#include "../data_types.cuh"
#include <cuda_fp16.h>

// FlashAttention-style K-tiled attention with online softmax.
// O(BK) SMEM per head (K-block + V-block), not O(seq_k).
// ponytail: FMA dot products; tensor core MMA when prefill perf matters.

#define ATTN_BK 64

__device__ void task_attention(const TaskDesc& task, void** buffers,
                               const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* Q = (const __half*)buffers[task.buffer_indices[1]];
    const __half* K = (const __half*)buffers[task.buffer_indices[2]];
    const __half* V = (const __half*)buffers[task.buffer_indices[3]];

    const uint32_t batch = task.dimensions[0];
    const uint32_t num_heads = task.dimensions[1];
    const uint32_t seq_q = task.dimensions[2];
    const uint32_t head_dim = task.dimensions[3];
    const uint32_t seq_k = task.dimensions[4];
    const uint32_t num_kv_heads = task.dimensions[5] ? task.dimensions[5] : num_heads;

    const float scale = rsqrtf((float)head_dim);
    const bool is_causal = (task.strides[0] != 0);
    const uint32_t vec_dim = head_dim / 8;
    const bool use_vec = (head_dim % 8 == 0);

    // Tile over (batch, head, query_position)
    const uint32_t total_work = batch * num_heads * seq_q;
    const uint32_t active_tiles = min((uint32_t)task.num_tiles, (uint32_t)gridDim.x);
    const uint32_t work_per_tile = (total_work + active_tiles - 1) / active_tiles;
    const uint32_t work_start = tile_id * work_per_tile;
    const uint32_t work_end = min(work_start + work_per_tile, total_work);

    extern __shared__ char smem_raw[];
    // SMEM: scores[ATTN_BK] | K_block[ATTN_BK * head_dim] | V_block[ATTN_BK * head_dim] | scratch[8]
    float* scores = (float*)smem_raw;
    __half* k_smem = (__half*)(scores + ATTN_BK);
    __half* v_smem = k_smem + ATTN_BK * head_dim;
    float* scratch = (float*)(v_smem + ATTN_BK * head_dim);

    for (uint32_t w = work_start; w < work_end; w++) {
        const uint32_t sq = w % seq_q;
        const uint32_t bh = w / seq_q;
        const uint32_t b = bh / num_heads;
        const uint32_t h = bh % num_heads;
        const uint32_t kv_h = h * num_kv_heads / num_heads;

        const __half* q_row = Q + ((uint64_t)(b * num_heads + h) * seq_q + sq) * head_dim;
        const uint64_t kv_base = (uint64_t)(b * num_kv_heads + kv_h) * seq_k * head_dim;
        __half* out_row = out + ((uint64_t)(b * num_heads + h) * seq_q + sq) * head_dim;

        float m = -1e30f;
        float l = 0.0f;

        // Per-thread output accumulators
        // Vectorized: thread t handles dims [t*8, t*8+8)
        // Scalar: thread t handles dim t
        const uint32_t my_d_vec = threadIdx.x * 8;
        const bool has_vec = use_vec && (my_d_vec < head_dim);
        const bool has_scalar = !use_vec && (threadIdx.x < head_dim);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
        float as = 0.f;

        // Causal: only attend to keys <= query position
        const uint32_t max_sk = is_causal ? min(seq_k, sq + 1) : seq_k;

        for (uint32_t sk_base = 0; sk_base < max_sk; sk_base += ATTN_BK) {
            const uint32_t bk_len = min((uint32_t)ATTN_BK, max_sk - sk_base);

            // Load K and V blocks cooperatively
            const uint32_t kv_elems = bk_len * head_dim;
            for (uint32_t i = threadIdx.x; i < kv_elems; i += blockDim.x) {
                k_smem[i] = K[kv_base + sk_base * head_dim + i];
                v_smem[i] = V[kv_base + sk_base * head_dim + i];
            }
            __syncthreads();

            // --- Phase 1: QK^T dot products (threads parallel over sk) ---
            float my_max = -1e30f;
            for (uint32_t sk = threadIdx.x; sk < bk_len; sk += blockDim.x) {
                float dot = 0.f;
                if (is_causal && sk_base + sk > sq) {
                    dot = -1e30f;
                } else {
                    const __half* k_row = k_smem + sk * head_dim;
                    if (use_vec) {
                        const float4* q4 = reinterpret_cast<const float4*>(q_row);
                        const float4* k4 = reinterpret_cast<const float4*>(k_row);
                        for (uint32_t v = 0; v < vec_dim; v++) {
                            float4 qv = q4[v];
                            float4 kv = k4[v];
                            const __half2* qh = reinterpret_cast<const __half2*>(&qv);
                            const __half2* kh = reinterpret_cast<const __half2*>(&kv);
                            #pragma unroll
                            for (int p = 0; p < 4; p++) {
                                float2 qf = __half22float2(qh[p]);
                                float2 kf = __half22float2(kh[p]);
                                dot += qf.x * kf.x + qf.y * kf.y;
                            }
                        }
                    } else {
                        for (uint32_t d = 0; d < head_dim; d++)
                            dot += __half2float(q_row[d]) * __half2float(k_row[d]);
                    }
                    dot *= scale;
                }
                scores[sk] = dot;
                my_max = fmaxf(my_max, dot);
            }

            // --- Online softmax: update running max and rescale ---
            float m_new = fmaxf(m, block_reduce_max(my_max, scratch));
            float alpha = __expf(m - m_new);

            // Rescale accumulators
            l *= alpha;
            if (has_vec) {
                a0 *= alpha; a1 *= alpha; a2 *= alpha; a3 *= alpha;
                a4 *= alpha; a5 *= alpha; a6 *= alpha; a7 *= alpha;
            }
            if (has_scalar) as *= alpha;

            // Compute P = exp(score - m_new) and sum
            float my_sum = 0.f;
            for (uint32_t sk = threadIdx.x; sk < bk_len; sk += blockDim.x) {
                float p = __expf(scores[sk] - m_new);
                scores[sk] = p;
                my_sum += p;
            }
            l += block_reduce_sum(my_sum, scratch);
            m = m_new;

            __syncthreads();

            // --- Phase 2: P @ V accumulation (threads parallel over head_dim) ---
            if (has_vec) {
                float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
                float t4 = 0.f, t5 = 0.f, t6 = 0.f, t7 = 0.f;
                for (uint32_t sk = 0; sk < bk_len; sk++) {
                    float pw = scores[sk];
                    if (pw == 0.f) continue;
                    const float4 v4 = *reinterpret_cast<const float4*>(
                        v_smem + sk * head_dim + my_d_vec);
                    const __half2* vh = reinterpret_cast<const __half2*>(&v4);
                    float2 f0 = __half22float2(vh[0]);
                    float2 f1 = __half22float2(vh[1]);
                    float2 f2 = __half22float2(vh[2]);
                    float2 f3 = __half22float2(vh[3]);
                    t0 += pw * f0.x; t1 += pw * f0.y;
                    t2 += pw * f1.x; t3 += pw * f1.y;
                    t4 += pw * f2.x; t5 += pw * f2.y;
                    t6 += pw * f3.x; t7 += pw * f3.y;
                }
                a0 += t0; a1 += t1; a2 += t2; a3 += t3;
                a4 += t4; a5 += t5; a6 += t6; a7 += t7;
            }
            if (has_scalar) {
                float ts = 0.f;
                for (uint32_t sk = 0; sk < bk_len; sk++)
                    ts += scores[sk] * __half2float(v_smem[sk * head_dim + threadIdx.x]);
                as += ts;
            }

            __syncthreads();
        }

        // --- Write output: acc / l ---
        float inv_l = (l > 0.f) ? 1.f / l : 0.f;
        if (has_vec) {
            float4 o4;
            __half2* oh = reinterpret_cast<__half2*>(&o4);
            oh[0] = __float22half2_rn(make_float2(a0 * inv_l, a1 * inv_l));
            oh[1] = __float22half2_rn(make_float2(a2 * inv_l, a3 * inv_l));
            oh[2] = __float22half2_rn(make_float2(a4 * inv_l, a5 * inv_l));
            oh[3] = __float22half2_rn(make_float2(a6 * inv_l, a7 * inv_l));
            *reinterpret_cast<float4*>(&out_row[my_d_vec]) = o4;
        }
        if (has_scalar)
            out_row[threadIdx.x] = __float2half(as * inv_l);

        __syncthreads();
    }
}
