#include "../data_types.cuh"
#include <cuda_fp16.h>

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
    const uint32_t total_heads = batch * num_heads;
    const uint32_t active_tiles = min((uint32_t)task.num_tiles, (uint32_t)gridDim.x);
    const uint32_t heads_per_tile = (total_heads + active_tiles - 1) / active_tiles;
    const uint32_t head_start = tile_id * heads_per_tile;
    const uint32_t head_end = min(head_start + heads_per_tile, total_heads);

    extern __shared__ char smem_raw[];
    float* smem = (float*)smem_raw;
    // smem layout: [seq_k scores] [8 warp scratch]
    float* warp_scratch = smem + seq_k;

    const bool is_causal = (task.strides[0] != 0);
    const uint32_t vec_dim = head_dim / 8;

    for (uint32_t bh = head_start; bh < head_end; bh++) {
        const uint32_t b = bh / num_heads;
        const uint32_t h = bh % num_heads;
        const uint32_t q_offset = b * num_heads + h;
        const uint32_t kv_h = h * num_kv_heads / num_heads;
        const uint32_t kv_offset = b * num_kv_heads + kv_h;

        for (uint32_t sq = 0; sq < seq_q; sq++) {
            const __half* q_row = Q + (q_offset * seq_q + sq) * head_dim;

            // --- Q * K^T with vectorized dot product ---
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                float dot = 0.0f;
                if (is_causal && sk > sq) {
                    dot = -1e30f;
                } else {
                    const __half* k_row = K + (kv_offset * seq_k + sk) * head_dim;
                    // Vectorized: load 8 halves at a time via float4
                    if (vec_dim > 0 && (head_dim % 8 == 0)) {
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
                        for (uint32_t d = 0; d < head_dim; d++) {
                            dot += __half2float(q_row[d]) *
                                   __half2float(k_row[d]);
                        }
                    }
                    dot *= scale;
                }
                smem[sk] = dot;
            }
            __syncthreads();

            // --- Softmax: max reduction via warp shuffles ---
            float max_score = -1e30f;
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                max_score = fmaxf(max_score, smem[sk]);
            }
            float row_max = block_reduce_max(max_score, warp_scratch);

            // --- Softmax: exp and sum ---
            float sum_exp = 0.0f;
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                float e = __expf(smem[sk] - row_max);
                smem[sk] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / block_reduce_sum(sum_exp, warp_scratch);

            // --- Normalize attention weights ---
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                smem[sk] *= inv_sum;
            }
            __syncthreads();

            // --- Attn * V with vectorized accumulation ---
            __half* out_row = out + (q_offset * seq_q + sq) * head_dim;

            if (vec_dim > 0 && (head_dim % 8 == 0)) {
                // Process 8 output dimensions at a time
                for (uint32_t dbase = threadIdx.x * 8; dbase < head_dim; dbase += blockDim.x * 8) {
                    if (dbase + 8 > head_dim) break;
                    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    for (uint32_t sk = 0; sk < seq_k; sk++) {
                        float w = smem[sk];
                        if (w == 0.0f) continue;
                        const __half* v_row = V + (kv_offset * seq_k + sk) * head_dim + dbase;
                        float4 v4 = *reinterpret_cast<const float4*>(v_row);
                        const __half2* vh = reinterpret_cast<const __half2*>(&v4);
                        #pragma unroll
                        for (int p = 0; p < 4; p++) {
                            float2 vf = __half22float2(vh[p]);
                            acc[p * 2]     += w * vf.x;
                            acc[p * 2 + 1] += w * vf.y;
                        }
                    }
                    // Store result
                    float4 o4;
                    __half2* oh = reinterpret_cast<__half2*>(&o4);
                    #pragma unroll
                    for (int p = 0; p < 4; p++) {
                        oh[p] = __float22half2_rn(make_float2(acc[p * 2], acc[p * 2 + 1]));
                    }
                    *reinterpret_cast<float4*>(&out_row[dbase]) = o4;
                }
            } else {
                for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                    float acc = 0.0f;
                    for (uint32_t sk = 0; sk < seq_k; sk++) {
                        acc += smem[sk] *
                               __half2float(V[(kv_offset * seq_k + sk) * head_dim + d]);
                    }
                    out_row[d] = __float2half(acc);
                }
            }
            __syncthreads();
        }
    }
}
