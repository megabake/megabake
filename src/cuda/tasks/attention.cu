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

    const float scale = rsqrtf((float)head_dim);
    const uint32_t total_heads = batch * num_heads;
    const uint32_t heads_per_tile = (total_heads + task.num_tiles - 1) / task.num_tiles;
    const uint32_t head_start = tile_id * heads_per_tile;
    const uint32_t head_end = min(head_start + heads_per_tile, total_heads);

    extern __shared__ char smem_raw[];
    float* smem = (float*)smem_raw;

    for (uint32_t bh = head_start; bh < head_end; bh++) {
        const uint32_t b = bh / num_heads;
        const uint32_t h = bh % num_heads;
        const uint32_t qkv_offset = (b * num_heads + h);

        for (uint32_t sq = 0; sq < seq_q; sq++) {
            const __half* q_row = Q + (qkv_offset * seq_q + sq) * head_dim;

            const bool is_causal = (task.strides[0] != 0);

            float max_score = -1e30f;
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                if (is_causal && sk > sq) {
                    smem[sk] = -1e30f;
                } else {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += __half2float(q_row[d]) *
                               __half2float(K[(qkv_offset * seq_k + sk) * head_dim + d]);
                    }
                    dot *= scale;
                    smem[sk] = dot;
                    if (dot > max_score) max_score = dot;
                }
            }
            __syncthreads();

            // Reduce max across threads
            smem[seq_k + threadIdx.x] = max_score;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s)
                    smem[seq_k + threadIdx.x] = fmaxf(smem[seq_k + threadIdx.x],
                                                       smem[seq_k + threadIdx.x + s]);
                __syncthreads();
            }
            float row_max = smem[seq_k];

            float sum_exp = 0.0f;
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                float e = expf(smem[sk] - row_max);
                smem[sk] = e;
                sum_exp += e;
            }
            __syncthreads();

            // Reduce sum
            smem[seq_k + threadIdx.x] = sum_exp;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s)
                    smem[seq_k + threadIdx.x] += smem[seq_k + threadIdx.x + s];
                __syncthreads();
            }
            float inv_sum = 1.0f / smem[seq_k];

            // Normalize attention weights
            for (uint32_t sk = threadIdx.x; sk < seq_k; sk += blockDim.x) {
                smem[sk] *= inv_sum;
            }
            __syncthreads();

            // Compute output: attn_weights @ V
            __half* out_row = out + (qkv_offset * seq_q + sq) * head_dim;
            for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                float acc = 0.0f;
                for (uint32_t sk = 0; sk < seq_k; sk++) {
                    acc += smem[sk] *
                           __half2float(V[(qkv_offset * seq_k + sk) * head_dim + d]);
                }
                out_row[d] = __float2half(acc);
            }
            __syncthreads();
        }
    }
}
