#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_rope(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];
    const __half* cos_cache = (const __half*)buffers[task.buffer_indices[2]];
    const __half* sin_cache = (const __half*)buffers[task.buffer_indices[3]];

    const uint32_t batch = task.dimensions[0];
    const uint32_t seq_len = task.dimensions[1];
    const uint32_t num_heads = task.dimensions[2];
    const uint32_t head_dim = task.dimensions[3];
    const uint32_t half_dim = head_dim / 2;
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    const uint32_t total_positions = batch * seq_len;
    const uint32_t pos_per_tile = (total_positions + num_tiles - 1) / num_tiles;
    const uint32_t pos_start = tile_id * pos_per_tile;
    const uint32_t pos_end = min(pos_start + pos_per_tile, total_positions);

    for (uint32_t pos = pos_start; pos < pos_end; pos++) {
        uint32_t seq_idx = pos % seq_len;
        for (uint32_t h = 0; h < num_heads; h++) {
            uint32_t base_offset = pos * num_heads * head_dim + h * head_dim;
            for (uint32_t d = threadIdx.x; d < half_dim; d += threads) {
                float x0 = __half2float(in0[base_offset + d]);
                float x1 = __half2float(in0[base_offset + d + half_dim]);
                float c = __half2float(cos_cache[seq_idx * half_dim + d]);
                float s = __half2float(sin_cache[seq_idx * half_dim + d]);
                out[base_offset + d] = __float2half(x0 * c - x1 * s);
                out[base_offset + d + half_dim] = __float2half(x1 * c + x0 * s);
            }
        }
    }
}
