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
    const int num_tiles = min((int)task.num_tiles, (int)gridDim.x);
    const int layout_bhsd = task.strides[0];
    const uint32_t cos_stride = task.dimensions[4] ? task.dimensions[4] : half_dim;

    const uint32_t total_positions = batch * seq_len;
    const uint32_t pos_per_tile = (total_positions + num_tiles - 1) / num_tiles;
    const uint32_t pos_start = tile_id * pos_per_tile;
    const uint32_t pos_end = min(pos_start + pos_per_tile, total_positions);

    for (uint32_t pos = pos_start; pos < pos_end; pos++) {
        uint32_t b = pos / seq_len;
        uint32_t s = pos % seq_len;
        for (uint32_t h = 0; h < num_heads; h++) {
            uint32_t base_offset;
            if (layout_bhsd) {
                base_offset = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim + s * head_dim;
            } else {
                base_offset = pos * num_heads * head_dim + h * head_dim;
            }
            for (uint32_t d = threadIdx.x; d < half_dim; d += threads) {
                float x0 = __half2float(in0[base_offset + d]);
                float x1 = __half2float(in0[base_offset + d + half_dim]);
                float c = __half2float(cos_cache[s * cos_stride + d]);
                float sv = __half2float(sin_cache[s * cos_stride + d]);
                out[base_offset + d] = __float2half(x0 * c - x1 * sv);
                out[base_offset + d + half_dim] = __float2half(x1 * c + x0 * sv);
            }
        }
    }
}
