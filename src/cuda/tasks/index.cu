#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_index(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id) {
    const uint32_t total_elems = task.dimensions[0];
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    const uint32_t elems_per_tile = (total_elems + num_tiles - 1) / num_tiles;
    const uint32_t start = tile_id * elems_per_tile;
    const uint32_t end = min(start + elems_per_tile, total_elems);

    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* src = (const __half*)buffers[task.buffer_indices[1]];
    const int64_t* idx = (const int64_t*)buffers[task.buffer_indices[2]];
    const uint32_t inner_size = task.dimensions[1];

    if (task.op_code != INDEX_GATHER && task.op_code != INDEX_INDEX_SELECT)
        return;

    if (inner_size % 8 == 0) {
        const uint32_t inner_vec = inner_size / 8;
        const uint32_t total_vec = (end - start) / 8;
        const uint32_t vec_start = start / 8;
        for (uint32_t i = threadIdx.x; i < total_vec; i += threads) {
            uint32_t gi = vec_start + i;
            uint32_t row = gi / inner_vec;
            uint32_t col = gi % inner_vec;
            int64_t src_row = idx[row];
            ((float4*)out)[gi] = ((const float4*)src)[src_row * inner_vec + col];
        }
    } else {
        for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
            uint32_t row = i / inner_size;
            uint32_t col = i % inner_size;
            int64_t src_row = idx[row];
            out[i] = src[src_row * inner_size + col];
        }
    }
}
