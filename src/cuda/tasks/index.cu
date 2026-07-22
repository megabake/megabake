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

    switch (task.op_code) {
        case INDEX_GATHER: {
            __half* out = (__half*)buffers[task.buffer_indices[0]];
            const __half* src = (const __half*)buffers[task.buffer_indices[1]];
            const int64_t* idx = (const int64_t*)buffers[task.buffer_indices[2]];
            const uint32_t inner_size = task.dimensions[1];
            for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
                uint32_t row = i / inner_size;
                uint32_t col = i % inner_size;
                int64_t src_row = idx[row];
                out[i] = src[src_row * inner_size + col];
            }
            break;
        }
        case INDEX_INDEX_SELECT: {
            __half* out = (__half*)buffers[task.buffer_indices[0]];
            const __half* src = (const __half*)buffers[task.buffer_indices[1]];
            const int64_t* idx = (const int64_t*)buffers[task.buffer_indices[2]];
            const uint32_t inner_size = task.dimensions[1];
            const uint32_t num_selected = task.dimensions[2];
            for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
                uint32_t sel = i / inner_size;
                uint32_t col = i % inner_size;
                int64_t src_row = idx[sel];
                out[i] = src[src_row * inner_size + col];
            }
            break;
        }
        default:
            break;
    }
}
