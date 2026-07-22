#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_embedding(const TaskDesc& task, void** buffers,
                                const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const int64_t* indices = (const int64_t*)buffers[task.buffer_indices[1]];
    const __half* table = (const __half*)buffers[task.buffer_indices[2]];

    const uint32_t num_indices = task.dimensions[0];
    const uint32_t embed_dim = task.dimensions[1];
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    const uint32_t indices_per_tile = (num_indices + num_tiles - 1) / num_tiles;
    const uint32_t start = tile_id * indices_per_tile;
    const uint32_t end = min(start + indices_per_tile, num_indices);

    for (uint32_t idx = start; idx < end; idx++) {
        int64_t token_id = indices[idx];
        const __half* row = table + token_id * embed_dim;
        __half* dst = out + idx * embed_dim;
        for (uint32_t j = threadIdx.x; j < embed_dim; j += threads) {
            dst[j] = row[j];
        }
    }
}
