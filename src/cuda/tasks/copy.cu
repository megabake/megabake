#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_copy(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];

    const uint32_t total_elems = task.dimensions[0];
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    const uint32_t elems_per_tile = (total_elems + num_tiles - 1) / num_tiles;
    const uint32_t start = tile_id * elems_per_tile;
    const uint32_t end = min(start + elems_per_tile, total_elems);

    if (task.op_code == COPY_CAT) {
        const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];
        const __half* in1 = (const __half*)buffers[task.buffer_indices[2]];
        const uint32_t in0_last = task.dimensions[1];
        const uint32_t out_last = task.dimensions[2];
        const uint32_t in1_last = out_last - in0_last;

        for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
            uint32_t row = i / out_last;
            uint32_t col = i % out_last;
            if (col < in0_last)
                out[i] = in0[row * in0_last + col];
            else
                out[i] = in1[row * in1_last + (col - in0_last)];
        }
    } else {
        const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];

        // strides[0] = ndim (0 means flat copy)
        // strides[1..4] = source strides per dim
        // strides[5] = source offset (in elements)
        // dimensions[1..4] = source shape per dim
        const int ndim = task.strides[0];
        if (ndim <= 0) {
            for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
                out[i] = in0[i];
            }
        } else {
            const uint32_t src_offset = (uint32_t)task.strides[5];
            for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
                uint32_t remainder = i;
                uint32_t src_idx = src_offset;
                for (int d = 0; d < ndim; d++) {
                    uint32_t dim_size = task.dimensions[1 + d];
                    uint32_t below = 1;
                    for (int dd = d + 1; dd < ndim; dd++)
                        below *= task.dimensions[1 + dd];
                    uint32_t coord = remainder / below;
                    remainder = remainder % below;
                    src_idx += coord * (uint32_t)task.strides[1 + d];
                }
                out[i] = in0[src_idx];
            }
        }
    }
}
