#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_matmul(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* A = (const __half*)buffers[task.buffer_indices[1]];
    const __half* B = (const __half*)buffers[task.buffer_indices[2]];

    const uint32_t M = task.dimensions[0];
    const uint32_t N = task.dimensions[1];
    const uint32_t K = task.dimensions[2];

    const uint32_t TILE_M = 16;
    const uint32_t TILE_N = 16;

    const uint32_t tiles_n = (N + TILE_N - 1) / TILE_N;
    const uint32_t tiles_m = (M + TILE_M - 1) / TILE_M;
    const uint32_t total_tiles = tiles_m * tiles_n;

    const uint32_t tiles_per_sm = (total_tiles + task.num_tiles - 1) / task.num_tiles;
    const uint32_t tile_start = tile_id * tiles_per_sm;
    const uint32_t tile_end = min(tile_start + tiles_per_sm, total_tiles);

    for (uint32_t t = tile_start; t < tile_end; t++) {
        const uint32_t tile_row = t / tiles_n;
        const uint32_t tile_col = t % tiles_n;
        const uint32_t m_start = tile_row * TILE_M;
        const uint32_t n_start = tile_col * TILE_N;

        for (uint32_t local = threadIdx.x; local < TILE_M * TILE_N; local += blockDim.x) {
            uint32_t local_m = local / TILE_N;
            uint32_t local_n = local % TILE_N;
            uint32_t gm = m_start + local_m;
            uint32_t gn = n_start + local_n;

            if (gm < M && gn < N) {
                float acc = 0.0f;
                // strides[0] != 0 means B is transposed: physical layout (N, K)
                if (task.strides[0] != 0) {
                    for (uint32_t k = 0; k < K; k++) {
                        acc += __half2float(A[gm * K + k]) * __half2float(B[gn * K + k]);
                    }
                } else {
                    for (uint32_t k = 0; k < K; k++) {
                        acc += __half2float(A[gm * K + k]) * __half2float(B[k * N + gn]);
                    }
                }
                out[gm * N + gn] = __float2half(acc);
            }
        }
    }
}
