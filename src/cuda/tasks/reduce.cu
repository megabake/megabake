#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_reduce(const TaskDesc& task, void** buffers,
                             const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];
    const __half* weight = (task.buffer_indices[2] != 0xFFFFFFFF)
                           ? (const __half*)buffers[task.buffer_indices[2]]
                           : nullptr;

    const uint32_t num_rows = task.dimensions[0];
    const uint32_t row_size = task.dimensions[1];
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    extern __shared__ char smem_raw[];
    float* smem = (float*)smem_raw;

    const uint32_t rows_per_tile = (num_rows + num_tiles - 1) / num_tiles;
    const uint32_t row_start = tile_id * rows_per_tile;
    const uint32_t row_end = min(row_start + rows_per_tile, num_rows);

    for (uint32_t row = row_start; row < row_end; row++) {
        const __half* row_in = in0 + row * row_size;
        __half* row_out = out + row * row_size;

        switch (task.op_code) {
            case REDUCE_RMSNORM: {
                float sum_sq = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    sum_sq += v * v;
                }
                smem[threadIdx.x] = sum_sq;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                float rms = rsqrtf(smem[0] / (float)row_size + 1e-5f);
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]) * rms;
                    if (weight)
                        v *= __half2float(weight[j]);
                    row_out[j] = __float2half(v);
                }
                __syncthreads();
                break;
            }
            case REDUCE_LAYERNORM: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                smem[threadIdx.x] = sum;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                float mean = smem[0] / (float)row_size;

                float sum_sq = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float diff = __half2float(row_in[j]) - mean;
                    sum_sq += diff * diff;
                }
                smem[threadIdx.x] = sum_sq;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                float inv_std = rsqrtf(smem[0] / (float)row_size + 1e-5f);

                const __half* ln_weight = weight;
                const __half* ln_bias = (task.buffer_indices[3] != 0xFFFFFFFF)
                                        ? (const __half*)buffers[task.buffer_indices[3]]
                                        : nullptr;

                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = (__half2float(row_in[j]) - mean) * inv_std;
                    if (ln_weight) v *= __half2float(ln_weight[j]);
                    if (ln_bias) v += __half2float(ln_bias[j]);
                    row_out[j] = __float2half(v);
                }
                __syncthreads();
                break;
            }
            case REDUCE_SOFTMAX: {
                float max_val = -1e30f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_val) max_val = v;
                }
                smem[threadIdx.x] = max_val;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
                    __syncthreads();
                }
                float row_max = smem[0];

                float sum_exp = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum_exp += expf(__half2float(row_in[j]) - row_max);
                }
                smem[threadIdx.x] = sum_exp;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                float inv_sum = 1.0f / smem[0];

                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = expf(__half2float(row_in[j]) - row_max) * inv_sum;
                    row_out[j] = __float2half(v);
                }
                __syncthreads();
                break;
            }
            case REDUCE_SUM: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                smem[threadIdx.x] = sum;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    out[row] = __float2half(smem[0]);
                __syncthreads();
                break;
            }
            case REDUCE_MEAN: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                smem[threadIdx.x] = sum;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] += smem[threadIdx.x + s];
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    out[row] = __float2half(smem[0] / (float)row_size);
                __syncthreads();
                break;
            }
            case REDUCE_MAX: {
                float max_v = -1e30f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_v) max_v = v;
                }
                smem[threadIdx.x] = max_v;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s)
                        smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    out[row] = __float2half(smem[0]);
                __syncthreads();
                break;
            }
            case REDUCE_ARGMAX: {
                float max_v = -1e30f;
                int max_idx = 0;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_v) { max_v = v; max_idx = j; }
                }
                smem[threadIdx.x] = max_v;
                ((int*)smem)[threads + threadIdx.x] = max_idx;
                __syncthreads();
                for (int s = threads / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        if (smem[threadIdx.x + s] > smem[threadIdx.x]) {
                            smem[threadIdx.x] = smem[threadIdx.x + s];
                            ((int*)smem)[threads + threadIdx.x] =
                                ((int*)smem)[threads + threadIdx.x + s];
                        }
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0)
                    ((int64_t*)out)[row] = (int64_t)((int*)smem)[threads];
                __syncthreads();
                break;
            }
        }
    }
}
