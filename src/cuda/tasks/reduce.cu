#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? smem[threadIdx.x] : -1e30f;
        val = warp_reduce_max(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

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
    const int num_tiles = min((int)task.num_tiles, (int)gridDim.x);

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
                sum_sq = block_reduce_sum(sum_sq, smem);

                uint32_t eps_bits = task.dimensions[2];
                float eps = eps_bits ? __uint_as_float(eps_bits) : 1e-5f;
                float rms = rsqrtf(sum_sq / (float)row_size + eps);
                int weight_plus_one = task.strides[0];
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]) * rms;
                    if (weight) {
                        float w = __half2float(weight[j]);
                        if (weight_plus_one) w += 1.0f;
                        v *= w;
                    }
                    row_out[j] = __float2half(v);
                }
                break;
            }
            case REDUCE_LAYERNORM: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                float mean = block_reduce_sum(sum, smem) / (float)row_size;

                float sum_sq = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float diff = __half2float(row_in[j]) - mean;
                    sum_sq += diff * diff;
                }
                float var_sum = block_reduce_sum(sum_sq, smem);
                float inv_std = rsqrtf(var_sum / (float)row_size + 1e-5f);

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
                break;
            }
            case REDUCE_SOFTMAX: {
                float max_val = -1e30f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_val) max_val = v;
                }
                float row_max = block_reduce_max(max_val, smem);

                float sum_exp = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum_exp += expf(__half2float(row_in[j]) - row_max);
                }
                float total_exp = block_reduce_sum(sum_exp, smem);
                float inv_sum = 1.0f / total_exp;

                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = expf(__half2float(row_in[j]) - row_max) * inv_sum;
                    row_out[j] = __float2half(v);
                }
                break;
            }
            case REDUCE_SUM: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                sum = block_reduce_sum(sum, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(sum);
                break;
            }
            case REDUCE_MEAN: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                sum = block_reduce_sum(sum, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(sum / (float)row_size);
                break;
            }
            case REDUCE_MAX: {
                float max_v = -1e30f;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_v) max_v = v;
                }
                max_v = block_reduce_max(max_v, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(max_v);
                break;
            }
            case REDUCE_ARGMAX: {
                float max_v = -1e30f;
                int max_idx = 0;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_v) { max_v = v; max_idx = j; }
                }
                // Warp-level argmax
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
                    int other_i = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
                    if (other_v > max_v) { max_v = other_v; max_idx = other_i; }
                }
                // Cross-warp via smem
                const int lane = threadIdx.x & 31;
                const int wid = threadIdx.x >> 5;
                if (lane == 0) {
                    smem[wid] = max_v;
                    ((int*)smem)[8 + wid] = max_idx;
                }
                __syncthreads();
                if (wid == 0) {
                    int nwarps = blockDim.x >> 5;
                    max_v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -1e30f;
                    max_idx = (threadIdx.x < nwarps) ? ((int*)smem)[8 + threadIdx.x] : 0;
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
                        int other_i = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
                        if (other_v > max_v) { max_v = other_v; max_idx = other_i; }
                    }
                }
                if (threadIdx.x == 0)
                    ((int64_t*)out)[row] = (int64_t)max_idx;
                __syncthreads();
                break;
            }
        }
    }
}
