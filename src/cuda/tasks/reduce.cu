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
    const uint32_t row_size_v = row_size & ~7u;
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
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        sum_sq += f.x * f.x + f.y * f.y;
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    sum_sq += v * v;
                }
                sum_sq = block_reduce_sum(sum_sq, smem);

                uint32_t eps_bits = task.dimensions[2];
                float eps = eps_bits ? __uint_as_float(eps_bits) : 1e-5f;
                float rms = rsqrtf(sum_sq / (float)row_size + eps);
                int weight_plus_one = task.strides[0];

                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    float4 o4;
                    __half2* oh = (__half2*)&o4;
                    if (weight) {
                        float4 w4 = *((const float4*)(weight + j));
                        __half2* wh = (__half2*)&w4;
                        for (int p = 0; p < 4; p++) {
                            float2 f = __half22float2(vh[p]);
                            float2 wf = __half22float2(wh[p]);
                            if (weight_plus_one) { wf.x += 1.0f; wf.y += 1.0f; }
                            oh[p] = __float22half2_rn(make_float2(f.x * rms * wf.x, f.y * rms * wf.y));
                        }
                    } else {
                        for (int p = 0; p < 4; p++) {
                            float2 f = __half22float2(vh[p]);
                            oh[p] = __float22half2_rn(make_float2(f.x * rms, f.y * rms));
                        }
                    }
                    *((float4*)(row_out + j)) = o4;
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
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
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        sum += f.x + f.y;
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                float mean = block_reduce_sum(sum, smem) / (float)row_size;

                float sum_sq = 0.0f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        float d0 = f.x - mean, d1 = f.y - mean;
                        sum_sq += d0 * d0 + d1 * d1;
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    float diff = __half2float(row_in[j]) - mean;
                    sum_sq += diff * diff;
                }
                float var_sum = block_reduce_sum(sum_sq, smem);
                float inv_std = rsqrtf(var_sum / (float)row_size + 1e-5f);

                const __half* ln_weight = weight;
                const __half* ln_bias = (task.buffer_indices[3] != 0xFFFFFFFF)
                                        ? (const __half*)buffers[task.buffer_indices[3]]
                                        : nullptr;

                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    float4 o4;
                    __half2* oh = (__half2*)&o4;
                    float4 w4, b4;
                    __half2 *wh = nullptr, *bh = nullptr;
                    if (ln_weight) { w4 = *((const float4*)(ln_weight + j)); wh = (__half2*)&w4; }
                    if (ln_bias)   { b4 = *((const float4*)(ln_bias + j));   bh = (__half2*)&b4; }
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        float v0 = (f.x - mean) * inv_std;
                        float v1 = (f.y - mean) * inv_std;
                        if (wh) { float2 wf = __half22float2(wh[p]); v0 *= wf.x; v1 *= wf.y; }
                        if (bh) { float2 bf = __half22float2(bh[p]); v0 += bf.x; v1 += bf.y; }
                        oh[p] = __float22half2_rn(make_float2(v0, v1));
                    }
                    *((float4*)(row_out + j)) = o4;
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    float v = (__half2float(row_in[j]) - mean) * inv_std;
                    if (ln_weight) v *= __half2float(ln_weight[j]);
                    if (ln_bias) v += __half2float(ln_bias[j]);
                    row_out[j] = __float2half(v);
                }
                break;
            }
            case REDUCE_SOFTMAX: {
                float max_val = -1e30f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        max_val = fmaxf(max_val, fmaxf(f.x, f.y));
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    max_val = fmaxf(max_val, __half2float(row_in[j]));
                }
                float row_max = block_reduce_max(max_val, smem);

                float sum_exp = 0.0f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        sum_exp += expf(f.x - row_max) + expf(f.y - row_max);
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    sum_exp += expf(__half2float(row_in[j]) - row_max);
                }
                float total_exp = block_reduce_sum(sum_exp, smem);
                float inv_sum = 1.0f / total_exp;

                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    float4 o4;
                    __half2* oh = (__half2*)&o4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        oh[p] = __float22half2_rn(make_float2(
                            expf(f.x - row_max) * inv_sum,
                            expf(f.y - row_max) * inv_sum));
                    }
                    *((float4*)(row_out + j)) = o4;
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    float v = expf(__half2float(row_in[j]) - row_max) * inv_sum;
                    row_out[j] = __float2half(v);
                }
                break;
            }
            case REDUCE_SUM: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        sum += f.x + f.y;
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                sum = block_reduce_sum(sum, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(sum);
                break;
            }
            case REDUCE_MEAN: {
                float sum = 0.0f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        sum += f.x + f.y;
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    sum += __half2float(row_in[j]);
                }
                sum = block_reduce_sum(sum, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(sum / (float)row_size);
                break;
            }
            case REDUCE_MAX: {
                float max_v = -1e30f;
                for (uint32_t j = threadIdx.x * 8; j < row_size_v; j += threads * 8) {
                    float4 v4 = *((const float4*)(row_in + j));
                    __half2* vh = (__half2*)&v4;
                    for (int p = 0; p < 4; p++) {
                        float2 f = __half22float2(vh[p]);
                        max_v = fmaxf(max_v, fmaxf(f.x, f.y));
                    }
                }
                for (uint32_t j = row_size_v + threadIdx.x; j < row_size; j += threads) {
                    max_v = fmaxf(max_v, __half2float(row_in[j]));
                }
                max_v = block_reduce_max(max_v, smem);
                if (threadIdx.x == 0)
                    out[row] = __float2half(max_v);
                break;
            }
            // ponytail: argmax stays scalar — float4 argmax needs per-element index tracking, complex for rare op
            case REDUCE_ARGMAX: {
                float max_v = -1e30f;
                int max_idx = 0;
                for (uint32_t j = threadIdx.x; j < row_size; j += threads) {
                    float v = __half2float(row_in[j]);
                    if (v > max_v) { max_v = v; max_idx = j; }
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
                    int other_i = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
                    if (other_v > max_v) { max_v = other_v; max_idx = other_i; }
                }
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
