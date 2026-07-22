#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_elementwise(const TaskDesc& task, void** buffers,
                                 const int* dyn_dims, int tile_id) {
    const uint32_t total_elems = task.dimensions[0];
    const int threads = blockDim.x;
    const int num_tiles = task.num_tiles;

    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];
    const __half* in1 = (task.buffer_indices[2] != 0xFFFFFFFF)
                        ? (const __half*)buffers[task.buffer_indices[2]]
                        : nullptr;

    const uint32_t elems_per_tile = (total_elems + num_tiles - 1) / num_tiles;
    const uint32_t start = tile_id * elems_per_tile;
    const uint32_t end = min(start + elems_per_tile, total_elems);

    const uint32_t in1_numel = task.dimensions[2];
    const uint32_t in1_repeat = task.dimensions[3];

    for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
        __half a = in0[i];
        __half val;

        switch (task.op_code) {
            case ELEM_ADD: {
                float bf;
                if (in1 == nullptr) bf = __int_as_float(task.dimensions[1]);
                else if (in1_numel) bf = __half2float(in1[(in1_repeat > 1) ? (i / in1_repeat) : (i % in1_numel)]);
                else bf = __half2float(in1[i]);
                val = __float2half(__half2float(a) + bf);
                break;
            }
            case ELEM_MUL: {
                float bf;
                if (in1 == nullptr) bf = __int_as_float(task.dimensions[1]);
                else if (in1_numel) bf = __half2float(in1[(in1_repeat > 1) ? (i / in1_repeat) : (i % in1_numel)]);
                else bf = __half2float(in1[i]);
                val = __float2half(__half2float(a) * bf);
                break;
            }
            case ELEM_SUB: {
                float bf;
                if (in1 == nullptr) bf = __int_as_float(task.dimensions[1]);
                else if (in1_numel) bf = __half2float(in1[(in1_repeat > 1) ? (i / in1_repeat) : (i % in1_numel)]);
                else bf = __half2float(in1[i]);
                val = __float2half(__half2float(a) - bf);
                break;
            }
            case ELEM_DIV: {
                float bf;
                if (in1 == nullptr) bf = __int_as_float(task.dimensions[1]);
                else if (in1_numel) bf = __half2float(in1[(in1_repeat > 1) ? (i / in1_repeat) : (i % in1_numel)]);
                else bf = __half2float(in1[i]);
                val = __float2half(__half2float(a) / bf);
                break;
            }
            case ELEM_SILU: {
                float af = __half2float(a);
                float sigmoid = 1.0f / (1.0f + expf(-af));
                val = __float2half(af * sigmoid);
                break;
            }
            case ELEM_GELU: {
                float af = __half2float(a);
                float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (af + 0.044715f * af * af * af)));
                val = __float2half(af * cdf);
                break;
            }
            case ELEM_RELU: {
                float af = __half2float(a);
                val = __float2half(fmaxf(af, 0.0f));
                break;
            }
            case ELEM_SIGMOID: {
                float af = __half2float(a);
                val = __float2half(1.0f / (1.0f + expf(-af)));
                break;
            }
            case ELEM_TANH_OP: {
                float af = __half2float(a);
                val = __float2half(tanhf(af));
                break;
            }
            case ELEM_EXP: {
                val = __float2half(expf(__half2float(a)));
                break;
            }
            case ELEM_LOG: {
                val = __float2half(logf(__half2float(a)));
                break;
            }
            case ELEM_RSQRT: {
                val = __float2half(rsqrtf(__half2float(a)));
                break;
            }
            case ELEM_NEG: {
                val = __hneg(a);
                break;
            }
            case ELEM_ABS: {
                val = __habs(a);
                break;
            }
            case ELEM_CAST: {
                val = a;
                break;
            }
            case ELEM_POW: {
                float af = __half2float(a);
                float exp_val = __int_as_float(task.dimensions[1]);
                // --use_fast_math replaces powf with __powf which
                // uses exp2(y*log2(x)) — NaN for negative x.
                // Use fabsf to handle negative bases safely.
                val = __float2half(powf(fabsf(af), exp_val));
                break;
            }
            default:
                val = a;
                break;
        }
        out[i] = val;
    }
}
