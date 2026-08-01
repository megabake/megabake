#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ float _apply_unary(float af, uint16_t op_code) {
    switch (op_code) {
        case ELEM_SILU: {
            float sigmoid = 1.0f / (1.0f + __expf(-af));
            return af * sigmoid;
        }
        case ELEM_GELU:
            return af * 0.5f * (1.0f + erff(af * 0.7071067811865476f));
        case ELEM_GELU_TANH: {
            float c = 0.7978845608028654f * (af + 0.044715f * af * af * af);
            return af * 0.5f * (1.0f + tanhf(c));
        }
        case ELEM_RELU:
            return fmaxf(af, 0.0f);
        case ELEM_SIGMOID:
            return 1.0f / (1.0f + __expf(-af));
        case ELEM_TANH_OP:
            return tanhf(af);
        case ELEM_EXP:
            return __expf(af);
        case ELEM_LOG:
            return logf(af);
        case ELEM_RSQRT:
            return rsqrtf(af);
        case ELEM_NEG:
            return -af;
        case ELEM_ABS:
            return fabsf(af);
        case ELEM_COS:
            return cosf(af);
        case ELEM_SIN:
            return sinf(af);
        default:
            return af;
    }
}

__device__ __forceinline__ float _apply_binary(float af, float bf, uint16_t op_code) {
    switch (op_code) {
        case ELEM_ADD: return af + bf;
        case ELEM_MUL: return af * bf;
        case ELEM_SUB: return af - bf;
        case ELEM_DIV: return af / bf;
        default: return af;
    }
}

__device__ void task_elementwise(const TaskDesc& task, void** buffers,
                                 const int* dyn_dims, int tile_id, uint32_t dispatch_flags) {
    const uint32_t total_elems = task.dimensions[0];
    const int threads = blockDim.x;
    const int num_tiles = min((int)task.num_tiles, (int)gridDim.x);

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
    const uint16_t op = task.op_code;

    // Vectorized path: unary ops or same-shape binary (no broadcast)
    const bool can_vectorize = (in1_numel == 0) &&
                               (start % 8 == 0) &&
                               (op != ELEM_POW) && (op != ELEM_CAST) &&
                               (op != ELEM_WHERE) && (op != ELEM_MASKED_FILL) &&
                               (op != ELEM_CLAMP);

    if (can_vectorize) {
        const uint32_t vec_end = start + ((end - start) / 8) * 8;

        if (in1 == nullptr && op <= ELEM_DIV) {
            // Binary with scalar constant
            float scalar = __int_as_float(task.dimensions[1]);
            for (uint32_t i = start + threadIdx.x * 8; i < vec_end; i += threads * 8) {
                float4 a4 = *reinterpret_cast<const float4*>(&in0[i]);
                __half2* ah = reinterpret_cast<__half2*>(&a4);
                float4 o4;
                __half2* oh = reinterpret_cast<__half2*>(&o4);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 af = __half22float2(ah[k]);
                    float2 of;
                    of.x = _apply_binary(af.x, scalar, op);
                    of.y = _apply_binary(af.y, scalar, op);
                    oh[k] = __float22half2_rn(of);
                }
                *reinterpret_cast<float4*>(&out[i]) = o4;
            }
        } else if (in1 != nullptr) {
            // Binary with same-shape tensor
            for (uint32_t i = start + threadIdx.x * 8; i < vec_end; i += threads * 8) {
                float4 a4 = *reinterpret_cast<const float4*>(&in0[i]);
                float4 b4 = *reinterpret_cast<const float4*>(&in1[i]);
                __half2* ah = reinterpret_cast<__half2*>(&a4);
                __half2* bh = reinterpret_cast<__half2*>(&b4);
                float4 o4;
                __half2* oh = reinterpret_cast<__half2*>(&o4);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 af = __half22float2(ah[k]);
                    float2 bf = __half22float2(bh[k]);
                    float2 of;
                    of.x = _apply_binary(af.x, bf.x, op);
                    of.y = _apply_binary(af.y, bf.y, op);
                    oh[k] = __float22half2_rn(of);
                }
                *reinterpret_cast<float4*>(&out[i]) = o4;
            }
        } else {
            // Unary
            for (uint32_t i = start + threadIdx.x * 8; i < vec_end; i += threads * 8) {
                float4 a4 = *reinterpret_cast<const float4*>(&in0[i]);
                __half2* ah = reinterpret_cast<__half2*>(&a4);
                float4 o4;
                __half2* oh = reinterpret_cast<__half2*>(&o4);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    float2 af = __half22float2(ah[k]);
                    float2 of;
                    of.x = _apply_unary(af.x, op);
                    of.y = _apply_unary(af.y, op);
                    oh[k] = __float22half2_rn(of);
                }
                *reinterpret_cast<float4*>(&out[i]) = o4;
            }
        }

        // Scalar tail
        for (uint32_t i = vec_end + threadIdx.x; i < end; i += threads) {
            float af = __half2float(in0[i]);
            float result;
            if (in1 != nullptr) {
                result = _apply_binary(af, __half2float(in1[i]), op);
            } else if (op <= ELEM_DIV) {
                result = _apply_binary(af, __int_as_float(task.dimensions[1]), op);
            } else {
                result = _apply_unary(af, op);
            }
            out[i] = __float2half(result);
        }
        return;
    }

    // Scalar path: broadcast, POW, CAST, or unaligned
    for (uint32_t i = start + threadIdx.x; i < end; i += threads) {
        __half a = in0[i];
        __half val;

        switch (op) {
            case ELEM_ADD: case ELEM_MUL: case ELEM_SUB: case ELEM_DIV: {
                float bf;
                if (in1 == nullptr) bf = __int_as_float(task.dimensions[1]);
                else if (in1_numel) bf = __half2float(in1[(in1_repeat > 1) ? (i / in1_repeat) : (i % in1_numel)]);
                else bf = __half2float(in1[i]);
                val = __float2half(_apply_binary(__half2float(a), bf, op));
                break;
            }
            case ELEM_CAST:
                val = a;
                break;
            case ELEM_POW: {
                float af = __half2float(a);
                float exp_val = __int_as_float(task.dimensions[1]);
                val = __float2half(powf(fabsf(af), exp_val));
                break;
            }
            default:
                val = __float2half(_apply_unary(__half2float(a), op));
                break;
        }
        out[i] = val;
    }
}
