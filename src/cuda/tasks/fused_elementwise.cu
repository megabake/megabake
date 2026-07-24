#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_fused_elementwise(const TaskDesc& task, void** buffers,
                                       const int* dyn_dims, int tile_id) {
    const int numel = task.dimensions[0];
    const int num_uops = task.dimensions[1];

    const int active_tiles = min((int)task.num_tiles, (int)gridDim.x);
    const int per_sm = (numel + active_tiles - 1) / active_tiles;
    const int start = tile_id * per_sm;
    const int end = min(start + per_sm, numel);

    // Pre-decode micro-ops
    int ops[8], dsts[8], s1s[8], s2s[8];
    int actual_uops = min(num_uops, 8);
    for (int u = 0; u < actual_uops; u++) {
        const uint32_t uop = (uint32_t)task.strides[u];
        ops[u] = uop & 0xFF;
        dsts[u] = (uop >> 8)  & 0xF;
        s1s[u]  = (uop >> 12) & 0xF;
        s2s[u]  = (uop >> 16) & 0xF;
    }

    // Vectorized path: process 4 elements per iteration
    const int vec_end = start + ((end - start) / 4) * 4;

    for (int base = start + (int)threadIdx.x * 4; base < vec_end; base += (int)blockDim.x * 4) {
        float regs[4][8];

        for (int u = 0; u < actual_uops; u++) {
            const int op  = ops[u];
            const int dst = dsts[u];
            const int s1  = s1s[u];
            const int s2  = s2s[u];

            switch (op) {
                case UOP_LOAD: {
                    const __half* src = (const __half*)buffers[task.buffer_indices[s1]];
                    // Load 4 consecutive halves (2 × half2 = 1 × float2)
                    float2 v2a = __half22float2(*reinterpret_cast<const __half2*>(&src[base]));
                    float2 v2b = __half22float2(*reinterpret_cast<const __half2*>(&src[base + 2]));
                    regs[0][dst] = v2a.x;
                    regs[1][dst] = v2a.y;
                    regs[2][dst] = v2b.x;
                    regs[3][dst] = v2b.y;
                    break;
                }
                case UOP_STORE: {
                    __half* d = (__half*)buffers[task.buffer_indices[dst]];
                    __half2 h2a = __float22half2_rn(make_float2(regs[0][s1], regs[1][s1]));
                    __half2 h2b = __float22half2_rn(make_float2(regs[2][s1], regs[3][s1]));
                    *reinterpret_cast<__half2*>(&d[base]) = h2a;
                    *reinterpret_cast<__half2*>(&d[base + 2]) = h2b;
                    break;
                }
                case UOP_ADD:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = regs[e][s1] + regs[e][s2];
                    break;
                case UOP_MUL:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = regs[e][s1] * regs[e][s2];
                    break;
                case UOP_SUB:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = regs[e][s1] - regs[e][s2];
                    break;
                case UOP_DIV:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = regs[e][s1] / regs[e][s2];
                    break;
                case UOP_SILU:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float v = regs[e][s1];
                        regs[e][dst] = v / (1.0f + __expf(-v));
                    }
                    break;
                case UOP_GELU:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float v = regs[e][s1];
                        regs[e][dst] = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
                    }
                    break;
                case UOP_RELU:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = fmaxf(regs[e][s1], 0.0f);
                    break;
                case UOP_TANH_U:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = tanhf(regs[e][s1]);
                    break;
                case UOP_NEG:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = -regs[e][s1];
                    break;
                case UOP_EXP:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = __expf(regs[e][s1]);
                    break;
                case UOP_SIGMOID:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float v = regs[e][s1];
                        regs[e][dst] = 1.0f / (1.0f + __expf(-v));
                    }
                    break;
                case UOP_RSQRT:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = rsqrtf(regs[e][s1]);
                    break;
                case UOP_LOG:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = logf(regs[e][s1]);
                    break;
                case UOP_ABS:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) regs[e][dst] = fabsf(regs[e][s1]);
                    break;
                case UOP_GELU_TANH:
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        float v = regs[e][s1];
                        float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                        regs[e][dst] = v * 0.5f * (1.0f + tanhf(c));
                    }
                    break;
            }
        }
    }

    // Scalar tail
    for (int idx = vec_end + (int)threadIdx.x; idx < end; idx += (int)blockDim.x) {
        float regs[8];

        for (int u = 0; u < actual_uops; u++) {
            const int op  = ops[u];
            const int dst = dsts[u];
            const int s1  = s1s[u];
            const int s2  = s2s[u];

            switch (op) {
                case UOP_LOAD:
                    regs[dst] = __half2float(
                        ((const __half*)buffers[task.buffer_indices[s1]])[idx]);
                    break;
                case UOP_STORE:
                    ((__half*)buffers[task.buffer_indices[dst]])[idx] =
                        __float2half(regs[s1]);
                    break;
                case UOP_ADD:     regs[dst] = regs[s1] + regs[s2];          break;
                case UOP_MUL:     regs[dst] = regs[s1] * regs[s2];          break;
                case UOP_SUB:     regs[dst] = regs[s1] - regs[s2];          break;
                case UOP_DIV:     regs[dst] = regs[s1] / regs[s2];          break;
                case UOP_SILU: {
                    float v = regs[s1];
                    regs[dst] = v / (1.0f + __expf(-v));
                    break;
                }
                case UOP_GELU: {
                    float v = regs[s1];
                    regs[dst] = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
                    break;
                }
                case UOP_RELU:    regs[dst] = fmaxf(regs[s1], 0.0f);        break;
                case UOP_TANH_U:  regs[dst] = tanhf(regs[s1]);              break;
                case UOP_NEG:     regs[dst] = -regs[s1];                    break;
                case UOP_EXP:     regs[dst] = __expf(regs[s1]);             break;
                case UOP_SIGMOID: {
                    float v = regs[s1];
                    regs[dst] = 1.0f / (1.0f + __expf(-v));
                    break;
                }
                case UOP_RSQRT:   regs[dst] = rsqrtf(regs[s1]);             break;
                case UOP_LOG:     regs[dst] = logf(regs[s1]);               break;
                case UOP_ABS:     regs[dst] = fabsf(regs[s1]);              break;
                case UOP_GELU_TANH: {
                    float v = regs[s1];
                    float c = 0.7978845608028654f * (v + 0.044715f * v * v * v);
                    regs[dst] = v * 0.5f * (1.0f + tanhf(c));
                    break;
                }
            }
        }
    }
}
