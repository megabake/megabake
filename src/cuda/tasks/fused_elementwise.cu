#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ float apply_uop_unary(int op, float a) {
    switch (op) {
        case UOP_SILU:    return a / (1.0f + __expf(-a));
        case UOP_GELU:    return a * 0.5f * (1.0f + erff(a * 0.7071067811865476f));
        case UOP_RELU:    return fmaxf(a, 0.0f);
        case UOP_TANH_U:  return tanhf(a);
        case UOP_NEG:     return -a;
        case UOP_EXP:     return __expf(a);
        case UOP_SIGMOID: return 1.0f / (1.0f + __expf(-a));
        case UOP_RSQRT:   return rsqrtf(a);
        case UOP_LOG:     return logf(a);
        case UOP_ABS:     return fabsf(a);
        case UOP_GELU_TANH: {
            float c = 0.7978845608028654f * (a + 0.044715f * a * a * a);
            return a * 0.5f * (1.0f + tanhf(c));
        }
        default: return a;
    }
}

__device__ __forceinline__ float apply_uop_binary(int op, float a, float b) {
    switch (op) {
        case UOP_ADD: return a + b;
        case UOP_MUL: return a * b;
        case UOP_SUB: return a - b;
        case UOP_DIV: return a / b;
        default: return a;
    }
}

__device__ __forceinline__ void exec_scalar_uop(
    int op, float* regs, int dst, int s1, int s2, int idx,
    const TaskDesc& task, void** buffers
) {
    if (op == UOP_LOAD) {
        regs[dst] = __half2float(((const __half*)buffers[task.buffer_indices[s1]])[idx]);
    } else if (op == UOP_LOAD_BROADCAST) {
        const __half* src = (const __half*)buffers[task.buffer_indices[s1]];
        uint32_t bc = task.dimensions[3 + s2];
        uint32_t bc_n = bc >> 16, bc_r = bc & 0xFFFF;
        regs[dst] = __half2float(src[(bc_r > 1) ? (idx / bc_r) % bc_n : idx % bc_n]);
    } else if (op == UOP_STORE) {
        ((__half*)buffers[task.buffer_indices[dst]])[idx] = __float2half(regs[s1]);
    } else if (op >= UOP_ADD && op <= UOP_DIV) {
        regs[dst] = apply_uop_binary(op, regs[s1], regs[s2]);
    } else {
        regs[dst] = apply_uop_unary(op, regs[s1]);
    }
}

__device__ void task_fused_elementwise(const TaskDesc& task, void** buffers,
                                       const int* dyn_dims, int tile_id, uint32_t dispatch_flags) {
    const int numel = task.dimensions[0];
    const int num_uops = task.dimensions[1];
    const int threads = blockDim.x;

    const int active_tiles = min((int)task.num_tiles, (int)gridDim.x);
    const int per_sm = (numel + active_tiles - 1) / active_tiles;
    const int start = tile_id * per_sm;
    const int end = min(start + per_sm, numel);

    // Pre-decode micro-ops (up to 32, first 8 from strides, rest from program buffer)
    int ops[32], dsts[32], s1s[32], s2s[32];
    int actual_uops = min(num_uops, 32);
    int from_strides = min(actual_uops, 8);
    for (int u = 0; u < from_strides; u++) {
        const uint32_t uop = (uint32_t)task.strides[u];
        ops[u]  = uop & 0xFF;
        dsts[u] = (uop >> 8)  & 0xF;
        s1s[u]  = (uop >> 12) & 0xF;
        s2s[u]  = (uop >> 16) & 0xF;
    }
    if (actual_uops > 8) {
        const uint32_t* prog = (const uint32_t*)buffers[task.dimensions[2]];
        for (int u = 8; u < actual_uops; u++) {
            uint32_t uop = prog[u - 8];
            ops[u]  = uop & 0xFF;
            dsts[u] = (uop >> 8)  & 0xF;
            s1s[u]  = (uop >> 12) & 0xF;
            s2s[u]  = (uop >> 16) & 0xF;
        }
    }

    // Scan for special ops
    bool has_special = false;
    int reduce_at = -1;
    for (int u = 0; u < actual_uops; u++) {
        if (ops[u] == UOP_LOAD_BROADCAST) has_special = true;
        if (ops[u] >= UOP_REDUCE_SUM && ops[u] <= UOP_REDUCE_MEAN) {
            reduce_at = u;
            has_special = true;
            break;
        }
    }

    if (!has_special) {
        // ====== FAST PATH: vectorized + scalar tail ======
        const int vec_end = start + ((end - start) / 4) * 4;

        for (int base = start + (int)threadIdx.x * 4; base < vec_end;
             base += threads * 4) {
            float regs[4][16];

            for (int u = 0; u < actual_uops; u++) {
                const int op  = ops[u];
                const int dst = dsts[u];
                const int s1  = s1s[u];
                const int s2  = s2s[u];

                if (op == UOP_LOAD) {
                    const __half* src = (const __half*)buffers[task.buffer_indices[s1]];
                    float2 v2a = __half22float2(
                        *reinterpret_cast<const __half2*>(&src[base]));
                    float2 v2b = __half22float2(
                        *reinterpret_cast<const __half2*>(&src[base + 2]));
                    regs[0][dst] = v2a.x;
                    regs[1][dst] = v2a.y;
                    regs[2][dst] = v2b.x;
                    regs[3][dst] = v2b.y;
                } else if (op == UOP_STORE) {
                    __half* d = (__half*)buffers[task.buffer_indices[dst]];
                    __half2 h2a = __float22half2_rn(
                        make_float2(regs[0][s1], regs[1][s1]));
                    __half2 h2b = __float22half2_rn(
                        make_float2(regs[2][s1], regs[3][s1]));
                    *reinterpret_cast<__half2*>(&d[base])     = h2a;
                    *reinterpret_cast<__half2*>(&d[base + 2]) = h2b;
                } else if (op >= UOP_ADD && op <= UOP_DIV) {
                    #pragma unroll
                    for (int e = 0; e < 4; e++)
                        regs[e][dst] = apply_uop_binary(op, regs[e][s1], regs[e][s2]);
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; e++)
                        regs[e][dst] = apply_uop_unary(op, regs[e][s1]);
                }
            }
        }

        // Scalar tail
        for (int idx = vec_end + (int)threadIdx.x; idx < end; idx += threads) {
            float regs[16];
            for (int u = 0; u < actual_uops; u++)
                exec_scalar_uop(ops[u], regs, dsts[u], s1s[u], s2s[u], idx,
                                task, buffers);
        }

    } else if (reduce_at >= 0) {
        // ====== REDUCE PATH: scalar only ======
        const int rd_dst = dsts[reduce_at];
        const int rd_src = s1s[reduce_at];
        const int rd_op  = ops[reduce_at];
        float partial = (rd_op == UOP_REDUCE_MAX) ? -1e30f : 0.0f;

        // Phase 1: execute pre-reduce uops, accumulate
        for (int idx = start + (int)threadIdx.x; idx < end; idx += threads) {
            float regs[16];
            for (int u = 0; u < reduce_at; u++)
                exec_scalar_uop(ops[u], regs, dsts[u], s1s[u], s2s[u], idx,
                                task, buffers);
            if (rd_op == UOP_REDUCE_MAX)
                partial = fmaxf(partial, regs[rd_src]);
            else
                partial += regs[rd_src];
        }

        // Block reduce — use dynamic SMEM to avoid shifting extern __shared__ base
        extern __shared__ char smem_fe[];
        float* _rs = (float*)smem_fe;
        if (rd_op == UOP_REDUCE_MAX)
            partial = block_reduce_max(partial, _rs);
        else
            partial = block_reduce_sum(partial, _rs);
        if (rd_op == UOP_REDUCE_MEAN)
            partial /= (float)numel;

        // Phase 2: if post-reduce uops exist, re-execute all uops per element
        // with the reduce result injected as a scalar register
        bool has_post = (reduce_at + 1 < actual_uops);
        if (has_post) {
            for (int idx = start + (int)threadIdx.x; idx < end; idx += threads) {
                float regs[16];
                for (int u = 0; u < actual_uops; u++) {
                    if (u == reduce_at) {
                        regs[rd_dst] = partial;
                        continue;
                    }
                    exec_scalar_uop(ops[u], regs, dsts[u], s1s[u], s2s[u],
                                    idx, task, buffers);
                }
            }
        } else if (threadIdx.x == 0) {
            // Reduce is last op — write scalar to output[0]
            __half* d = (__half*)buffers[task.buffer_indices[0]];
            d[0] = __float2half(partial);
        }

    } else {
        // ====== BROADCAST PATH: scalar only, with LOAD_BROADCAST ======
        for (int idx = start + (int)threadIdx.x; idx < end; idx += threads) {
            float regs[16];
            for (int u = 0; u < actual_uops; u++)
                exec_scalar_uop(ops[u], regs, dsts[u], s1s[u], s2s[u], idx,
                                task, buffers);
        }
    }
}
