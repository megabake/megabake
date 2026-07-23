#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_fused_elementwise(const TaskDesc& task, void** buffers,
                                       const int* dyn_dims, int tile_id) {
    const int numel = task.dimensions[0];
    const int num_uops = task.dimensions[1];

    const int per_sm = (numel + (int)task.num_tiles - 1) / (int)task.num_tiles;
    const int start = tile_id * per_sm;
    const int end = min(start + per_sm, numel);

    for (int idx = start + (int)threadIdx.x; idx < end; idx += (int)blockDim.x) {
        float regs[8];

        for (int u = 0; u < num_uops && u < 8; u++) {
            const uint32_t uop = (uint32_t)task.strides[u];
            const int op  = uop & 0xFF;
            const int dst = (uop >> 8)  & 0xF;
            const int s1  = (uop >> 12) & 0xF;
            const int s2  = (uop >> 16) & 0xF;

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
            }
        }
    }
}
