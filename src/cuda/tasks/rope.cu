#include "../data_types.cuh"
#include <cuda_fp16.h>

__device__ void task_rope(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id) {
    __half* out = (__half*)buffers[task.buffer_indices[0]];
    const __half* in0 = (const __half*)buffers[task.buffer_indices[1]];
    const __half* cos_cache = (const __half*)buffers[task.buffer_indices[2]];
    const __half* sin_cache = (const __half*)buffers[task.buffer_indices[3]];

    const uint32_t batch = task.dimensions[0];
    const uint32_t seq_len = task.dimensions[1];
    const uint32_t num_heads = task.dimensions[2];
    const uint32_t head_dim = task.dimensions[3];
    const uint32_t half_dim = head_dim / 2;
    const int threads = blockDim.x;
    const int num_tiles = min((int)task.num_tiles, (int)gridDim.x);
    const int layout_bhsd = task.strides[0];
    const uint32_t cos_stride = task.dimensions[4] ? task.dimensions[4] : half_dim;

    const uint32_t total_positions = batch * seq_len;
    const uint32_t pos_per_tile = (total_positions + num_tiles - 1) / num_tiles;
    const uint32_t pos_start = tile_id * pos_per_tile;
    const uint32_t pos_end = min(pos_start + pos_per_tile, total_positions);

    const uint32_t half_dim_v = half_dim & ~7u;

    for (uint32_t pos = pos_start; pos < pos_end; pos++) {
        uint32_t b = pos / seq_len;
        uint32_t s = pos % seq_len;
        for (uint32_t h = 0; h < num_heads; h++) {
            uint32_t base_offset;
            if (layout_bhsd) {
                base_offset = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim + s * head_dim;
            } else {
                base_offset = pos * num_heads * head_dim + h * head_dim;
            }

            const __half* x0_ptr = in0 + base_offset;
            const __half* x1_ptr = in0 + base_offset + half_dim;
            __half* out0_ptr = out + base_offset;
            __half* out1_ptr = out + base_offset + half_dim;
            const __half* cos_ptr = cos_cache + s * cos_stride;
            const __half* sin_ptr = sin_cache + s * cos_stride;

            for (uint32_t d = threadIdx.x * 8; d < half_dim_v; d += threads * 8) {
                float4 x0_4 = *((const float4*)(x0_ptr + d));
                float4 x1_4 = *((const float4*)(x1_ptr + d));
                float4 c_4  = *((const float4*)(cos_ptr + d));
                float4 s_4  = *((const float4*)(sin_ptr + d));

                __half2* x0h = (__half2*)&x0_4;
                __half2* x1h = (__half2*)&x1_4;
                __half2* ch  = (__half2*)&c_4;
                __half2* sh  = (__half2*)&s_4;

                float4 o0_4, o1_4;
                __half2* o0h = (__half2*)&o0_4;
                __half2* o1h = (__half2*)&o1_4;

                for (int p = 0; p < 4; p++) {
                    float2 v0 = __half22float2(x0h[p]);
                    float2 v1 = __half22float2(x1h[p]);
                    float2 cv = __half22float2(ch[p]);
                    float2 sv = __half22float2(sh[p]);

                    o0h[p] = __float22half2_rn(make_float2(
                        v0.x * cv.x - v1.x * sv.x,
                        v0.y * cv.y - v1.y * sv.y));
                    o1h[p] = __float22half2_rn(make_float2(
                        v1.x * cv.x + v0.x * sv.x,
                        v1.y * cv.y + v0.y * sv.y));
                }

                *((float4*)(out0_ptr + d)) = o0_4;
                *((float4*)(out1_ptr + d)) = o1_4;
            }

            for (uint32_t d = half_dim_v + threadIdx.x; d < half_dim; d += threads) {
                float x0 = __half2float(x0_ptr[d]);
                float x1 = __half2float(x1_ptr[d]);
                float c = __half2float(cos_ptr[d]);
                float sv = __half2float(sin_ptr[d]);
                out0_ptr[d] = __float2half(x0 * c - x1 * sv);
                out1_ptr[d] = __float2half(x1 * c + x0 * sv);
            }
        }
    }
}
