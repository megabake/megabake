#include <cooperative_groups.h>
#include "data_types.cuh"

#define PROFILE_TASK_BEGIN(timings) \
    long long _prof_t0 = 0; \
    if ((timings) && threadIdx.x == 0) _prof_t0 = clock64();

#define PROFILE_TASK_END(task_idx, sm_id, timings, grid_dim) \
    if ((timings) && threadIdx.x == 0) \
        (timings)[(task_idx) * (grid_dim) + (sm_id)] = clock64() - _prof_t0;

// Task function declarations
__device__ void task_matmul(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_attention(const TaskDesc& task, void** buffers,
                               const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_elementwise(const TaskDesc& task, void** buffers,
                                 const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_reduce(const TaskDesc& task, void** buffers,
                             const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_embedding(const TaskDesc& task, void** buffers,
                                const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_index(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_copy(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_rope(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id, uint32_t dispatch_flags);
__device__ void task_fused_elementwise(const TaskDesc& task, void** buffers,
                                       const int* dyn_dims, int tile_id, uint32_t dispatch_flags);

__device__ __forceinline__ void dispatch_task(
    const TaskDesc& task, void** buffers, const int* dyn_dims, int tile_id,
    uint32_t dispatch_flags
) {
    switch (task.op_type) {
        case OP_MATMUL:      task_matmul(task, buffers, dyn_dims, tile_id, dispatch_flags);      break;
        case OP_ATTENTION:   task_attention(task, buffers, dyn_dims, tile_id, dispatch_flags);   break;
        case OP_ELEMENTWISE: task_elementwise(task, buffers, dyn_dims, tile_id, dispatch_flags); break;
        case OP_REDUCE:      task_reduce(task, buffers, dyn_dims, tile_id, dispatch_flags);      break;
        case OP_EMBEDDING:   task_embedding(task, buffers, dyn_dims, tile_id, dispatch_flags);   break;
        case OP_INDEX:       task_index(task, buffers, dyn_dims, tile_id, dispatch_flags);       break;
        case OP_COPY:        task_copy(task, buffers, dyn_dims, tile_id, dispatch_flags);        break;
        case OP_ROPE:        task_rope(task, buffers, dyn_dims, tile_id, dispatch_flags);        break;
        case OP_FUSED_ELEMENTWISE: task_fused_elementwise(task, buffers, dyn_dims, tile_id, dispatch_flags); break;
    }
}

extern "C"
__global__ void __launch_bounds__(256, 1) megakernel(
    const TaskDesc* __restrict__ tasks,
    int num_tasks,
    void** __restrict__ buffers,
    const int* __restrict__ dyn_dims,
    long long* task_timings,
    const SMQueueEntry* __restrict__ sm_queues,
    const int* __restrict__ sm_queue_lens,
    int* dep_count,
    int* tile_remaining,
    const int* __restrict__ succ_list,
    const int* __restrict__ succ_offset,
    int max_queue_len,
    int scheduler_type
) {
    const int sm_id = blockIdx.x;

    if (scheduler_type == 0) {
        namespace cg = cooperative_groups;
        cg::grid_group grid = cg::this_grid();

        for (int i = 0; i < num_tasks; i++) {
            const TaskDesc& task = tasks[i];
            PROFILE_TASK_BEGIN(task_timings);
            if (sm_id < static_cast<int>(task.num_tiles))
                dispatch_task(task, buffers, dyn_dims, sm_id, 0);
            PROFILE_TASK_END(i, sm_id, task_timings, gridDim.x);
            grid.sync();
        }
        return;
    }

    extern __shared__ char smem[];

    const int queue_len = sm_queue_lens[sm_id];
    bool prefetch_pending = false;

    for (int q = 0; q < queue_len; q++) {
        SMQueueEntry entry = sm_queues[sm_id * max_queue_len + q];
        int tid = entry.task_id;
        int tile = (int)(entry.tile_id & TILE_ID_MASK);

        if (threadIdx.x == 0)
            while (atomicAdd(&dep_count[tid * CACHE_LINE_INTS], 0) != 0) {}

        if (prefetch_pending) {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();

        uint32_t dflags = 0;
        if (prefetch_pending)
            dflags |= DISPATCH_PREFETCHED;
        if (entry.tile_id & QFLAG_HANDOFF)
            dflags |= DISPATCH_HANDOFF;

        PROFILE_TASK_BEGIN(task_timings);
        dispatch_task(tasks[tid], buffers, dyn_dims, tile, dflags);
        PROFILE_TASK_END(tid, sm_id, task_timings, gridDim.x);

        __syncthreads();
        __threadfence();

        if (threadIdx.x == 0) {
            int rem = atomicSub(&tile_remaining[tid * CACHE_LINE_INTS], 1);
            if (rem == 1) {
                for (int s = succ_offset[tid]; s < succ_offset[tid + 1]; s++)
                    atomicSub(&dep_count[succ_list[s] * CACHE_LINE_INTS], 1);
            }
        }

        prefetch_pending = false;
        if (q + 1 < queue_len) {
            SMQueueEntry next = sm_queues[sm_id * max_queue_len + q + 1];
            if (next.prefetch_buf_idx != 0xFFFFFFFFu && next.prefetch_bytes > 0) {
                int next_tid = next.task_id;
                const TaskDesc& nt = tasks[next_tid];
                int M = nt.dimensions[0];
                int K = nt.dimensions[2];
                int N = nt.dimensions[1];
                int active_tiles = min(max((int)nt.num_tiles, 1), (int)gridDim.x);
                int cols_per_tile = (N + active_tiles - 1) / active_tiles;
                int next_tile = (int)(next.tile_id & TILE_ID_MASK);
                int col_start = next_tile * cols_per_tile;

                // Match skinny matmul bk computation
                int cols = (int)blockDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
                const int SMEM_BUD = 220 * 1024;
#else
                const int SMEM_BUD = 96 * 1024;
#endif
                int bk = SMEM_BUD / (2 * (M + 2 * cols));
                bk = bk & ~7;
                if (bk > K) bk = K;
                if (bk < 8) bk = 8;

                int smem_b_offset = M * bk * 2;
                char* dst = smem + smem_b_offset;
                int n_end = min(col_start + cols_per_tile, N);
                int actual_cols = min(cols, n_end - col_start);
                const char* src = (const char*)buffers[next.prefetch_buf_idx]
                                  + (int64_t)col_start * K * 2;
                int bytes = min((int)next.prefetch_bytes, actual_cols * bk * 2);

                for (int t = (int)threadIdx.x * 16; t < bytes; t += (int)blockDim.x * 16) {
                    uint32_t s_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(dst + t));
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                                 :: "r"(s_addr), "l"(src + t));
                }
                asm volatile("cp.async.commit_group;\n");
                prefetch_pending = true;
            }
        }
    }
}
