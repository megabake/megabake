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
                            const int* dyn_dims, int tile_id);
__device__ void task_attention(const TaskDesc& task, void** buffers,
                               const int* dyn_dims, int tile_id);
__device__ void task_elementwise(const TaskDesc& task, void** buffers,
                                 const int* dyn_dims, int tile_id);
__device__ void task_reduce(const TaskDesc& task, void** buffers,
                             const int* dyn_dims, int tile_id);
__device__ void task_embedding(const TaskDesc& task, void** buffers,
                                const int* dyn_dims, int tile_id);
__device__ void task_index(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id);
__device__ void task_copy(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id);
__device__ void task_rope(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id);
__device__ void task_fused_elementwise(const TaskDesc& task, void** buffers,
                                       const int* dyn_dims, int tile_id);

__device__ __forceinline__ void dispatch_task(
    const TaskDesc& task, void** buffers, const int* dyn_dims, int tile_id
) {
    switch (task.op_type) {
        case OP_MATMUL:      task_matmul(task, buffers, dyn_dims, tile_id);      break;
        case OP_ATTENTION:   task_attention(task, buffers, dyn_dims, tile_id);   break;
        case OP_ELEMENTWISE: task_elementwise(task, buffers, dyn_dims, tile_id); break;
        case OP_REDUCE:      task_reduce(task, buffers, dyn_dims, tile_id);      break;
        case OP_EMBEDDING:   task_embedding(task, buffers, dyn_dims, tile_id);   break;
        case OP_INDEX:       task_index(task, buffers, dyn_dims, tile_id);       break;
        case OP_COPY:        task_copy(task, buffers, dyn_dims, tile_id);        break;
        case OP_ROPE:        task_rope(task, buffers, dyn_dims, tile_id);        break;
        case OP_FUSED_ELEMENTWISE: task_fused_elementwise(task, buffers, dyn_dims, tile_id); break;
    }
}

extern "C"
__global__ void __launch_bounds__(256, 1) megakernel(
    const TaskDesc* __restrict__ tasks,
    int num_tasks,
    void** __restrict__ buffers,
    const int* __restrict__ dyn_dims,
    long long* task_timings
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int sm_id = blockIdx.x;

    for (int i = 0; i < num_tasks; i++) {
        const TaskDesc& task = tasks[i];

        PROFILE_TASK_BEGIN(task_timings);

        if (sm_id < static_cast<int>(task.num_tiles)) {
            dispatch_task(task, buffers, dyn_dims, sm_id);
        }

        PROFILE_TASK_END(i, sm_id, task_timings, gridDim.x);

        grid.sync();
    }
}
