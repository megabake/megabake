# Phase 0: Contained Per-Task Profiling

## Context

Need per-task cycle-level profiling inside the megakernel to measure where time goes (which op types dominate, SM utilization, barrier overhead). This data drives priority order for all subsequent optimization phases. User wants minimal footprint on main codebase — profiling contained in its own module with one-liners at integration points.

## Design: 2 files created, 2 files get small changes

### New files

**1. `src/cuda/profiling.cuh`** — CUDA-side timing macros (~10 lines)

```cuda
#pragma once

// Record clock64() delta per task per SM.
// Gated by task_timings != nullptr — zero effective overhead when disabled
// (branch predictor learns after first task).

#define PROFILE_TASK_BEGIN(task_id, sm_id, timings) \
    long long _prof_t0 = 0; \
    if ((timings) && threadIdx.x == 0) _prof_t0 = clock64();

#define PROFILE_TASK_END(task_id, sm_id, timings, grid_dim) \
    if ((timings) && threadIdx.x == 0) \
        (timings)[(task_id) * (grid_dim) + (sm_id)] = clock64() - _prof_t0;
```

No compile flag needed. Always compiled in. Cost when profiling off: 2 null-pointer checks per task per SM (predicted-not-taken after first iteration). Negligible vs `grid.sync()` cost.

**2. `src/megabake/runtime/profiler.py`** — Python profiler module (~120 lines)

Self-contained. Owns everything:
- Device buffer allocation (`num_tasks * num_sms * 8` bytes)
- Post-kernel data collection (copy back to host)
- Analysis: per-task table, per-op-type summary, barrier cost, SM utilization
- JSON save to `benchmarks/baselines/`

```python
class TaskProfiler:
    def __init__(self, num_tasks, num_sms):
        """Allocate device buffer for timing data."""

    @property
    def timings_ptr(self) -> int:
        """Device pointer to pass as kernel arg."""

    def collect(self, tasks) -> TaskProfileResult:
        """Copy back, compute per-task and per-op-type stats."""

    def print_report(self, result):
        """Pretty-print table:
        task_id | op_type | dims | median_cycles | max_cycles | active_sms | idle_sms | barrier_wait
        """

    def save_json(self, result, path):
        """Save baseline JSON for before/after comparison."""
```

### Changes to existing files

**1. `src/cuda/megakernel.cu`** — 5 lines added

```diff
 #include <cooperative_groups.h>
 #include "data_types.cuh"
+#include "profiling.cuh"

 extern "C"
 __global__ void __launch_bounds__(256, 1) megakernel(
     const TaskDesc* __restrict__ tasks,
     int num_tasks,
     void** __restrict__ buffers,
-    const int* __restrict__ dyn_dims
+    const int* __restrict__ dyn_dims,
+    long long* task_timings,
+    int enable_timing
 ) {
     // ...
     for (int i = 0; i < num_tasks; i++) {
         const TaskDesc& task = tasks[i];
+        PROFILE_TASK_BEGIN(i, sm_id, task_timings);
         if (sm_id < static_cast<int>(task.num_tiles)) {
             dispatch_task(task, buffers, dyn_dims, sm_id);
         }
+        PROFILE_TASK_END(i, sm_id, task_timings, gridDim.x);
         grid.sync();
     }
 }
```

**2. `src/megabake/runtime/launcher.py`** — ~5 lines in `_launch_cooperative()`

Add 2 optional params (`task_timings_ptr=0`, `enable_timing=0`). Pack as args 5-6 in the ctypes array. When both are 0, kernel sees `nullptr` and skips profiling.

```diff
 def _launch_cooperative(
     d_tasks_ptr, num_tasks, d_buffers_ptr, d_dyn_dims_ptr, num_sms,
+    task_timings_ptr=0, enable_timing=0,
 ):
     # ... existing setup ...
+    arg_timings = ctypes.c_void_p(task_timings_ptr)
+    arg_enable = ctypes.c_int(enable_timing)
-    args = (ctypes.c_void_p * 4)(...)
+    args = (ctypes.c_void_p * 6)(
         ...,  # existing 4
+        ctypes.cast(ctypes.pointer(arg_timings), ctypes.c_void_p),
+        ctypes.cast(ctypes.pointer(arg_enable), ctypes.c_void_p),
     )
```

### Test harness integration

**`benchmarks/test_harness.py`** — ~15 lines

- Add `--task-profile` flag to argparser
- In `run_megabake()`: when flag set, create `TaskProfiler`, pass `timings_ptr` through to launcher, collect and print after kernel completes

This touches the test harness, not the core library — fully contained.

## What profiler reports

```
task_id | op_type      | dims         | median_cycles | max_cycles | active_sms | idle_sms | barrier_wait
--------|--------------|--------------|---------------|------------|------------|----------|-------------
0       | MATMUL       | 1x576x1536   | 42000         | 43200      | 32         | 0        | 1200
1       | REDUCE       | 1x576        | 1800          | 2100       | 1          | 31       | 300
2       | ELEMENTWISE  | 576          | 450           | 510        | 1          | 31       | 60

Summary by op_type:
  MATMUL:      65.2%  (avg 42000 cycles, 32/32 SMs)
  REDUCE:      12.1%  (avg 1800 cycles, 1/32 SMs)
  ATTENTION:   18.4%  (avg 28000 cycles, 32/32 SMs)
  ...

Total barrier wait: ~250us
Total SM idle time: ~2000us
```

## What we learn

- Which op types dominate runtime (validates/refutes REDESIGN.md claims)
- SM utilization per task (confirms "97% idle on norms" or not)
- Barrier cost per task (max - median across SMs)
- Total idle time (sum of barrier_wait * idle_sm_count)
- Whether attention or matmul dominates on real models

## Footprint summary

| File | Change |
|------|--------|
| **New:** `profiling.cuh` | ~10 lines (2 macros) |
| **New:** `profiler.py` | ~120 lines (allocate/collect/report/save) |
| `megakernel.cu` | +1 include, +2 macro lines, +2 kernel params |
| `launcher.py` | +5 lines (2 optional params, 2 arg packing lines) |
| `test_harness.py` | +15 lines (flag + profiler integration) |
| `cuda_compiler.py` | 0 changes |
| `loader.py` | 0 changes |
| `__init__.py` | 0 changes |

## Verification

```bash
# Normal run (no profiling, identical behavior to before):
python benchmarks/test_harness.py mlp_silu

# Profiled run:
python benchmarks/test_harness.py mlp_silu --task-profile

# Should see per-task timing table
# Output values must match (profiling doesn't affect computation)
```
