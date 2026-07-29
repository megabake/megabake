# Megabake Redesign: Profile-Driven Incremental Plan

## Context

REDESIGN.md proposes 15 optimizations across 7 stages over 15+ weeks. Problem: priorities ranked by theoretical analysis, not measured data. Some claims are factually wrong. We need profiling data FIRST, then let numbers drive priority — but paper-validated wins with low risk (cp.async, weight transpose) proceed without waiting.

Codebase is ~3,650 lines (2,367 Python + 1,279 CUDA). Goal: simplify while improving. Each phase independently valuable, profiles before and after, commits atomically.

### REDESIGN.md Errors Found

1. **"Skinny matmul has element-by-element loads"** — Wrong. `src/cuda/tasks/matmul.cu:100-106` already loads B (weights) with `float4`:
   ```cuda
   float4 bv = *(const float4*)(b_ptr + k);
   ```
   A (activation) is loaded element-by-element into SMEM, but A is tiny at M<=4 (1-4 rows × K). Weight loading IS vectorized. What's missing: `cp.async` prefetch overlap (load next K-chunk while computing current).

2. **"No vectorization in reduce/rope/index"** — Correct. Confirmed by reading code. These DO use element-by-element `__half` loads.

3. **"Attention has no tensor cores"** — Correct. `attention.cu:86-117` uses scalar FMA for Q*K^T dot products. Has float4 loads for vectorized dot product path but unpacks to scalar floats — no HMMA/WGMMA instructions.

---

## Current Architecture (what exists today)

```
Python compiler (2,367 lines):
  __init__.py:11      compile()                  Entry point: Module -> CompiledModel
  __init__.py:28      run()                      Entry point: CompiledModel -> output
  graph_walker.py:782 compile_model()            FX graph -> TaskDesc list
  graph_walker.py:154 _fuse_tasks()              MATMUL+SILU/GELU epilogue fusion
  graph_walker.py:226 _fuse_elementwise_chains() Elementwise chain -> micro-op program
  graph_walker.py:344 _eliminate_redundant_copies() Redundant COPY elimination
  graph_walker.py:499 _find_rmsnorm()            Decomposed RMSNorm pattern detection
  graph_walker.py:660 _find_rope()               Decomposed RoPE pattern detection
  buffer_planner.py   Liveness + first-fit-decreasing arena packing
  tiling.py           Per-task tile count (fixed rules)
  serializer.py       Binary schedule encoding
  op_table.py         ATen op -> (OpType, op_code) mapping

CUDA runtime (1,279 lines):
  megakernel.cu:44    Kernel entry (__launch_bounds__(256,1))
  megakernel.cu:55-63 BSP dispatch loop (grid.sync per task)
  matmul.cu:61-134    Skinny matvec (M<=4): N-split, float4 loads, CUDA cores
  matmul.cu:136-338   SM90 CuTe GEMM: WGMMA 128x128x64, 3-stage cp.async pipeline
  matmul.cu:342-574   SM80 CuTe GEMM: mma.sync 128x128x32, 2-stage pipeline
  matmul.cu:46-56     apply_epilogue(): SiLU/GELU/GELU_TANH activation
  matmul.cu:13-44     matmul_scalar(): fallback when B not transposed
  attention.cu        185 lines: materialized scores, scalar FMA, O(seq_k) SMEM
  elementwise.cu      183 lines: vectorized (float4) unary/binary ops
  fused_elementwise.cu 188 lines: micro-op interpreter
  reduce.cu           212 lines: RMSNorm/LayerNorm/Softmax/Sum/Mean/Max/ArgMax
  rope.cu             47 lines: rotary embeddings
  index.cu            45 lines: gather/scatter/index_select
  embedding.cu        38 lines: embedding lookup
  copy.cu             75 lines: flat copy + cat

Host runtime:
  launcher.py:136     _launch_cooperative() (4 kernel args: tasks, num_tasks, buffers, dyn_dims)
  launcher.py:171     cuLaunchCooperativeKernel (num_sms blocks, 256 threads)
  loader.py:123       Schedule deserialization + execution
  cuda_compiler.py    nvcc compilation (all .cu concatenated into one TU)
```

### Current Op Types (12)

```python
# src/megabake/data_types.py:6-18
MATMUL=0x01, ATTENTION=0x02, ELEMENTWISE=0x03, REDUCE=0x04,
EMBEDDING=0x05, INDEX=0x06, COPY=0x07, ROPE=0x08,
MATMUL_SILU=0x09, MATMUL_GELU=0x0A, FUSED_ELEMENTWISE=0x0B,
MATMUL_GELU_TANH=0x0C
```

### Current TaskDesc Struct

```c
// src/cuda/data_types.cuh:84-91
struct __align__(16) TaskDesc {
    uint16_t op_type;        // OpType enum
    uint16_t op_code;        // sub-operation code
    uint32_t num_tiles;      // tiles for this task
    uint32_t buffer_indices[8]; // indices into buffers[] pointer array
    uint32_t dimensions[8];  // M, N, K, etc.
    int32_t  strides[8];     // strides + epilogue flags (strides[1] available)
};
// 56 bytes, 16-byte aligned
// buffer_indices[3-7] available for bias/residual/etc.
// strides[1] available for epilogue flags
```

### Current Megakernel Loop (BSP)

```cuda
// src/cuda/megakernel.cu:55-63
for (int i = 0; i < num_tasks; i++) {
    const TaskDesc& task = tasks[i];
    if (sm_id < static_cast<int>(task.num_tiles))
        dispatch_task(task, buffers, dyn_dims, sm_id);
    grid.sync();  // ALL SMs wait here, even idle ones
}
```

Problem: 1-tile task (e.g., RMSNorm) uses 1 SM, other 31 SMs idle at barrier.

### Current Skinny Matmul (M<=4)

```cuda
// src/cuda/tasks/matmul.cu:61-134
// N-split: each SM owns cols_per_tile columns
// K-chunked: bk = min(K, 51200/max(M,1)), A cached in SMEM per chunk
// Weight loads: float4 (lines 100-106)
// Activation loads: float4 from SMEM (lines 108-113)
// Accumulation: scalar FMA on unpacked half2 pairs (lines 114-117)
// NO cp.async — loads are synchronous, no overlap between K-chunks
```

---

## Phase 0: Per-Task Profiling

**Goal:** Measure where time goes inside the megakernel. Every subsequent priority decision depends on this data.

**Why this comes first:** REDESIGN.md estimates "~2000us SM idle time" and "~250us barrier cost" for SmolLM2-135M. These are guesses. We need numbers.

### What to change

**`src/cuda/megakernel.cu`** — Add timing around dispatch, guarded by compile flag:

```cuda
// New kernel signature adds: long long* task_timings, int enable_timing
// task_timings layout: [num_tasks * gridDim.x] — per-SM, per-task

for (int i = 0; i < num_tasks; i++) {
    long long t_start = 0;
    if (enable_timing && threadIdx.x == 0)
        t_start = clock64();

    if (sm_id < task.num_tiles)
        dispatch_task(task, ...);

    if (enable_timing && threadIdx.x == 0)
        task_timings[i * gridDim.x + sm_id] = clock64() - t_start;

    grid.sync();
}
```

This captures:
- Per-task execution time on each SM
- Which SMs were active vs idle (idle SMs record ~0 cycles between start/sync)
- Barrier wait time (difference between fastest and slowest SM per task)

**`src/megabake/runtime/launcher.py`** — When profiling enabled:
1. Allocate device buffer: `num_tasks * num_sms * sizeof(int64)` bytes
2. Pass as extra kernel arg (args grows from 4 to 6: +task_timings, +enable_timing)
3. After kernel: cudaMemcpy back, return alongside output

**`benchmarks/test_harness.py`** — Add `--task-profile` flag:
- Call `megabake.run()` with profiling enabled
- Print table: task_id, op_type, op_code, dimensions, median_cycles, max_cycles, active_sms, idle_sms
- Group by op_type for summary
- Save JSON to `benchmarks/baselines/`

### How to verify

```bash
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
python benchmarks/test_harness.py mlp_silu --task-profile
# Should see per-task breakdown with cycle counts
```

### What we learn

- Which op_types dominate runtime (validates REDESIGN.md Section 2 rankings)
- SM utilization per task (validates "97% idle on norms" claim)
- Barrier overhead (time between last SM finishing and sync completing)
- Whether attention or matmul dominates on real models

---

## Phase 1: Vectorize Reduce/Rope/Index

**Goal:** Replace element-by-element `__half` loads with `float4` (8 halves per load). Single-pass RMSNorm. Lowest risk, highest confidence optimization.

**Why:** Each `__half` load is 2 bytes. `float4` is 16 bytes = 8 halves. Same memory bus, 8x fewer instructions. 256 threads × 2 bytes = 512 bytes/cycle vs 256 threads × 16 bytes = 4096 bytes/cycle.

### 1a: reduce.cu — Vectorize + Single-Pass RMSNorm

**Current RMSNorm** (`reduce.cu:71-91`):
```cuda
// Pass 1: compute sum of squares (lines 73-76)
for (j = threadIdx.x; j < row_size; j += threads) {
    float v = __half2float(row_in[j]);  // <-- element-by-element
    sum_sq += v * v;
}
sum_sq = block_reduce_sum(sum_sq, smem);
float rms = rsqrtf(sum_sq / row_size + eps);

// Pass 2: normalize (lines 83-91, re-reads entire row)
for (j = threadIdx.x; j < row_size; j += threads) {
    float v = __half2float(row_in[j]) * rms;  // <-- re-reads same data
    if (weight) v *= __half2float(weight[j]);
    row_out[j] = __float2half(v);
}
```

Two problems: (1) element-by-element loads, (2) reads row twice.

**New RMSNorm** — single pass, float4, row cached in registers:

```cuda
// For hidden <= 4096 with 256 threads: 16 elements/thread = 16 fp32 registers
float cached[MAX_ELEMS_PER_THREAD];
float sum_sq = 0.0f;
int my_count = 0;

// Single vectorized load pass
for (j = threadIdx.x * 8; j < row_size_aligned; j += threads * 8) {
    float4 v4 = *(const float4*)(row_in + j);
    __half2* vh = (__half2*)&v4;
    for (int p = 0; p < 4; p++) {
        float2 vf = __half22float2(vh[p]);
        cached[my_count++] = vf.x;
        cached[my_count++] = vf.y;
        sum_sq += vf.x * vf.x + vf.y * vf.y;
    }
}
// Handle scalar tail for unaligned remainder

sum_sq = block_reduce_sum(sum_sq, smem);
float rms = rsqrtf(sum_sq / row_size + eps);

// Write pass from cached values (no re-read from global memory)
my_count = 0;
for (j = threadIdx.x * 8; j < row_size_aligned; j += threads * 8) {
    float4 o4;
    __half2* oh = (__half2*)&o4;
    for (int p = 0; p < 4; p++) {
        float x = cached[my_count++] * rms;
        float y = cached[my_count++] * rms;
        if (weight) { /* apply weight via float4 load */ }
        oh[p] = __float22half2_rn(make_float2(x, y));
    }
    *(float4*)(row_out + j) = o4;
}
```

Register budget: hidden=4096, 256 threads = 16 elements/thread = 16 fp32 registers. Well within 255 register limit. For hidden=8192: 32 registers/thread, still fine.

**Apply same float4 pattern to:** LayerNorm (lines 94-120, currently 3 passes down to 2), Softmax (lines 122-141, currently 3 passes down to 2), Sum/Mean/Max (lines 143-172, straightforward float4).

**Where:** `src/cuda/tasks/reduce.cu` — rewrite all reduce paths with float4 + register caching where applicable.

### 1b: rope.cu — Vectorize

**Current** (`rope.cu:37-44`):
```cuda
for (d = threadIdx.x; d < half_dim; d += threads) {
    float x0 = __half2float(in0[base + d]);           // <-- scalar
    float x1 = __half2float(in0[base + d + half_dim]); // <-- scalar
    float c  = __half2float(cos_cache[s * stride + d]); // <-- scalar
    float sv = __half2float(sin_cache[s * stride + d]); // <-- scalar
    out[base + d]            = __float2half(x0*c - x1*sv);
    out[base + d + half_dim] = __float2half(x1*c + x0*sv);
}
```

**New** — load x0/x1/cos/sin as float4, compute on float2 pairs, store as float4:

```cuda
for (d = threadIdx.x * 4; d < half_dim_aligned; d += threads * 4) {
    float4 x0_4 = *(const float4*)(in0 + base + d);
    float4 x1_4 = *(const float4*)(in0 + base + d + half_dim);
    float4 c_4  = *(const float4*)(cos_cache + s * stride + d);
    float4 s_4  = *(const float4*)(sin_cache + s * stride + d);
    // Unpack half2 pairs, compute rotary, repack
    // ... (same math, float4 in/out)
}
// Scalar tail for unaligned remainder
```

4x fewer load/store instructions. Same computation.

**Where:** `src/cuda/tasks/rope.cu` — rewrite inner loop with float4.

### 1c: index.cu — Vectorize

**Current** (`index.cu:20-25`):
```cuda
for (i = start + threadIdx.x; i < end; i += threads) {
    uint32_t row = i / inner_size;
    uint32_t col = i % inner_size;
    out[i] = src[src_row * inner_size + col];  // <-- scalar half load
}
```

**New** — when `inner_size % 8 == 0`, use float4:

```cuda
if (inner_size % 8 == 0) {
    uint32_t inner_vec = inner_size / 8;
    for (i = start/8 + threadIdx.x; i < end/8; i += threads) {
        uint32_t row = i / inner_vec;
        uint32_t col = i % inner_vec;
        int64_t src_row = idx[row];
        ((float4*)out)[i] = ((const float4*)src)[src_row * inner_vec + col];
    }
} else {
    // scalar fallback (existing code)
}
```

**Where:** `src/cuda/tasks/index.cu` — add vectorized path, keep scalar fallback.

### How to verify

```bash
# Profile before (Phase 0 baselines exist)
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
# Make changes
# Profile after
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
# Compare reduce task cycle counts — expect 2-4x reduction
# Run correctness check
python benchmarks/test_harness.py rmsnorm_mlp --profile
# max_diff should be < 1e-3
```

---

## Phase 2: Consolidate Op Types (12 to 9)

**Goal:** Simplify dispatch. Remove redundant op_type categories. Smaller switch, less I-cache pressure, simpler compiler code.

**Why:** Three matmul variants (MATMUL_SILU=0x09, MATMUL_GELU=0x0A, MATMUL_GELU_TANH=0x0C) already call the same `task_matmul()` function — see `megakernel.cu:27-38`. The distinction exists only for `apply_epilogue()` (matmul.cu:46-56) which checks `task.op_type`. Moving to flag bits in `task.strides[1]` removes 3 op_types.

Similarly, COPY is just elementwise identity. FUSED_ELEMENTWISE is elementwise with a micro-op program. One op_type with a subtype flag covers all three.

### What to change

**`src/megabake/data_types.py`** — Shrink OpType enum:
```python
class OpType(IntEnum):
    MATMUL      = 0x01  # was also 0x09, 0x0A, 0x0C
    ATTENTION   = 0x02
    ELEMENTWISE = 0x03  # was also 0x07 (COPY), 0x0B (FUSED_ELEMENTWISE)
    REDUCE      = 0x04
    EMBEDDING   = 0x05
    INDEX       = 0x06
    ROPE        = 0x08

# Epilogue flags (stored in task.strides[1])
EPILOGUE_NONE      = 0x00
EPILOGUE_SILU      = 0x01
EPILOGUE_GELU      = 0x02
EPILOGUE_GELU_TANH = 0x04
EPILOGUE_BIAS      = 0x08   # Phase 3
EPILOGUE_RESIDUAL  = 0x10   # Phase 3

# Elementwise subtypes (stored in task.op_code upper byte, NOT strides[1])
# strides[] is reserved for fused elementwise micro-op program
ELEM_SUBTYPE_SIMPLE = 0x00
ELEM_SUBTYPE_FUSED  = 0x0100  # upper byte of op_code
ELEM_SUBTYPE_COPY   = 0x0200  # upper byte of op_code
```

**`src/cuda/data_types.cuh`** — Mirror above as `#define` constants.

**`src/cuda/megakernel.cu`** — Dispatch shrinks from 12 cases to 7:
```cuda
switch (task.op_type) {
    case OP_MATMUL:      task_matmul(task, ...);      break;
    case OP_ATTENTION:   task_attention(task, ...);   break;
    case OP_ELEMENTWISE: task_elementwise(task, ...); break;  // handles copy + fused
    case OP_REDUCE:      task_reduce(task, ...);      break;
    case OP_EMBEDDING:   task_embedding(task, ...);   break;
    case OP_INDEX:        task_index(task, ...);       break;
    case OP_ROPE:        task_rope(task, ...);        break;
}
```

**`src/cuda/tasks/matmul.cu`** — `apply_epilogue()` (lines 46-56) reads flags from `task.strides[1]` instead of `task.op_type`:
```cuda
__device__ __forceinline__ float apply_epilogue(float v, uint32_t flags) {
    if (flags & EPILOGUE_SILU)      return v / (1.0f + __expf(-v));
    if (flags & EPILOGUE_GELU)      return v * 0.5f * (1.0f + erff(v * 0.7071f));
    if (flags & EPILOGUE_GELU_TANH) { /* existing tanh approx */ }
    return v;
}
```

Update all call sites: SM90 epilogue (lines 326-334), SM80 epilogue (lines 561-569), skinny epilogue (line 131) — pass `task.strides[1]` instead of `task.op_type`.

**`src/cuda/tasks/elementwise.cu`** — Add routing at top:
```cuda
__device__ void task_elementwise(const TaskDesc& task, ...) {
    uint32_t subtype = task.op_code & 0xFF00;  // upper byte of op_code
    if (subtype == ELEM_SUBTYPE_FUSED) {
        task_fused_elementwise_impl(task, ...);
        return;
    }
    if (subtype == ELEM_SUBTYPE_COPY) {
        task_copy_impl(task, ...);
        return;
    }
    // existing elementwise logic
}
```

**`src/cuda/tasks/copy.cu`** — Gut to just the impl function, called from elementwise.

**`src/megabake/schedule_compiler/graph_walker.py`** — Where it currently emits `OpType.MATMUL_SILU` (around line 154-187 in `_fuse_tasks()`), emit `OpType.MATMUL` with `strides[1] = EPILOGUE_SILU`. Same for COPY and FUSED_ELEMENTWISE.

**`src/megabake/schedule_compiler/tiling.py`** — Remove matmul variant checks (lines 10-15 currently check MATMUL_SILU/GELU/GELU_TANH), COPY/FUSED_ELEMENTWISE checks. All fold into parent op_type.

**`src/cuda/tasks/embedding.cu`** — Add `__noinline__`:
```cuda
__device__ __noinline__ void task_embedding(...)
```

**`src/cuda/tasks/index.cu`** — Add `__noinline__`:
```cuda
__device__ __noinline__ void task_index(...)
```

### How to verify

```bash
python benchmarks/test_harness.py mlp_silu --profile        # correctness
python benchmarks/test_harness.py rmsnorm_mlp --profile     # correctness
python benchmarks/test_harness.py llama_decoder --profile   # correctness
# All max_diff should match baseline
# Kernel count unchanged (same tasks, different encoding)
```

---

## Phase 3: Matmul Epilogue Fusion — Bias + Residual

**Goal:** Eliminate separate bias-add and residual-add tasks by fusing into matmul writeback.

**Why:** A biased linear currently emits:
```
Task N:   MATMUL (C = A @ B)
Task N+1: ELEMENTWISE ADD (C += bias)    <-- extra task
          grid.sync()                     <-- extra barrier
```
Fusing: `C[i] = A@B[i] + bias[col]` in matmul epilogue. Saves ~5us per biased linear (task dispatch + barrier).

**Depends on:** Phase 2 (epilogue flag mechanism in `strides[1]`).

### What to change

**`src/cuda/tasks/matmul.cu`** — Extend epilogue in both SM80 and SM90 GEMM paths. The epilogue loop (SM90: lines 326-334, SM80: lines 561-569) currently does:
```cuda
for (int i = 0; i < size(tCrC); ++i) {
    if (elem_less(tCidC(i), make_shape(M, N))) {
        tCgC(i) = half_t(apply_epilogue(tCrC(i), op));
    }
}
```

New epilogue needs bias and residual buffer pointers. Use available TaskDesc slots:
- `buffer_indices[3]` = bias buffer (1D, length N)
- `buffer_indices[4]` = residual buffer (2D, M × N)

```cuda
uint32_t flags = task.strides[1];
const half_t* bias_ptr = (flags & EPILOGUE_BIAS)
    ? (const half_t*)buffers[task.buffer_indices[3]] : nullptr;
const half_t* res_ptr = (flags & EPILOGUE_RESIDUAL)
    ? (const half_t*)buffers[task.buffer_indices[4]] : nullptr;

for (int i = 0; i < size(tCrC); ++i) {
    if (elem_less(tCidC(i), make_shape(M, N))) {
        float v = tCrC(i);
        if (bias_ptr) {
            int col = get<1>(tCidC(i));
            v += __half2float(bias_ptr[col]);
        }
        v = apply_epilogue(v, flags);  // activation after bias
        if (res_ptr) {
            int row = get<0>(tCidC(i));
            int col = get<1>(tCidC(i));
            v += __half2float(res_ptr[row * N + col]);
        }
        tCgC(i) = half_t(v);
    }
}
```

Also extend `matmul_skinny` epilogue (line 131) similarly.

**`src/megabake/schedule_compiler/graph_walker.py`** — New fusion pattern in `_fuse_tasks()` (after existing matmul+activation fusion at line 154).

Prerequisites — build these helpers first (they don't exist yet):
- `find_consumers(task_idx, tasks, buffer_sizes)`: scan tasks[task_idx+1:] for any task whose `buffer_indices` references task's output buffer. Use `buffer_indices[2]` (output buffer) of the producer.
- `get_other_input(consumer, producer_output_buf)`: return the `buffer_indices` entry that is NOT the producer's output.
- `is_1d_bias(buffer_id, expected_size, buffer_sizes)`: check `buffer_sizes[buffer_id] == expected_size * 2` (fp16 = 2 bytes per element, 1D with length N).
- Task removal: mark task as no-op (set `num_tiles=0` or filter later), don't delete from list mid-iteration (breaks indices).

```python
# Pattern: MATMUL -> consumer is ELEMENTWISE ADD -> consumer's other input is 1D (bias)
for i, task in enumerate(tasks):
    if task.op_type != OpType.MATMUL:
        continue
    consumers = find_consumers(i, tasks, buffer_sizes)
    for ci, consumer in consumers:
        if consumer.op_type == OpType.ELEMENTWISE and consumer.op_code == ElemCode.ADD:
            other_buf = get_other_input(consumer, task.buffer_indices[2])
            if is_1d_bias(other_buf, task.dimensions[1], buffer_sizes):
                task.strides[1] |= EPILOGUE_BIAS
                task.buffer_indices[3] = other_buf
                tasks[ci].num_tiles = 0  # mark dead, filter later
```

Similar pattern for residual add (2D operand matching M × N — check `buffer_sizes[buf] == M * N * 2`).

### How to verify

```bash
# Need a model with bias. layernorm_mlp has LayerNorm which has bias.
python benchmarks/test_harness.py layernorm_mlp --task-profile
# Count tasks before: should see MATMUL + ELEMENTWISE ADD sequences
# Count tasks after: ELEMENTWISE ADD tasks should disappear
# max_diff should match baseline
```

---

## Phase 4: Skinny Matvec — cp.async Prefetch + Weight Pre-Transposition

**Goal:** Overlap weight loading with computation in the M<=4 decode path. Pre-transpose weights to column-major for coalesced reads.

**Why:** Current skinny matmul (matmul.cu:61-134) has float4 weight loads but NO software pipelining. It loads a K-chunk, computes, loads next K-chunk, computes — purely serial. The SM90 CuTe GEMM path (lines 284-323) already uses cp.async 3-stage pipeline for exactly this purpose, proving the technique works in this codebase. Applying it to the skinny path overlaps weight loading with FMA compute.

Weight pre-transposition: current skinny matvec assumes B is in (N,K) layout (transposed). Host-side pre-transposition to column-major ensures coalesced reads across warps.

**Independent of other phases.** Paper-validated: Hazy Research uses identical technique for decode matvec.

### What to change

**`src/cuda/tasks/matmul.cu`** — Rewrite `matmul_skinny()` (lines 61-134) with cp.async double-buffering:

```cuda
__device__ void matmul_skinny(const TaskDesc& task, void** buffers,
                               const int* dyn_dims, int tile_id) {
    // Existing: N-split across SMs, A in SMEM (lines 61-77) — keep
    // ...

    // SMEM: two weight buffers for double-buffering
    extern __shared__ char smem[];
    half* smem_A = (half*)smem;                    // A rows cached (existing)
    half* smem_B0 = smem_A + M * bk;              // weight chunk 0
    half* smem_B1 = smem_B0 + cols_this_tile * bk; // weight chunk 1

    // Prefetch first K-chunk into smem_B0
    for (int t = threadIdx.x; t < chunk_elems; t += blockDim.x) {
        cp_async_cg(&smem_B0[t], &B[offset + t]);
    }
    cp_async_commit_group();

    int cur_buf = 0;
    for (int k_start = 0; k_start < K; k_start += bk) {
        int next_k = k_start + bk;
        half* cur_B = (cur_buf == 0) ? smem_B0 : smem_B1;
        half* nxt_B = (cur_buf == 0) ? smem_B1 : smem_B0;

        // Start loading NEXT K-chunk (overlapped with compute below)
        if (next_k < K) {
            for (int t = threadIdx.x; t < chunk_elems; t += blockDim.x) {
                cp_async_cg(&nxt_B[t], &B[next_offset + t]);
            }
            cp_async_commit_group();
        }

        // Wait for CURRENT chunk
        cp_async_wait_group<1>();  // allow 1 group in flight
        __syncthreads();

        // Compute: float4 loads from SMEM, FMA accumulate (existing lines 100-118)
        for (int my_n = col_start + threadIdx.x; my_n < col_end; my_n += blockDim.x) {
            for (int k = 0; k < actual_bk; k += 8) {
                float4 bv = *(const float4*)(cur_B + (my_n - col_start) * bk + k);
                float4 av = *(const float4*)(smem_A + m * bk + k);
                // FMA (existing)
            }
        }

        cur_buf ^= 1;
    }
    // Store results (existing lines 125-133)
}
```

SMEM budget: M=1, K=4096, bk=4096 (fits in one chunk): A = 8 KB, B×2 = 2 × cols_per_sm × bk × 2. For 32 SMs with N=4096: cols_per_sm ≈ 128, B_chunk = 128 × 4096 × 2 = 1 MB — too large for SMEM. Must use smaller bk. With bk=256: B_chunk = 128 × 256 × 2 = 64 KB × 2 = 128 KB. Fits in 228 KB SMEM. Current code already computes appropriate bk (line 75): `bk = min(K, 51200 / max(M, 1))`.

**`src/megabake/runtime/launcher.py`** — Weight pre-transposition for decode:

```python
# In run_tasks() or weight loading path:
def _maybe_transpose_weights(self, task, weight_tensor):
    """Pre-transpose weight to column-major for decode (M<=4)."""
    if task.op_type == OpType.MATMUL and task.dimensions[0] <= 4:
        if not weight_tensor.is_contiguous():
            weight_tensor = weight_tensor.contiguous()
        # Store as (K, N) instead of (N, K) for coalesced column reads
        weight_tensor = weight_tensor.t().contiguous()
    return weight_tensor
```

Cache transposed weights — don't re-transpose on every call.

### How to verify

```bash
# Standalone matvec benchmark
python benchmarks/test_harness.py linear_256x512 --profile
# Compare M=1 matmul against Phase 0 baseline
# Measure bandwidth utilization: bytes_transferred / (time * peak_bandwidth)
# Target: >= 50% utilization (up from ~25%)

# Full model
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile
# max_diff < 1e-3
```

### Expected gain

M=1 matmul 1.5-2x faster from cp.async overlap alone. Weight pre-transposition adds another 10-20% on non-coalesced shapes. Combined: `linear_256x512` from 0.27x to ~0.45-0.55x vs cuBLAS.

---

## Phase 5: Counter-Based Scheduler

**Goal:** Replace BSP (grid.sync per task) with counter-based sync + static per-SM assignment. Eliminate barrier overhead and SM idling.

**Why:** Current BSP loop (megakernel.cu:55-63):
- ~100 tasks × 2.5us barrier = ~250us wasted on barriers
- 1-tile tasks (norms): 1 SM active, 31 idle, all pay barrier cost
- Independent tasks (Q/K/V projections) serialized unnecessarily

Counter-based sync: each SM walks its own queue, waits only on data dependencies, never on global barriers.

### Phase 5a: Dependency DAG + Counter-Based Sync

**`src/megabake/schedule_compiler/graph_walker.py`** — Extract dependency edges:

```python
def build_dependency_dag(tasks, buffer_producers, buffer_consumers):
    """Build DAG: task A -> task B if A produces a buffer B consumes.

    buffer_producers/consumers already tracked during FX walk
    (buffer_map, buffer_sizes at graph_walker.py:829-835).
    """
    edges = []
    for buf_id, producer in buffer_producers.items():
        for consumer in buffer_consumers.get(buf_id, []):
            if producer != consumer:
                edges.append((producer, consumer))

    dep_count = [0] * len(tasks)
    successors = [[] for _ in range(len(tasks))]
    for (src, dst) in edges:
        dep_count[dst] += 1
        successors[src].append(dst)

    return dep_count, successors
```

### Phase 5b: Static Per-SM Assignment with Critical-Path Priority

**`src/megabake/schedule_compiler/scheduler.py`** — NEW file:

```python
def _critical_path_lengths(tasks, successors):
    """Compute longest-path-to-exit for each task (critical path priority).

    Assumes task IDs are in topological order (graph_walker emits them this way).
    Iterating in reverse guarantees all successors are computed before predecessors.
    """
    n = len(tasks)
    cpl = [0] * n
    for task_id in reversed(range(n)):
        max_succ = 0
        for s in successors[task_id]:
            max_succ = max(max_succ, cpl[s])
        cpl[task_id] = estimate_cycles(tasks[task_id]) + max_succ
    return cpl

def _topo_sort_by_priority(dep_count, successors, priority):
    """Kahn's algorithm with max-priority tie-breaking.

    Returns task IDs in topological order, preferring highest-priority (longest
    critical path) tasks first when multiple tasks have zero dependencies.
    """
    import heapq
    n = len(dep_count)
    in_deg = list(dep_count)
    # Max-heap via negated priority
    ready = [(-priority[i], i) for i in range(n) if in_deg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, task_id = heapq.heappop(ready)
        order.append(task_id)
        for s in successors[task_id]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                heapq.heappush(ready, (-priority[s], s))
    return order

def assign_tasks_to_sms(tasks, dep_count, successors, num_sms):
    """Static per-SM task assignment via bin-packing.

    Critical-path priority: tasks on longest remaining chain scheduled first.
    Returns: sm_queues[sm_id] = [(task_id, tile_id), ...]
    """
    cpl = _critical_path_lengths(tasks, successors)
    sorted_tasks = _topo_sort_by_priority(dep_count, successors, cpl)

    sm_queues = [[] for _ in range(num_sms)]
    sm_load = [0] * num_sms

    for task_id in sorted_tasks:
        task = tasks[task_id]
        if task.num_tiles == 1:
            sm = min(range(num_sms), key=lambda s: sm_load[s])
            sm_queues[sm].append((task_id, 0))
            sm_load[sm] += estimate_cycles(task)
        else:
            n = min(task.num_tiles, num_sms)
            for tile in range(task.num_tiles):
                sm = tile % n
                sm_queues[sm].append((task_id, tile))
                sm_load[sm] += estimate_cycles(task) / n

    return sm_queues

def estimate_cycles(task):
    """Rough cost estimate for bin-packing. Doesn't need to be exact."""
    if task.op_type == OpType.MATMUL:
        M, N, K = task.dimensions[0], task.dimensions[1], task.dimensions[2]
        return 2 * M * N * K  # proportional to FLOPs
    if task.op_type == OpType.REDUCE:
        return task.dimensions[0] * task.dimensions[1]  # proportional to elements
    return task.dimensions[0] * 100  # generic fallback
```

### Phase 5c: New Data Structures

**`src/megabake/data_types.py`** — New structures:

```python
@dataclass
class SMQueueEntry:
    task_id: int
    tile_id: int
    STRUCT_FORMAT = "<II"
    STRUCT_SIZE = 8
```

Extend `ScheduleHeader` (mirrors data_types.cuh:108-122):
```python
@dataclass
class ScheduleV2Header(ScheduleHeader):
    num_sms: int = 0
    max_queue_len: int = 0
    num_edges: int = 0
    scheduler_type: int = 0  # 0=BSP, 1=counter
```

**`src/cuda/data_types.cuh`** — Mirror:
```cuda
struct SMQueueEntry {
    uint32_t task_id;
    uint32_t tile_id;
};
```

### Phase 5d: Device Execution Loop

**`src/cuda/megakernel.cu`** — New execution loop:

```cuda
extern "C"
__global__ void __launch_bounds__(256, 1) megakernel(
    const TaskDesc* __restrict__ tasks, int num_tasks,
    void** __restrict__ buffers, const int* __restrict__ dyn_dims,
    // New args for counter-based scheduler:
    const SMQueueEntry* sm_queues,
    const int* sm_queue_lens,
    int* dep_count,           // [num_tasks] dependency counters
    int* tile_remaining,      // [num_tasks] tile completion counters
    const int* succ_list,
    const int* succ_offset,
    int scheduler_type
) {
    if (scheduler_type == 0) {
        // BSP fallback (existing code, lines 55-63)
        namespace cg = cooperative_groups;
        cg::grid_group grid = cg::this_grid();
        for (int i = 0; i < num_tasks; i++) {
            if (blockIdx.x < tasks[i].num_tiles)
                dispatch_task(tasks[i], buffers, dyn_dims, blockIdx.x);
            grid.sync();
        }
        return;
    }

    // Counter-based scheduler
    int sm_id = blockIdx.x;
    int queue_len = sm_queue_lens[sm_id];
    int queue_base = sm_id * MAX_QUEUE_LEN;

    for (int q = 0; q < queue_len; q++) {
        SMQueueEntry entry = sm_queues[queue_base + q];
        int task_id = entry.task_id;
        int tile_id = entry.tile_id;

        // Wait for dependencies (spin on THIS task's counter only)
        if (threadIdx.x == 0) {
            while (atomicAdd(&dep_count[task_id * CACHE_LINE_INTS], 0) != 0) {}
        }
        __syncthreads();

        // Execute
        dispatch_task(tasks[task_id], buffers, dyn_dims, tile_id);
        __syncthreads();

        // Signal successors when LAST tile of this task completes
        if (threadIdx.x == 0) {
            int remaining = atomicSub(&tile_remaining[task_id * CACHE_LINE_INTS], 1);
            if (remaining == 1) {
                // Last tile done — decrement all successors' dep counts
                for (int s = succ_offset[task_id]; s < succ_offset[task_id + 1]; s++) {
                    atomicSub(&dep_count[succ_list[s] * CACHE_LINE_INTS], 1);
                }
            }
        }
    }
}

// Cache-line padding to prevent false sharing between dep_count entries
#define CACHE_LINE_INTS 32  // 128 bytes / 4 bytes per int
```

**Key design decisions:**
- `dep_count[]` padded to 128 bytes per entry (one cache line) to prevent false sharing between SMs polling different tasks
- `tile_remaining[]` separate from `dep_count[]` — `dep_count` tracks dependency satisfaction, `tile_remaining` tracks tile completion within a single task
- `is_last_tile` determined by `atomicSub` returning 1 (old value was 1, new value is 0)

**`src/megabake/schedule_compiler/serializer.py`** — Serialize new data:
- Per-SM queue arrays (flat: task_id, tile_id pairs)
- Per-SM queue lengths
- dep_count[] initial values (padded to cache lines)
- tile_remaining[] initial values (= num_tiles per task, padded)
- successor_list[] (flat with offset array)

**`src/megabake/runtime/launcher.py`** — Upload new buffers, add `--scheduler=bsp|counter` flag. Kernel args grow from 4 to 11.

### How to verify

```bash
# A/B test
python benchmarks/test_harness.py rmsnorm_mlp --scheduler=bsp --task-profile
python benchmarks/test_harness.py rmsnorm_mlp --scheduler=counter --task-profile
# Compare: counter should show lower total time, no barrier gaps

# Correctness (output must match BSP exactly)
python benchmarks/test_harness.py mlp_silu --scheduler=counter --profile
python benchmarks/test_harness.py llama_decoder --scheduler=counter --profile

# Stress test for deadlocks
python -c "
import megabake, torch
model = ...  # SmolLM2-135M
for i in range(1000):
    out = megabake.run(compiled, model, input_ids)
    print(f'{i}: ok')
"
```

### Expected gain

Eliminate ~250us barriers + ~2000us SM idle time. SmolLM2-135M from ~1.0x to ~1.3-1.5x vs torch.compile.

**Risk:** Counter polling cache-line bouncing. Mitigated by padding dep_count[] entries to 128-byte cache-line boundaries.

---

## Phase 6: Attention Rewrite — FlashAttention

**Goal:** Replace naive attention with K-tiled FlashAttention using tensor cores. Single biggest performance gap on real transformer models.

**Why the current implementation is slow** (`src/cuda/tasks/attention.cu`):

1. **O(seq_k) SMEM** (line 69-70): `float* smem` holds ALL scores for one query. seq_k=2048 needs 8KB per query position. Limits batch size and seq length.

2. **Scalar dot product** (lines 86-117): Q*K^T computed with scalar FMA loops. Even the "vectorized" path (line 94) unpacks float4 to scalar floats and uses FMA. No HMMA/WGMMA tensor core instructions.

3. **Serial over queries** (line 82): `for (sq = 0; sq < seq_q; sq++)` — one query at a time. No parallelism across query positions.

4. **Three passes over scores** (lines 121-141): max reduction, exp+sum, normalize. Each touches all seq_k scores in SMEM.

### FlashAttention algorithm (what to implement)

```
For each query block Bq (block_q queries):
  Load Q[Bq] to SMEM
  Initialize: acc = 0, max_old = -inf, sum_old = 0

  For each key block Bk:
    cp.async: prefetch K[Bk+1] to SMEM        <- overlap with compute
    Load K[Bk] from SMEM (already prefetched)

    S = Q @ K^T                                <- TENSOR CORES (WGMMA/mma.sync)
    if causal: mask S where key_pos > query_pos <- REQUIRED for autoregressive models

    max_new = max(max_old, row_max(S))
    P = exp(S - max_new)
    sum_new = sum_old * exp(max_old - max_new) + row_sum(P)

    Load V[Bk] from SMEM
    acc = acc * exp(max_old - max_new) + P @ V <- TENSOR CORES
    max_old = max_new
    sum_old = sum_new

  O = acc / sum_new
  Store O to global memory
```

Key properties:
- O(1) extra SMEM per head (only K-block + V-block + Q-block, not full score matrix)
- Tensor cores for both S = Q @ K^T and O = P @ V
- Online softmax: never materializes full score matrix
- cp.async prefetch hides K/V load latency

### Causal masking

```cuda
// After computing S = Q @ K^T:
for (int i = 0; i < BQ; i++) {
    int q_pos = sq_block + i;
    for (int j = 0; j < BK; j++) {
        int k_pos = sk_block + j;
        if (k_pos > q_pos) {
            S[i][j] = -INFINITY;  // masked positions get exp(-inf) = 0
        }
    }
}
```

Can skip entire K-block if `sk_block > sq_block + BQ - 1` (all positions masked). Early exit saves compute on long sequences.

### What to change

**`src/cuda/tasks/attention.cu`** — Full rewrite. ~300 lines replacing ~185. Structure:

```cuda
__device__ void task_attention(const TaskDesc& task, ...) {
    // Parse dimensions (same as current)
    constexpr int BQ = 64;   // query block
    constexpr int BK = 64;   // key/value block

    // Reuse CuTe MMA atoms from matmul.cu
    // SM90: WGMMA for S = Q @ K^T (BQ x HD) @ (BK x HD)^T = (BQ x BK)
    // SM80: mma.sync

    extern __shared__ char smem[];
    // SMEM layout: Q_block | K_block | V_block | K_next (double buffer)
    // Total: BQ*HD + 2*BK*HD + BK*HD = ~48-64KB for BQ=BK=64, HD=128

    for each (batch, head) assigned to this tile_id:
        uint32_t kv_h = h * num_kv_heads / num_heads;  // GQA mapping (existing line 79)

        for (sq_block = 0; sq_block < seq_q; sq_block += BQ):
            load Q[sq_block:sq_block+BQ] to SMEM

            float acc[BQ][HD] = 0;  // in registers
            float max_old = -inf, sum_old = 0;

            cp_async_load(K[0:BK] -> smem_K);

            for (sk_block = 0; sk_block < seq_k; sk_block += BK):
                // Causal early exit
                if (is_causal && sk_block > sq_block + BQ - 1)
                    break;

                cp_async_wait();

                if (sk_block + BK < seq_k)
                    cp_async_load(K[sk_block+BK : sk_block+2*BK] -> smem_K_next);

                // S = Q @ K^T via tensor cores
                wgmma_gemm(smem_Q, smem_K, S_frag);

                // Causal mask
                if (is_causal) mask_upper(S_frag, sq_block, sk_block);

                // Online softmax
                max_new = row_max(S_frag);
                rescale = exp(max_old - max_new);
                P = exp(S_frag - max_new);
                sum_new = sum_old * rescale + row_sum(P);

                // Load V, accumulate
                cp_async_load(V[sk_block:sk_block+BK] -> smem_V);
                cp_async_wait();

                scale_acc(acc, rescale);
                wgmma_gemm(P, smem_V, acc);

                max_old = max_new;
                sum_old = sum_new;
                swap(smem_K, smem_K_next);

            normalize_and_store(acc, sum_old, out);
}
```

**`src/megabake/schedule_compiler/tiling.py`** — Update attention tiling:
```python
# Current (line 18): tiles = batch * num_heads
# New: also consider query blocks if seq_q > BQ
elif op_type == OpType.ATTENTION:
    batch, num_heads, seq_q = dims[0], dims[1], dims[2]
    query_blocks = (seq_q + 63) // 64
    return min(batch * num_heads * query_blocks, max_sms)
```

### SMEM budget

```
228 KB per SM (H200)
Q block: 64 * 128 * 2 = 16 KB
K block x2 (double buffer): 2 * 64 * 128 * 2 = 32 KB
V block: 64 * 128 * 2 = 16 KB
Total: 64 KB — fits comfortably
```

### How to verify

```bash
# Correctness (attention numerically sensitive — online softmax rescaling)
python benchmarks/test_harness.py llama_decoder --profile
# max_diff should be < 1e-2

# Performance
python benchmarks/test_harness.py llama_decoder --task-profile
# Attention task should be dramatically faster

# If model available:
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile
```

### Expected gain

gemma-2b from 0.49x to ~0.8-1.0x vs torch.compile. Attention 5-10x faster.

**Risk:** FlashAttention inside persistent kernel shares SMEM with matmul. With `__launch_bounds__(256, 1)`, full 228 KB available. FlashAttention needs ~64 KB. Fits — no conflict since only one task runs per SM at a time.

---

## Phase 7: Paged SMEM + Weight Prefetch Overlap

**Goal:** Overlap weight loading for task N+1 with compute of task N. This is the single biggest bandwidth utilization unlock — the difference between ~50% and ~78% of peak HBM bandwidth (per Hazy Research measurements on H100).

**Depends on:** Phase 5 (static per-SM assignment — must know next task at compile time to plan prefetch).

**Why:** Current execution:
```
Task 1: [  LOAD WEIGHTS  ][  COMPUTE  ][STORE]
Task 2:                                        [  LOAD WEIGHTS  ][  COMPUTE  ][STORE]
```

With prefetch:
```
Task 1: [  LOAD WEIGHTS  ][  COMPUTE  ][STORE]
Task 2:         [ PREFETCH (cp.async) ][  COMPUTE  ][STORE]
Task 3:                       [ PREFETCH (cp.async) ][  COMPUTE  ][STORE]
```

After warmup (task 1), every subsequent task has weights pre-loaded in SMEM pages. Load time fully overlapped with previous task's compute.

### What to change

**Paged SMEM design:**

```cuda
// SMEM divided into fixed pages
#define SMEM_TOTAL    (228 * 1024)   // 228 KB per SM on H200
#define PAGE_SIZE     (14 * 1024)     // 14 KB per page (matches Hazy's H100 design)
#define NUM_PAGES     16              // 228K / 14K = 16 pages
#define PAGE_PTR(p)   (&smem_pool[(p) * PAGE_SIZE])

__shared__ char smem_pool[SMEM_TOTAL];
```

Page budget per task:
- Matmul compute scratch: 4-6 pages (56-84 KB)
- Weight prefetch (next task): 2-4 pages (28-56 KB)
- Activation handoff: 1 page (14 KB, for hidden<=4096)
- Remaining: available for double-buffering

**`src/megabake/schedule_compiler/scheduler.py`** — Extend `assign_tasks_to_sms()` with page planning:

```python
def plan_smem_pages(sm_queues, tasks, num_pages=16):
    """Assign SMEM pages per task per SM.

    Returns: page_plans[sm_id][queue_pos] = {
        'compute_pages': [0,1,2,3],     # pages for this task's scratch
        'prefetch_pages': [4,5],         # pages to prefetch next task's weights
        'handoff_pages': [6],            # page for output -> next task input
    }
    """
    page_plans = []
    for sm_id, queue in enumerate(sm_queues):
        sm_plan = []
        for q, (task_id, tile_id) in enumerate(queue):
            task = tasks[task_id]
            compute_need = estimate_smem_pages(task)
            prefetch_need = 0
            if q + 1 < len(queue):
                next_task = tasks[queue[q + 1][0]]
                prefetch_need = estimate_weight_pages(next_task)

            # Simple double-buffer allocation
            if q % 2 == 0:
                compute_pages = list(range(0, compute_need))
                prefetch_pages = list(range(8, 8 + prefetch_need))
            else:
                compute_pages = list(range(8, 8 + compute_need))
                prefetch_pages = list(range(0, prefetch_need))

            sm_plan.append({
                'compute_pages': compute_pages,
                'prefetch_pages': prefetch_pages,
            })
        page_plans.append(sm_plan)
    return page_plans
```

**`src/cuda/data_types.cuh`** — Extend TaskDesc or add per-SM-queue metadata:

```cuda
struct SMQueueEntryV2 {
    uint32_t task_id;
    uint32_t tile_id;
    uint8_t  compute_page_start;  // first page for compute
    uint8_t  compute_page_count;
    uint8_t  prefetch_page_start; // first page for next task's weights
    uint8_t  prefetch_page_count;
    uint32_t prefetch_weight_offset; // offset into weight buffer to prefetch
    uint32_t prefetch_bytes;         // how many bytes to prefetch
};
```

**`src/cuda/megakernel.cu`** — Extend counter-based loop with prefetch:

```cuda
// Counter-based scheduler with weight prefetch
for (int q = 0; q < queue_len; q++) {
    SMQueueEntryV2 entry = sm_queues[queue_base + q];

    // Wait for dependencies (weights for THIS task already prefetched)
    if (threadIdx.x == 0) {
        while (atomicAdd(&dep_count[entry.task_id * CACHE_LINE_INTS], 0) != 0) {}
    }
    __syncthreads();

    // Compute using current pages (weights already in SMEM from previous iteration's prefetch)
    dispatch_task_with_smem(tasks[entry.task_id], entry.tile_id,
                            PAGE_PTR(entry.compute_page_start),
                            entry.compute_page_count * PAGE_SIZE);
    __syncthreads();

    // Signal successors
    if (threadIdx.x == 0) {
        int remaining = atomicSub(&tile_remaining[entry.task_id * CACHE_LINE_INTS], 1);
        if (remaining == 1) {
            for (int s = succ_offset[entry.task_id]; s < succ_offset[entry.task_id + 1]; s++) {
                atomicSub(&dep_count[succ_list[s] * CACHE_LINE_INTS], 1);
            }
        }
    }

    // Start prefetch for NEXT task's weights (overlapped with nothing — will overlap
    // with next iteration's dep_count wait and dispatch)
    if (q + 1 < queue_len) {
        SMQueueEntryV2 next = sm_queues[queue_base + q + 1];
        if (next.prefetch_bytes > 0) {
            char* dst = PAGE_PTR(next.compute_page_start);  // next task's compute pages
            const char* src = (const char*)buffers[tasks[next.task_id].buffer_indices[1]]
                              + next.prefetch_weight_offset;
            for (int t = threadIdx.x * 16; t < next.prefetch_bytes; t += blockDim.x * 16) {
                cp_async_cg(dst + t, src + t);
            }
            cp_async_commit_group();
        }
    }
}
```

### SMEM handoff (bonus — comes free with paged SMEM)

Once paged SMEM exists, inter-task handoff is straightforward:

```cuda
// RMSNorm writes output to SMEM page instead of HBM
if (task.handoff_page >= 0) {
    // Write normalized output to SMEM page
    half* smem_out = (half*)PAGE_PTR(task.handoff_page);
    for (j = threadIdx.x * 8; j < row_size_aligned; j += threads * 8) {
        *(float4*)(smem_out + j) = output_vec;  // instead of global memory write
    }
} else {
    // Normal global memory write
    *(float4*)(row_out + j) = output_vec;
}
```

Next matmul reads from SMEM page (30 cycles) instead of L2 (200 cycles). 60 norms × 170 cycle savings = ~7us saved.

**Compiler decides handoff eligibility:**
- Producer and consumer on same SM (guaranteed by static assignment)
- Intermediate fits in 1-2 SMEM pages (<=28 KB)
- Both tasks are memory-bound (handoff saves latency, not bandwidth)

### How to verify

```bash
# Standalone benchmark: chain of 10 matmuls with/without prefetch
python benchmarks/test_harness.py matmul_chain --prefetch=on --task-profile
python benchmarks/test_harness.py matmul_chain --prefetch=off --task-profile
# Measure bandwidth utilization — target >= 70%

# Full model
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile
# Expect 15-25% improvement over Phase 6
```

### Expected gain

Bandwidth utilization from ~50% to ~70-78%. SmolLM2-135M from ~1.3x to ~1.5-1.8x vs torch.compile.

**Risk:** Two concerns:
1. SMEM page planning at compile time requires knowing exact data sizes per task per SM. Mitigate: start with weight prefetch only (always know weight size at compile time). Add inter-task handoff as second step.
2. `dispatch_task_with_smem()` doesn't exist — every task kernel currently takes `(TaskDesc, buffers, dyn_dims, tile_id)`. Adding SMEM pointer args means changing every task function signature. Mitigate: pass SMEM base pointer through existing `buffers[]` array (add a slot for "current SMEM page") rather than changing all function signatures. Task kernels that need SMEM just read `buffers[SMEM_SLOT]`.

---

## Phase 8: CUTLASS Multi-Config for Prefill

**Goal:** Compile 3 CUTLASS tile variants for M>=16 prefill instead of fixed 128×128. Compile-time selection per shape.

**Why:** Current CuTe GEMM uses fixed BM=128, BN=128 (SM90: BK=64, SM80: BK=32). For shapes where M or N is not a multiple of 128, SM utilization drops. Example: M=64, N=4096 → 1×32=32 tiles. With 64×128 tiles: 1×32=32 tiles but each tile processes entire M in one shot (better data reuse).

**Independent of Phases 5-7.** But lower priority — matters only for prefill (M>=16), not decode.

### What to change

**`src/cuda/tasks/matmul.cu`** — Add 2 additional tile configs alongside existing 128×128:

```cuda
// Config 0: 64x128x32 — rectangular, good for MLP shapes where M is small
// Config 1: 128x128x64 — current default (SM90) / 128x128x32 (SM80)
// Config 2: 128x256x64 — large N (very wide layers)

// SM90 path: dispatch by config_id from TaskDesc
__device__ void task_matmul_gemm_sm90(const TaskDesc& task, ...) {
    uint32_t config_id = (task.strides[1] >> 8) & 0xFF;  // upper byte of strides[1]
    switch (config_id) {
        case 0: gemm_sm90<64, 128, 32, 2>(task, ...); break;
        case 1: gemm_sm90<128, 128, 64, 3>(task, ...); break;  // existing
        case 2: gemm_sm90<128, 256, 64, 3>(task, ...); break;
    }
}

template<int BM, int BN, int BK, int STAGES>
__device__ void gemm_sm90(const TaskDesc& task, ...) {
    // Same CuTe GEMM code as current (lines 136-338), parameterized by template args
    // MMA atom: BM<=64 uses SM90_64xBNx16, BM>=128 uses SM90_64x128x16 with 2 warpgroups
    // ...
}
```

**`src/megabake/schedule_compiler/tiling.py`** — Tile selection at compile time:

```python
TILE_CONFIGS = [
    (0, 64, 128, 32, 2),    # config_id, BM, BN, BK, stages
    (1, 128, 128, 64, 3),   # current default
    (2, 128, 256, 64, 3),   # wide N
]

def select_matmul_config(M, N, K, num_sms):
    """Pick tile config that maximizes SM utilization × wave efficiency."""
    if M <= 4:
        return SKINNY_MATVEC_ID  # handled separately

    best_score, best_id = 0, 1  # default to 128x128
    for (config_id, BM, BN, BK, stages) in TILE_CONFIGS:
        tiles = math.ceil(M / BM) * math.ceil(N / BN)
        sm_util = min(tiles, num_sms) / num_sms
        waves = math.ceil(tiles / num_sms)
        wave_eff = tiles / (waves * num_sms)
        score = sm_util * wave_eff
        if score > best_score:
            best_score, best_id = score, config_id
    return best_id
```

Store `config_id` in upper byte of `task.strides[1]`: `task.strides[1] |= (config_id << 8)`.

**`src/megabake/schedule_compiler/tiling.py`** — Update tile count computation to use selected config's BM/BN:

```python
if op_type == OpType.MATMUL and M > 4:
    config = TILE_CONFIGS[selected_config_id]
    BM, BN = config[1], config[2]
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    return min(tiles, max_sms)
```

### How to verify

```bash
# Prefill benchmarks at various shapes
python benchmarks/test_harness.py prefill_matmul --M=64 --N=4096 --K=4096 --profile
python benchmarks/test_harness.py prefill_matmul --M=128 --N=4096 --K=4096 --profile
python benchmarks/test_harness.py prefill_matmul --M=512 --N=11008 --K=4096 --profile
# Compare against cuBLAS — target within 15%

# Verify no decode regression
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile
```

### Expected gain

Prefill matmul within 10-15% of cuBLAS on common shapes (up from 0.2-0.3x). No decode impact.

---

## Phase 9+: Data-Driven Extensions

Not planned in detail. Proceed only when Phase 0-8 profiling shows need.

### Chunked Dependencies (from MPK)

**What:** Split large tasks into chunks with separate dep counters. Consumer starts on chunk 0 while producer still generating chunks 1-N. Eliminates pipeline stalls between large producer + small consumer.

**When to do it:** If Phase 5 profiling shows inter-task transition latency > 5% of runtime.

### INT8 Weight-Only Quantization (from Ada-MK)

**What:** INT8 weights + FP16 scale per group, dequant in registers during matmul. Halves weight memory traffic = ~2x decode speedup on bandwidth-bound workloads.

**When to do it:** When users request quantization support. High ROI — dequant is free on decode (compute is idle).

### BF16 Support

**What:** Add bfloat16 dtype alongside existing fp16. Matmul atoms support both. Mostly plumbing.

**When to do it:** When users request it.

### Hybrid cuBLAS + CUDA Graphs for Prefill

**What:** CUDA Graph wrapping cuBLAS GEMM nodes + megakernel for glue ops. For compute-bound prefill (M>=32) where megakernel GEMM can't match cuBLAS.

**When to do it:** Last resort. Breaks persistent kernel advantages (no prefetch overlap, no SMEM handoff at cuBLAS boundaries). Only if Phase 8 CUTLASS multi-config still >20% behind cuBLAS.

---

## Rejected / Deferred from REDESIGN.md

| Proposal | Verdict | Reason |
|----------|---------|--------|
| Two-level IR (Graph IR + Schedule IR) | Deferred | Flat TaskDesc + per-SM queues sufficient at current scale. Add when compiler complexity demands it |
| Cost-model fusion | Deferred | Rule-based fusion fine for <10 patterns. Cost model adds complexity without proven benefit |
| Layout propagation | Deferred | B already assumed transposed. Phase 4 adds col-major pre-transpose |
| Shape bucketing | Deferred | Feature request, not performance |
| Skinny matvec "float4 fix" | Rejected | Already has float4 weight loads. cp.async IS valid (Phase 4) |

---

## Commit Discipline

Each phase: 1-3 commits.
1. Implementation commit
2. Profile results commit (JSON in `benchmarks/baselines/`)
3. Cleanup commit if needed

Commit messages include before/after numbers.

---

## Verification Checklist (every phase)

```bash
# 1. All built-ins pass
python benchmarks/test_harness.py mlp_silu
python benchmarks/test_harness.py rmsnorm_mlp
python benchmarks/test_harness.py llama_decoder

# 2. Numerical correctness
# max_diff < 1e-3 for all models (< 1e-2 for attention changes)

# 3. Performance (compare against Phase 0 baselines)
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile

# 4. No regression on unrelated benchmarks
```

---

## Phase Summary

| Phase | What | Effort | Expected Impact | Depends On |
|-------|------|--------|----------------|------------|
| 0 | Per-task profiling infrastructure | 2 days | Enables data-driven decisions | — |
| 1 | Vectorize reduce/rope/index + single-pass RMSNorm | 3 days | 5-8% overall | Phase 0 |
| 2 | Consolidate 12 op_types to 7 + __noinline__ | 1.5 days | 2-5% I-cache | — |
| 3 | Matmul epilogue fusion (bias + residual) | 3 days | 5-12% on biased models | Phase 2 |
| 4 | Skinny matvec cp.async + weight pre-transpose | 1 week | 1.5-2x on M=1 matmul | — |
| 5 | Counter-based scheduler + static per-SM assignment | 2 weeks | Eliminate ~2250us waste | — |
| 6 | FlashAttention with tensor cores | 2-3 weeks | 5-10x on attention | — |
| 7 | Paged SMEM + weight prefetch overlap | 2 weeks | 50% to 78% BW utilization | Phase 5 |
| 8 | CUTLASS multi-config (3 tiles, prefill) | 2 weeks | 1.5-3x on prefill matmul | — |
| 9+ | Chunked deps, INT8, BF16, hybrid cuBLAS | varies | varies | Profiling data |
