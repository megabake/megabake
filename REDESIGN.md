# Megabake: Architectural Overhaul

*Authoritative system document. Replaces previous PLAN.md, inductor-integration.md, and hf-kernel-hub-integration.md.*

---

## 0. Executive Summary

Megabake compiles `torch.export` FX graphs into a single cooperative CUDA megakernel. The thesis is sound -- megakernel research (Hazy Research 2.5x over vLLM, MPK 1.0-1.7x over SGLang, Ada-MK in production at ByteDance) proves persistent kernels beat torch.compile on decode workloads. The current implementation does not deliver on this thesis.

| | Current | Target |
|---|---------|--------|
| **Perf vs torch.compile** | 0.17-0.32x (results.json) | 1.5-2.5x decode, 0.9-1.1x prefill |
| **Op coverage** | 40 hardcoded ATen ops, crash on unknown | Inductor decomps (1100+), graph split fallback, never crash |
| **Attention** | CUDA-core FMA, O(seq_k) SMEM, no tensor cores | FlashAttention: tensor cores, K-tiled, online softmax |
| **Scheduler** | BSP grid.sync every task, 97% SM idle on norms | Counter-based, static per-SM queues, zero idle SMs |
| **Bandwidth util** | ~25% | ~75-78% (paged SMEM + weight prefetch) |
| **Fusion** | 5 peephole passes | Matmul epilogue flags + expanded micro-op interpreter + cost model |
| **Reduce/rope/index** | Scalar fp16 loads | float4 vectorized, single-pass norms |

**Core thesis**: Inductor decomps for breadth, dedicated kernels for speed, micro-op interpreter for the long tail, graph splitting as safety net. Four pillars, zero compilation failures, competitive performance.

---

## 1. Current State

Full analysis in `current-project-state.md`. Key facts verified against actual code:

- **4,300 lines** of code (Python + CUDA). ~10,000 lines of planning docs. Only Phase 0 (profiling) implemented of 9-phase roadmap.
- **Matmul** (`matmul.cu`, 576 lines): Production-grade CuTe GEMM. SM90 WGMMA with 3-stage cp.async pipeline. SM80 mma.sync with 2-stage pipeline. Skinny matvec (M<=64) with float4 B loads but no cp.async between K-chunks.
- **Attention** (`attention.cu`, 143 lines): float4 dot products (not purely scalar as old docs claimed), GQA, causal masking. But CUDA-core FMA, O(seq_k) SMEM, serial queries, 3-pass softmax. 3-5x slower than FlashAttention.
- **Elementwise** (`elementwise.cu`, 183 lines): Well-vectorized float4. **Fused elementwise** (`fused_elementwise.cu`, 188 lines): Clever micro-op interpreter, 17 opcodes, limited to 8 uops/8 regs/8 bufs.
- **Reduce/rope/index**: Entirely unvectorized. Scalar `__half2float()` loads throughout.
- **Graph walker** (`graph_walker.py`, 1275 lines): Handles StridedView tracking, 2-variant RMSNorm detection, RoPE detection, addmm bias extraction, identity elimination, constant folding. But all-or-nothing -- unknown op = `RuntimeError`.
- **Scheduler**: BSP `grid.sync()` after every task. 1-tile tasks idle 31/32 SMs.

---

## 2. Architectural Principles

### P1: Inductor Pre-Grad Passes for Breadth

Use Inductor's pre-lowering optimization passes (`pre_grad_passes`) as a single pipeline stage. This gives megabake Inductor's decomposition table (1153 decomps), 100+ declarative pattern matches (RMSNorm all variants, RoPE, SDPA variants, split/cat fusion, batch fusion, binary folding), constant folding, CSE, and DCE -- all operating on plain FX graphs, outputting ATen ops. No Inductor IR, scheduler, or codegen -- those assume independent kernel launches.

This replaces megabake's hand-written `_find_rmsnorm_patterns` (150 lines), `_find_rope_patterns` (90 lines), and `_constant_fold` (105 lines) with Inductor's battle-tested equivalents. Every new pattern Inductor adds upstream becomes available to megabake automatically.

### P2: Dedicated Kernels for Depth

MATMUL, ATTENTION, REDUCE, ROPE, EMBEDDING get hand-optimized CUDA. These ops constitute ~80-90% of transformer inference runtime. Micro-op interpreter cannot match hand-tuned tensor core GEMM or FlashAttention.

### P3: Micro-Op Interpreter as Universal Pointwise

The existing fused_elementwise interpreter (`fused_elementwise.cu`) is the right primitive. Expand it: 8→32 uops, 8→16 regs, 8→16 buffers, add reduction ops, broadcast-aware LOAD. Any chain of pointwise + simple reductions compiles to a micro-op program. This is megabake's equivalent of Inductor's `Pointwise`/`Reduction` IR -- expressed as GPU bytecode instead of Python closures.

### P4: Graph Splitting as Safety Net

Unsupported ops split the megakernel into segments. Run each segment as a cooperative launch, unsupported ops via eager PyTorch between segments. Compilation never fails. Graceful degradation:

```
Best:     100% fused in one megakernel
Good:     95% fused, 5% via eager between segments
Okay:     Multiple small megakernel segments + eager glue
Baseline: Pure torch.compile fallback
```

### P5: Counter-Based Scheduler

Replace BSP `grid.sync()` with per-dependency atomic counters + static per-SM task assignment. Each SM walks its own queue, spins only on its own tasks' counters. Zero global barriers, zero cross-SM contention, zero idle SMs. Both Hazy Research and MPK converged on this independently.

### P6: Paged SMEM + Weight Prefetch

Divide SMEM into fixed pages. While computing task N, `cp.async` loads task N+1's weights into SMEM pages. After warmup, every task has weights pre-loaded. Raises bandwidth utilization from ~50% to ~78% (Hazy Research measurements). Requires P5 (static SM assignment) -- must know next task at compile time.

### P7: Hybrid Decode/Prefill

Decode (M<=4): Pure megakernel. Bandwidth-bound. Megakernel advantages (continuous memory streaming, weight prefetch, SMEM handoff) compound maximally.

Prefill (M>=16): Megakernel with CUTLASS multi-config. No cuBLAS fallback -- pure megakernel at all batch sizes. Optimize CUTLASS tile configs until they match cuBLAS quality. The persistent kernel advantages (prefetch overlap, SMEM handoff, zero launch overhead) only hold if the entire forward pass stays inside the megakernel. Breaking out to cuBLAS destroys those advantages.

---

## 3. Compilation Pipeline

```
torch.nn.Module
  │
  ▼
[1] torch.export(model, args, strict=False)
  │  Output: ExportedProgram (FX graph + state_dict)
  │  File: graph_walker.py:783
  │
  ▼
[2] Inductor Pre-Grad Optimization (CHANGED — replaces stages 2+3+4 from old pipeline)
  │  OLD: core_aten_decompositions() + hand-written _find_rmsnorm_patterns (150 lines)
  │       + _find_rope_patterns (90 lines) + _constant_fold (105 lines)
  │  NEW: Single call to Inductor's pre_grad_passes():
  │    - select_decomp_table(): 1153 decompositions (142 more than core_aten)
  │    - PatternMatcherPass: 100+ declarative patterns (RMSNorm, RoPE, GeGLU,
  │      SwiGLU, SDPA variants, split/cat, binary folding, pad_mm, etc.)
  │    - Constant folding (more robust than megabake's _constant_fold)
  │    - Common subexpression elimination
  │    - Dead code elimination
  │  Output: optimized FX graph of ATen ops (NOT Inductor IR)
  │  Deletes: _find_rmsnorm_patterns, _find_rope_patterns, _constant_fold (~345 lines)
  │  File: graph_walker.py:392-766 → new inductor_passes.py (~50 lines)
  │
  ▼
[3] Graph Walk + Lowering (CHANGED)
  │  OLD: ATEN_OP_MAP lookup → RuntimeError on unknown
  │  NEW: Three-tier lowering:
  │    Tier 1: ATEN_OP_MAP → dedicated CUDA task (unchanged)
  │    Tier 2: Fusable pointwise/reduction → micro-op program
  │    Tier 3: Unknown op → mark as graph split point
  │  File: graph_walker.py main loop + new micro_op_lowering.py
  │
  ▼
[4] Fusion Passes (CHANGED)
  │  OLD: _fuse_tasks, _fuse_elementwise_chains, _eliminate_redundant_copies
  │  NEW:
  │    [a] Matmul epilogue fusion (bias + activation + residual via flags)
  │    [b] Expanded elementwise chain fusion (32 uops, 16 regs, reduction ops)
  │    [c] Redundant copy elimination (keep existing)
  │    [d] Op type consolidation (12 → 8 types)
  │  Note: Inductor's pre_grad_passes may fuse ops megabake wants separate
  │        (e.g., matmul+bias). Run megabake-specific fusion AFTER Inductor
  │        passes to override where needed.
  │  File: new fusion.py
  │
  ▼
[5] Dependency DAG Extraction (NEW)
  │  Build producer/consumer graph from buffer_indices.
  │  Output: dep_count[], successor_list[], successor_offset[]
  │  File: new dependency.py
  │
  ▼
[6] Static Per-SM Assignment + Page Planning (NEW)
  │  Topological sort with critical-path priority.
  │  Bin-pack tasks onto SMs. Plan SMEM page allocation per task.
  │  Output: sm_queues[sm_id] = [(task_id, tile_id), ...]
  │  File: new scheduler.py
  │
  ▼
[7] Buffer Planning (KEEP)
  │  Existing plan_buffers() with liveness + first-fit-decreasing.
  │  File: buffer_planner.py (82 lines)
  │
  ▼
[8] Serialization (EXTEND)
  │  Add: per-SM queue arrays, dep_count[], successor_list[], page plans.
  │  Keep: TaskDesc, BufferDesc, WeightMapping, ScheduleHeader format.
  │  File: serializer.py
  │
  ▼
[9] CUDA Compilation (KEEP)
  │  nvcc concatenation build. Add __noinline__ on cold paths.
  │  File: cuda_compiler.py
  │
  ▼
[10] Cooperative Launch (EXTEND)
  │   Add: per-SM queue buffers, dep_count arrays, scheduler_type flag.
  │   Keep: cuLaunchCooperativeKernel, arena allocation, cached runner.
  │   File: launcher.py, loader.py
```

### New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `schedule_compiler/inductor_passes.py` | Inductor pre_grad_passes wrapper + megabake-specific exclusions | ~50 |
| `schedule_compiler/micro_op_lowering.py` | Lower arbitrary pointwise chains to micro-op programs | ~120 |
| `schedule_compiler/fusion.py` | Epilogue fusion, expanded chain fusion, cost-model decisions | ~200 |
| `schedule_compiler/dependency.py` | DAG extraction from buffer producers/consumers | ~60 |
| `schedule_compiler/scheduler.py` | Static per-SM assignment, critical-path sort, page planning | ~200 |

### Deleted Code (replaced by Inductor pre_grad_passes)

| Function | File | Lines | Replaced by |
|----------|------|-------|------------|
| `_find_rmsnorm_patterns()` | `graph_walker.py:512-663` | 150 | Inductor PatternMatcherPass |
| `_find_rope_patterns()` | `graph_walker.py:666-754` | 90 | Inductor PatternMatcherPass |
| `_constant_fold()` | `graph_walker.py:392-496` | 105 | Inductor constant folding pass |
| `_build_users_map()` | `graph_walker.py:499-510` | 12 | No longer needed (patterns handled by Inductor) |

---

## 4. Inductor Integration Layer

### 4a. Architecture: Option B (Pre-Grad Passes)

Use Inductor's pre-lowering optimization passes as a single pipeline stage. These passes operate on plain FX graphs (ATen ops in, ATen ops out). No Inductor IR (`TensorBox`, `Pointwise`, `SchedulerNode`) touches megabake's pipeline.

```python
# schedule_compiler/inductor_passes.py
def optimize_graph(ep):
    """Run Inductor's pre-grad passes on the exported program.

    Takes ExportedProgram with ATen ops, returns ExportedProgram with
    optimized ATen ops. No Inductor IR involved — output is still a
    plain FX graph that megabake's graph_walker consumes directly.
    """
    from torch._inductor.decomposition import select_decomp_table

    # Step 1: Decompose with Inductor's table (1153 decomps)
    decomp_table = select_decomp_table()

    # Preserve ops with dedicated megabake CUDA kernels
    for op in [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.embedding.default,
    ]:
        decomp_table.pop(op, None)

    # Restore ops Inductor excludes but megabake handles natively
    for op in [
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten._softmax.default,
        torch.ops.aten.native_layer_norm.default,
    ]:
        decomp_table.pop(op, None)

    ep = ep.run_decompositions(decomp_table)

    # Step 2: Run Inductor's pre-grad passes on the FX graph
    # These include: pattern matching (100+ patterns), constant folding,
    # CSE, DCE — all on plain FX graphs, no Inductor IR
    gm = ep.graph_module
    from torch._inductor.fx_passes.pre_grad import pre_grad_passes
    pre_grad_passes(gm)

    return ep
```

### 4b. What This Gives Megabake

**Decompositions** (from `select_decomp_table()`):
- 1153 decompositions vs current 1011 from `core_aten_decompositions()`
- New coverage: batch_norm, group_norm, dropout, log_softmax, upsample, and 130+ more
- All decompose into ATen primitives megabake already handles (add, mul, div, exp, reduce, etc.)

**Pattern matching** (from `pre_grad_passes` → `PatternMatcherPass`):
- 100+ declarative patterns: RMSNorm (all decomp variants), RoPE, GeGLU, SwiGLU, SDPA variants, split/cat fusion, binary folding, pad_mm, batch fusion
- Robust across PyTorch versions (declarative DAG matching, not hand-written graph walks)
- New patterns added upstream become available automatically

**Graph optimization** (from `pre_grad_passes`):
- Constant folding (more robust than megabake's `_constant_fold`)
- Common subexpression elimination
- Dead code elimination

**Deletes ~345 lines of megabake code**:
- `_find_rmsnorm_patterns()` (150 lines of fragile graph walking)
- `_find_rope_patterns()` (90 lines of fragile graph walking)
- `_constant_fold()` (105 lines)
- `_build_users_map()` (12 lines, only used by pattern functions)

### 4c. Handling Inductor/Megabake Fusion Conflicts

`pre_grad_passes` may fuse ops that megabake wants to handle differently. Example: Inductor might fold `matmul + bias` into a single pattern node, but megabake wants them separate so it can apply its own epilogue fusion (bias via `strides[1]` flags).

Strategy: run megabake-specific fusion passes AFTER Inductor's pre-grad passes. If Inductor fused something megabake needs to split, megabake's graph walker sees the fused result as a single op and handles it. If Inductor left ops separate that megabake wants to fuse (matmul + bias + activation), megabake's fusion pass (Section 8a) catches them.

In practice, most conflicts are benign -- Inductor's pre-grad patterns optimize graph structure (CSE, DCE, constant folding), not op fusion. The fusion-level patterns (like split/cat elimination) produce cleaner graphs for megabake to consume.

### 4d. API Stability

`pre_grad_passes` has been stable across PyTorch 2.4-2.6. The function signature is `pre_grad_passes(gm: GraphModule) -> None` (mutates in place). If it breaks in a future version, the fix is typically a one-line import path change.

Megabake already depends on `torch.export` (also internal-ish), so this class of API stability risk is already accepted. The fallback if `pre_grad_passes` breaks: revert to `select_decomp_table()` only (decomps without pattern matching) and re-add hand-written patterns temporarily.

### 4e. What NOT to Take

| Inductor Component | Take? | Why |
|---|---|---|
| `select_decomp_table()` | **YES** | Stable API (`Dict[OpOverload, Callable]`). No coupling. |
| `pre_grad_passes()` | **YES** | Operates on plain FX graphs. Output is ATen ops, not Inductor IR. |
| Lowering registry (`@register_lowering`) | **NO** | Returns `Pointwise`/`Reduction` IR nodes coupled to `V.graph`, `TensorBox`. Megabake's micro-op interpreter serves same purpose. |
| `scheduler.py` (fusion scheduler) | **NO** | Assumes independent kernel launches. Megabake needs per-SM assignment within one kernel. |
| `codegen/` (Triton/C++ codegen) | **NO** | Wrong execution model entirely. |
| `FallbackKernel` | **NO** | Concept useful (graph splitting), but implementation coupled to Inductor's `ExternKernelAlloc`. Reimplement for megabake (Section 5, Tier 3). |
| `post_grad_passes()` | **NO** | Operates on Inductor IR, not FX graphs. Wrong level of abstraction. |

---

## 5. Op Coverage Strategy

### Three Tiers

**Tier 1: Dedicated CUDA Kernel** (maximum performance)

| Op Type | CUDA File | Coverage |
|---------|-----------|----------|
| MATMUL | `matmul.cu` | mm, addmm, bmm, linear |
| ATTENTION | `attention.cu` | scaled_dot_product_attention |
| REDUCE | `reduce.cu` | RMSNorm, LayerNorm, softmax, sum, mean, max, argmax |
| ROPE | `rope.cu` | Rotary position embeddings |
| EMBEDDING | `embedding.cu` | Token embedding lookup |

**Tier 2: Micro-Op Interpreter** (good performance, general coverage)

Any chain of pointwise ops and simple reductions. The expanded `fused_elementwise.cu` interpreter handles:
- All unary: silu, gelu, relu, sigmoid, tanh, exp, log, rsqrt, neg, abs, cos, sin, pow
- All binary: add, mul, sub, div
- Reductions: sum, mean, max (NEW)
- Broadcast-aware LOAD (NEW)
- Index ops: gather (NEW)
- Cast/where/masked_fill/clamp

Ops that Inductor decomposes into these primitives automatically become supported without new CUDA code. Example: `log_softmax` decomposes to `log(softmax(x))`, both handled by micro-ops.

**Tier 3: Graph Splitting** (correctness guarantee, slower)

When the graph walker encounters an op not in Tier 1 or Tier 2:

```python
# In graph_walker main loop:
if mapping is None and not _can_lower_to_micro_ops(node):
    # Mark split point. Don't crash.
    split_points.append(node)
    continue
```

Runtime execution with splits:

```
megakernel_segment_1 → sync → eager_op(PyTorch) → sync → megakernel_segment_2
```

Each segment gets its own cooperative launch. Data flows via HBM between segments (standard PyTorch tensors). Performance degrades proportionally to number of splits, but correctness is guaranteed.

### Coverage Expansion via Inductor Decomps

With `select_decomp_table()`, these ops decompose into Tier 1/2 primitives automatically:

| Op | Decomposes to | Tier |
|----|--------------|------|
| batch_norm | mean + var + normalize (elementwise chain) | Tier 2 |
| group_norm | reshape + mean + var + normalize | Tier 2 |
| dropout | mul + bernoulli mask | Tier 2 (training only) |
| log_softmax | softmax + log | Tier 1 (softmax) + Tier 2 (log) |
| layer_norm | Already in Tier 1 (REDUCE) | Tier 1 |
| upsample | interpolate → index-based | Tier 2/3 |
| conv2d | Does not decompose to matmul | Tier 3 (graph split) |

---

## 6. Kernel Rewrites

### 6a. Attention: FlashAttention Rewrite (Highest Impact)

**Current** (`attention.cu`, 143 lines):
```
- float4 CUDA-core FMA dot products (no tensor cores)
- Full float[seq_k] score array in SMEM → O(seq_k) memory
- Serial query loop: for (sq = 0; sq < seq_q; sq++)
- Three-pass softmax: max, exp+sum, normalize
- GQA support present (kv_h = h * num_kv_heads / num_heads)
- Causal masking present (sk > sq → -1e30f)
```

**Target**: FlashAttention-style tiled attention inside the megakernel.

```
For each query block Bq (BQ=64 queries):
    Load Q[Bq] to SMEM
    acc = 0, max_old = -inf, sum_old = 0

    For each key block Bk (BK=64):
        if causal and sk_block > sq_block + BQ - 1: break  // early exit

        cp.async: prefetch K[Bk+1] to SMEM              // overlap with compute
        S = Q @ K^T                                       // TENSOR CORES (WGMMA/mma.sync)
        if causal: mask S where key_pos > query_pos

        max_new = max(max_old, row_max(S))                // online softmax
        P = exp(S - max_new)
        sum_new = sum_old * exp(max_old - max_new) + row_sum(P)

        acc = acc * exp(max_old - max_new) + P @ V        // TENSOR CORES
        max_old = max_new, sum_old = sum_new

    O = acc / sum_new
    Store O to global memory
```

**Key properties**:
- O(1) extra SMEM per head (K-block + V-block + Q-block, NOT full score matrix)
- Tensor cores for both S = Q@K^T and O = P@V
- Online softmax: never materializes full score matrix
- cp.async prefetch hides K/V load latency
- Causal early exit skips entire K-blocks

**SMEM budget** (H200, 228KB per SM):
```
Q block: 64 * 128 * 2 = 16 KB
K block x2 (double buffer): 2 * 64 * 128 * 2 = 32 KB
V block: 64 * 128 * 2 = 16 KB
Total: 64 KB — fits comfortably in 228 KB
```

**Tiling update** (`tiling.py`):
```python
# OLD: tiles = batch * num_heads
# NEW: also consider query blocks
elif op_type == OpType.ATTENTION:
    batch, num_heads, seq_q = dims[0], dims[1], dims[2]
    query_blocks = (seq_q + 63) // 64
    return min(batch * num_heads * query_blocks, max_sms)
```

**Keep**: GQA mapping (`kv_h = h * num_kv_heads / num_heads`), causal masking logic. Reuse CuTe MMA atoms from matmul.cu (SM90 WGMMA / SM80 mma.sync).

**Expected gain**: Attention 3-5x faster. Models like gemma-2b where attention dominates: overall 2-3x improvement.

### 6b. Reduce: Vectorize + Single-Pass Norms

**Current** (`reduce.cu`, 171 lines): All paths use scalar `__half2float(row_in[j])`. RMSNorm reads row twice. LayerNorm reads row three times.

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

Register budget: hidden=4096, 256 threads → 16 elements/thread → 16 fp32 registers. Within 255 register limit. hidden=8192 → 32 regs/thread, still fine.

Apply same float4 pattern to: LayerNorm (3→2 passes), Softmax (3→2 passes), Sum/Mean/Max.

### 6c. Rope: Vectorize

**Current** (`rope.cu`, 47 lines): Scalar `__half2float(in0[base + d])`.

**New**: Load x0/x1/cos/sin as float4, compute on float2 pairs, store as float4.

```cuda
for (d = threadIdx.x * 4; d < half_dim_aligned; d += threads * 4) {
    float4 x0_4 = *(const float4*)(in0 + base + d);
    float4 x1_4 = *(const float4*)(in0 + base + d + half_dim);
    float4 c_4  = *(const float4*)(cos_cache + s * stride + d);
    float4 s_4  = *(const float4*)(sin_cache + s * stride + d);
    // Unpack half2 pairs, compute rotary, repack, store as float4
}
```

4x fewer load/store instructions. Same computation. Keep BHSD/BSHD layout support.

### 6d. Index: Vectorize

**Current** (`index.cu`, 45 lines): Scalar `out[i] = src[src_row * inner_size + col]`.

**New**: When `inner_size % 8 == 0`, use float4:

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
    // existing scalar fallback
}
```

### 6e. Matmul Skinny: cp.async Prefetch

**Current** (`matmul.cu:61-134`): float4 weight loads but synchronous `__syncthreads()` between K-chunks. No overlap between load and compute.

**New**: Double-buffered SMEM with cp.async:

```cuda
// SMEM: two weight buffers for double-buffering
half* smem_B0 = smem_A + M * bk;
half* smem_B1 = smem_B0 + cols_this_tile * bk;

// Prefetch first K-chunk
for (t = threadIdx.x; t < chunk_elems; t += blockDim.x)
    cp_async_cg(&smem_B0[t], &B[offset + t]);
cp_async_commit_group();

for (k_start = 0; k_start < K; k_start += bk) {
    // Start loading NEXT K-chunk (overlapped with compute)
    if (next_k < K) {
        for (t = threadIdx.x; t < chunk_elems; t += blockDim.x)
            cp_async_cg(&nxt_B[t], &B[next_offset + t]);
        cp_async_commit_group();
    }
    cp_async_wait_group<1>();
    __syncthreads();

    // Compute: float4 loads from SMEM, FMA accumulate (existing)
    // ...

    cur_buf ^= 1;
}
```

SMEM budget: M=1, bk=256, cols_per_sm≈128 → B_chunk = 128 × 256 × 2 = 64KB × 2 = 128KB. Fits 228KB.

Also: Host-side weight pre-transposition to column-major for decode. Cache transposed weights -- don't re-transpose on every call.

### 6f. Matmul Epilogue Fusion

**Current**: 3 redundant op types (MATMUL_SILU=0x09, MATMUL_GELU=0x0A, MATMUL_GELU_TANH=0x0C). `apply_epilogue()` reads `task.op_type`. Bias in `buffer_indices[3]` is set by graph walker but ignored by CUDA kernel.

**New**: Epilogue flags in `task.strides[1]`:

```cuda
#define EPILOGUE_SILU      0x01
#define EPILOGUE_GELU      0x02
#define EPILOGUE_GELU_TANH 0x04
#define EPILOGUE_BIAS      0x08
#define EPILOGUE_RESIDUAL  0x10

__device__ __forceinline__ float apply_epilogue(float v, uint32_t flags,
    const half_t* bias, const half_t* residual, int col, int idx) {
    if (flags & EPILOGUE_BIAS)      v += __half2float(bias[col]);
    if (flags & EPILOGUE_SILU)      return v / (1.0f + __expf(-v));
    if (flags & EPILOGUE_GELU)      return v * 0.5f * (1.0f + erff(v * 0.7071f));
    if (flags & EPILOGUE_GELU_TANH) { /* existing tanh approx */ }
    if (flags & EPILOGUE_RESIDUAL)  v += __half2float(residual[idx]);
    return v;
}
```

Branch prediction makes this zero-cost -- same branch taken for entire tile.

Buffer layout: `buffer_indices[3]` = bias (1D, length N), `buffer_indices[4]` = residual (2D, M × N). Graph walker already sets `buffer_indices[3]` for addmm -- just need CUDA kernel to read it.

---

## 7. Scheduler Redesign

### 7a. Dependency DAG

Extract from buffer producers/consumers during graph walk. Each task that writes to a buffer is a producer; each task that reads it is a consumer.

```python
# schedule_compiler/dependency.py
def build_dependency_dag(tasks):
    producers = {}  # buffer_id → task_idx
    edges = []
    for i, task in enumerate(tasks):
        # Output buffer
        out_buf = task.buffer_indices[0]
        if out_buf != UNUSED_BUFFER:
            producers[out_buf] = i
        # Input buffers
        for buf in task.buffer_indices[1:]:
            if buf != UNUSED_BUFFER and buf in producers:
                src = producers[buf]
                if src != i:
                    edges.append((src, i))

    dep_count = [0] * len(tasks)
    successors = [[] for _ in range(len(tasks))]
    for src, dst in edges:
        dep_count[dst] += 1
        successors[src].append(dst)
    return dep_count, successors
```

### 7b. Static Per-SM Assignment

Critical-path-priority topological sort, then bin-pack onto SMs:

```python
# schedule_compiler/scheduler.py
def assign_tasks_to_sms(tasks, dep_count, successors, num_sms):
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
```

Benefits:
- Single-tile tasks (norms): 1 SM works, 31 SMs work on their own tasks
- Independent tasks (Q/K/V projections): different SMs work simultaneously
- Compiler knows each SM's full task order → enables weight prefetch planning

### 7c. Device Execution Loop

```cuda
// megakernel.cu — counter-based scheduler
extern "C"
__global__ void __launch_bounds__(256, 1) megakernel(
    const TaskDesc* tasks, int num_tasks, void** buffers, const int* dyn_dims,
    long long* task_timings,
    // Counter-based scheduler args:
    const SMQueueEntry* sm_queues, const int* sm_queue_lens,
    int* dep_count, int* tile_remaining,
    const int* succ_list, const int* succ_offset,
    int scheduler_type
) {
    if (scheduler_type == 0) {
        // BSP fallback (existing code)
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

    for (int q = 0; q < queue_len; q++) {
        SMQueueEntry entry = sm_queues[sm_id * MAX_QUEUE_LEN + q];

        // Wait for dependencies
        if (threadIdx.x == 0)
            while (atomicAdd(&dep_count[entry.task_id * CACHE_LINE_INTS], 0) != 0) {}
        __syncthreads();

        // Execute
        dispatch_task(tasks[entry.task_id], buffers, dyn_dims, entry.tile_id);
        __syncthreads();

        // Signal successors when LAST tile completes
        if (threadIdx.x == 0) {
            int remaining = atomicSub(&tile_remaining[entry.task_id * CACHE_LINE_INTS], 1);
            if (remaining == 1) {
                for (int s = succ_offset[entry.task_id]; s < succ_offset[entry.task_id + 1]; s++)
                    atomicSub(&dep_count[succ_list[s] * CACHE_LINE_INTS], 1);
            }
        }
    }
}

#define CACHE_LINE_INTS 32  // 128 bytes / 4 bytes — prevent false sharing
```

### 7d. Paged SMEM + Weight Prefetch

Depends on 7b (static per-SM assignment -- must know next task at compile time).

```cuda
#define SMEM_TOTAL   (228 * 1024)
#define PAGE_SIZE    (14 * 1024)
#define NUM_PAGES    16
#define PAGE_PTR(p)  (&smem_pool[(p) * PAGE_SIZE])

__shared__ char smem_pool[SMEM_TOTAL];
```

Lifecycle:
```
Task N:  compute phase uses pages 0-3
         cp.async loads Task N+1 weights into pages 4-7
Task N completes: releases pages 0-3
Task N+1: uses pages 4-7 (already warm)
          cp.async loads Task N+2 weights into pages 0-3 (recycled)
```

Page budget per task:
- Matmul compute scratch: 4-6 pages (56-84 KB)
- Weight prefetch (next task): 2-4 pages (28-56 KB)
- Activation handoff: 1 page (14 KB for hidden<=4096)

SMEM handoff (free once paged infrastructure exists):
- RMSNorm writes output to SMEM page instead of HBM
- Next matmul reads from SMEM page (30 cycles vs L2's 200 cycles)
- 60 norms × 170 cycle savings ≈ 7us saved

---

## 8. Fusion Framework

### 8a. Matmul Epilogue Fusion

Pattern match after graph walk:

```python
# fusion.py
def fuse_matmul_epilogues(tasks, buffer_sizes):
    for i, task in enumerate(tasks):
        if task.op_type != OpType.MATMUL:
            continue
        consumers = find_consumers(i, tasks)
        for ci, consumer in consumers:
            if consumer.op_type == OpType.ELEMENTWISE:
                if consumer.op_code == ElemCode.ADD:
                    other_buf = get_other_input(consumer, task.buffer_indices[0])
                    if is_1d_bias(other_buf, task.dimensions[1], buffer_sizes):
                        task.strides[1] |= EPILOGUE_BIAS
                        task.buffer_indices[3] = other_buf
                        tasks[ci].num_tiles = 0  # mark dead
                elif consumer.op_code in (ElemCode.SILU, ElemCode.GELU, ElemCode.GELU_TANH):
                    task.strides[1] |= EPILOGUE_MAP[consumer.op_code]
                    task.buffer_indices[0] = consumer.buffer_indices[0]
                    tasks[ci].num_tiles = 0

    return [t for t in tasks if t.num_tiles > 0]
```

Expected savings: 3-6 eliminated tasks per transformer layer × ~5us = 15-30us.

### 8b. Expanded Micro-Op Interpreter

Lift limits:

| | Current | New |
|---|---------|-----|
| Max uops | 8 | 32 |
| Max registers | 8 | 16 |
| Max buffer slots | 8 | 16 |
| Uop storage | `task.strides[0..7]` | Dynamic SMEM (uop array pre-loaded) |
| Reduction ops | None | REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN |
| Broadcast LOAD | None | LOAD_BROADCAST with repeat stride |

Micro-op lowering pass: given a chain of fusable ATen ops, automatically generate a micro-op program:

```python
# micro_op_lowering.py
def lower_to_micro_ops(chain_tasks, buffer_sizes):
    """Convert a chain of elementwise + simple reduction ATen ops
    into a micro-op program for the fused_elementwise interpreter."""
    program = []
    reg_alloc = {}
    # ... (register allocation + instruction emission)
    return program, buffer_slots
```

Any ATen op that decomposes into elementwise primitives automatically becomes a micro-op candidate. This is megabake's answer to Inductor's `Pointwise.create(inner_fn)` -- same expressiveness, different representation (GPU bytecode vs Python closures).

### 8c. Op Type Consolidation

12 → 8 op types:

```python
class OpType(IntEnum):
    MATMUL      = 0x01  # absorbs MATMUL_SILU/GELU/GELU_TANH (epilogue via strides[1])
    ATTENTION   = 0x02
    ELEMENTWISE = 0x03  # absorbs COPY (subtype 0x0200) and FUSED_ELEMENTWISE (subtype 0x0100)
    REDUCE      = 0x04
    EMBEDDING   = 0x05
    INDEX       = 0x06
    ROPE        = 0x08
    EXTERN      = 0x0D  # NEW: graph-split fallback (marker for split point)
```

Dispatch shrinks from 12 `case` branches to 8. Less I-cache pressure. `__noinline__` on cold paths (embedding, index) to reduce register pressure.

---

## 9. Prefill Strategy

For M>=16 (compute-bound regime), CUTLASS multi-config. No cuBLAS -- CUTLASS uses identical WGMMA/mma.sync instructions. Close the gap with tile variety:

```python
TILE_CONFIGS = [
    # (config_id, BM, BN, BK, stages) — SM90 defaults, SM80 uses BK/2
    (0, 64, 64, 64, 3),     # small M and N (e.g., M=64, N=256)
    (1, 64, 128, 64, 3),    # rectangular, common MLP shapes
    (2, 128, 128, 64, 3),   # current default, large square-ish shapes
    (3, 128, 256, 64, 3),   # very wide N (e.g., MLP up-projection)
    (4, 256, 128, 64, 3),   # very tall M (e.g., long-context prefill)
]

def select_matmul_config(M, N, K, num_sms):
    if M <= 4:
        return SKINNY_MATVEC_ID
    best_score, best_id = 0, 1
    for config_id, BM, BN, BK, stages in TILE_CONFIGS:
        tiles = math.ceil(M / BM) * math.ceil(N / BN)
        sm_util = min(tiles, num_sms) / num_sms
        waves = math.ceil(tiles / num_sms)
        wave_eff = tiles / (waves * num_sms)
        score = sm_util * wave_eff
        if score > best_score:
            best_score, best_id = score, config_id
    return best_id
```

Config stored in upper byte of `strides[1]`: `task.strides[1] |= (config_id << 8)`.

No cuBLAS fallback. If CUTLASS multi-config leaves a gap on specific shapes, add more tile configs (e.g., 64×64 for small M, 256×128 for wide layers) until the gap closes. CUTLASS uses the same WGMMA/mma.sync instructions as cuBLAS -- the gap is tile selection and pipeline tuning, not fundamental. Breaking out of the megakernel to call cuBLAS kills prefetch overlap and SMEM handoff, costing more than the GEMM quality difference saves.

---

## 10. Distribution (HF Hub)

Architecture (brief -- see `hf-kernel-hub-integration.md` for full detail):

**Kernel binary** (model-independent, compiled per hardware):
```
megabake/megabake-runtime  ← one repo on HF Kernels Hub
  torch213-cxx11-cu128-sm80/  ← A100
  torch213-cxx11-cu130-sm90/  ← H100/H200
```

**Task schedule** (model-specific, pure data):
```
meta-llama/Llama-3-8B/
  megabake/
    decode-sm_90-fp16.schedule   ← binary task schedule
    prefill-sm_90-fp16.schedule
    index.json                   ← maps (hardware, dtype) → schedule file
```

**API**:
```python
# End user
model = megabake.load("meta-llama/Llama-3-8B")
output = model(input_ids)

# Model author
megabake.bake(pt_model, example_input, push_to_hub=True)
```

Graceful fallback: if no schedule matches hardware/dtype, fall back to `torch.compile`. Megabake is an acceleration layer, not a hard dependency.

---

## 11. Execution Roadmap

### Phase 1: Coverage + Easy Kernel Wins (Weeks 1-2)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Inductor pre_grad_passes integration | 1 day | 142 new decomps + 100 patterns + CSE/DCE. Deletes 345 lines. | new `inductor_passes.py`, delete from `graph_walker.py` |
| Graph splitting for unsupported ops | 2 days | Compilation never fails | `graph_walker.py`, new split logic |
| Vectorize reduce.cu (float4 all paths) | 2 days | 2-4x per reduce task | `reduce.cu` |
| Single-pass RMSNorm (register cache) | 1 day | 2x on RMSNorm | `reduce.cu` |
| Vectorize rope.cu (float4) | 1 day | 4x on RoPE | `rope.cu` |
| Vectorize index.cu (float4 when aligned) | 0.5 day | 2-4x on gather/index_select | `index.cu` |

**Entry criteria**: Current tests pass.
**Exit criteria**: All existing benchmarks still correct (max_diff < 1e-3). New models compile via graph splitting. Reduce/rope/index profiled 2x+ faster.

### Phase 2: Matmul + Fusion (Weeks 2-4)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Epilogue fusion (bias + act + residual via flags) | 3 days | 5-12% on biased models | `matmul.cu`, `graph_walker.py`, new `fusion.py` |
| Op type consolidation (12→8) | 1.5 days | 2-5% I-cache | `data_types.py`, `data_types.cuh`, `megakernel.cu`, `tiling.py` |
| Skinny matvec cp.async | 1 week | 1.5-2x on M<=4 matmul | `matmul.cu` |
| Weight pre-transposition | 2 days | 10-20% on decode matvec | `launcher.py` |
| Expand micro-op interpreter | 1 week | General pointwise coverage | `fused_elementwise.cu`, new `micro_op_lowering.py` |

**Entry criteria**: Phase 1 complete.
**Exit criteria**: Biased-linear models show fewer tasks in profile. Skinny matvec bandwidth util >= 65%. Micro-op interpreter handles 30+ uop chains.

### Phase 3: Scheduler (Weeks 4-6)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Dependency DAG extraction | 2 days | Prerequisite | new `dependency.py` |
| Static per-SM assignment | 3 days | Prerequisite | new `scheduler.py` |
| Counter-based device loop | 1 week | Eliminate ~250us barriers + ~2000us SM idle | `megakernel.cu`, `data_types.cuh` |
| Extended serialization | 2 days | Wire up new data structures | `serializer.py`, `launcher.py`, `loader.py` |
| BSP fallback flag | 0.5 day | Safety | `megakernel.cu` |

**Entry criteria**: Phase 2 complete.
**Exit criteria**: Counter-based scheduler matches BSP output (bitwise). >= 500us saved vs BSP on SmolLM2-135M. 1000-iteration stress test passes.

### Phase 4: Attention (Weeks 6-9)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| FlashAttention rewrite | 2-3 weeks | 3-5x on attention tasks | `attention.cu` |
| Tiling update for query blocks | 0.5 day | More tiles for attention | `tiling.py` |

**Entry criteria**: Phase 3 complete (counter-based scheduler lets attention tiles run on subset of SMs while others work).
**Exit criteria**: Attention within 30% of cuDNN FlashAttention at seq_k=2048. max_diff < 1e-2. gemma-2b >= 0.8x vs torch.compile.

### Phase 5: Prefetch + Prefill (Weeks 9-12)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| Paged SMEM infrastructure | 1 week | Prerequisite | `megakernel.cu`, `data_types.cuh` |
| Weight prefetch overlap | 1 week | 50%→78% BW util | `megakernel.cu`, `scheduler.py` |
| SMEM handoff (norm→matmul) | 0.5 week | ~7us latency savings | `reduce.cu`, `megakernel.cu` |
| CUTLASS multi-config (3 tiles) | 2 weeks | 1.5-3x prefill matmul | `matmul.cu`, `tiling.py` |

**Entry criteria**: Phase 3 complete (requires static per-SM assignment).
**Exit criteria**: Bandwidth util >= 75% in standalone matvec chain. Prefill matmul within 15% of standalone CUTLASS benchmark on same shapes.

### Phase 6: Distribution + Polish (Weeks 12-15)

| Task | Effort | Impact | Files |
|------|--------|--------|-------|
| HF Hub integration | 1 week | Distribution | new `distribution/` |
| INT8/FP8 weight quantization | 1-2 weeks | 2x decode speedup | `matmul.cu`, new dequant kernel |
| BF16 support | 1 week | Broader model support | All CUDA files |

**Entry criteria**: Phases 1-5 complete.
**Exit criteria**: Top-10 HF models compile and run. Performance targets met.

---

## 12. Performance Targets

### Decode (M<=4, batch=1)

| Phase | vs torch.compile | vs CUDA Graphs | Mechanism |
|-------|-----------------|----------------|-----------|
| Current | 0.17-0.32x | worse | — |
| After Phase 1 (vectorize) | 0.3-0.5x | worse | Faster reduce/rope/index |
| After Phase 2 (matmul+fusion) | 0.5-0.8x | worse | cp.async skinny, epilogue fusion |
| After Phase 3 (scheduler) | 0.9-1.3x | 0.7-1.0x | No barriers, no idle SMs |
| After Phase 4 (attention) | 1.2-1.8x | 0.9-1.4x | Tensor core attention |
| After Phase 5 (prefetch) | 1.5-2.5x | 1.2-1.8x | 78% bandwidth util |

### Prefill (M>=16, batch=1)

| Phase | vs torch.compile | Mechanism |
|-------|-----------------|-----------|
| Current | 0.17-0.32x | — |
| After Phase 5 (3 tiles) | 0.8-1.0x | CUTLASS multi-config |
| After tile tuning (5+ tiles) | 0.9-1.1x | More tile configs for edge shapes |

---

## 13. Verification Strategy

Every phase follows this protocol:

```bash
# 1. Correctness: all built-in benchmarks
python benchmarks/bench_compare.py
# max_diff < 1e-3 for all models (< 1e-2 for attention changes)

# 2. Performance: per-task profiling (Phase 0 infrastructure)
python benchmarks/test_harness.py mlp_silu --task-profile
python benchmarks/test_harness.py llama_decoder --task-profile
# Compare against previous phase baselines in benchmarks/baselines/

# 3. Coverage: track which models compile
python -c "import megabake; megabake.compile(model, x)"
# Log: model_name, success/fail/split, num_segments, num_tasks

# 4. Regression: no benchmark regression
# Phase N perf >= Phase N-1 perf on all models

# 5. Stress test (scheduler phases)
for i in range(1000):
    out = megabake.run(compiled, model, x)
# Must not deadlock, output must be deterministic
```

Commit discipline: each phase gets 1-3 commits. Include before/after profile numbers in commit messages. Save baselines to `benchmarks/baselines/`.

---

## 14. Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FlashAttention inside persistent kernel: SMEM contention | High | Medium | `__launch_bounds__(256,1)` gives full 228KB. FA needs ~64KB. No conflict -- one task per SM at a time. |
| Counter polling cache-line bouncing | Medium | Medium | Pad `dep_count[]` to 128-byte boundaries. Each SM polls only its own tasks. |
| Inductor import cost | Low | Certain | One-time startup. `pre_grad_passes` imports pull in `lowering.py` (~2s). Acceptable for a compiler that runs nvcc (~10-60s). |
| `pre_grad_passes` API breaks | Medium | Low | Stable across PyTorch 2.4-2.6. Fallback: revert to `select_decomp_table()` only + temporary hand-written patterns. |
| Graph splitting perf overhead | Medium | Low | Each split = cooperative launch overhead (~48us). Minimize splits via expanded micro-op coverage. |
| `matmul_scalar` hit on real models | High | Low | Graph walker always detects transpose via StridedView (`strides[0]=1`). Add assertion + warning if scalar path triggered. |
| SMEM overflow in attention (long sequences) | High | Medium | FlashAttention rewrite eliminates this: O(1) SMEM per head via K-tiling. Until Phase 4, cap seq_k to SMEM capacity and warn. |
| Register pressure from dispatch switch | Medium | Low | `__noinline__` on cold paths (embedding, index). 8 op types (down from 12). LTO optimizes globally. |
| Benchmark inconsistency (README vs results.json) | Low | Certain | Resolve before Phase 1: re-run benchmarks, update README, commit results.json. Single source of truth. |
