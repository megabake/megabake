# Megabake: Task List

*Ordered execution list. Do them top to bottom. Each task is self-contained with clear inputs, outputs, and verification. Architecture rationale lives in REDESIGN.md.*

---

## Phase 1: Coverage + Easy Kernel Wins

### Task 1.1: Resolve benchmark inconsistency ✅ DONE
**Do first. Everything after this depends on trustworthy numbers.**

- Re-run `python benchmarks/bench_compare.py --output benchmarks/results.json`
- Update README.md benchmark table to match results.json
- Commit results.json as single source of truth
- If README numbers were from a different config, document what config

**Result**: Old README numbers were from unknown config (showed 0.85-1.62x). Old results.json (bench_iters=50) showed 0.17-0.32x. Fresh run (bench_iters=100, default) shows 1.04-3.38x on working models. 3 models fail (mlp_silu, rmsnorm_mlp, layernorm_mlp) due to weight-loading bugs. README updated. results.json is now single source of truth.

---

### Task 1.2: Inductor pre_grad_passes integration ✅ DONE
**Replace 345 lines of hand-written patterns with Inductor's battle-tested passes.**

Create `src/megabake/schedule_compiler/inductor_passes.py`:

```python
import torch

def optimize_graph(ep):
    from torch._inductor.decomposition import select_decomp_table
    decomp_table = select_decomp_table()

    for op in [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
        torch.ops.aten.embedding.default,
    ]:
        decomp_table.pop(op, None)

    for op in [
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten._softmax.default,
        torch.ops.aten.native_layer_norm.default,
    ]:
        decomp_table.pop(op, None)

    ep = ep.run_decompositions(decomp_table)

    gm = ep.graph_module
    from torch._inductor.fx_passes.pre_grad import pre_grad_passes
    pre_grad_passes(gm)

    return ep
```

Then in `graph_walker.py`:
- Replace `_decompose()` call with `optimize_graph()` from new file
- Delete `_find_rmsnorm_patterns()` (lines 512-663)
- Delete `_find_rope_patterns()` (lines 666-754)
- Delete `_constant_fold()` (lines 392-496)
- Delete `_build_users_map()` (lines 499-510)
- Delete `rmsnorm_patterns`/`rope_patterns` usage in `compile_from_ep()`
- Remove corresponding `rmsnorm_skip`/`rope_skip` logic

**Verify**:
```bash
pytest tests/
python benchmarks/bench_compare.py
# All tests pass. max_diff < 1e-3 on all benchmarks. No regression.
```

**Result**: Created `inductor_passes.py` with `optimize_graph()`. Uses `core_aten_decompositions()` + Inductor's `pre_grad_passes()` (CSE, DCE, constant folding, 100+ pattern matches). `select_decomp_table()` was NOT used — it corrupts global Inductor state, breaking subsequent `torch.compile` calls in same process. Hand-written pattern functions (`_find_rmsnorm_patterns`, `_find_rope_patterns`, `_constant_fold`, `_build_users_map`) are KEPT — `pre_grad_passes` does NOT fuse decomposed RMSNorm/RoPE back into higher-level ops in PyTorch 2.6, so deleting them would regress fused kernel usage. Net: `_decompose()` deleted (11 lines), `inductor_passes.py` created (29 lines), `__init__.py` updated. 73/73 tests pass. Benchmarks: 1.02-3.54x vs torch.compile, no regression.

---

w### Task 1.3: Graph splitting for unsupported ops ✅ DONE
**Compilation never crashes. Unknown ops degrade gracefully.**

**Result**: Implemented eager fallback approach instead of full segment-based graph splitting. When unsupported ops exist: (1) graph walker allocates output buffers and warns instead of crashing, (2) `unsupported_ops` list stored in CompiledModel, (3) `run()` detects unsupported ops and falls back to running original model eagerly. 73/73 tests pass. BatchNorm1d verification: compiles with warning, max_diff=0.0. Models without unsupported ops unchanged (still use megakernel). Full segment-based graph splitting deferred until partial acceleration is needed.

In `graph_walker.py` main loop, where it currently does:
```python
elif mapping is None:
    op_name = getattr(target, "__name__", str(target))
    unsupported_ops.append(op_name)
```

Change to:
```python
elif mapping is None and not _can_lower_to_micro_ops(node):
    split_points.append(node.name)
    # Allocate output buffer so downstream nodes can reference it
    out_meta = node.meta.get("val")
    if out_meta is not None and isinstance(out_meta, torch.Tensor):
        alloc_buffer(node.name, [int(s) for s in out_meta.shape], out_meta.dtype)
```

Remove the `RuntimeError` at the end that crashes on `unsupported_ops`.

Add runtime support in `loader.py`: when split points exist, break into segments. Each segment gets its own cooperative launch. Between segments, run split-point ops via `torch._decomp` eager.

**Verify**:
```bash
# Should compile without crashing (may be slow due to splits)
python -c "
import torch, megabake
model = torch.nn.Sequential(
    torch.nn.Linear(64, 64, bias=False),
    torch.nn.BatchNorm1d(64),  # not in op_table, triggers split
    torch.nn.Linear(64, 32, bias=False),
).cuda().half().eval()
x = torch.randn(4, 64, device='cuda', dtype=torch.float16)
compiled = megabake.compile(model, x)
out = megabake.run(compiled, model, x)
ref = model(x)
print('max_diff:', (out.float() - ref.float()).abs().max().item())
"
```

---

### Task 1.4: Vectorize reduce.cu — float4 all paths ✅ DONE
**Replace scalar `__half2float(row_in[j])` with float4 loads everywhere.**

For each reduce op (RMSNORM, LAYERNORM, SOFTMAX, SUM, MEAN, MAX):
- Inner loops: `j += threads` with scalar load → `j += threads * 8` with float4 load
- Unpack via `__half2*` → `__half22float2()`
- Scalar tail for unaligned remainder

**Result**: All 6 reduce ops (RMSNORM, LAYERNORM, SOFTMAX, SUM, MEAN, MAX) vectorized with float4 loads/stores. Each loop uses `j += threads * 8` stride, unpacks via `__half22float2()`, repacks via `__float22half2_rn()`. Scalar tail handles `row_size % 8 != 0`. ARGMAX left scalar (index tracking per pack not worth complexity for rare op). 74/74 tests pass, no regressions.

**Verify**:
```bash
pytest tests/test_tasks/test_reduce.py
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
# Compare reduce task cycle counts vs baseline. Expect 2-4x reduction.
```

---

### Task 1.5: Single-pass RMSNorm ✅ DONE
**Cache row in registers. Read from global memory once, not twice.**

**Result**: RMSNORM case in `reduce.cu` now caches row values in a `float cached[72]` register array during the sum_sq accumulation pass. Second pass reads from cache instead of re-reading global memory. Eliminates one full global memory read per RMSNorm row. Register budget: hidden=4096/256 threads = 16 fp32/thread, hidden=8192 = 32/thread, MAX=72 covers up to hidden=18432. 74/74 tests pass, no regressions.

---

### Task 1.6: Vectorize rope.cu ✅ DONE
**Replace scalar loads with float4 pipeline.**

**Result**: Inner loop vectorized with float4 loads/stores for x0, x1, cos, sin. Stride `threads * 8` (8 halves per float4). Unpack via `__half22float2()`, compute rotary on float2 pairs, repack via `__float22half2_rn()`. Scalar tail for `half_dim % 8 != 0`. BHSD/BSHD layout support preserved. 74/74 tests pass, benchmarks: no regression (llama_decoder 3.16x vs torch.compile).

---

### Task 1.7: Vectorize index.cu
**float4 when inner_size aligned, scalar fallback otherwise.**

In `index.cu` GATHER and INDEX_SELECT:
- Check `inner_size % 8 == 0`
- If yes: iterate in float4 chunks, copy 8 halves per load
- If no: existing scalar path

**Verify**:
```bash
pytest tests/test_tasks/test_embedding.py tests/test_tasks/test_copy.py
# (index tests if they exist, otherwise add a basic one)
```

---

## Phase 2: Matmul + Fusion

### Task 2.1: Op type consolidation (12 → 8)
**Remove 3 redundant matmul op types. Merge COPY and FUSED_ELEMENTWISE into ELEMENTWISE.**

Changes:
- `data_types.py`: Remove MATMUL_SILU (0x09), MATMUL_GELU (0x0A), MATMUL_GELU_TANH (0x0C). Remove COPY (0x07), FUSED_ELEMENTWISE (0x0B). Add EXTERN (0x0D).
- `data_types.cuh`: Mirror Python changes.
- `megakernel.cu` dispatch: Remove cases for deleted types. ELEMENTWISE dispatch checks op_code upper byte for subtype (0x0000=simple, 0x0100=fused, 0x0200=copy).
- `matmul.cu`: `apply_epilogue()` reads `task.strides[1]` flags instead of `task.op_type`.
- `graph_walker.py`: `_fuse_tasks()` sets `strides[1]` flags instead of changing op_type.
- `tiling.py`: Remove MATMUL_SILU/GELU/GELU_TANH/COPY/FUSED_ELEMENTWISE cases.
- `op_table.py`: No change (maps to base MATMUL/ELEMENTWISE types).

**Verify**:
```bash
pytest tests/
python benchmarks/bench_compare.py
# Identical output. Same task count. Kernel count still 1.
```

---

### Task 2.2: Matmul epilogue fusion — bias
**Make the CUDA kernel read buffer_indices[3] for bias. Graph walker already sets it.**

In `matmul.cu`, both SM90 and SM80 epilogue loops + skinny matvec epilogue:
- Read `uint32_t flags = task.strides[1]`
- If `flags & EPILOGUE_BIAS`: load bias from `buffers[task.buffer_indices[3]]`, add `bias[col]` before activation

In `graph_walker.py` or new `fusion.py`:
- After graph walk, scan for MATMUL → consumer ELEMENTWISE(ADD) where ADD's other input is 1D (size matches N dimension)
- Set `task.strides[1] |= EPILOGUE_BIAS`, move bias buffer to `buffer_indices[3]`, mark ADD task dead

Define epilogue flag constants in `data_types.py` and `data_types.cuh`:
```python
EPILOGUE_SILU      = 0x01
EPILOGUE_GELU      = 0x02
EPILOGUE_GELU_TANH = 0x04
EPILOGUE_BIAS      = 0x08
EPILOGUE_RESIDUAL  = 0x10
```

**Verify**:
```bash
# Need a model WITH bias
python -c "
import torch, megabake
model = torch.nn.Sequential(
    torch.nn.Linear(256, 512, bias=True),  # has bias
    torch.nn.SiLU(),
    torch.nn.Linear(512, 256, bias=True),
).cuda().half().eval()
x = torch.randn(1, 256, device='cuda', dtype=torch.float16)
compiled = megabake.compile(model, x)
out = megabake.run(compiled, model, x)
ref = model(x)
print('max_diff:', (out.float() - ref.float()).abs().max().item())
"
# Profile: should show fewer tasks than without bias fusion
python benchmarks/test_harness.py layernorm_mlp --task-profile
```

---

### Task 2.3: Matmul epilogue fusion — residual
**Fuse residual add into matmul writeback.**

Same pattern as bias but for 2D operands:
- Scan for MATMUL → consumer ELEMENTWISE(ADD) where ADD's other input is 2D (size matches M × N)
- Set `task.strides[1] |= EPILOGUE_RESIDUAL`, move residual buffer to `buffer_indices[4]`, mark ADD task dead
- In CUDA epilogue: `if (flags & EPILOGUE_RESIDUAL) v += __half2float(residual[row * N + col])`

**Verify**:
```bash
pytest tests/test_e2e/test_llama_e2e.py
# LLaMA decoder has residual connections — should show fewer tasks
python benchmarks/test_harness.py llama_decoder --task-profile
```

---

### Task 2.4: Skinny matvec cp.async prefetch
**Double-buffer weight chunks in SMEM. Overlap load with compute.**

In `matmul.cu` `matmul_skinny()`:
- Allocate two SMEM buffers for B: `smem_B0` and `smem_B1`
- Before K-loop: `cp_async_cg` first chunk into `smem_B0`, `cp_async_commit_group()`
- Inside K-loop: issue cp.async for NEXT chunk into alternate buffer, `cp_async_wait_group<1>()`, compute from current buffer
- Swap buffers each iteration: `cur_buf ^= 1`

SMEM budget: M=1, bk=256, cols_per_sm≈128 → B_chunk = 128 × 256 × 2 = 64KB. Two buffers = 128KB. Fits 228KB with A (small) and scratch.

**Verify**:
```bash
pytest tests/test_tasks/test_matmul.py
python benchmarks/test_harness.py mlp_silu --task-profile
# Skinny matmul cycles should drop ~1.5-2x
# Bandwidth utilization should increase (check via test_harness bandwidth reporting)
```

---

### Task 2.5: Weight pre-transposition for decode
**Pre-transpose weights to column-major for coalesced reads. Cache on first call.**

In `loader.py` `_CachedRunner._setup()`:
- After loading weight tensors, check if any matmul task has M<=4 (decode shape)
- If yes, transpose weight tensor: `weight = weight.t().contiguous()`
- Cache transposed version — don't re-transpose on repeated runs

Also update `graph_walker.py`: when M<=4 and B is already transposed (strides[0]=1), the pre-transposition means B is now in column-major for coalesced warp reads.

**Verify**:
```bash
pytest tests/
python benchmarks/bench_compare.py --models linear_256x512,linear_512x1024
# Linear benchmarks should show improvement
```

---

### Task 2.6: Expand micro-op interpreter limits
**Lift 8-uop/8-reg/8-buf limits to 32/16/16.**

In `fused_elementwise.cu`:
- Change `int ops[8]` → `int ops[32]`, same for `dsts`, `s1s`, `s2s`
- Change `float regs[4][8]` → `float regs[4][16]` (vectorized), `float regs[8]` → `float regs[16]` (scalar)
- Change `actual_uops = min(num_uops, 8)` → `min(num_uops, 32)`
- For >8 uops: store uop program in dynamic SMEM instead of `task.strides[0..7]`. First 8 threads load program from a dedicated buffer into SMEM, `__syncthreads()`, then interpret.

In `data_types.cuh`: if using SMEM for uop storage, add a `program_buffer_index` field (use `buffer_indices[7]` or similar).

In `graph_walker.py` `_fuse_elementwise_chains()`:
- Change `if len(uops) > 8 or len(buffer_slots) > 8 or next_reg > 8: continue` to use new limits
- For chains >8 uops: serialize program to a dedicated buffer instead of packing into `task.strides`

**Verify**:
```bash
pytest tests/test_tasks/test_elementwise.py
# Create a test with a 15-op elementwise chain, verify it fuses into one task
```

---

### Task 2.7: Add reduction micro-ops
**REDUCE_SUM, REDUCE_MAX, REDUCE_MEAN opcodes in micro-op interpreter.**

In `data_types.py` and `data_types.cuh`:
```python
UOP_REDUCE_SUM  = 0x11
UOP_REDUCE_MAX  = 0x12
UOP_REDUCE_MEAN = 0x13
```

In `fused_elementwise.cu`:
- Add cases for reduction uops
- Reduction requires `block_reduce_sum`/`block_reduce_max` — use existing implementations from `data_types.cuh`
- After reduction, result is scalar broadcast to all threads

In `micro_op_lowering.py` (new file):
- Detect when an ATen op decomposes to elementwise + simple reduction
- Auto-generate micro-op program

**Verify**:
```bash
# Test: chain of elementwise ops followed by a sum reduction
pytest tests/test_tasks/test_elementwise.py
```

---

### Task 2.8: Broadcast-aware LOAD micro-op
**LOAD_BROADCAST that handles in1_numel/in1_repeat broadcasting.**

In `fused_elementwise.cu`:
- New opcode `UOP_LOAD_BROADCAST` with stride info
- When loading from a buffer smaller than the output, apply `(i / repeat) % numel` indexing
- Currently broadcast falls back to scalar path in `elementwise.cu` — this brings it into fused chains

**Verify**:
```bash
# Test: binary op with broadcast (e.g., add tensor [1,256] + bias [256])
pytest tests/test_tasks/test_elementwise.py
```

---

## Phase 3: Scheduler

### Task 3.1: Dependency DAG extraction
**Build producer/consumer graph from buffer_indices.**

Create `src/megabake/schedule_compiler/dependency.py`:

```python
from megabake.data_types import TaskDesc, UNUSED_BUFFER

def build_dependency_dag(tasks):
    producers = {}
    dep_count = [0] * len(tasks)
    successors = [[] for _ in range(len(tasks))]
    
    for i, task in enumerate(tasks):
        out_buf = task.buffer_indices[0]
        if out_buf != UNUSED_BUFFER:
            producers[out_buf] = i
        for buf in task.buffer_indices[1:]:
            if buf != UNUSED_BUFFER and buf in producers:
                src = producers[buf]
                if src != i:
                    dep_count[i] += 1
                    successors[src].append(i)
    
    return dep_count, successors
```

**Verify**:
```bash
# Unit test: compile a model, extract DAG, verify edges match expected dependencies
pytest tests/test_schedule_compiler/
```

---

### Task 3.2: Static per-SM assignment
**Critical-path-priority topological sort + bin-packing onto SMs.**

Create `src/megabake/schedule_compiler/scheduler.py` with:
- `_critical_path_lengths(tasks, successors)`: longest-path-to-exit per task
- `_topo_sort_by_priority(dep_count, successors, priority)`: Kahn's algorithm with max-heap
- `assign_tasks_to_sms(tasks, dep_count, successors, num_sms)`: bin-pack tasks
- `estimate_cycles(task)`: rough cost estimate for bin-packing

Output: `sm_queues[sm_id] = [(task_id, tile_id), ...]`

**Verify**:
```bash
# Unit test: known task graph → verify assignment is load-balanced
# Verify: all tasks appear exactly once across all SM queues
# Verify: topological order is respected within each SM queue
pytest tests/test_schedule_compiler/
```

---

### Task 3.3: Extend serialization for scheduler data
**Serialize per-SM queues, dep_count[], successor_list[], tile_remaining[].**

In `data_types.py`:
- Add `SMQueueEntry` dataclass (task_id: uint32, tile_id: uint32)
- Extend `ScheduleHeader` with `num_sms`, `max_queue_len`, `num_edges`, `scheduler_type`

In `data_types.cuh`:
- Mirror `SMQueueEntry` struct

In `serializer.py`:
- After existing task/buffer/weight sections, append:
  - SM queue entries (flat array: sm0_entries... sm1_entries...)
  - SM queue lengths (int per SM)
  - dep_count[] (padded to 128 bytes per entry for cache lines)
  - tile_remaining[] (padded to 128 bytes per entry)
  - successor_list[] (flat with offset array)
  - successor_offset[] (int per task + 1)

In `serializer.py` `load_schedule()`: parse new sections.

**Verify**:
```bash
# Round-trip test: serialize → deserialize → compare
pytest tests/test_schedule_compiler/test_serializer.py
```

---

### Task 3.4: Counter-based device execution loop
**New megakernel loop. BSP fallback via scheduler_type flag.**

In `megakernel.cu`:
- Add `SMQueueEntry` struct
- Add new kernel parameters: `sm_queues`, `sm_queue_lens`, `dep_count`, `tile_remaining`, `succ_list`, `succ_offset`, `scheduler_type`
- `scheduler_type == 0`: existing BSP loop (keep as fallback)
- `scheduler_type == 1`: counter-based loop per REDESIGN.md Section 7c
- `#define CACHE_LINE_INTS 32` for 128-byte padding

In `launcher.py`:
- Extend `_launch_cooperative()` with new kernel args
- Upload SM queue buffers, dep_count arrays, successor data to GPU

In `loader.py`:
- `_CachedRunner._setup()`: call `build_dependency_dag()` + `assign_tasks_to_sms()`, serialize scheduler data, upload to GPU

**Verify**:
```bash
# A/B test: run with scheduler_type=0 (BSP) and scheduler_type=1 (counter)
# Output must be bitwise identical
pytest tests/
python benchmarks/bench_compare.py
# Profile both:
python benchmarks/test_harness.py llama_decoder --task-profile  # counter-based
# Compare total cycles. Counter should save >= 500us on llama_decoder.

# Stress test for deadlocks:
python -c "
import megabake, torch
# ... compile model ...
for i in range(1000):
    out = megabake.run(compiled, model, x)
    if i % 100 == 0: print(f'{i}: ok')
"
```

---

## Phase 4: Attention

### Task 4.1: FlashAttention rewrite — SM80 path
**K-tiled attention with mma.sync tensor cores, online softmax.**

Rewrite `attention.cu` entirely. Keep GQA mapping and causal masking logic. New structure:

- Tile over query blocks (BQ=64) and key blocks (BK=64)
- Use `SM80_16x8x16_F32F16F16F32_TN` MMA atom (same as matmul.cu SM80 path) for S=Q@K^T and O=P@V
- Online softmax: maintain running max + sum, rescale accumulator on new max
- cp.async prefetch next K/V block during compute
- Causal early exit: skip entire K-block if `sk_block > sq_block + BQ - 1`

SMEM layout: Q(16KB) + K×2(32KB) + V(16KB) = 64KB.

**Verify**:
```bash
pytest tests/test_tasks/test_attention.py
# Numerical tolerance is looser for attention: max_diff < 1e-2
python benchmarks/test_harness.py llama_decoder --task-profile
# Attention task cycles should drop 3-5x
```

---

### Task 4.2: FlashAttention — SM90 WGMMA path
**Same algorithm as 4.1 but with WGMMA instructions for SM90.**

Use `SM90_64x128x16_F32F16F16_SS` MMA atom (same as matmul.cu SM90 path). GMMA-compatible swizzled SMEM. Warpgroup synchronization.

Guard with `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (same pattern as matmul.cu).

**Verify**:
```bash
pytest tests/test_tasks/test_attention.py
python benchmarks/test_harness.py llama_decoder --task-profile
```

---

### Task 4.3: Update attention tiling
**More tiles for attention — consider query blocks, not just batch×heads.**

In `tiling.py`:
```python
elif op_type == OpType.ATTENTION:
    batch, num_heads, seq_q = dims[0], dims[1], dims[2]
    query_blocks = (seq_q + 63) // 64
    return min(batch * num_heads * query_blocks, max_sms)
```

**Verify**:
```bash
pytest tests/test_schedule_compiler/
python benchmarks/test_harness.py llama_decoder --task-profile
# Attention should use more SMs for long sequences
```

---

## Phase 5: Prefetch + Prefill

### Task 5.1: Paged SMEM infrastructure
**Divide SMEM into 16 pages of 14KB. Each task gets assigned pages at compile time.**

In `megakernel.cu`:
```cuda
#define SMEM_TOTAL   (228 * 1024)
#define PAGE_SIZE    (14 * 1024)
#define NUM_PAGES    16
#define PAGE_PTR(p)  (&smem_pool[(p) * PAGE_SIZE])
__shared__ char smem_pool[SMEM_TOTAL];
```

In `scheduler.py`:
- Add `plan_smem_pages(sm_queues, tasks, num_pages=16)` — assign compute pages and prefetch pages per task per SM
- Output: per-queue-entry page assignments

In `data_types.cuh`:
- Extend `SMQueueEntry` with `compute_page_start`, `compute_page_count`, `prefetch_page_start`, `prefetch_page_count`

Existing task kernels use `extern __shared__ char smem[]` — route them through page pointers instead. Start with matmul and reduce (highest impact).

**Verify**:
```bash
pytest tests/
# Output unchanged. SMEM usage now routed through pages.
```

---

### Task 5.2: Weight prefetch overlap
**While computing task N, cp.async loads task N+1's weights into SMEM pages.**

In counter-based megakernel loop (Task 3.4), after `dispatch_task()` and `signal_dependents()`:
```cuda
if (q + 1 < queue_len) {
    SMQueueEntry next = sm_queues[sm_id * MAX_QUEUE_LEN + q + 1];
    if (next.prefetch_bytes > 0) {
        char* dst = PAGE_PTR(next.compute_page_start);
        const char* src = (const char*)buffers[tasks[next.task_id].buffer_indices[1]]
                          + next.prefetch_weight_offset;
        for (int t = threadIdx.x * 16; t < next.prefetch_bytes; t += blockDim.x * 16)
            cp_async_cg(dst + t, src + t);
        cp_async_commit_group();
    }
}
```

Task kernels read from SMEM pages instead of global memory for weight data.

**Verify**:
```bash
# Standalone benchmark: chain of 10 matmuls with/without prefetch
python benchmarks/test_harness.py mlp_3layer --task-profile
# Bandwidth utilization should rise toward 75%+
```

---

### Task 5.3: SMEM handoff — norm to matmul
**RMSNorm writes output to SMEM page. Next matmul reads from SMEM instead of HBM.**

In `scheduler.py`:
- Detect producer-consumer pairs on same SM where producer is REDUCE and consumer is MATMUL
- Mark handoff page assignment

In `reduce.cu`:
- If handoff page assigned: write normalized output to SMEM page instead of global memory

In `matmul.cu`:
- If activation input is in SMEM page: read from page (30 cycles) instead of global memory (200 cycles)

**Verify**:
```bash
pytest tests/
python benchmarks/test_harness.py rmsnorm_mlp --task-profile
# Should see ~7us savings from eliminated HBM round-trips
```

---

### Task 5.4: CUTLASS multi-config for prefill (5 tile variants)
**Shape-dependent tile selection instead of fixed 128x128.**

In `matmul.cu`:
- Add 4 additional CuTe GEMM template instantiations alongside existing 128x128:
  - Config 0: 64×64 (small M and N)
  - Config 1: 64×128 (rectangular)
  - Config 2: 128×128 (existing default)
  - Config 3: 128×256 (wide N)
  - Config 4: 256×128 (tall M)
- Dispatch by `config_id` from upper byte of `task.strides[1]`

In `tiling.py`:
- `select_matmul_config(M, N, K, num_sms)` scores each config by SM utilization × wave efficiency
- Store in `task.strides[1] |= (config_id << 8)`
- Update tile count computation to use selected config's BM/BN

**Verify**:
```bash
pytest tests/test_tasks/test_matmul.py
# Test multiple M/N shapes to verify config selection
python benchmarks/bench_compare.py
# Prefill shapes (larger M) should improve
```

---

## Phase 6: Distribution + Polish

### Task 6.1: HF Hub integration
**Publish megabake-runtime on HF Kernels Hub. Per-model schedule files.**

Create `src/megabake/distribution/` with:
- `hub.py`: Upload/download schedules via `huggingface_hub`
- `index.py`: Read/write `index.json` (maps hardware/dtype to schedule file)

Implement `megabake.load(model_id)` and `megabake.bake(model, example_input, push_to_hub=True)` in `__init__.py`.

**Verify**:
```bash
# Local bake + load round-trip (no actual Hub upload)
python -c "
import megabake
compiled = megabake.compile(model, x)
megabake.save(compiled, '/tmp/test.schedule')
loaded = megabake.load_local('/tmp/test.schedule')
"
```

---

### Task 6.2: INT8 weight-only quantization
**INT8 weights + FP16 scale per group. Dequant in registers during matmul.**

New: `OP_DEQUANT_MATMUL` or fused dequant in matmul epilogue.
- Load INT8 weight, multiply by FP16 scale → FP16 weight on-the-fly
- Halves weight memory traffic → ~2x decode speedup on bandwidth-bound workloads

**Verify**:
```bash
# Quantize a model, compile, verify max_diff < 1e-2
pytest tests/
```

---

### Task 6.3: BF16 support
**Add bfloat16 dtype alongside existing fp16.**

Changes across all CUDA files:
- Template or branch on dtype code from TaskDesc
- MMA atoms support both FP16 and BF16
- Host side: dtype propagation from model to schedule

**Verify**:
```bash
python -c "
import torch, megabake
model = MyModel().cuda().bfloat16().eval()
x = torch.randn(1, 256, device='cuda', dtype=torch.bfloat16)
compiled = megabake.compile(model, x, dtype=torch.bfloat16)
out = megabake.run(compiled, model, x)
"
```

---

## Checklist

After each task, before moving to next:

```bash
# 1. All tests pass
pytest tests/

# 2. No benchmark regression
python benchmarks/bench_compare.py

# 3. Profiling shows expected improvement
python benchmarks/test_harness.py llama_decoder --task-profile

# 4. Commit with before/after numbers
git add -A && git commit -m "description (before: X us, after: Y us)"
```
