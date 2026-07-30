# Before vs After: Full Execution of REDESIGN.md

---

## Compilation Pipeline

**Before:**
```
torch.export → core_aten_decomps → hand-written RMSNorm/RoPE patterns
  → constant fold → ATEN_OP_MAP lookup (crash on unknown)
  → 3 peephole fusion passes → buffer plan → serialize → nvcc → BSP launch
```

**After:**
```
torch.export → Inductor pre_grad_passes (decomps + 100 patterns + CSE + DCE)
  → 3-tier lowering (dedicated kernel / micro-op / graph split)
  → epilogue fusion + expanded chain fusion + cost model
  → dependency DAG → static per-SM assignment + page planning
  → buffer plan → serialize (with SM queues + dep_counts) → nvcc → counter-based launch
```

---

## Op Coverage

| | Before | After |
|---|--------|-------|
| ATen ops handled | 40 (hardcoded in `op_table.py`) | 1100+ via Inductor decomps + micro-op lowering |
| On unknown op | `RuntimeError`, compilation crashes | Graph split, runs via eager PyTorch between segments |
| Pattern matching | 2 hand-written patterns (RMSNorm, RoPE), 240 lines, breaks across PT versions | 100+ declarative patterns from Inductor, auto-updated with upstream |
| Constant folding | Custom `_constant_fold()` (105 lines) | Inductor's battle-tested constant folding |
| Models that compile | Simple MLPs + LLaMA-style decoders only | Any `torch.export`-able model. Vision models, encoder transformers, MoE (via graph splitting for unsupported ops) |

---

## Kernel Performance

| Kernel | Before | After |
|--------|--------|-------|
| **Attention** | CUDA-core FMA, O(seq_k) SMEM, serial queries, 3-pass softmax. 3-5x slower than FlashAttention. | Tensor core (WGMMA/mma.sync), K-tiled, online softmax, cp.async KV prefetch, O(1) SMEM. Within 30% of FlashAttention. |
| **RMSNorm** | Scalar loads, 2 passes (reads row twice) | float4 vectorized, single pass (register-cached), 2x faster |
| **LayerNorm** | Scalar loads, 3 passes | float4, 2 passes, ~1.5x faster |
| **Softmax** | Scalar, 3 passes | float4, 2 passes |
| **Rope** | Scalar element-by-element | float4 pipeline, 4x fewer load/store instructions |
| **Index** | Scalar `out[i] = src[...]` | float4 when inner_size % 8 == 0, 2-4x faster |
| **Skinny matvec** (M<=4) | float4 B loads, synchronous K-chunks (no overlap) | cp.async double-buffered SMEM, load/compute overlap. ~1.5-2x faster. |
| **Matmul epilogue** | 3 redundant op types. Bias in buffer_indices[3] ignored by CUDA. Separate ADD task + barrier per bias. | Runtime flags in strides[1]. Bias/act/residual fused into matmul writeback. 3-6 fewer tasks per transformer layer. |
| **Matmul prefill** | Fixed 128x128 tiles | 5 CUTLASS tile configs, compile-time shape-dependent selection |
| **Elementwise** | Already float4 vectorized (good) | Same (keep) |
| **Fused elementwise** | 8 uops, 8 regs, 8 bufs | 32 uops, 16 regs, 16 bufs, reduction ops, broadcast LOAD |

---

## Scheduler

| | Before | After |
|---|--------|-------|
| Sync mechanism | `grid.sync()` after every task | Per-dependency atomic counters, no global barrier |
| SM utilization on RMSNorm (batch=1) | 1/32 active (97% idle), all 32 pay barrier | 1 SM runs RMSNorm, 31 SMs run their own queued tasks |
| Barrier overhead | ~2.5-10us × 50-100 tasks = 250-500us wasted | Zero barriers. Counter signal ~0.1-0.3us per dependency edge |
| Independent tasks (Q/K/V projections) | Serialized (one after another, barrier between each) | Parallel (different SMs run Q, K, V simultaneously) |
| Weight loading | Serial: finish task, load weights for next task | Paged SMEM: cp.async prefetches next task's weights DURING current task's compute |
| Bandwidth utilization | ~25-50% | ~75-78% |
| Inter-task data | All through HBM (or L2 if lucky) | SMEM handoff for norm→matmul (30 cycles vs 200 cycles) |

---

## Fusion

| | Before | After |
|---|--------|-------|
| Matmul + activation | Works (MATMUL_SILU/GELU/GELU_TANH op types) | Works (epilogue flags, cleaner) |
| Matmul + bias | **Broken**. Graph walker sets buffer_indices[3], CUDA kernel ignores it. | Works. `EPILOGUE_BIAS` flag, kernel reads bias and adds in writeback. |
| Matmul + residual | Not supported | Works. `EPILOGUE_RESIDUAL` flag, buffer_indices[4]. |
| Matmul + bias + act + residual | 4 separate tasks + 3 barriers | 1 task, zero barriers |
| Elementwise chains | 8 uops max, 8 buffers, no reductions | 32 uops, 16 buffers, reduction/broadcast support |
| Op types | 12 (with 3 redundant matmul variants) | 8 (consolidated, cleaner dispatch) |

---

## Code Size

| | Before | After |
|---|--------|-------|
| `graph_walker.py` | 1275 lines (does everything) | ~600 lines (graph walk + lowering only) |
| Hand-written patterns | 345 lines (`_find_rmsnorm_patterns`, `_find_rope_patterns`, `_constant_fold`, `_build_users_map`) | 0 lines (deleted, replaced by `inductor_passes.py` ~50 lines) |
| New Python files | 0 | `inductor_passes.py` (50), `micro_op_lowering.py` (120), `fusion.py` (200), `dependency.py` (60), `scheduler.py` (200) = ~630 lines |
| `attention.cu` | 143 lines (naive) | ~350-400 lines (FlashAttention with tensor cores) |
| `reduce.cu` | 171 lines (scalar) | ~200 lines (float4, single-pass norms) |
| `megakernel.cu` | 78 lines (BSP loop) | ~150 lines (counter-based + BSP fallback) |
| Net Python | ~2400 lines | ~2700 lines (+300 net: +630 new, -345 deleted, small edits elsewhere) |
| Net CUDA | ~1900 lines | ~2300 lines (+400 net: attention rewrite, extended megakernel, vectorized kernels) |
| **Net total** | ~4300 lines | ~5000 lines |
| Planning docs | ~10,000 lines across 7 docs | REDESIGN.md is authoritative (~1000 lines). Others kept as historical reference. |

---

## Performance (Expected)

| Model | Before | After Phase 5 | Change |
|-------|--------|--------------|--------|
| linear_256x512 | 0.27x vs torch.compile | 1.5-2.0x | ~6-7x improvement |
| mlp_silu | 0.17x | 1.5-2.0x | ~9-12x improvement |
| rmsnorm_mlp | 0.17x | 1.5-2.0x | ~9-12x improvement |
| llama_decoder | 0.32x | 1.5-2.5x | ~5-8x improvement |
| SmolLM2-135M decode | ~0.88x (README) / worse (results.json) | 1.5-2.5x | ~2-3x improvement |
| gemma-2b decode | ~0.49x | 1.2-1.8x | ~2.5-3.5x improvement |

Sources of improvement stacking:
- Vectorized reduce/rope/index: ~1.2-1.5x
- Skinny matvec cp.async: ~1.3-1.5x
- Epilogue fusion (fewer tasks): ~1.1-1.2x
- Counter-based scheduler (no barriers, no idle SMs): ~1.3-1.5x
- FlashAttention (tensor cores): ~1.5-3x on attention-heavy models
- Weight prefetch overlap (78% bandwidth): ~1.3-1.5x

These multiply, not add. Combined: 1.5-2.5x over torch.compile on decode.

---

## What Stays the Same

These don't change -- already correct:

- `matmul.cu` SM90 WGMMA and SM80 mma.sync GEMM paths (production-grade, keep)
- `elementwise.cu` float4 vectorized paths (well done, keep)
- `StridedView` shape tracking in `shape_ops.py` (correct approach, keep)
- `buffer_planner.py` liveness + first-fit arena (works fine, keep)
- Binary schedule format (TaskDesc/BufferDesc/WeightMapping/ScheduleHeader, keep)
- `_CachedRunner` in `loader.py` (clean caching, keep)
- Per-task profiling infrastructure (essential, keep)
- `pyproject.toml`, test structure, benchmark harness structure (keep)

---

## What Doesn't Exist Yet That Will

Things the project gains that it has zero of today:

1. **Compilation never fails** -- graph splitting means any model produces output
2. **Dependency-aware scheduling** -- tasks know what they depend on
3. **Per-SM task queues** -- each SM has its own work list
4. **Weight prefetch** -- next task's weights loading during current compute
5. **SMEM paging** -- shared memory managed as fixed-size pages
6. **SMEM handoff** -- norm output stays in SMEM for next matmul
7. **Tensor core attention** -- WGMMA/mma.sync for QK^T and PV
8. **Online softmax** -- no materialized score matrix
9. **Matmul bias fusion** -- bias applied in matmul writeback, not separate task
10. **Micro-op reduction ops** -- sum/mean/max in interpreter
11. **Automatic micro-op lowering** -- any pointwise chain auto-compiles to bytecode
