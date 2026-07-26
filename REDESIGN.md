# Megabake: First-Principles Redesign Analysis

---

## 0. Current Architecture Diagram

Bottlenecks marked with `[!]`/`[!!]`/`[!!!]` severity. All boxes 81 chars wide.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ COMPILE TIME (Python)                                                         │
│                                                                               │
│ torch.nn.Module                                                               │
│   │                                                                           │
│   ▼                                                                           │
│ torch.export.export(strict=False)                                             │
│   │                                                                           │
│   ▼                                                                           │
│ run_decompositions(core_aten)                                                 │
│   Preserves: SDPA, SiLU, GELU                                                 │
│   │                                                                           │
│   ▼                                                                           │
│ Pattern Matching                                                              │
│   ├─ RMSNorm detection (pow>mean>add>rsqrt>mul)                               │
│   ├─ RoPE detection (slice>neg>cat>mul+mul>add)                               │
│   └─ Constant folding (unsupported ops only)                                  │
│   │                                                                           │
│   ▼                                                                           │
│ Graph Walk (graph_walker.py)                                                  │
│   ├─ Shape ops: zero-cost stride manipulation                                 │
│   ├─ Supported ops: emit TaskDesc                                             │
│   ├─ Identity elimination (mul*1, add+0, cast)                                │
│   └─ Matmul B-transpose: strides[0] flag                                      │
│   │                                                                           │
│   ▼                                                                           │
│ Fusion (3 fixed peephole passes)                                              │
│   ├─ MATMUL + ELEMENTWISE > MATMUL_SILU/GELU                                  │
│   ├─ Elementwise chains > FUSED_ELEMENTWISE                                   │
│   └─ Redundant COPY elimination                                               │
│   [!!] No matmul bias fusion                                                  │
│   [!!] No matmul residual-add fusion                                          │
│   [!!] No norm+scale fusion                                                   │
│   │                                                                           │
│   ▼                                                                           │
│ Buffer Planning + Serialization                                               │
│   ├─ Liveness analysis (task-index based)                                     │
│   ├─ First-fit-decreasing arena packing                                       │
│   └─ Binary: Header | Tasks | Buffers | Weights                               │
│                                                                               │
│ Tiling (tiling.py)                                                            │
│   [!!!] Fixed 128x128 matmul tiles, no autotuning                             │
│   [!!]  Fixed 4096-element tiles for elementwise                              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ CUDA COMPILATION                                                              │
│                                                                               │
│ cuda_compiler.py concatenates ALL .cu into one TU:                            │
│   matmul.cu (576 lines) + attention.cu (186 lines)                            │
│   + elementwise.cu + fused_elementwise.cu                                     │
│   + reduce.cu + embedding.cu + copy.cu + rope.cu                              │
│   + index.cu + megakernel.cu (BSP dispatch loop)                              │
│                                                                               │
│ nvcc -cubin --use_fast_math -std=c++17                                        │
│   [!!] Single binary = all code in one cubin                                  │
│   [!!] I-cache pressure from 1600+ lines of code                              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME — HOST (loader.py + launcher.py)                                      │
│                                                                               │
│ ├─ Load cubin via cuModuleLoad                                                │
│ ├─ Allocate workspace arena (one cudaMalloc)                                  │
│ ├─ Load weights: .contiguous().cuda().half()                                  │
│ ├─ Build pointer array [workspace + weight ptrs]                              │
│ ├─ Upload TaskDesc[] to GPU                                                   │
│ └─ cuLaunchCooperativeKernel(num_sms, 256, 100KB)                             │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME — DEVICE (megakernel.cu)                                              │
│                                                                               │
│ __global__ megakernel()  __launch_bounds__(256, 1)                            │
│ 1 block per SM, 256 threads per block                                         │
│                                                                               │
│ for (i = 0; i < num_tasks; i++) {                                             │
│     if (blockIdx.x < tasks[i].num_tiles)                                      │
│         dispatch_task(tasks[i]);                                              │
│     else                                                                      │
│         [!!!] SM SITS IDLE — wasted cycles                                    │
│                                                                               │
│     grid.sync();  [!!!] BARRIER AFTER EVERY TASK                              │
│ }                 ~2.5 us x ~100 = 250 us wasted                              │
│                                                                               │
│ dispatch_task() switch(op_type):                                              │
│                                                                               │
│ MATMUL  task_matmul()                                                         │
│   B not transposed: scalar fallback                                           │
│     [!!!] NO TENSOR CORES on this path                                        │
│   M <= 4: skinny matmul (all SMs, N-split)                                    │
│   else: CuTe GEMM (SM90 WGMMA / SM80 mma.sync)                                │
│     [!!!] FIXED 128x128 TILES — 0.2-0.3x cuBLAS                               │
│     [!!]  No split-K / stream-K / tile autotuning                             │
│     [!!]  No bias/residual epilogue fusion                                    │
│                                                                               │
│ ATTENTION  task_attention()                                                   │
│   [!!!] SCALAR DOT PRODUCTS — no tensor cores                                 │
│   [!!!] FULL SCORE MATRIX IN SMEM — O(seq_k)                                  │
│   [!!]  Serial over query positions                                           │
│   [!!]  No K-dimension tiling (not FlashAttention)                            │
│                                                                               │
│ REDUCE  task_reduce()                                                         │
│   [!!] ELEMENT-BY-ELEMENT FP16 LOADS — no float4                              │
│   [!!] RMSNorm: 2 global memory passes                                        │
│   [!!] LayerNorm: 3 global memory passes                                      │
│                                                                               │
│ ROPE  task_rope()                                                             │
│   [!!] ELEMENT-BY-ELEMENT — no vectorization                                  │
│   [!!] Serial head loop, cos/sin re-read per head                             │
│                                                                               │
│ INDEX  task_index()                                                           │
│   [!!] NO VECTORIZATION — scalar element access                               │
│                                                                               │
│ ELEMENTWISE — vectorized (float4) — OK                                        │
│ FUSED_ELEMENTWISE — uop interpreter — OK                                      │
│   [!] Max 8 uops, 8 registers (caps fusion depth)                             │
│ EMBEDDING — vectorized when dim%8==0 — OK                                     │
│ COPY — vectorized flat path — OK                                              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────────────────────┐
│ BOTTLENECK SEVERITY LEGEND                                                    │
│                                                                               │
│ [!!!] Critical    >10% of runtime or >3x gap vs opt                           │
│ [!!]  Significant 2-10% of runtime or 2-3x gap                                │
│ [!]   Minor       <2% impact                                                  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 0b. Proposed Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ COMPILE — HIGH-LEVEL IR (Graph IR)                                            │
│                                                                               │
│ torch.nn.Module                                                               │
│   > torch.export.export(strict=False)                                         │
│   > run_decompositions(core_aten)                                             │
│   │                                                                           │
│   ▼                                                                           │
│ [1] Pattern Matching + Op Recognition                                         │
│     RMSNorm, LayerNorm, RoPE, GeGLU, SwiGLU, GQA                              │
│     Zero-cost shape ops via StridedView                                       │
│   │                                                                           │
│   ▼                                                                           │
│ [2] Graph Optimization                                                        │
│     ├─ Constant folding                                                       │
│     ├─ Dead code elimination                                                  │
│     ├─ Common subexpression elimination  (NEW)                                │
│     └─ Identity op elimination                                                │
│   │                                                                           │
│   ▼                                                                           │
│ [3] Cost-Model-Driven Fusion Planning                                         │
│     ├─ Matmul epilogue: bias+act+residual  (NEW)                              │
│     ├─ Elementwise chain > FUSED_ELEMENTWISE                                  │
│     ├─ Single-pass norm+scale  (NEW)                                          │
│     └─ Fuse IFF cost_fused < cost_split                                       │
│   │                                                                           │
│   ▼                                                                           │
│ [4] CUTLASS Autotuning  (NEW)                                                 │
│     For each matmul shape (M, N, K):                                          │
│     ├─ Tiles: 64x64, 64x128, 128x128, 128x256                                 │
│     ├─ Score = SM_util x wave_eff x occupancy                                 │
│     ├─ Split-K for bandwidth-bound (M<=4)                                     │
│     ├─ Best config encoded in TaskDesc                                        │
│     └─ Cached per (M, N, K, sm_version)                                       │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ COMPILE — LOW-LEVEL IR (Schedule IR)                                          │
│                                                                               │
│ [5] Memory Planning                                                           │
│     ├─ DAG-based liveness analysis  (NEW)                                     │
│     ├─ Layout propagation: col-major for B  (NEW)                             │
│     ├─ Weight pre-transposition for decode  (NEW)                             │
│     └─ Arena allocation (128-byte alignment)                                  │
│   │                                                                           │
│   ▼                                                                           │
│ [6] Scheduling                                                                │
│     ├─ Build dependency DAG  (NEW)                                            │
│     ├─ Topological sort, critical-path priority                               │
│     ├─ Compute dep_counts[] + successor_lists[]                               │
│     └─ Upload DAG metadata for device scheduler                               │
│   │                                                                           │
│   ▼                                                                           │
│ [7] Serialization                                                             │
│     Enhanced TaskDesc with:                                                   │
│     ├─ tile_config (which CUTLASS variant)                                    │
│     ├─ epilogue flags (bias, act, residual)                                   │
│     └─ Dependency edges for event-driven sched                                │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ CUDA COMPILATION (multi-config)                                               │
│                                                                               │
│ CUTLASS GEMM — multiple tile configs compiled:                                │
│   ├─ gemm_64x64_bk32   small shapes, high SM util                             │
│   ├─ gemm_64x128_bk32  rectangular (MLP N=11008)                              │
│   ├─ gemm_128x128_bk64 large shapes (current)                                 │
│   ├─ gemm_128x256_bk64 very large N                                           │
│   └─ gemm_splitk       M<=4 decode, K-split                                   │
│   dispatch_matmul() reads config from TaskDesc                                │
│                                                                               │
│ FlashAttention kernel  (NEW):                                                 │
│   ├─ K-tiled (block_kv = 64 or 128)                                           │
│   ├─ Tensor cores: WGMMA (SM90) / mma.sync (SM80)                             │
│   ├─ Online softmax (no materialized scores)                                  │
│   └─ O(1) SMEM per head (vs O(seq_k) current)                                 │
│                                                                               │
│ Improved task kernels:                                                        │
│   ├─ reduce.cu: float4 loads + single-pass norms                              │
│   ├─ rope.cu:   float4 loads + SMEM cos/sin cache                             │
│   └─ index.cu:  float4 vectorized                                             │
│                                                                               │
│ nvcc -cubin --use_fast_math -dlto                                             │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ RUNTIME — EVENT-DRIVEN SCHEDULER                                              │
│                                                                               │
│ Host:                                                                         │
│   ├─ Same arena allocation + weight caching                                   │
│   ├─ Weight pre-transposition (decode, cached)                                │
│   ├─ Upload TaskDescs + dependency DAG                                        │
│   └─ cuLaunchCooperativeKernel (unchanged)                                    │
│                                                                               │
│ Device:                                                                       │
│   while (task_id = ready_queue.pop()) {                                       │
│       if (blockIdx.x < tasks[task_id].num_tiles)                              │
│           dispatch_task(tasks[task_id]);                                      │
│       tile_barrier(num_tiles);  // participating                              │
│       if (tid==0 && bid==0)                                                   │
│           signal_dependents(atomicSub);                                       │
│   }                                                                           │
│                                                                               │
│   Benefits vs BSP:                                                            │
│   ├─ No idle SMs (pick up next ready task)                                    │
│   ├─ Independent tasks run in parallel                                        │
│   ├─ No global barrier for unrelated tasks                                    │
│   └─ Straggler tolerance                                                      │
│                                                                               │
│   dispatch_task() switch(op_type):                                            │
│   MATMUL    > CUTLASS variant by config_id                                    │
│               fused epilogue: bias+act+residual                               │
│               split-K for M<=4                                                │
│   ATTENTION > FlashAttention with tensor cores                                │
│   REDUCE    > float4 + single-pass RMSNorm                                    │
│   ROPE      > float4 + SMEM cos/sin cache                                     │
│   (others)  > float4 vectorized                                               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │                                        
                                        ▼                                        
┌───────────────────────────────────────────────────────────────────────────────┐
│ FUTURE: HYBRID PATH (prefill / large batch)                                   │
│                                                                               │
│ CUDA Graph wrapping cuBLAS + megakernel nodes:                                │
│   cuBLAS > megakernel(norm+act) > cuBLAS > ...                                │
│   cuGraphLaunch(graph, stream) — zero overhead                                │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 0c. Why You Can't Use cuBLAS Inside a Megakernel (And What To Do Instead)

### The Obvious Question

"Why not call cuBLAS/cuDNN from inside the megakernel and get the best of both worlds?"

### Why It's Impossible

1. **cuBLAS is a host-side API.** `cublasGemmEx()` enqueues a kernel onto a CUDA stream from host code. Cannot be called from `__device__` code.

2. **CUDA Dynamic Parallelism (CDP) doesn't help.** CDP lets device code launch child kernels, but:
   - Child kernels can't use WGMMA (no tensor core access from CDP)
   - CDP has ~50 us overhead per child launch (worse than host launches)
   - Cooperative kernels (required for grid.sync) forbid CDP entirely

3. **cuBLAS is closed-source.** No stable device-side ABI. Kernel names/signatures change between CUDA versions. Cannot link cuBLAS kernels into a custom binary.

### What We CAN Do: Use CUTLASS Better

cuBLAS's advantage is NOT magic hardware access. It's:
- Autotuned tile sizes per shape
- WGMMA/HMMA instruction selection (**we already have this via CuTe**)
- Software pipelining with optimal stage counts (**we already have this**)
- Split-K and Stream-K for load balancing (**missing**)
- Epilogue fusion in registers (**missing**)
- Multiple specialized kernel variants (**missing — we have one**)

**CUTLASS is open-source and already linked into megabake.** The CuTe GEMM in `matmul.cu` uses CUTLASS types (`SM90_64x128x16_F32F16F16_SS`, `SM80_16x8x16_F32F16F16F32_TN`). These use the SAME tensor core instructions as cuBLAS.

The gap is in TUNING, not in hardware access:

| What cuBLAS has | What megabake has | Gap |
|----------------|------------------|-----|
| ~50 tile configs per arch | 1 fixed (128×128) | 50x fewer configs |
| Split-K for bandwidth-bound | N-split only | Missing entirely |
| Stream-K for load balancing | None | Missing entirely |
| Bias+act+residual epilogue | Activation only (SiLU/GELU) | Partial |
| Per-shape autotuning DB | No tuning | Missing entirely |
| CUTLASS 3.x persistent kernels | Single-tile dispatch | Structural gap |

### The Strategy: CUTLASS Multi-Config Autotuning

**Phase 1 (Weeks 1-2): Compile multiple tile configs**
```
Compile 5 CUTLASS GEMM variants into the megakernel binary:
  gemm_64x64    → small shapes, maximize SM utilization
  gemm_64x128   → rectangular (common in MLP: M small, N=11008)
  gemm_128x128  → current default, large shapes
  gemm_128x256  → very large N, fewer tiles needed
  gemm_splitk   → M<=4 decode, split K across all SMs
```

Each is a template instantiation of the same CuTe pipeline with different `BM, BN, BK, STAGES` constants. Same WGMMA/mma.sync instructions. ~2-3 KB extra cubin per variant.

**Phase 2 (Week 3): Compile-time tile selection**
```python
def select_tile_config(M, N, K, num_sms):
    best_score = 0
    for (TM, TN, BK, STAGES) in TILE_CONFIGS:
        tiles = ceil(M/TM) * ceil(N/TN)
        util = min(tiles, num_sms) / num_sms
        waves = ceil(tiles / num_sms)
        efficiency = tiles / (waves * num_sms)
        smem = (TM*BK + TN*BK) * STAGES * 2
        score = util * efficiency * (1.0 if smem <= MAX_SMEM else 0.5)
        if score > best_score:
            best_score = score
            best_config = config_id
    return best_config
```

Store `config_id` in TaskDesc. `dispatch_matmul()` reads it and calls the right variant.

**Phase 3 (Weeks 4-6): CUTLASS epilogue fusion**
```cuda
// In matmul epilogue, before writing to global memory:
if (has_bias)    acc += __half2float(bias[col]);
if (has_silu)    acc = acc * sigmoidf(acc);
if (has_residual) acc += __half2float(residual[row * N + col]);
out[row * N + col] = __float2half(acc);
```

This eliminates the separate ELEMENTWISE_ADD task + barrier per biased linear.

**Expected result:** GEMM within 10-15% of cuBLAS on most shapes, within 5% on decode shapes. Combined with launch overhead elimination, this makes the pure megakernel approach viable.

### Comparison of Strategies

| Strategy | GEMM quality | Launch overhead | Complexity | Timeline |
|----------|-------------|-----------------|------------|----------|
| Current (1 CuTe config) | 0.2-0.3x cuBLAS | 0 (1 kernel) | Low | Done |
| CUTLASS multi-config (recommended) | 0.85-0.95x cuBLAS | 0 (1 kernel) | Medium | 4-6 weeks |
| Hybrid cuBLAS + megakernel | 1.0x cuBLAS | ~0 (CUDA Graphs) | High | 6-8 weeks |
| cuBLAS from device code | Impossible | N/A | N/A | N/A |

**Recommendation:** CUTLASS multi-config first. It keeps the pure megakernel architecture and closes most of the GEMM gap. Fall back to hybrid only if the remaining 5-15% gap matters for target workloads.

---

## 1. Architecture Critique

### What Megabake Is

A cooperative megakernel compiler: torch.export FX graph → flat `TaskDesc` array → single `cuLaunchCooperativeKernel` that loops over tasks with `grid.sync()` barriers. One block per SM, 256 threads per block, BSP execution.

### Fundamental Design Errors

**Error 1: Writing your own GEMM inside a persistent kernel.**
Matmul is 80-95% of transformer inference time. cuBLAS has hundreds of person-years of autotuning per shape. Megabake has ONE CuTe GEMM with fixed 128x128 tiles.

Proof from your own data:
```
linear_256x512:  megabake 237 us vs torch.compile 65 us → 0.27x
```
That's a single matmul. No fusion to save you. Your GEMM is 3.6x slower than cuBLAS on a basic shape. Every transformer layer pays this tax 7 times (Q/K/V/O projections + gate/up/down MLP).

**Error 2: BSP execution serializes everything.**
Grid.sync() after every task. ~100 tasks per SmolLM2-135M forward pass × 2.5 us = 250 us pure barrier cost. But worse: tasks with few tiles leave most SMs idle, and those idle SMs still hit the barrier.

For a 256-element elementwise op (1 tile needed), 31 of 32 SMs are idle. The wall-clock cost is the SAME as if all 32 SMs were working.

**Error 3: Attention kernel has no tensor cores and no flash-style tiling.**
Q*K^T and Attn*V are dense matrix multiplies. Running them with scalar FMA in fp32 leaves 99% of H200 compute dark. FlashAttention uses tensor cores for both and tiles in the sequence dimension. Your implementation:
- Materializes full score matrix in shared memory: O(seq_k) SMEM
- Serial loop over query positions
- Scalar dot products (no HMMA/WGMMA)

This is why gemma-2b is 0.49x. Longer sequences amplify the gap.

**Error 4: No vectorization in most kernels.**
`reduce.cu`: element-by-element fp16 loads. 256 threads × 1 half per load = 512 bytes/cycle. With float4 vectorization: 256 × 16 bytes = 4096 bytes/cycle. 8x bandwidth difference.

`rope.cu`: same problem. Element-by-element.

`index.cu`: same.

**Error 5: Redundant global memory passes.**
RMSNorm reads input 2x (sum-of-squares pass, then normalize pass). LayerNorm reads 3x. Each pass is a full global memory traversal. A fused single-pass with online statistics halves the bandwidth.

**Error 6: The uber-kernel design causes register pressure and I-cache pollution.**
All 11 task implementations (576-line matmul, 186-line attention, etc.) are compiled into ONE binary. `SKINNY_MAX_M = 64` declares 64 float accumulators per thread even when M=1. The compiler can't fully optimize away dead code across the dispatch switch. SM90 matmul alone uses 96KB shared memory.

---

## 2. Fundamental Performance Bottlenecks (Ranked)

| # | Bottleneck | Impact | Evidence |
|---|-----------|--------|----------|
| 1 | GEMM quality gap vs cuBLAS | 3-5x per matmul | linear benchmarks: 0.19-0.30x |
| 2 | Attention: no tensor cores, no flash tiling | 2-5x on attention | gemma-2b: 0.49x overall |
| 3 | SM underutilization (tiling) | 50-97% idle SMs on small tasks | PLAN.md analysis |
| 4 | BSP barrier overhead | ~250-500 us per forward | ~100 barriers × 2.5 us |
| 5 | No vectorization in reduce/rope/index | 2-8x bandwidth waste | Element-by-element loads |
| 6 | Multi-pass reductions | 2-3x bandwidth for norms | RMSNorm: 2 passes, LayerNorm: 3 |
| 7 | Matmul epilogue not fused (bias, residual add) | ~7.5 us × N biased linears | Extra task + barrier per bias |

---

## 3. Theoretical Performance Analysis

### H200 MIG 2g.35gb (32 SMs) Hardware Limits

| Resource | Value |
|----------|-------|
| Peak FP16 tensor core TFLOPS | ~150 TFLOPS (32/132 of full H200) |
| HBM3e bandwidth | ~387-672 GB/s measured |
| L2 cache | ~12.5 MB (32/132 of 50 MB) |
| SMEM per SM | 228 KB |
| Registers per SM | 65536 × 32-bit |
| Cooperative launch overhead | ~48 us |
| grid.sync() cost | ~2.5 us |

### Batch-1 Decode Roofline

For SmolLM2-135M (135M params = 270 MB in FP16):
```
Minimum time = model_bytes / bandwidth
             = 270 MB / 500 GB/s
             = 540 us

Current megabake:       8844 us  (16.4x above floor)
Current torch.compile:  7802 us  (14.4x above floor)
Theoretical target:     ~800-1200 us (1.5-2.2x above floor, accounting for compute)
```

Where does the 16.4x overhead come from?

| Source | Estimated us | % |
|--------|-------------|---|
| Weight reads (bandwidth-limited) | ~540 | 6.1% |
| Matmul compute overhead vs optimal | ~3000 | 33.9% |
| Attention compute | ~1500 | 17.0% |
| BSP barriers (~100) | ~250 | 2.8% |
| SM idle time (underutilization) | ~2000 | 22.6% |
| Activation memory traffic (unneeded HBM round-trips) | ~200 | 2.3% |
| Elementwise/reduce compute | ~800 | 9.0% |
| Kernel launch + scheduler | ~554 | 6.3% |

The matmul quality gap and SM underutilization account for ~57% of total runtime.

### Where 2x Over torch.compile Is Possible

**Possible (batch-1, seq=1 decode, small-medium models):**
- Eliminate launch overhead: torch.compile launches 433 kernels × ~5 us = ~2165 us overhead (28% of its 7802 us)
- Without CUDA Graphs, megabake's 1-launch advantage is worth ~1867 us
- If GEMM quality matches cuBLAS, megabake wins by eliminating launches + L2 reuse
- Theoretical: ~1.5-2x over torch.compile WITHOUT CUDA Graphs

**Impossible (any of these):**
- Large batch (>=32): matmul is compute-bound, cuBLAS wins, fusion savings negligible
- Long sequences (>=2048): FlashAttention vs naive attention is an unbridgeable gap
- Against torch.compile + CUDA Graphs: CUDA Graphs eliminate launch overhead, leaving only the GEMM quality gap (which favors cuBLAS)
- Against TensorRT: fully optimized fused kernels with cuBLAS-grade GEMM

**The honest assessment:** 2x over torch.compile without CUDA Graphs is achievable on batch-1 decode with competitive GEMM. 2x over torch.compile WITH CUDA Graphs requires innovations beyond what any existing system has demonstrated.

---

## 4. Hardware Bottleneck Analysis

### Decode (M=1): Memory-Bandwidth Bound

```
Matmul M=1, N=4096, K=4096:
  FLOPS = 2 × 1 × 4096 × 4096 = 33.5 MFLOP
  Bytes = 4096 × 4096 × 2 = 33.5 MB (weight matrix)
  Arithmetic intensity = 1 FLOP/byte
  At 500 GB/s: minimum 67 us
  Tensor core throughput: 33.5 MFLOP / 150 TFLOP/s = 0.00022 us
  → Pure memory bound. Tensor cores idle 99.99% of time.
```

Implication: for decode, GEMM quality doesn't matter IF you can saturate memory bandwidth. cuBLAS saturates it because it parallelizes across N columns. Megabake's skinny matmul does this too (uses all SMs via N-splitting). The gap must be in load efficiency.

### Prefill (M=512+): Compute-Bound

```
Matmul M=512, N=4096, K=4096:
  FLOPS = 2 × 512 × 4096 × 4096 = 17.2 GFLOP
  Bytes = 33.5 MB (weights) + 512 × 4096 × 2 (A) + 512 × 4096 × 2 (C) ≈ 41.5 MB
  Arithmetic intensity = 414 FLOP/byte
  At 150 TFLOP/s: 115 us
  At 500 GB/s: 83 us
  → Compute bound. Tensor core efficiency matters.
```

### L2 Cache Budget

L2 ≈ 12.5 MB. For decode intermediates:
```
SmolLM2-135M hidden=576:
  Activation buffer: 576 × 2 = 1.15 KB
  MLP intermediate: 1536 × 2 = 3.07 KB
  → ALL intermediates fit in L2. Megakernel L2 reuse advantage: ~1 us total.
```

For larger models (LLaMA-7B hidden=4096):
```
  Activation: 4096 × 2 = 8 KB
  MLP intermediate: 11008 × 2 = 22 KB
  → Still trivially fits L2. Megakernel advantage remains minimal for batch-1.
```

L2 reuse matters more for batch>1 where intermediates are `batch × hidden × 2` bytes. At batch=64, hidden=4096: 512 KB per intermediate, ~6 intermediates per layer = 3 MB, still fits L2.

**Bottom line:** For decode, L2 reuse from megakernels is worth almost nothing. The advantage is purely in launch overhead elimination.

---

## 5. Missing Compiler Optimizations

### Critical (directly cause benchmark losses)

1. **Single CUTLASS tile config (128×128).** Megabake already uses CuTe/CUTLASS with the same tensor core instructions as cuBLAS (WGMMA on SM90, mma.sync on SM80). The gap is tuning: cuBLAS has ~50 tile configs per arch, megabake has 1. Calling cuBLAS from device code is impossible (host-side API, CDP forbidden in cooperative kernels, closed-source). The fix is CUTLASS multi-config autotuning: compile 5 tile variants, select best per shape at compile time. See Section 0c for full analysis.

2. **No matmul bias fusion.** Every biased linear emits MATMUL + ELEMENTWISE_ADD + 2 barriers. cuBLAS epilogue handles this in-register.

3. **No matmul residual-add fusion.** `y = linear(x) + x` could fuse the add into matmul epilogue.

4. **No cross-task register/SMEM forwarding.** If RMSNorm output feeds matmul input, the data goes: RMSNorm registers → global memory → matmul shared memory. Could go: RMSNorm registers → shared memory → matmul directly.

5. **No persistent-thread reduction.** RMSNorm/LayerNorm read input 2-3x from global memory. Single-pass with online Welford algorithm reads once.

6. **No split-K for matmul.** When M is small but K is large, splitting K across SMs and reducing is faster than the current N-splitting approach. cuBLAS does this.

### Important (would help but not critical)

7. **No vectorized loads in reduce/rope/index.** Easy 2-4x bandwidth improvement on these kernels.

8. **No operator reordering.** Tasks execute in graph order. Reordering to improve L2 locality (e.g., keep Q*K^T close to softmax) is possible within dependency constraints.

9. **No recomputation.** Sometimes recomputing a value is cheaper than storing it (saves memory bandwidth). Particularly useful for activations in deep networks.

10. **No layout propagation.** All tensors assumed row-major. Column-major for matmul B operand would eliminate transposes.

---

## 6. Proposed New Architecture

### Core Principle: Hybrid Megakernel

Don't put matmul inside the persistent kernel. Instead:

```
Phase A: Large matmuls → cuBLAS (separate kernels or CUDA Graphs)
Phase B: Everything between matmuls → megakernel (fused norm+activation+residual)
```

This keeps cuBLAS quality for the 80-95% of compute that is matmul, while megakernel handles the memory-bound "glue" operations.

### Alternative: Pure Megakernel (if cuBLAS quality is achievable)

If GEMM quality can match cuBLAS within 5%, the pure megakernel is strictly better because it eliminates ALL launch overhead and enables cross-task fusion.

**Recommendation: pursue both in parallel.** Hybrid is the safe path. Pure megakernel is the moonshot that could hit 2x.

### IR Design

Two-level IR:

**High-Level IR (Graph IR):**
```
- Nodes: op_type, input edges, output edge, shape, dtype, layout
- Edges: buffer_id, shape, strides, liveness interval
- Graph-level metadata: dependency DAG, critical path, memory plan
```

**Low-Level IR (Schedule IR):**
```
- Fused regions: sets of ops that execute in one kernel tile
- Per-region: thread mapping, register allocation, shared memory layout
- Cross-region: dependency edges, synchronization points
```

The current flat `TaskDesc` array conflates both levels. Separating them enables:
- Graph-level optimization (fusion decisions, reordering, recomputation)
- Schedule-level optimization (tiling, register allocation, vectorization)

### Compilation Pipeline

```
torch.export FX graph
    │
    ▼
[1] Decompose + Pattern Match
    Detect: RMSNorm, LayerNorm, RoPE, GeGLU, SwiGLU, MHA/GQA
    Preserve detected patterns as single high-level ops
    │
    ▼
[2] Graph Optimization
    - Constant folding
    - Dead code elimination
    - Common subexpression elimination
    - Identity op elimination (mul by 1, add 0, cast to same dtype)
    │
    ▼
[3] Fusion Planning (cost-model driven)
    - Matmul epilogue fusion (bias, activation, residual add)
    - Elementwise chain fusion
    - Reduction + elementwise fusion (norm + scale)
    - Producer-consumer fusion (when intermediate fits in SMEM/registers)
    Decision: fuse vs. split based on cost model, not fixed rules
    │
    ▼
[4] Memory Planning
    - Liveness analysis
    - Arena allocation with alignment
    - Layout propagation (row-major vs. column-major)
    - Recomputation decisions (recompute vs. store)
    │
    ▼
[5] Scheduling
    - Topological sort with memory-locality heuristic
    - Tile assignment per fused region
    - Synchronization insertion (minimize barriers)
    │
    ▼
[6] Code Generation
    Option A (near-term): Enhanced TaskDesc with fused epilogues
    Option B (long-term): PTX/SASS code generation for fused regions
    │
    ▼
[7] Runtime
    - Cooperative kernel launch
    - Event-driven scheduler (replace BSP)
    - Autotuned tile sizes
```

### Key Differences from Current Architecture

| Aspect | Current | Proposed |
|--------|---------|----------|
| Matmul | Custom CuTe, fixed 128×128 | CUTLASS multi-config (5 tile variants, compile-time autotuned per shape) |
| Attention | Naive materialized scores | FlashAttention with tensor cores |
| Scheduling | BSP grid.sync() every task | Event-driven with dependency DAG |
| Fusion | 3 fixed peephole passes | Cost-model-driven fusion planner |
| Reductions | Multi-pass global reads | Single-pass online algorithms |
| Vectorization | Matmul/elementwise only | All kernels vectorized |
| IR | Flat TaskDesc array | Two-level Graph IR + Schedule IR |
| Code gen | Interpreted uber-kernel | JIT per fused region (long-term) |

---

## 7. Cost Model

### Fuse vs. Split Decision

```
cost_fused = max(compute_time_fused, memory_time_fused) + register_pressure_penalty
cost_split = cost_task_A + cost_task_B + barrier_cost + intermediate_memory_traffic

fuse when: cost_fused < cost_split
```

**Intermediate memory traffic:**
```
intermediate_bytes = numel × dtype_size
if intermediate_bytes <= L2_capacity:
    traffic_cost = intermediate_bytes / L2_bandwidth  (~3-5 TB/s)
else:
    traffic_cost = intermediate_bytes / HBM_bandwidth  (~500 GB/s)
```

For decode (hidden=4096, batch=1): intermediate = 8 KB → always in L2 → traffic_cost ≈ 0.002 us. Fusion saves almost nothing for decode intermediates.

**Barrier cost:** ~2.5 us per grid.sync(). This is the dominant savings from fusion.

**Register pressure penalty:**
```
regs_per_thread = fused_region_register_demand
if regs_per_thread > 128:
    penalty = occupancy_loss × compute_time  (significant)
if regs_per_thread > 255:
    penalty = INFINITY  (cannot compile)
```

### Recomputation vs. Storage

```
cost_store = write_bytes / bandwidth + read_bytes / bandwidth
cost_recompute = compute_flops / throughput

recompute when: cost_recompute < cost_store AND recomputation doesn't extend critical path
```

For activations (SiLU, GELU): ~5 FLOPS per element, ~2-4 bytes per element. At 150 TFLOPS/s and 500 GB/s:
```
cost_store = 4B / 500 GB/s = 0.008 ns/element
cost_recompute = 5 / 150 TFLOP/s = 0.033 ns/element
```
Storage wins for inference (no backward pass). Don't recompute.

### Tile Size Selection

```
For matmul (M, N, K):
  tile_candidates = [(64,64), (64,128), (128,64), (128,128), (128,256), (256,128)]
  for each (TM, TN):
    num_tiles = ceil(M/TM) × ceil(N/TN)
    sm_utilization = min(num_tiles, num_sms) / num_sms
    waves = ceil(num_tiles / num_sms)
    wave_efficiency = num_tiles / (waves × num_sms)
    smem_usage = (TM × BK + TN × BK) × stages × 2  # bytes
    occupancy = f(smem_usage, register_count)
    
    score = sm_utilization × wave_efficiency × occupancy
  pick tile with best score
```

Current fixed 128x128 gives wave_efficiency of:
```
M=256, N=512: 2 × 4 = 8 tiles / 32 SMs = 25% utilization
M=1, N=4096: skinny path uses all 32 SMs = 100% utilization (good)
M=128, N=11008: 1 × 86 = 86 tiles / 32 SMs = 2.69 waves, 84% efficiency
```

---

## 8. Fusion Strategy

### Tier 1: Matmul Epilogue Fusion (highest ROI)

Fuse into matmul writeback:
- Bias add: `y = Wx + b` (save 1 task + 1 barrier)
- Activation: `y = SiLU(Wx)` (already done for some)
- Residual add: `y = Wx + residual` (save 1 task + 1 barrier)
- Combined: `y = SiLU(Wx + b) + residual` (save 3 tasks + 3 barriers)

Implementation: add epilogue lambda to CuTe GEMM. Accumulator stays in registers, apply bias/activation/residual before writing to global memory.

Expected savings per transformer layer: 3-6 eliminated tasks × 7.5 us = 22-45 us.

### Tier 2: Norm + Scale Fusion

Fuse RMSNorm into single pass:
```
// Current: pass1 reads X for sum_sq, pass2 reads X again for normalize
// Proposed: single pass, accumulate sum_sq with warp shuffles,
//           then normalize in same pass
// Requires the row to fit in register file or SMEM
```

For hidden=4096: 4096 × 2 = 8 KB per row. With 256 threads: 16 elements per thread = 32 bytes = 8 registers. Feasible to cache entire row in registers, do single-pass norm.

Expected savings: 50% bandwidth reduction on every norm operation.

### Tier 3: Elementwise Chain Fusion (already implemented)

Current implementation is reasonable. The micro-op interpreter avoids code generation complexity. Cap of 8 uops per chain is sufficient for most patterns.

Improvement: raise to 16 uops, add float constant loads as uops instead of encoding in dimensions.

### Tier 4: Producer-Consumer SMEM Handoff (long-term)

When op A's output is op B's input and both run on the same SMs:
```
Op A writes to shared memory instead of global
grid.sync() or tile-level signal
Op B reads from shared memory
```

This only helps when:
- Both ops use the same number of tiles (or multiples)
- Intermediate is small enough for SMEM (~100 KB per SM)
- Both ops are memory-bound (compute-bound ops don't benefit)

For decode: intermediates are 1-8 KB, L2 latency is ~0.002 us. SMEM handoff saves ~0.001 us. Not worth the complexity.

For prefill (batch=64, hidden=4096): intermediates are 512 KB, exceeds SMEM. Not feasible.

**Verdict: SMEM handoff is not worthwhile for transformer inference.** L2 handles it.

---

## 9. Megakernel Strategy

### When Megakernels Win

Quantitative model:
```
megakernel_time = Σ task_time_i + N_barriers × barrier_cost + launch_overhead
separate_time  = Σ (task_time_i + launch_overhead_i) + intermediate_traffic
```

Megakernel wins when:
```
N_tasks × launch_overhead > N_barriers × barrier_cost + cooperative_launch_overhead
```

Plugging in H200 MIG numbers:
```
N × 5 us > N × 2.5 us + 48 us
2.5N > 48
N > 19.2
```

With >=20 tasks, megakernel saves on launch overhead. Transformer layers have ~10-15 tasks each, so >=2 layers makes megakernel worthwhile.

### When Megakernels Lose

1. **GEMM quality gap exceeds launch savings:**
   ```
   If custom GEMM is 2x slower than cuBLAS:
   Per matmul penalty = cuBLAS_time × 1.0 (100% overhead)
   With 7 matmuls/layer × 16 layers = 112 matmuls
   Need: 112 × cuBLAS_time < launch_savings
   ```
   This is why GEMM quality is the make-or-break issue.

2. **Register explosion:** The uber-kernel compiles all task types into one binary. NVCC must allocate registers for the worst case. `__launch_bounds__(256, 1)` helps (tells compiler max 1 block per SM), but the matmul's 96 KB SMEM still limits occupancy to 1 block regardless.

3. **Instruction cache pressure:** Combined kernel binary is large. I-cache on H200 is 64 KB per SM. If the hot path (matmul inner loop) doesn't fit in I-cache because attention code is also loaded, performance suffers.

### Optimal Megakernel Size

Based on the cost model:
```
Optimal = all tasks between two large matmuls

Pattern:  [cuBLAS GEMM] → [megakernel: norm + activation + residual + small ops] → [cuBLAS GEMM]
```

This is the hybrid approach: large matmuls as separate kernels, everything else fused into megakernels.

For pure megakernel approach: entire model. The cost is GEMM quality; the benefit is zero inter-kernel overhead.

### Partitioning Heuristic

```
partition_score(region) = 
    barrier_savings × 2.5us
    + launch_savings × 5us  
    - gemm_quality_penalty
    - register_pressure_penalty
    - icache_pressure_penalty

Split region when partition_score < 0
```

---

## 10. Memory Planning Strategy

### Current: First-Fit-Decreasing Arena

Adequate. O(n^2) in buffer count, but buffer count is typically <200. Not a bottleneck.

### Improvements

1. **Layout propagation:** Matmul B expects column-major (transposed). Currently, transpose is encoded as `strides[0]=1` flag. Better: propagate layout preferences backward from consumers. If matmul B wants column-major, store the preceding op's output as column-major. Eliminates implicit transposes.

2. **Alignment to 128 bytes** (not current 256): 128 bytes = one cache line = one vectorized load. 256 is over-aligned.

3. **Weight packing:** Weights are currently stored as-is from the state dict. For decode M=1 matmuls, weight access pattern is column-by-column. Row-major storage means non-coalesced reads. Pre-transposing weights to column-major would improve memory bandwidth.

4. **Buffer aliasing across independent ops:** Current liveness analysis is task-index-based. With an event-driven scheduler, liveness becomes DAG-based. Two independent branches can share buffers if they don't execute simultaneously — but this requires knowing the schedule at compile time.

---

## 11. Scheduling Strategy

### Replace BSP with Event-Driven

Current BSP:
```cuda
for task in tasks:
    if blockIdx.x < task.num_tiles:
        dispatch(task)
    grid.sync()  // ALL SMs barrier, even idle ones
```

Proposed event-driven:
```cuda
while (task = ready_queue.pop()):
    execute(task)
    for downstream in task.dependents:
        if atomic_decrement(downstream.remaining_deps) == 0:
            ready_queue.push(downstream)
```

**Expected savings:**
- Eliminate barriers for independent tasks (Q/K/V projections can run in parallel)
- No idle SM overhead (SMs that finish early pick up next ready task)
- Straggler tolerance

**Costs:**
- Atomic contention on ready queue (~0.1-0.5 us per pop)
- Non-determinism (harder to profile)
- Register overhead for scheduler state

**Net estimate for SmolLM2-135M:** Save 500-1000 us (per PLAN.md analysis). Brings megabake from 8844 us to ~7800-8300 us.

### Task Ordering Heuristic

Within the ready set, prioritize:
1. Tasks on the critical path (longest remaining chain)
2. Tasks whose outputs are consumed by the most dependents
3. Tasks with the most tiles (maximize SM utilization)

---

## 12. Runtime Architecture

### Near-Term (keep cooperative kernel)

```
Host:
  1. Parse schedule, upload TaskDescs to GPU
  2. Compute dependency DAG edges, upload dep_counts[] and successor_lists[]
  3. Allocate arena + pointer array
  4. Copy weights (first call only, cached)
  5. Copy inputs to arena (fast path: fp16+contiguous+CUDA → write data_ptr directly)
  6. cuLaunchCooperativeKernel(megakernel, num_sms, 256, args, smem, stream)
  7. Read output from arena, clone

Device:
  Event-driven loop:
    Pop task from ready queue
    Dispatch based on op_type
    Signal dependents via atomics
```

### Long-Term (CUDA Graphs integration)

```
For hybrid architecture:
  Build CUDA Graph:
    cuBLAS node → megakernel node → cuBLAS node → megakernel node → ...
  
  cuGraphLaunch(graph, stream)
```

This eliminates ALL launch overhead while using cuBLAS for matmuls.

### Memory Management

Keep the single-arena approach. It works well:
- One `cudaMalloc` for all workspace
- Liveness-based buffer sharing
- Weight tensors separate (pinned, cached)

Add: pre-transposed weight format for decode (column-major for N-major access).

---

## 13. Benchmarking Methodology

### Current Gaps

1. **No CUDA Graphs baseline.** torch.compile + CUDA Graphs is the real competition.
2. **No per-task profiling.** Can't identify which tasks are slow.
3. **No roofline analysis.** Don't know if kernels are compute or memory bound.
4. **No nsight profiling integration.** Manual CUDA event timing only.
5. **Benchmark shapes don't match production.** Tests use tiny configs (hidden=64, heads=2).

### Proposed Benchmark Suite

```
Tier 1: Microbenchmarks (per-kernel)
  - matmul: M={1,4,16,128,512}, N={1024,4096,11008}, K={1024,4096}
    Compare: megabake vs cuBLAS vs Triton
  - attention: H={8,32}, S_q={1,32,128}, S_k={128,2048,8192}, D={64,128}
    Compare: megabake vs FlashAttention vs cuDNN
  - reduce/norm: hidden={768,1024,4096,8192}
    Compare: megabake vs Triton vs apex

Tier 2: Layer benchmarks
  - Full transformer layer, realistic configs
  - Compare: megabake vs torch.compile vs torch.compile+CUDA Graphs

Tier 3: Model benchmarks
  - SmolLM2-135M, LLaMA-3.2-1B, LLaMA-3.1-8B, Gemma-2B
  - Batch sizes: 1, 4, 16, 64
  - Seq lengths: 1 (decode), 32, 128, 512, 2048
  - Compare: all backends

Tier 4: Production benchmarks
  - Tokens/second throughput
  - Time-to-first-token
  - Memory footprint
```

### Per-Kernel Profiling

Add optional timing mode:
```cuda
// In megakernel dispatch loop:
if (ENABLE_PROFILING && threadIdx.x == 0 && blockIdx.x == 0) {
    clock_t start = clock64();
    dispatch_task(task, ...);
    clock_t end = clock64();
    task_timings[i] = end - start;
}
```

This identifies which tasks are bottlenecks without external profiler overhead.

---

## 14. Validation Plan

### For each proposed optimization:

**Optimization 1: Matmul epilogue fusion (bias)**
- Bottleneck: extra ELEMENTWISE_ADD + barrier per biased linear
- Hypothesis: fusing bias into matmul epilogue eliminates 1 task + 1 barrier
- Expected effect: save ~7.5 us per biased linear
- Predicted speedup: GPT-2 (144 biased linears): 1080 us saved → ~12% improvement
- Benchmark: compare SmolLM2-135M and GPT-2 before/after
- Falsification: if latency doesn't decrease by >5 us per eliminated task, the barrier cost assumption is wrong

**Optimization 2: Event-driven scheduler**
- Bottleneck: ~100 grid.sync() barriers × 2.5 us = 250 us
- Hypothesis: event-driven eliminates most barriers, enables task overlap
- Expected effect: save 200-500 us on SmolLM2-135M
- Predicted speedup: 8844 → ~8300-8600 us (3-6%)
- Benchmark: A/B test BSP vs event-driven on same schedule
- Falsification: if atomic contention exceeds barrier savings, net negative

**Optimization 3: Vectorized reductions**
- Bottleneck: element-by-element fp16 loads → 12.5% of peak bandwidth
- Hypothesis: float4 loads → 100% bandwidth utilization
- Expected effect: reduce/norm kernels 2-4x faster
- Predicted speedup: norms are ~5% of total → overall ~2-4% improvement
- Benchmark: standalone RMSNorm benchmark, hidden=4096
- Falsification: if kernel is compute-bound (not bandwidth-bound), vectorization won't help

**Optimization 4: Single-pass RMSNorm**
- Bottleneck: 2 passes over input → 2x bandwidth
- Hypothesis: cache row in registers, single-pass reduces bandwidth by 2x
- Expected effect: RMSNorm 2x faster
- Predicted speedup: norms ~5% of total → ~2.5% overall
- Benchmark: standalone RMSNorm, row_size={1024,4096,8192}
- Falsification: if row doesn't fit in registers (row_size > 256 threads × 16 elements = 4096), need SMEM buffer, reducing the gain

**Optimization 5: FlashAttention-style tiled attention with tensor cores**
- Bottleneck: O(seq_k) SMEM, no tensor cores, serial over queries
- Hypothesis: tiled attention with WGMMA matches FlashAttention performance
- Expected effect: attention 5-10x faster
- Predicted speedup: gemma-2b attention is ~40% of runtime → ~2-3x overall improvement
- Benchmark: standalone attention, H=32, D=128, S_k={128,2048,8192}
- Falsification: if WGMMA utilization is low due to persistent kernel constraints, gap remains

---

## 15. Progressive Implementation Roadmap

### Stage 1: Quick Wins + CUTLASS Foundation (Weeks 1-3)

**Changes:**
- Vectorize reduce/rope/index kernels (float4 loads) — 1 day each
- Single-pass RMSNorm (cache row in registers, one global memory read) — 2 days
- Matmul bias epilogue fusion (add bias in-register before store) — 3 days
- CUTLASS multi-config: compile 5 tile variants into cubin — 1 week
  - `gemm_64x64_bk32` (small M×N, high SM utilization)
  - `gemm_64x128_bk32` (rectangular, common MLP shapes)
  - `gemm_128x128_bk64` (current default, large shapes)
  - `gemm_128x256_bk64` (very large N)
  - `gemm_splitk` (M<=4 decode, split K across all SMs with atomic reduction)
- Compile-time tile selector: score each config per (M,N,K), store config_id in TaskDesc

**Implementation detail — CUTLASS multi-config:**
```cuda
// In matmul.cu, replace single gemm path with:
__device__ void dispatch_matmul(const TaskDesc& task, ...) {
    uint8_t config = task.strides[1];  // tile config ID, set by compiler
    switch (config) {
        case 0: gemm_64x64(A, B, C, M, N, K, ...); break;
        case 1: gemm_64x128(A, B, C, M, N, K, ...); break;
        case 2: gemm_128x128(A, B, C, M, N, K, ...); break;  // current
        case 3: gemm_128x256(A, B, C, M, N, K, ...); break;
        case 4: gemm_splitk(A, B, C, M, N, K, ...); break;
    }
}
// Each gemm_* is a CuTe template instantiation with different BM,BN,BK,STAGES.
// Same WGMMA/mma.sync instructions. Different tiling.
```

**Implementation detail — compile-time tile selector:**
```python
# In tiling.py, replace fixed 128x128:
TILE_CONFIGS = [
    (0, 64, 64, 32, 2),    # config_id, BM, BN, BK, stages
    (1, 64, 128, 32, 2),
    (2, 128, 128, 64, 3),
    (3, 128, 256, 64, 3),
]

def select_matmul_config(M, N, K, num_sms):
    if M <= 4:
        return SPLITK_CONFIG_ID  # always split-K for decode
    
    best_score, best_id = 0, 2  # default to 128x128
    for (config_id, BM, BN, BK, stages) in TILE_CONFIGS:
        tiles = math.ceil(M/BM) * math.ceil(N/BN)
        sm_util = min(tiles, num_sms) / num_sms
        waves = math.ceil(tiles / num_sms)
        wave_eff = tiles / (waves * num_sms)
        score = sm_util * wave_eff
        if score > best_score:
            best_score, best_id = score, config_id
    return best_id
```

**Expected gain:** SmolLM2-135M from 0.88x to ~1.05-1.15x vs torch.compile

**Benchmark:**
- Matmul microbenchmarks: each config vs cuBLAS across M={1,4,16,128,512}, N={1024,4096,11008}, K={1024,4096}
- SmolLM2-135M + gemma-2b E2E before/after

**Risk:** Multiple CUTLASS instantiations increase cubin size (~2-3 KB per variant) and compile time (~30s extra). Mitigate: compile once, cache cubin.

**Success criteria:**
- matmul within 20% of cuBLAS on decode shapes (M<=4)
- matmul within 30% of cuBLAS on prefill shapes (M=128+)
- RMSNorm 1.5x+ faster in standalone benchmark
- Overall SmolLM2-135M >= 1.0x vs torch.compile

### Stage 2: FlashAttention (Weeks 4-6)

**Changes:**
- Implement FlashAttention-style tiled attention inside megakernel
- Tensor cores for Q*K^T and Attn*V:
  - SM90: WGMMA (same atoms as matmul)
  - SM80: mma.sync (same atoms as matmul)
- Tile over K in blocks of 64-128 (configurable)
- Online softmax: maintain running max and sum, rescale on new max
- O(1) extra SMEM per head (accumulator in registers, K-block in SMEM)
- GQA support preserved (kv_head = h * num_kv_heads / num_heads)

**Implementation sketch:**
```
for each query block (Bq):
    load Q block to SMEM
    acc = 0, max_old = -inf, sum_old = 0
    for each key block (Bk):
        load K block to SMEM
        S = Q @ K^T   (tensor cores)     ← WGMMA/mma.sync
        if causal: mask S
        max_new = max(max_old, rowmax(S))
        P = exp(S - max_new)
        sum_new = sum_old * exp(max_old - max_new) + rowsum(P)
        acc = acc * exp(max_old - max_new) + P @ V   (tensor cores)
        max_old, sum_old = max_new, sum_new
    O = acc / sum_new
    store O to global memory
```

**Expected gain:** gemma-2b from 0.49x to ~0.8-1.0x vs torch.compile

**Benchmark:**
- Standalone attention: H={8,32}, D={64,128}, S_q={1,32}, S_k={128,2048,8192}
  Compare: megabake vs FlashAttention-2 vs cuDNN
- gemma-2b E2E

**Risk:** FlashAttention inside persistent kernel shares SMEM with matmul (both need ~64-96 KB). With `__launch_bounds__(256, 1)`, only 1 block/SM, so full SMEM budget available (228 KB on H200). Should fit. If not: dynamically partition SMEM per task type.

**Success criteria:**
- attention within 30% of cuDNN FlashAttention at S_k=2048
- gemma-2b >= 0.8x vs torch.compile
- No regression on SmolLM2-135M

### Stage 3: Event-Driven Scheduler (Weeks 7-9)

**Changes:**
- Replace BSP grid.sync() with atomic dependency counters + ready queue
- Build dependency DAG at compile time (graph_walker emits edges)
- Device-side lock-free FIFO queue in global memory
- Intra-task barrier only (participating SMs, not full grid)
- Keep BSP as fallback (compile-time flag)

**Implementation detail — dependency DAG:**
```python
# In graph_walker.py, after fusion passes:
# Build adjacency list from buffer producers/consumers
dep_edges = []
for i, task in enumerate(tasks):
    for j, later_task in enumerate(tasks[i+1:], i+1):
        if buffers_overlap(task.outputs, later_task.inputs):
            dep_edges.append((i, j))

dep_counts = [0] * num_tasks
successors = [[] for _ in range(num_tasks)]
for (src, dst) in dep_edges:
    dep_counts[dst] += 1
    successors[src].append(dst)
```

**Expected gain:** ~500-1000 us saved on SmolLM2-135M (6-11%)

**Benchmark:** A/B test BSP vs event-driven on same schedule. Both SmolLM2-135M and gemma-2b.

**Risk:** Atomic contention, non-determinism, harder debugging. Mitigate: keep BSP as `--scheduler=bsp` flag, add deterministic mode that forces BSP ordering on event-driven queue.

**Success criteria:** >=200 us saved with no numerical regression, no deadlocks

### Stage 4: CUTLASS Parity Push + Epilogue Fusion (Weeks 10-14)

**Changes:**
- CUTLASS epilogue fusion: bias + activation + residual add, all in-register
  - Eliminates 2-6 tasks + barriers per transformer layer
  - Uses CUTLASS `EpilogueWithVisitor` or custom epilogue functor
- Split-K with atomic reduction for bandwidth-bound shapes
- Stream-K for better load balancing across SMs on uneven tile counts
- Weight pre-transposition: store weights as column-major for decode N-major access
- Per-shape tuning database: run each config once at compile time, store winner

**Implementation detail — epilogue fusion:**
```cuda
// After CUTLASS GEMM accumulation, before store:
template<bool HasBias, bool HasSiLU, bool HasResidual>
__device__ void fused_epilogue(float* acc, const __half* bias, 
                                const __half* residual, __half* out,
                                int row, int col, int N) {
    float v = *acc;
    if constexpr (HasBias)     v += __half2float(bias[col]);
    if constexpr (HasSiLU)     v *= 1.0f / (1.0f + expf(-v));
    if constexpr (HasResidual) v += __half2float(residual[row * N + col]);
    *out = __float2half(v);
}
```

Epilogue variant selected by `op_type` + flags in TaskDesc.

**Expected gain:** matmul within 5-10% of cuBLAS → overall ~1.2-1.5x vs torch.compile

**Benchmark:** full matmul shape sweep vs cuBLAS, E2E on all models

**Risk:** Many template instantiations (5 tiles × 8 epilogue combos = 40 kernels). Mitigate: only compile epilogue combos that appear in the model's schedule. Binary size ~100-200 KB extra.

**Success criteria:** matmul within 10% of cuBLAS across all shapes. Models with bias (GPT-2) show >10% E2E improvement from epilogue fusion.

### Stage 5: Advanced Optimizations (Weeks 15-18)

**Changes:**
- Matmul + residual add fusion (common in transformer: `x = proj(x) + residual`)
- Cross-task register forwarding for small intermediates (decode only, <8 KB)
- Layout propagation: column-major output when next consumer is matmul B
- Cost model integration: all fusion decisions gated by quantitative model
- CUTLASS 3.x persistent kernel integration (multiple output tiles per CTA)

**Expected gain:** ~1.3-1.6x vs torch.compile on decode workloads

**Risk:** Complexity. Mitigate: cost model prevents unprofitable fusions. Each optimization must show standalone microbenchmark win before integration.

### Stage 6: Production Hardening (Weeks 19-24)

**Changes:**
- CUDA Graphs integration for hybrid path (prefill/large batch fallback)
- Shape bucketing for dynamic shapes (compile per bucket, dispatch by shape)
- BF16 support (CUTLASS already supports, wire through dtype propagation)
- Comprehensive test coverage (all tile configs, all epilogue combos)
- Per-task profiling infrastructure (clock64 timing per task, exportable)
- nsight Systems/Compute integration guide

**Expected gain:** Reliability and production readiness. No raw performance change.

---

## 16. Prioritized Highest-ROI Changes

| Priority | Change | Effort | Expected Impact | ROI | Stage |
|----------|--------|--------|----------------|-----|-------|
| 1 | Vectorize reduce/rope/index (float4 loads) | 1 day | 2-4% overall | Very high | 1 |
| 2 | Single-pass RMSNorm | 2 days | 2-3% overall | Very high | 1 |
| 3 | Matmul bias epilogue fusion | 3 days | 5-12% on biased models | High | 1 |
| 4 | CUTLASS multi-config autotuning (5 tile variants) | 1 week | 10-30% on matmul | High | 1 |
| 5 | Compile-time tile selector (score per shape) | 2 days | Enables #4 | High | 1 |
| 6 | FlashAttention with tensor cores | 2-3 weeks | 2-5x on attention | High | 2 |
| 7 | Event-driven scheduler | 2 weeks | 6-11% overall | Medium | 3 |
| 8 | CUTLASS epilogue fusion (bias+act+residual) | 1-2 weeks | 5-12% on transformers | High | 4 |
| 9 | Split-K matmul with atomic reduction | 1 week | 10-20% on decode | Medium | 4 |
| 10 | Weight pre-transposition (col-major for decode) | 2 days | 5-10% on decode | Medium | 4 |
| 11 | Stream-K load balancing | 1 week | 5-10% on uneven tile counts | Medium | 4 |
| 12 | Matmul residual-add epilogue | 1 week | 3-5% overall | Medium | 5 |
| 13 | Hybrid cuBLAS + CUDA Graphs (prefill fallback) | 3-4 weeks | 1.0x cuBLAS on prefill | Strategic | 6 |

**Execution order:**
- **This week:** #1, #2, #3 (easy wins, near-zero risk, compound with everything later)
- **Weeks 2-3:** #4, #5 (CUTLASS multi-config — the single biggest unlock for GEMM quality inside the megakernel. Uses the same WGMMA/mma.sync instructions as cuBLAS. Closes 70-80% of the GEMM gap without leaving the pure megakernel architecture.)
- **Weeks 4-6:** #6 (FlashAttention — biggest single-item impact for real models)
- **Weeks 7-9:** #7 (event-driven scheduler — multiplicative with all GEMM improvements)
- **Weeks 10-14:** #8-11 (CUTLASS parity push — the long tail to match cuBLAS within 5-10%)
- **Weeks 15+:** #12-13 (advanced fusion, hybrid path for prefill)

---

## Key Conclusions

1. **GEMM quality is the bottleneck.** Not fusion, not scheduling, not the megakernel concept. Your matmul is 3-5x slower than cuBLAS on many shapes. Fix this first or the megakernel concept can never win.

2. **2x over torch.compile is achievable** on batch-1 decode without CUDA Graphs, IF matmul quality reaches parity with cuBLAS. The math: eliminate 433 kernel launches × 5 us = 2165 us saved, minus ~298 us megakernel overhead = ~1867 us net advantage on a 7802 us baseline = 1.31x from launch savings alone. Add fusion savings and the number climbs to 1.5-2x.

3. **2x over torch.compile + CUDA Graphs is very unlikely.** CUDA Graphs eliminates the launch overhead advantage. What remains is L2 reuse (worth ~1 us for decode) and fusion savings (~100-200 us). Not enough.

4. **The attention kernel must be rewritten.** No tensor cores + materialized scores + serial query loop = 5-10x slower than FlashAttention. This is the single biggest gap on real models.

5. **The megakernel concept is sound for decode.** Batch-1, single-token decode is the sweet spot: all operations are memory-bound, intermediates trivially fit in L2, and launch overhead is a significant fraction of total time. This is the right market to target.

6. **Don't fight cuBLAS on prefill.** For batch>=32 or seq>=512, the matmul is compute-bound and cuBLAS's autotuned tensor core utilization is unbeatable without years of engineering. Use hybrid architecture for prefill.
