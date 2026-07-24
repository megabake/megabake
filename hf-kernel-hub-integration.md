# HF Kernel Hub Integration: Combined Analysis

## Context

Megabake compiles PyTorch models into a single persistent CUDA megakernel using BSP (Bulk Synchronous Parallel) execution. The vision: `torch.export` → task schedule + cubin artifact, distributed via HF Hub, zero compilation for end users.

**Current performance reality** (H200 MIG, 32 SMs):

| Model | Eager (us) | torch.compile (us) | Megabake (us) | vs compile |
|-------|-----------|-------------------|--------------|------------|
| SmolLM2-135M | 14,158 | 7,802 | 8,856 | 0.88x (slower) |
| gemma-2b | 11,034 | 7,401 | 15,042 | 0.49x (slower) |

We have 1 kernel launch vs 262-433, but we're losing. The Hub integration strategy must address both **performance** and **ecosystem** — fixing the engine and distributing it are equally critical.

---

## Part 1: Why Megabake Is Slower (Root Causes)

### RC-1: Naive Attention (dominant for gemma-2b)

`src/cuda/tasks/attention.cu` implements textbook O(N²) SDPA — materializes the full attention matrix in shared memory, two-pass softmax, no sequence tiling, no online softmax. torch.compile dispatches cuDNN/FlashAttention which is 5-10x faster for this op. With gemma-2b having 8 heads and more attention compute, this single bottleneck explains most of the 0.49x regression.

### RC-2: Grid Sync Barrier After Every Task

`src/cuda/megakernel.cu:62` — `grid.sync()` after each of ~50-80 tasks. Each barrier costs ~5-10us, totaling 250-500us of pure synchronization overhead. The design doc acknowledges a 10-15% gap vs event-driven scheduling, but the real gap is larger because many tasks are trivially lightweight (single-row reductions, small elementwise ops) yet each still pays the full barrier cost.

### RC-3: Massive SM Underutilization (wave quantization)

`src/megabake/schedule_compiler/tiling.py` uses fixed tile sizes:
- **Matmul (decode, M=1, N=2048):** `ceil(1/128) * ceil(2048/128) = 16 tiles` on 32 SMs → 50% idle
- **Attention (batch=1, heads=8):** 8 tiles on 32 SMs → 75% idle
- **RMSNorm (batch=1):** 1 tile on 32 SMs → 97% idle

Every idle SM still pays the barrier cost. The BSP model makes this a multiplicative penalty — utilization × barrier overhead compounds per task.

### RC-4: No Cross-Task Data Fusion

Despite being a single kernel, the BSP model forces every intermediate result to global memory before the barrier. matmul output → HBM → norm reads back → HBM → elementwise reads back. torch.compile's inductor fuses epilogue chains (matmul → bias → activation → norm) into single kernels that keep data in registers/SMEM.

Megabake does fuse matmul+activation (MATMUL_SILU/GELU) and elementwise chains, but not across task-type boundaries.

### RC-5: No Bias Fusion in Matmul Epilogue

`src/cuda/tasks/matmul.cu` — the CuTe tensor-core paths ignore `buffer_indices[3]` (bias). The graph walker sets it (`graph_walker.py:1131`), but the CUDA kernel doesn't read it. Every `addmm` requires a separate elementwise ADD task + barrier.

---

## Part 2: How HF Kernel Hub Can Help

### Strategy A: Fix the Engine (Performance)

These address the root causes directly. HF Hub kernels serve as references, building blocks, and fallback implementations.

#### A-1: Replace Naive Attention with FlashAttention

**Approach:** Use `kernels-community/flash_attn` (flash_attn_func, flash_attn_varlen_func) as either:
- A **standalone kernel call** at graph-split points around attention ops (hybrid approach — break the megakernel at attention boundaries, call FlashAttention, then resume)
- A **reference implementation** to guide writing a FlashAttention-style tiled attention task inside the megakernel (online softmax, sequence-dimension tiling, register accumulation)

**Impact:** This alone would likely flip gemma-2b from 0.49x to >1x vs torch.compile. Attention dominates runtime in multi-head transformer models.

**Effort:** 2-3 days for the hybrid (break-out) approach. 1-2 weeks to implement proper tiled attention inside the megakernel.

#### A-2: Hardware-Aware Tiling

**Approach:** Leverage Hub's per-architecture kernel variants as a model for megabake's tiling strategy:
- Adaptive tile sizes based on problem dimensions and SM count (not fixed 128×128)
- Skinny matmul path for decode (M=1-4): tile only along N, all SMs participate
- Attention tiles = `batch * heads * ceil(seq_q / BLOCK_Q)` instead of just `batch * heads`
- Reduce tiles = `ceil(num_rows / rows_per_sm)` with multiple rows per SM for small batch

**Impact:** Brings SM utilization from 50-97% idle to near-full on decode workloads.

#### A-3: Hub Kernels as Correctness References

Use Hub's production-quality kernels as ground-truth for testing each megakernel task:

```python
from kernels import get_kernel
flash_attn = get_kernel("kernels-community/flash-attn")
rmsnorm_k = get_kernel("kernels-community/triton-layer-norm")

def test_attention_task():
    mb_out = run_single_task(OP_ATTENTION, q, k, v)
    ref_out = flash_attn.flash_attn_func(q, k, v)
    assert torch.allclose(mb_out, ref_out, atol=1e-3)
```

Hub kernels are battle-tested in TGI and transformers — stronger correctness signal than vanilla PyTorch.

**Effort:** 1 day. **Impact:** Catches subtle numerical issues early, especially for attention and norms.

#### A-4: Fallback Kernels for Graph Splits

When the megakernel encounters an unsupported op (graph split), pull optimized implementations from the Hub instead of falling back to eager PyTorch:

```python
from kernels import get_kernel

def execute_with_fallback(segments, model, inputs):
    for seg in segments:
        if seg.type == "megakernel":
            output = megabake.run(seg.schedule, model, *seg.inputs)
        else:
            kernel = get_kernel(f"kernels-community/{seg.op_name}")
            output = kernel.forward(*seg.inputs)
```

This means megabake doesn't need 100% op coverage to handle 100% of models. The Hub fills gaps with optimized implementations, and the user experience degrades gracefully:

```
Best:     100% fused in megakernel
Good:     95% fused, 5% via Hub kernels (still fast)
Okay:     Multiple megakernel segments + Hub kernels between them
Baseline: Pure torch.compile (no megakernel at all)
```

**Effort:** 2 days. **Impact:** Unlocks models with custom/unsupported ops without writing new task kernels.

---

### Strategy B: Distribute the Engine (Ecosystem)

These leverage the Hub for packaging, distribution, and integration — turning megabake from a tool into an invisible acceleration layer.

#### B-1: Publish megabake-runtime as a Kernel Repo

Publish the compiled megakernel (cubin + scheduler + all task implementations) as a Hub kernel repo. HF's build infrastructure handles the variant matrix:

```
megabake/megabake-runtime
  build/
    torch213-cxx11-cu128-x86_64-linux/   ← SM80 (A100)
    torch213-cxx11-cu130-x86_64-linux/   ← SM90 (H100/H200)
    torch213-cxx11-cu132-x86_64-linux/   ← SM100 (Blackwell)
```

Users load with `get_kernel("megabake/megabake-runtime")`. This replaces `cuda_compiler.py` entirely — no runtime compilation, no nvcc dependency, no build failures on user machines.

**Effort:** 1 week. **Impact:** Solves distribution permanently. The "compile once, distribute everywhere" vision from the design doc becomes real.

#### B-2: Register Megakernel as torch.compile Custom Op

Register the megakernel launch as a `TORCH_LIBRARY` custom op with a FakeTensor kernel:

```cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("run_megakernel(Tensor tasks, Tensor buffers, ...) -> Tensor");
    ops.impl("run_megakernel", torch::kCUDA, &run_megakernel_impl);
}
```

```python
@torch.library.register_fake("megabake::run_megakernel")
def run_megakernel_fake(tasks, buffers, ...):
    return torch.empty_like(buffers[0])
```

When megabake handles transformer layers but custom heads/loss functions sit on top, torch.compile sees the megakernel as a single opaque op — no graph breaks. The megakernel and torch.compile coexist.

**Effort:** 1 day. **Impact:** Composability with the broader PyTorch ecosystem.

#### B-3: Per-Architecture Task Repos (Modular Kernel Development)

Publish heavy tasks as independent Hub repos:

```
megabake/matmul-sm90       ← Hopper WGMMA GEMM
megabake/matmul-sm100      ← Blackwell TCGEN05
megabake/attention-sm90    ← Hopper TMA-based attention
megabake/megakernel-core   ← BSP scheduler + lightweight tasks
```

Each evolves independently. A better SM90 matmul is a version bump, not a full megakernel recompile.

**Catch:** These must be compiled together into one cubin. The Hub doesn't compose kernel repos into one binary. This becomes a build-service pattern: individual repos are source, the combined `megabake-runtime` is the published artifact.

**Effort:** 2 weeks. **Impact:** Enables community contributions to individual task kernels. Long-term play.

#### B-4: Auto-Load via Transformers Integration

HF transformers already has `hub_kernels.py` integration. Models can declare preferred Hub kernels:

```python
@use_kernel_forward_from_hub("LlamaDecoderLayer")
class LlamaDecoderLayer(nn.Module):
    ...
```

If megabake publishes a kernel matching `LlamaDecoderLayer`, transformers downloads and uses it automatically. Zero-config acceleration — users don't install megabake, don't call `megabake.compile()`, don't know it exists.

**Effort:** 1 week (after B-1). **Impact:** The dream end-state. Requires performance to be competitive first.

#### B-5: Agentic Kernel Development for New Tasks

Hub supports agentic kernel development (scaffold → build → benchmark → optimize). When megabake encounters an unsupported op:

1. Agent generates a task implementation (CUDA code)
2. `kernel-builder` compiles it
3. Published to Hub as `megabake/custom-op-name`
4. Next user with the same model gets it automatically

**Effort:** Long-term. **Impact:** Organic op coverage growth without manual kernel authoring.

---

## Part 3: Recommended Roadmap

### Phase 1: Fix Performance (weeks 1-3)
**Goal:** Beat torch.compile on SmolLM2-135M and gemma-2b.

| Priority | Action | Source | Expected Impact |
|----------|--------|--------|----------------|
| P0 | Replace naive attention — either break out to FlashAttention (hybrid) or implement tiled attention in megakernel | A-1 | Fix the 0.49x gemma-2b regression |
| P0 | Adaptive tiling — skinny matmul for decode, more attention tiles, multi-row reduce | A-2 | Fix 50-97% SM idling |
| P1 | Reduce barrier frequency — batch lightweight tasks, fuse matmul epilogues (bias+activation+norm) | RC-2, RC-4, RC-5 | Eliminate 250-500us barrier overhead |
| P1 | Hub kernels as test references | A-3 | Correctness confidence |

### Phase 2: Ecosystem Integration (weeks 3-5)
**Goal:** Make megabake distributable and composable.

| Priority | Action | Source | Expected Impact |
|----------|--------|--------|----------------|
| P0 | Publish megabake-runtime on Hub | B-1 | Eliminate runtime compilation |
| P0 | Fallback to Hub kernels for unsupported ops | A-4 | 100% model coverage |
| P1 | torch.compile custom op registration | B-2 | No graph breaks in mixed usage |

### Phase 3: Invisible Acceleration (weeks 5+)
**Goal:** Users load a model and it's fast — they never see megabake.

| Priority | Action | Source | Expected Impact |
|----------|--------|--------|----------------|
| P0 | Transformers auto-load integration | B-4 | Zero-config UX |
| P1 | Per-architecture task repos | B-3 | Community kernel contributions |
| P2 | Agentic kernel development | B-5 | Organic op coverage |

---

## Part 4: Key Disagreements Between Analyses

| Topic | Performance Analysis | Ecosystem Analysis | Resolution |
|-------|---------------------|-------------------|------------|
| **Priority** | Fix the engine first — can't distribute a slow artifact | Distribution and integration are the highest leverage | **Both right, but sequenced:** performance first (Phase 1), then distribution (Phase 2). Publishing a 0.49x artifact is counterproductive. |
| **Hybrid vs pure megakernel** | Break out heavy ops (attention, large matmuls) to standalone optimized kernels | Keep the megakernel pure, use Hub for distribution only | **Hybrid is pragmatic for now.** The BSP model fundamentally limits cross-task fusion. Break out the 3-4 ops that eat 80% of runtime, keep the megakernel for the long tail. Revisit pure megakernel when event-driven scheduling lands. |
| **torch.compile interop (B-2)** | Premature — need to be faster than torch.compile first | High value, 1-day effort | **Agree it's cheap, but sequence it after Phase 1.** Registering a custom op for a slower kernel is useful for testing but not for users. |
| **Per-arch task repos (B-3)** | Not addressed | 2-week effort, enables community | **Good long-term play but the composition problem (no Hub mechanism to merge repos into one cubin) makes it a build-service concern, not a distribution concern.** Defer to Phase 3. |
| **Matmul quality** | CuTe WGMMA/MMA paths use tensor cores properly; the main issue is tiling/sizing, not the kernel itself | Not analyzed | **Matmul kernels are solid** — the SM90 WGMMA path and SM80 MMA path are legitimate tensor-core GEMMs. The problem is tiling strategy (fixed 128×128 regardless of problem size) and missing epilogue fusion, not the core compute. |
| **The "scalar matmul" fallback** | When B is not transposed, falls back to `matmul_scalar` — a catastrophically slow element-by-element path | Not analyzed | **Must verify this path isn't hit on real models.** If any linear layer produces a non-transposed B, the entire model tanks. Worth adding an assertion or graph-walker fix. |

---

## Summary

The HF Kernel Hub is not a silver bullet for performance — megabake's problems are in its own engine (naive attention, barrier overhead, tiling strategy). But the Hub is the right answer for everything around the engine: distribution, hardware targeting, fallback coverage, ecosystem integration, and the long-term vision of invisible acceleration.

Fix the engine. Then distribute the engine. Then make the engine invisible.
