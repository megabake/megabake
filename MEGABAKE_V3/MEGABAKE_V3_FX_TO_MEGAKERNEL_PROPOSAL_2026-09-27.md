# MegaBake V3: a full-step FX graph's path to an optimized CUDA megakernel

Worked design walkthrough, revised 2026-09-27. The filename is retained from the original proposal; the [architecture](MEGABAKE_V3_ARCHITECTURE.md) and [IR contracts](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) are now normative. This is a thought experiment and search procedure, not an implemented performance result.

## 1. I arrive as a captured Hugging Face step

I am a full cached inference step represented by Torch FX. My inputs include a new token or token batch, positions/valid lengths, stable model weights and an old cache/state. My outputs include logits or declared hidden values and required new state. My nodes may spell the same mathematics as `aten.addmm`, `mm`, `view`, `transpose`, `mul`, `rsqrt`, `scaled_dot_product_attention`, or a decomposed sequence. Some layers may be windowed, use grouped-query attention, have a different norm or carry recurrent state.

I first want the compiler to prove that it has **all** of me. A custom `torch.compile` backend can be called on only one graph-break fragment; that fragment does not represent a whole decode step. I want a direct FX/ExportedProgram entry with lifted parameter bindings, guards, aliases, effects, exact output ownership and numerical policy. The Hugging Face helper may prepare a supported static cache, but my graph and bindings, rather than a model-class name, determine meaning. [PyTorch custom backend interface](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)

Before changing me for performance, establish the best equivalent `torch.compile` path on the selected CUDA device. It may select cuBLAS/cuBLASLt and strong attention kernels. Record its exact hot shapes, dtype, strides, cache layout, selected operations and complete-call latency. Without that target, a faster isolated MegaBake body could still leave the full model slower. [Performance protocol](MEGABAKE_V3_PERFORMANCE_MODEL.md)

## 2. I become indexed semantics, not a list of FatOps

I keep my original executable FX reference. A conservative normalizer makes state effects and selected ATen forms explicit, then proves shape, stride, alias, mask, cast and guard facts. The compiler attaches an `IndexedTensorProgram` to my regions:

```text
projection: Y[b,n] = cast_out(sum_k X[b,k] * W[n,k] + bias[n])
norm:       each output uses the exact reduction axis, epsilon and cast points
attention:  exact score scale, mask, causal/window alignment, head mapping, cache state
cache:      state_next[index] receives the declared value after required producers
```

I do **not** need a new `FatOp(CheckpointFamily)` to reach code generation. Supported maps, broadcasts, reductions, contractions, views, indexing and functional state effects have generated CUDA lowerings. An optional `RMSNorm`, `RoPE`, `SDPA`, gated-MLP or recurrent recognizer can propose an optimized algorithm if it proves equivalence to my FX region. A custom op without supplied semantics receives a strict diagnostic; its ordinary fallback is reported separately. An unfamiliar primitive may require new semantics and a new high-performance algorithm, but an unfamiliar arrangement of familiar primitives should not.

Every rewrite retains my other live outputs. A cache write cannot disappear because its returned tensor is unused. A transposed view cannot be discarded unless the consumer's indexing map incorporates it. Split-K or fused cast order must satisfy the declared numerical policy.

## 3. My repeated structure is verified

If my model contains many decoder blocks, the compiler verifies their boundaries and records a `RepeatRegion`: iteration count, per-layer weights/state, carried residual, exceptional variants and reference behavior. A config layer list helps find candidate structure; it is not proof. One layer may be full attention while another is recurrent or windowed. The target may emit a device loop, partial unroll or flat specialized worker program. The reason to represent repetition is whole-region storage, code size and cross-layer movement, not merely nicer names.

For my next layer, the weight addresses may be known before its input activation. This becomes a legal early-address fact. It does not force a CUDA preload; the eventual body, shared storage, bandwidth and participant assignment decide whether one pays. Inferact's [hand-specialized TPU program](https://inferact.ai/blog/tpu-megakernels) demonstrates the cross-layer opportunity under different VMEM/DMA hardware.

## 4. I preserve algebraic alternatives

The compiler enumerates a bounded set of equivalent algorithm and layout choices before choosing tiles or scheduling:

| My region | Alternatives I want retained |
|---|---|
| `Y = XW^T` | K-parallel SIMT GEMV, `Y^T = WX^T` tensor-core work, padded narrow batch, smaller N tiles, split-K and larger-batch GEMM |
| Shared normalized activation to Q/K/V | Separate linears, packed QKV contraction, fused legal positional/cache epilogue |
| Gate and up | Separate linears, packed weights, paired hidden outputs with exact activation/casts |
| Down projection | Full hidden materialization plus high-quality full-K body, owner-held chunk continuation, or paid split partials |
| Attention | Streaming online softmax, split-context combine where parallelism pays, conservative reference expansion |
| Layer sequence | Flat specialization, template loop, partial unroll and alternate activation ownership |

A large packed QKV body may make matrix math faster and head readiness later. A streamed down projection may begin early but lose a superior full-K tensor-core body. Those are measurable tradeoffs, not reasons to always fuse or always stream. An equivalence rule includes its guard, reference relation, required preparation and numerical allowance; the compiler retains a correct unfused alternative.

## 5. My linear bodies confront cuBLAS

For a batch-one projection with W stored `[N,K]`, a generated K-parallel SIMT body gives warp lanes chunks of K, reduces in the declared accumulator precision and applies the epilogue after full reduction. This removes the old V2 serial-per-output mapping recorded in the [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md). A second tactic computes `Y^T = W X^T` so output channels become the large tensor-core axis. The narrow batch dimension may be padded, with activation initialization charged per invocation and only real outputs stored.

If N=576, 64-channel output tiles yield nine tensor-core tasks before a K split; a SIMD/SIMT mapping can expose more independent work. If N=4096, 64 such tiles may make the tensor-core path more viable. Split-K creates extra tasks but adds partial storage, a combine and a numerical-order question. Mirage's [Hopper task body](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh) is one implementation reference. Hazy reports a target-dependent CUDA-core/tensor-core tradeoff for its low-batch model. [Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)

The body provider must include global loads, layout, MMA or SIMT reduction, epilogue, stores and any staging. A bare tensor-core instruction or a host-callable cuBLAS kernel is not a solution. Evaluate cuBLASDx's **pipelined** global GEMM path, adapted CUTLASS C++ collectives and owned SIMT kernels. cuBLASDx exposes host descriptor, `__grid_constant__` handle, block-size, divisibility and persistent tile-lifecycle requirements; it may be fast for one shape yet incompatible with the mixed entry. [cuBLASDx pipeline guide](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html), [CUTLASS collective API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)

For each hot shape, benchmark the selected vendor path, owned standalone tactics and the same tactics in a lean persistent entry. Keep a Pareto set of latency, registers, shared storage, block compatibility, output granularity and epilogue/stage access. An isolated deficit can be paid by a real fused-work saving; a repeated 4–6x deficit on dominant math should send the project back to body algorithms before building a more elaborate queue.

## 6. I become symbolic tile work

Once viable body/tile alternatives exist, derive each logical task family's iteration domain and exact read/write/reduction maps. A Q attention task waits for the needed Q, mapped K/V, current cache write and valid lengths, not all unrelated heads. A down-projection `reduce_update(I,J)` reads only hidden chunk J, matching weight slice and its owner's accumulator; `reduce_finalize(I)` waits for all required chunks. My full hidden vector is still an alternative if the continued body is inferior.

Producer sets come from overlapping index regions and effects. Represent them symbolically and compress equivalent events before target dispatch. Luminal's [block-domain expressions](https://blog.luminal.com/p/compiling-models-to-megakernels) and MPK's [event fusion](https://arxiv.org/html/2512.22219v2) are useful design examples. Logical tasks do not contain a CTA ID, fixed lookahead count or CUDA barrier. A changed body tile may regenerate this graph.

Five moments remain distinct: source address known, input data ready, result computed, result visible to another worker, and old storage safely reusable. Async copies may distinguish source-read retirement from destination visibility. My state effects and output ownership extend lifetimes beyond a convenient topological ordering.

## 7. The selected SM decides my physical program

The CUDA backend queries the selected device/visible partition and proposes compatible body mixtures, block configurations, tile assignments, shared/global/tensor-memory layouts, event protocols and static worker programs. A barrier program is a matched control. Head-ready attention, chunked MLP and next-task weight staging become optional measured variants. Irregular expert work may justify a dynamic/hybrid dispatcher; regular dense decode starts with a low-overhead static plan.

The planner searches over the repeated region because an activation handoff and a weight slot can affect later layers. Hazy's [B200 breakdown](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) makes activation publication/reload a material candidate cost. Inferact's TPU VMEM strategy motivates lifetime visibility but cannot be copied to CUDA's much smaller per-CTA storage. [Inferact](https://inferact.ai/blog/tpu-megakernels)

One CUDA entry has a common compiled block shape and resource envelope. The backend compiles selected bodies together, inspects registers, spills, shared memory, code size and actual cooperative residency, then revises the body/tile/schedule choice if the entry loses. A body that is fastest alone can lose through entry-wide register or participant pressure; MPK notes this resource issue explicitly. [MPK](https://arxiv.org/html/2512.22219v2)

The same indexed semantics can produce different target entries: Ampere/Ada warp MMA, Hopper `sm_90a` TMA/WGMMA, data-center Blackwell tcgen05/TMEM or a different Blackwell target. Compile and tune by supported feature set and CUDA/compiler/provider version. `_a` architecture-specific code has exact-target compatibility; portable source does not imply one portable fast cubin. [CUDA compute capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

## 8. I earn or fail the strict result

The compiled entry first proves output, state, descriptor, numerical, storage and progress correctness. It then faces the best equivalent `torch.compile` complete-step baseline on the same target, inputs, cache, numerical and output contract. Record raw unprofiled samples, resources, grid count and setup. A vendor-preserving multi-grid plan can be selected for a product call when it is faster; its speed is a separate result.

I want the compiler to report my exact status: capture failed, indexed semantics unsupported, strict grid incorrect/illegal, strict compiled but slower, or strict measured win. A second structurally different HF family tests whether the generic primitive and algorithm-provider boundaries carry over. A second SM generation tests whether source-level tactic families are correctly retargeted and tuned. Neither result is inferred from one model or one GPU.

## 9. The remaining hard limit

This pipeline puts V3 in a stronger position because it attacks the two earlier blockers directly: model-family FatOp maintenance and vendor-relative body quality. It is still possible that no device-callable body mixture for a declared cell beats cuBLAS/cuBLASLt enough to repay one-grid resources and coordination. That finding would be a strict loss requiring better body algorithms or a changed qualifying objective, not another schedule diagram.

## 10. What I expect to see after each transformation

| After | Concrete artifact | If it is missing |
|---|---|---|
| Capture | Original FX callable, flat binding/output/state signature, cache-position contract | Stop: a fragment or uncached wrapper is not my full step |
| Normalization/facts | Versioned copied graph, origin map, proven shapes/strides/aliases/casts/effects | Stop the unsafe transform; keep original reference |
| Indexed lowering | Every live FX output/effect mapped to a typed indexed op or a strict diagnostic | A leftover `ReferenceRegion` cannot be counted as device coverage |
| Algorithm enumeration | Guarded alternatives plus unfused reference and preparation costs | Keep the unfused algorithm; do not guess a model-family pattern |
| Body generation | Exact-shape/target SIMT and legal MMA candidates with numerical and stage contracts | Report body-quality gap before investing in a queue |
| Logical tiling | Parametric work domains, exact producer maps, reduction owners and lifetimes | Reject missing writer or unsafe reuse before CUDA emission |
| CUDA plan | Worker/block/storage/event program with compatible bodies and progress argument | Reject nonresident producer waits or incompatible CTA roles |
| Compile/admit | Actual mixed-entry register/shared/spill/code report and cooperative worker bound | Replan; a standalone body's occupancy is insufficient |
| Runtime/measurement | Correct advancing state/logits, one compute-grid trace, matched raw complete-call samples | Report strict loss/incorrect/unsupported distinctly from fallback |

The [implementation handbook](MEGABAKE_V3_IMPLEMENTATION.md) gives per-card edit surfaces and evidence handoffs that can be assembled into bounded dependency-closed batches. My first unfamiliar composition can reach a correct generic SIMT entry even when no optional recognizer matches; my full cached step then exposes whether the resulting body mixture is competitive with the best validated `torch.compile` path. A different SM reuses my indexed meaning but chooses new legal body, block and schedule candidates.
