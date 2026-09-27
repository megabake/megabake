# MegaBake V3: deep research behind the revised compiler

Research synthesis revised 2026-09-27. This file records the case for the [normative architecture](MEGABAKE_V3_ARCHITECTURE.md) and its unresolved risks. It does not contain a measured MegaBake V3 speedup. The [FX walkthrough](MEGABAKE_V3_FX_TO_MEGAKERNEL_PROPOSAL_2026-09-27.md) applies the conclusion to one representative full-step graph.

## 1. The question the research must answer

The north star is a compiler from complete Torch FX inference steps to one owned CUDA compute grid that consistently beats the strongest equivalent `torch.compile` execution on declared Hugging Face Transformers workload/target cells. CUDA is primary; the semantic and dependence boundary should permit a later TPU backend. The two hard problems are **generic model coverage without manually authored family FatOps** and **device-callable math near enough to private library kernels after full-entry composition**.

A universal guarantee is impossible: some FX graphs already need only one excellent kernel, and the one-grid envelope can cost more than it saves. “Consistently” must mean a predeclared measured matrix with strict coverage and loss rate reported, not a theorem over arbitrary FX and GPUs. Vendor-preserving multi-grid execution is a legitimate separately scored product path.

The historical [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) reports V2 hot-shape deficits of 15.49 versus 2.63 us and 95.78 versus 23.39 us on two batch-one projections, and a 4.85 ms V2 forward versus 1.78 ms low-overhead compiled comparison. These are old, potentially L2-hot or uncached, specific-H200-MIG observations, not V3 predictions. They are large enough that a launch-only or scheduling-only rewrite is not a credible default solution.

## 2. Lessons from the primary work

| Source | Directly established mechanism | V3 use | Caveat |
|---|---|---|---|
| [Mirage MPK v2](https://arxiv.org/html/2512.22219v2) | SM-level task graph, reference-to-CUDA task-body generation, event fusion, AOT/JIT dispatch and resource discussion | Generate bodies as well as dependencies; compact symbolic task relations; preserve schedulers as alternatives | Its measured models/hardware and superoptimizer do not prove cuBLAS parity for our exact FX shapes |
| [Hazy no-bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) | Hand-built Llama-1B instruction schedule, weight staging and detailed activation/sync cost | Search activation ownership and full handoff cost; test static programs | It is hand-specialized; B200/Hopper body choices differ and do not transfer unchanged |
| [Luminal megakernels](https://blog.luminal.com/p/compiling-models-to-megakernels) | Symbolic block work/data/barrier strides and dynamic work queue | Parametric tile domains and exact dependency relations | Queue policy is a target cost choice, not generic coverage or a universal win |
| [Inferact TPU](https://inferact.ai/blog/tpu-megakernels) | Hand-specialized repeated Pallas program, VMEM lifetimes and cross-layer DMA on TPU v7 | Verified repeat regions; separate address readiness from activation readiness; target-specific physical planner | TPU VMEM, gridless execution and mesh collectives are not CUDA shared memory/CTA scheduling |
| [NVIDIA cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html) | Pipelined global GEMM inside a CUDA kernel with host descriptor and participation constraints | Important exact-shape device-body candidate | Not a callable private cuBLASLt kernel or guarantee of its selected tactic's speed |
| [CUTLASS collectives](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) | Modular MMA/data-movement/epilogue components below host adapters | Controllable SM-specific tensor-core bodies | Adapted body still requires full-entry resource and protocol verification |

Additional context: [Event Tensor](https://arxiv.org/html/2604.13327) supports static and dynamic schedules from fine-grained dependencies and shows their tradeoff; [FlashAttention](https://arxiv.org/abs/2205.14135) motivates IO-aware attention algorithms; [FlashInfer](https://arxiv.org/html/2501.01005) shows that decode attention and cache layouts form their own workload space. These sources do not replace exact MegaBake baselines.

## 3. The previous V3 architecture's decisive flaw

The earlier V3 documentation correctly separated logical and target plans, guarded numerical/state behavior, tracked asynchronous lifetimes, and called for joint body/fusion/schedule selection. Its canonical semantic stage still centered a named FatOp set with ordinary ATen reference regions left over. An unfamiliar HF graph could therefore be reference-correct yet lack any strict device-code path until another family-specific matcher or kernel was supplied. The pipeline-first implementation work could make major progress while the best cuBLAS-relative math remained too slow.

The revised architecture makes indexed tensor computation and effects the **canonical supported program**. Its generic CUDA generator can lower supported maps, reductions, contractions, indexing and state updates regardless of their arrangement. Named norms, attention, gated MLPs and recurrent patterns are guarded algorithm choices. Their absence cannot block generic lowering of supported primitives. An unknown operator still needs a semantic definition; generic compilation does not invent a fast novel algorithm.

The second change is a true math-tactic search. For one projection `Y = XW^T`, alternatives include K-parallel SIMT GEMV, `Y^T = W X^T` with legal padding, smaller output tiles, split-K and target-specific GEMM pipelines. Each has a full global-load/reduction/epilogue implementation, not just an MMA primitive. The provider returns a Pareto set of latency, tile parallelism, resources and stage capabilities. Hazy's different Hopper/B200 matvec findings and Mirage's swapped small-batch body demonstrate why one universal M=1 kernel is weak. [Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [Mirage body](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh)

## 4. Body quality and composition are coupled

A cuBLASLt result is a strong performance comparison, not a component that can be linked into the owned grid. The cuBLASDx block API leaves global loading/tiling to the caller. Its [pipeline API](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html) moves global stages inside a device-callable GEMM path but requires host-created metadata, tile divisibility, particular block participation and a `__grid_constant__` handle. Fixed-block and reusable-accumulator options can disable optimizations. It is a tactic to measure beside CUTLASS-derived and generated SIMT bodies.

A body that wins alone can lose inside a model because the entry's block size, register count, shared memory, spills, code size and participant roles are common constraints. MPK explicitly calls out the fixed maximum register requirement across task types. [MPK discussion](https://arxiv.org/html/2512.22219v2) Conversely, a slower standalone body with local epilogue fusion or smaller resource footprint may win the complete step. An isolated per-op slowdown is a warning; it is not a hard veto if a guarded composite removes enough work. A conservative break-even bound caps exploration and the compiled full invocation decides. [Performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md)

This is why the body-provider interface includes global movement, output tile, numerical order, descriptor lifetime, acceptable entry block shapes, scratch and stage capability. An internally pipelined K loop is not automatically an externally preloadable operation or a continued reduction over producer-ready chunks. If no legal embedded tactic closes the budget for a dominant shape, more task scheduling cannot establish a strict win for that cell. Improve the algorithm/provider or report the loss.

## 5. Whole-region dependence and portability

The common program should express that layer N+1's weight address is known while its activation is not, that a head or hidden chunk can become ready before its enclosing operator finishes, and that a buffer can be reused only after every old access retires. It should not prescribe a CTA cohort, shared-memory page or fixed one/two-step lookahead. CUDA and a future TPU backend can generate different physical programs from the same exact dependence and effect facts.

A verified repeated region is essential for whole-model decisions: per-layer bindings, carried residual/state, exceptional branches and code-size choices become explicit. Inferact's hand-specialized 92-layer TPU program shows the value of this view, while its large VMEM and mesh-specific collectives show why a copied CUDA-style scheduler would be the wrong common IR. [Inferact](https://inferact.ai/blog/tpu-megakernels)

Symbolic task domains and access maps avoid explicit quadratic producer/consumer enumeration. The selected target may compact events, use static worker programs for regular dense decode, or add dynamic/hybrid dispatch for routed experts and variable work. Luminal's queue and MPK's scheduler are implementation candidates. They do not create math bodies or prove that queue overhead is paid on dense batch-one inference. [Luminal](https://blog.luminal.com/p/compiling-models-to-megakernels), [MPK](https://arxiv.org/html/2512.22219v2)

Portability across NVIDIA SMs is also source- and algorithm-level. The compiler shares indexed semantics and tactic families, then compiles feature-gated bodies for the target. Ampere/Ada warp MMA, Hopper `sm_90a` WGMMA/TMA and Blackwell-specific instructions have different legality/resource behavior; `_a` code has exact-target compatibility. CUDA/toolchain/provider versions belong in artifact keys. [CUDA feature sets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)

## 6. Audit of the V3 file set after revision

| File | Revised ownership |
|---|---|
| [README](MEGABAKE_V3_README.md) | North star, status and one canonical pipeline |
| [Architecture](MEGABAKE_V3_ARCHITECTURE.md) | Indexed semantic center, algorithm/body search, target plan and strict result |
| [IR](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) | FX capture, exact indexed semantics, guarded algorithms, repeat regions and verified plans |
| [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md) | Generated/competitive providers, cuBLASDx/CUTLASS constraints and admission experiment |
| [Pipeline](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) | Optional schedules and non-negotiable correctness/progress protocol |
| [Hardware](MEGABAKE_V3_HARDWARE_MODEL.md) | SM/CUDA feature gating, actual-entry resources and calibration provenance |
| [Performance](MEGABAKE_V3_PERFORMANCE_MODEL.md) | Break-even, strongest baseline, raw evidence and honest scorecard |
| [Implementation](MEGABAKE_V3_IMPLEMENTATION.md) | Body-quality and FX vertical path before broad scheduler machinery |
| [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) | Historical observations, discrepancies and revised diagnostic implications |
| [Diagrams](MEGABAKE_V3_DATAFLOW_DIAGRAM.md) | Search feedback, strict/external paths and target boundary |
| [Research ledger](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md) | Provenance, adopted decisions and falsifiers |
| [FX walkthrough](MEGABAKE_V3_FX_TO_MEGAKERNEL_PROPOSAL_2026-09-27.md) | Worked step-by-step application without model-name assumptions |

The two research files explain the normative design; they do not supersede the architecture, IR or body contracts with alternate versions.

## 7. Falsifiable sequence

1. Capture a complete cached FX step and strong equivalent compiled/vendor baseline. Inventory hot exact shapes and removable critical-path work.
2. Implement an indexed generic path and a K-parallel SIMT body; compare with at least one target-supported tensor-core tactic inside a lean persistent entry.
3. If the math budget is plausible, emit one correct graph-derived block, first with simple scheduling, then optional fusion/handoff/pipeline candidates.
4. Recover a verified repeated region and compile a whole model step; measure raw complete-call samples and state correctness.
5. Hold out a structurally different HF family and test another SM generation when hardware exists. Report every strict loss, unsupported node and fallback.

Failure has meaning. A poor standalone body suggests an algorithm/layout/mainloop problem. A good standalone body that becomes poor in the entry suggests resource or participation incompatibility. A good entry that loses the complete call suggests handoff, scheduling, setup or an already strong baseline. A new family's need for checkpoint-name branches means the indexed semantic/body abstraction has failed its intended breadth.

## 8. Uncertainty that remains

No document can establish cuBLAS-relative body quality or strict win rate without the actual GPU, model, context, compiler and library paths. The historical H200 MIG results cannot be transplanted to another target. This architecture improves the odds by making the bottleneck measurable and addressable early, while leaving a clear result if the one-grid constraint proves too costly for a declared cell.

## 9. The implementation consequence in this checkout

The repository's first V3 commits already supplied useful CPU contracts, diagnostics, reference fixtures, capture/normalization/facts, and named semantic matchers. The decisive missing bridge is still visible in `src/megabake/v3/frontend/semantic.py`: unmatched FX becomes a `ReferenceRegion` whose callable is the whole original program. It can be checked for correctness, but there is no generic indexed CUDA body for it. `composites.py` checks duplicate named operations rather than complete original-FX output/effect coverage. `layers.py` fingerprints candidate groups but does not encode executable per-iteration state/weight bindings. The [implementation inventory](MEGABAKE_V3_IMPLEMENTATION.md#2-what-the-current-repository-gives-us) and [IR migration](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#13-migration-from-the-currently-checked-in-semantic-frontend) specify what to retain and replace.

The first high-risk GPU experiment is also more constrained than a library name suggests. The current pinned PyTorch wheel is CUDA 12.4 and CUTLASS package is 3.8; the current cuBLASDx release documents a CUDA 13.0+ toolkit and CUTLASS 4.4.1+ requirement. [NVIDIA requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) Therefore a cuBLASDx win in an isolated newer toolchain is evidence for that provider and target, not an automatic V3 main-build body. A strong strict path still needs a target-qualified generated SIMT/MMA tactic family and whole-entry composition evidence.

No document can make a weak body fast by specifying it more carefully. The specification can ensure that an agent measures the vendor gap early, knows which algorithm/layout/resource variable to change next, and cannot hide a strict loss behind an unrelated launch or fallback. The [38 V3R work orders plus the HF capture split](MEGABAKE_V3_IMPLEMENTATION.md) make each research claim falsifiable at the point where it matters.
