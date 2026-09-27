# MegaBake V3: FX-to-megakernel compiler design

Status: revised architecture and implementation handoff, 2026-09-27. These documents specify intended behavior. The checkout has partial CPU V3 capture/contracts/matchers but no demonstrated indexed strict full-step compiler or measured V3 win. Historical V2 measurements are labelled separately in the [GPU evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md).

## North star and first scope

Accept a complete Torch FX `GraphModule` or `ExportedProgram` for a Hugging Face Transformers inference step. For supported, guarded cells, generate one owned CUDA compute grid that preserves the declared numerical, output and state contract and beats the strongest equivalent `torch.compile` execution on the same device. CUDA is the first backend. The shared semantic and dependence contracts remain usable by a later accelerator backend; they do not imply that a TPU backend exists.

The first measured regime is single-GPU, cached autoregressive decode at batch one and small batch with FP16/BF16 policies and fixed or guarded context buckets. Prefill, recurrent hybrids, MoE, multimodal models, quantization, distributed inference and multi-token serving controllers are separate workload extensions. A direct FX entry is primary. A Hugging Face helper may capture and bind a full step, but a graph-break fragment cannot count as a full-model megakernel.

Report three levels separately: **captured full step**, **correct strict one-grid compilation**, and **measured strict win**. A vendor-preserving multi-grid plan or ordinary compiled fallback can be a useful product result; it is not a strict win. Universal speedup for every FX graph, model and GPU is not a theorem.

## The revised center of the architecture

```text
FX/ExportedProgram + invocation/state/numerical contract
  -> conservative normalization + original executable reference
  -> IndexedTensorProgram: maps, reductions, contractions, indexing,
     state effects, guarded control and verified repeated regions
  -> guarded algorithm/layout alternatives (named patterns are optional)
  -> LogicalExecutionPlan: parametric tile work and exact dependencies
  -> backend body tactics + target-specific physical-plan search
  -> compiled/admitted artifact and measured selector
```

The compiler does not require a new `FatOp` for a new arrangement of supported primitive operations. A named attention, norm or gated-MLP recognition proposes a faster algorithm with an exact guard and reference expansion. A genuinely new mathematical primitive still needs a semantic contract; competitive performance may need a new reusable algorithm provider. The default structured-compute lowering establishes strict coverage where its primitive subset is supported, not an automatic performance win.

The CUDA backend searches **algorithm, body, tile, layout, ownership and schedule together**. A body compiler generates bounded target-qualified schedules for supported indexed work; reusable hardware providers supply SIMT/MMA building blocks rather than checkpoint kernels. It preserves several device-callable candidates instead of promoting only the fastest isolated kernel. The first critical experiment is whether embedded SIMT and tensor-core bodies can approach the selected cuBLAS/cuBLASLt tactics for hot exact shapes after full-entry resource costs. cuBLASDx is a measured candidate, not a promise of private cuBLAS parity. [Body-source contracts](MEGABAKE_V3_KERNEL_REUSE.md) and the [performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) define this gate.

Logical work is independent of a physical CUDA CTA or TPU TensorCore. The common plan records index/access maps, effects, reductions and legal readiness. CUDA chooses its worker program, storage, synchronization, body participants and invocation. A future TPU backend would make different physical choices from the same semantic program. [Inferact's TPU implementation](https://inferact.ai/blog/tpu-megakernels) motivates cross-layer lifetime and movement opportunities, not a CUDA-shaped common scheduler.

## Document map and authority

| Document | Owns |
|---|---|
| [Architecture](MEGABAKE_V3_ARCHITECTURE.md) | Product boundary, compiler stages, optimization search and strict-result contract |
| [IR and reuse](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) | Capture, indexed tensor semantics, algorithm alternatives, repeated regions and logical/target schemas |
| [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md) | Generated and adapted body providers, cuBLASDx/CUTLASS constraints and body-quality admission |
| [Pipelining and scheduling](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) | Optional physical schedules, typed readiness, storage, publication and progress |
| [Hardware model](MEGABAKE_V3_HARDWARE_MODEL.md) | Feature-gated SM/toolchain profiles, legality and measured target costs |
| [Performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) | Baselines, break-even analysis, measurements and claims |
| [Implementation](MEGABAKE_V3_IMPLEMENTATION.md) | Vertical work order, small reviewable tasks and evidence gates |
| [Dataflow diagrams](MEGABAKE_V3_DATAFLOW_DIAGRAM.md) | Compiler/search and execution relationships |
| [GPU evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md) | Historical V2 facts and limits of their interpretation |
| [Research and decisions](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md) | Primary-source lessons and decisions to revisit |
| [Deep research](MEGABAKE_V3_DEEP_RESEARCH_2026-09-27.md) | Research comparison and unresolved hypotheses |
| [FX walkthrough](MEGABAKE_V3_FX_TO_MEGAKERNEL_PROPOSAL_2026-09-27.md) | End-to-end transformation of a representative full-step FX graph |

The architecture owns decisions; the IR, body, pipeline, hardware and performance files own their respective contracts. The research and walkthrough documents explain those contracts rather than defining competing architectures.

## Immediate priorities

1. Capture a complete cached step, validate its reference meaning and inventory the strongest matched `torch.compile`/vendor paths at exact hot shapes.
2. Build a lean embedded K-parallel SIMT body and a target-supported tensor-core tactic. Compare standalone and minimal persistent-entry versions, including padding, descriptors and resources.
3. Lower a real FX block through the indexed tensor program and one legal CUDA entry. Keep a barrier schedule as a control; add fine-grained pipeline candidates only when their body interfaces support them.
4. Recover repeated structure and compile a correct full step. Search whole-region activation ownership and movement, then measure complete latency.
5. Evaluate a structurally different held-out Hugging Face family and, when hardware is available, another SM generation. A source build on another target is a legality check, not a performance claim.

A high-quality body is necessary but not sufficient. Hazy's [Llama-1B analysis](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) shows activation handoff and synchronization can remain substantial after weight stalls shrink. Mirage's [MPK compiler](https://arxiv.org/html/2512.22219v2) demonstrates the value of generating device task bodies and compact tile dependencies. Luminal's [block-domain approach](https://blog.luminal.com/p/compiling-models-to-megakernels) motivates symbolic work maps. None of those results substitutes for MegaBake's own same-device, same-contract measurements.

## Giving a coherent batch to an implementation agent

Start with the [implementation handbook](MEGABAKE_V3_IMPLEMENTATION.md#1-assignment-protocol-for-an-implementation-agent). Assign a bounded dependency-closed batch, usually two to four V3R cards, with one observable outcome, prerequisite handoffs and the documents named by the cards' traceability rows. A substantial GPU card can stand alone. The agent must inspect the real checkout, declare the batch before editing, add positive and negative cases for each card, run available checks, then return one `TaskHandoff` per attempted card and a short batch summary. CPU/source success cannot be reported as CUDA correctness or speed. The [IR document](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#10-concrete-records-an-implementer-must-be-able-to-construct) provides minimum records; the [architecture stage table](MEGABAKE_V3_ARCHITECTURE.md#10-compiler-handoff-contract-stage-by-stage) says where each record is consumed.

The first implementation route is G0 complete cached capture/baseline, G1 vendor-relative body quality, G2 generic indexed FX block, G3 correct one-grid full step, and G4 held-out family/target scorecard. Body probes can proceed before the whole frontend is ready, but their speed is local evidence until a complete equivalent step is measured. Optional head-ready attention, streamed MLP, prefetch and dynamic dispatch are measured candidate optimizations, not prerequisites for a first strict result.

Two facts deserve special attention in this checkout. First, `frontend/semantic.py` currently leaves unmatched FX as `ReferenceRegion`s; those are reference oracles, not strict CUDA coverage. Second, the pinned CUDA 12.4 PyTorch/CUTLASS 3.8 environment does not establish availability of the current cuBLASDx pipeline, whose documented requirements begin at CUDA 13 and CUTLASS 4.4.1. Treat it as a separately pinned provider experiment. [Implementation inventory](MEGABAKE_V3_IMPLEMENTATION.md#2-what-the-current-repository-gives-us), [NVIDIA requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html)
