# MegaBake V3: research provenance and decision ledger

Status: source-informed architectural decisions, revised 2026-09-27. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) is normative. This document separates observations, inferences and unmeasured proposals. No cited external result is a MegaBake V3 benchmark.

## 1. Evidence vocabulary

| Label | Meaning |
|---|---|
| `SOURCE` | Directly inspected paper, documentation, code or tracked repository artifact |
| `HISTORICAL` | Prior MegaBake/V2 report or user recollection with stated provenance |
| `INFERENCE` | Reasoned architectural consequence, not measured V3 behavior |
| `HYPOTHESIS` | Proposed tactic or schedule requiring compilation and target measurement |
| `V3_MEASURED` | Reserved for raw retained V3 device/correctness/performance evidence; none is supplied by these documents |

The original V2 source/evidence audit used repository base `3695f06de14322fd8ac3e111c693612d53b56319`; the later backend-boundary revision used `e84001d56df1535d5cc9d033cda435bd04f74630`. Current implementation work must inspect its checkout rather than treating those hashes as present code. Historical GPU data remains in the [V3 evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md).

## 2. What the primary work establishes

### Mirage MPK

[MPK v2](https://arxiv.org/html/2512.22219v2) lowers tensor programs to SM-level tasks, derives dependencies from overlapping regions, compresses events and generates task CUDA bodies using block-level superoptimization. Its runtime combines ahead-of-time and just-in-time dispatch. The paper reports event fusion reducing event count by 37–118x for its evaluated models and notes that register use of a mixed mega-kernel is fixed by its task types. The inspected [Hopper CUTLASS-based task](https://github.com/mirage-project/mirage/blob/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh) is device-callable and uses a small-batch operand orientation with target-specific TMA/WGMMA roles.

`INFERENCE`: task-body generation is as central as task scheduling. Exact access maps should be expressed symbolically before thousands of task instances are materialized. Reusable device-callable bodies are possible, but MPK does not prove a generic FX contraction equals the best cuBLASLt tactic. Its queue/runtime is a candidate model, not a requirement for dense V3.

### Hazy Research

The [low-latency Llama-1B report](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) describes hand-authored instruction families and a static model-shaped schedule with overlapping weight loads. Its B200 runtime breakdown reports approximately 250 us for activation store/consistency/reload, 200 us for RMSNorm/matvec work, 30 us waiting for weights and 40 us low-level warp synchronization in one 600 us forward. It reports CUDA cores preferable to tensor cores for its Hopper low-batch work and a different marginal tensor-core choice on Blackwell. Its [open implementation](https://github.com/HazyResearch/Megakernels) illustrates staged load, compute, store and retirement roles.

`INFERENCE`: activation ownership and handoff belong in the same search as weight prefetch. Tensor-core preference changes by target and shape. Handwritten instructions demonstrate a fast physical design, not a generic HF FX compiler or an assurance that their body can be dropped into V3's block configuration.

### Luminal

The [megakernel article](https://blog.luminal.com/p/compiling-models-to-megakernels) describes block operations with symbolic launch/data/barrier expressions and a dynamic global queue. It derives fine-grained barriers from producer/consumer tile regions. The inspected [CUDA-lite source](https://github.com/luminal-ai/luminal) also distinguishes opaque host operations, launchable kernels and composable block operations; a kernel-level GEMV is not automatically a device-callable block body.

`INFERENCE`: use parametric tile domains and compact dependence relations; choose static or dynamic physical scheduling by workload. A universal queue incurs traffic and cannot be the generic-coverage mechanism. The article's design is not a broad matched scorecard for every model/SM.

### Inferact TPU megakernels

Inferact's [Kimi K3 TPU v7 post](https://inferact.ai/blog/tpu-megakernels) and [source](https://github.com/inferact/tpu-megakernels) describe a hand-specialized Pallas program over 92 layers, VMEM lifetime control and DMA of next-layer weights before their activations are available. Its 16-chip topology drives its sharding and collective choices. The reported headline comparison uses TPU versus GB200 serving configurations; speculative numbers include the draft algorithm.

`INFERENCE`: recover verified repeated/stateful regions and expose early-address/late-activation dependence in the common compiler. Let each backend generate its own physical memory and schedule program. TPU VMEM capacity and gridless sequential execution cannot be copied into a CUDA CTA abstraction. The cross-hardware serving result does not establish a same-device torch.compile win for MegaBake.

### NVIDIA and PyTorch contracts

The [cuBLASDx pipelined API](https://docs.nvidia.com/cuda/cublasdx/using_pipelines.html) exposes global-memory GEMM staging with host-created handles, required block configuration and descriptor/lifecycle constraints; the [requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html) currently include CUDA 13.0+. [CUTLASS collectives](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) provide a lower device-code reuse boundary. [CUDA feature-set rules](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) distinguish baseline, architecture-specific and family-specific code. The historical [PyTorch 2.6 flow](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/compile_fx.py) supports an early functional-FX handoff but no stable promise of “all optimizations before fusion.”

`INFERENCE`: cuBLASDx is an important experiment, not a universal cuBLAS-quality answer or mandatory provider. The CUDA body compiler and whole-entry cache must be feature/toolchain qualified. PyTorch normalization remains a version-pinned adapter, while original FX is the semantic oracle.

## 3. Decision ledger

| ID | Decision | Reason | Revisit when |
|---|---|---|---|
| D01 | Retain one owned compute grid as the strict CUDA research result | User north star; honest result accounting | User changes the qualifying output contract |
| D02 | Accept complete FX/ExportedProgram plus explicit state/invocation contract | FX fragments omit full-step effects and costs | A public capture route supplies an equivalent complete contract |
| D03 | Replace FatOp-first `SemanticGraph` with `IndexedTensorProgram` | Generic arrangements of supported operations need strict codegen without family matchers | Indexed vocabulary demonstrably cannot express needed semantics compactly |
| D04 | Keep named patterns as guarded algorithm choices | Specialized attention/norm/MLP algorithms are valuable; exact reference must survive | A proven more general equivalence engine replaces bounded rules |
| D05 | Promote verified repeated/stateful regions | Cross-layer movement and code-size planning require explicit iteration/state | Flat graph proves equally expressive and manageable |
| D06 | Generate a generic device body path for supported indexed work | Strict coverage cannot depend only on manually supplied bodies | A different proven generator provides the same coverage |
| D07 | Put vendor-competitive body tactic generation on the critical path | Historical 4–6x hot-body gaps can swamp launch savings | Exact target data shows another dominant bottleneck first |
| D08 | Keep a Pareto set of body, tile, layout and stage alternatives | Fast isolated kernels may damage mixed-entry residency or readiness | Measurements show a safe narrower ranking |
| D09 | Evaluate cuBLASDx pipelined, CUTLASS-derived and owned SIMT tactics by exact shape and SM | No single provider guarantees private cuBLAS parity | One provider proves both quality and composition across target matrix |
| D10 | Keep logical indexed dependence target-neutral and let backend originate schedules | CUDA cohorts/lookahead cannot express TPU gridless/VMEM programs generally | Multiple implemented backends justify a different shared seam |
| D11 | Start dense CUDA with static resident programs and barrier control; add dynamic/hybrid when needed | Predictable decode avoids queue tax; irregular MoE needs another option | Measured dense imbalance or queue benefit changes selection |
| D12 | Make head-ready, streamed-MLP and weight-lookahead mechanisms optional measured candidates | The primary goal is complete-step speed; each can worsen body quality/resources | A workload proves one is required for correctness, not merely performance |
| D13 | Check actual compiled entry resources and cooperative admission | Registers/shared/code and block roles are whole-entry constraints | Fundamental invariant |
| D14 | Separate strict grid and vendor-preserving external plans | cuBLAS host APIs remain strong but are separate grids | Qualifying result definition changes |
| D15 | Rank by complete-call measured latency against strongest equivalent torch.compile | Kernel counts and isolated-body sums are insufficient | Fundamental performance contract |
| D16 | Require guarded numerical/state equivalence before accepted timing | Rewrites, padding and split-K may alter behavior | Fundamental correctness contract |
| D17 | Qualify code and tuning by SM feature set, CUDA/toolchain/provider versions | `_a`/`_f` compatibility and resource results vary | Target documentation establishes a broader safe key |
| D18 | First implement a thin body-quality/FX-block/full-step vertical slice | A large planning framework cannot rescue noncompetitive math | Measured early slice justifies expansion |
| D19 | A portability claim waits for a real second backend | Neutral types alone are not executable TPU support | Second backend passes correctness and measurement |
| D20 | Generate bounded body schedule families from indexed semantics | A hand-written model-kernel registry would preserve the original breadth problem; local shape/target search attacks the cuBLAS gap | Generated candidates prove too costly or a stronger reusable generator appears |
| D21 | Split tiny FX capture from real HF full-step capture | Generic semantics can progress while model export issues are diagnosed; a tiny fixture cannot be mistaken for benchmark coverage | A single verified capture path makes this split unnecessary |
| D22 | Keep current cuBLASDx in an isolated versioned provider lane | Checked-in CUDA12/CUTLASS3.8 pins do not establish its CUDA13/CUTLASS4.4.1 requirements | Main environment is deliberately migrated and remeasured |
| D23 | Predeclare all primary `must_win` workload cells and fresh validation trials | “Consistently beats” otherwise becomes movable after seeing favorable shapes | A different explicit claim/metric is chosen before tuning |

## 4. Deliberately deferred alternatives

| Alternative | Why deferred | Potential role |
|---|---|---|
| Universal equality saturation or full MLIR migration | Larger infrastructure than bounded indexed choices initially require | Search complexity or new transformations justify it |
| Binary extraction/rewrite of private cuBLAS kernels | Does not yield a stable device ABI; substantial correctness research | Separate project with explicit budget |
| Mandatory global queue/paged allocator | Dense low-batch schedule may be cheaper without it | Irregular work or storage pressure measured on target |
| Long-lived multi-token controller | Full cached step and state need proof first | Later generation-level scheduling |
| TPU/multi-GPU backend implementation | CUDA strict feasibility and body quality are current risks | Explicit future target program |
| Quantization or weaker precision as a rescue | Changes the declared comparison | Separate numerically matched workload |
| Per-checkpoint FatOp/kernel files | Does not scale to new model arrangements | Never a default coverage path |

## 5. Unresolved falsifiers

1. Can any legal device-callable low-batch tactic approach the selected vendor kernels **inside** a mixed persistent entry on more than one SM generation?
2. Does one grid retain enough resource headroom for different body roles without large spills, block-shape compromise or code-cache pressure?
3. Can indexed lowering cover a held-out HF family's familiar primitive arrangements without a new model-name branch?
4. Do activation ownership, head readiness or cross-layer loads save more than their publication, scratch and body-quality costs?
5. What strict win rate remains when the baseline uses its best equivalent capture/autotune path and real cache state advances?

Each question has an explicit experiment in the [implementation plan](MEGABAKE_V3_IMPLEMENTATION.md). A strict loss is evidence that selects the next body or physical-plan investigation, not permission to relabel an external fallback as a megakernel win.
