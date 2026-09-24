# MegaBake V3: research provenance and decision ledger

Research date: 2026-09-09; revised 2026-09-24 for an explicit backend boundary. Original source
audit base: `3695f06de14322fd8ac3e111c693612d53b56319`; portability revision base:
`e84001d56df1535d5cc9d033cda435bd04f74630`.
Scope: read-only code/history/reference research plus new V3 documentation. No GPU execution,
model-weight download, compiler implementation, dependency change or benchmark repair was performed.

## 1. Evidence vocabulary

- **Observed in source:** directly inspectable in the pinned repository/checkout.
- **Reported measurement:** stated by a paper or existing project report; not reproduced here.
- **Documented capability:** described by the cited API/toolchain documentation.
- **Inference:** a conclusion from evidence, with assumptions stated.
- **Proposal:** a V3 design or future experiment, not an existing feature.

All performance statements in the V3 series are either historical reports or labelled mathematical
examples. There are no new V3 performance results. External repository state and versioned APIs
are recorded so a future implementation does not mistake moving `main` documentation for a lockfile.

## 2. How V2 evolved on this branch

The inspection covered the recent ten commits, slightly beyond the requested five to eight so
the original design additions were included.

| Commit | Date | Relevant change |
|---|---|---|
| `9548f03` | 2026-08-02 | New architecture/dataflow/IR planning set |
| `f3cca49` | 2026-08-05 | Tile IR added |
| `e6db728` | 2026-08-06 | Helion-style tuning added to the plan |
| `167a4f5` | 2026-08-07 | Further IR analysis |
| `bad4ee4` | 2026-09-05 | GraCE paper/analysis and substantial architectural revision |
| `e743aae` | 2026-09-07 | Dependency/environment work |
| `353f292` | 2026-09-07 | GPU-based revision, GPU reanalysis and implementation ledger |
| `725b1bf` | 2026-09-07 | Merge of rewrite work |
| `a712c46` | 2026-09-07 | Cleanup of older discussion documents |
| `3695f06` | 2026-09-07 | Architecture-only simplification and revised implementation references |

The final commit changed only `MEGABAKE_V2_ARCHITECTURE.md` (48 added, 179 removed lines). That
helps explain the confusion: the files no longer describe one consistently scoped next step.

### Concrete inconsistencies, not just a feeling of complexity

The latest architecture narrows the source strategy toward selected MPK/Hopper task references
and optional cuBLASDx. Meanwhile, the older
[IR/reuse plan](MEGABAKE_V2_IR_AND_REUSE_PLAN.md) still requires broad multi-family enumeration and
contains stricter external-persistent-system exclusion language. The
[implementation ledger](MEGABAKE_V2_IMPLEMENTATION.md) still mandates four-family searches, a
second A100 target before later composite work, six model families and an 80% strict-win release
matrix. Some are valid future product objectives, but they are not the same minimal experiment.

The GraCE revision also introduced a substantial external-graph/binding path alongside the
strict persistent path. That can be useful for a hybrid system; it does not resolve the later
measured one-grid body deficit. Maintaining both as equally immediate architecture commitments
expanded scope without closing the principal uncertainty.

V3 supersedes these choices as a coherent proposal. V2 and `grace_hack.md` remain historical
records, unchanged. This is not a claim that every V2 idea was wrong: much of the metadata,
resource, state and baseline discipline remains useful.

## 3. What the inspirations actually establish

### Mirage MPK

The [MPK paper, arXiv v1](https://arxiv.org/html/2512.22219v1), is a CMU-led collaboration, not the
Stanford Hazy project. It combines tile-region dependencies, fine-grained execution and cross-task
pipelining. Selected A100/H100/B200 serving experiments report up to 1.7x over compared systems;
their workloads and worker/scheduler allocations are not MegaBake defaults or guarantees.

Importantly, §6.6 reports a 1.2–1.3x cross-task-pipelining ablation for Qwen3-8B's final linear
layer on B200. That is a **region-level** benefit, not a whole-model prediction. It substantiates
the opportunity beyond launch removal, while leaving V3's exact body/stage integration untested.

V3 adopts explicit readiness and staged movement in its logical and target execution-plan
contracts, with generated bounded schedules rather than requiring the entire MPK runtime. Current
source integration is audited in
[kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md#5-mirage-mpk-useful-device-code-not-a-free-generic-fx-frontend).

### Stanford Hazy / ThunderKittens megakernel work

[Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) demonstrates a
hand-tuned Llama-1B decoding design with weight prefetching and fine-grained readiness. It reports
sub-millisecond H100 decoding and approximately 680 us on B200 under its setup. This is evidence
that benefits beyond host launch removal exist when math and scheduling are carefully engineered.
It is not evidence that a generic compiler automatically reproduces those results.

The distinction matters: Hazy's work and MPK are related inspiration, but should not be treated
as one Stanford project or one implementation that can simply be cloned into a generic FX backend.

The revision additionally cloned and inspected Hazy's source, including its entry/configuration,
matvec stage interfaces, gate/up, additive down projection, page-release ordering and latency
scheduler. The [source audit](MEGABAKE_V3_KERNEL_REUSE.md#hazys-staged-implementation-an-additional-reference)
records the concrete capability and compatibility distinctions. In particular, numerical order,
volatile polling and a 640-thread default are not inherited as V3 correctness/performance rules.

### Luminal

The [historical megakernel article](https://blog.luminal.com/p/compiling-models-to-megakernels)
explores block-level composition. The current pinned backend was also inspected, because a blog
and a moving implementation can differ. Its host/kernel/block distinction and preservation of
legal alternatives are the useful design takeaways. Its existing external-library/graph paths
must not be counted as proof that every selected operation becomes one in-grid body.

V3 does not require adopting Luminal's whole search engine. Preserve legal alternatives and test
a bounded set first. The specific current files are linked in [kernel reuse §6](MEGABAKE_V3_KERNEL_REUSE.md#6-luminal-preserve-alternatives-and-distinguish-execution-levels).

### GraCE

The published [OSDI 2026 paper](https://www.usenix.org/system/files/osdi26-ghosh.pdf) addresses CUDA
Graph applicability and replay costs. Its lessons inform the baseline and binding contract. The
vendor path is not a mechanism for embedding private vendor code into MegaBake's grid; the
[reuse analysis](MEGABAKE_V3_KERNEL_REUSE.md#2-why-the-cublas-extraction-route-does-not-solve-composition)
records the API distinction and unresolved reproduction detail.

The supplied local PDF was also text-inspected:
[local GraCE paper](<GraceCE_OSDI_26_CudaGraphPytorchCompile (1).pdf>).
Its SHA-256 is `163af52d11cf0d7d8bbe9b164fdc520f711c8a4453ba4db81b5597ad2947b2c8`.
The published PDF and local file are separate artifacts; identical filenames/titles would not
establish byte-identical versions. The prior [grace_hack.md](grace_hack.md) was reviewed as project
analysis, not elevated above the primary API contract.

### Inferact TPU megakernels and the portability boundary

The later [Inferact TPU megakernels repository](https://github.com/Inferact/tpu-megakernels/tree/aa0094ef9add6a1f21b1697fc7371ffccf68e8ea)
was cloned and source-inspected after the original V3 planning commit. Its single published commit
postdates the 2026-09-09 plan. It contains Pallas/Mosaic fused decode implementations for a
multi-host 32-device Kimi configuration and an eight-device Qwen configuration. The kernels use
explicit HBM/VMEM staging, DMA semaphores, remote copies/collectives, state aliases and target
layouts. CPU/interpreter tests do not establish TPU lowering, capacity, scheduling or performance,
as its own README states.

This source is evidence that the semantic ideas behind V3—stateful full-step composition, explicit
staged movement, bounded scratch, continued reductions and fused collectives—are not inherently
CUDA-only. It is also evidence against pretending CUDA execution vocabulary is portable: TPU
TensorCore/mesh placement, VMEM and semaphore/remote-DMA contracts are not warps, CTA shared memory
or a cooperative grid. The repository is two hand-specialized implementations, not a generic FX
compiler or a drop-in MegaBake backend.

The distinction is consistent with JAX's primary [TPU pipelining documentation](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html),
which describes HBM/VMEM/SMEM and semaphore-tracked movement, and its
[distributed Pallas documentation](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html),
which describes mesh placement and remote DMA. These moving APIs would need version pinning and
device tests before a future adapter; they are architecture evidence, not a V3 dependency.

The resulting design decision is to separate `LogicalExecutionPlan` from backend-qualified
`TargetExecutionPlan` now, while implementing only CUDA. This is a narrower response than adding a
universal Tile IR or TPU work to the milestone: it preserves exact actions, footprints and lifetime
obligations above the adapter and keeps physical bodies, spaces, synchronization, topology,
codegen and runtime below it. Portability remains unclaimed until a second adapter is implemented,
verified and measured.

### vLLM's proposed semantic dialect

The [referenced RFC](https://github.com/vllm-project/vllm/issues/32358) is useful for keeping
functional reference semantics while choosing implementations later. V3's FatOps follow that
idea inside FX. The issue is a design reference, not evidence of a stable reusable V3 frontend.
Its status/project labels are not used as a technical correctness argument.

## 4. Source snapshots and reproducibility

Reference repositories were cloned outside the project into a temporary research directory.
They are not vendored dependencies and are not part of the V3 file changes.

| Reference | Inspected revision | Source entry |
|---|---|---|
| MegaBake | `3695f06de14322fd8ac3e111c693612d53b56319` | Current repository before V3 additions |
| MegaBake V3 portability revision base | `e84001d56df1535d5cc9d033cda435bd04f74630` | V3 planning commit before this documentation edit |
| Luminal | `d18376d184172616ab3980309f524299a02595ef` | [Pinned tree](https://github.com/luminal-ai/luminal/tree/d18376d184172616ab3980309f524299a02595ef) |
| Mirage | `17e9e36de583fe26a55be3fa0a6030f5c56a34d8` | [Pinned tree](https://github.com/mirage-project/mirage/tree/17e9e36de583fe26a55be3fa0a6030f5c56a34d8) |
| Hazy Megakernels | `7309cec801537b61fea3b50d7dfe454a6cde578e` | [Pinned tree](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e) |
| Inferact TPU megakernels | `aa0094ef9add6a1f21b1697fc7371ffccf68e8ea` | [Pinned tree](https://github.com/Inferact/tpu-megakernels/tree/aa0094ef9add6a1f21b1697fc7371ffccf68e8ea) |
| PyTorch flow/IR | `v2.6.0` | [Inductor source](https://github.com/pytorch/pytorch/tree/v2.6.0/torch/_inductor) |
| cuBLASDx pipeline/requirements | `0.7.1` docs | [Versioned requirements](https://docs.nvidia.com/cuda/cublasdx/0.7.1/requirements_func.html) |
| MPK paper | `2512.22219v1` | [Versioned paper](https://arxiv.org/html/2512.22219v1) |
| Gemma semantic example | Transformers `v4.50.0`; example, not the benchmark pin | [Norm and configurable gated MLP](https://raw.githubusercontent.com/huggingface/transformers/v4.50.0/src/transformers/models/gemma/modeling_gemma.py) |
| HF config / Transformers model source | Moving main, accessed 2026-09-09 | [Config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json), [model code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |

Pin the last row to a checkpoint/code revision before implementation or model benchmarking. The
current research only needed its metadata/semantic distinction; it did not load model weights.

### Additional primary documentation used

The [hardware model](MEGABAKE_V3_HARDWARE_MODEL.md) links CUDA property/cooperative APIs, architecture
tuning guides, MIG profiles and relevant memory restrictions alongside their claims. The
[kernel reuse document](MEGABAKE_V3_KERNEL_REUSE.md) links device math and launch APIs alongside
their contracts. The [IR plan](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) links pinned PyTorch flow, export,
the exact HF config and model semantics. This keeps citations near their evidence instead of
making an unannotated bibliography stand in for an argument.

The pipeline revision also checked CUDA async-copy and pipeline contracts, CUTLASS stage
acquire/release, PTX bulk-store completion, and CCCL execution/memory scopes. Direct citations
accompany the proposed [stage, publication and progress contracts](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md).
CCCL's cited `unstable` pages and unversioned CUDA/CUTLASS pages are accessed 2026-09-09; they are
research references, not evidence that the historical toolkit supports every documented feature.

## 5. Decision ledger

| ID | Decision | Basis | Revisit when |
|---|---|---|---|
| D01 | Preserve strict one-grid research objective; separate fallback score | User goal and meaningful result accounting | User explicitly changes objective |
| D02 | Handoff in normalized FX before GraphLowering | Pinned source; later IR is not optimized ATen | A concrete later pass provides essential reusable value |
| D03 | FatOps within FX with reference expansions | Semantic selection without another graph framework | FX cannot express a required transformation cleanly |
| D04 | LayerSummary is analysis, not mandatory IR | Layer names alone add no executable semantics | Structured iteration/state transformation is needed |
| D05 | Backend-qualified TargetProfile separates legality from measured costs | APIs/manuals do not reveal achieved performance | Two implemented backends require a richer shared property model |
| D06 | Primary bounded pipelined regions; barrier entry is a control | User objective and concrete source mechanisms; old queue/static timing is not a pipeline ablation | Measurements select a coarse schedule for a particular cell |
| D07 | Explicit tile stages, readiness and reduction footprints now | Consumers can start on partial tensor regions; movement can start on address readiness | Never remove without replacing the information |
| D08 | Logical tiles independent of worker residency | Current source conflates them; shape evidence contradicts it | Fundamental invariant, not a tuning preference |
| D09 | Small GEMV/tensor-core portfolio; cuBLASDx early candidate | Current mapping deficit; library APIs are viable but constrained | Specific hot shapes justify more tactics |
| D10 | Whole-entry compile/resource validation | Isolated resources/performance do not establish composition | Fundamental correctness/performance requirement |
| D11 | No private cuBLAS binary extraction on critical path | Handles/preludes are not device bodies | Separate funded research question with explicit need |
| D12 | Stateful cached decode is the first meaningful model target | Current wrapper disables cache | Different explicitly declared workload becomes primary |
| D13 | First win on one available GPU before portability expansion | Current uncertainty is viability, not release breadth | First useful result or explicit portability requirement |
| D14 | Numerical/state validation precedes accepted timing | Existing error-only metrics are insufficient | Never waive to manufacture speedup |
| D15 | No unconditional X multiplier or universal win-rate promise | Counterexamples and break-even model | Report actual measured matrix, not a theorem |
| D16 | Joint bounded body/fusion/tile/layout/schedule selection | Isolated body or greedy cover choices can preclude overlap | A measured decomposition of decisions is equally effective |
| D17 | Separate publication, source retirement and finalization | Async stores and continued reductions have different milestones | Fundamental correctness contract |
| D18 | Static staging slots and region-owned activation chunks initially | Bounded overlap without a general page allocator or multi-consumer ring | Storage pressure measurably warrants finer reclamation |
| D19 | First block tests all three pipeline patterns, with controlled ablations | One-grid count does not establish non-launch benefits | Never label an untested mechanism a success |
| D20 | Progress proof includes worker order, resources and cooperative support | Data-DAG acyclicity alone is insufficient | Fundamental legality requirement |
| D21 | Split logical and target execution plans; keep search jointly target-aware | TPU source confirms common action/lifetime ideas but incompatible physical execution vocabulary | The split duplicates decisions without enforcing a boundary |
| D22 | Define a narrow BackendAdapter now; implement only CUDA in V3 | Avoid CUDA leakage without expanding the first performance experiment | A second implemented backend needs a richer proven contract |
| D23 | Portability is not a V3 success claim | Neutral schemas and mocks do not establish another compiler/runtime/device path | A second backend passes device correctness and measurement gates |

## 6. Alternatives deliberately deferred

| Alternative | Why not first? | Useful future role |
|---|---|---|
| Full Inductor loop-IR interception | Inherits lowering choices and loses easy high-level semantics | Reuse a specific proven pass through an adapter |
| Standalone Layer IR | No required new transformation yet | Structured repeated/stateful regions |
| General instruction-level Tile IR / hardware DSL | Logical/target plan split supplies the present seam without modeling either ISA | Two implemented backends expose shared instruction-level transformations that adapters cannot express cleanly |
| Mandatory equality saturation | Larger optimizer than the first candidate set needs | Complex overlapping rewrite search |
| MPK's full queue/runtime and general allocator | Bounded V3 stages/events already supply initial overlap | Measured dynamic imbalance or storage limits beyond static templates |
| Mandatory hybrid planner and GraCE binding layer | Does not solve current strict body deficit | Product fallback with costly dynamic bindings |
| Four-source universal kernel tournament | Multiplies work before exact hot shapes are addressed | Broader portfolio once infrastructure is useful |
| Six-model/two-target release matrix first | Confuses product breadth with first research validation | Later release confidence |
| Quantization/batching as the initial rescue | Changes precision or workload | Separate throughput/memory optimization track |

These are sequencing decisions, not claims that the alternatives are inherently bad.

## 7. Remaining uncertainties

- Whether any selected tensor-core library body is competitive after the common launch/resource
  contract, especially at M=1 and on the next available GPU.
- Whether cuBLASDx's current pipeline metadata and repeated-tile lifetimes fit a multi-layer entry
  economically; this is a compatibility experiment, not implemented support.
- Which head/chunk sizes, producer/consumer assignments and lookahead depths reduce complete
  latency after bandwidth contention and whole-entry resource costs.
- Whether owner-held down-projection accumulators fit economically, or materialization/explicit
  partial finalizers are better for some shapes under the agreed numerical policy.
- How much real cached attention changes the relative costs for the actual small and 2B models.
- The exact historical checkpoint, raw traces and numerical policy behind the user's recalled
  speedups; available source/prose cannot reconstruct all of them.
- Whether later code-size/resource effects require structured loops or additional outlining.

None of these is resolved by a more detailed architectural diagram. Each has a corresponding
experiment in [the implementation plan](MEGABAKE_V3_IMPLEMENTATION.md).

## 8. Bottom line

The main idea is sound as a research direction: keep generic graph semantics, recognize useful
computations, and compose high-quality device implementations with better coordination. The
current evidence does not establish that a universal megakernel beats a strong compiled baseline.

The recommended simplification is two semantic compiler representations followed by logical and
target forms of one compute-and-movement plan with bounded schedule templates. It is **not**
removing the stage/readiness information needed for megakernel benefits or inserting an unrelated
graph IR. FatOps supply semantic choices; joint planning and stage-capable target bodies turn those
choices into useful fusion and overlap. Extra hierarchy, binary extraction and a large release
framework do not substitute for either competitive math or effective execution.

## 9. What this revision corrects

The earlier V3 architecture treated static phases as the first backend and pipelines as a later
response to measured tails. That sequencing was too restrictive for the user's objective, and
the historical queue/static result did not justify it. This revision changes architecture, plan
schema, device-body capabilities, memory/progress contracts, hardware calibration, diagrams,
performance objectives and experiments together. It is not just a stronger milestone list.

The new [pipeline specification](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) provides bounded
head-ready attention, streamed gated MLP and weight-lookahead designs with explicit rejection
conditions. The performance model adds action-level calculations, optimized controls, interference
and conditional size/rate scenarios. None is described as implemented or GPU-validated.

The 2026-09-24 portability revision additionally separates logical obligations from physical
target realization, generalizes the TargetProfile contract, and moves CUDA source/compiler/runtime
ownership behind a named adapter. It deliberately leaves all device milestones and performance
gates CUDA-only. The Inferact TPU source motivates the seam; it is not incorporated code or TPU
support in MegaBake.

The remaining uncertainty is profitability and implementation correctness, not whether these
mechanisms have an architectural home. That is the strongest conclusion the present evidence
supports without manufacturing performance results.

The original 2026-09-09 revision recorded checks over all 11 V3 files, the three original research
checkouts and the numerical examples/tables. The 2026-09-24 backend-boundary revision rechecked its
changed local links/anchors, fenced-block balance and whitespace, and pinned the separately cloned
TPU repository revision. These are documentation/analysis checks, not GPU/TPU tests, a
rendered-diagram test or a formal proof of the proposed implementation.
