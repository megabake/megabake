# MegaBake V3: research and design

Research date: 2026-09-09; backend boundary revised 2026-09-24. Original source audit base:
`3695f06de14322fd8ac3e111c693612d53b56319`; portability revision base:
`e84001d56df1535d5cc9d033cda435bd04f74630`.
Status: pipeline-first revision, source-audited; no V3 implementation or new GPU measurements.

The recommendation is to build a **target-extensible FX compiler with a small semantic dialect and
a specialized pipelined CUDA first backend**. CUDA is the only implementation and performance
target in V3. Keep the single-grid objective for that backend, but optimize tile-level
compute and movement, not just launches. Competitive math, useful fusion, early consumer
readiness and bounded cross-task weight staging are designed and selected together.

This revision replaces the earlier static-first V3 decision throughout the series. The phase
executor is now a control; pipelines are part of the first generated block, not a later milestone.

The central pipeline is:

```text
FX / ExportedProgram
  -> normalize, preserve semantics, collect tensor facts
  -> SemanticGraph: FatOps + ordinary ATen regions
       optional LayerSummary over the same graph
  -> LogicalExecutionPlan: tiles, footprints, staged actions and lifetime constraints
  -> jointly choose target body, layout, physical transport and schedule through BackendAdapter
  -> TargetExecutionPlan: resolved spaces, events, placement and invocation
  -> CUDA adapter generates one specialized cooperative entry
  -> compile, check resources, validate, measure, select
```

`TargetProfile` is a backend-qualified input to planning. It contains queried capabilities,
documented features, and measurements with provenance. It is not another program IR. The logical
plan contains no CUDA hierarchy, memory-space, primitive or launch names; the target plan is not
launchable until one adapter resolves all of them. `LayerSummary` is also an analysis, until a real
transformation needs structured layer/loop semantics.

## Read this first

| Document | Question it answers |
|---|---|
| [Architecture](MEGABAKE_V3_ARCHITECTURE.md) | What should we actually build, and what should we defer? |
| [Dataflow and diagrams](MEGABAKE_V3_DATAFLOW_DIAGRAM.md) | Where does each decision happen? |
| [IR and reuse plan](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) | What do FatOps, layer information, and execution plans mean? Where do we leave Inductor? |
| [Pipelining and scheduling](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) | How do ready heads, streamed MLPs, weight lookahead, safe storage and progress actually work? |
| [GPU evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md) | What do the existing numbers establish, and what is missing? |
| [Performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) | When can a megakernel win, and how large could that win be? |
| [Hardware model](MEGABAKE_V3_HARDWARE_MODEL.md) | How do we obtain and use the proposed hardware fact table? |
| [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md) | What can cuBLASDx, CUTLASS, MPK, and GraCE actually contribute? |
| [Atomic implementation handbook](MEGABAKE_V3_IMPLEMENTATION.md) | How do we execute the design through 92 bounded task cards with dependencies, before/after behavior, tests and evidence gates? |
| [Research and decisions](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md) | What changed from V2, which sources were inspected, and why? |

The architecture owns the decisions; the IR document owns representation contracts; the pipeline
document owns scheduling/lifetime protocols; the performance document owns formulas and measurement
definitions. Other files explain those contracts instead of
introducing parallel versions. V2 files remain historical records and were not edited.

For implementation assignments, use the handbook's [task-family index](MEGABAKE_V3_IMPLEMENTATION.md#6-atomic-tasks),
[agent handoff packet](MEGABAKE_V3_IMPLEMENTATION.md#8-ready-to-use-agent-assignment-and-handoff),
and [worked correctness examples](MEGABAKE_V3_IMPLEMENTATION.md#9-worked-contracts-and-adversarial-examples).
Assign one task plus its prerequisite handoffs; GPU-unvalidated work is not a completed device gate.
The last four tasks are measurement-triggered alternatives, not mandatory expansion. The plan
changes documentation only; it does not claim that these interfaces or tests already exist.

## Six conclusions that matter

1. The current code has concrete mapping and resource problems. More IR layers will not repair
   its serial per-output K reduction or its universal shared-memory reservation.
2. The latest V2 report gives approximately `4.85 ms` for MegaBake and `1.78 ms` for an exported
   low-overhead compiled baseline. Those are historical prose reports, not newly reproduced
   results. Their raw real-model benchmark artifacts are absent from the tracked tree.
3. The checked-in “decode” wrapper uses `use_cache=False`. Sequence-one results do not establish
   performance for cached autoregressive decoding with a growing context.
4. FatOps are useful when they preserve semantics and enable implementation choice. A layer name
   alone does not establish those semantics. FlashAttention is an implementation of attention,
   not a peer semantic category to full attention.
5. A private `CUfunction` does not provide an in-grid device body. cuBLASDx and adapted
   CUTLASS/CuTe code can provide such bodies, subject to their execution contracts.
6. Extensibility is a checked boundary, not a second implementation. Common semantic, footprint,
   lifetime and dependency contracts are target-neutral; CUDA owns its bodies, physical plan,
   code generation and runtime. A future backend supplies those pieces and must earn its own
   correctness/performance evidence.

Evidence and primary references are attached to the detailed claims in the linked files.

## Where the non-launch gains come from

| Mechanism | Concrete architectural support |
|---|---|
| Useful fusion | Joint composites/layouts; paired gate/up gating, compatible epilogues and positional/cache writes |
| Earlier computation | Published head/K-chunk readiness; attention and continued down-projection updates |
| Better data movement | Body load/compute stages; bounded next-task/layer weight lookahead with explicit storage leases |
| Reduced tails | Consumer-aligned tile sizes and balanced worker/cohort schedules with provable progress |

These mechanisms are candidates selected by net latency, not switches forced on every model.
The [performance calculations](MEGABAKE_V3_PERFORMANCE_MODEL.md#9-conscious-model-size-calculations-not-promises)
show how improved effective execution can matter even when launch savings become negligible.

## What confidence is possible now

There is no defensible unconditional “V3 will beat torch.compile by X” claim. A compiler cannot
strictly beat every input: a graph already reduced to one excellent kernel can offer no removable
launch or intermediate work. We can prove transformation and synchronization properties, derive
conditional break-even inequalities, and measure a supported workload matrix later.

The useful promise is more concrete: every proposed optimization has a named source of savings,
a cost to measure, and a rejection condition. The first performance milestone is a correct,
generated cached-decode step that wins on both a small model and a roughly 2B model on one GPU.
It also needs measured non-launch improvement over a matched phase control, with ablations
distinguishing fusion, matrix-data lookahead and tile readiness. The backend seam is defined now;
implementing or claiming portability follows that result. There is no calibrated per-model-size
forecast without a GPU.

The initial objective retains the model's FP16 or BF16 policy. Quantization, continuous batching,
multi-GPU execution, and recurrent hybrid models are separate extensions. They do not substitute
for proving the requested reference-precision megakernel objective.
