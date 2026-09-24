# MegaBake V3: pipelining, fusion, movement and progress

Status: normative proposed design, 2026-09-09; backend boundary revised 2026-09-24; no
implementation or GPU validation. This document
owns the scheduling/lifetime protocol. [Architecture](MEGABAKE_V3_ARCHITECTURE.md) owns scope,
[IR](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) owns the plan schema, and
[performance](MEGABAKE_V3_PERFORMANCE_MODEL.md) owns the quantitative model and ablations.

## 1. The optimization unit

An operator is too coarse for readiness. A machine instruction is unnecessarily fine for the
common planner. Use a **logical body tile with declared capabilities and exact footprints**.
`LogicalExecutionPlan` represents target-independent actions and conditions;
`TargetExecutionPlan` binds them to device-body stages and mechanisms. Neither is a general
machine-instruction IR.

A tile can publish less than an entire tensor, but never more than it has actually computed.
A load can start before all computation inputs are ready when its own address/data dependencies
are ready. Output visibility and scratch reuse are different events. These three distinctions
are the minimum information needed to optimize more than launches.

The common planner supports finite, statically bounded pipelined regions. Templates generate
partial orders, placement relations and producer/consumer cohort requirements from graph structure
and shape facts. They are not hand-written engines selected by checkpoint name. A backend resolves
them to target worker programs only when its execution/progress model supports them. The first CUDA
backend retains a barrier-control lowering for every supported region. A generic task queue,
distributed work stealing, paged scratch allocator and multi-token controller are not prerequisites.

## 2. Actions and capability contracts

The names below describe compiler contracts, not existing callable APIs:

| Action | Prerequisites | Completion/token |
|---|---|---|
| `reserve(slot)` | Prior users released the slot; capacity/alignment legal | Exclusive staging lease |
| `preload(tile, slot)` | Address/data dependencies and lease; legal descriptor | Copy-issued, then load-complete |
| `compute(tile)` | Required inputs acquired; local loads complete; participants available | Declared compute outputs complete |
| `reduce_begin(owner)` | Accumulator storage available | Initialized accumulator continuation |
| `reduce_update(owner, k_chunk)` | This chunk ready; predecessor update complete | Updated, still private accumulator |
| `reduce_finalize(owner)` | All required chunks incorporated exactly once | Final cast/epilogue and output |
| `publish(region)` | Writers joined; required stores visible at the selected target scope | Cross-worker data-ready token |
| `release(slot)` | Every operation accessing this storage has finished its access | Slot can be overwritten |
| `join(region)` | All required work, readers and async operations completed | Safe coarse boundary/reuse point |

Capabilities compose rather than form a performance ranking:

- `ATOMIC_TILE`: one indivisible call; the tile's declared memory accesses are drained at return.
  It can participate in head-ready scheduling between calls, but cannot be interrupted mid-call.
- `PRELOADABLE`: load/compute interfaces accept separately owned staging and expose completion.
  This can support cross-task weight lookahead, not just internal K-loop double buffering.
- `STREAM_REDUCTION`: begin/update/finalize preserve a supported accumulator across input chunks.
  The contract fixes ownership, update order, required casts and permitted reassociation.
- `EARLY_RELEASE`: optional finer lifetime information permits releasing fragments before the
  whole tile ends. Conservative bodies keep scratch until full completion.

A library exposing only a monolithic mainloop must not be assigned imaginary preload or update
methods. Adapt a proven component, choose another body, or keep a coarse boundary. The candidate
report states which capability prevents a proposed overlap.

### Separate completion events

The common rule is that an asynchronous movement can finish accessing its source before its
destination is visible to a consumer. The former can release source staging; it cannot publish the
destination. Each adapter supplies distinct completion tokens when its mechanism distinguishes
these points. In the first CUDA adapter, PTX explicitly distinguishes read completion from full
bulk-group completion.
[PTX async bulk wait contract](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

Load completion, MMA completion, output visibility and source retirement can likewise differ.
Each backend adapter must supply the required joins, barriers, visibility operations and progress
rules for its instructions. In CUDA, device acquire/release atomics alone do not complete an
outstanding TMA store or make unsafe shared-memory reuse legal. Do not replace an
instruction-specific protocol with a generic barrier assumption.

## 3. Joint bounded planning

For each recognized region, retain a small menu of alternative semantic covers. For each cover:

1. Enumerate a few compatible bodies and logical tile shapes, including the unfused alternative.
2. Derive input/output and reduction footprints. Partition outputs by **consumer readiness**:
   complete head groups and hidden K chunks, not arbitrary equal counts of FX nodes.
3. Choose logical edge transport: view, local forwarding, materialization, collective/remote
   movement where required, or explicit pure recomputation. Ask the adapter for physical spaces
   and protocols; propagate layout requirements and price conversions and duplicate loads.
4. Instantiate bounded schedules: barrier control, weight lookahead, and supported producer/consumer
   overlap. Search only a few cohort splits and staging depths, initially zero/one/two lookahead
   slots where legal; internal body stages are separate and also consume storage.
5. Derive logical lifetimes against possible overlap, then ask the adapter to allocate physical
   storage and insert target publication/release operations. Check semantic, memory, participation
   and progress obligations in both forms.
6. Rank by a resource-constrained makespan estimate and uncertainty. Compile a bounded set of
   finalists; use actual entry resources to reject or revise choices. Measure survivors later.

An initial engineering cap might be a beam of eight alternatives per region and a few common
block configurations. That is a configurable search-budget hypothesis, not a claimed optimum.
Retain at least one legal control and mechanism-distinct alternative when costs are unknown.
Record why a candidate was pruned; an inaccurate prior must not silently delete every pipeline.

Static code structure is specialized; token positions, input addresses and valid lengths can
remain guarded runtime data. Repeated regions can reuse parameterized templates. Do not unroll
every layer/tile merely to avoid a descriptor load if code size then damages instruction delivery.

### What the objective actually considers

The estimator tracks dependency readiness, finite worker/cohort capacity, staging/accumulator
storage, and shared memory-system/compute resources. Pairwise measurements refine interference
where contenders share HBM/L2/issue bandwidth. It must not let two memory-bound operations each
consume the full measured memory bandwidth simultaneously.

Minimizing the sum of standalone body latencies, maximizing theoretical occupancy, or minimizing
the number of materialized tensors are each insufficient. The target is complete invocation time
under the fixed semantics and one-grid contract, with a memory/setup budget and measured uncertainty.

## 4. Pipeline A: projection heads into decode attention

Suppose projections produce query heads and grouped K/V heads. For each query group, derive the
specific Q, K and V producer tiles, RoPE positions, cache writes, valid length and mask needed by
its attention body. That set—not the entire QKV tensor—defines its readiness event.

```text
projection cohort: [head group 0 Q/K/V + RoPE/cache] [group 1] [group 2] ...
                                      |                |
attention cohort:                 [attention 0]     [attention 1] ...
```

This is a dependency sketch, not a to-scale timing claim. Producer and consumer execution must
have compatible resident capacity; the planner may trade producer throughput for earlier attention.
Per-query-head readiness can be finer than a whole GQA group if the implementation benefits.
Publishing an entire group together is a conservative option, not an inherent attention rule.

Required conditions:

- Each advertised head region contains every dimension needed by the consumer. If several tiles
  construct one head, all contribute to its completion event.
- The current-token cache write is fully published before dependent attention reads. Old valid
  cache contents belong to the initialized session. No consumer reads capacity beyond valid length.
- Shared K/V heads map correctly to their query heads. RoPE pairing/scaling and masks stay exact.
- A fused QKV/RoPE/cache body retains any other live FX outputs; extra consumers cannot disappear.
- Complete output-projection reduction still needs all relevant attention values. Early attention
  is useful without early output projection; optionally stream that projection using a separately
  verified reduction continuation, not an incorrect same-index dependency.

Consumer-aligned layout can require different weight packing or projection tiling. Include those
choices in selection. A broad fused QKV GEMM that delays every head until its last tile can defeat
the desired schedule despite winning as an isolated kernel.

## 5. Pipeline B: a genuinely streamed MLP

For a conventional gated MLP, with numerical boundaries supplied by its reference region:

```text
g = W_gate x; u = W_up x; h = phi(g) * u; y = W_down h
```

Here phi is the reference's actual activation: SiLU for SwiGLU, not a mandatory replacement for
GELU or another expression. Divide the intermediate dimension into chunks `J0 ... Jn-1`. A producer computes corresponding
gate/up values and gating for one chunk, applying required intermediate casts, then publishes
`h[Jj]`. Every down-projection output tile depends on all K chunks eventually, but its computation
can start with the first one:

```text
owner accumulator A[I] = 0
for J in the specified chunk order:
    acquire h[J]
    A[I] = supported_accumulate(A[I], W_down[I,J], h[J])
finalize A[I] with the permitted cast/bias/residual policy
```

The planner tests three variants:

| Variant | Potential advantage | Main cost or constraint |
|---|---|---|
| Materialize all h, then full down | Simple ownership and best full-reduction body may win | No gate/down compute overlap |
| Output-owner continuation | Start K updates early; no global split-K partials | Long-lived accumulators, staged body needed, fixed owner capacity |
| Independent partials + finalizer | More flexible workers and partial readiness | Additional partial traffic/reduction, initialization and numerical-order policy |

The first pipeline experiment includes the owner-continuation variant; global partials are a
bounded alternative when it is unsuitable. Keep only as many active output tiles per owner as
the actual accumulator/storage budget supports. More tiles than owners may require passes or
another assignment; do not assume all output accumulators fit in registers at once.

Use distinct global offsets for all `h` chunks within the region initially. This keeps the small
activation arena simple and avoids per-chunk multi-consumer ring reclamation. The buffer is reused
only after every reader completes at the region join. This does **not** remove h's cross-CTA
write/read: it removes materialized g/u when local fusion permits, and overlaps ready work.

No output is final after one K chunk. Do not add a residual, round the running accumulation to
BF16, or increment a final-output event after each update unless that is the reference's actual
semantics. Reordered floating reductions need an explicit numerical policy and tests; algebraic
linearity alone is not bitwise equivalence.

## 6. Pipeline C: weight movement across task boundaries

Weights and their addresses are often known before the next activation. A worker can reserve a
slot and issue loads for its next assigned weight tile while current arithmetic or output work
continues. It waits for the activation and copy completion only before using that tile.

```text
weight address + free slot -> preload next tile -> load-complete --+
current computation -------> next activation becomes ready ------+-> next compute
```

This applies across tiles of one operator, between different operators, and across layer boundaries
when the dependency/ownership proof permits it. It does not imply prefetching the entire next
matrix into shared memory, or reusing different layers' distinct weights. A next-layer activation
still waits for its preceding reduction/norm. Cross-layer lookahead is an architectural capability
now, with bounded depth and measured selection, not a promise that every layer boundary hides work.

Start with explicit, fixed logical staging slots. A backend binds them to reachable physical spaces
and may use asynchronous movement; alignment, completion, visibility, participation and capacity
rules remain mandatory. The first CUDA adapter considers global-to-shared transfers.
An API named async can fall back to synchronous movement only when the target plan records the
different mechanism and preserves the logical completion/retirement conditions.
[CUDA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)

For CUDA, compare unified loading by compute participants against supported loader/consumer warp roles.
Dedicated loader warps are a target choice with a resource cost, not a logical-plan requirement. CUDA's
pipeline interface makes stage acquire/release and warp-converged commit behavior explicit.
[CUDA pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)

Prefetch can lose through HBM contention, cache pollution, duplicated loads, descriptor overhead,
register pressure or lost residency. Price the **joint** workload. Avoid issuing far-future work
when ready critical-path traffic needs the same memory bandwidth. Descriptor prefetch is separately
labelled; prefetching a task record is not evidence that matrix data movement was overlapped.

## 7. Tail-aware scheduling without a universal queue

For `N` identical tiles over `P` workers, the final wave may have fewer than P active workers.
That idle capacity is an opportunity only if other useful, resource-compatible work is ready.
An early next-weight load, a completed head's attention or a ready K-chunk update can qualify.

Initial choices are consumer-aligned tile sizes, balanced static assignment and producer/consumer
cohort ratios. Compare fixed cohorts with generated mixed worker sequences where progress is
provable. Each scheduling boundary is a body-supported stage boundary, not preemption of arbitrary
GEMM code. Do not dedicate scheduler CTAs by default; they reduce arithmetic capacity.

Small tiles improve readiness but add publication, setup and scheduling work and may degrade the
math body. Large tiles amortize that work but delay consumers. Choose their tradeoff jointly.
Aggregate events only for consumers sharing the same complete dependency set; a single global
event for an entire operator can reintroduce the very tail the plan intended to remove.

Dynamic ready-frontier dispatch becomes justified when measured imbalance cannot be handled by
these bounded schedules. It must inherit the same ownership, credit and progress contracts. A
general queue is not needed to demonstrate asynchronous execution on a known graph.

## 8. Storage allocation and bounded lifetimes

Allocate against the plan's partial order. Two storage uses may overlay only when all possible
executions enforce release-before-reuse. A fast-looking simulated timeline is not a safety proof.
Conservatively treat incomparable lifetimes as potentially overlapping.

Physical local staging accounts for current body internal stages, next-tile lookahead, output
stores, synchronization metadata, alignment and padding. Accumulators have independent live ranges.
The backend must account for the complete compiled resource envelope under its execution model;
variable logical roles do not automatically recover physical capacity. In CUDA, this means per-CTA
staging and a shared/register envelope reserved throughout the launch rather than phase-local occupancy.

Initially:

- Per-local-group preload rings have a small static slot count, explicit release and phase/generation
  state where slots cycle. Reuse waits for all previous async readers of that slot. The CUDA
  realization uses per-CTA storage.
- Cross-worker activation chunks use unique remotely reachable addresses during a region; coarse
  verified joins allow whole-region arena reuse across layers. The CUDA realization uses global
  address-space chunks between CTAs.
- A later cross-worker ring needs acknowledgements from **every** consumer, bounded credits and
  generation validation. Producer completion alone never permits overwriting a published chunk.
- Inputs, output retention and session state cannot alias scratch merely because graph execution
  has advanced to the next operator.

Static slot assignment avoids a general page allocator while capturing bounded overlap. Early
release can improve the schedule; it is accepted only when the body exposes a verifiable access
completion point. This is a performance/correctness interface, not guessed liveness from source.

## 9. Cross-worker publication and the first CUDA protocol

The logical contract requires a finite checked producer set, destination visibility before
publication, acquisition before consumption, invocation-safe initialization/generation and a
separate last-reader condition for reclamation. It does not prescribe atomics, polling, semaphores
or a particular memory scope. The adapter must choose a mechanism whose scope contains every
declared producer/consumer and must expose the mechanism's progress requirements.

The following is the conservative first CUDA protocol to implement and test, not a claim of
existing verification. Use correctly aligned device-scope atomic objects in device memory, in one
compatible memory synchronization domain. Every event has a finite, checked set of unique producer
contributions.

1. The session serializes invocations. The new grid initializes its own event counters to zero,
   then every worker reaches a cooperative grid barrier before using them. This work is timed.
2. An event for produced data has a positive expected count. Graph inputs are separately declared
   ready; empty dependency sets are compile-time identities, not accidental zero-ready events.
3. Each producer completes all required writes, including full async-store visibility. Its writers
   join into the publishing thread using the backend's correct protocol.
4. The publisher performs device-scope acquire-release `fetch_add` on the event exactly once for
   its declared contribution. The RMW chain orders all earlier contributions transitively.
5. A consumer's acquire load observes the expected count before reading that region. One polling
   thread can distribute readiness to its CTA through a proper shared handoff and CTA barrier.
6. No event counter is reset/reused during the invocation in this first cross-worker scheme.
   Counter bounds and contribution multiplicities are checked; unexpected extra arrivals are an
   error, not additional useful work. At region joins, all pending accesses must finish.

This defines publication, not resource reclamation: a produced activation remains live until its
last reader. If later event reuse/epochs are introduced, their wraparound, initialization and
generation checks require their own proof. Do not copy historical relaxed/volatile polling into
V3 as a substitute for that proof. Atomic scope and memory location both matter.
[CUDA C++ memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)

Per-producer flags are an alternative if a shared event's RMW contention is measured to dominate.
Compare their polling/space cost; neither form justifies a per-element global atomic by default.

## 10. Forward progress is part of legality

Acyclic tensor dependencies do not prove a schedule deadlock-free. For example, worker A can wait
for a consumer assigned behind blocked work on worker B while B waits for a resource held by A.
An early prefetch can also consume the only slot needed by a prerequisite task.

The logical verifier checks the augmented ordering/wait-for graph, including abstract participant
order, stage ownership, resource credits and collective participation. Target lowering then adds
physical worker order, primitive participation, placement/residency and backend progress edges and
checks the combined graph again. A logical proof alone cannot certify a target whose workers or
collectives do not have the assumed progress behavior. For the initial templates:

- Every blocked consumer has a producer that can run in the admitted resident worker/cohort set.
- A waiting continuation does not hold a slot needed by that producer; dedicated accumulator
  ownership is included in the capacity proof.
- Per-worker generated order plus dependency edges has no wait cycle. For cyclic buffer reuse,
  prove the bounded credit protocol or reject the template.
- Waits occur only at body-declared stage boundaries and cannot prevent required participants from
  reaching the waited-on operation.
- All required all-worker joins occur in a backend-legal uniform sequence; workers required by a
  collective do not exit early.

For the first CUDA lowering, one warp must not wait for a CTA-wide operation that requires that
same warp to proceed first. All cooperative-grid joins are uniform, idle CTAs do not return early,
and no ordinary oversubscribed grid substitutes for the validated cooperative launch. The CUDA
execution model provides eventual progress for threads in a cooperative grid once a
device thread makes progress; ordinary grids do not provide that same whole-grid guarantee.
This does not cure a logical circular wait. Use documented atomic/synchronization polling and
validate the actual cooperative entry's residency and target support.
[CUDA execution model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/execution_model.html)

Those CCCL pages are moving `unstable` documentation accessed 2026-09-09, not the project's pinned
implementation toolchain. Before implementation, verify the applicable contract against the
selected CUDA/CCCL release. The CUDA design does not rely on ordinary-grid scheduling fairness.
A future backend must supply and test its own corresponding progress argument; it does not inherit
CUDA's cooperative-grid guarantee through the common abstraction.

## 11. Required validation and mechanism evidence

CPU plan validation/model tests should cover:

- Tiny legal schedules under many action interleavings; missing/duplicate producers and premature
  publication rejected; immutable semantic and effect coverage preserved.
- Head mapping, reduction K coverage, non-divisible tiles, empty/masked attention semantics,
  chunk finalization, and intermediate cast boundaries.
- Scratch poison/reuse simulations, two live preloads, slow stores, slow consumers and event
  reuse attempts; worker-order and slot-credit cycles intentionally rejected.

Device tests add repeated invocations, alternate valid positions/buckets, legal tail shapes,
state comparison, supported sanitizers, and deliberately perturbed producer/consumer timing in
debug variants. Sanitizer success does not prove cross-CTA publication or liveness. Test the
publication and async-source-retirement litmus cases independently before a full model.

The plan report names each proposed optimization, removed materialization, enabled early edge,
copy stage, remaining join, and rejection reason. Diagnostic device traces relate logical tile
IDs and events to worker activity; do not infer cross-SM time order from uncalibrated `clock64`
values. Use a validated common timing source or causal event ordering. Instrumented runs explain
behavior; uninstrumented trials decide speed.

The primary block experiment must exercise head-ready attention, streamed MLP and cross-task
weight lookahead, plus controls disabling each. Whole-model reporting includes mechanism speedup
over a matched owned-body phase control and end-to-end speedup over the strong external baseline.
A supported optimization may be rejected for a particular cell; absence of a tested pipeline
must not be presented as evidence that launch-only composition exhausts the design space.

## 12. Scope of the claim

This architecture gives the compiler the required information and lowering mechanisms to pursue
the benefits. It does not prove they are profitable on every shape, nor that current library
bodies expose all required stages. The measurable hypothesis is that joint body/layout/schedule
selection recovers enough effective execution efficiency to beat matched alternatives.

If source-compatible competitive bodies cannot support useful overlap in the common resource
envelope, record that failure. The remedy is a specific body/placement change or a scoped strict
loss—not an unsupported claim that one launch necessarily wins, and not another mandatory IR.
