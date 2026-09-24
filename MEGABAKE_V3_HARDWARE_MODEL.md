# MegaBake V3: hardware facts without a hardware-discovery project

Status: proposed, 2026-09-09; backend boundary revised 2026-09-24. No device was queried or benchmarked for V3.

## 1. Verdict on the DFS idea

The fact table is useful; automatic architectural discovery by DFS is not the right mechanism.
Traversal requires an existing graph. CUDA exposes some properties, manuals describe additional
features, and experiments reveal achieved costs. None exposes a universal complete hierarchy with
all bandwidths and latencies attached.

Use a compact backend-qualified `TargetProfile`. Represent only facts that change legality, tiling,
placement/residency, movement, synchronization or a measured cost. The common schema does not
assume an SM, warp, CTA or cooperative CUDA launch. A relational table plus a few scope/movement
edges is sufficient. Each backend populates it through a versioned adapter. V3 implements only the
CUDA adapter; the abstraction is an extension seam, not evidence of another working target.
Do not build a general chip reverse-engineering framework before compiling a decoder step.

## 2. Three classes of facts

| Class | Examples | Source | May prune legality? |
|---|---|---|---|
| Queried capability | Compute resources, local-memory limits, topology, supported invocation mechanisms | Backend runtime/compiler API | Yes, subject to API meaning |
| Documented feature | Body/instruction family, synchronization scope, movement and collective restrictions | Versioned architecture/ISA/backend documentation | Yes, with precise target gating |
| Calibrated cost | Shape latency, achieved bandwidth, barrier cost, composition penalty | Experiment with provenance | No: selects/ranks, not a correctness proof |

Unknown costs remain `UNKNOWN`. A documented peak is not a measurement. A historical result from
another MIG profile is not local calibration. Estimates may rank exploratory candidates, but must
carry their assumptions and may not be presented as predicted measured performance.

## 3. A small schema

```text
TargetProfile {
  identity: backend, target kind/version, visible partition/topology,
            runtime, compiler and body-provider versions,
  execution_domains: participant/group/device/mesh domains and capacities,
  capabilities: supported body, invocation, movement, synchronization and collective mechanisms,
  spaces: stable IDs with reachability, capacity/alignment and access restrictions,
  movements: supported source/destination edges, issuers and required scopes/protocols,
  synchronization: supported scopes, participation, ordering, visibility and progress contracts,
  topology: addressability, collective/remote links and backend-visible placement constraints,
  contention_domains: shared bandwidth/issue resources relevant to chosen actions,
  stage_requirements: operation completion, source retirement, destination visibility and descriptor lifetime,
  costs: [experiment_key, value, unit, conditions, uncertainty, provenance],
  source_versions, unknown_fields
}
```

Profiles use backend-owned stable identifiers such as `cuda:cta_shared`; the common planner treats
them as opaque spaces with declared properties. It may compare reachability, capacity and movement
edges, but must not branch on spelling such as `shared`, `VMEM` or a numeric architecture suffix.
Logical target requirements request properties—local cooperative storage, an asynchronous copy
with distinct completion/retirement, a collective group—not a CUDA primitive name. The adapter
either binds those requirements to precise mechanisms or rejects the candidate.

Global memory is an address space. HBM/DRAM and L2 are physical parts of the memory system. Registers
are not a universally addressable cache tier. Texture/constant paths, shared/L1 partitioning and
distributed shared memory do not fit a single “four levels” ladder.

For pipelining, record movement **edges**, not just capacities: source/destination spaces, issuing
participants, alignment, descriptor prerequisites, completion scope and source-retirement meaning.
Bandwidth-sharing domains are a small cost-model annotation, not an automatically discovered
microarchitecture. An async engine does not provide independent unlimited memory bandwidth.

The first CUDA profile instantiates the common schema with these resources:

| CUDA storage/resource | Ownership and relevant use | Important restriction |
|---|---|---|
| Registers | Thread/warp computation and accumulators | Allocation/liveness can limit residency; spills use local memory |
| CTA shared memory | Cooperative tiles and async staging | Not accessible from arbitrary other CTAs |
| Cluster distributed shared memory | Optional cluster-local exchange | Requires supported launch and cluster synchronization/lifetime |
| Global address space | Weights, outputs, state and inter-CTA buffers | Physical accesses may hit cache instead of HBM |
| L2 | Shared cache for global traffic | Residency/contention are workload-dependent |
| Tensor memory, where supported | Specialized tensor-core accumulator storage | Architecture-specific allocation and access protocol |

Facts about tensor memory are not generalizable to every GPU marketed as Blackwell. Use instruction
and target-specific support, as illustrated by the
[tcgen05 programming documentation](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/mma_docs/tcgen05_programming.html).

## 4. Backend discovery and CUDA realization

The common discovery interface accepts an explicit backend/target selection, records absent facts
as unknown and never changes the active device as a side effect. It is lazy so CPU-only planning
does not load CUDA. Offline profiles use the same schema and provenance rules. Cache keys include
the backend ID and exact target identity; similarly named resources from two backends are never
assumed compatible.

For the first CUDA implementation, obtain only what the planner uses.

At session/device initialization, query documented properties and attributes, including visible
SMs, compute capability, block/thread limits, shared-memory opt-in limits, register limits and
cooperative support. The [device property API](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
is the starting point, not a calibrated performance model.

After compiling each complete entry, inspect function attributes and compiler resource reports;
set the requested dynamic shared-memory attribute where required; ask CUDA occupancy APIs about
that function/block/shared-memory combination. Validate cooperative grid limits before launch.
The [cooperative-groups documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
describes the required launch/participation model.

A first residency bound for an ordinary non-cluster cooperative entry is:

```text
resident_workers <= visible_SMs * supported_active_CTAs_per_SM(actual_entry, block, dynamic_smem)
```

Use the API's actual resource accounting, not only hand division: allocation granularity and other
limits matter. Cluster launches require their own support/occupancy checks. Cap the initial planner
to a few valid worker counts; maximum residency is not necessarily maximum throughput.

The current `num_sms` grid and fixed huge shared-memory allocation are not a substitute for this
check. Nor does a one-CTA-per-SM-sized grid guarantee physical placement or future resource legality.

The pipelined plan's producer/consumer cohorts must all fit this admitted worker set. Holding a
continued-reduction accumulator while another body runs can alter register pressure. Current tile
stages, next-task preloads, delayed output stores and metadata can require simultaneous shared
storage. Measure the complete role mixture; neither `max(isolated_smem)` nor a per-phase occupancy
estimate is a sufficient composition model. See [storage and progress](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md).

## 5. CUDA architecture distinctions worth encoding

| Target family | Facts that can change a candidate | V3 consequence |
|---|---|---|
| Ampere | MMA/copy mechanisms and resource differences between individual SM targets | Use an appropriate device body, not a Hopper fallback selected by product name |
| Hopper | TMA, WGMMA, optional clusters and associated barriers | Consider a supported pipelined body; include setup/storage/participation costs |
| Newer targets | Different MMA and accumulator-storage mechanisms | Add explicit feature entries and a tested adapter |

The [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) and
[Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) guides explain why one
architecture-wide shared-memory constant and numeric `SM >= 90` feature logic are insufficient.
For Hopper, TMA is asynchronous and an issuing block can do independent work before waiting;
useful overlap does not universally require separate producer/consumer warps. Warp-specialized
pipelines are one design, with their own resource tradeoff.

Distinguish targets such as `sm_80`, `sm_90a`, `sm_100a` and `sm_120` through supported features and
toolchain compatibility. Do not create an architecture suffix or assume every later numeric target
supports every earlier architecture-specific instruction. Unsupported compilation is a clear
candidate rejection, not a silently degraded unknown body.

A future non-CUDA adapter must provide equivalent facts rather than emulate this table by analogy.
At minimum it must identify its compute/group/mesh domains, locally and remotely reachable spaces,
layout restrictions, async movement issuers and completion semantics, synchronization and collective
participation scopes, invocation admission rules, compiler resource reports, and runtime ownership.
For example, a TPU adapter would need to model HBM/VMEM/SMEM and semaphore-backed DMA/remote-copy
contracts as their own mechanisms; it must not rename VMEM to CUDA shared memory or pretend a TPU
mesh barrier is a cooperative-grid barrier. Such an adapter is deliberately outside the first
implementation ledger.

## 6. H200 MIG: what the historical profile tells us

The latest V2 report describes 60 visible SMs in `3g.71gb`; older tracked JSON describes 32 visible
SMs in `2g.35gb`. Keep these separate. The
[MIG profile table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
assigns H200 `3g.71gb` three sevenths of SM resources and four eighths of memory/L2 resources.
These resource fractions are not an achieved-bandwidth measurement.

The full [H200 product specification](https://www.nvidia.com/en-us/data-center/h200/) lists
4.8 TB/s bandwidth. Multiplying by one half gives a **2.4 TB/s proportional estimate**, not a
guaranteed effective bandwidth or an unconditional model-latency bound. Query the actual available
partition and calibrate it when a GPU returns.

Do not plan around an L2 persistence feature that is disabled in MIG mode: NVIDIA documents this
restriction for the [persisting-L2 set-aside](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html).
Ordinary L2 caching still matters; disabling this set-aside does not mean every access goes to HBM.

## 7. The minimum useful calibration set

The following set is for the first CUDA backend. Another backend keeps the provenance and
whole-entry principles but substitutes its own invocation, synchronization and counter mechanisms.

1. Exact hot linear shapes at matched dtype/layout, with both standalone and persistent-composed
   versions. Record the cache/weight working set.
2. End-to-end empty/minimal invocation under the chosen input/output contract; ordinary stream and
   replayed graph controls are separate experiments.
3. Grid joins, cross-worker publication/acquire and stage transitions under selected entry
   configurations; include event fan-in/fan-out and contended versus uncontended cases.
4. Streaming read and representative gather/state access bandwidth at relevant sizes. A copy
   benchmark alone does not predict GEMV's achieved bandwidth.
5. Real attention/state cost at short and long context; do not infer it from seq-one uncached code.
6. Two-tile and cross-task load/compute/store experiments: no lookahead versus one/two slots;
   measure fill/drain, useful lead time, source retirement and actual overlapped elapsed time.
7. Representative concurrent producer/consumer pairs: QKV with attention, and gate/up with down
   updates. Record joint throughput, physical traffic, cohort split and the actual resource report.
8. Logical-tile tail sweeps near worker-count multiples and different chunk sizes; compare
   static barriers with ready-consumer schedules at fixed body/numerical contracts.

These calibrations serve the first pipeline decision, not a general hardware-survey project.
Obtain only the cells used by current candidates. Peak FLOP tables are insufficient for skinny
matrices, and isolated warm-cache microkernels are insufficient for a full model.

The planner uses legality facts to admit a load/stage, a proven lifetime to allocate its slot,
and calibrated joint costs to decide whether issuing it early is useful. No GPU means the third
step stays unresolved. V3 can specify a legal candidate without claiming it achieves that overlap.

## 8. Cache keys and offline use

Correctness/legality keys include graph semantics, guards, dtype/layout, backend and target feature set,
numerical flags, selected body versions and toolchain ABI. Performance keys additionally include
workload bucket, visible resource profile, relevant clock/power conditions and experiment settings,
body mixture, layout, cohort assignment and pipeline depth. Changing those invalidates associated
joint-cost estimates even if the standalone shape is unchanged.
Device identity is useful provenance; it need not force recompilation across demonstrably compatible
devices. A new resource profile does require rechecking launch legality and invalidates old tuning
assumptions where applicable.

Without a GPU, the CUDA adapter emits a source/plan report with unresolved costs. If a suitable toolkit is present,
cross-compilation can test syntax and some resource constraints; it cannot establish runtime
correctness, actual occupancy behavior or latency. V3 performs no such new GPU validation.

The fact table earns its place by pruning illegal bodies and attaching measured costs to legal
ones. It is not a source of automatic performance proofs.
