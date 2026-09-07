# MegaBake V2.3 Architecture

Status: generic-compiler redesign with its first empirical validation on H200. The GraCE/CUDA
investigation, source audit, Nsight Compute profile, exact-shape GEMV probes, and strong-baseline
replay inform the contracts; they do not make Hopper the deployment target. This document is
normative with:

- [`MEGABAKE_V2_DATAFLOW_DIAGRAM.md`](./MEGABAKE_V2_DATAFLOW_DIAGRAM.md)
- [`MEGABAKE_V2_IR_AND_REUSE_PLAN.md`](./MEGABAKE_V2_IR_AND_REUSE_PLAN.md)
- [`grace_hack.md`](./grace_hack.md)
- [`MEGABAKE_V2_GPU_REANALYSIS.md`](./MEGABAKE_V2_GPU_REANALYSIS.md)

## 1. Executive decision

MegaBake V2.3 is a **generic traffic-first compiler that emits bucket- and target-specialized
one-grid inference artifacts**. The compiler and upper IR are reusable across hardware; kernel
families, machine pipelines, launch envelopes, packed layouts, and measured winners are not.

Its research north star is to emit one owned GPU compute grid that beats the strongest equivalent
baseline for each supported target/model/bucket. “One grid” describes an invocation, not one
universal binary, algorithm, or launch envelope shared by all workloads. “Most optimal” means the
best legal measured candidate in a declared search space with a recorded margin; it is not an
unprovable claim of a global optimum.

The design has two execution planes:

1. **Persistent plane** — owned `DEVICE_CALLABLE` implementations execute inside resource-specific
   persistent CUDA grids.
2. **Graph plane** — CUDA Graphs orchestrate persistent entries and separately launched high-quality
   kernels such as cuBLASLt, cuDNN, or standalone generated kernels.

Strict research mode requires one persistent grid. The graph plane is the measured product fallback
and oracle path; a hybrid win does not count toward strict-one-grid coverage. One host submission
and one GPU grid remain different quantities.

The compiler has only three real representations:

1. `FXGraph + FactTables`
2. `RegionGraph`
3. `PlanTemplate`, frozen as `FinalExecutionPlan` after code generation and measurement

V2.3 retains these V2.2 contracts:

- weight, activation, KV-cache, and materialization bytes are first-class costs;
- precision and prepacked layout are first-class plan choices;
- fixed buckets use a small compiled `PhaseProgram`, not arbitrary per-worker busy-wait DAGs;
- a cheap producer may be recomputed `PER_WORKER` or `PER_CLUSTER` instead of materialized to HBM;
- persistent entrypoints are split by compiled resource class rather than sharing one maximum-SMEM
  universal kernel;
- per-call binding, scheduler reset, and output ownership are explicit `InvocationContract` costs.

The GPU evidence adds five corrections:

- logical phase work is independent of resident worker count;
- a generated artifact has an explicit compatible `EntryLaunchEnvelope`;
- `M<=4` does not select one linear family: warp-reduction, CTA/split-K, and padded-M tensor-core
  tactics have measured shape-dependent crossovers;
- semantic end-to-end bandwidth, semantic GPU-body bandwidth, and counter-derived physical DRAM
  bandwidth are distinct measurements;
- strict acceptance is gated by the strongest low-overhead/static-address compiled baseline and a
  cross-model/hardware coverage scorecard.

The first validation milestone remains narrow: reference-precision decode on the available H200 at
`M=1..4`. It validates compiler contracts and exposes failure modes; it is not the main target or a
portable tuning result. A generic-CUDA performance claim additionally requires at least one non-
Hopper architecture family and then a broader predeclared hardware matrix.
Weight-only quantization and continuous batching are separate numerical/serving regimes, but they
are represented from the beginning because they attack the fundamental decode bottleneck: weight
traffic.

The first H200 result validates the launch-fusion motivation but not the current implementation. On
SmolLM2-135M, the present grid measured 4.77 ms, 184 registers/thread, a 1,072-byte stack frame,
225.28 KB dynamic SMEM/CTA, 12.5% occupancy, 6.42% compute throughput, and 2.33% DRAM throughput.
An export-based `torch.compile(mode="reduce-overhead")` measured 1.782 ms median. The current strict
grid is therefore about 2.7x slower than the relevant low-overhead baseline.

None of those numerical values, kernel crossovers, or launch shapes enter a target-neutral rule.
They are H200 observations stored under its complete target fingerprint.

## 2. Physical facts that determine the architecture

### 2.1 Persistent launch is not persistent model storage

For a batch-one linear layer, nearly every weight is consumed once per token. Model weights are far
larger than the combined register, shared-memory, and cache capacity of the target device or
partition.
Keeping a grid resident removes dispatch boundaries; it does not keep a multi-gigabyte model
resident on chip.

Therefore, the main decode levers are:

- increase achieved bandwidth for each weight stream;
- reduce weight bytes with an accepted precision policy;
- reuse each loaded weight tile across multiple active tokens or requests;
- eliminate avoidable activation/KV/materialization traffic and synchronization.

No amount of schedule cleverness changes this roofline.

### 2.2 GraCE does not make cuBLAS device-callable

For an immutable vendor kernel, a GraCE prelude writes new values into existing parameter slots of
a captured graph node. It changes neither the vendor kernel ABI nor its code, function identity,
block shape, resource allocation, or grid boundary.

Consequently:

- cuBLAS and cuBLASLt candidates are `GRAPH_NODE`, never `DEVICE_CALLABLE`;
- a live private `CUfunction` is a host handle and diagnostic oracle;
- direct relaunch of a frozen private plan is a research experiment, not a deployment interface;
- one-grid composition requires MegaBake-owned device code, optionally built from public
  CuTe/CUTLASS or cuBLASDx components.

### 2.3 One host submission is not one GPU grid

V2.3 counts these separately:

- host submissions;
- GPU grids;
- persistent segments;
- phase joins and atomics;
- HBM materializations;
- binding and copy operations.

A graph containing twenty kernel nodes is one host submission and twenty GPU grids.

### 2.4 On-chip storage belongs to a CTA or cluster

Registers belong to threads. Shared memory belongs to a CTA. Distributed shared memory belongs to a
declared block cluster. None belongs to an abstract physical SM across arbitrary tasks.

On-chip forwarding is legal only when a compiled `FusionContract` proves:

- producer scope and consumer scope;
- thread/warp ownership;
- layout, dtype, size, and alignment;
- synchronization and lifetime;
- combined compiled resources.

`blockIdx.x` is a worker slot, not a stable SM identifier.

### 2.5 Recomputation can beat communication

Decode activations are tiny relative to linear weights. It can be cheaper for every output-tile CTA
to recompute RMSNorm, keep the normalized input in its own shared memory, and immediately stream its
weight rows than to materialize the normalized tensor and coordinate a global handoff.

V2.3 therefore permits these value transports:

```text
ALIAS_VIEW
MATERIALIZE_HBM
REGISTER_FORWARD
SMEM_FORWARD
RECOMPUTE_PER_WORKER
RECOMPUTE_PER_CLUSTER
```

Recomputation is a measured implementation choice. It never changes graph semantics.

### 2.6 Resource allocation is entrypoint-wide

A monolithic persistent entry inherits the maximum register, shared-memory, launch-attribute, and
warp-structure requirements of its reachable paths. A light pointwise phase does not regain
occupancy merely because the heavy WGMMA path is inactive.

V2.3 normally emits separate entries for resource classes such as:

```text
LINEAR_HEAVY      TMA/WGMMA or large mixed-dtype pipelines
SKINNY_STREAMING  M=1..4 GEMV/skinny GEMM variants
ATTENTION_HEAVY   paged/split-KV attention pipelines
COMPACT           reductions, RoPE, pointwise and small composites
```

CUDA Graphs may connect these entries cheaply. A single-grid artifact is retained only when its
measured fusion savings exceed its resource penalty.

The H200 profile makes this concrete: the current universal entry uses 184 registers/thread, a
1,072-byte stack frame, and 225.28 KB dynamic SMEM/CTA. Registers and SMEM independently limit it to
one 256-thread CTA/SM and 12.5% occupancy. The decode-only generator must make prefill WGMMA and
other unreachable heavy paths disappear before ptxas; source-level branch inactivity is not enough.

### 2.7 Machine pipelines belong in generated variants

Useful TMA/cp.async overlap requires producer and consumer warps executing concurrently. Prefetch
issued after one task completes and immediately before the next waits is not such a pipeline.

The following are compile-time properties of `KernelVariant` or `CompositeVariant` code:

- TMA/cp.async issue, waits, tensor maps, and barriers;
- producer and consumer warp roles;
- MMA/WGMMA atoms and shapes;
- stage count and shared-memory layout;
- register-accumulator layout;
- epilogue and on-chip forwarding.

The scheduler selects a variant and work range. It does not interpret machine-level prefetch or warp
role opcodes.

### 2.8 Generic compiler, specialized artifacts

Portability is layered:

```text
target-neutral front/middle end
  FX semantics -> facts -> regions -> traffic -> legal composites

architecture-family backend
  legal instruction families -> candidate generators -> code-object format

exact-device specialization
  discovered resources -> calibration -> compile -> profile -> measured winner
```

The initial backend is CUDA, but its interface is accelerator-neutral. A backend must implement
capability discovery, candidate enumeration, compilation/linking, resource inspection, profiling,
launch/graph recipes, baseline construction, and artifact serialization. A future ROCm or other
accelerator backend can reuse the upper representations but must supply equivalents for these
machine services; CUDA concepts are not silently assumed portable.

Within CUDA, architecture-family plugins own instruction legality and optimized bodies. For
example, an SM80 target such as A100 must not receive Hopper-only TMA/WGMMA code; it selects SM80
SIMT, `cp.async`/MMA, and eligible CUTLASS bodies. An SM90 target may add TMA/WGMMA candidates. An
unknown target may run a correctness-oriented portable CUDA body or product fallback, but MegaBake
does not call it optimally supported until its backend candidates beat the target's baselines.

The H200 database may seed search priors such as “try warp-reduction GEMV,” but never reuses an
H200 winner, resource envelope, occupancy choice, or crossover as an A100 result. Generic compiler
logic plus per-target generation/autotuning is the intended combination.

Hardware coverage has explicit levels:

```text
COMPILE_SUPPORTED       code generation and resource inspection succeed
CORRECTNESS_VALIDATED   representative artifacts pass numerical and memory-safety gates
TUNED                   candidates were calibrated and autotuned on real target hardware
STRICT_WIN              one-grid artifact beats that target's strongest equivalent baseline
HYBRID_ONLY             product fallback works, but strict target is not yet won
```

An H200-only laboratory can establish compile coverage for other CUDA architectures and catch
illegal instruction/resource assumptions by cross-compiling. It cannot establish their tuning or
performance levels. Those require a hardware farm, remote runners, or user-device first-run tuning
with cached, versioned results.

## 3. Orthogonal workload contracts

Performance claims are meaningless unless three policies and one compound invocation contract are
fixed independently.

### 3.1 Performance objective

```text
SINGLE_REQUEST_LATENCY
BOUNDED_LATENCY_THROUGHPUT
PREFILL_THROUGHPUT
```

`SINGLE_REQUEST_LATENCY` does not wait for other requests. `BOUNDED_LATENCY_THROUGHPUT` may perform
continuous batching within an explicit waiting and fairness budget. Results from one objective are
not presented as results for another.

### 3.2 Numerical policy

```text
REFERENCE_FP16_BF16
FP8
W8A16
W4A16
```

Every policy has its own correctness criteria and strongest equivalent baseline. A W4A16 result is
never compared as if it preserved reference-precision semantics.

### 3.3 Execution policy

```text
STRICT_SINGLE_GRID
HYBRID_GRAPH
CORRECTNESS_FIRST
```

- `STRICT_SINGLE_GRID` permits only `DEVICE_CALLABLE` candidates in exactly one grid.
- `HYBRID_GRAPH` selects the fastest measured legal mixture of persistent and graph nodes.
- `CORRECTNESS_FIRST` permits host escapes for unsupported operations.

### 3.4 Invocation contract

```text
binding_mode = STATIC_SESSION | DYNAMIC_BINDINGS
output_mode  = RUN_INTO | BORROWED_OUTPUT | OWNED_OUTPUT
reset_mode   = IN_ENTRY_PARALLEL_RESET | EPOCH_TAGGED | GRAPH_MEMSET
```

- `STATIC_SESSION` uses stable device addresses and can avoid per-call pointer updates; it may be
  combined with any output mode.
- `RUN_INTO` binds caller-owned output storage.
- `BORROWED_OUTPUT` returns a view valid until the documented reuse point.
- `OWNED_OUTPUT` guarantees independent lifetime using a pool/allocation or copy; all associated
  work is charged to latency and operation counts.

Inputs also carry explicit dtype, layout, alignment, alias, and lifetime requirements. Hidden
contiguous/cast/copy operations are forbidden in a performance result.

### 3.5 Independent budgets

```text
max_host_submissions
max_gpu_grids
max_persistent_segments
max_hbm_materializations
max_binding_bytes
max_batch_wait_us
allow_host_fallback
```

These are policy constraints, not aliases for semantic region count.

## 4. Architecture overview

```text
CompileRequest
  -> WorkloadContract + WorkloadBucketKey
  -> FXGraph + FactTables
  -> RegionGraph with traffic and recomputation opportunities
  -> TargetArchitectureKey + TargetFingerprint + DeviceCaps
  -> TargetBackend selection and feature-legal candidate generation
  -> CandidateSet + MeasuredBaselineSuite
  -> PlanTemplate[]
  -> generate/compile/resource-inspect/replay/refine
  -> FinalExecutionPlan
       |- PrecisionLayoutPlan
       |- InvocationContract
       |- PersistentSegment[]
       |- CudaGraphSegment[]
       `- fallback chain
  -> ArtifactPack
```

There are three compiler representations, not one IR per pipeline stage. Precision, binding,
buffers, resources, and measurements are plan objects and contracts.

## 5. Compiler representations

### 5.1 `FXGraph + FactTables`

This layer preserves exported PyTorch semantics and records:

- shapes and bucket constraints;
- dtype, stride, layout, alignment, offset, and aliasing;
- parameter identity and prepack eligibility;
- view versus materialization facts;
- mutation/effects, including KV-cache append and sampling state;
- input/output roles and lifetimes;
- numerical-policy legality.

It may canonicalize, decompose, fold constants, perform CSE/DCE, and preserve views. It may not
select persistent execution, precision tactics, kernel families, or worker schedules.

### 5.2 `RegionGraph`

`RegionGraph` recovers transformer computations large enough to expose meaningful traffic and
composition choices.

Initial primitive regions include:

- `LinearRegion` and `MatmulRegion`;
- `NormRegion` and `PointwiseRegion`;
- `DecodeAttentionRegion` and `PrefillAttentionRegion`;
- `RopeRegion` and `KVCacheRegion`;
- `EmbeddingRegion`, `SamplingRegion`, and `GenericRegion`.

Bounded composite opportunities include:

- `NormLinearComposite`;
- `QKVProjectionComposite`;
- `GatedMLPComposite` for gate/up/activation/multiply;
- `RopeKVAppendComposite`;
- linear epilogue composites;
- decode-attention composites appropriate to paged/GQA layouts.

Each region records a `TrafficEstimate` rather than one undifferentiated operation count:

```text
weight_bytes
activation_read_bytes
activation_write_bytes
kv_read_write_bytes
workspace_bytes
flops
parallel_work
```

Each edge records legal value transports, including materialization, on-chip forwarding, and
recomputation scope. A region is not assumed to equal one CUDA launch.

### 5.3 `PlanTemplate` to `FinalExecutionPlan`

`PlanTemplate` fixes a small legal execution neighborhood:

```text
PlanTemplate
  WorkloadContract
  segment templates and dependencies
  candidate configuration sets
  PrecisionLayoutPlan alternatives
  ValueTransportPlan alternatives
  BufferPlan
  BindingSchema
  InvocationContract alternatives
  policy and resource constraints
```

After lowering, compilation, correctness testing, profiling, and bounded search, it becomes:

```text
FinalExecutionPlan
  selected precision/layout and prepack descriptors
  selected value transports and composites
  exact PersistentSegment[] and CudaGraphSegment[]
  exact InvocationContract and state-reset policy
  measured latency/throughput/traffic breakdown
  strongest equivalent baseline
  compatibility guards and fallback chain
```

Compiled feedback may change tile/stage/vector configurations, worker counts, value transport,
resource segmentation, and persistent-versus-graph selection. It may not reopen graph semantics.

## 6. Execution objects

### 6.1 `KernelVariant`

```text
KernelVariant
  semantic_region_kind
  implementation_family
  execution_capability
  supported_shape_layout_precision
  compile_time_parameters
  InputOutputContract
  ResourceEnvelope
  FusionEndpoint[]
  measured_results
```

`ExecutionCapability` is one of:

```text
DEVICE_CALLABLE
GRAPH_NODE
HOST_ONLY
```

Only `DEVICE_CALLABLE` enters a persistent entry.

### 6.2 `ResourceEnvelope`

```text
threads_per_cta
warp_group_structure
static_smem_bytes
dynamic_smem_bytes
registers_per_thread_actual
spill_bytes_actual
stack_bytes_actual
local_memory_bytes_actual
max_resident_ctas_per_sm
cluster_shape
launch_attributes
tma_tensor_map_requirements
minimum_arch_toolkit
```

Post-compile resource values are mandatory. Resource estimates are sufficient only for early
rejection.

### 6.3 `FusionContract` and `ValueTransportPlan`

```text
FusionContract
  producer_variant
  consumer_variant
  producer_scope = ONCE | PER_WORKER | PER_CLUSTER
  transport = REGISTER_FORWARD | SMEM_FORWARD | RECOMPUTE
  layout_dtype_alignment
  thread_warp_ownership
  synchronization_protocol
  lifetime
  recompute_cost
  combined_resource_envelope
```

`REGISTER_FORWARD` and `SMEM_FORWARD` require static composition into a `CompositeVariant`.
`RECOMPUTE_PER_WORKER` permits a cheap semantically identical producer inside every consuming CTA.
HBM materialization is represented explicitly when no such contract wins.

### 6.4 `PersistentSegment`

A persistent segment is one owned grid containing one compatible resource class:

```text
PersistentSegment
  resource_class
  entrypoint
  EntryLaunchEnvelope
  resident_worker_count
  PhaseProgram
  selected_variant_ids
  binding_map
  state_reset_policy
  combined_resource_envelope
```

`resident_worker_count` is not automatically the SM count. It is bounded by cooperative residency
for the compiled `EntryLaunchEnvelope` and may include multiple CTAs/SM. Each phase has a separate
logical work count and may activate only a subset of resident workers.

```text
EntryLaunchEnvelope
  entrypoint_id
  target_and_bucket_key
  compatible_variant_ids
  threads_per_cta and warp_topology
  static_smem_bytes and dynamic_smem_bytes
  registers_per_thread_actual
  stack_bytes_actual and local_memory_bytes_actual
  resident_ctas_per_sm
  resident_worker_count
  maximum_legal_cooperative_grid_blocks
  launch_attributes
  rejected_variant_ids_and_reason
```

### 6.5 Fixed-bucket `PhaseProgram`

The default scheduler is deliberately small:

```text
PhaseProgram
  PhaseDesc[]
  WorkDesc[]
  CompletionEpoch[]

PhaseDesc
  composite_or_variant_id
  active_worker_count
  logical_work_count
  distribution = STATIC_RANGE | ATOMIC_CURSOR | ALL_ACTIVE_WORKERS
  logical_to_worker_mapping
  work_range_or_cursor
  binding_slice
  completion = NONE | CTA_JOIN | GRID_JOIN
```

Rules:

- static ranges are preferred when work is known;
- logical output/reduction tiles are independent of resident CTA count; a worker may execute several
  logical tiles;
- one atomic cursor is allowed for meaningful imbalance, not as the default for every task;
- joins occur between composite phases, not every FX node;
- counters are preallocated and reset inside the entry, epoch-tagged, or represented by graph
  memset nodes; they are never cloned per invocation;
- arbitrary per-worker dependency polling is absent;
- release and profiling binaries are separate.

### 6.6 Serving scheduler

Dynamic request admission is a separate runtime mode, not hidden inside the fixed-bucket executor.
It may add:

- continuous/in-flight batching;
- bounded waiting and fairness;
- layer/shape/precision-compatible work queues;
- grouped persistent GEMM visitors;
- a dedicated scheduler CTA when measurements justify its resource cost.

Its primary purpose is to reuse a weight tile across several tokens, not merely to provide a more
general task abstraction.

### 6.7 `CudaGraphSegment`

```text
CudaGraphSegment
  covered_regions
  GraphRecipe
  workspace_plan
  binding_update_map
  environment_guards
```

Graph recipes use public operations and owned nodes. They may contain resource-specific persistent
entries, vendor kernels, memsets, and binding preludes. Live graph/function handles and private
parameter packs are never serialized.

Programmatic dependent launch is an optional graph-edge tactic only when a downstream kernel has a
substantial prefix independent of its producer and uses the required synchronization protocol. It
is measured like any other candidate; ordinary full dependencies remain the default.

## 7. Precision and layout architecture

`PrecisionLayoutPlan` is a plan object because byte width and physical layout often dominate decode
performance. It may mix precisions across tensors and operations when the numerical policy permits.

```text
PrecisionLayoutPlan
  default_policy
  TensorPrecisionPlan[]
  OperationPrecisionPlan[]
  PrepackDescriptor[]

TensorPrecisionPlan
  value_or_parameter_id
  storage_dtype
  scale_dtype_and_granularity
  zero_point_policy
  physical_layout

OperationPrecisionPlan
  region_or_variant_id
  input_dtypes
  accumulator_dtype
  output_dtype
```

The initial candidate families are:

- reference FP16/BF16;
- FP8 where activation conversion/scaling cost is measured;
- W8A16;
- W4A16 groupwise/per-channel layouts.

Weights are packed once at load/build time into the exact layout required by the selected skinny or
WGMMA variant. Transposes, scale swizzles, padding, and alignment are part of the artifact, not
performed during invocation.

Quantized plans require model-quality validation in addition to numerical kernel checks. They are
accepted only against an equivalent quantized baseline.

## 8. Binding and invocation architecture

V2.3 has a fast static path and a general dynamic path.

### 8.1 Static-session fast path

Model weights, arena buffers, inputs, outputs, KV pages, and scheduler state have stable addresses.
The invocation may need only a small device-resident sequence/token update or no binding transfer at
all.

This is the preferred serving contract.

### 8.2 Dynamic binding path

A compact pinned-host `BindingBlock` is copied once into a stable device `BindingTable`:

```text
BindingBlock -> one asynchronous update -> BindingTable
                                            |- persistent entries
                                            `- one batched GraCE-style prelude
```

The prelude updates only confirmed existing pointer/scalar slots of graph nodes. It is an ancestor
of every edited node, checks every status, and is used only when it beats host update or fixed
addresses.

### 8.3 Output lifetime

The runtime never hides an output clone. The chosen `InvocationContract.output_mode` states whether
output is:

- written into caller storage;
- borrowed until the next reuse event;
- double-buffered;
- placed in an independently owned pool/allocation or copied into an owned tensor.

Every copy appears in trace, operation count, and latency.

## 9. Measurement and cost model

### 9.1 Analytical seed

For each region/plan, estimate a vector rather than one score:

```text
T_compute       = flops / calibrated_compute_rate
T_weight        = weight_bytes / calibrated_shape_family_bandwidth
T_activation    = activation_and_materialization_bytes / calibrated_bandwidth
T_kv            = kv_bytes / calibrated_kv_bandwidth
T_orchestration = grids + joins + atomics + binding + copies

semantic_effective_bw_e2e  = semantic_bytes / complete_invocation_time
semantic_effective_bw_body = semantic_bytes / relevant_gpu_active_interval
physical_dram_bw           = counter_observed_dram_bytes / measured_interval
```

The seed lower bound is based on the dominant non-overlapped terms, with explicit contention when
independent branches share HBM. It prunes impossible plans; measurements decide winners.

Bandwidth is calibrated on the actual MIG instance and by shape family. Product peak is only a
mathematical lower bound. For the measured H200 `3g.71gb`, NVIDIA's 4/8 memory fraction implies about
2.4 TB/s from the 4.8 TB/s full-device specification; the old 500 GB/s name-based estimate was
wrong. L2 persisting-cache set-aside is not an architecture dependency because CUDA disables it
under MIG.

### 9.2 Unified candidate tournament

Every supported dense `LinearRegion` enters one uniform `CandidateSet` with four implementation
families: `CUBLASDX`, `CUTLASS_COLLECTIVE`, `CUTE_COMPOSED`, and `NATIVE_CUDA`. cuBLASDx is a
first-class required candidate family, not a late experiment: for every target/shape/precision cell
supported by its pinned public API, the generator must emit and measure legal shared-memory,
register-accumulator, and pipelined forms where applicable.

“Try everything” means exhaustive coverage of the bounded legal family, not an unbounded Cartesian
product. Target features, descriptor support, alignment, numerical policy, CTA topology, and
estimated resource limits reject illegal candidates first. The remaining search uses staged
successive halving:

1. compile and inspect each exact-shape body;
2. measure it standalone under a common input/output contract;
3. measure survivors inside a minimal persistent worker;
4. measure survivors inside the intended composite and candidate `EntryLaunchEnvelope`;
5. compile separate whole-entry beam candidates and select by the requested end-to-end objective.

All families compete through the same correctness, precision, traffic, and timing gates. Search
budget may favor cuBLASDx because it is the vendor-supported in-kernel BLAS path, but no family
receives a performance assumption. The final release entry contains only selected bodies; placing
all alternatives behind runtime branches would charge the entry for their combined code and
worst reachable resources.

### 9.3 Baseline suite

For each equivalent workload and numerical policy, measure as legal:

1. eager/reference correctness;
2. ordinary `torch.compile`;
3. strongest `torch.compile` CUDA-Graph/static-address mode;
4. direct vendor/CUTLASS operation baselines for hot regions;
5. static vendor graph replay;
6. GraCE-style indirect graph replay;
7. MegaBake strict and hybrid plans;
8. an equivalent serving engine for batching/quantized claims when available.

The first strong-baseline result is part of the design evidence: export plus
`torch.compile(mode="reduce-overhead")` measured 1.782 ms median versus 4.846 ms for the current
MegaBake invocation. Ordinary `torch.compile` at 5.691 ms remains useful launch-overhead evidence,
but it is not the acceptance baseline.

### 9.4 Mandatory measurements

- CPU wall latency and GPU event latency;
- full latency distribution and warm steady-state throughput;
- every CUDA kernel, memcpy, memset, and allocation event;
- GPU grid count and host submission count;
- semantic byte counts and both semantic effective-bandwidth definitions;
- counter-derived DRAM/L2 bytes, throughput, and cache rates;
- register count, stack frame, local-memory instructions/transactions, spills, SMEM, theoretical
  and achieved occupancy;
- scheduler/reset/binding time outside task bodies;
- model weight, activation, and KV bytes implied by the plan;
- correctness and model-quality results appropriate to numerical policy.

Profiler filtering may classify operations but may not remove them from totals. Task-local timers are
never used as a substitute for whole-entry time.

### 9.5 Acceptance

A performance artifact is accepted only when it:

- passes correctness, guard-region, and sanitizer checks;
- satisfies its numerical/quality policy;
- is stable across bucket shapes, layouts, and alignments;
- beats the strongest equivalent legal baseline by a configured noise margin;
- includes compatibility guards and a measured fallback.

A strict one-grid research artifact may be emitted slower, but its deficit must be explicit.

Project-level reliability is reported with a `SingleGridScorecard`: strict legality, strict win,
p50/p99 margin, and fallback result for every predeclared model x target x bucket x numerical-policy
cell. A hybrid fallback never increments the strict-win numerator.

## 10. Kernel program

### 10.1 Decode linear first

Generate exact `M=1`, `M=2`, and `M=4` families. `M`, dtype, layout, vector width, epilogue, and
scale format are compile-time constants wherever practical. Never retain an accumulator array sized
for an unused `M=64` path.

Candidate families:

1. cuBLASDx descriptors, including shared-memory, register-accumulator, and pipelined execution;
2. CUTLASS collectives adapted below the host launcher;
3. direct CuTe-composed tensor-core pipelines;
4. MegaBake-native CUDA, including warp/CTA/split-K streaming GEMV.

The four families share one candidate ABI and tournament. cuBLASDx is mandatory wherever its
pinned version supports the target and contract; being mandatory to search does not make it the
predetermined winner.

CuTe is also a foundational component underneath other NVIDIA paths. `CUTE_COMPOSED` specifically
means MegaBake directly controls the atoms, layouts, copies, and pipeline; candidate
canonicalization removes configurations that lower to an equivalent generated body.

Candidate tactics include warp-per-output K-parallel GEMV, CTA/split-K GEMV, padded-M tensor-core
projection, TMA warp specialization, and mixed-dtype W8/W4 variants. The 49,152-row SmolLM
vocabulary projection proves that a tensor-core path can win at `M=1`; exact M alone does not select
the family. Logical work count, resident worker count, block/warp topology, tile shape, stage count,
and SMEM are tuned together.

### 10.2 Transformer composites

Prioritize composites that remove traffic or joins:

1. replicated RMSNorm into decode linear;
2. fused gate/up dot products with SiLU-multiply epilogue;
3. grouped/concatenated QKV projection with output routing;
4. bias/residual/activation linear epilogues;
5. RoPE plus KV-cache append.

Mere adjacency or dispatch fusion is not sufficient.

### 10.3 Attention and KV cache

Attention is sequence-bucketed separately from linear. Decode candidates include:

- paged KV layouts;
- GQA/MQA-aware head mapping;
- single-block and split-KV/multi-block variants;
- reference, FP8, or INT8 KV cache under the numerical policy;
- TMA/vectorized movement and online softmax;
- a graph/vendor fallback until the owned variant wins.

The planner pivots variants using measured sequence-length and parallelism thresholds.

### 10.4 Code generation discipline

- emit only variants reachable from the bucket;
- do not concatenate every task family into one entry;
- select and validate one compatible `EntryLaunchEnvelope` for a strict artifact;
- keep logical work count separate from resident worker count;
- compile release and instrumented binaries separately;
- collect ptxas resources and disassembly for every candidate;
- reject unexpected local memory or spills in hot skinny kernels;
- retain source revision and license metadata for reused code.

## 11. Runtime model

The fixed-bucket runtime does only this:

1. fingerprint target and libraries;
2. select a compatible artifact and numerical/objective policy;
3. allocate stable arena, scheduler state, KV pages, and binding table;
4. load prepacked weights and owned code;
5. reconstruct, instantiate, and upload graph recipes once;
6. apply the documented invocation binding/state-reset operation;
7. launch one persistent entry or one top-level graph;
8. return output under the explicit lifetime contract.

It does not rediscover regions, rebuild schedules, infer vendor ABIs, clone dependency arrays, hide
output copies, or silently autotune on every process start.

For an autoregressive static session, a later graph recipe may use conditional graph nodes or device
tail launches for sampling/termination control. This remains a multi-grid GPU-controlled loop, not
one magically fused vendor kernel.

## 12. Implementation roadmap

### Phase 0 — freeze measurement truth

- count every CUDA operation and record CPU/GPU latency separately;
- store real-model benchmark artifacts;
- discover device/partition compute, memory, and cache resources; calibrate bandwidth; and remove
  product-name peak heuristics;
- attribute per-invocation counter/output clones and define `STATIC_SESSION`, `RUN_INTO`, and
  `BORROWED_OUTPUT` contracts;
- build ordinary, `reduce-overhead`/static-graph, and indirect-graph baselines;
- add an unfiltered operation timeline and `SingleGridScorecard` schema.

Exit: an empty/minimal invocation has a fully explained timeline.

This phase is bounded instrumentation work. The measured copies total only about 5.5 us of device
time; do not tune them while the entry body is roughly 3 ms behind the strong baseline.

### Phase 1 — first H200 validation, not a target assumption

- choose the hottest real `M=1` and `M=4` shapes;
- generate a decode-only entry with no dynamic `acc[64]`, prefill path, or unexpected hot-path
  local-memory traffic;
- enumerate the same exact shapes through cuBLASDx, CUTLASS collectives, direct CuTe, and
  MegaBake-native CUDA candidates;
- in a compatible pinned MathDx/CUDA toolchain, broadly search cuBLASDx descriptor tile, block,
  layout, alignment, architecture modifier, register/shared-memory output, and pipeline choices;
- stage the tournament from isolated bodies through a minimal persistent worker to separately
  compiled whole-entry beams;
- tune logical tiles, resident workers, launch envelope, and SMEM rather than reserving the device
  maximum;
- compare body, resources, SASS, traffic, and end-to-end phase time with cuBLASLt.

Exit: every frequent isolated shape reaches at least 80% of its vendor body, reaches 95% after
tuning or repays the deficit through measured fusion, and one strict segment has no unexplained
resource or timeline deficit.

### Phase 2 — architecture-family portability proof

- replay the same target-neutral facts, regions, and workload contract on at least one non-Hopper
  CUDA family, with SM80/A100 as the first intended case;
- select candidates from feature capability (`cp.async`/MMA or SIMT on SM80), never GPU-name
  conditionals or Hopper-only source;
- cross-compile and resource-inspect portable, SM80, and SM90 release artifacts in continuous
  validation even when only one performance machine is locally available;
- on real target hardware, calibrate bandwidth/launch costs, rebuild packed layouts, retune logical
  work and residency, and compare with that target's strongest baselines;
- keep H200 results only as candidate-order priors and create a separate scorecard cell.

Exit: one non-Hopper target independently passes correctness, entry-envelope, isolated-body, and
strict end-to-end gates. Until then, MegaBake is a generic compiler design with one validated CUDA
target, not a generally optimized CUDA compiler.

### Phase 3 — phase executor and traffic-removing composites

- implement the compact static `PhaseProgram`;
- add resource-class entry generation;
- implement `RECOMPUTE_PER_WORKER` RMSNorm-linear;
- implement GatedMLP and QKV composite candidates;
- demonstrate fewer HBM bytes/joins, not merely fewer dispatch cases.
- once body competitiveness is established, replace attributed per-call clones with preallocated
  epoch state and `RUN_INTO`/borrowed-output paths.

### Phase 4 — precision and prepacking

- add W8A16 and W4A16 prepack descriptors and kernels;
- measure activation conversion for FP8 plans;
- add model-quality gates and equivalent quantized baselines;
- make numerical policy part of artifact selection.

### Phase 5 — hybrid graph backend

- implement stable binding tables and fixed workspaces;
- build a batched GraCE-style prelude with version guards;
- connect resource-specific persistent and vendor nodes;
- accept the hybrid plan only when it beats both static graph and strict persistent alternatives.

### Phase 6 — attention, KV cache, and device-side decode control

- add paged/GQA decode attention and split-KV thresholds;
- add quantized KV policies;
- fuse RoPE/KV append where profitable;
- optionally keep sampling/termination in a conditional or tail-launched graph.

### Phase 7 — continuous batching

- introduce the separate admission scheduler;
- group compatible tokens so weight tiles serve multiple rows;
- tune batch wait against service-level latency;
- consider a dedicated scheduler CTA only after measuring its resource cost.

## 13. Explicit non-goals

- extracting and shipping private cuBLAS cubins;
- treating an opaque CUDA entry as a callable device subroutine;
- SASS lifting as a production dependency;
- a universal maximum-SMEM persistent entry;
- arbitrary per-worker busy-wait DAGs for fixed buckets;
- generic interpreted TMA, warp-role, or register-handoff instructions;
- claiming speedup by hiding copies from profiler counts;
- presenting quantized or batched throughput as reference batch-one latency;
- building a general training compiler before the decode vertical slice wins.

## 14. Design invariants

1. Semantics are fixed before execution planning.
2. Traffic and numerical policy are first-class.
3. Every candidate declares `DEVICE_CALLABLE`, `GRAPH_NODE`, or `HOST_ONLY`.
4. Vendor kernels always remain independent grids.
5. One host submission is never called one GPU grid.
6. Worker identity means CTA slot, not physical SM identity.
7. Machine pipelines live in compiled variants.
8. On-chip forwarding or recomputation requires a compiled contract.
9. Persistent entries are split when resource classes conflict.
10. Fixed buckets use phase programs; dynamic schedulers exist only for dynamic serving.
11. Every invocation copy/reset/lifetime operation is explicit and measured.
12. Compiled resources and replay measurements may revise the execution plan.
13. The strongest equivalent measured baseline gates acceptance.
14. Runtime binds and launches; it does not compile again.
15. Logical work count and resident worker count are never conflated.
16. A strict artifact has one measured, compatible entry launch envelope.
17. Hybrid coverage and strict-one-grid coverage are reported separately.
18. Target-neutral representations contain no product-name or instruction-family policy.
19. Candidate legality is driven by backend capabilities; performance selection is driven by exact-
    target measurements.
20. A result from one target fingerprint can seed but never validate another target.

## 15. Bottom line

The best MegaBake architecture is not one enormous universal binary containing every operator and a
general scheduler. It is a generic semantic/traffic compiler with pluggable target backends that
emits one resource-compatible, shape/precision/target-specific grid per supported bucket. It
recomputes cheap values when that avoids communication and, in the CUDA backend, uses CUDA Graphs
as the measured fallback for incompatible or vendor boundaries.

For batch-one reference decode, success means approaching the real weight-bandwidth floor while
removing materializations and joins. For serving throughput, success means reusing those streamed
weights across multiple tokens under a bounded latency policy. GraCE and private-kernel tracing are
valuable graph-binding and oracle tools; they are not the source of persistent composability.
