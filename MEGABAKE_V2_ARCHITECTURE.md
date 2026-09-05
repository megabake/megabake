# MegaBake V2.2 Architecture

Status: second redesign after the GraCE/CUDA investigation and a source-level performance audit of
the current runtime. This document is normative with:

- [`MEGABAKE_V2_DATAFLOW_DIAGRAM.md`](./MEGABAKE_V2_DATAFLOW_DIAGRAM.md)
- [`MEGABAKE_V2_IR_AND_REUSE_PLAN.md`](./MEGABAKE_V2_IR_AND_REUSE_PLAN.md)
- [`grace_hack.md`](./grace_hack.md)

## 1. Executive decision

MegaBake V2.2 is a **traffic-first, bucket-specialized inference compiler**. Its purpose is not to
maximize the amount of code placed in one kernel. Its purpose is to minimize measured end-to-end
cost under an explicit latency, throughput, numerical, and execution-domain contract.

The design has two execution planes:

1. **Persistent plane** — owned `DEVICE_CALLABLE` implementations execute inside resource-specific
   persistent CUDA grids.
2. **Graph plane** — CUDA Graphs orchestrate persistent entries and separately launched high-quality
   kernels such as cuBLASLt, cuDNN, or standalone generated kernels.

Strict research mode may require one persistent grid. Default performance mode is free to use both
planes in one top-level graph submission. One host submission and one GPU grid remain different
quantities.

The compiler has only three real representations:

1. `FXGraph + FactTables`
2. `RegionGraph`
3. `PlanTemplate`, frozen as `FinalExecutionPlan` after code generation and measurement

V2.2 changes the previous proposal in six important ways:

- weight, activation, KV-cache, and materialization bytes are first-class costs;
- precision and prepacked layout are first-class plan choices;
- fixed buckets use a small compiled `PhaseProgram`, not arbitrary per-worker busy-wait DAGs;
- a cheap producer may be recomputed `PER_WORKER` or `PER_CLUSTER` instead of materialized to HBM;
- persistent entrypoints are split by compiled resource class rather than sharing one maximum-SMEM
  universal kernel;
- per-call binding, scheduler reset, and output ownership are explicit `InvocationContract` costs.

The first performance proof remains narrow: reference-precision decode on Hopper at `M=1..4`.
Weight-only quantization and continuous batching are separate numerical/serving regimes, but they
are represented from the beginning because they attack the fundamental decode bottleneck: weight
traffic.

## 2. Physical facts that determine the architecture

### 2.1 Persistent launch is not persistent model storage

For a batch-one linear layer, nearly every weight is consumed once per token. Model weights are far
larger than the combined register, shared-memory, and L2 capacity of the target MIG partition.
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
- one-grid composition requires owned device code from CuTe/CUTLASS, Mirage MPK, cuBLASDx, or a
  MegaBake implementation.

### 2.3 One host submission is not one GPU grid

V2.2 counts these separately:

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

V2.2 therefore permits these value transports:

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

V2.2 normally emits separate entries for resource classes such as:

```text
LINEAR_HEAVY      TMA/WGMMA or large mixed-dtype pipelines
SKINNY_STREAMING  M=1..4 GEMV/skinny GEMM variants
ATTENTION_HEAVY   paged/split-KV attention pipelines
COMPACT           reductions, RoPE, pointwise and small composites
```

CUDA Graphs may connect these entries cheaply. A single-grid artifact is retained only when its
measured fusion savings exceed its resource penalty.

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
max_resident_ctas
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
  worker_count
  threads_per_worker
  dynamic_smem_bytes
  launch_attributes
  PhaseProgram
  selected_variant_ids
  binding_map
  state_reset_policy
  combined_resource_envelope
```

`worker_count` is not automatically the SM count. It is tuned with the actual MIG partition and
variant. Each phase may activate a subset of resident workers.

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
  distribution = STATIC_RANGE | ATOMIC_CURSOR | ALL_ACTIVE_WORKERS
  work_range_or_cursor
  binding_slice
  completion = NONE | CTA_JOIN | GRID_JOIN
```

Rules:

- static ranges are preferred when work is known;
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

V2.2 has a fast static path and a general dynamic path.

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
T_compute      = flops / calibrated_compute_rate
T_weight       = weight_bytes / calibrated_stream_bandwidth
T_activation   = activation_and_materialization_bytes / calibrated_bandwidth
T_kv           = kv_bytes / calibrated_kv_bandwidth
T_orchestration= grids + joins + atomics + binding + copies
```

The seed lower bound is based on the dominant non-overlapped terms, with explicit contention when
independent branches share HBM. It prunes impossible plans; measurements decide winners.

Bandwidth is calibrated on the actual MIG instance. Full-GPU peak bandwidth is not divided and
treated as measured truth. L2 persisting-cache set-aside is not an architecture dependency because
CUDA disables it under MIG.

### 9.2 Baseline suite

For each equivalent workload and numerical policy, measure as legal:

1. eager/reference correctness;
2. ordinary `torch.compile`;
3. strongest `torch.compile` CUDA-Graph/static-address mode;
4. direct vendor/CUTLASS operation baselines for hot regions;
5. static vendor graph replay;
6. GraCE-style indirect graph replay;
7. MegaBake strict and hybrid plans;
8. an equivalent serving engine for batching/quantized claims when available.

### 9.3 Mandatory measurements

- CPU wall latency and GPU event latency;
- full latency distribution and warm steady-state throughput;
- every CUDA kernel, memcpy, memset, and allocation event;
- GPU grid count and host submission count;
- achieved DRAM/L2 traffic and cache rates;
- register count, spills, SMEM, theoretical and achieved occupancy;
- scheduler/reset/binding time outside task bodies;
- model weight, activation, and KV bytes implied by the plan;
- correctness and model-quality results appropriate to numerical policy.

Profiler filtering may classify operations but may not remove them from totals. Task-local timers are
never used as a substitute for whole-entry time.

### 9.4 Acceptance

A performance artifact is accepted only when it:

- passes correctness, guard-region, and sanitizer checks;
- satisfies its numerical/quality policy;
- is stable across bucket shapes, layouts, and alignments;
- beats the strongest equivalent legal baseline by a configured noise margin;
- includes compatibility guards and a measured fallback.

A strict one-grid research artifact may be emitted slower, but its deficit must be explicit.

## 10. Kernel program

### 10.1 Decode linear first

Generate exact `M=1`, `M=2`, and `M=4` families. `M`, dtype, layout, vector width, epilogue, and
scale format are compile-time constants wherever practical. Never retain an accumulator array sized
for an unused `M=64` path.

Candidate sources:

1. selected pinned Mirage MPK Hopper task bodies;
2. CUTLASS/CuTe collectives adapted below the host launcher;
3. optional cuBLASDx device-callable components;
4. a small custom streaming GEMV only where it measures best.

Candidate tactics include tuned SIMT GEMV, transposed small-N WGMMA, TMA warp specialization, and
mixed-dtype W8/W4 variants. Worker count, tile shape, stage count, and SMEM are tuned together.

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

### Phase 0 — make measurement truthful and remove housekeeping

- count every CUDA operation and record CPU/GPU latency separately;
- store real-model benchmark artifacts;
- measure actual MIG bandwidth and resource limits;
- remove per-invocation counter clones and hidden output clones;
- add `STATIC_SESSION`, `RUN_INTO`, and `BORROWED_OUTPUT` paths;
- build ordinary, static-graph, and indirect-graph baselines.

Exit: an empty/minimal invocation has a fully explained timeline.

### Phase 1 — one competitive reference-precision decode linear

- choose the hottest real `M=1` and `M=4` shapes;
- generate exact-size variants with no dynamic interpreter arrays;
- port one MPK/CuTe TMA pipeline and retain a tuned SIMT candidate;
- tune worker count and SMEM rather than reserving the device maximum;
- compare body, resources, SASS, traffic, and end-to-end phase time with cuBLASLt.

Exit: isolated owned linear is competitive and one strict segment has no unexplained deficit.

### Phase 2 — phase executor and traffic-removing composites

- implement the compact static `PhaseProgram`;
- add resource-class entry generation;
- implement `RECOMPUTE_PER_WORKER` RMSNorm-linear;
- implement GatedMLP and QKV composite candidates;
- demonstrate fewer HBM bytes/joins, not merely fewer dispatch cases.

### Phase 3 — precision and prepacking

- add W8A16 and W4A16 prepack descriptors and kernels;
- measure activation conversion for FP8 plans;
- add model-quality gates and equivalent quantized baselines;
- make numerical policy part of artifact selection.

### Phase 4 — hybrid graph backend

- implement stable binding tables and fixed workspaces;
- build a batched GraCE-style prelude with version guards;
- connect resource-specific persistent and vendor nodes;
- accept the hybrid plan only when it beats both static graph and strict persistent alternatives.

### Phase 5 — attention, KV cache, and device-side decode control

- add paged/GQA decode attention and split-KV thresholds;
- add quantized KV policies;
- fuse RoPE/KV append where profitable;
- optionally keep sampling/termination in a conditional or tail-launched graph.

### Phase 6 — continuous batching

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

## 15. Bottom line

The best MegaBake architecture is not one enormous kernel containing every operator and a general
scheduler. It is a small compiler that knows when one grid is valuable, builds excellent
shape/precision-specific device bodies, recomputes cheap values when that avoids communication,
splits incompatible resource classes, and uses CUDA Graphs for the remaining boundaries.

For batch-one reference decode, success means approaching the real weight-bandwidth floor while
removing materializations and joins. For serving throughput, success means reusing those streamed
weights across multiple tokens under a bounded latency policy. GraCE and private-kernel tracing are
valuable graph-binding and oracle tools; they are not the source of persistent composability.
