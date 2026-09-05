# MegaBake V2.2 IR and Reuse Plan

This document defines the normative compiler objects, legality contracts, and external-project reuse
boundaries for the traffic-first architecture. It must remain consistent with:

- [`MEGABAKE_V2_ARCHITECTURE.md`](./MEGABAKE_V2_ARCHITECTURE.md)
- [`MEGABAKE_V2_DATAFLOW_DIAGRAM.md`](./MEGABAKE_V2_DATAFLOW_DIAGRAM.md)
- [`grace_hack.md`](./grace_hack.md)

## 1. Executive summary

V2.2 retains three compiler representations:

1. `FXGraph + FactTables`
2. `RegionGraph`
3. `PlanTemplate`, frozen as `FinalExecutionPlan`

It retains two execution planes:

- `PersistentSegment` for owned `DEVICE_CALLABLE` code;
- `CudaGraphSegment` for separately launched `GRAPH_NODE` implementations.

The important V2.2 changes are contracts, not more IR layers:

- `WorkloadContract` separates latency/throughput, precision, execution domain, and a compound
  `InvocationContract`;
- `TrafficEstimate` replaces raw operation count as the useful analytical seed;
- `PrecisionLayoutPlan` includes weight/KV width, scale format, and offline packing;
- `ValueTransportPlan` includes deliberate recomputation per worker or cluster;
- `PersistentSegment` belongs to one compatible resource class;
- fixed buckets use `PhaseProgram`, while continuous batching has a separate serving scheduler;
- `InvocationContract` makes binding, reset, and output-copy costs impossible to hide;
- `MeasuredBaselineSuite` requires equivalent numerical and service objectives.

The core implementation thesis is:

> Generate excellent exact-shape device bodies, make weight and KV traffic explicit, compose only
> where a compiled contract removes traffic or joins, and use CUDA Graphs for resource or vendor
> boundaries that should remain separate.

## 2. IR design principles

### Principle A — add an IR only when it enables a new semantic decision

- `FXGraph + FactTables` establishes semantics and static facts.
- `RegionGraph` establishes transformer-level opportunities, traffic, and legal groupings.
- `PlanTemplate` establishes execution alternatives and becomes the measured final plan.

Code generation, resource inspection, profiling, and serialization are stages, not new semantic IRs.

### Principle B — compare only equivalent contracts

Every result is keyed by:

```text
performance objective
numerical policy
execution policy
invocation contract
workload bucket
target fingerprint
```

W4A16 throughput is not evidence about reference-precision batch-one latency. A copied owned output
is not compared with a borrowed arena view without charging the copy.

### Principle C — execution capability is a legality property

```text
DEVICE_CALLABLE
GRAPH_NODE
HOST_ONLY
```

A cost model chooses among legal candidates. It cannot turn a host handle, cubin entry, Triton
launch, or cuBLAS graph node into an in-grid device call.

### Principle D — semantic grouping and machine composition are different

`RegionGraph` may identify a `NormLinearComposite` opportunity. Actual register forwarding, shared-
memory forwarding, or per-worker recomputation exists only after generation and validation of a
`CompositeVariant` and `FusionContract`.

Dispatch adjacency does not prove fusion.

### Principle E — recomputation is a transport tactic

A small pure producer may be recomputed by every consuming CTA or cluster. This is represented like
any other value-transport choice and charged for compute and duplicate reads. It is not a semantic
rewrite and is not assumed profitable.

### Principle F — compiled resources can revise an execution plan

Early resource bounds prune impossible plans. Actual ptxas registers, spills, shared memory,
occupancy limits, and launch attributes may force a different resource split, worker count, or
composite choice before finalization.

### Principle G — measurement decides performance

Analytical models identify rooflines and clear losers. They do not prove that a persistent grid,
TMA pipeline, WGMMA tactic, quantization scheme, recomputation, or graph prelude wins.

### Principle H — runtime objects are not artifacts

The following are context-local and never serialized as deployment contracts:

- `CUfunction`, `CUmodule`, `CUkernel`, or library handles;
- instantiated CUDA graph executables;
- device addresses;
- raw private vendor parameter packs;
- profiler-created activity/callback pointers.

Artifacts contain owned code, public reconstruction recipes, prepacks, guards, and schemas.

## 3. Stage 0 objects

## 3.1 `CompileRequest`

```text
CompileRequest
  model_or_exported_program
  example_inputs
  example_state
  service_constraints
  requested WorkloadContract
  target_selector
```

## 3.2 `WorkloadContract`

```text
WorkloadContract
  performance_objective
  numerical_policy
  execution_policy
  InvocationContract
  execution_budgets
  quality_and_tolerance_policy
```

Enums:

```text
performance_objective =
  SINGLE_REQUEST_LATENCY |
  BOUNDED_LATENCY_THROUGHPUT |
  PREFILL_THROUGHPUT

numerical_policy =
  REFERENCE_FP16_BF16 |
  FP8 |
  W8A16 |
  W4A16

execution_policy =
  STRICT_SINGLE_GRID |
  HYBRID_GRAPH |
  CORRECTNESS_FIRST
```

Budgets include host submissions, GPU grids, persistent segments, HBM materializations, binding
bytes, batch wait, and host fallback permission.

## 3.3 `WorkloadBucketKey`

Target-neutral identity for:

```text
graph/model family
decode or prefill mode
active-token/query bucket
sequence/KV bucket
hidden/head/projection family
semantic feature flags
input source-dtype/shape/layout family
```

Numerical and target identity are not silently folded into it; artifact keys combine the bucket with
`WorkloadContract`, `TargetFingerprint`, and backend versions.

## 3.4 `TargetFingerprint`

```text
TargetFingerprint
  GPU UUID/model
  MIG profile and visible SM count
  compute capability
  driver/runtime/toolkit versions
  cuBLAS/cuBLASLt/cuDNN build IDs
  owned codegen and backend revisions
```

## 3.5 `DeviceCaps`

Facts and calibrated primitives:

```text
register/SMEM/cluster limits
supported MMA/WGMMA/TMA modes
cooperative launch and graph capabilities
device-updatable graph-node capabilities
measured streaming and KV bandwidth
measured launch, graph, join, atomic, and binding costs
supported code-generation targets
```

Calibrate the actual MIG instance. Do not infer usable bandwidth from the full-device product number.
Persisting-L2 set-aside is not a required optimization because CUDA disables it in MIG mode.

## 4. Representation 1 — `FXGraph + FactTables`

### 4.1 Purpose

Preserve exported PyTorch semantics while exposing all facts needed for region formation, precision
legality, state handling, and invocation contracts.

### 4.2 Graph payload

- canonical ATen-like nodes and use-def topology;
- placeholders, parameters, buffers, constants, and outputs;
- effect ordering;
- source/debug mapping;
- graph signature and state identity.

### 4.3 Required fact tables

#### `ShapeFact`

```text
value_id
rank
static_dims
symbolic_dims
bucket_constraints
```

#### `LayoutFact`

```text
value_id
dtype
strides
storage_offset
alignment
contiguity_and_layout_class
```

#### `AliasLifetimeFact`

```text
alias_group
base_value
view_transform
mutation_visibility
lifetime_interval
external_ownership
```

#### `MaterializationFact`

```text
producer_value
consumer
required_or_optional
reason
legal_transport_classes
```

#### `StateEffectFact`

```text
effect_kind = pure | read_state | write_state | atomic | external
affected_values
ordering_constraints
session_lifetime
```

#### `ConstantParameterFact`

```text
value_id
parameter_identity
original_dtype_layout
prepack_allowed
sharing_and_lifetime
```

#### `PrecisionLegalityFact`

```text
value_or_region
legal_input_output_dtypes
legal_accumulator_dtypes
scale_or_zero_point_requirements
quality_policy_dependencies
```

### 4.4 Allowed decisions

- decompositions and canonical rewrites;
- constant folding, CSE, and DCE;
- shape/layout/alias/effect propagation;
- virtual view versus required copy;
- parameter/prepack eligibility;
- numerical-policy legality facts;
- motif-matching preparation.

### 4.5 Forbidden decisions

- persistent versus graph execution;
- kernel family or quantization tactic selection;
- semantic fusion winner;
- worker, tile, stage, TMA, or scheduling choices;
- private vendor ABI assumptions.

## 5. Representation 2 — `RegionGraph`

### 5.1 Purpose

Represent semantic computations at the scale where weight, activation, KV, synchronization, and
composition decisions are meaningful.

### 5.2 Primitive region kinds

```text
LinearRegion
MatmulRegion
NormRegion
PointwiseRegion
DecodeAttentionRegion
PrefillAttentionRegion
RopeRegion
KVCacheRegion
EmbeddingRegion
SamplingRegion
GenericRegion
```

`GenericRegion` is semantic. External execution is a candidate property, not a region kind.

### 5.3 Bounded composite opportunities

```text
NormLinearComposite
QKVProjectionComposite
GatedMLPComposite
LinearEpilogueComposite
RopeKVAppendComposite
DecodeAttentionComposite
```

These are alternate legal groupings, not promises that a fused implementation exists.

### 5.4 `Region`

```text
Region
  region_id
  region_kind
  internal_nodes
  input_values
  output_values
  effects
  shape_layout_signature
  TrafficEstimate
  candidate_family_ids
  composite_neighbor_ids
  materialization_constraints
```

### 5.5 `TrafficEstimate`

```text
TrafficEstimate
  weight_bytes_by_storage_dtype
  activation_read_bytes
  activation_write_bytes
  kv_read_bytes
  kv_write_bytes
  workspace_bytes
  flops_by_math_dtype
  parallel_work_units
  reuse_distance_hints
```

This is a semantic lower-bound description. Candidate-specific packing, redundant reads,
recomputation, spills, and workspace traffic are added later.

### 5.6 `RegionEdge`

```text
RegionEdge
  producer_region
  consumer_region
  value_id
  layout_relation
  effects_and_ordering
  required_materialization
  legal_value_transports
  possible_fusion_endpoints
```

### 5.7 Allowed decisions

- transformer motif recovery;
- primitive region boundaries;
- a bounded set of semantic composite alternatives;
- candidate-family legality;
- traffic estimates and lower bounds;
- potential value transports and recomputation scopes;
- effect-safe materialization alternatives.

### 5.8 Forbidden decisions

- claiming registers or shared memory remain live;
- assigning CTAs or physical SMs;
- encoding TMA/cp.async/WGMMA operations;
- treating one region as one launch;
- changing numerical policy;
- treating a private vendor handle as callable device code.

## 6. Representation 3 — `PlanTemplate` to `FinalExecutionPlan`

### 6.1 `PlanTemplate`

```text
PlanTemplate
  plan_id
  WorkloadContract
  WorkloadBucketKey
  SegmentTemplate[]
  candidate_configuration_sets
  PrecisionLayoutPlan alternatives
  ValueTransportPlan alternatives
  BufferPlan
  BindingSchema
  InvocationContract alternatives
  policy_and_resource_constraints
```

### 6.2 `SegmentTemplate`

```text
SegmentTemplate
  segment_kind = persistent | cuda_graph | host_fallback
  covered_regions
  candidate_family_set
  resource_class_candidates
  configuration_neighborhood
  predecessor_successor_segments
  materialized_boundaries
```

### 6.3 `FinalExecutionPlan`

```text
FinalExecutionPlan
  selected PrecisionLayoutPlan
  selected ValueTransportPlan[]
  selected PersistentSegment[]
  selected CudaGraphSegment[]
  optional HostFallbackSegment[]
  final BufferPlan
  final BindingSchema
  final InvocationContract
  measured performance/traffic/resource breakdown
  strongest_equivalent_baseline_id
  compatibility_guards
  fallback_chain
```

### 6.4 Finalization gates

A plan is final only after:

1. reachable variants and composites are generated;
2. ptxas/resource/disassembly metadata is collected;
3. whole-entry resource legality is rechecked;
4. region and composite correctness tests pass;
5. quantized plans pass quality policy;
6. all-operation end-to-end profiles are recorded;
7. bounded refinement terminates;
8. the candidate beats or deliberately falls back to the strongest equivalent baseline.

### 6.5 Legal compiled feedback

Stage 5 may change:

- tile, stage, vector, cluster, and warp configuration;
- active and resident worker counts;
- static range versus atomic cursor;
- `ValueTransportPlan` and composite selection;
- resource-class segmentation;
- persistent versus graph choice in non-strict mode;
- static versus dynamic binding tactic.

It may not reinterpret graph semantics, silently change numerical policy, or broaden the requested
service objective.

## 7. Execution and composition contracts

## 7.1 `ExecutionCapability`

```text
enum ExecutionCapability {
  DEVICE_CALLABLE,
  GRAPH_NODE,
  HOST_ONLY
}
```

Rules:

- only `DEVICE_CALLABLE` enters `PersistentSegment`;
- `GRAPH_NODE` declares its actual grid count or subgraph shape;
- `HOST_ONLY` is illegal when host fallback is disabled;
- capability is versioned with its backend/toolkit adapter;
- a private `CUfunction` or extracted entry never becomes `DEVICE_CALLABLE` by observation alone.

## 7.2 `KernelVariant`

```text
KernelVariant
  variant_id
  region_kind
  implementation_family
  execution_capability
  supported_shape_layout_precision
  compile_time_parameters
  InputOutputContract
  ResourceEnvelope
  FusionEndpoint[]
  backend_source_revision
```

For `GRAPH_NODE`, the implementation is a public operation/graph recipe, not an owned in-grid body.

## 7.3 `ResourceEnvelope`

```text
ResourceEnvelope
  resource_class
  threads_per_cta
  warp_group_structure
  static_smem_bytes
  dynamic_smem_bytes
  registers_per_thread_actual
  spill_bytes_actual
  local_memory_bytes_actual
  max_resident_ctas
  cluster_shape
  launch_attributes
  tma_tensor_map_requirements
  minimum_arch_toolkit
```

Actual post-compile values are mandatory for finalization. One entry is charged the worst reachable
path and any unioned shared storage. Incompatible resource classes should usually become separate
entries.

## 7.4 `FusionEndpoint`

```text
FusionEndpoint
  value_role
  layout_id
  dtype
  ownership
  supported_transport
  producer_or_consumer_scope
```

## 7.5 `FusionContract`

```text
FusionContract
  producer_variant
  consumer_variant
  producer_scope = ONCE | PER_WORKER | PER_CLUSTER
  transport = REGISTER_FORWARD | SMEM_FORWARD | RECOMPUTE
  layout_dtype_size_alignment
  thread_warp_ownership_map
  synchronization_protocol
  lifetime
  recompute_flops_and_bytes
  combined_resource_envelope
```

`REGISTER_FORWARD` and `SMEM_FORWARD` require static composition into one `CompositeVariant`.
`RECOMPUTE` requires purity/effect safety and produces semantically identical values at the selected
scope.

Example: a `NormLinearComposite` may choose `RECOMPUTE_PER_WORKER`, allowing each output-tile CTA to
normalize the small activation into local shared memory before streaming its weight rows.

## 7.6 `ValueTransportPlan`

```text
ValueTransportPlan
  edge_id
  transport =
    ALIAS_VIEW |
    MATERIALIZE_HBM |
    REGISTER_FORWARD |
    SMEM_FORWARD |
    RECOMPUTE_PER_WORKER |
    RECOMPUTE_PER_CLUSTER
  selected_fusion_contract
  traffic_and_recompute_cost
```

## 7.7 `PrecisionLayoutPlan`

```text
PrecisionLayoutPlan
  default_policy
  TensorPrecisionPlan[]
  OperationPrecisionPlan[]
  PrepackDescriptor[]

TensorPrecisionPlan
  value_or_parameter_id
  storage_dtype
  scale_dtype
  scale_granularity
  zero_point_policy
  physical_layout

OperationPrecisionPlan
  region_or_variant_id
  input_dtypes
  accumulator_dtype
  output_dtype
```

`PrepackDescriptor` records source parameter identity, physical packed layout, padding, transpose,
scale/zero-point swizzle, alignment, backend revision, and validation hash. Packing occurs once at
artifact build/load, never in the timed invocation.

## 7.8 `PersistentSegment`

```text
PersistentSegment
  segment_id
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

`worker_count` is tuned and may be below the visible SM count. `active_worker_count` is phase-local
and may be below the resident count.

## 7.9 `PhaseProgram`

```text
PhaseProgram
  PhaseDesc[]
  WorkDesc[]
  CompletionEpoch[]

PhaseDesc
  phase_id
  composite_or_variant_id
  active_worker_count
  distribution
  work_range_or_cursor
  binding_slice
  completion_policy

WorkDesc
  region_instance
  static_tile_or_range
  variant_specific_small_parameters

CompletionEpoch
  producer_phase
  consumer_phase
  completion_requirement
  state_slot
```

Initial distributions:

```text
STATIC_RANGE
ATOMIC_CURSOR
ALL_ACTIVE_WORKERS
```

Initial completion policies:

```text
NONE
CTA_JOIN
GRID_JOIN
```

Rules:

- static ranges are preferred for fixed work;
- atomics exist only for measured imbalance;
- phase joins follow composite/semantic dependencies, not every FX node;
- counters are preallocated and use `IN_ENTRY_PARALLEL_RESET`, `EPOCH_TAGGED`, or graph memsets;
- there is no generic prefetch, warp-role, register, or handoff bytecode;
- there is no arbitrary per-worker busy-wait DAG in the fixed-bucket path.

## 7.10 `ServingSchedule`

```text
ServingSchedule
  admission_policy
  max_batch_wait_us
  fairness_policy
  compatibility_key
  request_state_layout
  grouped_work_queues
  optional_scheduler_cta_contract
```

This object exists only for `BOUNDED_LATENCY_THROUGHPUT`. It groups compatible tokens so a loaded
weight tile services multiple rows. It is not required by the single-request compiler path.

## 7.11 `CudaGraphSegment` and `GraphRecipe`

```text
CudaGraphSegment
  covered_regions
  GraphRecipe
  workspace_plan
  binding_update_map
  environment_guards

GraphRecipe
  public_library_operations
  owned_kernel_and_persistent_nodes
  copy_memset_conditional_nodes
  node_dependencies_and_edge_kinds
  capture_or_explicit_build_method
  device_updatable_node_requests
  upload_requirements
```

The runtime reconstructs and uploads the graph. Conditional nodes or device tail launches may drive
autoregressive control, but every contained kernel remains its own grid.

`PROGRAMMATIC_DEPENDENT` is a legal edge candidate only when the downstream node has useful work
independent of the producer and performs the mandated device synchronization before consuming its
results. Otherwise the edge is a full dependency.

## 7.12 `BindingSchema` and `InvocationContract`

```text
BindingSchema
  slot_id
  semantic_value_id
  value_kind = pointer | scalar | small_record
  size_alignment
  update_frequency
  persistent_consumers
  graph_node_update_targets

InvocationContract
  binding_mode = STATIC_SESSION | DYNAMIC_BINDINGS
  output_mode = RUN_INTO | BORROWED_OUTPUT | OWNED_OUTPUT
  reset_mode = IN_ENTRY_PARALLEL_RESET | EPOCH_TAGGED | GRAPH_MEMSET
  input_layout_alignment_lifetime
  output_ownership_and_reuse_point
  binding_transfer_kind_and_bytes
  required_copy_memset_nodes
```

Dynamic bindings use one compact pinned-host `BindingBlock` and stable device `BindingTable`.
`STATIC_SESSION` may eliminate that transfer. A GraCE-style prelude may copy confirmed binding values
into existing vendor-node ABI slots.

If parameter recovery is unsafe, choose host graph updates, fixed captured addresses, or fallback.
Never guess offsets. Every output clone/copy is explicitly represented.

## 8. Measurement objects

## 8.1 `MeasuredBaselineSuite`

```text
MeasuredBaselineSuite
  suite_id
  WorkloadContract equivalence class
  WorkloadBucketKey
  TargetFingerprint
  BaselineTrace[]
  strongest_equivalent_baseline_id
```

Each `BaselineTrace` records:

```text
baseline kind and public operation recipe
complete CUDA kernel/copy/memset/allocation sequence
host submissions and GPU grids
launch and resource metadata
CPU wall and GPU event distributions
DRAM/L2 traffic and cache counters
optional private ABI diagnostics
environment hashes and confidence samples
```

Required baselines where legal:

- ordinary `torch.compile`;
- strongest static-address/CUDA-Graph `torch.compile` mode;
- direct cuBLASLt/CUTLASS region baselines for hot linears;
- static vendor graph replay;
- GraCE-style indirect graph replay;
- equivalent quantized or serving engine for non-reference claims.

Profiler classification may group copies separately but never remove them from totals.

## 8.2 `CandidateMeasurement`

```text
CandidateMeasurement
  correctness_and_quality_status
  CPU_and_GPU_latency_distributions
  steady_state_throughput
  operation_timeline
  traffic_counters
  compiled_resources
  scheduler_binding_reset_breakdown
  implied_semantic_byte_counts
  baseline_margin
```

Task-local timers are diagnostic only. Whole-entry/whole-invocation time is authoritative.

## 8.3 `AutotuneDB`

Keyed by contract, bucket, target, candidate/backend revisions, packed layout, and relevant
alignments. It stores resource results, correctness/quality status, distributions, winner, baseline
margin, provenance, and invalidation keys.

## 8.4 `ArtifactPack`

```text
ArtifactPack
  format_version
  WorkloadContract
  WorkloadBucketKey
  compatibility_guards
  FinalExecutionPlan
  owned_code_objects
  GraphRecipe[]
  BindingSchema
  InvocationContract
  BufferPlan
  PrepackDescriptor[] and packed weights
  fallback_chain
  debug_and_measurement_provenance
```

No process-local CUDA handles, raw addresses, or private vendor code are required by the artifact.

## 9. Cost model and search boundary

## 9.1 Analytical vector

```text
T_compute       = candidate_flops / calibrated_math_rate
T_weight        = candidate_weight_bytes / calibrated_stream_bandwidth
T_activation    = activation_and_materialization_bytes / calibrated_bandwidth
T_kv            = kv_bytes / calibrated_kv_bandwidth
T_orchestration = grid + phase_join + atomic + binding + reset + copy costs
```

The model also records insufficient parallelism and shared-resource contention. Independent
bandwidth-heavy branches are not assumed to overlap for free.

## 9.2 Search variables

Search is bounded to already legal choices:

- implementation family;
- exact M/N/K and layout specialization;
- tile/vector/stage/warp/cluster configuration;
- active/resident worker count;
- reference/FP8/W8/W4 tactic allowed by policy;
- weight prepack layout;
- composite and value transport;
- resource-class segment boundary;
- persistent versus graph candidate;
- binding and state-reset tactic.

Search does not rediscover graph semantics, enumerate arbitrary partitions, or invent a new
scheduler architecture.

## 10. Kernel-family contracts

## 10.1 Reference decode linear

Generate separate `M=1`, `M=2`, and `M=4` variants. Exact M is compile-time visible so accumulator
arrays, loops, and output ownership are bounded correctly. Reject hot variants with unexplained
local memory or spills.

Candidate tactics:

- coalesced SIMT streaming GEMV;
- transposed small-N tensor-core/WGMMA formulation;
- TMA producer plus compute warp groups;
- MPK/CuTe warp-specialized task;
- optional cuBLASDx block/device implementation.

Worker count, SMEM size, vector width, tile shape, and pipeline stage count are selected together.

## 10.2 Quantized linear

Initial families:

- W8A16 per-channel or groupwise;
- W4A16 groupwise/AWQ/GPTQ-compatible physical layouts;
- FP8 with measured activation scaling/conversion.

Scale metadata is packed adjacent or swizzled exactly as the device body expects. Model-quality
validation is mandatory.

## 10.3 Transformer composites

### `NormLinearComposite`

Candidate transports:

- materialize normalized activation;
- normalize once and SMEM-forward within one CTA when tiling permits;
- `RECOMPUTE_PER_WORKER` and retain a local normalized vector.

### `GatedMLPComposite`

Compute gate and up dot products together, apply SiLU and multiply before storing. The ideal variant
streams both weight rows but stores only the product and removes pointwise phases/joins.

### `QKVProjectionComposite`

Use concatenated or grouped packed weights, reuse the activation, route differing GQA output ranges,
and optionally apply compatible Q/K epilogues. It must preserve observable output/state semantics.

### `RopeKVAppendComposite`

Apply positional rotation and append K/V directly into the chosen paged-cache layout where effect
ordering and ownership permit.

## 10.4 Attention

Decode attention variants are keyed by sequence bucket, head dimension, GQA ratio, cache dtype, page
layout, and available parallelism. Candidate families include:

- single-block per head;
- split-KV/multi-block with reduction;
- GQA/MQA-specialized/XQA-like mapping;
- paged KV traversal;
- reference, FP8, or INT8 KV cache;
- TMA/vectorized loads and online softmax.

Prefill remains a different region/candidate family and may use a vendor graph node until owned
tensor-core kernels win.

## 10.5 Code generation

- generate only variants reachable from an artifact;
- split release and profiling builds;
- never concatenate every operation family into one universal entry by default;
- record ptxas resources, disassembly hash, backend revision, and source license;
- make phase descriptors compact and preferably constant/readonly;
- make hot paths static enough for compiler-visible loop and array bounds.

## 11. External project reuse

## 11.1 PyTorch Export and Inductor

### Reuse

- `torch.export` graph/signature machinery;
- decompositions, FakeTensor, and symbolic/bucketed shapes;
- suitable alias/layout/effect analysis;
- pattern matching, cleanup, and code-cache discipline;
- generated-epilogue and autotuning ideas.

### Avoid

- importing the launched-kernel scheduler as the persistent runtime;
- equating one fused Triton launch with all forms of persistent composition;
- relying on private internal APIs without versioned adapters.

Integration: Stages 1–3 and baseline construction.

## 11.2 CUTLASS and CuTe

### Reuse directly

- Hopper MMA/WGMMA atoms and layouts;
- TMA/copy pipelines and barriers;
- collective mainloops and epilogues;
- grouped/persistent tile-scheduler concepts;
- mixed-dtype INT4/INT8/FP8 building blocks and packed layouts.

### Adapt

- use layers below the ordinary host `Device` adapter;
- accept MegaBake worker/tile identity;
- expose `InputOutputContract`, `ResourceEnvelope`, and `FusionEndpoint`;
- generate only exact bucket-required instantiations.

### Avoid

- calling a host adapter “device composition”;
- dropping a whole universal kernel unchanged into a phase program;
- copying tile parameters without their pipeline/resource contract.

Integration: Stage 5, default owned compute backend.

## 11.3 Mirage MPK

### Reuse where license and interfaces permit

- pinned Apache-licensed Hopper TMA/WGMMA task bodies;
- decode/linear, norm, gated-MLP, and relevant attention implementations;
- low-level pipeline and synchronization helpers;
- tests and shape assumptions attached to the selected source revision.

### Adapt

- worker-local task identity and event concepts;
- resource-aware persistent execution;
- grouped/dynamic scheduler ideas only for the later serving path.

### Avoid initially

- importing the entire compiler/runtime;
- adopting a scheduler CTA or maximum resource reservation before fixed-bucket measurements justify
  it.

Integration: Stage 5 variants; later `ServingSchedule` work.

## 11.4 cuBLASDx

Use as an optional pinned `DEVICE_CALLABLE` backend for supported block/device BLAS operations and
register-accumulator fusion. It does not expose private cuBLAS host-library kernels. Its early-access
and toolkit constraints must be guarded, and every variant must pass whole-entry resource and replay
tests.

## 11.5 CUDA Graphs and GraCE

### Reuse directly

- public graph construction/capture, upload, replay, conditional, and device-launch APIs;
- fixed user-owned library workspaces;
- device-updatable graph nodes on confirmed stacks;
- explicit dependency edges and batched updates.

### Adapt from GraCE

- stable pointer/scalar cells or a compact `BindingTable`;
- one batched prelude for confirmed vendor-node parameter slots;
- profile-guided selection among static addresses, host updates, and device updates.

### Never infer

- the vendor function signature or body changed;
- a graph node became a device function;
- a private parameter pack is stable or portable;
- graph replay removed kernel grid boundaries.

Integration: baseline suite, hybrid graph construction, and optional GPU-controlled decode loop.

## 11.6 CUPTI, Nsight Compute, and CUDA binary tools

### Reuse

- API callbacks for actual launch arguments and live function identity;
- activity records for complete timelines and correlation;
- module resource callbacks for loaded code diagnostics;
- Nsight Compute roofline, memory, scheduler, and occupancy sections;
- `cuobjdump`/`nvdisasm` for owned/vendor comparison where permitted.

### Uses

- build `MeasuredBaselineSuite`;
- measure hidden scheduler/binding/reset time;
- detect spills/local memory and resource ceilings;
- learn vendor multi-kernel plans and seed owned tactics;
- compare traffic, issue behavior, and achieved bandwidth.

### Avoid

- treating activity records as argument-byte sources;
- treating private symbols/ABI fields as production contracts;
- redistributing proprietary binaries without authorization.

## 11.7 TensorRT-LLM or equivalent serving engines

Use only as an equivalent systems baseline and implementation reference for:

- continuous/in-flight batching;
- paged KV caches;
- GQA/MQA decode attention and multi-block thresholds;
- weight-only quantization and quantized KV caches.

Do not import its runtime architecture into the batch-one compiler by default.

## 11.8 Triton

An ordinary Triton entry is `GRAPH_NODE`. It becomes `DEVICE_CALLABLE` only if MegaBake owns a
separate lowering that emits a compatible in-grid device body and resource contract. Use Triton as a
strong launched candidate/baseline and source of code-generation ideas, not as an assumed callable
subroutine.

## 11.9 Helion, Luminal, and TileIR

- From Helion, reuse bounded late-search discipline and cached measured winners.
- From Luminal, reuse small-IR and specialization discipline.
- Keep TileIR deferred until CuTe/MPK exposes a demonstrated portability or maintenance gap and
  TileIR can emit a compatible body with resource metadata.

None replaces the three V2.2 representations.

## 11.10 Private cuBLAS extraction

Research uses:

- trace the live selected launch as an oracle;
- test frozen direct host relaunch to isolate dispatch;
- inspect the loaded module, resources, and SASS where legally allowed;
- compare owned variants against the exact vendor result.

Never a production dependency:

- no private handle, parameter pack, or cubin in `ArtifactPack`;
- no guessed offsets;
- no claim that a kernel entry is a device subroutine;
- no unsupported SASS lifting in the critical path.

## 12. Reuse decision matrix

| System | Direct reuse | Conceptual adaptation | Explicitly avoid |
|---|---|---|---|
| PyTorch Export/Inductor | export, facts, decompositions, matching | tuning/code-cache discipline | launched scheduler as persistent runtime |
| CUTLASS/CuTe | collectives, TMA/WGMMA, grouped schedulers, mixed dtype | worker/tile adapters and contracts | host adapter as fake device composition |
| Mirage MPK | selected pinned task bodies/helpers | workers/events; later serving scheduler | whole compiler/runtime initially |
| cuBLASDx | optional device-callable backend | block/full-device design | assuming it exposes cuBLAS internals |
| CUDA Graph/GraCE | public graph/update APIs | compact binding prelude | claiming vendor ABI/body modification |
| CUPTI/Nsight/binary tools | tracing, counters, diagnostics | candidate seeding | private ABI as deployment interface |
| TensorRT-LLM | equivalent serving baseline | batching/KV/attention ideas | forcing serving scheduler into B=1 path |
| Triton | graph-node candidate | generated-kernel ideas | assuming ordinary entry is callable |
| Helion/Luminal | none required | bounded search and small-IR discipline | importing semantic/compiler stacks |
| TileIR | deferred optional backend | future portability experiment | replacing region/plan IR |

## 13. Migration from V2.1

| V2.1 concept | V2.2 replacement | Reason |
|---|---|---|
| latency objective implicit | `WorkloadContract.performance_objective` | separate B=1 latency from serving throughput |
| FP16/BF16 first regime only | explicit `numerical_policy` and `PrecisionLayoutPlan` | weight width dominates decode traffic |
| one `MeasuredBaselineTrace` | `MeasuredBaselineSuite` of equivalent traces | require strongest comparable mode |
| `estimated_flops_bytes` | structured `TrafficEstimate` | distinguish weights, activations, KV, and workspace |
| materialize or same-worker handoff | `ValueTransportPlan` including recomputation scope | cheap producer duplication can beat communication |
| `WorkerProgram[] or PhaseProgram` | fixed-bucket `PhaseProgram` only | remove unnecessary arbitrary worker scheduling |
| dynamic worker machinery deferred | separate `ServingSchedule` | admit complexity only when batching reuses weights |
| segment has resource envelope | segment also has explicit `resource_class` | avoid universal maximum-SMEM entrypoints |
| binding-table update each invocation | static and dynamic binding tactics | stable sessions can eliminate updates |
| output behavior implicit | compound `InvocationContract` | combine binding/reset/output choices and prohibit hidden clones |
| task profiling as main breakdown | complete operation timeline plus task diagnostics | include scheduler, reset, copy, and idle gaps |

## 14. Suggested module layout

```text
src/megabake/v2/
  contract/
    workload.py
    invocation.py
    numerical.py
  capture/
    export.py
    signatures.py
  facts/
    shapes.py
    layouts.py
    aliases_lifetimes.py
    effects.py
    precision.py
  regions/
    ir.py
    recover.py
    traffic.py
    composites.py
  target/
    fingerprint.py
    caps.py
    calibrate.py
  baseline/
    suite.py
    trace.py
    torch_compile.py
    cuda_graph.py
    vendor.py
    cupti/
  plan/
    template.py
    precision_layout.py
    value_transport.py
    resources.py
    buffers.py
    binding.py
    enumerate.py
    finalize.py
  backend/
    variant.py
    cute/
    mpk/
    cublasdx/
    graph/
  tune/
    replay.py
    profile.py
    search.py
    database.py
    acceptance.py
  artifact/
    schema.py
    serialize.py
  runtime/
    loader.py
    session.py
    graph_cache.py
    binding.py
    serving.py
    launch.py

src/cuda_v2/
  runtime/
    persistent_entry.cu
    phase_program.cuh
    scheduler_state.cuh
    binding_table.cuh
  variants/
    skinny_linear/
    linear_heavy/
    composites/
    attention/
    compact/
  graph/
    binding_prelude.cu
  common/
    contracts.cuh
    synchronization.cuh
    quant_layouts.cuh
```

Each artifact generates or links only required variants. Instrumented and release entries are
different code objects.

## 15. Initial vertical-slice contract

The first milestone is one reference-precision real-model block, not every compiler component:

```text
one exported decode block at M=1 and M=4
  -> facts and TrafficEstimate
  -> NormRegion + LinearRegion + legal NormLinearComposite
  -> exact-size SIMT and MPK/CuTe DEVICE_CALLABLE candidates
  -> cuBLASLt GRAPH_NODE baseline/candidate
  -> fixed PhaseProgram and resource-specific entry
  -> STATIC_SESSION + RUN_INTO invocation paths
  -> compiled resource and all-operation timeline
  -> strict/hybrid replay against MeasuredBaselineSuite
  -> guarded ArtifactPack
```

Exit criteria:

- numerical agreement under the reference policy;
- no sanitizer or guard-region findings;
- zero unexplained copies, allocations, or launches;
- actual registers, local memory, spills, SMEM, and occupancy recorded;
- actual weight/activation traffic compared with the semantic lower bound;
- repeatable CPU/GPU latency distributions;
- isolated linear competitive with the vendor baseline;
- strict deficit, if any, completely attributed;
- no private vendor handle or code required by the artifact.

Only after this should the project add `GatedMLPComposite`, W4/W8, hybrid GraCE binding, attention,
or continuous batching in the roadmap order.

## 16. Primary implementation anchors

Reference snapshot: 2026-09-05.

- [GraCE OSDI page](https://www.usenix.org/conference/osdi26/presentation/ghosh)
- [CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUDA device graph-node update APIs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
- [CUDA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA L2 cache control and MIG restriction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [CUPTI callback API](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [cuBLAS CUDA Graph support](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support)
- [cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/)
- [CUTLASS repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS grouped scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [CUTLASS Hopper INT4/BF16 example](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu)
- [Mirage MPK repository](https://github.com/mirage-project/mirage)
- [Mirage MPK Hopper GEMM body](https://raw.githubusercontent.com/mirage-project/mirage/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh)
- [TensorRT-LLM paged attention and in-flight batching](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)
- [TensorRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/features/attention.html)
- [NVIDIA matrix-multiplication performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

## 17. Final design rule

MegaBake-specific code should focus on the pieces that existing systems do not provide together:
traffic-aware transformer region formation, explicit execution capability, exact invocation and
value-transport contracts, resource-compatible persistent composition, and measured selection
against strong graph/vendor baselines.

Everything else should be reused below a stable adapter: PyTorch for semantics, CuTe/MPK/cuBLASDx
for hard device math, CUDA Graphs for multi-grid orchestration, CUPTI/Nsight for truth, and serving
systems as equivalent batching/quantization baselines.
