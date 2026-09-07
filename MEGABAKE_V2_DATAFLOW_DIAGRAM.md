# MegaBake V2.3 Detailed Dataflow

This is the diagram-first companion to
[`MEGABAKE_V2_ARCHITECTURE.md`](./MEGABAKE_V2_ARCHITECTURE.md). Object names, stage numbers, and
execution distinctions are normative and match
[`MEGABAKE_V2_IR_AND_REUSE_PLAN.md`](./MEGABAKE_V2_IR_AND_REUSE_PLAN.md). The measured H200 evidence
and revised north-star gates are in
[`MEGABAKE_V2_GPU_REANALYSIS.md`](./MEGABAKE_V2_GPU_REANALYSIS.md).

## 1. Complete compile-time flow

```mermaid
flowchart TB
  subgraph Input["Inputs"]
    A["CompileRequest<br/>model or ExportedProgram<br/>example inputs and state<br/>service constraints"]
    B["Orthogonal contract axes<br/>performance objective<br/>numerical policy<br/>execution policy<br/>compound invocation contract"]
    C["Target discovery<br/>accelerator/backend/ISA<br/>device or partition identity"]
    D["Optional priors<br/>HF metadata<br/>previous tuning data"]
  end

  subgraph S0["Stage 0 — Contract, bucket, and target"]
    E["WorkloadContract"]
    F["WorkloadBucketKey<br/>target-neutral shapes and mode"]
    G["TargetArchitectureKey + TargetFingerprint"]
    H["DeviceCaps<br/>device/partition resources<br/>measured shape-family bandwidth<br/>launch/sync costs<br/>resource limits"]
    H0["TargetBackend registry<br/>portable + architecture-family generators"]
  end

  subgraph S1["Stage 1 — Export, canonicalize, facts"]
    I["FXGraph<br/>canonical semantics"]
    J["FactTables<br/>shape/layout/alias<br/>constant/effect/lifetime<br/>precision legality"]
  end

  subgraph S2["Stage 2 — Regions, traffic, transport"]
    K["RegionGraph<br/>primitive regions<br/>bounded composites"]
    L["TrafficEstimate<br/>weight/activation/KV/workspace bytes<br/>flops and parallel work"]
    M["ValueTransport alternatives<br/>HBM materialize<br/>register/SMEM forward<br/>recompute per worker/cluster"]
  end

  subgraph S3["Stage 3 — Candidates and baseline suite"]
    N["CandidateSet<br/>feature-legal portable + architecture candidates<br/>DEVICE_CALLABLE / backend node / host"]
    O["MeasuredBaselineSuite<br/>ordinary and strongest low-overhead torch.compile<br/>vendor region baselines<br/>static/indirect graphs<br/>equivalent serving baseline"]
  end

  subgraph S4["Stage 4 — Bounded plan enumeration"]
    P["Traffic/resource legality pruning"]
    Q["PlanTemplate set<br/>precision/layout alternatives<br/>value transport<br/>resource segments<br/>invocation contracts"]
  end

  subgraph S5["Stage 5 — Generate, compile, measure, refine"]
    R["Generate reachable variants<br/>exact shape/precision<br/>composites and graph recipes"]
    S["Compile and inspect<br/>entry launch envelope<br/>registers/stack/local/spills/SMEM<br/>residency/attributes/SASS"]
    T["Correctness and quality gates"]
    U["Replay and profile<br/>unfiltered GPU operations<br/>semantic and physical traffic<br/>end-to-end cost"]
    V["Bounded refinement<br/>tiles/stages/workers<br/>transport/resource splits"]
    W["FinalExecutionPlan"]
    X["AutotuneDB"]
  end

  subgraph S6["Stage 6 — Artifact emission"]
    Y["ArtifactPack<br/>target key + compatibility guards<br/>owned code and prepacks<br/>phase/graph recipes<br/>support level, strict score, fallbacks"]
  end

  A --> E
  B --> E
  A --> F
  C --> G
  C --> H
  G --> H0
  A --> I
  I --> J
  E --> J
  J --> K
  K --> L
  K --> M
  D -.priors.-> K
  K --> N
  L --> N
  H0 --> N
  E --> O
  F --> O
  G --> O
  H --> O
  D -.cached results.-> O
  N --> P
  O --> P
  L --> P
  M --> P
  H --> P
  P --> Q
  Q --> R
  H0 --> R
  R --> S
  S --> T
  T --> U
  U --> V
  V -->|bounded execution-level revision| Q
  U --> X
  V --> W
  W --> Y
  X --> Y
  E --> Y
  F --> Y
  G --> Y
```

Stage 5 may revise implementation configuration, value transport, or resource segmentation. It may
not reopen exported semantics or invent a different numerical policy.

## 1.1 Generic-to-target lowering

```mermaid
flowchart LR
  A["Target-neutral package<br/>FX facts, RegionGraph, contracts"] --> B["TargetArchitectureKey"]
  B --> C{"Backend + feature legality"}
  C --> D["Portable correctness candidates"]
  C --> E["Architecture-family candidates<br/>for example SM80 or SM90"]
  D --> F["Compile for exact device/partition"]
  E --> F
  F --> G["Inspect EntryLaunchEnvelope"]
  G --> H["Calibrate and autotune on target"]
  H --> I["Exact-fingerprint winner cache"]
  I --> J["Guarded target-specific ArtifactPack"]
```

The compiler is generic because the semantic/traffic pipeline and backend contract are shared. The
output is specialized because optimal instruction families, layouts, residency, and timing depend
on the exact target. Architecture-family measurements are search priors; only exact-fingerprint
measurements can select a deployed winner.

## 2. The three compiler representations

```mermaid
flowchart LR
  A["FXGraph + FactTables<br/>semantics and facts"]
  B["RegionGraph<br/>traffic-bearing semantic opportunities"]
  C["PlanTemplate<br/>bounded execution family"]
  D["FinalExecutionPlan<br/>same plan schema, measured and frozen"]

  A -->|motifs and legal groupings| B
  B -->|capabilities, transports, segment choices| C
  C -->|generate, compile, profile, select| D
```

`PrecisionLayoutPlan`, `ValueTransportPlan`, `InvocationContract`, and `ResourceEnvelope` are plan
objects, not additional semantic IRs.

## 3. Workload-contract axes

```mermaid
flowchart TB
  A["WorkloadContract"] --> B["Performance objective"]
  A --> C["Numerical policy"]
  A --> D["Execution policy"]
  A --> E["Invocation contract"]

  B --> B1["SINGLE_REQUEST_LATENCY"]
  B --> B2["BOUNDED_LATENCY_THROUGHPUT"]
  B --> B3["PREFILL_THROUGHPUT"]

  C --> C1["REFERENCE_FP16_BF16"]
  C --> C2["FP8"]
  C --> C3["W8A16"]
  C --> C4["W4A16"]

  D --> D1["STRICT_SINGLE_GRID"]
  D --> D2["HYBRID_GRAPH"]
  D --> D3["CORRECTNESS_FIRST"]

  E --> E1["binding mode<br/>STATIC_SESSION or DYNAMIC_BINDINGS"]
  E --> E2["output mode<br/>RUN_INTO, BORROWED_OUTPUT, OWNED_OUTPUT"]
  E --> E3["reset mode<br/>in-entry, epoch-tagged, graph memset"]
```

Changing any axis creates a different acceptance class. Quantized throughput cannot satisfy a
reference-precision batch-one latency claim.

## 4. Core object relationships

```mermaid
classDiagram
  class CompileRequest {
    +model_or_exported_program
    +example_inputs_and_state
    +service_constraints
  }

  class WorkloadContract {
    +performance_objective
    +numerical_policy
    +execution_policy
    +invocation_contract
    +budgets
  }

  class WorkloadBucketKey {
    +graph_family
    +decode_or_prefill
    +active_token_bucket
    +sequence_bucket
    +hidden_head_projection_family
    +source_dtype_layout_family
    +semantic_flags
  }

  class TargetFingerprint {
    +TargetArchitectureKey
    +device_partition_identity
    +visible_resource_identity
    +driver_toolkit_ids
    +library_build_ids
  }

  class TargetArchitectureKey {
    +accelerator_vendor
    +backend_and_ISA_family
    +object_link_format
    +feature_set
  }

  class TargetBackend {
    +discover_caps
    +enumerate_legal_candidates
    +compile_inspect_profile
    +execution_artifact_recipes
  }

  class DeviceCaps {
    +resource_limits
    +visible_compute_memory_cache_resources
    +mma_tma_features
    +measured_shape_family_bandwidths
    +measured_orchestration_costs
  }

  class FactTables {
    +shape_layout_alias
    +constants_and_prepack
    +effects_and_lifetimes
    +precision_legality
  }

  class RegionGraph {
    +primitive_regions
    +composite_opportunities
    +traffic_estimates
    +value_transport_options
  }

  class CandidateSet {
    +variant_families
    +execution_capabilities
    +legality_contracts
  }

  class MeasuredBaselineSuite {
    +equivalence_class
    +trace_set
    +traffic_resources_latency
    +environment_fingerprint
  }

  class PlanTemplate {
    +segment_templates
    +precision_layout_alternatives
    +value_transport_alternatives
    +invocation_contracts
  }

  class FinalExecutionPlan {
    +selected_precision_layout
    +selected_transports
    +persistent_and_graph_segments
    +entry_launch_envelopes
    +measured_provenance
  }

  class ArtifactPack {
    +TargetArchitectureKey
    +target_compatibility_range
    +compatibility_guards
    +owned_code_objects
    +prepacked_weights
    +final_plan_and_recipes
    +fallback_chain
  }

  CompileRequest --> WorkloadContract
  CompileRequest --> WorkloadBucketKey
  CompileRequest --> FactTables
  TargetArchitectureKey --> TargetFingerprint
  TargetArchitectureKey --> TargetBackend
  TargetFingerprint --> DeviceCaps
  FactTables --> RegionGraph
  RegionGraph --> CandidateSet
  WorkloadContract --> MeasuredBaselineSuite
  WorkloadBucketKey --> MeasuredBaselineSuite
  TargetFingerprint --> MeasuredBaselineSuite
  CandidateSet --> PlanTemplate
  TargetBackend --> CandidateSet
  MeasuredBaselineSuite --> PlanTemplate
  DeviceCaps --> PlanTemplate
  PlanTemplate --> FinalExecutionPlan
  FinalExecutionPlan --> ArtifactPack
```

## 5. Execution-domain decision

```mermaid
flowchart TD
  A["Candidate implementation"] --> B{"Callable as device code in the<br/>same linked executable?"}
  B -->|yes| C["DEVICE_CALLABLE"]
  B -->|no| D{"Representable using public<br/>CUDA Graph/runtime operations?"}
  D -->|yes| E["GRAPH_NODE"]
  D -->|no| F["HOST_ONLY"]

  C --> G["PersistentSegment candidate"]
  E --> H["CudaGraphSegment candidate<br/>one or more independent grids"]
  F --> I["HostFallbackSegment"]

  G --> J{"Execution policy"}
  H --> J
  I --> J
  J -->|STRICT_SINGLE_GRID| K["Exactly one all-DEVICE_CALLABLE plan"]
  J -->|HYBRID_GRAPH| L["Measure legal persistent/graph mixtures"]
  J -->|CORRECTNESS_FIRST| M["Allow guarded host fallback"]
```

A traced private `CUfunction` remains `GRAPH_NODE`/research diagnostic. Its numeric handle never
changes capability.

This diagram is the CUDA backend realization. Another accelerator backend supplies its own
separate-launch and orchestration objects beneath the same `DEVICE_CALLABLE` versus backend-node
legality distinction; it need not emulate CUDA Graph APIs in the target-neutral IR.

## 6. Traffic-first plan decision

```mermaid
flowchart TB
  A["Region or composite candidate"] --> B["TrafficEstimate"]
  B --> C["Weight bytes"]
  B --> D["Activation/materialization bytes"]
  B --> E["KV bytes"]
  B --> F["Workspace bytes"]
  B --> G["FLOPs and parallelism"]

  C --> H["PrecisionLayoutPlan<br/>FP16/BF16, FP8, W8, W4"]
  D --> I["ValueTransportPlan"]
  E --> H

  I --> I1["MATERIALIZE_HBM"]
  I --> I2["REGISTER_FORWARD"]
  I --> I3["SMEM_FORWARD"]
  I --> I4["RECOMPUTE_PER_WORKER"]
  I --> I5["RECOMPUTE_PER_CLUSTER"]

  H --> J["Calibrated lower bound"]
  I --> J
  F --> J
  G --> J
  J --> K["Prune only clear losers"]
  K --> L["Replay decides close cases"]
```

## 7. Resource-class segmentation

```mermaid
flowchart TD
  A["Selected DEVICE_CALLABLE variants"] --> B["Compile each candidate"]
  B --> C["Actual ResourceEnvelope"]
  C --> D{"Compatible EntryLaunchEnvelope?<br/>threads/warps/registers/stack/SMEM"}
  D -->|yes| E["One PersistentSegment"]
  D -->|no| F["Split by resource class"]

  F --> F1["LINEAR_HEAVY"]
  F --> F2["SKINNY_STREAMING"]
  F --> F3["ATTENTION_HEAVY"]
  F --> F4["COMPACT"]

  E --> G["Measure single-grid result"]
  F1 --> H["Top-level CUDA Graph"]
  F2 --> H
  F3 --> H
  F4 --> H
  H --> I["Measure segmented result"]
  G --> J["Choose under execution policy"]
  I --> J
```

Splitting adds grid transitions but can recover occupancy, reduce spills, and remove maximum-SMEM
reservation from compact phases. The result is measured, not assumed.

For `STRICT_SINGLE_GRID`, incompatibility rejects that plan or forces generation of a different
common envelope; it is not hidden behind a hybrid graph. A decode artifact must compile away
unreachable prefill/heavy paths before this decision.

## 8. Fixed-bucket persistent flow

```mermaid
flowchart TB
  A["PersistentSegment"] --> B["EntryLaunchEnvelope<br/>resident workers, threads, warps<br/>register/stack/SMEM ceilings"]
  A --> C["Preallocated scheduler state"]
  A --> D["PhaseProgram"]
  A --> E["Selected compiled variants"]

  C --> C1["parallel in-entry reset"]
  C --> C2["or epoch-tagged state"]
  C1 --> F["initial grid join"]
  C2 --> F

  D --> G["PhaseDesc"]
  G --> G1["active resident workers"]
  G --> G0["logical work count<br/>independent of worker count"]
  G --> G2["STATIC_RANGE"]
  G --> G3["ATOMIC_CURSOR when justified"]
  G --> G4["completion join"]

  E --> H["TMA/cp.async and warp roles<br/>inside variant"]
  E --> I["CompositeVariant"]
  I --> J["register/SMEM forward<br/>or per-worker recomputation"]

  F --> G
  G4 --> K["next composite phase"]
```

There is no per-task counter clone and no arbitrary dependency spin loop. Joins separate meaningful
composite phases rather than syntax-level FX operations. A worker may consume several logical
output or reduction tiles; visible SM count never defines the phase's semantic work count.

## 8.1 Decode-linear family decision

```mermaid
flowchart TD
  A["Exact M/N/K, layout, precision, epilogue<br/>target and MIG fingerprint"] --> B["Shape-family oracle measurements"]
  B --> C["cuBLASDx descriptors<br/>shared, register, pipeline"]
  B --> D["CUTLASS collectives"]
  B --> E["direct CuTe composition"]
  B --> F["MegaBake-native CUDA<br/>warp, CTA, split-K"]
  C --> G["Legality, correctness, compile-resource gate"]
  D --> G
  E --> G
  F --> G
  G --> H["Standalone body tournament"]
  H --> I["Embedded persistent-worker tournament"]
  I --> J["Separate composite and whole-entry beams"]
  J --> K["Select exact-fingerprint measured winner"]
```

`M<=4` is only a bucket fact. It does not select one algorithm. The measured SmolLM vocabulary head
uses a tensor-core family at `M=1`, while smaller projections use distinct GEMV mappings.

## 9. Transformer-composite flow

```mermaid
flowchart LR
  A["Residual/input"] --> B["RMSNorm"]
  B --> C["Q projection"]
  B --> D["K projection"]
  B --> E["V projection"]
  C --> F["QKVProjectionComposite"]
  D --> F
  E --> F

  B -."recompute normalized input<br/>inside each output CTA".-> F

  G["MLP input"] --> H["Gate projection"]
  G --> I["Up projection"]
  H --> J["SiLU"]
  J --> K["multiply"]
  I --> K
  K --> L["GatedMLPComposite<br/>stores only product"]
```

The compiler may concatenate/group QKV weights or compute gate/up dot products together. It accepts
the composite only when HBM traffic or joins fall and compiled resources remain legal.

## 10. Binding and output flow

```mermaid
flowchart TB
  A["InvocationContract"] --> B{"Stable session addresses?"}
  B -->|yes| C["STATIC_SESSION<br/>no pointer-table copy when possible"]
  B -->|no| D["Pinned host BindingBlock"]
  D -->|one async transfer| E["Stable device BindingTable"]

  C --> F["Persistent/graph execution"]
  E --> F
  E --> G["One-thread batched GraCE prelude"]
  G --> H["Confirmed existing vendor ABI slots"]
  H --> F

  F --> I{"Output contract"}
  I -->|RUN_INTO| J["caller-owned output"]
  I -->|BORROWED_OUTPUT| K["arena view with reuse lifetime"]
  I -->|OWNED_OUTPUT| L["independent pool/allocation or copy<br/>all work measured"]
```

The GraCE prelude edits launch values, not a vendor signature or body.

## 11. Serving-throughput flow

```mermaid
flowchart TB
  A["Incoming requests"] --> B["Admission scheduler"]
  B --> C["bounded wait/fairness policy"]
  C --> D["Group compatible layer/shape/precision work"]
  D --> E["Active-token batch"]
  E --> F["Load each weight tile once"]
  F --> G["Apply to multiple activation rows"]
  G --> H["Update paged KV and request state"]
  H --> I{"Requests complete?"}
  I -->|no| B
  I -->|yes| J["sampling/output"]
```

This scheduler exists because batching raises arithmetic intensity and amortizes weight bytes. It is
not part of the batch-one fixed-phase executor.

## 12. Runtime sequence

```mermaid
sequenceDiagram
  participant User as Caller
  participant Runtime as V2 Runtime
  participant Catalog as Artifact Catalog
  participant GPU as CUDA Device
  participant Graph as Cached Graph Exec

  User->>Runtime: run(inputs/state, WorkloadContract)
  Runtime->>Runtime: fingerprint target and validate input contract
  Runtime->>Catalog: lookup compatible ArtifactPack
  Catalog-->>Runtime: plan, code, prepacks, fallback

  alt first initialization
    Runtime->>GPU: allocate stable arena/KV/scheduler/binding state
    Runtime->>GPU: load prepacked weights and owned code
    Runtime->>Graph: reconstruct graph recipes
    Runtime->>Graph: instantiate and upload
  end

  alt dynamic bindings
    Runtime->>GPU: one compact BindingBlock update
  else static session
    Runtime->>Runtime: retain stable addresses
  end

  alt strict single-grid
    Runtime->>GPU: launch one persistent entry
  else hybrid graph
    Runtime->>Graph: one cudaGraphLaunch
    Graph->>GPU: persistent/vendor/copy nodes as recorded
  end

  GPU-->>Runtime: completion and optional telemetry
  Runtime-->>User: output under explicit lifetime contract
```

## 13. Baseline and acceptance flow

```mermaid
flowchart TB
  A["Same bucket + objective + numerical policy"] --> B["ordinary + strongest low-overhead<br/>torch.compile baselines"]
  A --> C["direct hot-region vendor baselines"]
  A --> D["static/indirect CUDA Graph baselines"]
  A --> E["equivalent serving/quantized baseline"]
  A --> F["MegaBake strict/hybrid candidates"]

  B --> G["MeasuredBaselineSuite"]
  C --> G
  D --> G
  E --> G
  G --> H["strongest equivalent legal baseline"]

  F --> I["correctness/quality"]
  F --> J["all-operation timeline"]
  F --> K["semantic + physical traffic<br/>entry resources/latency distribution"]
  I --> L{"All gates pass and<br/>candidate wins by margin?"}
  J --> L
  K --> L
  H --> L
  L -->|yes| M["freeze FinalExecutionPlan"]
  L -->|no, performance mode| N["refine or fallback"]
  L -->|no, explicit research mode| O["emit measured deficit label"]
  M --> P["SingleGridScorecard<br/>target support level + measurement origin<br/>strict legality/win/p50/p99"]
  N --> Q["hybrid fallback result<br/>not a strict win"]
```

The initial H200 evidence sets the bar: ordinary `torch.compile` measured 5.691 ms, but the legal
export + `reduce-overhead` mode measured 1.782 ms. Only the latter can gate a strict win for that
bucket on that target. It provides no performance status for A100 or another fingerprint.

## 13.1 Measurement-truth flow

```mermaid
flowchart LR
  A["Semantic plan bytes"] --> B["semantic BW, end-to-end"]
  A --> C["semantic BW, GPU body"]
  D["Unfiltered GPU timeline"] --> E["compute grids + copies + memsets"]
  F["Hardware counters"] --> G["physical DRAM/L2 bytes and throughput"]
  H["Device/partition resource fractions"] --> I["product-peak mathematical floor"]
  B --> J["CandidateMeasurement"]
  C --> J
  E --> J
  G --> J
  I --> J
```

These bandwidth quantities are not interchangeable. The H200 `3g.71gb` profile owns 4/8 memory
slices, making the previous hard-coded 500 GB/s estimate invalid.

## 14. Stage contracts

### Stage 0 — contract, bucket, and target

- Produces `WorkloadContract`, target-neutral `WorkloadBucketKey`, `TargetArchitectureKey`,
  `TargetFingerprint`, selected `TargetBackend`, and calibrated `DeviceCaps`.
- Does not choose regions or kernels.

### Stage 1 — export, canonicalize, facts

- Produces `FXGraph + FactTables`.
- Does not select execution plane, precision tactic, or schedule.

### Stage 2 — regions, traffic, transport

- Produces `RegionGraph`, `TrafficEstimate`s, and legal value-transport alternatives.
- Does not claim actual register/SMEM residency.

### Stage 3 — candidates and baseline suite

- Produces `CandidateSet` and `MeasuredBaselineSuite`.
- Enumerates cuBLASDx, CUTLASS collective, direct CuTe, and native CUDA families whenever legal.
- Labels every implementation capability, required target feature, and equivalence class.
- Never promotes a private handle or parameter pack into a portable candidate.

### Stage 4 — bounded plan enumeration

- Produces a small set of legal `PlanTemplate`s.
- Uses traffic and resources to reject clear losers, not to pronounce an analytical winner.

### Stage 5 — generate, compile, measure, refine

- Produces `FinalExecutionPlan` and `AutotuneDB` entries.
- Uses staged successive halving from standalone body to persistent worker, composite, and
  separately compiled whole-entry beams.
- Accepts only exact-fingerprint measurements as winners.
- May revise execution configuration, transports, and resource segmentation.
- Measures all work, including resets, copies, and binding.

### Stage 6 — artifact emission

- Produces `ArtifactPack` with target key/range, owned code, prepacks, recipes, contracts, support
  level, measurement provenance, and fallbacks.
- Does not serialize live CUDA handles or defer compilation decisions to runtime.

## 15. Strict invariants

1. Stage and object names match all V2.3 documents.
2. Workload objective, numerical policy, execution policy, and the compound invocation contract are
   orthogonal.
3. `RegionGraph` describes semantics and traffic opportunities, not launch count.
4. Only `DEVICE_CALLABLE` variants enter persistent segments.
5. Vendor work remains separate GPU grids even inside one graph submission.
6. Worker IDs identify CTA slots, not physical SMs.
7. Machine pipelines live in compiled variants.
8. Register/SMEM forwarding and recomputation require compiled contracts.
9. Resource-incompatible variants are split unless a measured one-grid result wins.
10. Fixed buckets use `PhaseProgram`; dynamic admission is a separate serving scheduler.
11. Every reset, copy, allocation, and output-lifetime cost is visible.
12. The strongest equivalent measured baseline gates acceptance.
13. Runtime binds and launches; it does not reconstruct compiler decisions.
14. Logical work count is independent of resident worker and visible SM counts.
15. A strict artifact has one compatible measured `EntryLaunchEnvelope`.
16. Strict-one-grid wins and hybrid fallback wins are separate scorecard fields.
17. `FXGraph + FactTables` and `RegionGraph` contain no CUDA product-name or instruction-family
    policy.
18. Target features, not GPU name tests, determine candidate legality.
19. Cross-target measurements may order a search but never select or validate another target's
    winner.
