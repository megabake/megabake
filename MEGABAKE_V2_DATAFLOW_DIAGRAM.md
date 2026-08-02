# Megabake V2 Detailed Dataflow Diagram

This file is the diagram-first companion to `MEGABAKE_V2_ARCHITECTURE.md`.

Its purpose is to answer, in strict dataflow terms:

- what goes into each stage
- what comes out of each stage
- how each stage transforms its inputs
- which objects are persistent artifacts versus transient compiler state
- where target modeling, baseline comparison, and tuning enter the pipeline

The diagrams below are intentionally redundant with the architecture doc. The
goal here is not brevity. The goal is to make the system traceable end to end.


## 1. End-to-End Compile-Time Dataflow

```mermaid
flowchart TB
  subgraph Inputs["Inputs and Context"]
    A["CompileRequest<br/>- nn.Module or FX/Export graph<br/>- example inputs<br/>- compile policy<br/>- target mode"]
    B["HF Front-End Priors (optional)<br/>- model family<br/>- hidden size / heads<br/>- rope type<br/>- activation / MLP style"]
    C["Target Discovery<br/>- device props<br/>- arch family<br/>- memory / SMEM / regs"]
    D["Region Census Corpus (offline optional)<br/>- exported HF models<br/>- region frequencies<br/>- common bucket shapes"]
  end

  subgraph Stage0["Stage 0: Target + Bucket Binding"]
    E["TargetFingerprint<br/>- hardware identity<br/>- capability hash"]
    F["TargetModel<br/>- tensor core modes<br/>- async copy / TMA support<br/>- occupancy model<br/>- launch overhead priors"]
    G["BucketKey<br/>- model family<br/>- mode<br/>- batch bucket<br/>- seq bucket<br/>- hidden/head family<br/>- dtype family<br/>- target fingerprint"]
  end

  subgraph Stage1["Stage 1: Export and Canonicalization"]
    H["ExportedProgram / FX Graph<br/>- raw exported graph<br/>- graph signature<br/>- state refs"]
    I["Canonical FX Graph<br/>- decomposed<br/>- normalized<br/>- DCE/CSE applied<br/>- pattern-friendly op forms"]
    J["FactTables<br/>- shape facts<br/>- dtype facts<br/>- layout/stride facts<br/>- alias facts<br/>- constant facts<br/>- materialization boundaries"]
  end

  subgraph Stage2["Stage 2: Region Formation"]
    K["Motif Recovery<br/>- RMSNorm / LayerNorm<br/>- RoPE<br/>- QKV bundles<br/>- attention blocks<br/>- MLP patterns<br/>- KV-cache updates"]
    L["RegionGraph<br/>- semantic regions<br/>- region edges<br/>- residency opportunities<br/>- fusion boundaries"]
  end

  subgraph Stage3["Stage 3: Candidate Enumeration + Baseline"]
    M["CandidateSet per Region<br/>- persistent matvec variants<br/>- attention variants<br/>- norm/pointwise variants<br/>- strict-fusion candidates<br/>- segmented candidates"]
    N["ReferencePlan<br/>- TorchCompile-style launched plan<br/>- vendor GEMM/attention assumptions<br/>- launch-gap estimate"]
  end

  subgraph Stage4["Stage 4: Cost Model + Plan Selection"]
    O["Plan Scoring<br/>- compute cost<br/>- launch/orchestration cost<br/>- occupancy / SMEM / reg penalties<br/>- handoff/prefetch benefits<br/>- policy constraints"]
    P["SelectedRegionPlan<br/>- chosen region family per region<br/>- chosen segmentation count<br/>- accepted fusion edges<br/>- chosen delegation decisions"]
  end

  subgraph Stage5["Stage 5: Schedule Synthesis"]
    Q["ScheduleProgram<br/>- per-SM programs<br/>- tile descriptors<br/>- dependency tokens<br/>- prefetch actions<br/>- handoff actions<br/>- release actions"]
    R["KernelBundle<br/>- target-specific code templates<br/>- generated kernel entrypoints<br/>- schedule interpreter / driver kernel"]
  end

  subgraph Stage6["Stage 6: Replay Extraction + Tuning"]
    S["RegionReplaySet<br/>- representative hot-region replays<br/>- optional short forward snippets"]
    T["Autotuner<br/>- region replay search<br/>- baseline comparison<br/>- keep only winning plans"]
    U["AutotuneDB<br/>- target fingerprint key<br/>- bucket key<br/>- region/schedule winners<br/>- win margins vs reference"]
  end

  subgraph Stage7["Stage 7: Artifact Emission"]
    V["ArtifactPack<br/>- BucketDescriptor<br/>- TargetFingerprint<br/>- ScheduleProgram blob<br/>- KernelBundle refs<br/>- weight layout metadata<br/>- debug map"]
  end

  A --> E
  A --> G
  B -. informs .-> K
  B -. informs .-> G
  C --> E
  C --> F
  D -. informs priors .-> K
  E --> G
  A --> H
  H --> I
  H --> J
  F --> I
  I --> K
  J --> K
  K --> L
  L --> M
  L --> N
  F --> M
  F --> N
  G --> M
  M --> O
  N --> O
  F --> O
  O --> P
  P --> Q
  F --> Q
  Q --> R
  P --> S
  R --> S
  S --> T
  N --> T
  T --> U
  U --> V
  Q --> V
  R --> V
```


## 2. Core Data Objects

```mermaid
classDiagram
  class CompileRequest {
    +model_or_graph
    +example_inputs
    +compile_policy
    +target_mode
  }

  class TargetFingerprint {
    +arch_family
    +sm_count
    +smem_budget
    +driver_runtime_id
  }

  class TargetModel {
    +tensor_core_modes
    +async_copy_modes
    +occupancy_model
    +launch_overhead_priors
    +memory_bw_priors
  }

  class BucketKey {
    +model_family
    +mode
    +batch_bucket
    +seq_bucket
    +hidden_family
    +dtype_family
    +target_fingerprint
  }

  class FactTables {
    +shape_facts
    +dtype_facts
    +layout_facts
    +alias_facts
    +constant_facts
    +materialization_boundaries
  }

  class RegionGraph {
    +regions
    +edges
    +residency_opportunities
    +fusion_boundaries
  }

  class CandidateSet {
    +per_region_candidates
    +strict_candidates
    +segmented_candidates
    +delegated_candidates
  }

  class ReferencePlan {
    +launched_regions
    +vendor_kernel_assumptions
    +launch_gap_estimate
    +baseline_latency_estimate
  }

  class ScheduleProgram {
    +region_count
    +sm_programs
    +tile_descriptors
    +dependency_tokens
    +prefetch_actions
    +handoff_actions
  }

  class KernelBundle {
    +kernel_entrypoints
    +codegen_family
    +target_specific_objects
  }

  class RegionReplaySet {
    +hot_region_replays
    +short_forward_snippets
  }

  class AutotuneDB {
    +entries
    +measured_winners
    +reference_margins
  }

  class ArtifactPack {
    +bucket_descriptor
    +target_fingerprint
    +schedule_blob
    +kernel_refs
    +weight_layout_metadata
    +debug_map
  }

  CompileRequest --> TargetFingerprint
  TargetFingerprint --> TargetModel
  CompileRequest --> BucketKey
  BucketKey --> FactTables
  FactTables --> RegionGraph
  RegionGraph --> CandidateSet
  CandidateSet --> ScheduleProgram
  CandidateSet --> ReferencePlan
  ScheduleProgram --> KernelBundle
  ScheduleProgram --> RegionReplaySet
  ReferencePlan --> RegionReplaySet
  RegionReplaySet --> AutotuneDB
  ScheduleProgram --> ArtifactPack
  KernelBundle --> ArtifactPack
  AutotuneDB --> ArtifactPack
```


## 3. Stage Contracts

Each stage below is defined by:

- **goes in**
- **does**
- **comes out**
- **must not do**


### Stage 0: Target + Bucket Binding

**Goes in**

- `CompileRequest`
- runtime device properties
- optional HF model metadata
- optional corpus priors from prior region-census runs

**Does**

- fingerprints the target device
- builds `TargetModel`
- constructs the `BucketKey`
- decides which workload regime this compile belongs to

**Comes out**

- `TargetFingerprint`
- `TargetModel`
- `BucketKey`

**Must not do**

- mutate graph semantics
- choose region boundaries
- choose final kernel family


### Stage 1: Export + Canonical FX + FactTables

**Goes in**

- raw PyTorch model or FX graph
- example inputs
- `TargetModel` only as a light advisory input

**Does**

- `torch.export`
- decomposition
- canonicalization
- dead-code elimination
- constant folding
- shape / dtype / layout / alias propagation
- virtual view tracking
- explicit copy insertion only where needed

**Comes out**

- canonical FX graph
- `FactTables`

**Must not do**

- choose semantic regions
- choose strict-vs-segmented plan
- emit task/schedule records


### Stage 2: Motif Recovery + RegionGraph

**Goes in**

- canonical FX graph
- `FactTables`
- optional HF priors
- optional offline region-census priors

**Does**

- recognizes transformer motifs and generic tensor motifs
- groups nodes into semantic regions
- records region-to-region edges
- records residency and fusion opportunities

**Comes out**

- `RegionGraph`

**Must not do**

- fix per-SM schedule
- choose final tile descriptors
- encode low-level runtime actions


### Stage 3: Candidate Enumeration + ReferencePlan

**Goes in**

- `RegionGraph`
- `TargetModel`
- compiler policy (`strict`, `budgeted`, `auto`)
- delegation policy

**Does**

- enumerates valid Megabake candidate families per region
- enumerates valid segmentation candidates
- builds a TorchCompile-style `ReferencePlan`

**Comes out**

- `CandidateSet`
- `ReferencePlan`

**Must not do**

- assume Megabake wins by default
- assume strict fusion is always best


### Stage 4: Cost Model + Plan Selection

**Goes in**

- `CandidateSet`
- `ReferencePlan`
- `TargetModel`
- policy constraints

**Does**

- scores candidate compute cost
- scores orchestration / launch cost
- estimates occupancy / SMEM / register penalties
- estimates handoff and prefetch benefits
- compares candidate plans against the reference launched plan

**Comes out**

- `SelectedRegionPlan`

This plan contains:

- chosen region family per region
- chosen segmentation plan
- chosen delegation decisions
- chosen residency strategy
- expected win / loss margin versus the baseline

**Must not do**

- emit final code objects
- tune by itself without replay validation


### Stage 5: ScheduleProgram Synthesis

**Goes in**

- `SelectedRegionPlan`
- `TargetModel`

**Does**

- assigns work to SMs
- builds tile descriptors
- builds dependency tokens
- inserts prefetch / handoff / release actions
- chooses persistent program structure

**Comes out**

- `ScheduleProgram`

**Must not do**

- rediscover graph semantics
- re-open region boundaries without explicit policy reason


### Stage 6: Target-Specific Code Generation

**Goes in**

- `ScheduleProgram`
- chosen region-family implementations
- `TargetModel`

**Does**

- lowers selected regions into target-specific code objects
- emits persistent schedule driver kernel logic
- binds schedule layout to codegen family

**Comes out**

- `KernelBundle`

**Must not do**

- choose bucket families
- choose launch count
- re-run semantic fusion logic


### Stage 7: Replay Extraction + Baseline-Gated Tuning

**Goes in**

- `ScheduleProgram`
- `KernelBundle`
- `ReferencePlan`

**Does**

- extracts hot region replays
- optionally extracts short forward snippets
- tunes tile choices / micro-variants
- measures against the reference launched baseline
- rejects candidates that lose in default performance mode

**Comes out**

- tuned entries in `AutotuneDB`
- measured win margins

**Must not do**

- pretend analytical scores are enough forever


### Stage 8: ArtifactPack Emission

**Goes in**

- tuned `ScheduleProgram`
- tuned `KernelBundle`
- `BucketKey`
- `TargetFingerprint`
- weight layout metadata
- debug metadata

**Does**

- serializes everything needed for runtime execution
- stores enough metadata to replay / inspect / debug the plan later

**Comes out**

- `ArtifactPack`

**Must not do**

- leave any runtime-critical scheduling decisions implicit


## 4. Runtime Sequence

```mermaid
sequenceDiagram
  participant User as Caller
  participant Loader as Runtime Loader
  participant Catalog as Artifact Catalog
  participant Target as Target Fingerprinter
  participant Weights as Weight Prepacker / Cache
  participant DB as AutotuneDB
  participant GPU as Persistent Program Launch

  User->>Loader: run(model_inputs, mode, policy)
  Loader->>Target: fingerprint current device
  Target-->>Loader: TargetFingerprint
  Loader->>Catalog: lookup ArtifactPack by BucketKey + TargetFingerprint
  Catalog-->>Loader: ArtifactPack
  Loader->>Weights: get prepacked weights / layouts
  Weights-->>Loader: bound weight buffers
  Loader->>DB: check tuned entries for this bucket
  DB-->>Loader: tuned winner or miss
  alt no tuned entry and tuning allowed
    Loader->>GPU: run replay tuning jobs
    GPU-->>Loader: measured winners
    Loader->>DB: persist winners
  end
  Loader->>GPU: bind buffers + launch ScheduleProgram
  GPU-->>Loader: outputs + optional telemetry
  Loader-->>User: result tensors
```


## 5. HF-Aware Front-End Without Losing Generality

The architecture should deliberately account for the fact that a large fraction
of real workloads will come from HF transformer models, without turning the core
IR into a transformer-only IR.

### HF should help with:

- workload priors
- bucket formation priors
- motif priors
- tuning corpus selection
- region-census tooling

### HF should not be required for:

- graph correctness
- semantic truth
- execution legality
- non-transformer support

The exported FX graph remains the source of truth. HF metadata is advisory.


## 6. Region Census Tooling

The architecture should include an offline tool that exports a corpus of HF
models and reports:

- recovered region kinds
- most common region transitions
- most common bucket shapes
- most common fusion boundary types
- rare / unsupported structures

This tooling is not part of the runtime path. It is part of architecture and
compiler design hygiene.


## 7. Strict Dataflow Invariants

These invariants are the easiest way to check that v2 is not drifting back
toward v1 mistakes.

### Invariant A
Layer 1 must not decide GPU scheduling.

### Invariant B
Layer 2 must not collapse into packed low-level task records.

### Invariant C
Layer 3 must not need to rediscover semantic regions.

### Invariant D
Runtime must not rebuild the schedule graph from scratch.

### Invariant E
Default performance mode must compare against a launched reference baseline.

If any of these fail, the architecture is drifting.
