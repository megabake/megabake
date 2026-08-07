# Megabake V2 IR and Reuse Plan

This document exists to answer two questions explicitly:

1. **What are all the IR levels in v2, what do they contain, and what are they
   allowed to decide?**
2. **Which parts of other projects do we intend to reuse directly, which parts
   only inspire the design, and where do those pieces integrate into Megabake?**

This is the “IR contract” companion to:

- `MEGABAKE_V2_ARCHITECTURE.md`
- `MEGABAKE_V2_DATAFLOW_DIAGRAM.md`


## 1. Executive Summary

Megabake v2 should have:

- **3 real compiler layers**
  - `FXGraph + FactTables`
  - `RegionGraph`
  - `ScheduleProgram`

- **4 auxiliary compiler objects**
  - `TargetModel`
  - `ReferencePlan`
  - `AutotuneDB`
  - `ArtifactPack`

The intended split is:

- **Inductor / PyTorch-native machinery** for early graph cleanup and fact
  propagation
- **Megabake custom logic** for region formation, persistent scheduling, and
  artifact emission
- **Luminal / Mirage / Hazy / MPK / TileIR ideas** influencing design and search
  strategy, but generally **not** imported as hard runtime dependencies


## 2. The IR Stack at a Glance

### Real compiler layers

1. `FXGraph + FactTables`
2. `RegionGraph`
3. `ScheduleProgram`

### Auxiliary objects

4. `TargetModel`
5. `ReferencePlan`
6. `AutotuneDB`
7. `ArtifactPack`

The philosophy is:

- keep **semantic IR count small**
- keep **analysis richness high**
- keep **persistent execution logic explicit**

## 2.1 Decision Principles Behind the IR Split

These principles explain why the IR stack looks the way it does.

### Principle A: New IR only when a new class of decisions becomes legal

We do **not** create a new IR just because a stage has a new pass.

We create a new IR only when the compiler has crossed a boundary where it is now
allowed to decide something qualitatively different.

That is why:

- `FXGraph + FactTables` can decide semantic facts
- `RegionGraph` can decide semantic execution regions
- `ScheduleProgram` can decide concrete hardware execution

### Principle B: Facts should live in side tables before they become execution structure

Many things are important but do not justify their own first-class IR:

- shapes
- strides
- alias groups
- materialization boundaries
- constants

These are best represented first as **facts**, not as separate graph layers.

This is one of the core lessons from trying to avoid v1 complexity.

### Principle C: Hardware knowledge should enter late and explicitly

We do not want hardware assumptions contaminating the semantic graph too early.

That is why:

- `TargetModel` is separate from `FXGraph + FactTables`
- `RegionGraph` can remain semantically meaningful across targets
- `ScheduleProgram` is the first place where hard hardware commitment is allowed

This is also why `TargetModel` is an auxiliary object, not the first IR layer.

### Principle D: Runtime simplicity forces schedule explicitness

V1 reconstructs too much late in the runtime.

To avoid that mistake:

- `ScheduleProgram` must already contain the real execution structure
- `ArtifactPack` must carry runtime-ready data
- the loader should bind and launch, not invent the schedule

This principle is one of the biggest reasons `ScheduleProgram` exists as its own
layer.

### Principle E: The project goal forces baseline-aware objects

Because the goal is not only “emit a megakernel,” but “beat `torch.compile`,”
the compiler needs objects that standard compilers often do not surface so
explicitly:

- `ReferencePlan`
- `AutotuneDB`

Those objects exist because the performance objective is first-class, not as an
afterthought.


## 3. Real Compiler Layers

## 3.1 `FXGraph + FactTables`

### Purpose

This is the first true working layer of the compiler. It keeps the graph close
to PyTorch semantics while attaching the information needed for region formation
and later hardware-aware reasoning.

### Comes from

- `torch.export`
- decomposition and normalization passes
- fake tensor and symbolic shape analysis
- alias / layout / constant analysis

### Contains

#### Graph payload

- canonicalized FX / Export graph
- ATen-ish op set after decomposition
- placeholders / parameters / constants
- use-def edges

#### Side tables

- `ShapeFact`
- `DTypeFact`
- `LayoutFact`
- `StrideFact`
- `AliasGroup`
- `ConstantFact`
- `MaterializationBoundary`
- `BucketKey`
- `MemoryEffect`

### Typical field-level content

#### `ShapeFact`

- `value_id`
- `rank`
- `static_dims`
- `symbolic_dims`
- `bucketed_dims`

#### `LayoutFact`

- `value_id`
- `layout_kind`
- `contiguous`
- `strides`
- `offset`

#### `AliasGroup`

- `group_id`
- `members`
- `view_of`
- `materialization_required`

#### `MaterializationBoundary`

- `src_value`
- `dst_value`
- `reason`
  - non-contiguous consumer requirement
  - unsupported view semantics
  - copy-for-correctness

### Allowed decisions

- canonical op rewriting
- shape / dtype / layout propagation
- constant folding
- explicit materialization boundaries
- metadata-only view preservation

### Forbidden decisions

- final fusion boundaries
- region family selection
- warp roles
- per-SM scheduling
- final persistent program layout

### Core design rule

This layer is **semantic and factual**, not hardware-specific.

### Why this layer looks like this

This layer exists because v1 lowered too early.

If we go directly from exported FX nodes to something task-like, we lose the
ability to:

- reason cleanly about views
- reason cleanly about aliasing
- keep pattern recovery robust across syntactic graph variations
- keep hardware-specific logic out of semantic cleanup

So this layer is deliberately boring and rich in facts.

The design choice to use **FactTables** instead of another graph IR is
intentional:

- it keeps us close to PyTorch-native semantics
- it avoids inventing a second graph just to hold facts
- it makes it easier to reuse export / Inductor infrastructure

### Why each key item exists

- `ShapeFact` exists because region formation, bucketization, and codegen all
  need shape truth, but shape should not be rediscovered by later layers.
- `LayoutFact` and `StrideFact` exist because view-vs-copy decisions are central
  to correctness and fusion, especially for export graphs.
- `AliasGroup` exists because a lot of tensor programs are really view graphs in
  disguise; without explicit aliasing, later residency planning becomes
  incorrect or overly conservative.
- `ConstantFact` exists because constants should be handled semantically early,
  not as surprise buffers late.
- `MaterializationBoundary` exists because "when does a view become real?" is a
  semantic question first and an execution question second.
- `BucketKey` is attached here because specialization begins as soon as we know
  what family of shapes we are compiling for.
- `MemoryEffect` exists because some values are pure, some are just metadata,
  and some actually mutate state such as KV-cache updates.

### Inspirations

- **Direct**: `torch.export`, fake tensor / symbolic shape reasoning, and
  Inductor-style graph cleanup
- **Conceptual**: MLIR / bufferization-style distinction between metadata views
  and real copies
- **Anti-inspiration**: v1’s early lowering into low-level task records


## 3.2 `RegionGraph`

### Purpose

This is the first Megabake-native IR. It groups cleaned graph structure into
semantic units that are meaningful for persistent execution and cost modeling.

### Comes from

- `FXGraph + FactTables`
- pattern recovery
- region clustering
- optional HF transformer priors
- optional region-census priors

### Contains

#### Region nodes

- `MatvecRegion`
- `MatmulRegion`
- `AttentionRegion`
- `NormPointwiseRegion`
- `RopeRegion`
- `KVCacheRegion`
- `ExternRegion`

#### Region edges

- producer / consumer tensor flow
- layout compatibility
- residency opportunity
- fusion constraints

#### Region metadata

- region kind
- inputs / outputs
- shape key
- layout key
- side-effect / statefulness marker
- candidate family list
- fusion constraints
- residency opportunities
- estimated bytes / flops

### Typical field-level content

#### `Region`

- `region_id`
- `region_kind`
- `input_values`
- `output_values`
- `internal_nodes`
- `shape_signature`
- `layout_signature`
- `stateful`
- `candidate_families`
- `residency_hints`

#### `RegionEdge`

- `src_region`
- `dst_region`
- `value_id`
- `layout_relation`
- `handoff_possible`
- `must_materialize`

#### `ResidencyHint`

- `value_id`
- `preferred_storage`
  - register
  - smem
  - hbm
- `reuse_distance`
- `producer_consumer_pair`

### Allowed decisions

- semantic region boundaries
- semantic fusion boundaries
- candidate family legality
- residency opportunity detection
- strict vs segmented plan search space definition

### Forbidden decisions

- exact tile descriptors
- exact prefetch instruction sequence
- exact per-SM action list
- final code object choice

### Core design rule

This layer is **semantic + execution-aware**, but not yet **schedule-specific**.

### Why this layer looks like this

This layer exists because once the graph is clean, the next hard question is no
longer "what is the shape?" but:

> what are the real semantic execution units?

That is what `RegionGraph` answers.

We do not want to jump directly from facts to a concrete schedule because that
would recreate v1 in a cleaner suit. We first need a layer where:

- transformer motifs are explicit
- fusion boundaries are explicit
- candidate kernel families are explicit
- residency opportunities are explicit

This layer is where the compiler stops thinking in ATen nodes and starts
thinking in execution regions.

### Why each key item exists

- `MatvecRegion` is distinct from `MatmulRegion` because the performance regime
  and kernel families are materially different. The traces strongly support this
  split.
- `AttentionRegion` exists because attention is structurally special even when
  it is not the first bottleneck.
- `NormPointwiseRegion` exists because norms and pointwise chains are typically
  memory-bound, fusion-friendly, and closely tied by residency opportunities.
- `KVCacheRegion` exists because cache updates are stateful and should not be
  hidden inside generic elementwise logic.
- `ExternRegion` exists as a safety valve so generality does not poison the fast
  path.
- `RegionEdge` exists because tensor flow alone is not enough; the compiler must
  also know whether layout is compatible and whether handoff is plausible.
- `ResidencyHint` exists because "keep this on chip if possible" should be
  represented before final scheduling, but should still remain a hint rather
  than a hard action at this layer.
- `candidate_families` exists because region formation and kernel selection must
  be decoupled.

### Why there is no separate first-class `MemoryIR` here

This is an important non-decision.

We deliberately do **not** make memory planning a fourth heavy semantic layer at
this point, because:

- most of the crucial memory information is already represented as facts or
  region annotations
- turning it into a full IR too early would increase complexity sharply
- the final concrete memory behavior should still be decided jointly with the
  schedule

So memory planning begins here as annotations and becomes hard execution
structure only in `ScheduleProgram`.

### Inspirations

- **Direct**: none from external libraries; this is primarily Megabake custom
- **Conceptual**: super-op formation, transformer motif recovery, and compiler
  partitioning
- **Transformer-specific prior**: HF model family structure and offline region
  census
- **Persistent-kernel influence**: regions are shaped by what wants to live
  together on chip, not only by graph syntax


## 3.3 `ScheduleProgram`

### Purpose

This is the persistent-execution IR. It is the first layer that fully commits to
how the chosen plan runs on the chosen target.

### Comes from

- selected `RegionGraph` plan
- `TargetModel`
- cost model results
- policy constraints
- tuning priors

### Contains

#### Program-level decisions

- region count
- segmentation plan
- chosen backend family per region
- target-specific tile family

#### Execution descriptors

- `SMProgram`
- `TileDescriptor`
- `DependencyToken`
- `PrefetchAction`
- `HandoffAction`
- `ReleaseAction`
- `WarpRoleDescriptor`

### Typical field-level content

#### `ScheduleProgram`

- `program_id`
- `bucket_key`
- `target_fingerprint`
- `region_instances`
- `sm_programs`
- `tile_descriptors`
- `dependency_table`
- `policy_mode`

#### `SMProgram`

- `sm_id`
- ordered action list
- region instance references
- local staging descriptors

#### `Action`

- `kind`
  - wait_dep
  - prefetch
  - run_region
  - handoff
  - release
- `payload`

#### `TileDescriptor`

- `region_instance_id`
- `tile_shape`
- `warp_roles`
- `smem_budget`
- `register_budget`

### Allowed decisions

- final persistent schedule
- exact region count
- exact tile / warp structure
- exact prefetch / handoff sequencing
- final policy-compliant execution plan

### Forbidden decisions

- rediscovering graph semantics
- rediscovering region structure
- reopening canonicalization / decomposition issues

### Core design rule

This layer is **hardware-aware and execution-committed**.

### Why this layer looks like this

This layer exists because at some point the compiler must stop discussing
options and commit to an actual execution program.

That commitment includes:

- how many regions really run
- in what order they run
- which SMs own what work
- where dependencies are released
- how prefetch and handoff actually happen

V1 did too much of this late in the runtime. `ScheduleProgram` exists to prevent
that.

### Why each key item exists

- `region_count` exists because segmentation is a first-class performance
  variable, not a side effect.
- `SMProgram` exists because persistent execution is not just a list of kernels;
  it is a per-SM action program.
- `TileDescriptor` exists because tiles are the unit where target geometry,
  resource use, and work assignment meet.
- `DependencyToken` exists because persistent programs need explicit dependency
  management rather than implicit kernel-launch ordering.
- `PrefetchAction` exists because overlap is one of the only mathematically large
  gains available once launch gaps shrink.
- `HandoffAction` exists because on-chip reuse must become explicit to be
  reliable.
- `WarpRoleDescriptor` exists because once we commit to execution, different
  warps may have different responsibilities.

### Why this is not just a fancy `TaskDesc[]`

This is the most important rationale in the whole document.

`ScheduleProgram` is not a renamed `TaskDesc[]` because it contains:

- explicit persistent ordering
- explicit dependency actions
- explicit prefetch and handoff actions
- explicit segmentation choice
- explicit policy outcome

That is qualitatively different from a flat packed task stream.

### Inspirations

- **Conceptual**: persistent kernels, task-graph execution, per-SM programs
- **Strong influence**: Hazy / MPK style execution thinking
- **Direct code reuse**: likely very little; this is mostly Megabake-specific


## 4. Auxiliary Compiler Objects

These are not full semantic IR layers, but they are first-class objects that
drive selection, tuning, and runtime behavior.

## 4.1 `TargetModel`

### Purpose

Represents the hardware capabilities and cost-model priors needed for schedule
selection and codegen.

### Contains

- architecture family
- tensor-core / MMA capabilities
- async copy / TMA capabilities
- register budget
- SMEM budget
- occupancy model
- memory bandwidth / latency priors
- launch-overhead priors

### Written by

- target fingerprinting
- device query
- calibration passes

### Read by

- candidate enumeration
- cost model
- schedule builder
- codegen

### Why this object is separate

`TargetModel` is separate because hardware capability is not a semantic graph
property.

If we mix it into the semantic IR too early:

- the graph becomes target-contaminated
- reuse across targets gets harder
- passes begin to silently assume one GPU family

Keeping `TargetModel` separate enforces late target binding.

### Inspirations

- **Direct**: device property queries and calibration
- **Conceptual**: backend target models in compilers and autotuners


## 4.2 `ReferencePlan`

### Purpose

Represents a TorchCompile-style launched baseline used as a comparison target.

### Contains

- launched region assumptions
- vendor GEMM / attention baseline assumptions
- launch-gap model
- expected end-to-end baseline estimate

### Written by

- candidate enumeration / baseline builder

### Read by

- cost model
- replay tuner
- plan acceptance logic

### Why this object exists at all

Most compiler designs stop after "choose the best internal plan."

Megabake cannot stop there because the real project goal is external:

> beat `torch.compile`

That means we need an explicit baseline object, not just internal scores.

### Inspirations

- **Direct**: our own trace-based reasoning
- **Conceptual**: A/B baseline comparison and shadow-plan evaluation


## 4.3 `AutotuneDB`

### Purpose

Stores empirical winners for `(target fingerprint, bucket key, region kind,
policy mode)` combinations.

### Contains

- candidate configuration
- measured runtime
- baseline margin
- selected winner
- confidence / sample count

### Written by

- region replay tuner
- optional short forward-run tuner

### Read by

- candidate enumeration
- plan selection
- runtime first-use logic

### Why this object is not optional

The architecture is explicitly saying:

- analytical modeling is necessary
- but analytical modeling is not enough

`AutotuneDB` is the object that turns that sentence into engineering structure.

It also prevents the compiler from re-learning the same region winners on every
run.

### Inspirations

- **Direct**: autotuning systems and cached winner tables
- **Conceptual**: Inductor / Triton tuning culture, plus search-oriented systems


## 4.4 `ArtifactPack`

### Purpose

The emitted object that runtime consumes.

### Contains

- `BucketDescriptor`
- `TargetFingerprint`
- serialized `ScheduleProgram`
- code object references
- weight layout metadata
- debug map
- optional tuned margins / provenance

### Why this is separate from `ScheduleProgram`

`ScheduleProgram` is a compile-time execution object.

`ArtifactPack` is a deployment / runtime object.

That distinction matters because runtime needs:

- serialization
- versioning
- portability across processes
- cached tuned results
- debugging provenance

without reopening compiler logic.

### Inspirations

- **Direct**: deployment-oriented artifact packaging
- **Conceptual**: AOT runtime bundles and serialized schedule objects


## 5. Which Project Solves What?

This section is the most important practical part of the document.

For each external project or system, we should be clear whether we are:

- reusing it directly
- adapting a pattern or idea
- deliberately *not* using it


## 5.1 Torch Export / Inductor

### Reuse directly

These are the strongest direct integration points:

- `torch.export`
- decomposition tables
- graph canonicalization
- fake tensor / symbolic shape machinery
- graph cleanup passes
- pattern matcher infrastructure where stable enough

### Use conceptually

- autotuning mindset
- candidate selection philosophy
- layout / shape reasoning discipline

### Avoid reusing directly

Likely too tied to launched-kernel execution:

- Inductor scheduler for final execution
- launched-kernel codegen assumptions as the central Megabake runtime model
- any IR layer that assumes “fusion means one launched Triton/C++ kernel”

### Integrates at

- `capture/export.py`
- `passes/canonicalize.py`
- `analysis/shapes.py`
- `analysis/layouts.py`
- `analysis/constants.py`
- early parts of `passes/recover_motifs.py`


## 5.2 Luminal

### Reuse directly

Probably none as a hard dependency.

### Use conceptually

Luminal’s real value here is:

- keeping the IR stack small
- specializing aggressively
- avoiding unnecessary compiler bureaucracy
- doing compile-time simplification early

### Avoid reusing directly

- any attempt to mirror Luminal’s implementation architecture too literally if
  it conflicts with PyTorch-native export / FX integration

### Integrates at

- IR-count discipline
- field design discipline
- "new pass, yes; new IR, only if necessary" rule

### Detailed Luminal analysis from the local clone

The local clone in `agent_space/luminal` reinforces that Luminal is most useful
to Megabake as an **IR-discipline and late-search reference**, not as a direct
replacement for our stack.

#### What Luminal actually has

From the inspected repository:

- `Graph` owns the compiler state and search state
- `HLIRGraph` is the user/model-facing graph
- a saturated **e-graph** holds rewrite/search alternatives
- `LLIRGraph` is the extracted executable backend graph
- `ShapeTracker` and symbolic `Expression` machinery carry shape/stride facts
- `DimBucket`-style profiling buckets and runtime maps attach dynamic-shape
  reasoning to compilation and profiling

The most important practical observation is:

> Luminal really has a **semantic graph → searchable rewrite space → extracted
> low-level graph** flow.

That is much closer to our:

- `FXGraph + FactTables`
- `RegionGraph`
- `ScheduleProgram`

split than it first appears from the README.

#### What we should copy from Luminal very directly in spirit

- every node/value should be paired with rich shape/layout truth
- symbolic and bucketed dimension reasoning should be first-class
- late-stage profiling search should be bounded and data-driven
- legality/resource filters should explicitly reject bad candidates
- the semantic core should remain as small and stable as possible

These are excellent fits for Megabake.

#### What we should explicitly not copy

- Luminal’s 15-op semantic HLIR as our Layer 1
- egglog/e-graph as the primary Megabake IR
- whole-graph genetic search over semantic alternatives
- implicit discovery of persistent schedule structure from primitive graph search

These do not fit because Megabake is:

- FX/export-first
- transformer-aware
- persistent-program oriented
- trying to keep the search at the late backend/schedule neighborhood

#### Concrete mapping to Megabake

- Luminal `ShapeTracker` / symbolic expressions
  - inspires Megabake `FactTables`

- Luminal search-space saturation + extracted executable form
  - inspires Megabake’s clean separation between `RegionGraph` and
    `ScheduleProgram`

- Luminal bounded profiling search
  - inspires Megabake’s Stage 7 replay tuning

- Luminal resource / legality filtering
  - inspires Megabake’s candidate validation and baseline-gated acceptance

#### Why this matters for the IR design

The important lesson is not:

> "copy Luminal’s IR"

The important lesson is:

> "copy Luminal’s discipline about keeping semantics small, facts explicit, and
> search late."

That is the part that genuinely strengthens Megabake’s IR design.


## 5.3 Hazy / MPK / Persistent Megakernel Research

### Reuse directly

Probably none as a hard code dependency.

### Use conceptually

These systems primarily inform:

- persistent execution structure
- per-SM program thinking
- on-chip residency planning
- software pipelining and handoff
- weight-streaming mindset
- launch-gap elimination as a first-class optimization target

### Avoid reusing directly

- overfitting to one paper’s exact runtime assumptions if they do not match
  export-based graph lowering or the chosen workload regime

### Integrates at

- `passes/build_schedule.py`
- `cost_model/residency.py`
- `cost_model/occupancy.py`
- `runtime` and `cuda_v2/runtime_program.cuh`


## 5.4 Mirage

### Reuse directly

Probably none as a hard dependency unless a very specific subcomponent proves
surprisingly reusable.

### Use conceptually

Mirage is most useful for:

- candidate search
- partition-space thinking
- schedule-space exploration
- plan selection under constraints

### Avoid reusing directly

- adopting another full compiler worldview if it clashes with PyTorch export and
  Megabake’s persistent-program model

### Integrates at

- `passes/enumerate_candidates.py`
- `cost_model/roofline.py`
- `autotune/search.py`


## 5.5 HF Transformers

HF is not exactly a compiler project, but it is central to the workload shape.

### Reuse directly

- model configs and metadata
- family priors
- test corpus
- wrapper utilities for exported decode / prefill steps

### Use conceptually

- transformer motif taxonomy
- region census input corpus
- bucket-family design

### Avoid reusing directly

- using HF metadata as the semantic source of truth instead of the exported FX graph

### Integrates at

- `frontends/hf.py`
- `corpus/region_census.py`
- `passes/recover_motifs.py`
- tuning corpus generation


## 5.6 TileIR

### Reuse directly

Potentially, but only in a narrow place:

- as an **optional backend kernel IR** for selected tile-centric region
  families

### Use conceptually

- stable low-level tile target
- target-specific backend lowering boundary
- tile-centric codegen separation from semantic graph IRs

### Avoid reusing directly

- as a replacement for `FXGraph + FactTables`
- as a replacement for `RegionGraph`
- as a replacement for `ScheduleProgram`
- as the core persistent runtime orchestration language, at least initially

### Integrates at

- `backend/kernel_ir.py`
- `backend/tileir.py`
- Stage 6 target-specific code generation
- `KernelBundle` backend variants for selected compute-heavy regions

### Why this integration is intentionally narrow

TileIR is valuable precisely because it can enhance the **backend lowering
boundary** without forcing changes upward into the semantic IR stack.

That means we add it only where all of the following are true:

- the region is already semantically formed
- the schedule is already chosen
- the region is tile-centric and compute-heavy
- the backend choice can be evaluated by replay tuning

This avoids repeating a common architecture mistake:

- seeing a powerful low-level backend IR
- then letting it leak upward and reshape semantic compiler layers that it was
  never meant to own

### Admission criteria

TileIR should be admitted only when:

1. the region family is something like `MatvecRegion`, `MatmulRegion`, or
   `AttentionRegion`
2. the default CUDA/CuTe path is either hard to retarget or clearly leaving
   performance on the table
3. the integration stays below `ScheduleProgram`
4. replay tuning shows a real win or materially cleaner backend portability

If those conditions are not met, Megabake should stay on the default
CUDA/CuTe-style backend path.


## 6. Integration Map: Where Each Project Enters the Pipeline

### Stage 1: Canonical graph and facts

Use mainly:

- `torch.export`
- Inductor decomposition / canonicalization / shape machinery

Do **not** use:

- Hazy / MPK / Mirage abstractions here

### Stage 2: Region formation

Use:

- Inductor-style pattern infrastructure
- HF priors
- offline region-census results

Influence from:

- Luminal discipline (small IR)

### Stage 3: Candidate enumeration and baseline generation

Use:

- custom Megabake logic

Influence from:

- Mirage search-space thinking
- TorchCompile-style launched reference modeling

### Stage 4: Cost model and plan selection

Use:

- custom Megabake logic
- `TargetModel`
- `ReferencePlan`
- `AutotuneDB`

Influence from:

- Hazy / MPK style runtime assumptions
- Mirage search / plan comparison ideas

### Stage 5: ScheduleProgram synthesis

Use:

- custom Megabake logic

Influence from:

- persistent-kernel research
- Hazy / MPK ideas

### Stage 6+: Codegen, tuning, and artifact emission

Use:

- custom Megabake logic
- replay tuning
- target-specific codegen
- optional TileIR lowering for selected backend kernel families

Influence from:

- Inductor tuning philosophy
- Mirage-style search mindset
- Hazy / MPK runtime philosophy
- TileIR as a possible late backend target for tile-centric kernels


## 7. Adopt / Adapt / Avoid

This is the short operational version.

### Adopt directly

- export
- decomposition
- canonicalization
- fake tensor / symbolic shapes
- pattern matcher infrastructure where stable

### Adapt conceptually

- Luminal: small IR discipline
- Hazy / MPK: persistent schedule design
- Mirage: candidate search and partition-space reasoning
- TileIR: optional backend kernel IR for selected compute-heavy regions
- HF: workload priors and corpus generation

### Avoid

- building too many IR layers because a paper had many concepts
- reusing a launched-kernel scheduler as if it were a persistent-program
  scheduler
- letting HF metadata replace the exported graph as semantic truth
- importing multiple full compiler worldviews that compete with one another


## 8. Final Design Rule

The most important rule is:

> **Use Inductor/PyTorch to solve standard graph-cleanup and fact-propagation
> problems. Use Megabake custom logic only once the problem becomes semantic
> region formation and persistent-program synthesis.**

That gives the cleanest split:

- PyTorch / Inductor solves the early, standard compiler work
- Megabake solves the execution-model-specific work
- TileIR may optionally enhance the backend lowering layer without changing the
  main IR stack
- Luminal / Mirage / Hazy / MPK shape the design, but do not bloat the codebase

That is how v2 stays both ambitious and implementable.
