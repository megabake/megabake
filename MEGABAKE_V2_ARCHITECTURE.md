# Megabake V2 Architecture

This document is the repo-local snapshot of the v2 design work. It is meant to
be the durable companion to the interactive canvas at
`/home/devuser/.cursor/projects/home-devuser-megabake/canvases/megabake-first-principles-redesign.canvas.tsx`.

The goal of v2 is not "more fusion" in the abstract. The goal is:

> Given a `torch.export` / FX graph for a narrow target workload, produce a
> shape-bucket-specific, hardware-parameterized execution artifact that can be
> specialized by target modeling and empirical tuning into a persistent
> megakernel schedule competitive with vendor-grade kernels on the operations
> that actually dominate the chosen regime.

The main lesson from v1 and the current traces is that `megabake` lowers into
low-level task records too early. That makes the project good at *running a flat
task list* but weak at *reasoning about graph regions, memory residency, target
capabilities, and schedule alternatives*.

## North Star

The north-star goal for Megabake v2 is:

> any exportable torch model FX graph should lower into a Megabake execution
> plan whose default objective is to beat `torch.compile`

This is the long-term project goal. The architecture should be evaluated by
whether it makes that goal increasingly achievable without collapsing back into
v1-style early lowering and ad hoc task emission.

## Default Objective

The compiler’s default optimization objective is:

> beat `torch.compile` on end-to-end latency by combining near-baseline or
> slightly better hot-region kernels with dramatically lower launch and
> orchestration overhead

This is different from:

- maximizing fusion count
- maximizing single-kernel purity
- maximizing coverage at any performance cost

Those can all be important secondary goals, but the default objective should be
winning the end-to-end latency comparison.

## First Proven Regimes

The architecture should support the north-star goal broadly, but the first
regimes proven in practice can be narrower.

That means:

- the compiler remains FX / export native and model-generic
- the first successful workload buckets can still be transformer-heavy
- decode may be proven before prefill
- selected batch / sequence / dtype / target buckets may be proven before the
  full space

This is not a retreat from the north star. It is the proof strategy for getting
there.

## Strict Fusion Mode vs Performance Mode

Megabake should explicitly separate:

1. **strict purity**
   - "can I turn this into one megakernel?"
2. **default performance**
   - "what plan actually beats `torch.compile`?"

That implies two legitimate operating modes:

- **Strict fusion mode**
  - best effort single-megakernel lowering
  - useful for research, purity, and architectural validation

- **Performance mode**
  - allow segmentation and, if policy allows, delegation
  - optimize for beating `torch.compile`, not for minimum region count

This split is necessary so the project can preserve the single-megakernel
ambition without forcing every workload into a shape that loses on actual
latency.


## 1. V2 Thesis

Megabake v2 should be an ahead-of-time compiler for a narrow inference regime,
but it should not be architected as a hardcoded "SM90 compiler." It should be a
hardware-parameterized compiler with late target binding.

The first validated workload can still be narrow, but the architecture should
look like this:

- early semantic IRs are hardware-neutral
- a formal `TargetModel` injects hardware knowledge later
- a formal `AutotuneDB` stores empirical schedule results per target and bucket
- codegen and schedule emission are target-specific only at the end
- the compiler can validate one backend first without baking that backend into
  every design decision

The winning design is:

1. Normalize the FX graph into a small canonical tensor IR.
2. Recover large semantic motifs such as matvec, attention, norm, rope, and
   epilogues.
3. Build explicit optimization regions and a real memory/dataflow plan.
4. Use a `TargetModel` and empirical tuning to choose profitable schedules for a
   bucketed workload regime.
5. Generate a per-bucket persistent schedule, not just a list of generic tasks.
6. Emit target-specific code late, after schedule and candidate selection have
   converged.


## 2. What V1 Gets Wrong Structurally

V1 has important pieces, but they are arranged in the wrong order:

- `graph_walker.py` mixes canonicalization, motif recovery, layout handling,
  lowering, fusion, buffer planning, and schedule emission in one pass.
- `TaskDesc` becomes the first serious optimization IR, but it is already too
  low-level to express region formation or schedule alternatives.
- scheduling is effectively rebuilt in `runtime/loader.py`, so the compiler does
  not really emit the final schedule artifact
- `dyn_dims` / `batch_range` / `seq_range` exist mostly as metadata, not as a
  real dynamic bucket or symbolic-shape mechanism
- decode-specialized paths exist only partially
- prefetch / SMEM handoff / page planning are mostly scaffolding

The tra

- launch count is not the main unsolved problem anymore
- raw kernel quality for decode ces reinforce this:matvec / GEMV / GEMM is the primary gap
- attention is not the first bottleneck on the captured decode runs

Therefore v2 should be built around *compiler structure* and *kernel quality*,
not around adding more peephole fusion to the current one-pass lowering path.


## 3. Design Goals

### Primary goals

1. **Emit a real compile-time schedule artifact**
   - no late reconstruction of DAGs and SM queues in the runtime

2. **Be shape-bucket-specific**
   - a schedule is specialized to concrete shape families, not one fully generic
     runtime interpreter

3. **Be hardware-parameterized, not hardware-hardcoded**
   - early IRs should stay architecture-neutral
   - target capabilities should enter through a formal `TargetModel`
   - only late schedule/codegen stages should become backend-specific

4. **Be region-first, not node-first**
   - optimize semantic regions such as matvec + epilogue, not a flat sequence of
     ATen nodes

5. **Model memory residency explicitly**
   - registers, SMEM, and HBM are all part of the compile-time plan

6. **Treat autotuning and profile-guided tuning as part of the compiler**
   - schedule quality comes from search and measurement, not only handwritten
     heuristics

### Secondary goals

- retain a clear correctness fallback path
- keep artifacts introspectable and debuggable
- preserve enough modularity to support a future prefill path without rewriting
  the whole compiler again


## 4. Non-goals for V2.0

These are intentionally *not* the first target:

- broad model coverage across arbitrary exported graphs
- training
- full dynamic shape support across wildly varying sequence lengths
- perfect prefill performance from day one, even though prefill must be
  represented architecturally from the start
- a large number of validated backends on day one
- broad BF16 / FP32 / mixed-precision surface before decode FP16 is solid

V2.0 should win one narrow thing first.


## 5. Target Regimes and Bucket Families

The compiler should be designed around explicit workload regimes and bucket
families, not a single universal "golden path."

A bucket key should be something like:

- model family
- mode (`decode`, later `prefill`)
- batch bucket
- sequence bucket
- hidden-size / head-dim family
- dtype family
- target fingerprint

The first validated bucket can still be narrow, for example:

- decoder-only transformer
- `seq_q = 1`
- KV-cache already present
- hidden sizes / head dims drawn from a small known family
- weight layout allowed to change at compile/prepack time

But two important caveats should be explicit:

1. the architecture should not assume low batch is the only profitable regime
2. persistent megakernel profitability is empirical, not monotonic in batch size

For some workloads, small decode batches will benefit most from persistent
scheduling and on-chip handoff. For others, medium-batch or more irregular
serving regimes may be better. The compiler should represent that as a tuning
and schedule-selection question, not as a thesis baked into the IR.

Prefill should therefore be treated as:

- an explicit bucket family from day one
- a first-class regime in the architecture
- not the first required performance win for v2.0

### 5.1 Optimization Objective and Compiler Policies

The compiler should be framed as solving a constrained optimization problem:

> minimize estimated latency  
> subject to legality, resource budgets, and user-selected fusion / delegation
> constraints

This is the right place to represent "one megakernel" versus "ten launches
instead of four hundred" versus "best latency overall."

The architecture should expose at least two policy dimensions.

#### Fusion policy

- `fusion_policy="strict"`
  - best effort single megakernel
  - equivalent to `max_regions = 1`
  - useful for research, purity, and architecture validation

- `fusion_policy="budgeted"`
  - constrain the compiler to a launch budget such as `max_regions = 10`
  - useful when the product goal is "dramatically fewer launches" rather than
    absolute monolithic purity

- `fusion_policy="auto"`
  - no explicit region-count cap
  - optimize purely for latency subject to legality and cost model

#### Delegation policy

- `delegation_policy="none"`
  - pure megakernel mode
  - no external library escape for the chosen region

- `delegation_policy="fallback_only"`
  - escape only when legality or correctness requires it

- `delegation_policy="perf"`
  - allow external kernel families when the cost model and tuning data say they
    win

#### Recommended interpretation

- `strict_fusion=True` should be treated as a shorthand for:
  - `fusion_policy="strict"`
  - `max_regions = 1`

- the practical middle ground should be:
  - `fusion_policy="budgeted"`
  - `max_regions = N`

This lets Megabake support both:

- a pure research mode that asks "can this become one megakernel?"
- a performance mode that asks "what is the fastest region decomposition?"

That split is especially important for prefill. Prefill may eventually want the
same persistent machinery, but it should not be forced into strict one-kernel
form if that destroys GEMM or attention quality for the current hardware.


### 5.2 Reliable Win Objective

The primary optimization target for v2 should be stated plainly:

> Megabake should reliably beat `torch.compile` end-to-end latency by combining
> near-baseline or slightly better kernel quality on the hot regions with
> dramatically lower orchestration and launch overhead.

This means the compiler should optimize **total latency**, not fusion count.

A useful mental model is:

- `torch.compile total = launched-kernel compute + launch / orchestration gap`
- `megabake total = persistent-region compute + tiny scheduling overhead`

The goal is therefore:

`megabake_compute + megabake_schedule_overhead < torch_compile_compute + torch_compile_launch_gap`

The current traces already show why this matters:

- **SmolLM2 decode**
  - `torch.compile`: about `1175 us` compute and `7607 us` inter-kernel gap
  - `megabake`: about `4333 us` compute and almost no gap
  - result: Megabake wins span despite worse raw compute

- **gemma-2b decode**
  - `torch.compile`: about `6197 us` compute and `1715 us` inter-kernel gap
  - `megabake`: about `11442 us` compute
  - result: Megabake loses because the compute gap is larger than the launch-gap advantage

So the architecture must be designed around two rules:

1. **always preserve the large launch-time advantage**
2. **never let the compute gap on hot regions grow larger than the launch-time advantage can pay for**

This is why:

- strict one-kernel mode is valuable
- but the default performance mode must be free to segment or delegate when that
  is the only way to keep the total-latency inequality favorable

The compiler should therefore carry an explicit notion of a **reference launched
plan** or **TorchCompile surrogate baseline** and compare candidate Megabake
plans against it during tuning.


## 6. V2 IR Stack

V2 should use **three real compiler layers**, not a large tower of first-class
IRs. The goal is to keep the semantics clean without repeating v1’s mistake of
lowering too early.

The rule is:

- many semantic passes are fine
- many heavyweight IR universes are not

### 6.1 Layer 1: FX Graph + Fact Tables

This is still fundamentally the exported FX / ATen graph, but it is paired with
rich analysis side tables.

**Before this layer**

- raw export graph
- decomposed transformer patterns
- many metadata-only shape/layout ops
- no trustworthy global fact table

**After this layer**

- graph is canonicalized and cleaned
- constants are folded where appropriate
- views are tracked as metadata until materialization is required
- every value has attached facts

**What this layer carries**

- graph topology
- canonical ATen-ish ops after decomposition
- shapes, dtypes, layouts, strides, offsets, alias groups
- constant values where known
- bucket key
- memory-effect markers (`pure`, `view`, `materialize`, `stateful update`)

**What this layer is allowed to decide**

- semantic normalization
- shape / layout / alias facts
- whether a view stays virtual or must materialize
- motif-friendly graph cleanup

**What this layer must not decide**

- warp roles
- SM assignment
- final kernel family
- final fusion budget
- final persistent schedule

This layer should aggressively reuse PyTorch / Inductor machinery:

- `torch.export`
- decompositions
- canonicalization
- constant folding
- CSE / DCE
- fake tensor / symbolic shape infrastructure
- pattern matcher infrastructure where possible

### 6.2 Layer 2: RegionGraph

This is the first truly Megabake-specific IR.

Instead of asking "what ATen nodes do I have?" the compiler now asks:

> what semantic regions do I have, and what are the candidate ways to execute
> them?

**Before this layer**

- a canonicalized graph of cleaned ATen-ish ops
- all semantic facts known
- no final execution structure yet

**After this layer**

- the graph is compressed into meaningful execution regions
- candidate kernel families exist per region
- fusion boundaries are explicit
- residency opportunities are explicit

**Region kinds should include at least**

- `MatvecRegion`
- `MatmulRegion`
- `AttentionRegion`
- `NormPointwiseRegion`
- `RopeRegion`
- `KVCacheRegion`
- `ExternRegion` for bring-up / correctness / non-golden-path cases

**What this layer carries**

- region kind
- input / output tensors
- shape and layout keys
- region adjacency
- residency opportunities
- candidate implementation families
- estimated byte / flop counts
- fusion / split constraints

**What this layer is allowed to decide**

- which nodes belong together semantically
- which candidate families are legal
- where fusion boundaries live
- where handoff / residency is even possible

**What this layer must not decide**

- exact per-SM program
- exact prefetch instruction sequence
- final warp-level implementation

This layer should be **HF-aware but not HF-dependent**:

- HF transformers metadata and configs can inform priors
- exported FX graph remains the source of truth
- the IR must stay generic enough for non-transformer models later

### 6.3 Layer 3: ScheduleProgram

This is the persistent-program IR.

This is the layer that finally answers:

> how does this exact RegionGraph become 1, 8, or 15 persistent regions on this
> target under this fusion policy?

**Before this layer**

- semantic regions are known
- candidate families are known
- target constraints and tuning priors are known

**After this layer**

- a concrete persistent execution program exists
- region count is fixed
- per-SM work is fixed
- prefetch and handoff actions are fixed
- the artifact can be emitted

**What this layer carries**

- region count / segmentation plan
- per-SM static programs
- tile descriptors
- warp-role descriptors
- dependency tokens
- prefetch actions
- handoff actions
- release / completion actions
- codegen-family choices

**What this layer is allowed to decide**

- exact persistent schedule
- exact segmentation count
- exact tile and warp strategy
- exact handoff and prefetch actions

**What this layer must not decide**

- raw semantic graph questions like "is this RMSNorm?" or "should this view fold?"

If those questions survive this late, the earlier layers failed.

### 6.4 Auxiliary Compiler Objects

These are not semantic IR layers, but they are first-class compiler inputs.

#### `TargetModel`

`TargetModel` should describe:

- architecture family
- tensor core / MMA capabilities
- async copy / TMA capabilities
- register and SMEM budgets
- occupancy model
- memory bandwidth and latency estimates
- launch overhead estimates

#### `AutotuneDB`

`AutotuneDB` should store:

- target fingerprint
- bucket key
- region kind
- candidate schedule choices
- measured performance
- selected winner
- reference baseline comparison, if available

#### `ArtifactPack`

The final emitted object should contain:

- schedule program blob
- bucket descriptor
- target fingerprint
- prepacked weight layout metadata
- debug maps back to regions / graph nodes
- cubin / fatbin references

### 6.5 Layer Boundary Sanity Check

This is the easiest way to avoid repeating v1.

If a layer violates these tests, the architecture is drifting:

- if Layer 1 starts deciding GPU schedule details, it is too low-level
- if Layer 2 starts looking like packed task records, it is too low-level
- if Layer 3 still needs to rediscover transformer semantics, earlier passes are too weak

The goal is not "many IRs." The goal is:

- **Layer 1 removes semantic uncertainty**
- **Layer 2 removes structural uncertainty**
- **Layer 3 removes execution uncertainty**


## 7. End-to-End Compiler Pipeline

This is the v2 compile pipeline from first principles after simplifying the IR
story. The guiding idea is:

- reuse Inductor / PyTorch for standard graph cleanup
- build Megabake-specific logic only once the problem becomes persistent-program
  synthesis
- compare candidate Megabake plans against a TorchCompile-style reference plan

### Stage 0: Workload regime, target fingerprint, and optional region census

Input:

- `nn.Module`
- example inputs
- target mode (`decode`, later `prefill`)
- target device

Output:

- `BucketKey`
- `TargetFingerprint`
- `TargetModel`

This stage decides what regime we are compiling for:

- model family
- mode
- batch bucket
- sequence bucket
- hidden-size / head-dim family
- dtype family
- target fingerprint

There are really two related activities here:

1. **per-compile target binding**
   - identify the bucket and target

2. **offline workload characterization**
   - optionally run a region census over a corpus of exported HF transformer
     graphs to learn the most common region shapes and transitions

That second part is not required for correctness, but it is highly valuable for
architecture and tuning priorities.

### Stage 1: Export, canonicalize, and build fact tables

Input:

- model + example inputs

Output:

- Layer 1: `FXGraph + FactTables`

Implementation bias:

- reuse `torch.export`
- reuse decomposition tables
- reuse graph cleanup / canonicalization
- reuse fake tensor / symbolic shape machinery
- reuse pattern matcher infrastructure where possible

Passes in this stage:

- decomposition
- canonicalization
- constant folding
- CSE / DCE
- shape / dtype / layout propagation
- alias analysis
- view folding
- copy insertion where required

At the end of this stage, the graph should be:

- semantically clean
- fact-rich
- still hardware-neutral

### Stage 2: Build the RegionGraph

Input:

- Layer 1: `FXGraph + FactTables`
- optional transformer priors from HF metadata

Output:

- Layer 2: `RegionGraph`

This is where Megabake becomes Megabake.

The compiler should:

- recover norms, RoPE, QKV bundles, attention, MLP patterns, KV-cache updates,
  and residual epilogues
- group nodes into semantic execution regions
- record region adjacency and residency opportunities
- attach candidate implementation families to each region

HF integration matters here, but as a **front-end prior**, not as a hard
dependency:

- HF configs can inform likely region shapes and motif families
- the exported FX graph remains the source of truth

### Stage 3: Enumerate candidates, including a launched reference baseline

Input:

- Layer 2: `RegionGraph`
- `TargetModel`
- compiler policy (`fusion_policy`, `delegation_policy`, `max_regions`)

Output:

- candidate sets per region
- a reference launched plan

This stage should generate two kinds of candidates:

1. **Megabake candidates**
   - persistent matvec variants
   - persistent attention variants
   - norm / pointwise microprogram regions
   - segmented persistent-region plans

2. **Reference launched candidates**
   - a TorchCompile-style surrogate plan consisting of strong launched kernels
     plus expected orchestration cost

This is critical for the project goal. If v2 wants to *reliably beat*
`torch.compile`, then its tuner needs an explicit baseline to beat, not just an
internal score.

### Stage 4: Cost modeling and plan selection

Input:

- region candidates
- reference launched plan
- `TargetModel`
- policy constraints

Output:

- selected region family per region
- selected segmentation plan
- selected residency strategy
- selected reference comparison margin

The cost model should answer:

1. which Megabake candidate wins for this region?
2. how many regions should the final program have?
3. does the chosen plan beat the reference launched plan by enough margin?

This is where:

- `strict` mode enforces `max_regions = 1`
- `budgeted` mode enforces a launch cap
- `auto` mode optimizes pure latency
- delegation constraints are applied

### Stage 5: Synthesize the ScheduleProgram

Input:

- chosen region plan
- chosen residency plan
- `TargetModel`

Output:

- Layer 3: `ScheduleProgram`

This stage produces the actual persistent execution program:

- region count
- per-SM programs
- tile descriptors
- warp roles
- dependency tokens
- prefetch actions
- handoff actions
- release actions

This stage must happen at compile time, not in the runtime loader.

### Stage 6: Generate target-specific code

Input:

- `ScheduleProgram`
- chosen region families

Output:

- target-specific code objects

Codegen families should include:

- decode matvec kernels
- attention kernels
- norm / pointwise executors
- persistent schedule interpreter / program loop

TileIR can fit here as an **optional backend kernel IR** for selected
tile-centric region families. The intended use is narrow and deliberate:

- do **not** replace `RegionGraph`
- do **not** replace `ScheduleProgram`
- do use TileIR as a possible lowering target for compute-heavy regions such as
  matvec / matmul / attention if it improves backend portability or kernel
  quality for those regions

The admission rule should be strict:

- CUDA/CuTe remains the default backend path
- TileIR is considered only after region boundaries and schedule are already fixed
- TileIR is considered only for tile-centric, compute-heavy regions
- TileIR stays only if replay tuning shows it beats or materially improves the
  default backend for that region family

In other words, TileIR is a candidate implementation detail of `KernelBundle`,
not a replacement for the main Megabake IR stack.

The important point is that codegen comes **after** schedule decisions, not
before.

### Stage 7: Region replay tuning and acceptance against the baseline

Input:

- target-specific candidate objects
- representative region replays
- optional short forward-pass snippets
- reference launched plan

Output:

- tuned winner stored in `AutotuneDB`
- measured margin over reference

The compiler should not tune only the whole model and not only synthetic
microbenchmarks.

It should:

- extract hot region replays from the actual model
- tune them on real bucket shapes
- validate interactions on short forward snippets
- keep only plans that beat the launched reference baseline by a margin

This is how "reliably beat `torch.compile`" becomes an engineering loop rather
than a hope.

### Stage 8: Emit the ArtifactPack

Input:

- tuned `ScheduleProgram`
- tuned code objects
- bucket and target metadata

Output:

- `ArtifactPack`

This should include:

- schedule program blob
- bucket descriptor
- target fingerprint
- weight layout / quantization descriptor
- code object references
- debug map


## 8. Persistent Runtime Model

The runtime should become intentionally small.

It should do only:

1. fingerprint the target and pick the matching artifact bucket
2. bind prepacked weights
3. bind input/output pointers
4. bind dynamic bucket fields if needed
5. optionally run first-use calibration / replay tuning if no tuned artifact exists
6. persist tuning results into `AutotuneDB`
7. launch the persistent program
8. collect optional telemetry

It should *not* rebuild the schedule graph, rediscover dependencies, or invent
the SM program late.


## 9. Persistent Program Model

At runtime, the GPU should conceptually execute something closer to this:

```text
for each SM:
    load SMProgram[sm_id]
    for each action in program:
        if action == WAIT_DEP:
            wait until dependency token reaches zero
        if action == PREFETCH:
            cp.async / TMA next weight tile or region payload
        if action == RUN_REGION:
            execute specialized kernel body for this tile
        if action == HANDOFF:
            publish data in agreed SMEM/register slot
        if action == RELEASE_DEP:
            decrement successor tokens
```

The important difference from v1 is that the "program" is already compiled. It
is not reconstructed from generic tasks in the loader.

The important difference from a hand-written backend is that the program is
chosen through target modeling and empirical replay tuning, not just hardcoded
per architecture.


## 10. Kernel Families

### 10.1 Decode matvec family

This is the first-class kernel family for v2.

Requirements:

- shape-specialized for `M = 1..4`
- tuned for exact hidden sizes / projection sizes
- support fused bias / activation / residual epilogues
- support weight-only quantized variants early
- explicit residency / prefetch strategy

The compiler should treat this family as more important than attention on the
initial decode target.

This does not mean decode is the only profitable regime. It means v2 should
prove itself on one regime first while leaving room for other bucket families to
win later through the same target-model + tuning pipeline.


### 10.2 Attention family

The attention family should be decode-specific first:

- single-query or small-query decode attention
- explicit K/V cache layout assumptions
- tensor-core capable variants where it matters
- hardware-specific blocking

Attention should still be architected properly, but it is not the first
performance hill to die on for the current decode traces.


### 10.3 Pointwise / norm family

This family should handle:

- RMSNorm / LayerNorm
- RoPE
- pointwise epilogues
- small reductions
- broadcast-heavy chains

These can remain microprogram / interpreter-friendly as long as they do not
become the decode bottleneck.

### 10.4 Backend Surface: Handwritten vs Generated

V1 accumulated many handwritten CUDA kernels because its architecture is roughly:

1. normalize the FX graph a bit
2. map recognized ops or patterns to `TaskDesc`
3. dispatch each task to a handwritten CUDA implementation

That naturally creates an ever-growing backend surface:

- handwritten matmul
- handwritten attention
- handwritten reduce
- handwritten rope
- handwritten embedding
- handwritten index
- handwritten copy
- handwritten fused elementwise
- then handwritten special cases for more patterns over time

That is useful for bring-up, but it is not the long-term shape of a compiler.
It makes the project feel more like an op-to-kernel selector than a region
compiler.

V2 should draw a much harder line:

> handwritten per-op kernels are a smell  
> handwritten backend primitives are normal

The right goal is not "zero handwritten CUDA." The right goal is:

> handwrite the machine-level substrate once, then generate the model-specific
> behavior automatically

#### What should remain handwritten in v2

Only a small backend substrate should be handwritten and maintained directly:

1. **Persistent runtime program**
   - the megakernel execution loop
   - dependency handling
   - prefetch and handoff primitives
   - warp / SM orchestration

2. **A small number of kernel families**
   - matvec / GEMM family
   - attention family
   - pointwise / reduction engine
   - optional data movement / layout engine

3. **Hardware utility layer**
   - cp.async / TMA wrappers
   - tensor-core / MMA / WGMMA wrappers
   - residency / paging helpers
   - synchronization helpers

This is the stable, low-level substrate of the compiler.

#### What should be generated in v2

Everything graph-specific or fusion-specific should come from the compiler:

- region decomposition
- fusion boundaries
- launch count / segmentation plan
- epilogue composition
- pointwise chains
- small reductions
- layout-specific glue logic
- residency and prefetch plans
- per-bucket schedules
- tuned tile choices

In other words, the compiler should generate:

- **which backend family to use**
- **how it is parameterized**
- **how regions are stitched together**

but it should not require a new handwritten CUDA file every time a new pattern
appears in the graph.

#### The v2 maintenance rule

If a new model feature requires:

- a new schedule
- a new epilogue
- a new pointwise chain
- a new launch segmentation

then the fix should land in compiler IR or schedule generation.

If it requires:

- a fundamentally new machine primitive
- a new tensor-core strategy
- a new persistent runtime mechanism

then it may justify new handwritten backend substrate.

That rule is how v2 avoids turning back into a curated kernel zoo.


## 11. Cost Model

The cost model should answer two questions:

1. Which kernel family should implement this region?
2. Should this boundary stay fused or split?

In practice it should answer a third question too:

3. Does the resulting Megabake plan beat the launched reference baseline by a
   sufficient margin?

Inputs:

- shape bucket
- target fingerprint
- dtype
- layout
- SMEM need
- register estimate
- bytes from HBM
- tensor-core availability
- expected occupancy
- predecessor / successor residency opportunities
- launched reference-plan estimate

Outputs:

- chosen candidate
- accepted fusion edges
- tile family
- prefetch policy
- region count / segmentation plan
- delegation decisions, if allowed by policy
- predicted win / loss margin against the reference launched plan

Even if the final architecture remains "pure megakernel" for a chosen bucketed
regime,
the cost model is still required. Otherwise the compiler has no formal reason
for its decisions.

This cost model should not remain purely analytical forever. It should be
calibrated against real measurements and eventually defer to `AutotuneDB` when
bucket-specific empirical winners exist.

The architecture should treat "beat `torch.compile`" as a first-class scoring
target, not a side effect. So the cost model and tuner should both compare
Megabake candidates against a reference launched plan and reject candidates that
lose unless the user explicitly requested strict purity.

The cost model must also honor compiler policy:

- in `strict` mode, it optimizes under `max_regions = 1`
- in `budgeted` mode, it optimizes under a launch budget
- in `auto` mode, it can choose the segmentation that minimizes latency
- if delegation is disabled, candidates from external kernels are illegal


## 12. Artifact Format

The new artifact format should be closer to:

```text
ArtifactHeader
BucketDescriptor
TargetFingerprint
WeightLayoutDescriptor[]
RegionDescriptor[]
TensorDescriptor[]
ScheduleDescriptor
  SMProgram[]
  DependencyTable
  TileDescriptor[]
  PrefetchPlan[]
  HandoffPlan[]
DebugMap
CodeObjectRefs
```

The schedule should already contain:

- dependency structure
- SM programs
- queue / tile plan
- prefetch instructions
- handoff instructions

The runtime should only bind pointers and launch.


## 13. Suggested Module Layout

V2 should probably become a parallel module tree, not an edit-in-place of
`graph_walker.py`.

Suggested Python structure:

```text
src/megabake/v2/
  capture/
    export.py
  frontends/
    hf.py
  target/
    model.py
    fingerprint.py
    calibrate.py
  ir/
    canonical.py
    region.py
    memory.py
    schedule.py
    artifact.py
  analysis/
    shapes.py
    layouts.py
    aliases.py
    constants.py
    patterns.py
  corpus/
    region_census.py
    workloads.py
  passes/
    canonicalize.py
    recover_motifs.py
    form_regions.py
    enumerate_candidates.py
    plan_memory.py
    build_schedule.py
    emit_artifact.py
  cost_model/
    roofline.py
    occupancy.py
    residency.py
  baseline/
    reference_plan.py
    torch_compile.py
  autotune/
    db.py
    search.py
    replay.py
    runners.py
  backend/
    kernel_ir.py
    cuda_cute.py
    tileir.py
  runtime/
    loader.py
    launcher.py
```

Suggested CUDA / kernel layout:

```text
src/cuda_v2/
  megakernel.cu
  runtime_program.cuh
  regions/
    matvec_decode.cu
    attention_decode.cu
    norm_pointwise.cu
  tileir/
    README.md
  common/
    data_types.cuh
    cp_async.cuh
    tensor_core.cuh
    residency.cuh
```


## 14. Migration Plan from V1

### Keep / adapt

- `inductor_passes.py` as inspiration for front-end normalization
- pieces of `shape_ops.py`
- `buffer_planner.py` concepts, but move them later into a proper memory IR
- serializer concepts, but redesign the payload
- CUDA building / loading machinery as scaffolding

### Rewrite

- `graph_walker.py`
- `tiling.py`
- `scheduler.py`
- `dependency.py`
- `runtime/loader.py`
- the current task-centric schedule format

### Eventually delete or de-emphasize

- one-pass task emission as the main compiler path
- runtime reconstruction of DAG + SM queues
- ad-hoc overload of `TaskDesc.strides[]` as the main carrier for dispatch
  semantics
- proliferation of op-specific handwritten CUDA files as the primary extension
  mechanism


## 15. Implementation Phases

### Phase A: Canonical graph compiler skeleton

Deliver:

- `TargetModel`
- HF-aware front-end hooks
- optional region census tooling
- Export Graph IR
- Canonical Tensor IR
- motif recovery

Do not build codegen yet beyond debug dumps.


### Phase B: Region IR and cost model skeleton

Deliver:

- region formation
- candidate enumeration
- reference launched-plan generator
- basic cost model
- target-parameterized candidate selection

Again, prioritize visibility over completeness.


### Phase C: Memory IR and schedule IR

Deliver:

- explicit residency plan
- compile-time SM program
- serialized schedule artifact

This is the point where scheduling leaves the runtime and becomes a compiler
product.


### Phase D: Decode matvec-first codegen

Deliver:

- one great decode matvec family
- fused epilogues
- weight prepack / quantization support

This is the first performance milestone.


### Phase E: Attention family

Deliver:

- decode-specialized attention family
- integrated schedule actions


### Phase F: Autotuning and artifact packaging

Deliver:

- replay-based tuning loop
- `AutotuneDB`
- bucketed artifact store
- simple runtime artifact selection
- acceptance against a TorchCompile-style baseline


## 16. Bottom Line

Megabake v2 should be designed as:

> a staged compiler that lowers an FX graph into a bucketed region graph, then
> into a target-parameterized memory-resident persistent schedule, then into
> late-bound target-specific code chosen and refined by empirical tuning against
> a TorchCompile-style launched baseline.

It should not be designed as:

> a better one-pass graph walker that emits slightly smarter low-level task
> records.

That is the main architectural line between the current project and the next
version, and it is the mechanism by which v2 should aim to reliably beat
`torch.compile`: slightly better or near-parity kernels on the hot regions, plus
consistently much lower orchestration cost.
