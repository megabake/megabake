# MegaBake V3: atomic implementation work orders

Status: implementation plan only, 2026-09-09. None of the tasks below is implemented by this
documentation change. The current repository base is `3695f06de14322fd8ac3e111c693612d53b56319`;
recheck the working tree before starting. Proposed paths, interfaces, commands and artifacts are
explicit implementation targets, not claims that those files or tools already exist.

This is the execution handbook for the complete pipeline-first V3 design. It replaces the earlier
E0–E5-only experiment outline with individually assignable work orders. A task should produce one
reviewable behavioral change and its tests, not an entire subsystem built from a vague objective.
The detailed plan does not authorize starting implementation during the documentation task.

## 1. How to give this plan to an implementation agent

Assign **one task ID**, its dependency handoffs, sections 1–5, and its required reading. The agent
should read the specified source files itself, perform the task, run its tests, and stop at its
acceptance boundary. It must not implement every later task because it has the full document.

A well-specified plan reduces the reasoning burden; it does not eliminate the need for review.
CUDA memory ordering, continued reductions and whole-entry composition still need a reviewer
competent in that area. Do not treat a model name, confidence statement or green mock test as
evidence of device correctness.

### 1.1 Authority and non-negotiable scope

- The [architecture](MEGABAKE_V3_ARCHITECTURE.md) owns the pipeline-first scope.
- [IR contracts](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) own semantic and ExecutionPlan meaning.
- [Pipeline contracts](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) own stages, publication, storage and progress.
- [Performance protocol](MEGABAKE_V3_PERFORMANCE_MODEL.md) owns timings, controls and claim rules.
- This document owns task order, concrete module seams, fixtures, handoffs and acceptance tests.
- If these disagree, identify the precise conflict and request a decision; do not silently
  change architecture while implementing a task.

Keep three representations: normalized FX, SemanticGraph within FX, and ExecutionPlan. TargetProfile
and LayerSummary remain analyses. Preserve the old backend as a control until an explicitly
authorized migration. No general Tile IR, new graph framework, binary cuBLAS extraction, generic
queue runtime, paged scratch allocator, quantization rescue or multi-GPU expansion is required here.

The target remains generic FX input with an explicit supported subset, one owned compute grid for
strict execution, batch-one cached decoding, and the model's declared FP16/BF16 policy. Initial
model evidence must include a small model and a roughly 2B model at declared short/long contexts.
Uncached sequence-one execution is a diagnostic, not a cached-decode result.

### 1.2 Definition of an atomic task

One task has one principal observable outcome, a bounded edit surface, an independently testable
contract and an evidence handoff. A task may touch implementation and test files together. It does
not mean one line or one file. If a device-library adaptation cannot meet that scope, split it into
numbered child tasks such as `MB3-057a`, each with its own test; retain the parent acceptance gate.
Do not turn “blocked” into permission for a broad rewrite.

Dependencies indicate minimum prerequisites, **not** a demand to complete every lower-numbered
task first. A focused body experiment can run while unrelated frontend matchers are unfinished.
No task requires an isolated kernel to beat the vendor before useful pipeline work may start.
An explicitly marked optional input is not a dependency gate. If supplied, it must satisfy its
own correctness/compatibility requirements; if absent, claims about that capability remain absent.

### 1.3 Completion is an evidence vector

Use these independent fields in each handoff:

| Field | Allowed values | Meaning |
|---|---|---|
| implementation | not_started / partial / patch_ready / reviewed | Code delivery and review, not measured success |
| cpu_validation | not_run / pass / fail / not_applicable | Actual CPU commands and results |
| cuda_compilation | not_run / pass / fail / not_applicable | Selected target/toolchain only |
| gpu_correctness | not_run / pass / fail / not_applicable | Numerical, state and applicable protocol tests |
| gpu_performance | not_measured / measured_win / measured_loss / inconclusive / not_applicable | Named comparison with raw samples |
| disposition | continue / revise / rejected_candidate / blocked_external | Next action and exact reason |

A GPU task may have `patch_ready` code while its GPU fields remain `not_run`. That does not satisfy
a downstream **GPU-validation** gate. A rejected library candidate can complete a compatibility
experiment; it cannot satisfy a dependency requiring a working body. A measured loss is evidence,
not a reason to conceal the result. Final research success is separate from task completion.

### 1.4 Reviews and hardware lanes

- **CPU:** may be fully validated without CUDA; optional dependencies must not initialize devices.
- **CT:** requires an explicitly selected CUDA toolkit/headers to validate compilation; no GPU
  performance or residency claim follows. Code may be drafted without CT access.
- **GPU:** requires the selected actual CUDA device for the stated acceptance gate.
- **R1:** ordinary implementation/test review.
- **R2:** graph, alias, numerical, ABI or measurement-contract review.
- **R3:** device synchronization, async lifetime, reduction or composition review. Require an
  independent competent reviewer before the feature is enabled in accepted strict runs.

Review requirements describe future execution of the plan. They are not a request to spawn agents
during this documentation-only turn. Parallel future tasks must use isolated worktrees or distinct
owned files; one integration owner resolves shared interface edits.

### 1.5 Every task inherits these acceptance rules

1. Confirm dependency artifacts describe the actual checkout, not an obsolete patch.
2. Add a regression that would fail before the task, then the implementation, then negative cases.
3. Do not change dtype, cast order, mask/state behavior, output ownership or benchmark work to win.
4. Unsupported cases fail with a node/guard/capability diagnostic; no hidden fallback in strict mode.
5. No new GPU work during import or CPU-only planning. No downloads, toolkit upgrades or paid
   resources silently introduced by a test; declare and obtain any needed access first.
6. New public behavior is opt-in until the integration tasks. Do not delete historical tests/results.
7. Preserve unrelated user changes. Reverting a failed candidate means a scoped reviewed patch,
   never resetting the worktree or deleting an entire build/model cache.
8. Report actual commands, passed/failed/skipped counts, artifacts and remaining uncertainty.
9. Performance estimates stay labelled estimates. No per-task universal percentage guarantee.

## 2. Repository map and proposed implementation seams

### 2.1 Existing code to inspect, not blindly preserve

| Existing path | Relevant current behavior | Planned treatment |
|---|---|---|
| [public entry](src/megabake/__init__.py) | compile_fx queries CUDA; public imports pull in runtime | Preserve legacy API; add an explicit V3 entry later |
| [normalization](src/megabake/schedule_compiler/inductor_passes.py) | core decompositions plus private pre_grad call | New pinned adapter with equivalence tests |
| [graph walker](src/megabake/schedule_compiler/graph_walker.py) | ATen to TaskDesc; early task fusion; one-output metadata | Coverage/reference inventory, not V3's semantic ABI |
| [shape helpers](src/megabake/schedule_compiler/shape_ops.py) | StridedView and view transformations | Reuse individually verified rules |
| [dependency builder](src/megabake/schedule_compiler/dependency.py) | Buffer-level producer edges and arena anti-dependencies | New exact tile/effect/readiness analysis |
| [scheduler](src/megabake/schedule_compiler/scheduler.py) | Estimated costs, per-worker queues and heuristic prefetch | Historical control; do not transplant its cost/lifetime assumptions |
| [buffer planner](src/megabake/schedule_compiler/buffer_planner.py) | Sequential live intervals | New partial-order allocation for V3 |
| [CUDA compiler](src/megabake/runtime/cuda_compiler.py) | Eager CUTLASS discovery, all-task concatenation, fast math | First make imports safe; V3 gets selected-source codegen |
| [launcher](src/megabake/runtime/launcher.py) | Global module/function, fixed launch envelope | Separate typed per-context V3 launcher |
| [loader](src/megabake/runtime/loader.py) | Global runner, half conversion, clones and arena | Session-owned V3 state/bindings |
| [HF adapter](src/megabake/integrations/transformers.py) | use_cache=False wrapper | Preserve; add a distinct stateful V3 adapter |
| [task bodies](src/cuda/tasks/matmul.cu) | Existing task ABI and skinny mapping | Compare against new bounded device bodies |
| [comparison harness](benchmarks/bench_compare.py), [model harness](benchmarks/test_harness.py) | Filtered profiles and mismatched reporting contracts | Keep historical routes; build a V3 evidence harness |

The eager header-resolution detail is observable in `_CUTLASS_INCLUDE = _resolve_cutlass_include()`
at module scope. CPU imports can fail before any compilation request. Fix that narrow seam first;
do not assume having CPU tests already means they collect on a machine without CUDA headers.

### 2.2 Proposed paths and ownership

These paths do **not** exist yet. Prefix aliases below are documentation shorthand, not environment
variables or extra IRs. Expand them literally when creating files:

```text
V3     = src/megabake/v3
CPU    = tests/test_v3/cpu
CTEST  = tests/test_v3/toolchain
GTEST  = tests/test_v3/gpu
CUDA   = src/cuda/v3
BENCH  = benchmarks/v3
ART    = artifacts/v3
```

Proposed organization:

```text
V3/
  contracts.py, diagnostics.py, target.py, backend.py
  frontend/
    capture.py, normalize.py, facts.py, effects.py, semantic.py
    match_linear.py, match_pointwise.py, match_norm.py
    match_rope.py, match_attention.py, match_state.py
    composites.py, layers.py, inventory.py
  plan/
    model.py, bodies.py, footprints.py, dependencies.py
    storage.py, verify.py, simulate.py, templates.py, costs.py, search.py
  codegen/
    source.py, compile.py, resources.py
  runtime/
    driver.py, session.py
CUDA/
  abi.cuh, entry support headers, selected bodies/
BENCH/
  fixtures.py, timing.py, trace.py, baselines.py
  bench_shapes.py, calibrate.py, ablations.py, scorecard.py
```

Do not create every empty module in the first task. Each owner creates only what its card needs.
Keep related small definitions together; splitting a module requires a concrete readability or
ownership reason, not a one-class-per-file rule. CPU tests import only Python/reference layers.

Shared device-test helpers live in `tests/test_v3/support/body_harness.py` and
`tests/test_v3/support/pipeline_harness.py` when their owning tasks need them. These helpers construct
test inputs/plans and invoke the same production emitter/runtime; they must not become a second
executor whose success substitutes for generated production code. Body “registration” edits mean
the corresponding BodySpec/source association in the existing V3 registry seam, not an unrelated
global legacy dispatch change. Serialize shared registration edits through the interface owner.

### 2.3 Public boundary to implement, not today's API

```python
normalize_fx(graph, example_args=None, *, input_spec, policy) -> NormalizedProgram
recognize(program) -> SemanticGraph
make_plans(semantic_graph, target, body_registry, options) -> CandidateSet
verify_plan(plan, semantic_graph, target) -> VerificationReport
emit_entry(verified_plan, body_registry) -> SourceArtifact
compile_entry(source_artifact, toolchain) -> CompiledArtifact
create_session(compiled_artifact, bindings, state, *, device, stream_policy) -> Session
session.run(*inputs) -> owned_output_tree
session.run_into(output_tree, *inputs) -> explicit caller-owned output_tree
```

These are shared contracts to implement incrementally. Use keyword-only contracts for choices
that alter semantics. A candidate lacking costs may be examined/emitted offline but cannot be
labelled a measured winner. Never make `normalize_fx` query a GPU.

## 3. Shared interfaces that prevent agents from inventing incompatible pieces

### 3.1 Identifiers, values and regions

Use stable invocation-independent IDs: `ValueId`, `OpId`, `TileId`, `ActionId`, `EventId`,
`BufferId`, `BodyId`. Deterministic integers or canonical strings are sufficient; no distributed
ID service. A structural hash must not contain raw pointers or Python object IDs.

A tensor value records logical shape, dtype, strides, storage offset, alias identity, role
(input/weight/state/intermediate/output), numerical provenance and origin FX nodes. Use explicit
unknown facts; do not replace unknown dtype size with two bytes. Initially specialize finite
shape buckets. Preserve symbolic constraints even if a body requires concrete dimensions.

Represent a logical footprint by value ID plus half-open multidimensional regions and read/write
mode. Compose indexing through proven views. Unknown/non-affine overlap is conservative, not
disjoint by default. A reduction update includes the full required activation/weight chunk and
its accumulator dependency. A finalizer is a distinct semantic action.

### 3.2 Shared records

Implement small dataclasses/enums first; no dependency on a new graph or schema framework.

| Record | Minimum contract |
|---|---|
| WorkloadSpec | checkpoint/config identity, batch/context/capacity, dtype, state/mask semantics, input origin, output ownership, timed unit |
| NumericalPolicy | reference expansion identity, intermediate casts, accumulation/reassociation allowances, dtype/op-specific tolerances and exceptional-value policy |
| NormalizedProgram | FX graph, input/output pytree and export signature, lifted bindings, constraints, effects, reference callable |
| SemanticGraph | FX dialect nodes, executable reference regions, origin mapping, facts and composite alternatives |
| BodySpec | semantic support predicate, version/features, thread roles, tile/footprint generator, scratch/accumulator requirements, capability set, source/descriptor dependencies |
| ExecutionPlan | the schema in IR §7; unresolved choices prohibited in a finalized plan |
| CostRecord | key, value/unit or unknown, measured/estimated provenance, interval, working-set and contention conditions |
| SourceArtifact | selected source/header hashes, entry name, typed argument schema, numerical flags, target requirements, plan hash |
| CompiledArtifact | source/toolchain identity, binary location/hash, entry attributes, compiler logs, unresolved runtime checks |
| VerificationReport | independent coverage/storage/events/participation/progress/guards results with counterexample IDs |
| TaskHandoff | task/dependency revisions, changed files, commands/results, evidence vector, remaining risks and next eligible tasks |

An implementation source path does not define its semantics. A source hash change invalidates
its compiled artifact and relevant performance costs. A changed numerical policy invalidates
reference comparisons. Re-run launch admission when the visible device profile changes.

### 3.3 Action/token rules

Use `reserve`, `preload`, `compute`, `reduce_begin`, `reduce_update`, `reduce_finalize`,
`publish`, `release` and `join` as the initial action families. Keep address-ready, data-ready,
load-complete, accumulator-ready, destination-visible and source-retired meanings distinct.

Atomic-tile bodies drain declared accesses before returning. Staged bodies may return outstanding
operation tokens only with explicit ownership/lifetime transfer. A same-CTA barrier is not a
substitute for an async completion, a cross-CTA publication, or a grid collective.

For the first inter-CTA event protocol: positive expected producer counts, same-grid zero
initialization plus a uniform grid barrier, exact unique producer contributions, device-scope
acquire-release publication and acquire consumption, and no within-invocation event-counter reuse.
Activation chunks have unique addresses inside a region. Small per-CTA staging rings use their
own explicit phase/release protocol. No consumer-refcount ring reclamation is implicitly present.

### 3.4 Fixed testing vocabulary

Create these fixtures incrementally; their definitions belong to the task that first needs them:

| Fixture | Definition and purpose |
|---|---|
| LINEAR_TINY | M=1, N=17, K=33; weights in explicit NK and KN views; tail/stride handling |
| LINEAR_HOT | (M,N,K): (1,576,576), (1,1536,576), (1,576,1536), (1,49152,576), (1,4096,4096); historical hints, not new measurements |
| GATE_TINY | H=32, I=65, chunks [0,16), [16,32), [32,48), [48,64), [64,65); separate SiLU and GELU references |
| NORM_VARIANTS | eps outside/inside relevant expression, weight versus 1+weight, multiply-before/after output cast; valid distinctions preserved |
| ATTENTION_TINY | B=1, Hq=4, Hkv=2, D=8, capacity=17, positions 0/1/15/16; GQA and valid-length edges |
| BLOCK_TINY | H=32, Hq=4, Hkv=2, D=8, I=65, two norm sites, residuals and explicit fixed-capacity KV |
| BLOCK_DEVICE | Start H=256, Hq=4, Hkv=2, D=64, I=512 with capacities 128/2048; adjust only as a declared synthetic fixture |
| REPEATED_TINY | Two copies of BLOCK_TINY with distinct weights/state; identical structure is not identical data |
| STATE_POISON | Unused cache/scratch filled with sentinels; cannot affect reads outside valid regions |
| SCHEDULE_NEGATIVE | Missing producer, duplicate K chunk, worker-order cycle, slot-credit cycle, early store publication and early scratch reuse |

CPU semantic tests can use FP32 and explicit casts; GPU tests must exercise each advertised FP16/BF16
policy. An unsupported CPU dtype operation is reported, not replaced with another reference without
recording that limitation. Synthetic fixtures never count as real-model speedup evidence.

### 3.5 Test and artifact conventions

Implementation cards name test files under CPU, CTEST or GTEST; inventory/audit cards instead name
their required evidence. After the implementation task creates its test:

```bash
python -m pytest -q tests/test_v3/cpu/test_<name>.py
python -m pytest -q tests/test_v3/toolchain/test_<name>.py
python -m pytest -q tests/test_v3/gpu/test_<name>.py
```

Use the literal filename from the card in place of the placeholder; do not run a nonexistent
placeholder command and report success. MB3-003 establishes skip/marker policy and commands.
A skipped GPU suite is not a passed GPU gate. Ordinary CPU tests must not be skipped wholesale
because CUDA is absent.

Proposed artifacts: `ART/tasks/MB3-NNN/handoff.json` plus small human-readable notes; benchmark
runs under `ART/runs/<run_id>/` with graph/plan/source hashes, environment, correctness, raw timings,
trace/resource evidence and selection/final-validation distinction. Large weights/binaries/traces
are not committed by default. A handoff records their durable location and hash; temporary paths
alone are insufficient for a final result.

## 4. Execution order and early useful results

Do not wait to implement the entire handbook before taking the first measurement. Use dependencies
to schedule these useful vertical slices:

| Slice | Exit, not just files written |
|---|---|
| CPU foundation | Imports work without headers; fixtures/specs and reference capture tests pass |
| Semantic slice | Tiny linear/gating/state graphs round-trip through normalized FX and FatOps |
| Offline plan slice | A tiny staged plan has exact dependencies, storage and a checked progress argument |
| Math probe | Same exact-shape body runs standalone and in a lean cooperative entry; no full model required |
| Pipeline probe | Real matrix-data lookahead and continued K updates have device litmus tests and controls |
| Generated block | One graph-driven block runs barrier and pipelined variants with state correctness |
| Model result | Small and roughly 2B models, matched baselines, full state/output contracts and mechanism evidence |

When GPU access returns early, prioritize MB3-042, MB3-046–049 and MB3-077–081 along their **actual
dependencies**, while CPU frontend work continues separately. Do not make a complete LayerSummary
or every matcher a prerequisite for LINEAR_TINY. Conversely, a handwired math probe is not a
completed generic FX compiler milestone.

The mandatory first-block mechanisms are ready-head attention, streamed gated MLP and cross-task
matrix-data lookahead. Each gets a correct experiment. A final cell may reject an unprofitable
mechanism, but it may not claim that launch-only composition tested these mechanisms.

## 5. Task card notation and expected effects

Every card specifies dependencies, lane/review, owned edit surface, before/after, procedure,
validation and expected effects. “Read” refers to the linked V3 document sections by number.
The short names architecture, IR, pipeline, performance, hardware, kernel reuse, GPU audit, dataflow
and research refer respectively to the matching documents in the [series index](MEGABAKE_V3_README.md).

Expected effects use **S** (semantics/correctness), **C** (clarity/diagnosability) and **P**
(performance). For plumbing and verifier tasks, P is usually “no direct GPU gain.” For an
optimization, P names the mechanism and the experiment that can reject it, not a fabricated
percentage. Removing materializations is a structural expectation; physical DRAM-byte and latency
improvements require measurement.

“Accept” includes the shared rules in §1.5 and a completed task handoff. For R3 work the handoff
must include the review status and all unrun device gates. No card permits loosening the fixed
numerical policy to manufacture a speedup.

## 6. Atomic tasks

Use this index to open a bounded task family; each card has a stable task-ID anchor for handoffs.

| Task IDs | Responsibility |
|---|---|
| [001–006](#mb3-001) | Environment, safe imports, test lanes, contracts and diagnostics |
| [007–012](#mb3-007) | Capture, normalization, tensor/effect facts and semantic dialect |
| [013–021](#mb3-013) | Matchers, state, composites, layer summary and exact-shape inventory |
| [022–033](#mb3-022) | Plan/body schemas, footprints, storage, legality and simulation |
| [034–040](#mb3-034) | Pipeline templates, tails, cost model and joint search |
| [041–048](#mb3-041) | Target, source generation, compiler, driver, admission and probe harness |
| [049–057](#mb3-049) | Math/vector/state/attention bodies and one library adaptation |
| [058–069](#mb3-058) | Staging, continuations, publication, generated pipelines and composition |
| [070–076](#mb3-070) | Session, invocation, fallback and graph-driven model integration |
| [077–084](#mb3-077) | Timing, baselines, calibration, ablations and real-model evidence |
| [085–088](#mb3-085) | Optional compile adapter, packaging, regression lanes and final audit |
| [089–092](#mb3-089) | Measurement-triggered extensions, not automatic scope |

<a id="mb3-001"></a>

### MB3-001 — Record the execution environment and legacy baseline status

**Depends:** none. **Lane/review:** CPU, R1.
**Own:** `ART/tasks/MB3-001/` notes; no compiler edits.
**Read:** GPU audit §§2–6; research ledger §§2,4; pyproject.toml, requirements.txt and current git status.

**Before → after:** later agents guess installed versions and existing failures → a dated,
reproducible starting-state record separates present environment from historical GPU reports.

**Do:** record commit/worktree status and Python/package versions without importing MegaBake's
runtime; inspect available test commands and CUDA/toolkit availability without installing anything.
Run the existing CPU-safe tests if their environment permits; retain collection errors. Record
whether original Gemma checkpoint identity/raw GPU artifacts are available or still unresolved.
Never transcribe historical version strings as locally observed versions.

**Validate/accept:** handoff lists actual commands, errors, skipped hardware checks and existing
user modifications; no performance claim. A missing dependency is explicit, not a successful test.
**Expected:** S: preserves comparison provenance; C: one baseline inventory; P: no direct gain.

<a id="mb3-002"></a>

### MB3-002 — Remove eager optional-header resolution from Python import

**Depends:** MB3-001. **Lane/review:** CPU, R1.
**Own:** existing `runtime/cuda_compiler.py`; `CPU/test_imports.py`.
**Read:** current import chain from public __init__.py through loader/launcher to cuda_compiler.py.

**Before → after:** importing the package can require CUTLASS headers → only an actual CUDA
compilation request resolves optional headers; normal CPU/reference imports work without them.

**Do:** move header resolution behind the compile boundary and preserve explicit overrides and
the existing actionable error. Keep driver initialization lazy. Do not remove CUTLASS from an
actual legacy compile or change its flags/selected sources in this task. Add a subprocess test
with controlled header discovery and a test proving compilation still attempts resolution.

**Validate/accept:** `CPU/test_imports.py` collects/runs without CUDA headers; existing pure shape
and serialization imports remain usable. Missing headers fail only on the relevant compile path.
**Expected:** S: no graph change; C: clean dependency boundary; P: no GPU gain, lower import coupling.

<a id="mb3-003"></a>

### MB3-003 — Establish separate CPU, toolchain and GPU test lanes

**Depends:** MB3-002. **Lane/review:** CPU, R1.
**Own:** `tests/test_v3/conftest.py`, `CPU/test_lanes.py`, test package setup, pytest markers.
**Read:** existing test skip conventions and §3.5 above.

**Before → after:** CUDA-dependent collection and skips blur coverage → CPU tests always collect;
CT/GPU tests declare their precise prerequisites and visibly skip when unavailable.

**Do:** register markers `v3_gpu`, `v3_toolchain`, `v3_slow`; provide lazy fixtures for a selected
device/toolchain and deterministic seeds. Do not apply a directory-wide CUDA skip to CPU tests.
Add a deliberately failing/skip-report self-test using pytest's available testing facilities or
a subprocess; never add dependencies merely for that convenience.

**Validate/accept:** CPU lane works with device discovery mocked to raise; CT/GPU collection does
not load weights or initialize the driver; handoff records lane commands and skip counts.
**Expected:** S: honest coverage; C: unambiguous gates; P: no direct GPU gain.

<a id="mb3-004"></a>

### MB3-004 — Freeze workload and numerical-policy records

**Depends:** MB3-003. **Lane/review:** CPU, R2.
**Own:** `V3/contracts.py`, `CPU/test_contracts.py`.
**Read:** architecture §§1,8–10; IR §§3–5,9; performance §§7–9.

**Before → after:** dtype/timing/state assumptions are implicit → immutable validated WorkloadSpec
and NumericalPolicy accompany every compilation and benchmark.

**Do:** implement §3.2 fields, canonical serialization and schema version. Separate accumulation,
output casts, permitted reassociation, exceptional-value behavior and tolerance by operation/dtype.
Require explicit context versus capacity, cache layout, timed unit, input origin and output ownership.
Pre-register benchmark cells and which must win before selection; unresolved checkpoint identity
stays unresolved. Begin with inference, fixed buckets and no dropout.

**Validate/accept:** `CPU/test_contracts.py` rejects negative capacities, invalid positions/shape
contracts and omitted policy fields; changing a cast/timing contract changes the contract hash.
Tolerance values need an explicit documented reference-based choice before device acceptance.
**Expected:** S: prevents silent semantic drift; C: shared vocabulary; P: no direct gain.

<a id="mb3-005"></a>

### MB3-005 — Build seeded reference fixtures and comparison helpers

**Depends:** MB3-004. **Lane/review:** CPU, R2.
**Own:** `tests/test_v3/fixtures.py`, `CPU/test_reference_fixtures.py`.
**Read:** §3.4; IR §4; kernel reuse §9.

**Before → after:** individual tests invent different inputs → reusable small graphs with known
weights, exact expression/cast boundaries and structured output/state comparisons.

**Do:** create LINEAR_TINY, GATE_TINY and NORM_VARIANTS using ordinary torch operations, no V3
codegen. Preserve separate SiLU/GELU variants and a multiple-live-output graph. Add finite,
zero, cancellation and policy-relevant extreme inputs. Comparisons separately check structure,
exact integer/index values, state regions, numerical tolerances and exceptional values.

**Validate/accept:** `CPU/test_reference_fixtures.py` proves seeds replay, altered state is caught,
swapped gate inputs fail, and a blanket cosine/max-error print cannot pass a wrong result.
**Expected:** S: executable oracle; C: consistent failures; P: no direct gain.

<a id="mb3-006"></a>

### MB3-006 — Define diagnostics and evidence handoff serialization

**Depends:** MB3-004. **Lane/review:** CPU, R1.
**Own:** `V3/diagnostics.py`, `CPU/test_diagnostics.py`.
**Read:** §1.3, §3.2, performance §8.

**Before → after:** free-text success/failure loses provenance → stable reason codes and structured
task/compile/benchmark evidence distinguish unsupported, incorrect, unmeasured and measured loss.

**Do:** implement TaskHandoff and diagnostic records with node/action IDs and artifact hashes.
Use JSON-safe enums and schema versions; never pickle arbitrary executable objects. Define
codes for unsupported semantics, missing facts, body incompatibility, invalid event/storage/progress,
missing toolchain/device, and failed numerical gate. Add a human-readable rendering.

**Validate/accept:** `CPU/test_diagnostics.py` round-trips records; missing GPU evidence cannot
serialize as a strict win; corrupted schema/version yields an actionable error.
**Expected:** S: prevents false success; C: actionable handoffs; P: no direct gain.

<a id="mb3-007"></a>

### MB3-007 — Preserve ExportedProgram signatures and lifted bindings

**Depends:** MB3-005, MB3-006. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/capture.py`, `CPU/test_export_capture.py`.
**Read:** IR §§2–3; graph_walker.py parameter-name handling.

**Before → after:** capture is treated as just a graph → NormalizedProgram retains parameter,
buffer, user-input, mutation, constraint and output-tree meanings.

**Do:** accept an ExportedProgram without recapturing it; map lifted placeholders through its
graph signature and preserve constants/nonpersistent buffers. Retain a reference callable and
input/output pytree specification. Give values deterministic IDs. Keep aliases and mutation
outputs visible. Do not convert all tensors to FP16 or assume the only output is logits.

**Validate/accept:** `CPU/test_export_capture.py`: linear parameters, registered buffers, nested
outputs, tied weights and mutation-signature fixtures bind correctly; missing bindings fail by name.
Executing the preserved reference uses the same inputs/weights as the original.
**Expected:** S: correct ABI/state capture; C: one binding map; P: no direct gain.

<a id="mb3-008"></a>

### MB3-008 — Add the bare GraphModule capture adapter

**Depends:** MB3-007. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/capture.py`, `CPU/test_graphmodule_capture.py`.
**Read:** IR §2; public compile_fx implementation.

**Before → after:** GraphModule capture is coupled to CUDA shape discovery → an explicit example
signature/reference path accepts supported CPU graphs without querying a device.

**Do:** require example_args/input_spec, normalize argument trees without flattening away meaning,
and capture/functionalize through the selected PyTorch API. Do not execute unknown side-effecting
callbacks merely to obtain metadata. Reject unsupported custom effects/control flow explicitly.
Reuse MB3-007's representation instead of creating a second capture format.

**Validate/accept:** `CPU/test_graphmodule_capture.py`: missing examples fail; simple multi-input
and nested-output graphs agree; device discovery is never called; caller graph/state is unchanged.
**Expected:** S: explicit capture limits; C: one frontend contract; P: no direct gain.

<a id="mb3-009"></a>

### MB3-009 — Implement the pinned normalization adapter

**Depends:** MB3-007, MB3-008. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/normalize.py`, `CPU/test_normalize.py`.
**Read:** IR §2 and current optimize_graph; inspect the actual selected PyTorch pass source.

**Before → after:** a private pass is called without a tested compatibility seam → an allowlisted,
version-checked transformation sequence preserves the export/reference contract.

**Do:** copy the decomposition table before modifying it; preserve useful attention/activation
operations initially. Call selected passes through one adapter, correctly handling both in-place
and replacement returns. Propagate graph signature, constraints and metadata. Unsupported PyTorch
versions receive an explicit compatibility error or a named minimal-normalization path, never a
pretend full optimization. Do not monkeypatch global Inductor state.

**Validate/accept:** `CPU/test_normalize.py`: original/normalized outputs agree; both pass return
styles work in mocks; two sequential compile requests do not contaminate each other; a normal
torch.compile reference is not modified by the adapter. Test the actual installed supported pin.
**Expected:** S: preserved semantics; C: bounded private-API dependency; P: no promised direct gain.

<a id="mb3-010"></a>

### MB3-010 — Collect tensor, view, alias and specialization facts

**Depends:** MB3-009. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/facts.py`, `CPU/test_facts.py`.
**Read:** IR §3; shape_ops.py and its existing tests.

**Before → after:** dimensions/strides are reconstructed ad hoc → every relevant value has proven
or unknown TensorFacts and an explicit specialization guard.

**Do:** track dtype bytes, shape, strides, storage offset, alignment if proven, alias set and
mutability. Reuse tested view rules; handle transpose, slice, squeeze and zero-stride expansion.
A noncontiguous reshape either remains an indexed view when proven or requires explicit copying.
Preserve symbolic constraints; specialize fixed shapes without claiming all symbolic cases work.
Unknown overlap/alignment cannot be used as proof of disjointness/alignment.

**Validate/accept:** `CPU/test_facts.py`: view chains match torch indexing; unknown dtype fails;
zero-stride writes and unsafe aliases are rejected; changing a shape/stride violates its guard.
**Expected:** S: indexing/alias safety; C: no magic defaults; P: enables copy elimination, no measured gain yet.

<a id="mb3-011"></a>

### MB3-011 — Make dead-code elimination respect state and effects

**Depends:** MB3-010. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/effects.py`, `CPU/test_effect_dce.py`.
**Read:** IR §§3,8; export mutation signatures from MB3-007.

**Before → after:** liveness may follow only visible tensor outputs → required state transitions
and externally observable effects remain roots of the normalized graph.

**Do:** classify pure supported operations and explicit state effects. Mark returned values,
required mutations and designated session state updates live; walk dependencies backward.
Delete only unreachable pure nodes. Preserve old/new state identities and order conflicting effects.
Do not treat arbitrary Python/custom operations as pure by default.

**Validate/accept:** `CPU/test_effect_dce.py`: dead arithmetic disappears; an ignored-but-required
cache update stays; deleting it changes the state comparison; unsafe alias/effect reorder fails.
**Expected:** S: correct state cleanup; C: explainable live roots; P: removes truly dead work only.

<a id="mb3-012"></a>

### MB3-012 — Define the semantic dialect registry and reference expansion

**Depends:** MB3-011. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/semantic.py`, `CPU/test_semantic_registry.py`.
**Read:** IR §4; architecture §§2,4.

**Before → after:** operations are old integer task codes → SemanticGraph is still FX, with
versioned semantic operations, facts, origin regions and executable reference expansions.

**Do:** define registration/support/expand interfaces, metadata rules and effect declarations.
Retain ordinary ATen reference regions for unmatched nodes. Register a minimal Linear definition
to test the protocol; separate the definition from device candidates. Validate expansion inputs,
outputs, live boundaries and semantic version. Avoid opaque model-layer nodes.

**Validate/accept:** `CPU/test_semantic_registry.py`: expansion round-trips a simple region,
unknown operations remain visible, duplicate/conflicting registrations fail, and no CUDA import occurs.
**Expected:** S: stable reference meaning; C: separates semantics from algorithms; P: no direct gain.

<a id="mb3-013"></a>

### MB3-013 — Recognize supported Linear forms without losing scales or views

**Depends:** MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_linear.py`, `CPU/test_match_linear.py`.
**Read:** IR §4; current _extract_dimensions and ATen op table.

**Before → after:** mm/addmm/linear handling relies on positional guesses → supported forms map
to a guarded Linear with complete transpose, bias, scale and cast behavior.

**Do:** match linear, supported mm and addmm forms using schema arguments and metadata. Preserve
addmm alpha/beta or reject unsupported values; do not silently assume one. Record arbitrary live
consumers and weight layout. Initially reject unsupported batched/broadcasted matmul forms with
the original reference region intact.

**Validate/accept:** `CPU/test_match_linear.py`: NK/KN views, bias, non-unit alpha/beta, odd shapes,
multiple outputs and mismatching ranks either expand equivalently or are explicitly unmatched.
**Expected:** S: exact linear semantics; C: explicit support boundary; P: enables good mapping, no gain claimed.

<a id="mb3-014"></a>

### MB3-014 — Recognize pointwise expressions and exact gating

**Depends:** MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_pointwise.py`, `CPU/test_match_pointwise.py`.
**Read:** IR §4; GATE_TINY and NumericalPolicy.

**Before → after:** activation names/old uop flags hide intermediate behavior → a small supported
expression DAG preserves operand order, broadcasts and cast boundaries.

**Do:** support the first fixtures' arithmetic, SiLU and declared GELU approximation, with constants
and casts as explicit nodes. Recognize SwiGLU only for its actual SiLU-gating definition. Keep a
GELU gate as Pointwise. Retain extra consumers rather than consuming an entire region greedily.

**Validate/accept:** `CPU/test_match_pointwise.py`: SiLU versus GELU, swapped gate inputs, broadcast,
FP32 intermediate and FP16/BF16 cast positions remain distinguishable and reference-equivalent.
**Expected:** S: no activation substitution; C: inspectable expression; P: enables local fusion later.

<a id="mb3-015"></a>

### MB3-015 — Recognize RMSNorm with guarded numerical variants

**Depends:** MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_norm.py`, `CPU/test_match_norm.py`.
**Read:** IR §4 and NORM_VARIANTS.

**Before → after:** similar-looking reductions risk one norm label → only semantically identified
RMSNorm regions match, with epsilon, axes, weight transform and cast order retained.

**Do:** match the explicit square/mean/rsqrt/multiply chain or preserved norm op where supported.
Require the correct reduction axes and epsilon placement. Store whether the scale is weight or
1+weight and where output rounding occurs. Reject layer norm, altered variance expressions and
unproven rewrites instead of approximating them.

**Validate/accept:** `CPU/test_match_norm.py`: all supported variants expand correctly; deliberately
changed axes/epsilon placement fail recognition; extra live reduction outputs remain available.
**Expected:** S: protects Gemma-like distinctions; C: named variants; P: enables lean reduction/fusion later.

<a id="mb3-016"></a>

### MB3-016 — Recognize RoPE without guessing conventions

**Depends:** MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_rope.py`, `CPU/test_match_rope.py`.
**Read:** IR §4; architecture §4.

**Before → after:** a model/config name may imply a positional kernel → recognized graph structure
supplies pairing, rotary dimension, positions, frequency/scaling and cast rules.

**Do:** begin with explicit half-split and interleaved pairing fixtures; preserve nonrotary channels.
Retain supplied cos/sin or their supported derivation. Reject an unsupported scaling rule rather
than substituting default RoPE. Ensure Q and K layouts are independently checked.

**Validate/accept:** `CPU/test_match_rope.py`: both pairings, partial rotary dimension, positions
near bucket boundaries and swapped sine signs are distinguished; reference expansion agrees.
**Expected:** S: positional correctness; C: no model-name inference; P: enables cache/epilogue fusion.

<a id="mb3-017"></a>

### MB3-017 — Preserve attention semantics in a guarded SDPA matcher

**Depends:** MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_attention.py`, `CPU/test_match_attention.py`.
**Read:** IR §5; kernel reuse §9; pipeline §4.

**Before → after:** attention is recognized by broad pattern/name → explicit Q/K/V head mapping,
scale, mask, causal alignment, valid lengths, dtype and output policy define each SDPA region.

**Do:** first support a preserved SDPA op and the exact decomposed reference forms needed by
fixtures. Keep boolean/additive mask meanings and softmax casts distinct. Restrict dropout to zero.
Never lower a recurrent/linear-attention label into SDPA. Avoid applying a prefill triangular
mask to decode merely because the query length is one.

**Validate/accept:** `CPU/test_match_attention.py`: MHA/GQA, non-square causal alignment, masked
rows and custom scale either match exactly or remain explicit reference regions.
**Expected:** S: correct attention identity; C: algorithm remains a later choice; P: no direct gain.

<a id="mb3-018"></a>

### MB3-018 — Define functional fixed-capacity cache updates

**Depends:** MB3-011, MB3-012. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/match_state.py`, fixture additions, `CPU/test_cache_semantics.py`.
**Read:** IR §8; pipeline §§4,9.

**Before → after:** cached state is absent/implicit → a pure reference transition exposes old/new
K/V, position, valid length and required state effects.

**Do:** implement ATTENTION_TINY and STATE_POISON with an explicit contiguous fixed-capacity cache.
Use supported functional indexing/scatter/update expressions, not data-dependent .item() branches
inside capture. Define position p as the slot written and valid length after update as p+1 for
the initial append-only contract. Reject overflow and unsupported overwrite policies. Preserve
the old state in the reference; in-place selection belongs to the storage/session tasks.

**Validate/accept:** `CPU/test_cache_semantics.py`: positions 0/1/15/16, untouched regions, original
state preservation, malformed GQA/state shapes and overflow. DCE retains the required update.
**Expected:** S: real state semantics; C: explicit capacity/position contract; P: enables cached decode, not a speed claim.

<a id="mb3-019"></a>

### MB3-019 — Enumerate overlapping composite candidates without committing a cover

**Depends:** MB3-013, MB3-014, MB3-015, MB3-016, MB3-018. **Lane/review:** CPU, R2.
**Own:** `V3/frontend/composites.py`, `CPU/test_composites.py`.
**Read:** architecture §§4,6; IR §4.

**Before → after:** early fusion can erase useful alternatives → bounded composite records coexist
with unfused semantics until body/tile/schedule selection.

**Do:** enumerate linear epilogues, paired gate/up plus exact gating, norm-linear, QKV grouping and
RoPE/cache combinations when graph facts permit. Each record names covered nodes, live boundaries,
effects and reference expansion. Do not remove matched nodes yet. Track pure recomputation as an
explicit alternative with multiplicity, never duplicate a state effect.

**Validate/accept:** `CPU/test_composites.py`: overlapping matches survive; extra consumers are
retained; final-cover validation can reject double-covered effects; unfused candidates always remain.
**Expected:** S: preserves live behavior; C: inspectable alternatives; P: exposes fusion/readiness tradeoffs.

<a id="mb3-020"></a>

### MB3-020 — Add LayerSummary structural fingerprints

**Depends:** MB3-012. **Lane/review:** CPU, R1.
**Own:** `V3/frontend/layers.py`, `CPU/test_layer_summary.py`.
**Read:** IR §6; dataflow §1.

**Before → after:** repeated layers are rediscovered or identified by name → a side analysis groups
verified repeated structures while keeping distinct parameter/state bindings.

**Do:** fingerprint semantic attributes, connectivity, shape/layout guards and numerical policy,
excluding runtime addresses and model naming. Preserve layer order and boundary values. Include
mask/norm differences; do not assume a config layer list proves equivalent subgraphs.

**Validate/accept:** `CPU/test_layer_summary.py`: renamed equal structures group; distinct weights
stay distinct bindings; a changed norm/mask breaks the structural match. No graph becomes opaque.
**Expected:** S: no rewrite; C: reusable structure report; P: enables tuning/code reuse, not weight reuse.

<a id="mb3-021"></a>

### MB3-021 — Emit the normalized graph and exact-shape inventory

**Depends:** MB3-010, MB3-012, MB3-013, MB3-017, MB3-018.
**Lane/review:** CPU, R1. **Own:** `V3/frontend/inventory.py`, `CPU/test_inventory.py`.
**Read:** GPU audit §§4,7; implementation §3.4.

**Before → after:** hot shapes are historical guesses → every captured workload has a reproducible
shape/layout/operation/state inventory and unsupported-region report.

**Do:** emit deterministic JSON plus concise text: M/N/K, multiplicity, dtype, strides, epilogue,
weight identity/ties, context/head/cache dimensions, live values and unsupported nodes. Separate
semantic accessed bytes from parameter storage and from unknown physical DRAM bytes. No CUDA query.

**Validate/accept:** `CPU/test_inventory.py`: LINEAR_TINY reports 1/17/33; repeated tied weights
are not falsely independent parameter storage; unknown costs remain unknown; report IDs map to FX.
**Expected:** S: coverage visibility; C: exact tuning inputs; P: guides experiments, no direct gain.

<a id="mb3-022"></a>

### MB3-022 — Implement the ExecutionPlan data model and deterministic serialization

**Depends:** MB3-004, MB3-006. **Lane/review:** CPU, R2.
**Own:** `V3/plan/model.py`, `CPU/test_plan_model.py`.
**Read:** IR §7; pipeline §2; shared interfaces §3.

**Before → after:** V3 plans exist only in prose → one typed model stores candidate/final plans,
actions, footprints, reductions, buffers, events, worker programs and unresolved decisions.

**Do:** implement the logical schema without a task interpreter or new graph library. Support
small explicit domains first and compact repeated domains where needed. Separate immutable
semantic IDs from physical worker assignments. A finalize operation rejects unresolved guards,
body choices or required synchronization, but permits unknown performance costs marked as such.

**Validate/accept:** `CPU/test_plan_model.py`: round-trip equality, stable ordering/hash, invalid
references and unknown schema rejection; changing worker count does not change logical work coverage.
**Expected:** S: explicit plan meaning; C: one representation; P: no direct gain.

<a id="mb3-023"></a>

### MB3-023 — Implement device-body capability and support registration

**Depends:** MB3-010, MB3-022. **Lane/review:** CPU, R2.
**Own:** `V3/plan/bodies.py`, `CPU/test_body_registry.py`.
**Read:** kernel reuse §7; pipeline §2.

**Before → after:** a callable body is assumed composable → support is a checked semantic,
shape/layout, target, thread-role, descriptor and stage-capability contract.

**Do:** implement BodySpec/support results with precise rejection reasons. Distinguish ATOMIC_TILE,
PRELOADABLE, STREAM_REDUCTION and optional EARLY_RELEASE. Register test bodies without claiming
their CUDA implementation exists. Include required source/header versions, explicit role masks,
scratch/alignment and accumulator lifetime. No external/child-grid body is admitted as strict.

**Validate/accept:** `CPU/test_body_registry.py`: missing stages, incompatible block participation,
unsupported dtype/alignment and host-only operations reject; costs cannot override legality.
**Expected:** S: honest compatibility; C: visible missing interfaces; P: enables joint body selection.

<a id="mb3-024"></a>

### MB3-024 — Derive linear/vector tile footprints independently of workers

**Depends:** MB3-022, MB3-023. **Lane/review:** CPU, R2.
**Own:** `V3/plan/footprints.py`, `CPU/test_linear_footprints.py`.
**Read:** IR §7; pipeline §§3,5; existing tiling.py only as a contrast.

**Before → after:** tile count follows SM count → output/reduction domains define logical work;
workers are assigned afterward.

**Do:** generate half-open output tiles for vector operations and Linear. Full linear outputs
read their entire K domain; continued updates read exactly their declared K chunk. Map footprints
through supported strides/views, retaining conservative alias information. Tail predicates cover
only valid elements; padding is an explicit initialized allocation when a body needs it.

**Validate/accept:** `CPU/test_linear_footprints.py`: enumerate LINEAR_TINY indices against a
reference, check exact once-only output/K coverage, compare plans with 1/2/7 workers and test K tails.
**Expected:** S: correct work partition; C: tiling/residency separation; P: enables useful-lane improvements.

<a id="mb3-025"></a>

### MB3-025 — Derive head, GQA and cache-update footprints

**Depends:** MB3-017, MB3-018, MB3-024. **Lane/review:** CPU, R2.
**Own:** `V3/plan/footprints.py`, `CPU/test_attention_footprints.py`.
**Read:** pipeline §4; IR §5.

**Before → after:** consumers depend on an entire QKV operator → the exact Q/K/V/cache regions
needed by a head determine its producers.

**Do:** map each query head to its actual KV head; include all head-dimension tiles and valid
cache positions. Distinguish current-slot writes from old initialized cache. A readiness group may
combine heads conservatively but may not omit their producers. Include RoPE and length/mask
dependencies. Output projection retains full input coverage unless explicitly split.

**Validate/accept:** `CPU/test_attention_footprints.py`: ATTENTION_TINY q0/q1 map to kv0 and q2/q3
to kv1; missing V/current-slot publication fails; poison beyond valid length is never in a read set.
**Expected:** S: exact readiness; C: explainable fan-in; P: enables head overlap, no speedup yet.

<a id="mb3-026"></a>

### MB3-026 — Verify semantic coverage, writers and reduction finalization

**Depends:** MB3-011, MB3-022, MB3-023, MB3-024. **Lane/review:** CPU, R2.
**Own:** `V3/plan/verify.py`, `CPU/test_plan_coverage.py`.
**Read:** IR §§7–9; pipeline §5.

**Before → after:** a plan can look complete while dropping work → explicit checks prove each
required semantic region/effect and each output/reduction contribution is represented.

**Do:** validate body support, graph boundary mapping, unique writers or declared reducers, exact
K coverage and finalizer order. Reject duplicate state effects, missing live outputs, overlaps
without a reduction protocol and finalization before the last required update. Numerical-policy
compatibility is a required body guard, not just a runtime comparison.

**Validate/accept:** `CPU/test_plan_coverage.py`: mutate valid plans to omit/duplicate a K chunk,
drop a residual/state output or round a continuation early; each yields an identifying diagnostic.
**Expected:** S: prevents incomplete megakernels; C: localized counterexamples; P: no direct gain.

<a id="mb3-027"></a>

### MB3-027 — Build exact producer sets and typed readiness events

**Depends:** MB3-024, MB3-025, MB3-026. **Lane/review:** CPU, R2.
**Own:** `V3/plan/dependencies.py`, `CPU/test_event_dependencies.py`.
**Read:** pipeline §§2,9; IR §7.

**Before → after:** a buffer name implies one producer → region intersections yield exact unique
producer contributions and typed prerequisites.

**Do:** intersect consumer footprints with producer writes, including view/alias mapping and
required effects. Distinguish input-ready facts from produced events with positive expected counts.
Deduplicate contribution IDs; aggregate events only for identical complete dependency sets.
Address-ready weights can be independent of activation-ready tokens. No within-invocation reuse
of inter-CTA event IDs in this first scheme.

**Validate/accept:** `CPU/test_event_dependencies.py`: multi-tile heads wait for all contributions;
duplicate counting, zero-ready produced data and source-retired-as-data-ready are rejected.
**Expected:** S: publication prerequisites; C: readable dependency sets; P: removes unnecessary whole-op waits later.

<a id="mb3-028"></a>

### MB3-028 — Generate barrier-control worker programs

**Depends:** MB3-024, MB3-026. **Lane/review:** CPU, R2.
**Own:** `V3/plan/templates.py`, `CPU/test_barrier_plan.py`.
**Read:** architecture §6; dataflow §4.

**Before → after:** the only executable control is the legacy task scheduler → V3 has a simple
same-body cooperative phase plan for comparison and correctness debugging.

**Do:** assign logical tiles over an explicit worker count; use balanced strip mining or a declared
costed static assignment. Every worker reaches the same ordered joins, even with no arithmetic.
Keep body calls atomic here. Do not force a poor producer/consumer cohort split into this control
to inflate later overlap gains.

**Validate/accept:** `CPU/test_barrier_plan.py`: zero-work workers, fewer/more tiles than workers,
two dependent regions and complete output coverage; joins have uniform participation.
**Expected:** S: reference schedule; C: simple comparison; P: control only, no pipeline-success claim.

<a id="mb3-029"></a>

### MB3-029 — Allocate buffers from partial-order access lifetimes

**Depends:** MB3-022, MB3-026, MB3-028. **Lane/review:** CPU, R3.
**Own:** `V3/plan/storage.py`, `CPU/test_storage_lifetimes.py`.
**Read:** IR §8; pipeline §8; old sequential buffer_planner.py.

**Before → after:** an estimated/topological interval can justify unsafe overlay → reuse requires
a proven release-before-next-use order for all relevant accesses.

**Do:** build lifetime/conflict records from actions, ownership and async source retirement.
Treat incomparable lifetimes as overlapping. Allocate aligned fixed slots by deterministic first-fit
over a conflict graph; do not add serializing edges silently just to fit memory. Separate global
activation arena, per-CTA shared staging and accumulator storage. Preserve output/state lifetimes.
For inter-CTA chunks, use unique region addresses and reuse only after the verified region join.

**Validate/accept:** `CPU/test_storage_lifetimes.py`: simultaneous current/next tiles do not overlay;
slow-store source stays live; two-reader buffers outlive both readers; address bounds/alignment hold.
**Expected:** S: safe reuse; C: explicit memory budget; P: lowers footprint where ordering proves it, otherwise may grow safely.

<a id="mb3-030"></a>

### MB3-030 — Verify event, async and slot lifecycle rules

**Depends:** MB3-027, MB3-029. **Lane/review:** CPU, R3.
**Own:** `V3/plan/verify.py`, `CPU/test_protocol_lifetimes.py`.
**Read:** pipeline §§2,8–9.

**Before → after:** event IDs and scratch ranges are structurally present → the plan checks their
initialization, publication, acquisition, retirement and reuse ordering.

**Do:** require same-grid initialization before use, positive produced-event counts and exact
producer multiplicity. Check destination visibility precedes publish, acquire precedes consumer
reads, and source retirement precedes slot overwrite. For a bounded per-CTA ring, verify each
iteration's phase and release-before-reacquire; reject unknown unbounded cycles. Count bounds must
fit the selected integer representation without wraparound.

**Validate/accept:** `CPU/test_protocol_lifetimes.py`: early publication, lost acquire, stale phase,
double reserve, counter overflow and overwrite-before-retirement all fail with action IDs.
**Expected:** S: protocol legality; C: explicit event/slot bugs; P: no direct gain.

<a id="mb3-031"></a>

### MB3-031 — Verify worker order, collective participation and progress

**Depends:** MB3-028, MB3-029, MB3-030. **Lane/review:** CPU, R3.
**Own:** `V3/plan/verify.py`, `CPU/test_progress.py`.
**Read:** pipeline §10; hardware §4.

**Before → after:** an acyclic tensor graph is assumed safe → generated worker/resource waits
must also satisfy a bounded progress argument and collective participation rules.

**Do:** add worker instruction-order, producer readiness and explicit slot-credit/release edges
to the wait-for/order model. Reject a cycle with a concrete cycle witness. For supported templates,
prove blocked consumers leave prerequisites runnable and do not retain required producer slots.
Reject body-internal hidden grid collectives. Record cooperative residency as a runtime proof
obligation; a CPU profile stub cannot discharge it.

**Validate/accept:** `CPU/test_progress.py`: acyclic-data/cyclic-worker example, slot deadlock,
idle-worker early return and nonuniform collective all reject; legal barrier/pipeline toy plans pass.
**Expected:** S: prevents known deadlock classes; C: wait-cycle diagnostics; P: no direct gain.

<a id="mb3-032"></a>

### MB3-032 — Execute tiny plans with a deterministic CPU reference interpreter

**Depends:** MB3-026, MB3-030. **Lane/review:** CPU, R2.
**Own:** `V3/plan/simulate.py`, `CPU/test_plan_reference.py`.
**Read:** IR §9; pipeline §11.

**Before → after:** plan tests inspect only metadata → tiny action plans can produce outputs/state
using reference tile functions and explicit token/storage transitions.

**Do:** interpret supported actions and reference bodies on small tensors. Track accumulator
continuations, validity and poisoned storage. Use logical completion order, not GPU latency
predictions. A step cannot consume an absent token. Compare completed results with SemanticGraph
expansion, including live residuals and state. Keep this debug interpreter out of GPU execution.

**Validate/accept:** `CPU/test_plan_reference.py`: valid linear/gating/cache plans agree; a modified
footprint or early finalizer fails; attempts to read poison before publication are identified.
**Expected:** S: differential plan oracle; C: reproducible debugging; P: none; interpreter timings are not GPU estimates.

<a id="mb3-033"></a>

### MB3-033 — Explore bounded interleavings and test verifier counterexamples

**Depends:** MB3-031, MB3-032. **Lane/review:** CPU, R3.
**Own:** `V3/plan/simulate.py`, `CPU/test_interleavings.py`.
**Read:** pipeline §11.

**Before → after:** one favorable action order passes → finite tiny schedules are tested under
many producer/consumer/store delays, including adversarial schedules.

**Do:** enumerate enabled transitions for deliberately small plans, canonicalize equivalent states
and impose an explicit exploration cap. Model data-ready and source-retired completions separately.
Report pass only for the stated explored bounds; reaching the cap is inconclusive, not proof.
Seed randomized larger tests separately. Save minimal failing action sequences.

**Validate/accept:** `CPU/test_interleavings.py`: known unsafe fixtures are found; legal two-slot
plans preserve outputs/storage under all fully explored tiny schedules; cap exhaustion is visible.
**Expected:** S: stronger finite validation; C: replayable races/deadlocks; P: no direct gain or formal CUDA proof.

<a id="mb3-034"></a>

### MB3-034 — Instantiate a head-ready projection/attention plan template

**Depends:** MB3-025, MB3-027, MB3-031, MB3-032.
**Lane/review:** CPU, R3. **Own:** `V3/plan/templates.py`, `CPU/test_head_ready_plan.py`.
**Read:** pipeline §4; dataflow §5.

**Before → after:** attention waits for the full QKV region → a generated candidate starts each
consumer when its exact head/GQA inputs and cache effects are published.

**Do:** partition producer and consumer work from footprint groups, instantiate bounded cohort
programs and retain a barrier-control alternative. Keep old-cache readiness separate from current
writes. Respect RoPE/valid-length inputs. Do not stream the output projection in this task; retain
its full dependency. Run coverage/lifetime/progress checks after assignment.

**Validate/accept:** `CPU/test_head_ready_plan.py`: delay unrelated group 1 and allow group 0
attention; delay group 0 V and prevent it; interpreter output/state equals the unfused reference.
**Expected:** S: same attention semantics; C: visible early edges; P: exposes overlap, CPU timing proves no speedup.

<a id="mb3-035"></a>

### MB3-035 — Instantiate an owner-held streamed gated-MLP plan

**Depends:** MB3-019, MB3-027, MB3-031, MB3-032.
**Lane/review:** CPU, R3. **Own:** `V3/plan/templates.py`, `CPU/test_mlp_stream_plan.py`.
**Read:** pipeline §5; kernel reuse §7.

**Before → after:** down projection starts only after every hidden element exists → ordered K
updates start on complete chunks and finalize only after full reduction coverage.

**Do:** use GATE_TINY's exact SiLU or GELU reference and tail chunk. Give each active output tile
one accumulator owner; bound simultaneous accumulator demand. Emit begin/update/finalize with
required casts, unique h chunk addresses and no early residual/output publication. Retain full
materialization. If owners cannot hold all active tiles, use a declared multipass assignment or
reject this candidate; do not invent free register storage.

**Validate/accept:** `CPU/test_mlp_stream_plan.py`: early first update is legal; missing/duplicated
chunk, tail omission and early cast/finalization reject; complete result matches the policy.
**Expected:** S: complete reduction; C: explicit ownership; P: enables gate/down overlap without mandatory global partials.

<a id="mb3-036"></a>

### MB3-036 — Insert bounded independent weight-preload actions

**Depends:** MB3-023, MB3-029, MB3-030, MB3-031.
**Lane/review:** CPU, R3. **Own:** `V3/plan/templates.py`, `CPU/test_weight_lookahead_plan.py`.
**Read:** pipeline §§6,8.

**Before → after:** next-task readiness blocks even independent movement → a PRELOADABLE body can
reserve storage and issue its weight loads before the next activation exists.

**Do:** instantiate zero/one/two legal lookahead slots from the body contract. Require invariant
weights/known address dependencies, safe descriptor lifetime and release-before-reuse. Include
current internal body stages and output-store storage. Delay compute until both activation and
load-complete tokens exist. Do not issue speculative activation reads. Rerun all affected verifiers.

**Validate/accept:** `CPU/test_weight_lookahead_plan.py`: delayed activation does not block legal
weight loading; missing space/mutable weight/unknown stage support rejects; output semantics unchanged.
**Expected:** S: safe early movement; C: separate address/data readiness; P: hides potential stalls, profitability unknown.

<a id="mb3-037"></a>

### MB3-037 — Enumerate balanced tile/cohort and tail-aware assignments

**Depends:** MB3-028, MB3-034, MB3-035, MB3-036.
**Lane/review:** CPU, R2. **Own:** `V3/plan/templates.py`, `CPU/test_assignments.py`.
**Read:** pipeline §7; performance §6.

**Before → after:** one fixed worker split determines results → a bounded candidate set considers
tile granularity, balanced barrier assignments and legal ready-consumer placements.

**Do:** enumerate a small documented set of producer/consumer ratios and compatible tile sizes.
Generate static mixed worker orders only when progress checks pass. Separate worker IDs from
physical SM identities. Keep final-wave work counts and assigned active tiles in reports.
No universal queue/work stealing or body preemption is introduced.

**Validate/accept:** `CPU/test_assignments.py`: N near P and 2P boundaries, N<P, heterogeneous toy
durations and the performance document's negative overlap example retain a competitive control.
**Expected:** S: coverage preserved; C: assignment choices visible; P: targets tails; no universal benefit.

<a id="mb3-038"></a>

### MB3-038 — Store measured, estimated and unknown costs without conflating them

**Depends:** MB3-004, MB3-006, MB3-023. **Lane/review:** CPU, R2.
**Own:** `V3/plan/costs.py`, `CPU/test_cost_records.py`.
**Read:** hardware §§2,7–8; performance §§2,4–5.

**Before → after:** guessed cycles are treated like comparable latency → unit-aware costs carry
shape/body/entry/cache/target conditions and evidence status.

**Do:** implement CostRecord keys and uncertainty ranges; distinguish body, stage, transition,
joint-pair and invocation measurements. Include numerical policy and source versions. Unknown is
not zero. A changed MIG profile, body mixture, cohort split or staging depth invalidates affected
cost applicability. Store physical traffic separately from semantic bytes.

**Validate/accept:** `CPU/test_cost_records.py`: incompatible units/provenance cannot merge;
unknown costs cannot declare a winner; hot-cache measurements are not silently reused as streaming.
**Expected:** S: honest inference; C: explainable cost provenance; P: enables tuning, no direct gain.

<a id="mb3-039"></a>

### MB3-039 — Implement the bounded resource-constrained schedule estimator

**Depends:** MB3-022, MB3-031, MB3-038. **Lane/review:** CPU, R2.
**Own:** `V3/plan/costs.py`, `CPU/test_schedule_cost.py`.
**Read:** performance §§2,5–6; pipeline §3.

**Before → after:** sums of isolated kernels imply overlap gains → an estimator accounts for
dependencies, worker capacity, staging and declared contention domains.

**Do:** support opaque-tile durations first and explicit stages where costs exist. Enforce finite
workers/slots and conservative shared-bandwidth contention or measured pair adjustments. Do not
allocate full device bandwidth independently to concurrent tasks. Never double-charge an effective
body rate and its already-included stalls. Produce ranges/unknown results rather than fake precision.

**Validate/accept:** `CPU/test_schedule_cost.py`: serial 32 versus ideal staged 23 us toy case,
extra 2 us coordination, positive/negative tail examples, saturated shared-bandwidth case and unknowns.
**Expected:** S: no semantic change; C: falsifiable estimates; P: better ranking, not measured performance.

<a id="mb3-040"></a>

### MB3-040 — Select bounded joint body/fusion/layout/schedule candidates

**Depends:** MB3-019, MB3-023, MB3-026, MB3-033, MB3-037, MB3-039.
**Lane/review:** CPU, R2. **Own:** `V3/plan/search.py`, `CPU/test_joint_search.py`.
**Read:** architecture §6; pipeline §3.

**Before → after:** fastest isolated bodies or greedy covers freeze later choices → bounded
whole-region candidates retain fusion/readiness alternatives through legality and cost evaluation.

**Do:** enumerate compatible body/tile/transport/cover combinations, instantiate schedules,
allocate/verify, and keep a configured small beam (initial default eight). Retain a legal control
and mechanism-distinct exploratory candidates when costs are unknown. Log pruning reasons and
budget exhaustion. Keep compile/measured feedback interfaces, but do not implement an autotuning
service or infinite search. Final plans contain no unresolved execution choices.

**Validate/accept:** `CPU/test_joint_search.py`: a slower isolated body can win a cheaper region;
an invalid fast candidate rejects; unknown costs retain alternatives; fixed inputs yield stable plans.
**Expected:** S: only verified covers; C: bounded explainable search; P: enables end-to-end selection, not a guaranteed win.

<a id="mb3-041"></a>

### MB3-041 — Define TargetProfile legality and offline feature records

**Depends:** MB3-004, MB3-006. **Lane/review:** CPU, R2.
**Own:** `V3/target.py`, `CPU/test_target_profile.py`.
**Read:** hardware §§1–5,8; research source versions.

**Before → after:** architecture names/numeric thresholds imply capabilities → a versioned profile
separates explicit supported features, resource limits and unknown/calibrated costs.

**Do:** define capability/space/movement records, target/toolchain identity and provenance. Start
with only target entries relevant to the chosen first GPU or explicit offline examples. No
SM>=90 suffix rule, automatic topology DFS, product-peak-as-measured-bandwidth or MIG set-aside
assumption. Offline construction must not query CUDA and must label unknown properties.

**Validate/accept:** `CPU/test_target_profile.py`: unsupported instruction/target pair rejects;
different visible-resource profiles remain distinct; unknown costs never become legality facts.
**Expected:** S: target safety; C: small fact table; P: no direct gain.

<a id="mb3-042"></a>

### MB3-042 — Query the actual selected CUDA device and resource profile

**Depends:** MB3-041. **Lane/review:** GPU, R2.
**Own:** `V3/target.py`, `GTEST/test_target_query.py`, CPU mocked-query tests.
**Read:** hardware §4 and the selected release's device-property APIs.

**Before → after:** device 0/product-string assumptions drive launch choices → the requested device
supplies visible SMs, compute capability, cooperative support and relevant resource limits.

**Do:** add a lazy explicit query function using available supported APIs. Record driver/toolkit,
device/MIG identity where obtainable and missing fields honestly. Do not switch the user's device
permanently or hard-code a historical partition. Keep queried facts separate from bandwidth priors.

**Validate/accept:** `GTEST/test_target_query.py` compares returned properties with direct API
queries; CPU mocks exercise missing fields/multiple device IDs. Actual GPU evidence is required
to mark querying validated; a mocked H200 profile is not an observed device.
**Expected:** S: correct device selection; C: real provenance; P: enables legal resource tuning.

<a id="mb3-043"></a>

### MB3-043 — Emit selected atomic-body source and typed argument manifests

**Depends:** MB3-022, MB3-023, MB3-026, MB3-028, MB3-041.
**Lane/review:** CPU, R2. **Own:** `V3/codegen/source.py`, `CUDA/abi.cuh`, `CPU/test_source_emission.py`.
**Read:** architecture §§6–7; kernel reuse §7.

**Before → after:** all legacy task sources enter one translation unit → a V3 source artifact
contains only selected reachable atomic bodies and an explicit entry argument schema.

**Do:** emit the barrier-control plan and a minimal standalone-body diagnostic wrapper. Include
only declared header dependencies, preserve numerical flags, and derive logical coordinates from
plan tile IDs rather than blockIdx. Keep generated joins uniform. Reject unverified semantic/body
coverage. Full pipeline-stage emission is MB3-064, not hidden scope in this task.

**Validate/accept:** `CPU/test_source_emission.py`: stable golden source/manifest; an unused heavy
body is absent; missing body/argument binding rejects; tail predicates and required joins are present.
Source-string tests establish emission only, not executable correctness.
**Expected:** S: explicit bindings; C: inspectable selected code; P: aims to reduce universal-entry costs, unmeasured.

<a id="mb3-044"></a>

### MB3-044 — Add isolated CUDA builds with complete artifact/cache keys

**Depends:** MB3-043. **Lane/review:** CPU+CT, R2.
**Own:** `V3/codegen/compile.py`, `CPU/test_build_keys.py`, `CTEST/test_build_smoke.py`.
**Read:** hardware §8; current cuda_compiler.py; kernel reuse §§3–4.

**Before → after:** a broad cache key and ambient flags can reuse the wrong binary → source,
target, ABI, compiler/header versions and numerical policy determine the artifact identity.

**Do:** generate an explicit command for one selected target; compile in a unique build directory,
preserve stdout/stderr/command and hash the binary. Resolve optional headers only when selected.
Do not inject fast math by default. Handle missing compiler, compilation failure, timeout and
stale cache entries with diagnostics. Do not modify the global environment or upgrade dependencies.

**Validate/accept:** `CPU/test_build_keys.py` checks key invalidation and mocked failures;
`CTEST/test_build_smoke.py` compiles a tiny declared entry when CT is available. No GPU claim.
**Expected:** S: reproducible code/policy; C: trustworthy artifacts; P: compile-cache reuse, not inference gain.

<a id="mb3-045"></a>

### MB3-045 — Collect compiler and function resource reports

**Depends:** MB3-044. **Lane/review:** CPU+CT, R2.
**Own:** `V3/codegen/resources.py`, `CPU/test_resource_reports.py`, `CTEST/test_resource_smoke.py`.
**Read:** hardware §4; GPU audit §6.

**Before → after:** isolated source estimates stand in for the composed entry → each artifact
records available registers, static/dynamic shared requirements, stack/local/spill information.

**Do:** preserve raw compiler output and parse supported fields with version-aware fixtures.
Distinguish static shared, requested dynamic shared and unknown resource fields. Never infer the
complete register count as max(body registers). Runtime function attributes will supplement this
record after loading; absence of a parser match cannot become zero usage.

**Validate/accept:** `CPU/test_resource_reports.py`: several raw-format fixtures, missing fields
and parse failures; `CTEST/test_resource_smoke.py` checks a real compiled test entry where available.
**Expected:** S: no silent resource assumptions; C: visible composition cost; P: guides pruning only.

<a id="mb3-046"></a>

### MB3-046 — Implement a lazy, typed, per-context CUDA driver wrapper

**Depends:** MB3-006, MB3-044. **Lane/review:** CPU+GPU, R3.
**Own:** `V3/runtime/driver.py`, `CPU/test_driver_abi.py`, `GTEST/test_driver_smoke.py`.
**Read:** existing launcher.py; kernel reuse §§1,7; selected CUDA driver ABI documentation.

**Before → after:** global module/function state and loosely typed calls → explicit artifact/device/
context handles, typed argument packing and checked driver return codes.

**Do:** define ctypes argument/return types or an equivalently narrow supported binding, including
64-bit pointer/stream handling. Reuse PyTorch's active compatible context; do not create a competing
context implicitly. Key modules by artifact and context/device. Keep module and argument storage
alive through use; expose unload only after safe completion. No import-time driver initialization.

**Validate/accept:** `CPU/test_driver_abi.py`: pointer width, argument ordering, errors, lazy loading;
`GTEST/test_driver_smoke.py`: a trivial entry writes expected output on the selected stream/device.
**Expected:** S: ABI/context safety; C: no hidden global runner; P: no speedup promise.

<a id="mb3-047"></a>

### MB3-047 — Enforce actual-entry cooperative launch admission

**Depends:** MB3-042, MB3-045, MB3-046. **Lane/review:** CPU+GPU, R3.
**Own:** `V3/runtime/driver.py`, `CPU/test_launch_admission.py`, `GTEST/test_launch_admission.py`.
**Read:** hardware §4; pipeline §10.

**Before → after:** num_sms and fixed shared memory are assumed safe → function attributes and
occupancy APIs admit the exact block/shared-memory/worker configuration before launch.

**Do:** set required dynamic-shared opt-in attributes when supported; query active CTAs for the
actual entry and derive the cooperative bound. Check block dimensions, visible device resources,
cooperative support and role requirements. Unknown mandatory legality facts reject. A smaller
worker-count candidate may be retried explicitly; no ordinary oversubscribed fallback grid.

**Validate/accept:** CPU mocks reject too many workers/shared bytes/threads; GPU test launches legal
cases and confirms unsafe requests are rejected **before** launch. Retain the admission report.
**Expected:** S: residency/progress preconditions; C: clear limits; P: enables worker/resource selection.

<a id="mb3-048"></a>

### MB3-048 — Build the standalone and lean-persistent body test harness

**Depends:** MB3-005, MB3-023, MB3-043, MB3-044, MB3-046, MB3-047.
**Lane/review:** CT+GPU, R2. **Own:** `tests/test_v3/support/body_harness.py`,
`GTEST/test_body_harness.py`.
**Read:** kernel reuse §8; performance §4.

**Before → after:** body quality is visible only through the entire old model → identical device
body logic can be validated standalone and under a controlled persistent launch envelope.

**Do:** accept an explicit BodySpec, exact shape/layout, input set and worker count. Generate both
wrappers from the same selected body; persistent wrapper strip-mines more logical tiles than
workers and repeats legal body calls with scratch reuse. A temporary trivial test body validates
the harness before optimized math exists. This is a diagnostic, not a generic compiler result.

**Validate/accept:** `GTEST/test_body_harness.py`: shared inputs, two sequential tiles, N<P/N>P,
output ownership, all-operation counts and actual resource/admission reports are saved.
**Expected:** S: isolated oracle; C: separates math from composition; P: measurement capability, not gain.

<a id="mb3-049"></a>

### MB3-049 — Implement one lean M=1 K-parallel SIMT linear body

**Depends:** MB3-023, MB3-024, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/linear_simt.cuh`, registration, `GTEST/test_linear_simt.py`.
**Read:** GPU audit §§6–7; kernel reuse §§7–8; pinned GEMV mapping references.

**Before → after:** a few threads each traverse K serially → a bounded warp/CTA mapping cooperates
along K with coalesced weight access and an exact output tile contract.

**Do:** start with NK-contiguous weights and M=1, one row/output or a small row group per warp.
Lanes accumulate strided K positions into FP32 and reduce with correct active participation.
Handle K/N tails by predicates/zero contributions; use compile-time bounded accumulators, not a
runtime-M acc[64]. Preserve required bias/output casts. Add only advertised FP16/BF16 support;
unsupported layouts use an explicit pack/candidate rejection, not scalar hidden fallback.

**Validate/accept:** `GTEST/test_linear_simt.py`: LINEAR_TINY/HOT, zero/cancellation/range tests,
multiple worker counts, standalone versus persistent correctness and resources. Measure later in
MB3-080; being functional is not vendor parity.
**Expected:** S: same Linear policy; C: simple mapping; P: targets useful lanes/bandwidth, size-dependent.

<a id="mb3-050"></a>

### MB3-050 — Generate lean pointwise device expressions with explicit casts

**Depends:** MB3-014, MB3-048. **Lane/review:** CT+GPU, R2.
**Own:** `CUDA/bodies/pointwise.cuh`, pointwise support in `V3/codegen/source.py`,
`GTEST/test_pointwise_body.py`.
**Read:** IR §4; kernel reuse §9.

**Before → after:** a broad uop interpreter handles a specialized graph → generated supported
expressions retain exact operand/broadcast/cast behavior without runtime opcode dispatch.

**Do:** emit only MB3-014's supported expression set, index through proven strides, and implement
the selected activation approximation. Cast at every semantic boundary; compiler flags and
intrinsics must match NumericalPolicy. Keep excess live outputs when required by the graph.
Reject unsupported expressions, rather than silently deleting them.

**Validate/accept:** `GTEST/test_pointwise_body.py`: SiLU/GELU variants, residual/bias broadcasts,
mixed intermediate dtypes, odd sizes and extra consumers under both wrappers.
**Expected:** S: exact expression policy; C: readable generated math; P: removes dispatch/materialization where selected.

<a id="mb3-051"></a>

### MB3-051 — Implement the declared RMSNorm device variants

**Depends:** MB3-015, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/rmsnorm.cuh`, registration, `GTEST/test_rmsnorm_body.py`.
**Read:** IR §4; NORM_VARIANTS; kernel reuse §9.

**Before → after:** norm handling is broad or interpreted → one lean reduction body explicitly
supports the required axis/epsilon/weight/cast variants.

**Do:** start one cooperative CTA per vector/row, FP32 reduction where required, correct tail
participation and an output pass matching the reference. Keep weight versus 1+weight and
multiply-before/after-cast as separate guarded cases. Do not push a scale through a linear.
No per-vocabulary-tile norm duplication is selected here.

**Validate/accept:** `GTEST/test_rmsnorm_body.py`: NORM_VARIANTS, odd dimensions, small/large values,
empty/unsupported axes diagnostics and repeated scratch reuse; compare policy-specific outputs.
**Expected:** S: model-correct normalization; C: small explicit variants; P: leaner body, measured benefit pending.

<a id="mb3-052"></a>

### MB3-052 — Implement embedding and required indexed-copy primitives

**Depends:** MB3-010, MB3-048. **Lane/review:** CT+GPU, R2.
**Own:** `CUDA/bodies/indexed_copy.cuh`, registration, `GTEST/test_indexed_copy.py`.
**Read:** IR §3; existing embedding/copy task tests.

**Before → after:** input/output and views rely on an opaque legacy path → V3 has bounded
embedding-row gather and required copy/layout conversion bodies with dtype-correct indexing.

**Do:** support the first fixtures' integer token IDs and floating row dtype; preserve index width.
Bounds-check via guards or device predicates according to the contract. Proven views emit no work;
non-view conversions are real timed actions. Do not assume every buffer is two-byte float storage.
Use explicit copy semantics for overlap; reject unsafe in-place overlap rather than memcpy guessing.

**Validate/accept:** `GTEST/test_indexed_copy.py`: first/last/invalid token, multiple returned rows,
strided slices, noncontiguous reshape copy, exact integer behavior and no-op views.
**Expected:** S: correct indices/dtypes; C: explicit copies; P: avoids unnecessary movement, no free conversions.

<a id="mb3-053"></a>

### MB3-053 — Implement RoPE device bodies for the recognized conventions

**Depends:** MB3-016, MB3-048. **Lane/review:** CT+GPU, R2.
**Own:** `CUDA/bodies/rope.cuh`, registration, `GTEST/test_rope_body.py`.
**Read:** MB3-016's semantic contract; pipeline §4.

**Before → after:** positional work is an assumed generic task → supported head tiles apply
the exact pairing/rotary extent/position/scaling policy and expose complete head output.

**Do:** use explicit cos/sin inputs or a separately supported generated expression. Handle partial
rotary dimensions without touching unrotated channels. Retain Q/K dtype/layout distinctions.
Describe each output footprint and require all needed pairs before publication.

**Validate/accept:** `GTEST/test_rope_body.py`: half-split/interleaved, partial rotation, boundary
positions, odd unsupported rotary dimensions and known signed reference vectors.
**Expected:** S: positional correctness; C: clear body guards; P: enables local fusion later.

<a id="mb3-054"></a>

### MB3-054 — Implement bounded in-grid cache-slot writes

**Depends:** MB3-018, MB3-025, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/cache_write.cuh`, registration, `GTEST/test_cache_write_body.py`.
**Read:** pipeline §§4,9; IR §8.

**Before → after:** cache updates are only a reference operation → a device body writes the
declared current K/V slot without modifying other live state.

**Do:** use the fixed-capacity contiguous contract first. Guard p/capacity and dtype/layout before
unsafe access. Distinguish value writes from later valid-length publication. Output-region
completion must include every writer. Do not allocate a new cache inside the kernel or assume
a nonempty prefix if p=0.

**Validate/accept:** `GTEST/test_cache_write_body.py`: STATE_POISON, slots 0/last, unchanged old
positions, overflow rejection and separate K/V completion. Inter-CTA publication follows MB3-063.
**Expected:** S: explicit state writes; C: visible ownership; P: removes external update grids when composed.

<a id="mb3-055"></a>

### MB3-055 — Implement one ordinary decode online-softmax attention body

**Depends:** MB3-017, MB3-025, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/decode_attention.cuh`, registration, `GTEST/test_decode_attention_body.py`.
**Read:** kernel reuse §9; pipeline §4; NumericalPolicy.

**Before → after:** a full score/probability materialization or legacy task is required → a
supported query-head tile streams valid K/V while retaining running softmax state.

**Do:** implement the documented m/l/weighted-value update, with safe first-valid-tile handling.
Apply exact masks, scale, GQA mapping, valid length and accumulation/output policy. All-masked
and empty cases follow the reference; never evaluate -inf-minus--inf blindly. Start single-owner
attention, not split-context or a copied prefill kernel. Declare accumulator/scratch footprint.

**Validate/accept:** `GTEST/test_decode_attention_body.py`: ATTENTION_TINY plus D=64/128 where
supported, context tails, GQA, current-slot visibility precondition and range/mask adversarial tests.
**Expected:** S: real cached attention; C: explicit algorithm/body boundary; P: avoids score IO, not assured whole-entry speed.

<a id="mb3-056"></a>

### MB3-056 — Resolve one device-library source and toolchain compatibility choice

**Depends:** MB3-001, MB3-023, MB3-041, MB3-044.
**Lane/review:** CPU+CT, R2. **Own:** `ART/tasks/MB3-056/compatibility.md`,
`CTEST/test_library_probe.py` and its minimal compile source.
**Read:** kernel reuse §§3–6; research pinned revisions; actual selected headers/documentation.

**Before → after:** “use cuBLASDx/CUTLASS/MPK” is an unspecified dependency → one versioned source
choice has a reproduced compile contract or an explicit rejection record.

**Do:** inspect the selected target and installed toolkit; test a minimal compatible cuBLASDx
interface early, with exact required block/scratch/descriptor rules. If incompatible, select an
appropriate adapted CUTLASS/CuTe/MPK source, not a four-library tournament. Record license notices
and dependency versions. Do not silently install CUDA 13+ or assume a 0.7.1 API works on 12.8.
Bound the probe to a small fixed set of descriptors/shapes and record each failure.

**Validate/accept:** compile probe/report identifies an eligible source for MB3-057, or a specific
external blocker. A rejected probe is completed research, not a working tensor-core body.
**Expected:** S: supported ABI only; C: resolved source choice; P: unknown until shape/composition tests.

<a id="mb3-057"></a>

### MB3-057 — Adapt one tensor-core tile body below its host launcher

**Depends:** MB3-023, MB3-048, MB3-056. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/linear_tensorcore.cuh`, its source/registration manifest,
`GTEST/test_tensorcore_body.py`.
**Read:** the chosen pinned source; kernel reuse §§3–5,7.

**Before → after:** library math exists only as an external call → one supported device tile runs
inside MegaBake's existing grid with explicit logical coordinates and resources.

**Do:** adapt the selected collective/block interface, retaining tested load/layout/MMA/epilogue
protocols. Declare exact role/block constraints and descriptor lifetime. If padding is necessary,
represent/initialize it and include its runtime cost. Do not invent masked-tail support or convert
a CUfunction into a device call. Initially advertise ATOMIC_TILE unless staged interfaces are
independently implemented and tested.

**Validate/accept:** `GTEST/test_tensorcore_body.py`: supported hot shapes, tail rejection/padding,
standalone and composed output/resource reports. Unsupported common-envelope combinations reject.
**Expected:** S: policy-compatible linear; C: narrow adapter; P: potential better math/data path, no cuBLAS parity guarantee.

<a id="mb3-058"></a>

### MB3-058 — Implement one target-gated async staging primitive

**Depends:** MB3-023, MB3-041, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/async.cuh`, `CTEST/test_async_compile.py`, `GTEST/test_async_lifetimes.py`.
**Read:** pipeline §§2,6,8; selected release's async-copy/proxy rules.

**Before → after:** a body cannot expose trustworthy copy-stage completion → one selected async
movement mechanism has explicit issue/wait/source-retirement contracts and alignment guards.

**Do:** begin with supported global-to-shared staging on the chosen target, including bytes,
participants and any barrier phase. Keep synchronous fallback labelled as such. If the selected
body uses async stores, implement/test source-read completion separately from destination visibility;
otherwise use ordinary stores and do not advertise async-store support. Specify required generic/
async-proxy ordering for the actual instructions. No guessed fence sequence.

**Validate/accept:** `GTEST/test_async_lifetimes.py`: patterned bytes, tail/alignment rejection,
two slots reused repeatedly, delayed consumer and applicable store-retirement litmus. CT-only
success leaves GPU protocol validation pending; independent review is required.
**Expected:** S: correct stage tokens; C: real capability boundary; P: enables movement overlap, not proved by API name.

<a id="mb3-059"></a>

### MB3-059 — Expose separate preload and compute stages for the SIMT body

**Depends:** MB3-049, MB3-058. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/linear_simt.cuh`, `GTEST/test_preloadable_linear.py`.
**Read:** pipeline §§2,6; BodySpec contract.

**Before → after:** linear is an indivisible tile call → its selected weight tile can be loaded
into caller-owned staging and consumed later with a verified lifetime.

**Do:** split only the supported tile's load and compute boundary, keeping the atomic wrapper as
a composition of these stages. Supply explicit slot/layout/byte metadata and load-complete token.
Do not load a missing activation merely because the weight address is known. Respect slot release
after every accessing operation, and keep internal versus cross-task buffering distinct.

**Validate/accept:** `GTEST/test_preloadable_linear.py`: compute receives prepared data, delayed
activation, two sequential tiles and zero/one/two legal slot configurations; atomic/staged math agrees.
**Expected:** S: unchanged Linear semantics; C: usable staging interface; P: enables real weight lookahead.

<a id="mb3-060"></a>

### MB3-060 — Expose owner-held FP32 reduction continuations

**Depends:** MB3-024, MB3-049, MB3-048. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/linear_continuation.cuh`, shared math helpers in `linear_simt.cuh`,
`GTEST/test_linear_continuation.py`.
**Read:** pipeline §5; IR §7.

**Before → after:** down projection requires all K inputs before invocation → begin/update/finalize
maintain a private supported accumulator across ordered K chunks.

**Do:** implement a bounded accumulator type, zero initialization, exact chunk update and final
cast/epilogue. Retain the same owner/participants until finalize; include the long live range in
resource reports. No BF16 rounding or residual addition after individual chunks unless explicitly
required by the reference. Start from the lean SIMT body; tensor-core continuation is optional.

**Validate/accept:** `GTEST/test_linear_continuation.py`: odd K/tail chunk, several partitions,
zero data, cancellation, repeated accumulators and policy-specific reference comparison; report
register/spill costs. Missing/duplicate chunks must already fail plan validation.
**Expected:** S: valid continued reduction; C: explicit finalization; P: enables MLP overlap without global partials.

<a id="mb3-061"></a>

### MB3-061 — Fuse paired gate/up output tiles with exact gating

**Depends:** MB3-014, MB3-049, MB3-050. **Lane/review:** CT+GPU, R3.
**Own:** `CUDA/bodies/gated_projection.cuh`, registration, `GTEST/test_gated_projection.py`.
**Read:** pipeline §5; IR §4; performance §6.

**Before → after:** gate and up vectors are separately written/read → a compatible tile computes
both projections and the actual gating expression before publishing h.

**Do:** align paired intermediate row ranges, retain required projection casts, then apply the
reference activation/multiply policy. Preserve extra live g/u outputs if requested, or reject
that fused alternative. Emit complete h chunks and metadata; do not claim h's cross-CTA traffic
disappears. Include doubled projection accumulators/staging in the resource budget.

**Validate/accept:** `GTEST/test_gated_projection.py`: SiLU and GELU references, I tail, extra
consumers and cast-order cases; compare materialized/fused bodies under the same policy.
**Expected:** S: exact gating; C: named composite; P: removes eligible g/u traffic, may lose through resources.

<a id="mb3-062"></a>

### MB3-062 — Fuse compatible QKV head work with positional/cache output

**Depends:** MB3-025, MB3-049, MB3-053, MB3-054.
**Lane/review:** CT+GPU, R3. **Own:** `CUDA/bodies/qkv_head.cuh`,
`GTEST/test_qkv_head_composite.py`.
**Read:** pipeline §4; architecture §8.

**Before → after:** projection/positional/cache stages all materialize independently → a supported
head/GQA tile performs compatible local work while preserving complete region boundaries.

**Do:** use consumer-aligned head ranges and verified weight/activation layout. Fuse only operations
whose threads own the required values; retain all externally live Q/K/V results. Describe exactly
which current-slot writes complete and what remains for the publication primitive. Do not fuse
the entire QKV tensor into an opaque body that cannot publish the intended head granularity.

**Validate/accept:** `GTEST/test_qkv_head_composite.py`: MHA/GQA, both relevant positional forms,
cache tails and extra outputs; compare with separate bodies and inspect live storage.
**Expected:** S: preserved head/cache semantics; C: explicit local ownership; P: less movement and earlier ready heads, unmeasured.

<a id="mb3-063"></a>

### MB3-063 — Implement conservative device publication and acquire primitives

**Depends:** MB3-030, MB3-046, MB3-047, MB3-048, MB3-058.
**Lane/review:** CT+GPU, R3. **Own:** `CUDA/events.cuh`, `GTEST/test_publication.py`.
**Read:** pipeline §9 and selected CUDA memory/async contracts.

**Before → after:** a counter increment is treated as sufficient → a reviewed protocol explicitly
orders all producers' completed data before dependent cross-CTA reads.

**Do:** use aligned device-scope atomics with checked count width. Initialize counters in the
owned grid, then uniformly join. Complete all writer/async-store obligations before the publishing
thread's acquire-release RMW; consumer acquire observes the exact expected contributions and
hands readiness to its CTA correctly. Keep event addresses unique within an invocation.
Do not copy volatile-only polling or use source-read completion as destination publication.

**Validate/accept:** `GTEST/test_publication.py`: many producers, delayed producer/store, repeated
fresh invocations, consumer CTA sharing and poisoned outputs; failures/timeouts are retained.
A simple value match once is insufficient; require R3 review of the actual instruction protocol.
**Expected:** S: safe cross-worker data flow; C: one publication primitive; P: synchronization cost to measure.

<a id="mb3-064"></a>

### MB3-064 — Lower bounded staged worker programs into one cooperative entry

**Depends:** MB3-031, MB3-034, MB3-035, MB3-036, MB3-043, MB3-058, MB3-063.
**Lane/review:** CPU+CT+GPU, R3.
**Own:** `V3/codegen/source.py`, `CUDA/abi.cuh`, `CPU/test_staged_emission.py`,
`GTEST/test_staged_entry.py`.
**Read:** pipeline §§2–3,8–10; IR §7.

**Before → after:** V3 can emit only atomic phase controls → one generated entry executes verified
reserve/preload/compute/update/publish/release programs with bounded worker/cohort roles.

**Do:** translate typed tokens to the corresponding body/runtime primitive, preserving argument
types and accumulator scope. Emit common initialization and uniform joins; idle workers cannot
return early. Preserve tracked outstanding operations across stage returns. Generate finite
loops/template repeats, not a CPU per-tile dispatcher or arbitrary queue interpreter. Reject any
unresolved action/capability/lifetime; never silently drop it.

**Validate/accept:** golden source checks plus `GTEST/test_staged_entry.py` with tiny synthetic
producer/consumer bodies; multiple tiles/workers, complete event lifecycle and admission reports.
**Expected:** S: executable verified plan; C: one bounded runtime; P: enables non-launch mechanisms, not yet model evidence.

<a id="mb3-065"></a>

### MB3-065 — Validate device protocol lifetimes under adversarial timing

**Depends:** MB3-033, MB3-063, MB3-064.
**Lane/review:** GPU, R3. **Own:** `GTEST/test_protocol_stress.py`, small debug-only delay hooks.
**Read:** pipeline §§9–11; MB3-030/031 counterexamples.

**Before → after:** codegen passes a favorable smoke case → selected device protocols survive
repeated invocations, delayed participants and safe staged-buffer reuse tests.

**Do:** perturb producer/consumer/store timing at legal boundaries; vary worker/tile counts,
ring iterations and event fan-in. Run applicable memory/race/synchronization diagnostics in a
separate test process. Apply an external test watchdog; a suspected deadlock is a failed isolated
test, not permission for an unproved early-return escape inside a grid collective. Never reset
a shared GPU automatically.

**Validate/accept:** save seeds, source/plan hash, tool output and failing/timeout cases. All required
correctness/litmus checks pass on the selected target; sanitizer success alone is not a proof.
**Expected:** S: concrete protocol evidence; C: replayable failures; P: debug hooks disabled for timing.

<a id="mb3-066"></a>

### MB3-066 — Run the generated head-ready pipeline on device

**Depends:** MB3-034, MB3-055, MB3-062, MB3-064, MB3-065.
**Lane/review:** GPU, R3. **Own:** head-specific wiring in `V3/plan/templates.py`,
`tests/test_v3/support/pipeline_harness.py`, `GTEST/test_head_pipeline.py`.
**Read:** pipeline §4; performance §6 ablations.

**Before → after:** head-ready execution exists as a plan template → actual projection/cache
producers and attention consumers run under that generated plan.

**Do:** bind supported head-composite and attention bodies, retain matched separate/barrier
controls, and verify actual common resources. Delay one head group in a diagnostic build to test
independent readiness. Keep output projection coarse here. This task connects existing components;
do not add another model-specific scheduler.

**Validate/accept:** `GTEST/test_head_pipeline.py`: values/state agree for short/long synthetic
contexts; group 0 may execute before unrelated group 1 completes but never before its own V/cache
data. Save resource and causal-event evidence; performance selection is MB3-082.
**Expected:** S: state-safe head overlap; C: concrete early-edge trace; P: candidate tail/pipeline benefit.

<a id="mb3-067"></a>

### MB3-067 — Run the generated streamed gated MLP on device

**Depends:** MB3-035, MB3-060, MB3-061, MB3-064, MB3-065.
**Lane/review:** GPU, R3. **Own:** MLP-specific wiring in `V3/plan/templates.py`,
`tests/test_v3/support/pipeline_harness.py`, `GTEST/test_mlp_pipeline.py`.
**Read:** pipeline §5; performance §6.

**Before → after:** owner continuations and chunk producers work separately → generated cohorts
overlap gated chunks with ordered down updates under one entry.

**Do:** bind GATE_TINY/BLOCK_DEVICE variants, allocate h once per region, retain owner accumulators
through all required chunks and finalize once. Include all live scratch/accumulators in admission.
Keep full-materialization control and both relevant activation policies. Explicitly reject an
owner assignment that exceeds the supported accumulator capacity.

**Validate/accept:** `GTEST/test_mlp_pipeline.py`: odd final chunk, multiple owner counts,
interleaved producer delays, repeated calls, exact state/residual/cast policy; no partial output is
published. Save evidence of early K updates and full resource cost.
**Expected:** S: complete streamed MLP; C: visible accumulator lifetime; P: potential overlap, not guaranteed.

<a id="mb3-068"></a>

### MB3-068 — Run cross-task matrix-data lookahead on device

**Depends:** MB3-036, MB3-059, MB3-064, MB3-065.
**Lane/review:** GPU, R3. **Own:** lookahead wiring in `V3/plan/templates.py`,
`tests/test_v3/support/pipeline_harness.py`, `GTEST/test_weight_lookahead.py`.
**Read:** pipeline §6; hardware §7.

**Before → after:** staged loads exist only within a body/prototype → the generated worker program
issues the next task's weight load while current useful work remains.

**Do:** connect caller-owned slots and declared body stages across at least two distinct task
bindings; test zero/one/two legal lookahead slots. Make next activation arrive late and ensure only
its independent weight data loads early. Include fill/drain, internal stages and resource growth.
Task-descriptor prefetch alone does not satisfy this task.

**Validate/accept:** `GTEST/test_weight_lookahead.py`: matrix data consumed from the prepared slot,
correct release across repeated tiles, no activation-before-ready reads, and a diagnostic event
sequence showing next-load issue before current work completes.
**Expected:** S: safe next-task movement; C: actual versus nominal prefetch; P: candidate stall reduction.

<a id="mb3-069"></a>

### MB3-069 — Test common-entry resource compatibility for mixed real bodies

**Depends:** MB3-045, MB3-047, MB3-049, MB3-066, MB3-067, MB3-068.
**Lane/review:** CT+GPU, R3. **Own:** `GTEST/test_composed_resources.py`, resource-report integration.
**Read:** architecture §7; hardware §4; GPU audit §6.
**Optional input:** validated MB3-057 body; mandatory for any tensor-core composition claim,
not for examining a valid SIMT mixture when the library route is unavailable.

**Before → after:** individually successful bodies imply a viable model → representative mixed
entries reveal register/spill/shared-memory and block-role incompatibilities before whole-model work.

**Do:** compile selected linear, norm, positional, attention and pipeline mixtures for a small
set of compatible common block shapes. Inspect true combined attributes, including continuation
live ranges and outstanding stages. Preserve only reachable alternatives per artifact. Reject or
revise incompatible candidates; do not average incompatible launch requirements.

**Validate/accept:** `GTEST/test_composed_resources.py`: admission uses actual functions; missing/
oversized resources reject before launch; standalone/composed reports share body/source hashes.
If the tensor-core candidate is rejected, preserve a valid SIMT path and report the portfolio gap.
**Expected:** S: valid common envelope; C: localized composition failures; P: identifies regressions, not a gain itself.

<a id="mb3-070"></a>

### MB3-070 — Create session-owned weights, state and typed workspace

**Depends:** MB3-004, MB3-018, MB3-029, MB3-046.
**Lane/review:** CPU+GPU, R2.
**Own:** `V3/runtime/session.py`, `CPU/test_session_bindings.py`, `GTEST/test_session_storage.py`.
**Read:** architecture §§8–9; IR §8; current loader.py.

**Before → after:** a global half-only runner owns mutable state → each V3 session has explicit
immutable weight bindings, per-session state/workspace and typed allocation ownership.

**Do:** validate binding shapes/dtypes/layouts; avoid unconditional .half(). Start with owned stable
weight storage/packing, report its setup/memory cost and deduplicate compatible tied storage.
Keep original source weights unchanged. Provide explicit rebinding/invalidation, not pointer-
identity-based correctness. Allocate byte-correct aligned workspace and independent state per
session. Do not implement lazy offload, paging or implicit cross-session sharing of mutable arenas.

**Validate/accept:** CPU binding failures and GPU tests for BF16/FP16/int inputs, tied weights,
changed source weights, session independence, state capacity and workspace bounds.
**Expected:** S: ownership/dtype correctness; C: no global mutable runner; P: stable reusable setup, cost reported.

<a id="mb3-071"></a>

### MB3-071 — Implement guarded step invocation and output ownership

**Depends:** MB3-047, MB3-064, MB3-070.
**Lane/review:** CPU+GPU, R3.
**Own:** `V3/runtime/session.py`, `CPU/test_session_guards.py`, `GTEST/test_session_run.py`.
**Read:** architecture §9; IR §8; performance §7.

**Before → after:** each call clones counters/outputs with implicit lifetime rules → a session
validates guards, launches one admitted entry and returns correctly owned outputs/state.

**Do:** define the initial serialized/fixed-stream policy explicitly. Preserve ordering across calls;
a host lock alone cannot serialize kernels on different streams. Bind new inputs/position without
hidden host-device assumptions. Counters initialize in the entry. `run` returns independent owned
outputs; `run_into` uses explicitly caller-owned storage. Retain inputs/descriptors/allocations
until asynchronous use completes and apply the selected stream-lifetime contract.

**Validate/accept:** `GTEST/test_session_run.py`: retain output A across call B; changed token/
position takes effect; guard violation rejects before execution; cross-stream misuse is rejected
or ordered as documented; all required allocation/copy/update operations are counted.
**Expected:** S: API/state lifetime safety; C: explicit call contract; P: less avoidable setup, measured later.

<a id="mb3-072"></a>

### MB3-072 — Add strict rejection and an explicit ordinary compiled fallback

**Depends:** MB3-006, MB3-007, MB3-071.
**Lane/review:** CPU+GPU, R2. **Own:** `V3/backend.py`, `CPU/test_fallback_policy.py`.
**Read:** architecture §§3,9–10; performance §8.

**Before → after:** an unsupported path may look like megakernel success → strict and ordinary
modes have separate, observable outcomes.

**Do:** strict mode raises a structured unsupported/legality diagnostic before changing state.
Ordinary mode may execute an explicitly identified equivalent compiled callable. Preserve output/
state/exception contracts and record which path ran. Do not catch CUDA execution/correctness faults
and replay a stateful step automatically; state may already have changed. Fallback is selected
before unsafe execution, not an after-the-fact way to hide errors.

**Validate/accept:** `CPU/test_fallback_policy.py` with mocks plus applicable GPU integration:
unsupported node, unavailable body, guard failure and compile failure report distinct outcomes;
runtime fault does not execute the step twice.
**Expected:** S: exactly-once state semantics; C: no hidden success; P: fallback usability, never a strict win.

<a id="mb3-073"></a>

### MB3-073 — Add a tensor-state Hugging Face one-step reference/export adapter

**Depends:** MB3-007, MB3-008, MB3-018, MB3-021.
**Lane/review:** CPU+GPU, R2.
**Own:** `src/megabake/integrations/transformers_v3.py`, `CPU/test_hf_step_adapter.py`,
`GTEST/test_hf_step_reference.py`.
**Read:** existing HF wrapper; IR §§2,5,8; actual pinned Transformers cache/model interfaces.

**Before → after:** the only HF wrapper disables cache → a separately named adapter represents
one real cached step with explicit tensor state, position and output bindings.

**Do:** begin with tiny configuration instances of the selected small-model family, no weights
download. Flatten/unflatten the selected cache representation through a versioned adapter;
retain the original HF step as an independent reference. Export fixed-capacity state transitions
without host .item() inside the captured path. Do not substitute a hand-written approximate
decoder for the actual model. Support Gemma's chosen interface in a separate child card if its
cache contract differs; do not hide that additional scope.

**Validate/accept:** reference/cache values agree with genuine advancing HF calls; untouched
cache regions and position rules hold; repeated export works on the pinned versions. Unsupported
cache objects produce a bounded diagnostic. GPU FP16/BF16 agreement remains a separate gate.
**Expected:** S: real cached workload; C: distinct from uncached legacy API; P: enables valid comparisons.

<a id="mb3-074"></a>

### MB3-074 — Connect the full generated decoder-block vertical slice

**Depends:** MB3-019, MB3-040, MB3-050, MB3-051, MB3-066, MB3-067, MB3-068, MB3-071.
**Lane/review:** CPU+GPU, R3.
**Own:** `V3/backend.py`, BLOCK_TINY/BLOCK_DEVICE fixtures, `GTEST/test_generated_block.py`.
**Read:** architecture §11; dataflow §5; pipeline §11.

**Before → after:** components/probes are manually connected → one ordinary FX block passes
through normalization, semantic matching, joint planning and generated execution.

**Do:** build the exact residual/norm/attention/gated-MLP graph from fixture torch operations.
Feed it through the public V3 pipeline, not preassigned node IDs or a model-name scheduler.
Integrate selected bindings/state and preserve every live residual/output. Generate barrier and
all supported pipeline candidates; inspect missing matches/capabilities. Keep the patch integration-
only: a newly discovered operator gap becomes a named bounded child task with a reproducer.

**Validate/accept:** `GTEST/test_generated_block.py`: output/state equality, alternate names with
identical structure, changed norm/mask not falsely matched, both schedule policies, repeated steps.
**Expected:** S: graph-driven block; C: integrated diagnostics; P: first complete mechanism experiment, not yet a model win.

<a id="mb3-075"></a>

### MB3-075 — Reuse repeated-layer templates and enable guarded boundary lookahead

**Depends:** MB3-020, MB3-036, MB3-064, MB3-074.
**Lane/review:** CPU+CT+GPU, R3.
**Own:** repeated-region support in `V3/plan/templates.py` and `V3/codegen/source.py`, `CPU/test_repeated_plan.py`,
`GTEST/test_repeated_layers.py`.
**Read:** IR §6; pipeline §6; architecture §7.

**Before → after:** each repeated layer duplicates code or ends all useful lookahead → verified
templates retain distinct bindings and can preload independent next-layer weights.

**Do:** start REPEATED_TINY with two layers, parameterized bindings and explicit layer-local state.
Compare bounded unrolling with loops/outlined helpers where supported. Retain activation/effect
dependencies across the boundary; only independent weights may preload early. Recheck all buffer
lifetimes and actual entry resources after structuring. Do not introduce an opaque Layer IR.

**Validate/accept:** `GTEST/test_repeated_layers.py`: distinct weights/state, second-layer
correctness, boundary preload lifetime, source-size/resource report and no weight-sharing fiction.
**Expected:** S: preserves repetition semantics; C: reusable generated structure; P: code-size/lookahead opportunity, measured later.

<a id="mb3-076"></a>

### MB3-076 — Integrate full-model FX compilation and the opt-in V3 API

**Depends:** MB3-021, MB3-040, MB3-052, MB3-069, MB3-072, MB3-073, MB3-075.
**Lane/review:** CPU+GPU, R3.
**Own:** `V3/backend.py`, narrow opt-in public API hook, `CPU/test_v3_api.py`,
`GTEST/test_generated_causal_lm.py`.
**Read:** architecture §§1–3,9–11; IR §10.

**Before → after:** V3 reaches blocks but not the user workload → a supported model capture,
embedding, repeated layers and full vocabulary head produce one admitted strict artifact.

**Do:** connect existing stages, select compiled/legal candidates and preserve complete logits/
state/output ownership. Add an explicit V3 API without changing the default legacy backend.
Expose reports for all unmatched/unsupported nodes. Leave the optional torch.compile backend hook
to MB3-085; the direct FX/export API is sufficient here.
Compiler selection must depend on graph semantics/guards, not a model-name engine switch.

**Validate/accept:** tiny full-model graph and selected real-model captures compile through the
same route; legacy API still behaves as before; strict traces contain one compute grid and full
state/logits work. Actual real-model numerical/performance campaign is MB3-083/084.
**Expected:** S: complete generic capture path for supported subset; C: user-facing diagnostics;
P: enables whole-model measurement, no automatic speedup.

<a id="mb3-077"></a>

### MB3-077 — Build raw-sample timing and uncertainty utilities

**Depends:** MB3-004, MB3-005, MB3-006. **Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/timing.py`, `CPU/test_timing_stats.py`, `GTEST/test_timing_smoke.py`.
**Read:** performance §7; GPU audit §5.

**Before → after:** timing scripts conflate launch, device and user latency → every sample has
an explicit timing view, workload contract, trial block and selection/validation role.

**Do:** implement separate host enqueue, ordered CUDA-event and synchronized wall-time paths.
Keep compilation/capture/warmup outside steady-state samples, but keep runtime-required copies,
resets and output production inside their declared contract. Support alternated backend blocks,
raw sample retention, median/dispersion and block-level resampling with a recorded seed. Reject
empty, non-finite and mismatched samples; do not treat iterations in one block as independent runs.
Make fixture restoration an explicit setup callback, distinct from required runtime work.

**Validate/accept:** CPU fixtures with known distributions verify units, median, pairing and
deterministic resampling; an inadequate sample design reports inconclusive uncertainty. GPU smoke
tests verify stream/event ordering and that required work is inside the interval. No p99 claim
from a handful of samples. Statistics code can pass CPU tests while GPU timing remains unvalidated.
**Expected:** S: equal timed work; C: reproducible uncertainty; P: measurement only, no speedup.

<a id="mb3-078"></a>

### MB3-078 — Produce complete operation and causal-stage trace evidence

**Depends:** MB3-006, MB3-045, MB3-046, MB3-077. **Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/trace.py`, `CPU/test_trace_classification.py`, `GTEST/test_trace_smoke.py`.
**Read:** GPU audit §5; performance §§6–7; pipeline §§9,11.

**Before → after:** filtered kernel counts can omit work and cannot establish overlap → a trace
ledger retains all operations and distinguishes submissions, compute grids, copies and resets.

**Do:** consume the available pinned profiler/driver evidence without relying on one kernel-name
prefix. Preserve unknown operations and raw trace locations. Attach plan/action IDs to optional
diagnostic stage records for preload, completion, consumer start and retirement. Define the stage
record schema now; connect it to generated actions in MB3-082 after MB3-064 exists. Do not order
raw per-SM cycle counters as a global timeline without a validated clock-domain method. Causal
tokens can establish readiness order without pretending to measure cross-SM elapsed time.

**Validate/accept:** traces containing elementwise kernels, opaque kernels, copies and graph
replays retain all entries; graph nodes and host submissions are not double-counted as grids.
GPU smoke trace matches a known small launch/copy sequence. Diagnostic overhead is reported and
never substituted for unprofiled latency. Missing timeline evidence is explicit.
**Expected:** S: exposes hidden work; C: auditable mechanisms; P: diagnostic, may slow instrumented runs.

<a id="mb3-079"></a>

### MB3-079 — Establish the strongest equivalent compiled-baseline ladder

**Depends:** MB3-004, MB3-005, MB3-073, MB3-077, MB3-078. **Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/baselines.py`, `BENCH/fixtures.py`, `CPU/test_baseline_contract.py`,
`GTEST/test_baseline_ladder.py`.
**Read:** performance §§7–8; architecture §10; GPU audit §§3–5.

**Before → after:** historical or convenient baseline settings define the target → each workload
selects its fastest validated equivalent eager/compiled/captured route with capture evidence.

**Do:** test legal default, reduce-overhead and max-autotune settings on the pinned environment,
including the matched export route and an explicit stable-address graph control when legal.
Record unsupported modes, graph breaks, recompiles, warmup/autotuning and actual capture. Use
identical weights, state, masks, complete logits and output ownership. Separate borrowed-output
engine comparisons from owned-output API comparisons. Give baseline selection its own trials;
freeze its choice before final validation. Do not require a working full V3 model to run this task.

**Validate/accept:** deliberately unequal cache length/dtype/output lifetime is rejected; selected
baselines pass state/reference checks. A faster validated mode replaces a slower one, even if it
reduces V3's apparent speedup. Unavailable real weights are reported; tiny fixtures test the harness
but cannot replace the real-model baseline gate.
**Expected:** S: equivalent reference target; C: honest comparator; P: may lower reported V3 speedup.

<a id="mb3-080"></a>

### MB3-080 — Measure exact-shape body quality and composition penalties

**Depends:** MB3-021, MB3-048, MB3-049, MB3-077, MB3-078.
**Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/bench_shapes.py`, `CPU/test_shape_report.py`, `GTEST/test_shape_probe.py`.
**Read:** GPU audit §§6–7; kernel reuse §8; performance §§4–5.
**Optional inputs:** MB3-057 tensor-core body and MB3-069 mixed entry, each only when validated.
Neither blocks the first SIMT/lean-entry measurements.

**Before → after:** a generic GEMM benchmark guesses the bottleneck → exact model inventory
shapes have standalone, lean-persistent and mixed-entry measurements with numerical/resource data.

**Do:** run LINEAR_HOT and the actual selected roughly 2B inventory, retaining layout, M/N/K,
dtype, casts and working-set regime. Compare legal external kernels, K-parallel SIMT and the selected
tensor-core adapter. Include padding, conversion and descriptor/setup costs in the appropriate
scope. Report hot-cache microtests separately from representative streamed weights. The mixed-entry
mode accepts an artifact later supplied by MB3-069; its absence must not block initial lean probes.
Report shape costs and composition ratios; do not force a universal utilization threshold.

**Validate/accept:** numerical failure suppresses performance eligibility; changing source/layout/
toolchain invalidates cost reuse. Each row includes actual entry registers/shared memory/spills,
raw samples and exact competitor. If the library probe was rejected, report that candidate as
unavailable and keep SIMT probe results, without claiming MB3-057's working-body gate passed.
**Expected:** S: body eligibility evidence; C: locates the math/resource gap; P: measured input to selection, not an optimization itself.

<a id="mb3-081"></a>

### MB3-081 — Calibrate the costs that the joint planner actually uses

**Depends:** MB3-038, MB3-042, MB3-058, MB3-063, MB3-066, MB3-067, MB3-068,
MB3-077, MB3-078, MB3-080. **Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/calibrate.py`, `CPU/test_calibration_records.py`, `GTEST/test_calibration_probe.py`.
**Read:** hardware §7; pipeline §§3,6–7; performance §§2,4,6.

**Before → after:** scheduling uses guessed independent stage costs → a small, versioned
calibration set includes contention, publication, staging, tails and actual entry envelopes.

**Do:** measure required stage combinations with zero/one/two allowed lookahead slots where legal,
body/cohort concurrency and logical tile counts around worker-capacity boundaries. Measure actual
matrix-data loads, not only descriptors. Include shared-memory/accumulator residency effects and
publication consumption under the selected topology. Store intervals and working-set conditions
in CostRecord; unsupported/missing measurements stay unknown. Optional physical traffic counters
must name their scope; a logical byte count is not a measured DRAM byte count.

**Validate/accept:** matching keys recover costs; stale or differently contended keys do not.
Held-out combinations reveal model residuals instead of having their outcomes copied into the
prediction. Calibration uncertainty can force a tie/measurement request. No topology-discovery
framework or comprehensive GPU benchmark suite is needed.
**Expected:** S: unchanged work; C: justified planning inputs; P: enables better selection; calibration has setup cost.

<a id="mb3-082"></a>

### MB3-082 — Run controlled generated-block mechanism ablations

**Depends:** MB3-074, MB3-078, MB3-079, MB3-080, MB3-081. **Lane/review:** CPU+GPU, R3.
**Own:** `BENCH/ablations.py`, `CPU/test_ablation_matrix.py`, `GTEST/test_block_ablations.py`.
**Read:** performance §6, especially the A–F and 2×2 controls; pipeline §11.

**Before → after:** a pipeline looks promising in diagrams → measured controls establish which
fusion, data-movement, readiness and tail mechanisms improve a complete generated block.

**Do:** generate A strongest equivalent baseline; B selected bodies as separate kernels in a
matched graph; C optimized barrier-persistent control; D C plus fusion/layout; E D plus matrix-data
lookahead; F E plus readiness/continued reductions/tail-aware assignment. Also generate the 2×2
lookahead/readiness matrix with fixed body/fusion choices and a separate fusion comparison. If
continuation changes a body, name that confound. Use competent phase assignments, not an intentionally
imbalanced C. Connect diagnostic stage records and prove the claimed early work actually executes.

**Validate/accept:** all comparable variants pass identical state/numerical policies; every one
of head-ready attention, streamed MLP and weight lookahead has a correct experiment, including
negative outcomes. Publish raw unprofiled samples, separate diagnostic traces, resource deltas,
A/F and C/F plus interactions. Do not add incremental percentages into a predicted whole-model gain.
**Expected:** S: equivalent alternatives; C: causal evidence; P: measured non-launch gain or an explicit rejected mechanism.

<a id="mb3-083"></a>

### MB3-083 — Validate real-model logits, state and advancing generation

**Depends:** MB3-076, MB3-079. **Lane/review:** GPU, R3.
**Own:** `GTEST/test_model_campaign.py`, declared model/case manifests and correctness artifacts.
**Read:** IR §§8–9; performance §§7–8; GPU audit §4.

**Before → after:** tiny graphs and synthetic blocks pass → exact selected model revisions pass
the full output/state contract at declared short/long contexts and across advancing steps.

**Do:** pin the small and roughly 2B checkpoint/config revisions and actual adapter versions.
Use explicitly available weights; do not silently download them or substitute a different Gemma.
Construct valid prompt/cache state using the genuine reference prefill outside the one-step timed
contract, then test positions, mask variants and supported capacity edges. Check complete logits,
every updated state tensor, preserved unused regions, multiple advancing steps and retained outputs.
Apply the previously fixed tolerances and exceptional-value policy; report maximum/aggregate errors
and the worst-case location. Token agreement or cosine similarity alone is insufficient.

**Validate/accept:** every advertised model/dtype/context cell has a full pass or explicit failure/
unsupported/missing-artifact status. Repeat runs detect leaked state and stale events. No retuning
of tolerances after observing a failure. Noncontiguous/alias variants outside the guarded subset
must be rejected before device mutation, not miscomputed.
**Expected:** S: genuine decode correctness; C: precise coverage; P: eligibility gate only.

<a id="mb3-084"></a>

### MB3-084 — Publish the preregistered real-model performance scorecard

**Depends:** MB3-076, MB3-079, MB3-082, MB3-083. **Lane/review:** CPU+GPU, R2.
**Own:** `BENCH/scorecard.py`, `CPU/test_scorecard.py`, `GTEST/test_scorecard_smoke.py`, run reports.
**Read:** architecture §§10–11; performance §§6–9; this document §10.

**Before → after:** candidate trials and anecdotal ratios stand in for success → fresh held-out
measurements classify every declared cell, with external and non-launch comparisons separated.

**Do:** freeze the model/context/dtype matrix, baseline choice, candidate, numerical policy and
practical margin before final runs. Include a small and roughly 2B model at declared short/long
valid lengths; capacity alone is not context. Report device, host and user latency, raw samples,
uncertainty, setup/compile time, peak memory, fallback status and all-operation trace evidence.
Use the performance document's proposed 5% lower-median margin only if explicitly adopted in the
manifest; it is an acceptance policy, not a predicted gain. Distinguish A/F from C/F for the cells
where both are valid and retain all losses/unsupported cells in the report.

**Validate/accept:** generated reports cannot label an incorrect/fallback/unmeasured cell a strict
win. An uncertainty interval overlapping 1 is inconclusive evidence even if the point estimate is
favorable. The measured research objective is complete only under §10, not when the report exists.
**Expected:** S: no changed workload; C: defensible final result; P: actual measured outcome, including loss.

<a id="mb3-085"></a>

### MB3-085 — Add a separately tested torch.compile backend adapter

**Depends:** MB3-007, MB3-008, MB3-072, MB3-076. **Lane/review:** CPU+GPU, R2.
**Own:** `V3/compile_backend.py`, `CPU/test_compile_adapter.py`,
`GTEST/test_compile_adapter.py`.
**Read:** IR §§2,10; current public API; pinned PyTorch backend interface source.

**Before → after:** V3 accepts its explicit FX/export API only → an optional versioned
torch.compile entry adapts captured graphs to exactly the same compiler and diagnostics.

**Do:** use the selected version's documented/backend-source signature and return convention.
Bind example inputs/guards correctly; retain state as explicit tensor arguments/results at this
boundary. Test how graph breaks/recompilation are surfaced, and define strict full-step admission.
Do not infer that one compiled subgraph means the entire Python model is one megakernel. No global
Inductor monkey-patching or new scheduling path. Preserve the direct export/FX API as the primary
debugging route. This convenience adapter is not a prerequisite for the first exported-model win.

**Validate/accept:** same graph produces equivalent plans through both paths; guard changes cause
the expected recapture/rejection and correct binding, not stale pointer reuse. A deliberately split
graph cannot be reported as a strict full-step result. CPU mocks test the seam, not GPU execution.
**Expected:** S: preserved graph/state contract; C: familiar integration; P: no intrinsic GPU gain.

<a id="mb3-086"></a>

### MB3-086 — Make selected-source compilation work from an installed package

**Depends:** MB3-043, MB3-044, MB3-076. **Lane/review:** CPU+CT, R2.
**Own:** `pyproject.toml`, package-resource resolution in `V3/codegen/source.py`, `CPU/test_package_resources.py`,
`CTEST/test_installed_compile.py`.
**Read:** pyproject.toml; V3 source/dependency manifests; kernel reuse §§3–5,7.

**Before → after:** compilation may rely on source-checkout-relative CUDA paths → an installed
artifact can locate all owned selected sources and diagnose external headers explicitly.

**Do:** choose one package-resource layout and include its CUDA sources/support headers in build
metadata. Preserve third-party license notices and record external dependency/version requirements;
do not copy proprietary or incompatible-distribution material. Build a wheel using available tooling
and inspect its content; use an isolated temporary installation for import/resource smoke tests.
No implicit network download or hidden legacy developer path. Keep compiler-cache location outside
read-only installed package resources and retain content-based invalidation.

**Validate/accept:** installed CPU import needs no CUDA headers; selected source resolution works
without the repository cwd; absent external headers produce an actionable compile diagnostic.
Actual installed-source CUDA compilation is a CT gate, not established by listing wheel files.
**Expected:** S: identical sources/flags; C: reproducible distribution; P: no direct execution gain.

<a id="mb3-087"></a>

### MB3-087 — Automate independent regression lanes without hiding unavailable coverage

**Depends:** MB3-003, MB3-076, MB3-086. **Lane/review:** CPU+CT+GPU, R2.
**Own:** `tests/test_v3/run_lanes.py`, `CPU/test_lane_selection.py`, `ART/tasks/MB3-087/runbook.md`;
CI workflow only if this repository's chosen CI service is in scope.
**Read:** existing pytest configuration; §1.3–1.5; hardware §8.

**Before → after:** a broad pytest result can hide GPU skips and legacy breakage → independent
CPU, compilation, device-correctness and opt-in performance jobs emit explicit evidence vectors.

**Do:** keep CPU tests runnable without device initialization; CT tests pin supported toolchains;
GPU jobs record device/driver identity and run protocol/numerical/session regressions. Retain legacy
tests and report their pre-existing failures separately. Do not require model downloads for ordinary
unit jobs. Gate slow real-model/performance jobs on declared hardware/weights and explicit invocation.
Use machine-readable test results and artifact hashes; do not add a noisy universal latency gate to
shared CI. Performance comparisons require the controlled hardware/protocol of MB3-084.

**Validate/accept:** no-GPU execution still exercises CPU tests and visibly reports unavailable GPU
coverage; intentional failures fail the relevant lane. An installed-package job does not accidentally
import the checkout. No implementation task is marked GPU-validated merely because CI is green.
**Expected:** S: prevents regressions; C: honest automated status; P: guards measured work, not a new optimization.

<a id="mb3-088"></a>

### MB3-088 — Audit the final handoff against architecture and measured evidence

**Depends:** MB3-084, MB3-086, MB3-087. **Lane/review:** CPU, R2 plus outstanding R3 reviews.
**Own:** `ART/final/evidence.json`, `ART/final/report.md`, narrowly factual V3 documentation updates.
**Read:** every V3 document; this document §§9–11; actual task/run artifacts.

**Before → after:** many merged tasks suggest completion → a reader can reproduce the supported
result, locate every unrun gate and distinguish implemented architecture from successful research.

**Do:** verify task/source/plan/model/toolchain hashes, review records, license provenance and durable
artifact locations. Compare implementation with the three-representation architecture and all three
pipeline experiments. List unsupported semantics, fallbacks, lost candidates and optional unimplemented
work. Update docs with measured facts only; retain historical assertions as historical. Recommend one
bounded next experiment from observed bottlenecks, or hand off the completed target if it is met.

**Validate/accept:** a fresh reader can follow one real-model result back to source, correctness,
baseline, raw timings, trace, resources and decision. Missing evidence prevents the associated claim;
it need not prevent delivering an honest audit. “All planned files exist” never closes the research goal.
**Expected:** S: no semantic expansion; C: trustworthy handoff; P: no new gain, no concealed failure.

<a id="mb3-089"></a>

### MB3-089 — Conditional: add split-context attention with an explicit combine

**Depends:** MB3-025, MB3-055, MB3-065, MB3-082. **Lane/review:** CPU+CT+GPU, R3.
**Own:** `CUDA/bodies/decode_attention_split.cuh`, its footprints/selection rule, `CPU/test_attention_partials.py`,
`GTEST/test_split_context_attention.py`.
**Read:** pipeline §4; kernel reuse §9; performance §§4–6.

**Trigger:** long-context evidence shows the single-owner attention body limits latency or exposes
a tail, and the measured budget can pay for extra partials/combine. Not mandatory before MB3-084.

**Before → after:** each head scans its context on one owner → selected context chunks produce
mathematically compatible softmax statistics/weighted-value partials and an explicit final combine.

**Do:** define the partial representation and numerically stable reference combine first. Specify
empty/fully masked chunks and valid lengths exactly. Allocate/verify unique partials, publish only
complete statistics, and make final output readiness depend on all required partials. Add the extra
traffic, storage, combine and common-entry resources to candidate cost/admission. Keep the unsplit body.

**Validate/accept:** irregular context sizes, GQA, mask edges and chunk-order tests agree under the
fixed policy; partial omission/duplication fails verification. Compare full attention region latency,
not just faster partial kernels. A loss rejects this candidate without weakening the working path.
**Expected:** S: explicit equivalent reduction; C: visible combine cost; P: conditional parallelism/tail gain.

<a id="mb3-090"></a>

### MB3-090 — Conditional: add global partials for an otherwise infeasible continued reduction

**Depends:** MB3-026, MB3-060, MB3-065, MB3-067, MB3-082. **Lane/review:** CPU+CT+GPU, R3.
**Own:** `CUDA/bodies/linear_partials.cuh`, its finalizer/plan support, `CPU/test_global_reduction_plan.py`,
`GTEST/test_global_reduction.py`.
**Read:** pipeline §5; performance §6; IR §§7–9.

**Trigger:** actual resources or poor parallelism make the owner-held accumulator candidate
uncompetitive/inadmissible on an important measured shape; a bounded alternative has a credible budget.

**Before → after:** only one CTA may own all K updates for an output tile → a declared alternative
uses unique complete FP32 partials and a separate finalizer with full producer coverage.

**Do:** define K partitioning, partial dtype, summation order, final cast and output/residual semantics.
Never disguise partial output as final output or double-apply bias/residual. Model all partial writes,
reads, events, extra scratch and finalization; re-run lifetime/progress/resource verification. Keep
owner continuation available. Do not introduce unrestricted atomic accumulation or a general queue.

**Validate/accept:** missing/duplicate partitions, incomplete finalization and early scratch reuse
are rejected. GPU correctness includes tails and numerically difficult inputs. Compare entire region
and actual entry, including finalization; faster producers alone do not justify selection.
**Expected:** S: new explicit lowering, same supported semantics; C: honest ownership tradeoff;
P: conditional capacity/parallelism gain minus traffic and synchronization.

<a id="mb3-091"></a>

### MB3-091 — Conditional: add one evidence-selected hot-shape body tactic

**Depends:** MB3-023, MB3-069, MB3-080, MB3-082. **Lane/review:** CPU+CT+GPU, R3.
**Own:** one selected existing body header or `CUDA/bodies/linear_hot.cuh`, `CPU/test_hot_body_guard.py`,
`GTEST/test_hot_body_tactic.py`.
**Read:** kernel reuse §§3–9; GPU audit §§6–7; performance §4.
**Conditional input:** MB3-057 when adapting its tensor-core implementation; a SIMT tactic
does not depend on an unavailable tensor-core adapter.

**Trigger:** the score/shape ledger identifies a substantial uncovered math/composition gap and
specifies the exact shape/layout/dtype/resource cell. “More kernels might help” is insufficient.

**Before → after:** the selected portfolio is poor for one material cell → one guarded alternative
addresses its measured cause without replacing the compiler with model-specific code.

**Do:** name one tactic and expected mechanism, for example a different lane/tile mapping, a legal
tensor-core shape, or a consumer-aligned epilogue. Start from the existing reusable interface and
license-compatible source. Add only advertised capabilities; a better atomic tile does not become
PRELOADABLE automatically. Record all conversions, padding, staging and actual common-entry resources.

**Validate/accept:** guarded-in shapes pass standalone/composed correctness; nearby unsupported
shapes choose an existing body or diagnostic. Fresh whole-region samples test the hypothesis after
selection. If it loses, retain the evidence and do not grow the default portfolio needlessly.
**Expected:** S: guarded lowering alternative; C: small evidence-driven portfolio; P: shape-specific, measured only.

<a id="mb3-092"></a>

### MB3-092 — Conditional: share or recompute norm statistics with an explicit cost decision

**Depends:** MB3-015, MB3-051, MB3-061, MB3-080, MB3-082. **Lane/review:** CPU+CT+GPU, R3.
**Own:** `CUDA/bodies/norm_projection.cuh`, its composite/plan support, `CPU/test_norm_projection_composite.py`,
`GTEST/test_norm_projection_composite.py`.
**Read:** kernel reuse §9; IR §4; pipeline §§3,5; performance §6.

**Trigger:** measured norm/materialization or redundant projection-side normalization is on the
critical path, and a precise alternative can preserve the declared intermediate rounding behavior.

**Before → after:** norm and projection always use the existing materialization → a bounded
alternative shares statistics/values or deliberately recomputes them with accounted cost.

**Do:** write the exact reference including epsilon, reduction dtype, weight versus 1+weight and
cast placement. Specify which data is shared, where it lives, who publishes/consumes it and when it
can retire. If normalized values are recomputed per output tile, count the duplicate work and reads.
Do not algebraically move a rounded cast through the projection unless the fixed policy allows it.
Retain the unfused reference candidate and update footprints/storage/resource proofs.

**Validate/accept:** semantic-variant near misses stay distinct; difficult rounding inputs pass the
fixed policy. Compare complete norm+projection latency and traffic/resource evidence, not removed
operation count. Reject unprofitable recomputation or an unproved numerical rewrite.
**Expected:** S: explicit numerical equivalence; C: visible reuse/recompute tradeoff;
P: conditional movement/fusion gain; no assumption that every norm should fuse.

## 7. Scheduling work without creating another architecture project

### 7.1 Useful routes through the dependency graph

The arrows below identify deliverables, not a replacement for the card dependencies. In particular,
do not read a numeric range as a serial prerequisite chain.

| Desired next evidence | Route and stopping point |
|---|---|
| CPU-safe development | 001 → 002 → 003; record existing failures before changing imports |
| Semantic correctness | 004–012, then the specific matcher; expand back to its preserved reference |
| Earliest linear device probe | Contracts/body/footprint/barrier prerequisites → 041–049; compile only selected probe bodies |
| First body timing | 077/078 plus 080's prerequisites; use the SIMT probe without waiting for an optional library body |
| Honest reference target | 073 plus 077–079; this does not need a finished V3 model |
| First verified pipeline | 027–036 plus the relevant body/staging/publication tasks → 065 → 066, 067 or 068 |
| First non-launch evidence | Generated block 074 plus measurement prerequisites → 082 |
| Actual model outcome | 075/076, baseline 079, correctness 083 → held-out scorecard 084 |
| Reproducible handoff | Packaging/regression 086/087 plus scorecard → 088 |

Library compatibility/adaptation 056/057 is an early bounded experiment, not a requirement that
every probe wait for tensor-core success. Do not skip investigating that quality gap because SIMT
is easier; equally, do not prevent useful SIMT/pipeline evidence when the library experiment has a
documented blocker. A source-adaptation failure does not justify binary extraction or silently
changing the requested precision.

### 7.2 Ownership rules for future parallel assignment

These rules concern how the user may distribute work later; they do not require parallel execution.

| Surface | Integration rule |
|---|---|
| contracts.py, plan/model.py, plan/bodies.py | One interface owner accepts schema changes; callers do not add incompatible parallel records |
| Matchers 013–018 | Can be assigned separately after the shared dialect/facts exist; central registry wiring is reviewed once |
| footprints.py, dependencies.py, verify.py | Shared ownership: serialize overlapping patches or use isolated worktrees and explicit integration review |
| templates.py tasks 028,034–037 | Shared implementation surface even when logical dependencies allow concurrency; no simultaneous uncoordinated edits |
| Device bodies | Associate each body with its own descriptor/source manifest; serialize registry edits and never add a giant generated dispatch switch |
| source.py / ABI / driver | Contract changes propagate through source hash, argument packing and tests together; one integration owner |
| Session 070/071 | Serial ownership; storage rules precede invocation rules |
| BENCH modules | Shared WorkloadSpec/run-manifest schema stays fixed; each task owns its own report logic |

A dependent task can draft pure code against a reviewed interface while an upstream GPU run is
pending, but its handoff must retain the unresolved evidence. Do not enable the composed strict path
by treating two individually unvalidated patches as if they validated each other.

### 7.3 When to split a card, and how to stop scope growth

Split a parent before implementation if one of these applies:

1. It requires two unrelated new mechanisms with independently shippable behavior.
2. The selected upstream body has separate descriptor, tile and epilogue adaptations that cannot
   reasonably be reviewed together. Give each a reproducer and preserve the parent integration gate.
3. Supporting the second model requires a genuinely different state adapter, activation, norm or
   layout contract. Do not bury that addition inside “make the model work.”
4. A correctness failure points to an earlier contract bug. Stop the optimization patch, identify
   the violated contract and make a bounded repair task. Do not simultaneously redesign the planner.

Child format: `MB3-<parent>a`, parent ID, exact new outcome, dependencies, owned files, failing test,
before/after, steps, acceptance, evidence lane and expected S/C/P. Children do not automatically
change the parent architecture or numerical policy. A GPU-dependent task may end with a reviewed
patch and clearly pending device gates; it must not invent an emulation result to remove the wait.

## 8. Ready-to-use agent assignment and handoff

### 8.1 Copyable assignment packet

Replace the bracketed fields. This is a template for future implementation, not an instruction to
implement code during the current documentation change.

```text
Implement only [MB3-NNN and title] from MEGABAKE_V3_IMPLEMENTATION.md.

Checkout / branch / base: [actual values]
Dependency handoffs: [paths plus revisions/hashes]
Current GPU/toolchain access: [actual availability; unknown is allowed]
Owned files: [expand the card's path aliases; list any approved shared-file edits]
Required evidence gate for this assignment: [CPU patch / CT / GPU correctness / measured result]
Independent reviewer for R3 work: [assignment or explicitly pending]

Read sections 1–5 and 8 of the implementation handbook, this card, its specified V3 sections,
and the actual affected source/tests. Use section 9's examples when applicable. Do not read
only a summary of a required contract. Confirm dependency revisions still match the checkout.

Before editing, state the current behavior, desired behavior and the regression that separates
them. List any missing prerequisite. Implement one bounded outcome, its positive/negative tests
and diagnostics. Preserve existing user changes and the legacy path. No model downloads,
toolchain upgrades, numerical-policy changes or new runtime architecture without authorization.

Do not implement later tasks. If the card is not atomic in the actual checkout, propose named
child cards with tests rather than expanding scope silently. If a required interface changed,
identify the exact conflict and stop relying on the stale interface.

Run the tests appropriate to available hardware. Report actual commands, passed/failed/skipped
counts, warnings and artifact paths/hashes. Distinguish mocks, compilation, actual device tests
and performance measurements. Never count unavailable hardware as a passed gate.

Finish with a TaskHandoff and a short before/after report. Include semantic changes, clarity
improvements, measured performance or unmeasured hypothesis, unrun gates, review needs, and the
next task that is genuinely eligible. Stop at this acceptance boundary.
```

### 8.2 Minimal machine-readable handoff shape

This is an illustrative schema instance, not a completed task report. Replace all placeholders;
store actual commands/results, not the example's explanatory strings.

```json
{
  "schema_version": 1,
  "task_id": "MB3-NNN",
  "title": "<assigned title>",
  "base_revision": "<actual revision>",
  "patch_revision_or_digest": "<actual revision or scoped diff digest>",
  "dependency_handoffs": [
    {"task_id": "MB3-NNN", "path": "<durable path>", "digest": "<digest>"}
  ],
  "owned_files": [],
  "changed_files": [],
  "before": "<observable prior behavior>",
  "after": "<observable delivered behavior>",
  "semantic_effect": "<preserved contract or explicitly authorized change>",
  "clarity_effect": "<diagnostic/interface improvement>",
  "status": {
    "implementation": "not_started",
    "cpu_validation": "not_run",
    "cuda_compilation": "not_run",
    "gpu_correctness": "not_run",
    "gpu_performance": "not_measured",
    "disposition": "continue"
  },
  "commands": [],
  "artifacts": [],
  "performance": {
    "hypothesis": "<mechanism or no direct GPU gain>",
    "comparison_contract": null,
    "measured_speedup": null,
    "speedup_interval": null,
    "raw_samples_artifact": null
  },
  "review": {"required_level": "R2", "reviewer": null, "outcome": "pending"},
  "remaining_gates": [],
  "known_limitations": [],
  "next_eligible_tasks": []
}
```

Each actual command record includes working directory, command, environment identity, exit code,
test counts, timestamp and log location. Each artifact includes kind, durable path, content hash
and the graph/plan/source/run identities it supports. Omit secrets, raw authentication material
and model weights from command logs. An uncommitted scoped diff digest is acceptable for task
handoff; do not commit, push or open a PR unless that action has been authorized separately.

### 8.3 Human review checklist

- Can a reviewer identify the single changed behavior without reading every future task?
- Does at least one regression fail on the prior behavior and pass on the new one?
- Are near-miss/unsupported cases rejected rather than silently broadened?
- Are input/output/state bindings, dtypes, casts and alias assumptions explicit?
- Do generated-source, compiled-resource and numerical artifacts refer to the same candidate?
- Does an R3 patch include its participants, ordering, lifetime and forward-progress argument?
- Are performance costs tested against the entire relevant region and actual composed entry?
- Is any hardware-dependent claim being made from CPU mocks, estimated costs or skipped tests?
- Did the patch introduce unrelated architecture, dependencies or fallback behavior?
- Can the next owner use the interface without guessing what a token or pointer owns?

## 9. Worked contracts and adversarial examples

These examples disambiguate the task cards. They are logical/reference contracts, not copy-ready
CUDA or a substitute for the selected CUDA release's instruction-level rules.

### 9.1 FX/export normalization: preserve bindings, not just node count

Given a captured graph with lifted parameter W, user input x, input state s, a returned state
update s_new and user output y, the normalized program must retain all five roles and the exact
export signature. `s_new` is observable even if y has no data dependency on it. DCE may remove
an unused pure temporary; it may not remove the state update or alter alias semantics.

Regression recipe for MB3-007/009/011:

1. Construct a tiny program returning nested outputs and a fixed-slot state update. Preserve an
   independent original reference and fresh inputs/state for each execution.
2. Normalize using the pinned allowlist, intentionally testing a pass that returns a replacement
   graph instead of only mutating its argument. Verify the replacement is actually propagated.
3. Execute both; compare output tree, lifted bindings, state updates, untouched state and guards.
4. Rename every FX node and repeat. Semantics must not depend on names.
5. Add a dead pure branch and a live state-only branch. Only the first may disappear.
6. Change one captured constraint or alias relationship; either preserve it correctly or diagnose
   unsupported input. Do not regenerate a guessed signature from placeholder order alone.

### 9.2 Exact footprints for a tail linear

For LINEAR_TINY, x has logical shape `[1,33]`, W is logically `[17,33]`, and
`y[0,n] = sum_k x[0,k] * W[n,k]` under the declared numerical policy. An update with output
rows `[8,16)` and K chunk `[16,32)` reads precisely x's K chunk and those W rows/K coordinates;
it updates the owner's accumulator for rows `[8,16)`. It does not yet write final y.

The last output tile `[16,17)` and K chunk `[32,33)` cannot be dropped. A transposed physical
view changes address calculation, not the logical operation. For the initial affine case:

```text
element address = base + storage_offset + sum(index[d] * stride[d])
byte address    = element address interpreted with the correct element-byte size
```

In implementation, keep pointer/byte arithmetic typed; the formula is not permission to add an
element offset to a byte pointer. Reject unsupported overlap/negative strides or represent them
correctly. Tests need both a contiguous layout and a proven noncontiguous view, with sentinels in
unused storage. A bounding address range is conservative overlap evidence, not exact disjointness
for arbitrary strided regions.

### 9.3 Streamed gate/down: publication is not reduction completion

Let `g = Linear_g(x)`, `u = Linear_u(x)`, `h = phi(g) * u`, `y = Linear_down(h)` with every
reference cast retained. The GATE_TINY intermediate dimension is 65. For one down-output owner:

```text
begin owner accumulator once
for chunks [0,16), [16,32), [32,48), [48,64), [64,65) in the selected legal order:
    wait for the whole h chunk's exact producers
    update the same accumulator from that chunk and matching down-weight K region
finalize only after all five distinct chunks
apply final bias/cast/residual at its reference location exactly once
publish the complete output tile
```

The first update can start before later h chunks exist. The final output cannot. Tests must
reject receiving chunk `[16,32)` twice even if the total number of updates is five. Counts alone
are not coverage. Partial sums must not be rounded to BF16 after each chunk unless the policy
explicitly calls for that behavior. Floating-point reassociation remains policy-controlled; the
symbolic identity of a sum does not establish bitwise equality to every vendor reduction tree.

For a cross-CTA producer/owner split, h still needs an exchange location. Fusion can remove g/u
materialization while h remains in global memory. Do not subtract h's traffic merely because the
diagram draws a fused MLP box. Multiple output owners consume the same h chunks; retain them until
all required readers finish or the conservative verified region join permits reuse.

### 9.4 Head readiness in grouped-query attention

For Hq=4 and Hkv=2 in ATTENTION_TINY's ordinary contiguous group mapping, query heads 0/1 use KV
head 0 and query heads 2/3 use KV head 1. This mapping is a fixture contract, not an inferred rule
for every graph; MB3-025 reads the actual semantic mapping.

An attention consumer for query head 0 needs its own positional Q, valid cache prefix for KV head
0, current K/V writes for KV head 0 when included, mask/scale/position metadata and all required
visibility tokens. It does not require query head 3. It also must not run merely because Q head 0
is ready while V's current slot is still outstanding.

Regressions: delay unrelated group 1 and permit group 0 to start; delay group 0 V and forbid that
start; poison unused cache slots beyond valid length; test position 0 and the last supported slot.
If a producer tile spans several readiness groups, its actual completion granularity may force a
coarser event. Split it only through a declared body capability/footprint change, not a fictional
early-ready annotation. The first pipeline retains a full dependency before output projection.

### 9.5 Online attention and the optional split-context combine

For a valid score tile s with values V, a reference online update tracks running maximum m,
normalizer l and unnormalized weighted-value vector a:

```text
m_new = max(m, max(s_valid))
alpha = exp(m - m_new)                  # special-case no previous valid values
p     = exp(s_valid - m_new)
l_new = alpha * l + sum(p)
a_new = alpha * a + sum(p[:, None] * V_valid, axis=context)
output = a / l                          # only under the defined nonempty policy
```

Mask/scale precede the update as the reference requires. A tile with no valid values must not
produce `exp(-inf - -inf)`. Empty/all-masked outputs follow the captured reference/policy, not an
assumed universal behavior. Accumulator type, exponential approximation and final cast belong to
NumericalPolicy and BodySpec and need actual device comparison.

For MB3-089, partials use compatible `(m_j, l_j, a_j)` values. The combine rescales each valid
partial to the global maximum before summing l/a. Simply adding independently normalized outputs
is wrong. Empty partials need an explicit validity contract. The combine's storage, synchronization
and final cast are part of the candidate's full cost.

### 9.6 Data readiness, source retirement and destination visibility

For a weight staging slot, this is a necessary logical sequence:

```text
reserve free slot/phase
  -> issue real matrix-data preload when addresses and source lifetime permit
  -> receive declared load-complete token
  -> compute only after load-complete AND activation-ready
  -> all readers of the slot retire
  -> release slot/phase
```

If the current body internally uses two shared-memory stages, next-task staging is additional
storage unless a proven lifetime permits reuse. “Double buffered” does not imply a free third
slot. Compiler estimates and actual common-entry admission must include all simultaneous storage
and owner-held accumulators.

For an async output store, source-read retirement can allow reuse of the source staging area;
it does not necessarily establish that a different CTA may read the destination. Destination
visibility must precede publication. Conversely, a consumer's acquisition of destination readiness
does not mean every later reader is done with that destination allocation. These are separate
lifetime questions. Initial ordinary-store bodies still need correct writer aggregation before
publication; omitting async stores does not eliminate cross-thread memory-order obligations.

### 9.7 Counter publication: necessary proof obligations

The first inter-CTA protocol uses one invocation-local counter per produced event. For an event
with P required producer contributions:

1. Verify `P > 0`, exact producer identities and count-width bounds on the CPU.
2. In the owned cooperative grid, initialize counters to zero; every worker participates in the
   uniform initialization barrier before publishing/polling. Events are not reused in that invocation.
3. Each producer finishes all required writers and applicable async destination-visibility work;
   the designated publisher has the required ordering from its participating threads.
4. The publisher performs the reviewed device-scope acquire-release atomic RMW exactly once for
   its contribution. The chosen protocol must establish the transitive ordering for all producers.
5. The consumer acquires the completion value, then correctly shares readiness with its own CTA
   before nonpublishing/nonpolling threads consume the data.

This is a review checklist, not a complete C++ implementation. An atomic increment by one thread
does not automatically order unrelated writes by every other thread. Independent R3 review must
examine the selected CUDA primitives, participation, proxy ordering where applicable and actual
compiled path. CPU interleaving tests validate the abstract protocol only.

### 9.8 An acyclic data DAG that still deadlocks

There are four actions with only these data dependencies: A produces B's input; C produces D's
input. The data DAG is acyclic. Assign worker instruction order as follows:

```text
worker 0: D, then A
worker 1: B, then C

combined precedence edges:
D -> A   (worker 0 order)
A -> B   (data)
B -> C   (worker 1 order)
C -> D   (data)
```

Worker 0 waits for C; worker 1 waits for A; neither reaches its producer. The combined graph has
a cycle. Checking only the tensor DAG would accept this broken plan. A legal assignment such as
`worker 0: A,D` and `worker 1: C,B` removes this particular cycle, but still needs storage-credit,
collective participation and actual residency checks.

Another failure: all active consumers hold the only staging slots while waiting for producers
that need those slots. Assigning enough CTAs does not fix resource-credit deadlock. A body with an
undocumented grid collective cannot safely appear under a divergent worker branch. Ordinary
oversubscribed launches are not an alternative to the cooperative residency gate.

### 9.9 Partial-order storage reuse

Suppose producer P fills X; consumers C0 and C1 read X. C0 finishes early; C1 has a delayed access
or an async source read. Reusing X after C0 alone is invalid. A topological listing that happens
to place C1 earlier is not a completion proof. The allocator requires a happens-before path from
the end of every relevant access to the next writer, or treats those lifetimes as overlapping.

For initial cross-CTA chunks, prefer distinct addresses within a region and one verified region
join before reuse. For shared staging, a static slot can be reused with a proven per-slot phase
protocol. Outputs returned with owned lifetime cannot be overlaid on scratch needed by the next
call. Tests should retain output A, run call B with different inputs, then verify A is unchanged.

### 9.10 Cost model: two illustrative acceptance calculations

The examples use invented toy durations and do not forecast GPU/model speedups.

For four chunks with a 5 us producer and 3 us consumer per chunk on independently available
resources, a barrier schedule takes `4*5 + 4*3 = 32 us`. Ideal pipelining takes
`5 + 3 + 3*max(5,3) = 23 us`. With 2 us of additional coordination the example becomes 25 us,
giving `32/25 = 1.28×` versus that control. If both stages contend for one saturated bandwidth
resource, their isolated durations do not justify 23 us; the estimator must use a shared-domain
constraint or measured joint costs. Actual common-entry resource changes also alter the durations.

For a tail example with four workers and five equal independent tiles taking t each, two waves
cost approximately 2t if overhead is ignored. Changing the mapping to eight half-duration tiles
could approach t only if it preserves total useful work, numerical semantics, resources and the
assumed half-duration cost. With per-tile overhead h it instead costs about `2*(t/2+h)`; different
body efficiency or extra memory traffic can remove the gain. MB3-037/039 must keep negative cases,
not assume smaller tiles always improve utilization.

For real performance expectations, fill measured body/resource/overlap records and the model-size
calculations in the performance document. Do not sum the expected P fields of task cards. Several
tasks enable the same eventual optimization; counting each as a separate gain would double-count it.

## 10. Exit gates: implementation progress versus research success

The earlier E0–E5 labels remain useful summaries, but no longer substitute for work orders.

| Gate | Atomic-task evidence | Required exit |
|---|---|---|
| E0: offline compiler contracts | Relevant 001–041, especially 007–040 | CPU reference round-trips, coverage/state/lifetime/progress checks, concrete counterexamples; no CUDA correctness claim |
| E1: trustworthy target and baseline | 042,046/047,073,077–079 | Actual target/admission and equivalent baseline/capture/timing evidence |
| E2: useful owned math | 048–062 and 080, applicable body subset | Exact-shape numerical/resource/latency records standalone and in a lean entry; library outcome reported |
| E3: non-launch pipeline evidence | 063–074 and 081/082 | All three pipeline experiments correct; generated block, diagnostic early-work evidence, controlled non-launch comparisons |
| E4: real-model result | 075/076 and 083/084 | Supported small and roughly 2B cached workloads, state correctness, one-grid trace, held-out baseline and mechanism results |
| E5: evidence-led expansion | Triggered 089–092 or separately approved child cards | One bounded gap addressed and remeasured; no automatic backend/runtime expansion |

These gates can progress in overlapping routes. E2 probes do not wait for all of E0's unrelated
frontend features. E3 work does not wait for every standalone body to beat its external competitor.
Final E4 correctness and performance, however, cannot bypass the full compiler/session proofs.

### 10.1 Freeze the target matrix before final timing

The first manifest includes at least one small model and one roughly 2B model, each with a declared
short and long valid context, checkpoint revision, actual supported dtype and cache capacity. Use
128/2048 as initial fixture candidates only if they are legal and representative for the chosen
model/target. No claimed coverage follows from parameter count alone. Fix the required win cells
before final timing and report the entire matrix, including cells not targeted for a win.

At minimum, a claimed first small-plus-2B success needs a preregistered winning cell for each model,
not a small-model win combined with an unmeasured larger model. A claim covering both context
ranges requires wins in those corresponding cells; do not generalize one successful context to
the whole model. The first evidence campaign includes both ranges regardless of outcome.

A strict successful cell requires all of:

1. Complete logits/state/output-ownership correctness under the fixed policy.
2. One admitted owned compute grid for the complete declared step, with no hidden computation
   delegated to fallback. Necessary copies/setup are separately disclosed and timed appropriately.
3. A win over the strongest validated equivalent baseline on fresh trials under the predefined
   practical/uncertainty rule; point estimates alone are insufficient for a robust-win claim.
4. Exact workload/target/source/plan identity and raw evidence sufficient to reproduce the result.

The revised research objective additionally requires measured non-launch improvement over a
matched competent phase control, explained by mechanism ablations. All three first-class pipelines
receive correct experiments; they need not all be profitable in every final cell. A launch-only
strict win remains a valid narrower result but does not meet this additional objective.

### 10.2 Failure tells us what to do next

| Observation | Bounded response; not permission for a rewrite |
|---|---|
| Original/normalized reference differs | Stop downstream performance work on that graph; isolate capture/pass/effect bug in 007–011 |
| FatOp expansion differs | Fix matcher guard/reference; keep original region, not a guessed semantic match |
| Abstract plan fails coverage/lifetime/progress | Reject it; repair the responsible contract/template with its counterexample |
| CPU model passes, CUDA protocol fails | Keep GPU gate failed; investigate primitive/codegen/participation, never loosen the abstract contract |
| Standalone body is poor | Use 080's exact shape evidence to trigger 091 or revise the selected adapter |
| Standalone good, composed poor | Inspect 069 resources and 081 concurrency; consider a leaner compatible body/common envelope |
| Prefetch issues loads but does not help | Inspect contention, staging and actual overlap; keep zero-lookahead candidate |
| Streamed MLP loses through accumulator demand | Adjust bounded active owners/tiles; trigger 090 only if measured budget supports it |
| Long-context attention dominates | Investigate 055 quality; trigger 089 if parallel partial/combine accounting is favorable |
| Norm/recompute work dominates | Trigger 092 only with exact cast/reference and region budget |
| Small model wins, 2B loses | Report both; distinguish body-rate/composition/context limits from launch overhead |
| Final confidence interval is inconclusive | Collect appropriate fresh independent blocks or report inconclusive; do not cherry-pick trials |
| No legal strict candidate | Report unsupported/inadmissible; normal fallback remains explicitly non-strict |

There is no finite document that proves future measured speedup for all models. This plan makes
progress observable and failures local. It deliberately allows a task/report to finish honestly
without declaring the overall performance target achieved.

## 11. Traceability to every V3 document

| Design authority | Concrete tasks and acceptance hooks |
|---|---|
| [README](MEGABAKE_V3_README.md) | 004 workload scope; 076 generic supported-FX route; 082 non-launch evidence; 084/088 honest outcome |
| [Architecture](MEGABAKE_V3_ARCHITECTURE.md) | 007–012 three representations; 019/034–040 joint planning; 063–076 specialized execution/session; 084 first result |
| [IR and reuse](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) | 007–021 signatures/facts/FatOps/state/LayerSummary; 022–033 coverage, lifetimes and verification; 085 optional backend adapter |
| [Pipelining and scheduling](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) | 027–037 readiness/ownership/tails; 058–068 stage contracts/protocol/generated pipelines; 069 resource composition; 081/082 costs and evidence |
| [Dataflow diagrams](MEGABAKE_V3_DATAFLOW_DIAGRAM.md) | 007–012 frontend handoff; 040 selection; 043/064 code generation; 070–076 invocation; 078/082 causal trace |
| [Hardware model](MEGABAKE_V3_HARDWARE_MODEL.md) | 038 cost provenance; 041/042 TargetProfile; 045/047 actual function resources/admission; 058 target-gated stages; 081 targeted calibration |
| [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md) | 023 body contract; 048–057 exact math/vector/attention/source adaptation; 058–062 stage/continuation/fusion capabilities; 069 composition |
| [GPU reanalysis](MEGABAKE_V3_GPU_REANALYSIS.md) | 001 evidence inventory; 049 skinny mapping; 069 common resources; 073 actual cached step; 077–080 corrected timing/counting/baselines/shapes |
| [Performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) | 004 fixed contract; 038/039 model and uncertainty; 077–084 raw timings, baseline ladder, calibration, A–F/2×2 ablations, fresh model scorecard |
| [Research and decisions](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md) | 001 historical provenance; 009 pinned normalization; 056 source/license compatibility; 088 decision audit; 089–092 bounded evidence-led alternatives |
| Implementation handbook | 001–092 task ownership, before/after, procedures, tests, S/C/P expectations, evidence vectors and handoff boundaries |

The following are intentionally **not** silently added to implementation scope: private CUDA-binary
embedding, generic learned hardware-topology discovery, a new Layer IR for config labels alone,
training/backward, paged serving/continuous batching, recurrent/hybrid-attention device bodies,
quantization, distributed execution, multi-GPU or a universal work-stealing scheduler. A real
requirement for one receives its own semantic contract and separately authorized task series.

## 12. What can be delivered before GPU access returns

CPU-only work can establish the import boundary, test lanes, workload/numerical records, capture,
normalization, facts/effects, semantic matchers, reference expansions, plan/footprint/storage/progress
checks, bounded simulation, candidate search, offline target schema and source-generation tests.
Pure portions of runtime argument packing, statistics, report validation and packaging can also be
tested. Toolchain probes may compile if the selected toolkit is actually available; never assume
the current machine can run nvcc merely because source files exist.

Device bodies and the launcher may be drafted with compile checks where possible, but their GPU
correctness, residency, synchronization, contention and latency remain unvalidated. Mark this in
the evidence vector and retain independent R3 review gates. CPU simulators do not reproduce the
CUDA memory model, compiler register allocation, real async engines or whole-entry occupancy.

When access returns, use the early routes in §7: establish the actual target and baseline, test
lean math and synchronization, then the generated mechanism probes and real-model campaign. Do
not spend the first GPU session implementing another IR or downloading unapproved model variants.
Do not fill performance tables with predictions formatted as measurements.

This document is a plan for delivering and testing the compiler. The documentation expansion
itself implements none of these tasks and establishes no new performance numbers.
