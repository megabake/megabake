# MegaBake V2.3 Implementation Ledger

Status: executable implementation plan. This document converts the normative architecture and the
H200 evidence into ordered, independently verifiable work. It is subordinate to, and must remain
consistent with:

- [`MEGABAKE_V2_ARCHITECTURE.md`](./MEGABAKE_V2_ARCHITECTURE.md)
- [`MEGABAKE_V2_DATAFLOW_DIAGRAM.md`](./MEGABAKE_V2_DATAFLOW_DIAGRAM.md)
- [`MEGABAKE_V2_IR_AND_REUSE_PLAN.md`](./MEGABAKE_V2_IR_AND_REUSE_PLAN.md)
- [`MEGABAKE_V2_GPU_REANALYSIS.md`](./MEGABAKE_V2_GPU_REANALYSIS.md)
- [`grace_hack.md`](./grace_hack.md)

If this ledger and a normative document disagree, stop, record the conflict, repair both documents,
and only then continue implementation. This ledger does not authorize private vendor code as a
production dependency, hidden work, relaxed numerical policy, or counting a hybrid graph as a
strict one-grid win.

## 1. Objective and exact completion claim

MegaBake V2.3 is complete only when a generic semantic/traffic compiler emits target-specialized
artifacts and, on **every target in the declared release hardware matrix**, its strict one-grid path
beats the strongest equivalent `torch.compile` mode on **at least 80% of the declared model/bucket
cells for that target**. The threshold is a release default, recorded before tuning, not selected
after results are known.

The statement is intentionally narrower and more falsifiable than “all hardware in existence”:

```text
hardware requirement = 100% of targets declared in objective_manifest.json are measured
model requirement    = strict wins / eligible declared cells >= 0.80 on every target
strict p50 win       = MegaBake p50 <= 0.95 * strongest-equivalent-baseline p50
strict p99 gate      = MegaBake p99 <= strongest-equivalent-baseline p99
correctness gate     = 100% of executed cells pass their declared numerical/quality policy
strict grid gate     = exactly one owned compute grid; hybrid/vendor grids never count
```

The initial release matrix must include the available H200/SM90 environment and at least one real
non-Hopper CUDA target, initially A100/SM80. Adding a target to the manifest makes it mandatory for
the release. Non-CUDA support is supplied through the same target-backend contract, but no such
performance claim is made until a real backend and hardware cell complete this ledger.

“Most models” initially means transformer inference model families declared in the manifest—not
general training and not every PyTorch program. The required cohort must cover, at minimum:

- dense decoder-only models with MHA and GQA/MQA;
- at least six independently implemented model families, including SmolLM, Llama, Gemma, Qwen,
  Mistral, and Phi-class architectures;
- decode batch/token buckets `M=1`, `M=2`, and `M=4`;
- short, medium, and long KV sequence buckets;
- at least one prefill bucket after decode wins;
- reference FP16/BF16 as the mandatory strict score; quantized and serving scores remain separate.

Unsupported semantics remain scorecard losses unless the manifest marked the cell ineligible before
implementation began and recorded an objective-independent reason. Model aliases and repeated sizes
of the same architecture do not inflate the denominator.

## 2. Measured starting point and required movement

The first laboratory cell is SmolLM2-135M, batch 1, sequence 1, decode, on H200 MIG `3g.71gb` with
60 visible SMs. These values are the starting evidence, not portable constants:

| Metric | Before V2.3 | Required after H200 vertical slice |
|---|---:|---:|
| MegaBake complete invocation p50 | `4,845.5 us` | `<= 0.95 * current strongest baseline`; with `1,781.7 us`, `<= 1,692.6 us` |
| MegaBake owned compute-grid time | `4,748.7 us` | low enough for the complete invocation gate; every residual interval attributed |
| Ordinary `torch.compile` p50 | `5,691.4 us` | diagnostic only; never the sole acceptance baseline |
| Strongest measured compiled p50 | `1,781.7 us` | remeasured and stored for every environment |
| Strongest measured compiled best | `1,737.9 us` | diagnostic; p50/p99 distributions remain authoritative |
| Strong-baseline summed CUDA nodes | `1,026.2 us` | region/body oracle only, not end-to-end acceptance |
| MegaBake registers/thread | `184` | candidate-specific; no unexplained resource inflation |
| MegaBake stack/thread | `1,072 B` | `0 B` expected on hot skinny paths; otherwise explicitly justified and faster |
| MegaBake dynamic SMEM/CTA | `225,280 B` | exact reachable maximum only; no universal maximum reservation |
| MegaBake occupancy | `12.5%` | mapping-specific; sufficient residency must be demonstrated by performance |
| MegaBake compute throughput | `6.42%` | diagnostic, not an isolated acceptance metric |
| MegaBake DRAM throughput | `2.33%` | improve toward calibrated shape/model-context bandwidth |
| Reported MegaBake maximum difference | `0.195312` | must pass the frozen reference policy, not merely be printed |

The H200 vertical slice therefore needs approximately a `2.86x` improvement over the current
MegaBake p50 to clear a 5% win margin if the baseline remains `1,781.7 us`. If the baseline changes,
the formula—not the historical number—decides acceptance.

## 3. How an implementation agent uses this ledger

### 3.1 Checkbox and evidence protocol

Every milestone and atomic task begins unchecked. An agent may change `[ ]` to `[x]` only when:

1. all dependencies are `[x]`;
2. the named artifact exists in the repository;
3. the stated verification command succeeds in the recorded environment;
4. the before and after measurements are stored, even when the result is a regression;
5. the stated gate passes; and
6. `benchmarks/v2/evidence/<TASK_ID>.json` records commands, repository revision and worktree hash,
   target fingerprint, raw artifact paths, before/after metrics, and a short conclusion.

`[!]` means blocked and must include a blocker record. It never means done. A performance task cannot
be completed using an estimate, another target's result, a task-local timer in place of invocation
latency, or a filtered profiler trace.

### 3.2 Required evidence schema

Every task evidence file must validate against `benchmarks/v2/schema/task_evidence.schema.json` and
contain:

```text
task_id, status, base_revision, worktree_diff_hash, timestamp
commands[] and exit_codes[]
input_artifact_hashes[] and output_artifact_hashes[]
TargetFingerprint or "host-only"
WorkloadContract and WorkloadBucketKey when applicable
before_metrics{}, after_metrics{}, acceptance_gate{}, pass
raw_logs[], profiles[], result_json[]
notes and follow_up_task_ids[]
```

Do not paste large profiler output into this ledger. Link the immutable evidence file.

### 3.3 Performance measurement protocol

Unless a task declares a stricter protocol:

- lock model, inputs, numerical policy, invocation contract, and target fingerprint;
- warm up at least 50 iterations;
- collect at least 200 timed iterations in each of 5 fresh processes;
- record CPU wall, GPU events, p50/p95/p99, best, standard deviation, and 95% bootstrap confidence
  interval;
- synchronize only at equivalent boundaries for all candidates;
- report every kernel, copy, memset, allocation, graph node, and host submission;
- capture clocks, power policy, MIG/partition identity, driver, toolkit, PyTorch, library builds, and
  profiler state;
- randomize candidate order when thermal or clock drift could bias a tournament;
- accept a strict p50 win only when the lower confidence bound on speedup exceeds `1.00` and the
  configured 5% nominal margin is met.

### 3.4 Numerical and quality protocol

The manifest freezes tolerances before performance tuning. Defaults are:

- FP32 primitive oracle: `rtol=1e-4`, `atol=1e-5`;
- FP16/BF16 primitive/composite oracle: `rtol=1e-2`, `atol=1e-2`, with accumulation rules recorded;
- full-model reference inference: all values finite, cosine similarity `>=0.9999`, normalized RMSE
  `<=1e-3`, and 100% next-token agreement on the fixed validation prompt set;
- no candidate may be worse than the equivalent compiled baseline on a declared error metric by
  more than 10% unless the numerical policy explicitly permits it;
- FP8/W8/W4 require their own model-quality suite and never satisfy a reference-precision cell.

If a model's natural scale makes a default inappropriate, change the manifest before tuning that
model and explain the invariant metric. Printing `MaxDiff` without a pass/fail decision is not a
correctness gate.

### 3.5 Global search budget

The initial bounded tournament budget per exact hot shape is:

```text
generate <= 256 legal configurations per implementation family
compile/resource-inspect <= 64 per family after analytical pruning
standalone timing <= 16 per family
embedded-worker timing <= 4 per family
retain >= 1 legal candidate per family through embedded-worker timing
whole-entry beam width <= 8
default wall budget <= 60 minutes per hot shape and <= 6 hours per model/target bucket
```

The manifest may override these numbers before a run. Exhausting a budget produces a measured best-
known candidate, not a claim of global optimality. Equivalent generated bodies are deduplicated.

### 3.6 Global invariants

- Target-neutral facts and regions contain no CUDA product or instruction policy.
- H200 measurements may order but never select an A100 candidate.
- Exactly one owned compute grid is required for a strict result.
- The strongest equivalent low-overhead `torch.compile` mode gates every cell.
- The release entry contains winners only; losing candidate bodies remain outside it.
- Logical work count and resident workers are independently selected.
- Hot skinny paths have no unexplained spills, stack, or local-memory traffic.
- Runtime binds and launches frozen artifacts; normal inference does not tune or compile.
- Hybrid results, quantized results, and serving-throughput results have separate scorecards.

## 4. Milestone dependency map

| Done | Milestone | Depends on | Exit contribution |
|---|---|---|---|
| [ ] | M0 — objective and measurement truth | none | trustworthy denominator, baselines, correctness, evidence |
| [ ] | M1 — V2 compiler contracts and skeleton | M0 | target-neutral compiler objects exist and serialize |
| [ ] | M2 — target backends, calibration, and baseline suite | M1 | exact-device facts and strong baselines are reproducible |
| [ ] | M3 — generated entry, phase executor, and invocation path | M2 | lean one-grid artifacts execute with measured resources |
| [ ] | M4 — four-family linear tournament | M3 | hot GEMMs are selected empirically, cuBLASDx first-class |
| [ ] | M5 — H200 strict vertical slice | M4 | first strict real-model win |
| [ ] | M6 — SM80/A100 portability proof | M5 | same compiler retargets and wins independently |
| [ ] | M7 — traffic-removing transformer composites | M6 | full blocks remove HBM traffic and joins |
| [ ] | M8 — precision and prepacking portfolios | M7 | separate FP8/W8/W4 quality-gated artifacts |
| [ ] | M9 — hybrid graph product fallback | M7 | unsupported strict cells have fast honest fallback |
| [ ] | M10 — attention, KV cache, prefill, and decode control | M7 | sequence buckets and full decode semantics are competitive |
| [ ] | M11 — continuous batching and serving throughput | M8, M10 | weights are reused across tokens under an SLO |
| [ ] | M12 — model/hardware coverage and release audit | M5–M11 | objective matrix passes and release is reproducible |

Milestones are sequential unless the dependency table explicitly permits overlap. Performance work
must not skip an earlier truth or correctness gate.

## 5. M0 — Objective and measurement truth

**Goal:** make every later speedup, loss, and coverage claim reproducible and impossible to game.

**Before:** default harnesses compare mainly with ordinary `torch.compile`, maximum differences are
printed rather than gated, the `1,781.7 us` result has no persisted machine-readable artifact, the
coverage denominator is undefined, and the README presents older weak-baseline claims.

**After:** one versioned manifest defines models, hardware, buckets, policies, baselines, thresholds,
statistics, and search budgets; every run emits validated raw results and scorecard cells.

**Milestone performance gate:** harness overhead measured with an empty callable is `<=2 us` or is
subtracted identically from all compared paths; repeated baseline p50 varies by `<=3%` across the
five-process protocol unless the variance is explained.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M0.1 | — | Add `benchmarks/v2/objective_manifest.json`, its schema, a minimal task-evidence schema, and a bootstrap validator; define release targets, model-family weights, buckets, policies, 80% threshold, and 5% margin. | Denominator and evidence mechanism exist only in prose. | Versioned manifest, schemas, bootstrap validator, and parser tests. | M0.1 evidence validates; 100% required fields enforced; changing denominator changes manifest hash. |
| [ ] | M0.2 | M0.1 | Complete the task/result schemas, raw-run convention, and full evidence validator; revalidate M0.1 with the final schema. | Only bootstrap evidence exists. | `benchmarks/v2/schema/` and `verify_evidence.py`. | M0.1 plus valid fixtures pass; every missing required field fails. Runtime delta `0 us`. |
| [ ] | M0.3 | M0.1 | Extend benchmark CLI with eager, ordinary Inductor, export+Inductor, `reduce-overhead`, static-address/CUDA-Graph, strict MegaBake, and hybrid modes. | Only ordinary compiled path is standard. | `benchmarks/v2/run.py` emits one equivalence-keyed result format. | 100% of legal modes run the same inputs/output contract; unsupported modes are explicit, never omitted. |
| [ ] | M0.4 | M0.3 | Implement five-process warmup/distribution runner and bootstrap comparison. | Single medians can hide noise. | Raw samples plus p50/p95/p99/CI in JSON. | Synthetic timing tests reproduce known ordering; runner overhead `<=2 us` or symmetric. |
| [ ] | M0.5 | M0.3 | Implement unfiltered CPU/GPU operation timeline and counts. | “One kernel” can hide copies or memsets. | Trace records kernels, grids, copies, memsets, allocations, submissions, and gaps. | Injected copy/memset tests are detected 100%; no filter changes totals. |
| [ ] | M0.6 | M0.1 | Implement reference numerical and model-quality policies with pass/fail output. | Harness prints `MaxDiff` only. | `quality_and_tolerance_policy` evaluator plus prompt corpus hashes. | All intentional corruptions fail; eager and compiled references pass 100%. |
| [ ] | M0.7 | M0.3–M0.6 | Reproduce and persist the H200 strong baseline and current MegaBake result. | `1,781.7 us` exists only in prose. | Immutable raw samples/profile/result under exact fingerprint. | Strongest legal mode selected automatically; numbers within 5% of prior run or discrepancy explained. |
| [ ] | M0.8 | M0.7 | Correct README benchmark language and link current V2 status/scorecard. | README claims zero launch overhead and 3.4x under a weaker baseline. | README distinguishes current implementation, strict objective, and historical results. | No speedup claim lacks result link, equivalence key, target, and baseline mode. Runtime delta `0 us`. |
| [ ] | M0.9 | M0.1 | Add an initially failing `SingleGridScorecard` generator. | No scorecard artifact exists. | `benchmarks/v2/scorecard.json` lists every declared cell including losses/unrun cells. | Cell count is 100% of the manifest Cartesian product after declared exclusions; silent drops = 0. |
| [ ] | M0.10 | M0.2–M0.9 | Add `verify_measurement_truth.py` as milestone gate. | Truth checks are manual. | One command validates schemas, artifacts, cell completeness, operation visibility, and baseline selection. | Command exits 0; evidence `M0.10.json` links all outputs. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_measurement_truth.py \
  --manifest benchmarks/v2/objective_manifest.json
```

**Done when:** all M0 tasks and the M0 milestone checkbox are `[x]`; the H200 cell remains a strict
loss until performance actually clears the gate.

## 6. M1 — V2 compiler contracts and skeleton

**Goal:** create the target-neutral objects and stable target-backend boundary described by the IR
plan without perturbing the current V1 runtime.

**Before:** `src/megabake/v2/` and `src/cuda_v2/` do not exist; current code directly lowers FX-like
operations into one fixed CUDA schedule.

**After:** V2 can ingest an exported program, construct facts and regions, choose a backend, build a
serializable plan template, and round-trip an artifact schema without executing generated math.

**Milestone performance gate:** schema/fact/region construction for SmolLM2-135M completes in
`<=1 s` after export, uses `<=1 GB` peak host memory, and adds `0 us` to V1 runtime because the V2
path is opt-in.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M1.1 | M0 | Create the module layout from the IR plan under `src/megabake/v2/`, `src/cuda_v2/`, and `tests/v2/`. | No V2 package. | Importable empty packages and ownership boundaries. | Import smoke test `<100 ms`; V1 test collection unchanged. |
| [ ] | M1.2 | M1.1 | Implement immutable `WorkloadContract`, numerical/execution/performance enums, invocation choices, and execution/search budgets. | Policies are implicit Python arguments. | Validated, hashable contract objects. | Round-trip identity 100%; invalid combinations rejected; runtime delta `0 us`. |
| [ ] | M1.3 | M1.1 | Implement `WorkloadBucketKey`, symbolic bucket constraints, and stable hashing. | Example input is the de facto bucket. | Decode/prefill/token/KV/layout buckets are explicit. | 100% of equivalent-input fixtures hash equally; 100% of boundary fixtures create the expected distinct keys. |
| [ ] | M1.4 | M1.1 | Implement `TargetArchitectureKey`, `TargetFingerprint`, `DeviceCaps`, and backend registry interfaces without CUDA-specific fields in neutral objects. | Hardware tests are scattered. | Capability-driven backend selection. | SM80 and SM90 fixtures differ only in backend facts; neutral serialization has no instruction policy. |
| [ ] | M1.5 | M1.2–M1.3 | Add export/canonicalization wrapper and graph signature/state identity. | Existing graph walker mixes capture and lowering. | Stable canonical graph input to facts. | Repeated export hashes match; parameters/buffers/effects preserved 100%. |
| [ ] | M1.6 | M1.5 | Implement shape, layout, alias/lifetime, materialization, effect, constant, and precision fact tables. | Required facts are inferred ad hoc. | Complete `FactTables` with source mapping. | Golden models have no unknown required fact; fact construction `<=1 s`. |
| [ ] | M1.7 | M1.6 | Implement `RegionGraph`, primitive regions, bounded composite opportunities, edges, and semantic `TrafficEstimate`. | One task roughly equals one operation. | Transformer-scale semantic regions independent of launch count. | Semantic byte totals agree with tensor metadata within 1 byte; no CUDA policy in graph. |
| [ ] | M1.8 | M1.4, M1.7 | Implement `ExecutionCapability`, `KernelVariant`, resource/fusion/value-transport contracts, and family identifiers. | No uniform candidate ABI. | Backends can describe device, graph, and host candidates uniformly. | Illegal device-callability and transport combinations fail unit tests 100%. |
| [ ] | M1.9 | M1.8 | Implement `PlanTemplate`, segment templates, buffer/binding schemas, and `FinalExecutionPlan` types. | Scheduler state is the plan. | Bounded alternatives and frozen selection are separate. | Final plan cannot contain unresolved alternatives; runtime delta `0 us`. |
| [ ] | M1.10 | M1.2–M1.9 | Implement versioned `ArtifactPack` serialization with hashes, guards, provenance, and rejection of live handles/addresses. | Artifacts are process-oriented. | Deterministic portable metadata package. | Byte-identical repeat serialization; corrupted hash/version rejected 100%. |
| [ ] | M1.11 | M1.10 | Add golden IR/artifact snapshots for a linear, MLP, decoder block, and full SmolLM graph. | No V2 regression oracle. | Human-readable snapshots and schema tests. | Unapproved snapshot diffs = 0; 100% of round trips are byte-exact. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_contract tests/v2/test_facts \
  tests/v2/test_regions tests/v2/test_artifact
```

**Done when:** V2 stops before target code generation with a deterministic plan template, all M1
gates pass, and V1 behavior is unchanged.

## 7. M2 — Target backends, calibration, and baseline suite

**Goal:** convert a neutral plan into target-legal candidate sets and equivalent measured baselines
using real device facts rather than names or H200 constants.

**Before:** current tiling contains SM-count defaults, bandwidth output used an incorrect product-
name heuristic, and baseline selection is not a compiler object.

**After:** CUDA SM80 and SM90 plugins enumerate legal families; exact fingerprints own calibration,
baseline, resource, and tuning data; cross-target data is prior-only.

**Milestone performance gate:** fingerprint lookup `<=5 ms` after initialization; cached calibration
load `<=20 ms`; baseline measurement variance meets M0; no target mismatch can reuse a measured
winner.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M2.1 | M1.4 | Implement CUDA discovery for compute capability, UUID, MIG/partition, visible SM/L2/memory resources, driver/toolkit, clocks, and library IDs. | Partial runtime properties only. | Exact `TargetFingerprint` and raw discovery dump. | H200 reports 60 visible SMs for the laboratory partition; mocked mismatches are rejected. |
| [ ] | M2.2 | M2.1 | Implement feature legality tables for portable CUDA, SM80, SM90, cooperative launch, graph features, MMA, `cp.async`, TMA, WGMMA, and library availability. | Product names choose tactics. | Candidate factories query capabilities. | Cross-compiled SM80 source contains no SM90-only instruction; 100% negative fixtures rejected. |
| [ ] | M2.3 | M2.1 | Calibrate launch, graph, grid-join, atomic, binding-copy, streaming bandwidth, shape-family bandwidth, and KV bandwidth. | Peak bandwidth substitutes for measurement. | Versioned `DeviceCaps` calibration artifact. | Repeated calibration p50 within 5%; inferred peak never stored as achieved bandwidth. |
| [ ] | M2.4 | M0.3, M2.1 | Implement `MeasuredBaselineSuite` builder keyed by full equivalence class and target. | Baselines are printed side by side. | Ordinary, strongest low-overhead, static graph, vendor-region, and applicable serving baselines. | Automatic winner matches the manual minimum in 100% of fixtures; a missing strong mode blocks strict acceptance. |
| [ ] | M2.5 | M2.4 | Build exact-shape cuBLAS/cuBLASLt oracle profiler for grid/block, resources, duration, physical bytes, and kernel family name. | Vendor comparison is aggregate only. | Hot-shape oracle table with no private deployment dependency. | All frequent Smol linears classified; duration sums reconcile with trace within 3%. |
| [ ] | M2.6 | M2.1 | Implement code-object resource inspection: registers, stack, local memory, spills, SMEM, cluster, launch attributes, and disassembly hash. | Resources require manual Nsight inspection. | Machine-readable `ResourceEnvelope` and `EntryLaunchEnvelope`. | Known bad `acc[64]` fixture reports stack/local traffic; injected spill detected 100%. |
| [ ] | M2.7 | M2.1–M2.6 | Implement `AutotuneDB` keying, trust levels, invalidation, and prior-only cross-target transfer. | No measured winner database. | SQLite or equivalent transactional cache with artifact hashes. | Driver/backend/layout change invalidates exact result; H200 result cannot select SM80 winner. |
| [ ] | M2.8 | M2.3–M2.7 | Add H200 calibration/baseline/oracle artifact and cross-compile-only SM80 artifact. | Measurements are prose. | Reproducible target folders under `benchmarks/v2/results/`. | All artifacts validate; only H200 is labeled measured, SM80 labeled compile-only. |
| [ ] | M2.9 | M2.8 | Add CI tests for target-neutral leakage and target guard failures. | Portability regressions are manual. | Fixture and cross-compilation checks. | Forbidden instruction/target reuse fixtures fail 100%; host-only CI stays green. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_target tests/v2/test_baseline \
  tests/v2/test_resources tests/v2/test_autotune_db
.venv/bin/python benchmarks/v2/calibrate.py --target current --output benchmarks/v2/results/
```

**Done when:** the same neutral region graph produces different legal SM80/SM90 candidate sets,
H200 has persisted measurements, and no exact winner crosses fingerprints.

## 8. M3 — Generated entry, phase executor, and invocation path

**Goal:** replace the universal maximum-resource program with generated bucket-specific entries that
contain only reachable winners and execute a compact static phase program.

**Before:** the current cubin includes broad task paths, uses grid-wide BSP sequencing, conflates
logical tiles with resident workers in key paths, and exposes the whole entry to the heaviest
register/SMEM requirements.

**After:** a V2 artifact generates one compatible entry envelope, independent logical work mapping,
explicit reset/binding/output behavior, and separate release/instrumented code objects.

**Milestone performance gate:** on H200, an empty/minimal V2 phase program adds `<=10 us` GPU time
over an empty cooperative entry and `<=50 us` complete invocation overhead after initialization;
there are zero unexpected allocations/copies and zero hot-path spills/local-memory accesses.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M3.1 | M2 | Define the versioned host/device ABI for `PhaseProgram`, `PhaseDesc`, `WorkDesc`, completion epochs, binding table, and scheduler state. | V1 `TaskDesc` is the universal ABI. | Matching Python/C++ layouts with ABI hash. | Size/offset static assertions and round-trip fuzzing pass 100%; runtime delta `0 us`. |
| [ ] | M3.2 | M3.1 | Implement `STATIC_RANGE`, `ATOMIC_CURSOR`, and `ALL_ACTIVE_WORKERS` distributions with independent logical and resident counts. | Tile count often follows visible SMs. | Workers loop over zero or more logical work items. | Synthetic coverage executes every logical item exactly once for worker counts 1 through 2x SM count. |
| [ ] | M3.3 | M3.1–M3.2 | Implement `NONE`, `CTA_JOIN`, and `GRID_JOIN` completion protocols. | Every task uses a broad grid barrier. | Joins occur only at required composite/phase dependencies. | 100% of race/sanitizer fixtures pass; a no-dependency two-phase fixture removes at least 1 grid join. |
| [ ] | M3.4 | M3.1 | Implement preallocated `IN_ENTRY_PARALLEL_RESET` and `EPOCH_TAGGED` state. | Per-invocation state may be cloned/reset externally. | No dependency-array clone on strict path. | Reset correctness over 10,000 invocations; reset GPU time `<=5 us` on H200. |
| [ ] | M3.5 | M1.9, M3.1 | Implement `STATIC_SESSION`, dynamic `BindingBlock`, `RUN_INTO`, `BORROWED_OUTPUT`, and owned-output accounting. | Binding/output lifetime is implicit. | Every transfer and lifetime is represented in the plan. | Static session has 0 binding HtoD bytes; dynamic binding uses one transfer and `<=5 us` GPU copy on H200. |
| [ ] | M3.6 | M3.1–M3.5 | Generate `persistent_entry.cu` from selected reachable variants and a bucket-specific phase program. | One source contains every task path. | Source manifest lists exactly the reachable bodies. | 0 dead/unselected fixture symbols in PTX/SASS; exactly 1 entry symbol and 1 strict compute grid. |
| [ ] | M3.7 | M3.6 | Implement mutually exclusive shared-storage overlay and reject incompatible warp/block/cluster contracts. | Maximum SMEM is broadly reserved. | Shared storage equals aligned maximum of reachable mutually exclusive phases, not sum or global maximum. | Golden layouts are 100% byte-exact; incompatible candidates are rejected before launch. |
| [ ] | M3.8 | M2.6, M3.6–M3.7 | Feed compiled entry resources back into plan finalization and regenerate/split when illegal. | Per-body estimates are trusted. | Final `EntryLaunchEnvelope` is post-link and measured. | Registers/stack/SMEM match compiler metadata exactly; 0 illegal cooperative grids launch. |
| [ ] | M3.9 | M3.6 | Produce separate release and instrumented binaries. | Profiling code can affect production resources. | Code-object kind is part of provenance. | Task-timer instructions in release SASS = 0; unexplained release-resource regressions = 0. |
| [ ] | M3.10 | M3.2–M3.9 | Build empty, copy-only, pointwise-only, and two-phase integration artifacts. | No isolated executor baseline. | Minimal artifacts and raw timelines establish fixed costs. | Empty/minimal overhead meets milestone numbers; 10,000 replays pass correctness and sanitizer. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_phase_program tests/v2/test_entry_codegen \
  tests/v2/test_invocation
.venv/bin/python benchmarks/v2/measure_executor.py --target current --protocol release
```

**Done when:** a generated minimal strict artifact contains one measured compatible entry, no
unreachable body, and all orchestration cost is visible.

## 9. M4 — Four-family linear tournament

**Goal:** make GEMM quality a measured compiler choice across cuBLASDx, CUTLASS collectives, direct
CuTe compositions, and MegaBake-native CUDA rather than a fixed handwritten kernel.

**Before:** the H200 strict grid spends about `4,748.7 us` in one resource-heavy entry while the
ordinary compiled path's CUDA kernels sum to about `985.1 us`; current skinny code retains
`float acc[64]` and a fixed mapping that is poor for frequent decode shapes.

**After:** each exact linear signature receives a bounded legal candidate tournament; standalone and
embedded measurements select a target-specific winner; only selected bodies enter the final entry.

**Milestone performance gate:** every frequent isolated shape reaches `>=80%` of its matched vendor
body during bring-up and `>=95%` before a strict-win claim, unless a fused composite demonstrably
repays the entire deficit. Hot skinny candidates have `0` unexplained stack bytes, spills, or
`LDL/STL` traffic.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M4.1 | M2.5 | Inventory every linear signature in the initial real models: frequency, M/N/K, dtype, layouts, alignment, epilogue, and packed-weight eligibility. | Shapes are inferred while executing. | Versioned shape inventory and hot-shape rank. | 100% of linear calls mapped; aggregate counts match profiler trace. |
| [ ] | M4.2 | M1.8, M4.1 | Define common device task ABI, input/output contract, fusion endpoints, and canonical generated-body signature for all four families. | Families have incompatible launch assumptions. | One `LinearCandidate` contract and deduplication key. | 100% of cross-family correctness fixtures consume identical buffers; duplicate equivalent configurations retained = 0. |
| [ ] | M4.3 | M3.6, M4.2 | Implement exact `M=1`, `M=2`, and `M=4` native warp-reduction GEMV generators. | Runtime `M`, `acc[64]`, fixed tiling. | Compile-time M and bounded accumulator fragments. | No `acc[64]`, unexpected stack, or local loads/stores; each hot shape reaches 80% gate or remains an explicit loser. |
| [ ] | M4.4 | M4.2 | Implement native CTA-per-output and split-K candidates with explicit reduction cost. | Insufficient parallelism has one mapping. | Legal split factors and reduction plans. | No global partials unless measured winner; correctness for odd K and tails; 80% body gate on selected shapes. |
| [ ] | M4.5 | M4.2 | Implement padded-M tensor-core candidates for large N and tune wasted rows versus coalescing. | All small M assumed GEMV-like. | Tensor-core candidate available at M=1–4 where legal. | Vocabulary-head shape includes at least 1 correct tensor-core candidate; the measured choice beats or explicitly rejects it. |
| [ ] | M4.6 | M4.2 | Build CUTLASS collective adapter below the host `Device` launcher with MegaBake tile identity and storage contract. | CUTLASS exists only as standalone/hand-integrated code. | Device-callable collective candidate generator. | Host launches in task path = 0; 100% of exact hot-shape fixtures are correct; resources are machine-readable. |
| [ ] | M4.7 | M4.2 | Build direct CuTe-composed candidate generator for atoms, layouts, copies, stages, and architecture-specific pipelines. | One manually fixed CuTe path. | Searchable direct-composition family. | SM80/SM90 legality tests pass; canonicalization removes duplicates with other families. |
| [ ] | M4.8 | M2.2 | Provision and pin compatible MathDx/cuBLASDx and CUDA toolchains without replacing the repository's working default environment. | `cublasdx.hpp` absent from CUDA 12.8 environment. | Versioned dependency lock, license metadata, compile probe, and isolated build route. | Official sample compiles/runs on H200; exact versions recorded; default V1 environment remains usable. |
| [ ] | M4.9 | M4.2, M4.8 | Implement cuBLASDx descriptor enumeration across tile size, precision, arrangement, leading dimensions, alignment, `BlockDim`, SM modifier, and accumulation form. | cuBLASDx is research prose only. | First-class `CUBLASDX` candidate generator. | At least 1 legal descriptor per supported hot cell; unsupported reason explicit; 100% of Section 3.5 budget limits enforced. |
| [ ] | M4.10 | M4.9 | Implement cuBLASDx shared-memory, returned/explicit register-accumulator, and pipelined global-memory forms with epilogue hooks. | One block example does not form a full task. | Persistent-worker-compatible tile/K pipeline. | 100% of required descriptor threads participate; sanitizer passes; epilogue meets the M0 reference tolerance. |
| [ ] | M4.11 | M4.3–M4.10 | Implement weight prepack descriptors per candidate layout with load-time caching and hashes. | Invocation may reinterpret/transpose weights. | Selected packed weights are artifact state. | Timed invocation performs 0 weight packing/transposes; packed bytes and padding recorded. |
| [ ] | M4.12 | M4.3–M4.11 | Add exhaustive boundary correctness suite for tails, alignments, layouts, alpha/beta, epilogues, and M=1/2/4. | Only broad e2e tolerances exist. | Family-neutral golden tests and sanitizer runs. | 100% legal cases pass M0 numerical policy; corrupted descriptor fixtures fail. |
| [ ] | M4.13 | M2.7, M4.12 | Implement analytical pruning, family quotas, randomized standalone timing, successive halving, and search-budget accounting. | Winner selection is manual. | Deterministic search given raw measurements and seed. | Caps from Section 3.5 enforced; at least one legal family survivor reaches embedded stage. |
| [ ] | M4.14 | M3, M4.13 | Benchmark survivors standalone, in a minimal persistent worker, with available linear epilogues, and in separately compiled vertical-entry beams; broader transformer composites are deferred to M7. | Standalone performance may hide entry poisoning. | Four measurement levels stored per survivor. | 4 measurement levels exist per survivor; standalone-only winners accepted = 0. |
| [ ] | M4.15 | M4.14 | Freeze exact-fingerprint winners and fallback order in `AutotuneDB` and artifact manifests. | Runtime could retune or select by heuristic. | Runtime loads guarded winner only. | Cold lookup selects correct winner in `<=5 ms`; mismatch triggers compile/install path or labeled fallback. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_linear_candidates
.venv/bin/python benchmarks/v2/tune_linear.py \
  --manifest benchmarks/v2/objective_manifest.json --target current --model SmolLM2-135M
```

**Done when:** each frequent H200 shape has correctness/resource/timing evidence for all legal
families, the winner satisfies the body gate or has a quantified fusion repayment plan, and the
release candidate contains no losing body.

## 10. M5 — H200 strict vertical slice

**Goal:** turn the first laboratory cell from a characterized strict loss into the first strict
real-model win without making H200 behavior generic compiler policy.

**Before:** SmolLM2-135M strict invocation is `4,845.5 us`; its one grid is `4,748.7 us`, uses 184
registers/thread, 1,072 stack bytes/thread, 225,280 B SMEM/CTA, and trails the `1,781.7 us` strong
baseline by about `2.7x`.

**After:** a reference-precision, batch-1, sequence-1 SmolLM2 artifact uses exactly one owned compute
grid, contains only reachable decode bodies, passes correctness, and beats the remeasured strongest
equivalent baseline with the global margin.

**Milestone performance gate:** if the strong baseline remains `1,781.7 us`, MegaBake p50 is
`<=1,692.6 us`; p99 does not exceed baseline p99; strict speedup lower confidence bound exceeds
`1.00`; body and orchestration intervals fully reconcile with invocation time.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M5.1 | M0.7, M4 | Freeze the exact H200 Smol workload/equivalence key, inputs, prompt corpus, output contract, and current before profile. | Several historical H200 profiles differ. | One canonical vertical-slice cell and immutable before artifact. | Reproduction p50 within 5% or environment difference explains it. |
| [ ] | M5.2 | M3.6, M5.1 | Generate a decode-only entry with M=1/2/4 specializations; remove unreachable prefill, heavy attention alternatives, dynamic `acc[64]`, and profiling paths. | Universal entry resources dominate. | Lean source/code-object manifest. | Unreachable symbols absent; hot path stack `0 B` unless a measured faster exception is recorded. |
| [ ] | M5.3 | M4.15, M5.2 | Install selected exact-shape linear winners and packed layouts for all repeated projections and vocabulary head. | One handwritten family serves all shapes. | Per-shape winner map inside the entry build. | Each installed body meets 95% vendor gate or has measured composite repayment. |
| [ ] | M5.4 | M3.2–M3.8, M5.3 | Tune logical tiles, resident workers, active workers, CTA topology, phase joins, and shared-storage envelope jointly. | Grid is effectively one CTA per visible SM with broad resources. | Measured compatible `EntryLaunchEnvelope`. | Illegal cooperative launches and unexplained spill/local traffic = 0; chosen mapping is the measured neighborhood minimum. |
| [ ] | M5.5 | M5.4 | Attribute and optimize the non-linear tail only after linear gates: copies, reductions, pointwise, RoPE, attention, reset, and barriers. | Aggregate megakernel timing hides phase cost. | Complete phase/idle/traffic breakdown and bounded fixes. | 100% GPU time attributed within 3%; every accepted change improves complete p50 by >=1% or enables a later composite. |
| [ ] | M5.6 | M3.5, M5.5 | Exercise `STATIC_SESSION + RUN_INTO` and explicit dynamic/owned-output alternatives. | Invocation overhead and output clone can be conflated. | Fast strict invocation plus separately scored general path. | Hidden static-path output clones/binding transfers = 0; 100% of other work is counted. |
| [ ] | M5.7 | M5.6 | Run full correctness, sanitizer, distribution, unfiltered timeline, counters, resource inspection, and strongest-baseline comparison. | Only development samples exist. | Immutable release-candidate evidence set. | M0 protocols pass; exactly one owned compute grid; numerical gate passes 100%. |
| [ ] | M5.8 | M5.7 | Record H200 scorecard cell and freeze artifact or retain explicit strict-loss status. | Headline can overstate progress. | Honest `STRICT_WIN` or measured loss with reason. | Mark M5 complete only for `STRICT_WIN`; historical `1.17x` ordinary-baseline result is never accepted. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_cell.py \
  --manifest benchmarks/v2/objective_manifest.json \
  --target h200_3g --model SmolLM2-135M --bucket decode_b1_m1 --strict
```

**Done when:** H200 is the first measured strict-win cell, not merely a characterized CUDA target.
If the gate fails, keep M5 unchecked and iterate M4/M5 using the stored attribution.

## 11. M6 — SM80/A100 portability proof

**Goal:** prove that generic means shared compiler semantics with independently generated and tuned
target artifacts, not H200 constants wrapped in an abstraction.

**Before:** only H200 has real V2 measurements; SM80 is at most cross-compiled; cuBLASDx/CuTe/
CUTLASS/native winners and resource envelopes are unvalidated on A100.

**After:** the identical target-neutral graph, facts, regions, and workload contract produce an
SM80-legal artifact, independently calibrated/tuned on A100, with an independent strict score.

**Milestone performance gate:** A100 complete p50 is `<=0.95 *` its strongest equivalent compiled
baseline, p99 does not regress, all numerical gates pass, and no H200 result is used as more than
candidate ordering.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M6.1 | M5 | Register a real A100/SM80 runner and immutable environment fingerprint in the objective manifest. | No non-Hopper performance machine. | Authorized repeatable target cell. | 100% of discovery/calibration commands pass; target is a real device, not an emulator or cross-compile-only result. |
| [ ] | M6.2 | M2.2, M6.1 | Cross-compile all reachable portable/SM80 generators and reject TMA/WGMMA/SM90-only paths. | SM90 paths may leak through templates. | Clean SM80 code-object set and legality report. | Disassembly scan finds 0 illegal instructions; all required symbols link. |
| [ ] | M6.3 | M2.3–M2.5, M6.1 | Calibrate A100 and build its ordinary/low-overhead/vendor baseline suite and shape oracle. | H200 priors are the only measurements. | A100-specific caps, raw samples, and baseline winner. | Five-process variance gate passes; no H200 timing/resource copied into exact fields. |
| [ ] | M6.4 | M4, M6.2–M6.3 | Run the four-family linear tournament using only SM80-legal configurations. | H200 winners could be tempting defaults. | Independent A100 winner map and packed layouts. | Every frequent body reaches 80%, then 95% or measured fusion repayment; winners may differ from H200. |
| [ ] | M6.5 | M3, M6.4 | Generate and tune an SM80 entry envelope, worker mapping, phase program, and invocation contract. | H200 residency/topology is unportable. | Post-link A100 artifact. | 100% of cooperative-legality checks pass; unexplained local traffic = 0; chosen mapping is the measured neighborhood minimum. |
| [ ] | M6.6 | M6.5 | Run the same canonical Smol vertical cell on A100 against its strongest baseline. | Genericity is structural only. | A100 correctness/performance/timeline evidence. | Strict p50/p99/one-grid gates pass on real hardware. |
| [ ] | M6.7 | M6.6 | Audit neutral artifacts for byte-identical semantics and target artifacts for intentional differences. | Portability boundary is assumed. | Diff report separates neutral identity from target specialization. | 100% of neutral graph/region/contract hashes match; target code/layout/resource hashes differ and are guarded. |
| [ ] | M6.8 | M6.7 | Record independent H200 and A100 scorecard cells and support levels. | One target could imply generic CUDA. | Two measured architecture-family cells. | Both are `STRICT_WIN`; otherwise M6 remains unchecked. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_portability.py \
  --manifest benchmarks/v2/objective_manifest.json --targets h200_3g,a100
```

**Done when:** both targets independently satisfy their baselines using the same upper compiler and
different guarded machine artifacts.

## 12. M7 — Traffic-removing transformer composites

**Goal:** turn kernel persistence into a traffic and synchronization advantage rather than merely a
different dispatcher.

**Before:** competitive isolated GEMMs still materialize normalized activations, separate gate/up
outputs, Q/K/V outputs, epilogues, and other intermediate values or join after syntax-level work.

**After:** legal composites recompute or forward values locally, remove measured HBM bytes/joins,
and are selected only when their combined entry resources improve the complete objective.

**Milestone performance gate:** each accepted composite either improves its equivalent phase p50 by
`>=5%` or reduces counter-measured DRAM bytes by `>=10%` while causing `<=1%` whole-entry regression
before downstream consumers are included; final whole-entry beams must improve or remain rejected.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M7.1 | M1.7, M3 | Implement candidate-specific `REGISTER_FORWARD`, `SMEM_FORWARD`, `RECOMPUTE_PER_WORKER`, `RECOMPUTE_PER_CLUSTER`, and HBM materialization costing. | Edges imply global materialization. | Measurable `ValueTransportPlan` alternatives. | Traffic accounting reconciles with counters within 10% on calibration fixtures. |
| [ ] | M7.2 | M7.1 | Implement `NormLinearComposite` with per-worker RMSNorm recomputation and compatible linear families. | Normalized activation is materialized/read repeatedly. | Local normalized input lifetime and combined resource contract. | Numerical gate passes; accepted variant meets 5% phase or 10% DRAM-byte gate. |
| [ ] | M7.3 | M7.1 | Implement `GatedMLPComposite`: gate/up traversal plus SiLU-multiply epilogue storing only product. | Gate and up intermediates reach HBM. | Fused output contract with no unnecessary intermediates. | Materialized bytes fall as predicted within 10%; complete phase p50 improves >=5%. |
| [ ] | M7.4 | M7.1 | Implement grouped/concatenated `QKVProjectionComposite` with GQA-aware routing. | Q/K/V projections schedule separately. | One legal composite family with packed routing metadata. | Correct Q/K/V layouts; accepted variant improves phase p50 >=5% or removes >=10% bytes. |
| [ ] | M7.5 | M7.1 | Implement bias, residual, activation, scaling, and cast `LinearEpilogueComposite` hooks for all winning linear families. | Epilogues launch/phase separately or materialize C. | Register-accumulator/store transform path. | Every supported epilogue correct; accepted fusion never increases complete phase p50. |
| [ ] | M7.6 | M7.1 | Implement RoPE plus KV-append composite contract, deferring attention consumption details to M10. | Rotated K/V can be copied twice. | Legal in-grid write path to selected KV layout. | Output/cache correctness over boundary positions; accepted path reduces bytes or p50 by milestone gate. |
| [ ] | M7.7 | M7.2–M7.6 | Extend plan enumeration and whole-entry beam search across composite, transport, and resource choices. | Fast local bodies can poison whole entry. | Bounded composite beams and rejection reasons. | Beam width/search budget enforced; final choice uses whole-entry p50 and resource legality. |
| [ ] | M7.8 | M7.7 | Replay accepted composites on H200 and A100 with counters and scorecards. | Benefits may be target-specific. | Per-target composite winner maps. | No cross-target assumed winner; each target's final strict p50 improves or artifact remains unchanged. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_composites tests/v2/test_value_transport
.venv/bin/python benchmarks/v2/tune_composites.py --targets h200_3g,a100
```

**Done when:** every accepted composite has a byte/latency/resource proof and both targets retain
strict correctness and performance gates.

## 13. M8 — Precision and prepacking portfolios

**Goal:** reduce the dominant weight-byte floor under explicit numerical policies without using a
quantized result to satisfy the reference-precision objective.

**Before:** reference FP16/BF16 streams most weights once per token; quantized layouts, scales,
conversion costs, quality baselines, and artifact selection are incomplete.

**After:** FP8, W8A16, and W4A16 have separate candidate/prepack/quality/baseline/scorecard paths,
selected by the same target-specific tournament.

**Milestone performance gate:** W8 packed weight bytes are `<=55%` and W4 bytes `<=30%` of the
reference representation including scale/zero-point metadata; each accepted policy beats its
strongest equivalent quantized baseline by 5% at p50 without p99 or quality regression.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M8.1 | M0.6, M1.2 | Freeze separate FP8/W8A16/W4A16 policies, calibration corpus, quality metrics, and equivalent baselines. | Quantization may silently change objective. | Manifest policies and quality reports. | Reference-score changes = 0; 100% of policy selections change the equivalence key and artifact hash. |
| [ ] | M8.2 | M4.11 | Extend prepack descriptors for scales, zero points, group size, transpose, swizzle, padding, and target-family layout. | Reference packing only. | Reproducible quantized pack cache. | Round-trip metadata exact; timed invocation performs 0 packing; corrupted hash rejected. |
| [ ] | M8.3 | M4, M8.2 | Add W8A16 native/library candidates across winning linear mappings. | No 8-bit weight task portfolio. | Four-family W8 candidate set where supported. | Packed bytes <=55%; kernel correctness passes; unsupported families explicit. |
| [ ] | M8.4 | M4, M8.2 | Add W4A16 groupwise candidates and metadata-aware epilogues. | No 4-bit task portfolio. | Legal W4 candidate set and pack layouts. | Packed bytes <=30%; dequant/accumulation correctness and boundary groups pass. |
| [ ] | M8.5 | M4, M8.1 | Add FP8 activation/weight conversion, scale, accumulator, and candidate choices. | Conversion cost is ignored. | Conversion represented as fused or materialized traffic. | 100% of conversion time/bytes counted; accepted plan meets the configured `<=0.95 * baseline` p50 gate end to end. |
| [ ] | M8.6 | M8.3–M8.5 | Run kernel, layer, perplexity/task-quality, and fixed-prompt validation for each policy. | Kernel closeness may hide model degradation. | Immutable quality artifact per model/policy. | 100% policy thresholds pass; otherwise cell is quality failure, not performance win. |
| [ ] | M8.7 | M2.4, M8.6 | Build strongest equivalent quantized baselines and per-policy tournaments on H200/A100. | Reference baseline might be reused unfairly. | Separate baseline suites and winners. | 5% p50/p99 gates use identical quantization/service policy. |
| [ ] | M8.8 | M8.7 | Emit separate quantized scorecards and guarded runtime selection. | One score could mix policies. | Reference, FP8, W8, W4 results remain disjoint. | Quantized cells added to the reference strict-win numerator = 0. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_precision.py \
  --policies reference,fp8,w8a16,w4a16 --targets h200_3g,a100
```

**Done when:** every enabled precision policy is quality-valid, fully costed, independently compared,
and selected only through a matching artifact guard.

## 14. M9 — Hybrid graph product fallback

**Goal:** provide the fastest legal product path for strict-loss or resource-incompatible cells while
preserving the integrity of the strict one-grid score.

**Before:** vendor kernels cannot execute inside the owned grid, resource-class splits are either
forced together or manually launched, and graph binding/output costs can be hidden.

**After:** public graph recipes combine selected persistent entries, vendor nodes, copies, memsets,
and guarded binding updates; hybrid performance is measured and reported separately.

**Milestone performance gate:** choose a MegaBake hybrid only if it is at least 5% faster at p50 than
the strongest equivalent pre-existing graph/vendor route and does not regress p99; otherwise use the
strongest baseline as the product fallback. Hybrid results contribute `0` strict wins.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M9.1 | M1.9, M2.4 | Implement versioned `CudaGraphSegment` and public `GraphRecipe` construction/capture with fixed workspaces. | Graph execution is harness-specific. | Reconstructable recipe, no serialized live handle. | Fresh-process reconstruction produces identical dependencies and outputs 100%. |
| [ ] | M9.2 | M3.5, M9.1 | Implement stable binding tables, host graph updates, static addresses, and measured selection. | Binding path may clone/copy implicitly. | Explicit binding-update map and bytes. | Static path 0 binding bytes; dynamic work appears in trace and meets declared contract. |
| [ ] | M9.3 | M9.1–M9.2 | Implement a batched one-thread binding prelude only for confirmed public node parameter slots with version guards and error status. | Pointer updates are speculative/manual. | Guarded optional prelude candidate. | 10,000 alternating replays correct; any API/version mismatch disables candidate safely. |
| [ ] | M9.4 | M3.8, M9.1 | Connect resource-specific persistent entries, vendor nodes, copies, and memsets into dependency-correct recipes. | Incompatible resource classes poison one entry. | Hybrid segmentation candidate set. | Operation/grid counts match the recipe exactly; 100% of hybrid fixtures are classified multi-grid. |
| [ ] | M9.5 | M9.1–M9.4 | Add static graph, host-update graph, prelude graph, and hybrid benchmark modes to baseline suite. | Product fallback has no common tournament. | Equivalent end-to-end distributions. | 100% of binding/workspace/output costs are included; automatic route matches the measured minimum. |
| [ ] | M9.6 | M9.5 | Add compatibility guards, graph upload cache, invalidation, and safe fallback chain. | Cached graph can outlive environment assumptions. | Fresh initialization and guarded reuse. | 100% of mismatch fixtures rebuild or fall back; stale graph executions = 0. |
| [ ] | M9.7 | M9.6 | Emit hybrid scorecard fields and public reporting distinct from strict results. | A one-submission graph might be called one kernel. | Separate grids, submissions, segments, and speedup. | Hybrid adds 0 to strict numerator; README/report labels it unambiguously. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_graph_recipe tests/v2/test_binding
.venv/bin/python benchmarks/v2/verify_hybrid.py --manifest benchmarks/v2/objective_manifest.json
```

**Done when:** every strict-loss cell has a guarded measured fallback, but no hybrid result can pass
the strict verifier.

## 15. M10 — Attention, KV cache, prefill, and decode control

**Goal:** extend the strict compiler from linear-heavy single-token evidence to sequence-dependent
transformer execution without reintroducing a universal resource envelope.

**Before:** attention/KV behavior is a broad current task or vendor fallback; sequence length, head
mapping, KV layout/precision, split-KV thresholds, prefill resources, and decode-loop control lack
complete target-specific portfolios.

**After:** decode and prefill buckets select legal attention/KV candidates, layouts, composites, and
resource-specific entries; static strict invocations and multi-grid device-controlled loops remain
separately classified.

**Milestone performance gate:** every selected attention body reaches `>=80%` of the matched vendor
body during bring-up and `>=95%` or repays its deficit through RoPE/KV/composite traffic removal;
each completed model/bucket still clears the global 5% strict p50 margin and p99 gate.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M10.1 | M0.1, M1.7 | Inventory attention signatures: batch, query/KV lengths, heads, KV heads, head dim, causal/window mask, dtype, and cache behavior. | One attention category hides regimes. | Short/medium/long decode and prefill buckets. | 100% declared model attention calls classified; bucket boundaries frozen before tuning. |
| [ ] | M10.2 | M10.1 | Implement paged and contiguous KV layout plans, page tables, allocation/lifetime, and optional FP8/INT8 policy fields. | KV layout is task-local. | Artifact-owned `PrecisionLayoutPlan` and cache state contract. | 100% of boundary/page-crossing fixtures pass; unrepresented bytes/metadata = 0. |
| [ ] | M10.3 | M2.4–M2.5, M10.1 | Build strongest equivalent attention baselines and exact-shape oracle profiles, including available fused/vendor kernels. | Whole-model attention time only. | Per-bucket body/grid/resource/traffic baselines. | Profile sums reconcile within 3%; numerical and layout contracts match. |
| [ ] | M10.4 | M3, M10.1–M10.3 | Implement native/CuTe/CUTLASS device-callable decode attention with online softmax and target-legal movement. | Broad reference task is not tuned. | Candidate family for each supported head/KV shape. | Numerical gate passes; no full score/probability materialization; 80% body gate. |
| [ ] | M10.5 | M10.4 | Implement GQA/MQA-aware head mapping and logical work distribution independent of residency. | KV heads can underutilize the grid. | Tunable query-head/KV-head work mapping. | 100% of heads are produced exactly once; chosen mapping is the measured minimum; unexplained idle-work anomalies = 0. |
| [ ] | M10.6 | M10.4–M10.5 | Implement single-block and split-KV/multi-block candidates with explicit reduction workspace and thresholds. | One algorithm spans all sequence lengths. | Measured pivot by sequence/parallelism. | Split path wins only where reduction-inclusive p50 improves; threshold cross-validation error <=5%. |
| [ ] | M10.7 | M7.6, M10.2–M10.6 | Integrate RoPE plus paged KV append and optional attention consumption without redundant HBM round trips. | RoPE/append/attention are separate phases. | Composite and transport alternatives. | Cache/output correct over long positions; accepted composite meets M7 byte/latency gate. |
| [ ] | M10.8 | M10.3–M10.6 | Implement a separate prefill portfolio and resource-class entry; never retain it in decode entry. | Prefill may inflate decode resources. | Prefill-specific tiles/pipelines/attention candidates. | Decode code object has 0 reachable prefill symbols; prefill reaches 80/95% body gates independently. |
| [ ] | M10.9 | M9, M10.2 | Implement sampling/termination state for static strict step and optional conditional/tail-launched graph loop. | Host drives every token or semantics are implicit. | Explicit one-step strict contract and separately scored multi-grid controller. | Device loop never counts as strict one grid; outputs match host-driven reference over 100 sequences. |
| [ ] | M10.10 | M10.1–M10.9 | Run full decoder models across KV buckets on H200 and A100 and update scorecards. | Only sequence-1 cell is proven. | Per-sequence artifacts, thresholds, distributions, and strict/hybrid outcomes. | Every executed cell correct; eligible strict cells meet global p50/p99 gates or remain losses. |

### Milestone verification

```bash
.venv/bin/python -m pytest -q tests/v2/test_attention tests/v2/test_kv_cache
.venv/bin/python benchmarks/v2/verify_sequence_buckets.py --targets h200_3g,a100
```

**Done when:** decode and prefill resources are isolated, each attention regime has a measured
winner/fallback, and sequence-dependent scorecard cells are honest.

## 16. M11 — Continuous batching and serving throughput

**Goal:** reuse each streamed weight tile across multiple active tokens under an explicit latency and
fairness policy, without presenting throughput as batch-one latency.

**Before:** strict fixed-bucket execution handles one request objective; weight bytes are reread for
each token and dynamic scheduler complexity has no measured justification.

**After:** a separate `ServingSchedule` admits, groups, executes, and retires requests with paged KV,
bounded wait, fairness, and an equivalent serving baseline/scorecard.

**Milestone performance gate:** under the frozen arrival trace and SLO, accepted MegaBake serving
throughput is `>=1.05x` the strongest equivalent serving baseline, p99 time-to-first-token and
inter-token latency remain within the manifest SLO, and no request exceeds the fairness bound.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M11.1 | M0.1, M10 | Freeze arrival traces, request-length distributions, max batch wait, fairness, TTFT/ITL SLOs, and serving baseline. | Throughput objective is informal. | Versioned serving `WorkloadContract`. | 100% of trace hashes are fixed before tuning; batch-one results accepted under this contract = 0. |
| [ ] | M11.2 | M1.9, M11.1 | Implement request state, admission queue, cancellation, completion, and paged-KV ownership. | Fixed session owns one state. | Correct concurrent request lifecycle. | Stress test has 0 leaks/use-after-free and correct outputs for 10,000 request events. |
| [ ] | M11.3 | M11.2 | Implement bounded waiting and layer/shape/precision compatibility grouping. | Requests execute independently. | Active-token batches with fairness timestamps. | Requests beyond the declared wait bound = 0; incompatible-policy groups = 0. |
| [ ] | M11.4 | M4, M7, M11.3 | Extend winning linears/composites to grouped active-token M and reuse one weight tile across rows. | Weight streamed per request. | Grouped persistent GEMM/visitor candidate. | Counter-derived weight bytes/token fall with batch size within 15% of predicted reuse. |
| [ ] | M11.5 | M10, M11.3 | Integrate batched attention/KV updates and per-request completion masks. | Attention state assumes fixed request. | Correct variable-length serving phases. | Post-finish mutations = 0; 100% of active outputs match an independently run reference. |
| [ ] | M11.6 | M11.3–M11.5 | Compare distributed scheduling with and without a dedicated scheduler CTA. | Scheduler CTA is assumed helpful or harmful. | Measured scheduling variants and resource cost. | Dedicated CTA selected only if complete throughput improves >=2% without SLO violation. |
| [ ] | M11.7 | M11.1–M11.6 | Run load sweeps, saturation curves, fairness, TTFT/ITL, power/traffic, and strongest-baseline comparison. | No serving score. | Serving scorecard and raw traces. | >=1.05x throughput, all SLO/fairness/quality gates, separate from strict latency score. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_serving.py \
  --manifest benchmarks/v2/objective_manifest.json --targets h200_3g,a100
```

**Done when:** throughput gains are explained by measured reuse and remain within the declared
service contract; none are reported as reference batch-one wins.

## 17. M12 — Model/hardware coverage and release audit

**Goal:** convert successful point cells into the predeclared “every release hardware target, most
model cells” result and ship a reproducible compiler/runtime rather than a benchmark branch.

**Before:** H200 and A100 vertical cells may pass, but model-family breadth, unsupported semantics,
artifact lifecycle, search cost, CI, documentation, and the final denominator are incomplete.

**After:** every declared matrix cell is run and classified; at least 80% strict-win on every target;
all cells are correct or explicitly unsupported losses; artifacts are guarded, reproducible, cached,
and documented; one command verifies the release.

**Milestone performance gate:** per target, eligible strict-win rate `>=80%`, geometric-mean strict
speedup over the strongest equivalent compiled baseline `>=1.05x`, no winning cell regresses p99,
100% measured-cell correctness, and 100% declared hardware coverage.

### Atomic tasks

| Done | ID | Depends | Atomic goal/work | Before | After and evidence | Acceptance gate |
|---|---|---|---|---|---|---|
| [ ] | M12.1 | M0.1, M5–M10 | Implement adapters and freeze representative checkpoints for SmolLM, Llama, Gemma, Qwen, Mistral, and Phi model families. | One real model dominates evidence. | Six independently weighted model families with immutable revisions. | 100% of the 6 model families export; alias variants add 0 denominator weight; all input/prompt hashes are stored. |
| [ ] | M12.2 | M1.7, M10, M12.1 | Close required semantic/operator gaps for declared models or record them as strict losses. | Unsupported ops can disappear from reporting. | Complete region/operator coverage report. | 100% graph nodes classified; no eager escape in strict path; unsupported cell remains denominator loss. |
| [ ] | M12.3 | M12.1–M12.2 | Generate/tune decode M=1/2/4 and short/medium/long KV cells on every release target. | Only vertical buckets are proven. | Full mandatory decode matrix. | Each cell has result/evidence; no unrun cell omitted; correctness 100%. |
| [ ] | M12.4 | M10.8, M12.1 | Generate/tune at least one declared prefill bucket per model family and target. | Decode-only claim could be called general inference. | Prefill matrix and separate artifacts. | 100% cells measured; resource-specific prefill does not alter decode artifacts. |
| [ ] | M12.5 | M2.7, M4.13, M12.3–M12.4 | Audit compile time, candidate counts, cache size, artifact size, and first-install tuning against frozen budgets. | Search could be operationally unbounded. | Per-cell search-cost report and cache policy. | No undeclared budget overrun; cache/artifact corruption tests pass; runtime performs 0 tuning. |
| [ ] | M12.6 | M1.10, M9.6, M12.3 | Validate artifact determinism, compatibility ranges, fresh-process loading, driver/library mismatch handling, and fallback. | Lab process state may be required. | Installable guarded artifact packs. | 100 clean-process replays correct; all negative compatibility fixtures reject safely. |
| [ ] | M12.7 | M12.3–M12.4 | Generate final strict, hybrid, precision, and serving scorecards directly from raw evidence. | Results can be selected manually. | Immutable scorecards with manifest denominator and exclusions. | 100% cells accounted; strict score contains exactly one-grid results only. |
| [ ] | M12.8 | M12.7 | Enforce per-target 80% strict-win, 1.05x geomean, p99, correctness, and hardware-coverage gates in CI/release script. | North-star prose has no executable final gate. | `benchmarks/v2/verify_release.py`. | Intentional loss/cell deletion/hybrid relabel makes verifier fail 100%. |
| [ ] | M12.9 | M12.5–M12.8 | Update README and all V2 documents from generated scorecards; label current, historical, strict, hybrid, and policy-specific numbers. | README and research narrative can diverge. | One consistent public story with evidence links. | Documentation checker finds 0 unbacked numeric claims and 0 stale target/model counts. |
| [ ] | M12.10 | M0–M11, M12.1–M12.9 | Run full unit, integration, sanitizer, cross-compile, reproducibility, and performance suites on every release target. | Subsystems passed independently. | Signed final audit bundle and release candidate. | 100% of correctness and performance/coverage gates pass in 2 fresh-runner passes per release target. |
| [ ] | M12.11 | M12.10 | Freeze release manifest, artifact/catalog versions, raw evidence hashes, known limits, and rollback procedure. | Development state can drift. | Reproducible V2.3 release record. | At least 1 fresh machine reproduces 100% of scorecard classifications; rollback restores the last valid catalog. |

### Milestone verification

```bash
.venv/bin/python benchmarks/v2/verify_release.py \
  --manifest benchmarks/v2/objective_manifest.json \
  --require-hardware-coverage 1.00 \
  --require-per-target-strict-win-rate 0.80 \
  --require-geomean-speedup 1.05
```

**Done when:** the command exits 0 on two fresh measurement passes for every declared target and all
M0–M12 milestone/task checkboxes are `[x]` with validated evidence.

## 18. Final completion checklist

The implementation agent performs this audit after M12 and may not replace it with a narrative
summary:

- [ ] C1 — All atomic task IDs have validated evidence files and no `[!]` remains.
- [ ] C2 — Objective manifest was frozen before the final tuning runs; its hash is in every scorecard.
- [ ] C3 — Every declared hardware target has real measurements, not cross-compile inference.
- [ ] C4 — Every declared model/bucket cell is a strict win, strict loss, or predeclared ineligible
      cell; none is missing.
- [ ] C5 — Per-target strict-win rate is at least 80%; no target is rescued by averaging with another.
- [ ] C6 — Each winning cell beats the strongest equivalent baseline by the configured p50 margin
      with confidence and has no p99 regression.
- [ ] C7 — Every strict result has exactly one owned compute grid; all copies/resets/submissions are
      visible and conform to its invocation contract.
- [ ] C8 — Reference, quantized, hybrid, and serving results remain separate.
- [ ] C9 — All measured cells pass their frozen correctness/quality policy.
- [ ] C10 — cuBLASDx, CUTLASS collective, direct CuTe, and native CUDA families were enumerated where
      legal; the release contains selected winners only.
- [ ] C11 — H200 winners/resources were never used as A100 measurements or policy.
- [ ] C12 — Normal runtime performs no compilation or tuning and rejects incompatible artifacts.
- [ ] C13 — README and every numeric claim link to immutable evidence.
- [ ] C14 — Full release verification passes twice on fresh runners.

## 19. Agent traversal and handoff record

On every new pass, an agent:

1. reads Sections 1–3 and the normative documents;
2. runs `verify_evidence.py`;
3. finds the first unchecked milestone whose dependencies are complete;
4. selects the lowest unchecked atomic task ID in that milestone;
5. records before evidence before editing;
6. implements only that task and its necessary tests;
7. records after evidence and evaluates the stated gate;
8. marks the task only after evidence validation;
9. reruns the milestone command; and
10. leaves the next exact task ID and any blockers in the handoff table.

| Pass | Agent/revision | Completed IDs | Failed or blocked IDs | Evidence index | Next task |
|---|---|---|---|---|---|
| 0 | planning document | none | none | not yet created | M0.1 |

## 20. Final release statement template

Only after Section 18 is complete may the project publish:

> MegaBake V2.3 is a generic traffic-first transformer-inference compiler. For every hardware target
> in release manifest `<hash>`, its strict single-grid artifacts beat the strongest equivalent
> `torch.compile` baseline on `<rate per target>` of the predeclared model/bucket cells, with
> geometric-mean speedup `<value>`, no winning-cell p99 regression, and all numerical policies
> passing. Hybrid, quantized, and serving results are reported separately.

Until then, the accurate statement is:

> MegaBake V2.3 is under implementation. Completed scorecard cells, strict losses, unrun cells, and
> product fallbacks are reported without extrapolation.
