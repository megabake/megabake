# MegaBake V3: implementation handbook and atomic work orders

Status: proposed execution specification, revised 2026-09-27. No card is certified complete by this document. The old `MB3-001`–`MB3-092` task numbers belong to the earlier FatOp/pipeline-first plan; the `V3R` numbers below describe the revised compiler. Existing MB3 code is inventory, not evidence that the corresponding V3R gate has passed.

Read [architecture](MEGABAKE_V3_ARCHITECTURE.md) for scope, [IR](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) for semantic contracts, [body design](MEGABAKE_V3_KERNEL_REUSE.md) for device-callable math, [pipeline](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) for scheduler legality, [hardware](MEGABAKE_V3_HARDWARE_MODEL.md) for target admission, and [performance](MEGABAKE_V3_PERFORMANCE_MODEL.md) for claims. If a card and its owning contract disagree, record the exact conflict in the handoff and repair the specification before implementing an ambiguous behavior.

## 1. Assignment protocol for an implementation agent

Assign a **bounded, coherent batch** of V3R cards when their dependencies and edit surfaces fit together; a single card is appropriate when it already produces a substantial reviewable result. A batch normally contains two to four cards and ends at one observable compiler or measurement milestone. Select cards by dependency and outcome, not by adjacent numbers. Give the agent this document, the selected cards' dependency handoffs and the owning documents named above. The agent must:

1. Inspect the current checkout and dependency handoffs. Existing filenames below are real as of this revision; `src/megabake/v3/semantics/`, `algorithms/`, `logical/`, `backends/cuda/`, `runtime/` and `benchmarks/v3/` are proposed destinations, not existing packages.
2. Before editing, list the batch's card IDs in dependency order, its single observable outcome, external prerequisites, edit surface, per-card pass/reject criteria and applicable CPU/toolchain/GPU lanes. Every dependency must already have a credible handoff or appear earlier in the batch; code presence alone does not certify a card. Start with the smallest batch that reaches the outcome.
3. For each card in order, state its observable before/after behavior, add its smallest positive and negative regression/fixture, implement it and run the applicable checks. Preserve unrelated changes. If a lane is unavailable, mark it `not_run` or `blocked_external`; a skip is not a pass. A failed prerequisite stops dependent cards; independent cards may continue only if they still serve the declared batch outcome.
4. Stop at the **declared batch boundary**. Do not silently implement downstream cards, turn an `UNKNOWN` fact into a guess, import CUDA in common semantics, or replace a strict failure with an external fallback.
5. Return one schema-valid `TaskHandoff` **per attempted card** (existing schema in `src/megabake/v3/diagnostics.py`) with task revision, dependency revisions, changed files, exact commands/results, evidence vector, diagnostic/counterexample results, artifact hashes, remaining risks and eligible next cards. Finish with a short batch summary that names the achieved outcome, blocked cards and next eligible batch. Raw GPU samples and target identity accompany any speed claim. Do not invent a batch-level `TaskHandoff` schema.

An agent can complete an experiment by documenting a failed candidate. Completion of the **compiler** requires the later gates; card completion does not imply a speedup. No agent should infer device correctness from source inspection, a CPU interpreter, a successful nvcc invocation or a standalone body timing.

A reusable batch assignment message is:

```text
Implement a coherent batch of V3R cards from MEGABAKE_V3_IMPLEMENTATION.md.
Inspect the checkout, card dependencies and existing handoffs first. Read each card's owning V3 contracts.
Select the smallest dependency-closed batch, usually two to four cards, that reaches one reviewable outcome.
Before editing, state its IDs in execution order, prerequisite evidence, pass/reject criteria and edit surface.
For each card, add its positive and negative cases, implement the specified behavior and run applicable checks.
If a prerequisite fails, stop its dependents and report the exact blocker. Do not expand beyond the declared batch.
Record exact commands and artifacts; mark unavailable lanes not_run, never pass.
Return one schema-valid TaskHandoff per attempted card and a concise batch outcome/next-batch summary.
Claim strict speed only with full-step GPU correctness, one-grid evidence and raw matched samples.
```

The following batches are **suggested assignment boundaries**, not additional dependencies. They cover the required first strict path; rows with independent prerequisites can proceed concurrently when separate workspaces and interface ownership are available. A batch may stop after an attempted card if its next dependency lacks the required evidence.

| Batch | Cards in dependency order | Prerequisite handoffs | Reviewable outcome |
|---|---|---|---|
| A — typed step | `001 → 002` | None | Complete step manifest and equivalent tiny stateful FX capture |
| B — real workload | `002H → 003 → 004` | A; declared HF checkpoint and GPU for baseline/inventory | Full cached-step capture, strongest matched baseline and exact hot-shape inventory; split after `002H` if model capture is unresolved |
| C — first embedded math | `005 → 006` | `001`, exact shape descriptor; `004` for final G1 evidence | Standalone/lean-entry harness and K-parallel SIMT body against vendor math |
| D — competing math | `007 → 009` | `003–006` and named target; `007` may find no legal fast tensor-core tactic | Tensor-core Pareto or precise rejection, followed by conservative body-quality go/no-go; `008` is optional and separate |
| E — indexed frontend | `010 → 011 → 012` | `002` | Proven normalization/facts and origin-complete indexed meaning |
| F — generic stateless math | `013 → 014` | E | Generic map/reduce/view and contraction lowering from indexed semantics |
| G — state and structure | `015 → 016` | E | Typed cache effects and executable repeated regions |
| H — semantic choices | `017 → 018` | F and G | Guarded algorithms with complete original-FX/effect coverage verification |
| I — logical plan | `019 → 020` | H | Tile domains, exact access maps, dependencies and lifetimes |
| J — target/body contract | `021 → 022` | `005–007`, `019`; `020` needed by later physical search | Named CUDA profile and compatible device-callable body variants |
| K — physical artifact | `023 → 024` | I and J | Whole-entry candidate search, selected-source compilation and actual-entry admission |
| L — first generic grid | `025 → 026` | F, H and K | One-grid worker execution of an unfamiliar supported FX composition: G2 |
| M — exact attention | `027` | L and `015` | Correct cached attention/KV body and state transition |
| N — full strict step | `028` | B, G and M; all listed card dependencies | Complete cached-step session and measured strict result: G3 |
| O — held-out family | `035` | N | Structural breadth check on a predeclared second decoder family |
| P — second CUDA target | `036`, when hardware exists | N and `021–024` | Rebuilt/admitted entry and measured result on another target |
| Q — scorecard/API | `037 → 038` | O; P when target hardware exists | Complete predeclared scorecard and transparent public V3 compiler: G4 |

B and C can overlap only with an exact provisional shape descriptor; C's measurements count toward G1 after B's real-shape inventory. E–I can progress while B–D measure math quality. J is the main convergence point. A substantial R3 device card may stand alone when pairing would hide its failure mode. Treat `008` as an isolated provider experiment because its toolchain can differ. Trigger optional `029`–`034` only from measured full-step evidence after N; batch an optional card with its own ablation and report, not with an unrelated mechanism. A failed D go/no-go should redirect body work before a large scheduler effort, while E–I remain useful for strict generic correctness.

### 1.1 Evidence vocabulary that matches the current code

`EvidenceVector.gpu_performance` uses `not_measured`, `measured_win`, `measured_loss`, `inconclusive` or `not_applicable`; `TaskHandoff.claim` currently supports `not_measured`, `strict_win`, `strict_loss`, `incorrect` or `unsupported`. These are the values in the checked-in enums. Extend the schema deliberately if the product needs a separate `fallback_only` or `inconclusive` claim; never write a status that only exists in prose. A `strict_win` also requires `cuda_compilation=pass`, `gpu_correctness=pass`, `gpu_performance=measured_win` and retained raw samples. The existing schema checks some of these conditions; future reporting must also validate equivalence and grid count.

CPU lane: no CUDA device query, header lookup, model download or GPU claim during import/collection. CT lane: selected nvcc and headers compile source for a named SM but do not establish runtime legality. GPU lane: target device correctness, actual-entry resources, launch trace and latency. R2 review covers FX/effect/numerical/benchmark semantics. R3 review covers CUDA collective participation, async memory, event generations, forward progress and cooperative residency; a second competent reviewer must inspect an R3 device protocol before it is enabled for accepted strict runs.

### 1.2 Non-negotiable invariants inherited by every card

- Original FX/ExportedProgram reference and every live output/state effect remain executable and traceable. Named pattern recognition is optional; supported indexed primitives have a strict code-generation path.
- The first benchmark is a **complete cached inference step**, not `use_cache=False` sequence-one inference. Initial state is fixed-capacity KV with uniform batch position; old valid length `L` satisfies `0 <= L < C`, the new token uses position `L`, and returned valid length is `L+1`. Other cache/state conventions require a separate declared contract.
- Eval/inference mode, dropout disabled, exact output tree/ownership, weight version, cache layout, masks and numerical policy are fixed before tuning. Every baseline and candidate uses the same callable semantics.
- A strict artifact performs all required device compute in one owned CUDA grid. Host cuBLAS/cuBLASLt, child grids and a captured sequence of kernels belong to an `ExternalPlan`/baseline. Count copies, memset, descriptor updates, binding and return work in the chosen complete-call view.
- The actual **composed** entry determines block shape, registers, shared/local storage, code footprint, occupancy and cooperative grid size. No estimate from a standalone body admits a different binary.
- Unknown cost is `UNKNOWN`, never zero. An unavailable GPU/toolchain is `unvalidated`, not a fabricated measurement. Every final predeclared workload cell remains in the scorecard, including losses and unsupported cases.

## 2. What the current repository gives us

This inventory is specific to the current checkout and is a migration guide. Recheck it before each card; do not delete working code simply because its old abstraction is no longer canonical.

| Existing path | Useful work already present | Required change or limit |
|---|---|---|
| `src/megabake/v3/contracts.py`, `diagnostics.py` | Immutable workload/numerical records, hashes and handoffs | `WorkloadSpec` is an envelope, not yet a typed full cached-step ABI; status fields must stay aligned with real enum values |
| `src/megabake/v3/frontend/capture.py` | Lifted bindings, output signature, effect facts, original callable | Bare `GraphModule` is re-exported with `strict=False`; a direct FX route must preserve exact user-supplied signature and detect changed semantics |
| `frontend/normalize.py`, `facts.py`, `effects.py` | Isolated normalization, tensor facts, effect-aware DCE | Facts are incomplete for symbolic aliases/index maps/cast boundaries; string-based view recognition is not proof of aliasing |
| `frontend/semantic.py`, `match_*.py`, `composites.py` | Reusable recognizer guards and candidate ideas | `SemanticGraph` is a selected named-op cover with leftover `ReferenceRegion`s; those references do **not** supply strict device bodies. Move recognizers behind indexed semantics as guarded algorithm proposals |
| `frontend/layers.py`, `inventory.py` | Structural fingerprints and diagnostics | `LayerSummary` is not an executable `RepeatRegion`; inventory must include all live FX/effects, not only recognized ops |
| `tests/test_v3/fixtures.py`, `tests/test_v3/cpu/` | `LINEAR_TINY`, `GATE_TINY`, `NORM_VARIANTS`, `ATTENTION_TINY`, `STATE_POISON`, CPU regression lanes | Reuse and extend these; they do not establish CUDA compilation, correctness or a cached full-step benchmark |
| `src/megabake/__init__.py`, `src/megabake/integrations/transformers.py` | Legacy public entry and HF wrapper | Current HF wrapper calls `use_cache=False`; keep legacy behavior separately while introducing an explicit V3 cached-step path |
| `src/megabake/schedule_compiler/`, `src/cuda/tasks/`, `src/megabake/runtime/` | Legacy task implementations and one-grid launcher | Useful as controls or selectively verified code; current source concatenates every task, uses `--use_fast_math` and a numeric SM rule; it is not a V3 body ABI |
| `benchmarks/bench_compare.py`, `benchmarks/test_harness.py` | Historical timing paths | Current real-model route is uncached and filters work in counts; build a new complete-step evidence route rather than relabeling those numbers |
| `requirements.txt`, `pyproject.toml` | Existing environment and package metadata | Pinned PyTorch 2.6.0+cu124 and CUTLASS 3.8; current cuBLASDx requires CUDA 13.0+ and CUTLASS 4.4.1+. Its probe needs a separate, explicit toolchain lane |

The two commits implementing old MB3-001–021 mainly establish CPU frontend support. Keep their useful contracts, fixtures and diagnostics. V3R-010–018 must replace the canonical named-cover assumption without erasing existing reference tests; deprecate old names only after the new path and its equivalence checks exist. The older `compile_fx` in `src/megabake/__init__.py` is a legacy callable: a V3 result must expose which compiler path ran.

## 3. First supported slice and fixed test oracles

The first strict subset is an eval-mode, deterministic FP16 or BF16, single-GPU decoder step with stable dense weights, batch 1 or a declared small batch with **uniform** valid length, fixed-capacity contiguous KV state in an explicitly declared layout, no dropout, and no graph break. Start with `[B,Hkv,C,D]` cache as the canonical fixture layout. The original captured output tree is the contract; an adapter may select last-token logits only if the same adapter is used for the reference and all baselines. Nonuniform lengths, paged KV, MoE routing, recurrent state, quantization, training and multi-GPU work are later guarded extensions. A genuinely unsupported primitive is a precise diagnostic, not a claim of broad coverage.

Use checked-in CPU fixture families before downloading a model:

| Fixture | Distinction the compiler must preserve | Minimal negative case |
|---|---|---|
| `LINEAR_TINY`: M=1, N=17, K=33 | Weight orientation, bias, tail K/N, alpha/beta | Treating `[N,K]` as `[K,N]` or losing nonunit alpha/beta |
| `GATE_TINY`: hidden 32, intermediate 65 | SiLU versus GELU, chunk tail and casts | Calling a GELU gate SwiGLU |
| `NORM_VARIANTS`: width 32 | Epsilon inside/outside rsqrt, one-plus-weight and cast placement | Moving epsilon or an FP16 cast |
| `ATTENTION_TINY`: Hq=4, Hkv=2, D=8, C=17 | GQA head map, cache append, positions 0/1/15/16 | Reading unwritten cache or wrong mask alignment |
| `STATE_POISON` | Untouched cache slots preserve sentinel values | A broad overwrite that still gives plausible logits |

Add a synthetic **unfamiliar composition** to those fixtures: transpose/view → `addmm` → explicit cast → reduction → gated pointwise → state write. Do not register its whole pattern in advance. It is the G2 proof that supported primitives, not a model-family FatOp, drive strict codegen. For the full model, select checkpoints/revisions and short/long context buckets in a `WorkloadSpec` before final tuning. The [performance contract](MEGABAKE_V3_PERFORMANCE_MODEL.md) defines the scorecard and held-out trials.

### 3.1 Reference and numerical checks

For each transformed region, compare exact output structure, shape/dtype, integer/index values, cache untouched regions and exceptional-value positions. Use the policy's operation/dtype tolerance for floating values. Run multiple seeds and edge cases, including zero, cancellation, tails, masks and final cache capacity. A result that matches logits while corrupting state fails. A rule that is algebraically valid in real arithmetic but moves a declared cast fails unless an explicitly recorded tolerance and validation permit it. Preserve an untransformed path for any rejected rule.

### 3.2 Minimum primitive subset before the first HF claim

| Kind | First generic semantic/body route | Out of first subset without an explicit extension |
|---|---|---|
| Scalar maps | `add/sub/mul/div`, `neg`, comparison/`where`, explicit `to`, `exp`, `rsqrt`, supported SiLU/GELU spelling | Arbitrary Python callback, RNG, training/backward |
| Reductions | `sum/mean/max` over proved finite axes, keepdim, accumulator and final cast | Unbounded custom reduction or unsupported precision |
| Contractions | `mm/bmm/addmm`, weight views/transposes, alpha/beta/bias, bounded K | Sparse/complex or opaque vendor-only op without reference expansion |
| Views/indexing | Proven `view/reshape/transpose/permute/slice/select/expand` maps and bounded gather | Unknown alias used for write or unbounded indirect index |
| State | Fixed-capacity append/functional write and exact old/new cache path | Paged, nonuniform per-example lengths or undeclared in-place alias |
| Attention | Conservative SDPA/decomposed reference plus V3R-027 guarded online-softmax body, causal/window/GQA as declared | Nonzero dropout or unsupported mask/state semantics |

This table is a **minimum G2/G3 engineering target**, not a statement that all listed forms are implemented today. The full HF capture inventory may add more primitive forms; a strict full-step result requires every live form to be lowered or a precise unsupported diagnostic. A named `RMSNorm`, `RoPE` or `SwiGLU` recognizer improves algorithms but is not required for generic coverage when its constituent primitives are supported.

## 4. Handoff locations and dependency graph

Proposed V3 ownership (paths may be combined until a second user exists):

```text
src/megabake/v3/
  contracts.py, diagnostics.py                 existing; extend carefully
  frontend/                                    existing capture/normalize/facts; migrate matchers
  semantics/                                   proposed indexed values/ops/effects/reference/verifier
  algorithms/                                  proposed guarded equivalence choices and repeat regions
  logical/                                     proposed tile domains, dependence, storage, verifier
  backends/cuda/                               proposed profile, body providers, physical plan, codegen
  runtime/                                     proposed typed session, dispatch, V3 API
benchmarks/v3/                                 proposed manifests, baseline/body/full-step reports
src/cuda/v3/                                   proposed generated entry and reusable CUDA body components
tests/test_v3/{cpu,toolchain,gpu}/             CPU exists; toolchain/GPU are proposed
```

The common `semantics/`, `algorithms/` and `logical/` packages must import without initializing CUDA. Existing source in a legacy package remains usable as an oracle or control until a focused migration card changes it. Do not import all legacy `.cu` task files into the selected V3 entry.

| Gate | Required cards | Observable exit | Allowed outcome |
|---|---|---|---|
| G0 workload truth | 001–004, including 002H | Complete cached reference, strongest matched baseline, exact hot-shape inventory | valid / unsupported capture / unavailable GPU |
| G1 math feasibility | 005–007, 009; 008 optional | Correct target-qualified bodies, lean-entry resource and cuBLAS gap evidence | plausible / loss / unavailable GPU |
| G2 generic compilation | 010–026 | Unfamiliar supported FX composition compiles to a correct one-grid block | pass / diagnostic / unavailable toolchain or GPU |
| G3 full strict step | 027–028 | Correct advancing cache and output, admitted one-grid entry, full-call score | win / loss / incorrect / unavailable GPU |
| G4 repeatability | 035, 037–038; 036 when available | Held-out family, second SM when available, all cells reported | per-cell result; no universal inference |

G1 body work can proceed in parallel with G0 capture on already known exact shapes; its latency only becomes full-step evidence after G0. Early body probes use a provisional descriptor with the same indexed-contract fields; V3R-022 integrates them after V3R-012/014. G2 semantics must be established before a full strict step. V3R-029–034 are optional optimization cards triggered by measured opportunity; they are not G3 or G4 correctness prerequisites. A body candidate rejected by G1 is not a reason to stop generic correctness work, but it should prevent a large scheduler project from being sold as the performance solution.

## 5. Atomic cards: workload truth and competitive body probes

### V3R-001 — Freeze a complete invocation manifest

**Depends / lane / review:** none; CPU, R2. **Edit surface:** extend `src/megabake/v3/contracts.py` only if a typed field is missing; add manifest fixtures under `benchmarks/v3/` and contract tests under `tests/test_v3/cpu/`.

**Implement:** keep `WorkloadSpec` for benchmark intent and add the versioned typed `StepABI/v1` defined in the [IR ABI contract](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#14-first-strict-cached-step-abi-and-public-compiler-surface). Distinguish model inputs, lifted stable weights, old/new state paths, caller-owned versus runtime-owned outputs, cache layout/capacity, uniform valid length, mask/position convention, setup amortization and declared benchmark cells. Set `timed_unit=cached_step`. Record checkpoint revision, graph hash, PyTorch/CUDA/Transformers versions and fixed inputs/seeds without embedding raw tensor pointers in hashes. Do not interpret free-text `state_semantics` as a launcher ABI.

**Pass:** serialize/deserialize/hash a complete step; changing state layout or numerical policy changes the hash; advancing and fixed-state replay are distinct modes. **Reject:** `use_cache=False` or missing old/new state cannot be labelled cached decode. **Handoff:** one JSON manifest, its hash and a table of not-yet-measured cells.

### V3R-002 — Capture ABI on a tiny complete FX step

**Depends / lane / review:** 001; CPU, R2. **Edit surface:** `frontend/capture.py`, an optional distinct HF cached-step helper under `src/megabake/v3/`, and CPU capture fixtures; do not change legacy `from_pretrained` behavior.

**Implement:** accept `ExportedProgram` with full graph signature/lifted bindings/effects; accept bare `GraphModule` with explicit `example_args`, input/output tree and state bindings. Do not make re-export a hidden semantic requirement: if using `torch.export.export(..., strict=False)` as an adapter, compare its resulting graph signature and reference behavior, and reject changed or fragmented capture. Track every lifted weight/state binding, tied weight, mutation and live user output. Capture a tiny stateful decoder step; V3R-002H owns the chosen HF step.

**Pass:** two consecutive calls to original and captured **tiny** reference agree in output tree, logits and advancing state at positions 0 and a nonzero position. **Reject:** a graph-break fragment, unbound lifted parameter, unknown Python callback or omitted cache effect. **Handoff:** capture signature, node/effect inventory and exact unsupported-node diagnostics; this unblocks generic semantic work but not G0's real-model comparison.

### V3R-002H — Capture the declared Hugging Face cached step

**Depends / lane / review:** 001–002; CPU/GPU as the model requires, R2. **Edit surface:** a distinct V3 HF capture helper and `benchmarks/v3/` workload manifest/fixture; do not relabel the legacy `use_cache=False` wrapper.

**Implement:** choose checkpoint/revision, valid lengths, cache capacity/layout, dtype and output tree in the manifest before optimizing. Build an eval-mode one-token step with explicit old/new cache and model weights, then capture a **complete** FX/ExportedProgram and validate it against the original HF callable over advancing steps. Record graph breaks and unsupported custom operations with exact FX origins; if a checkpoint cannot be captured under the declared contract, keep it as an unsupported cell and record the reason before choosing another declared trial.

**Pass:** complete graph signature contains all inputs/outputs/effects; logits and state agree at short and long valid lengths and over consecutive calls. **Reject:** partial graph, stale replayed cache, silently changed output tree or `use_cache=False` substitute. **Handoff:** pinned model/revision, full-step reference/capture hash and unsupported-node report. This is a G0 dependency for the model baseline, not a dependency for V3R-010's tiny generic semantics.

### V3R-003 — Establish the matched strongest `torch.compile` baseline

**Depends / lane / review:** 001, 002H; GPU, R2. **Edit surface:** new `benchmarks/v3/` harness and report schema; leave historical benchmark JSON untouched.

**Implement:** run the same reference adapter/weights/inputs/cache/output lifetime through legal `torch.compile` default, `reduce-overhead`, `max-autotune` and stable-address graph capture where applicable. Warm compilation/autotune and verify real capture, graph breaks, recompiles, vendor calls, preparation and output ownership. Keep all device operations in unfiltered traces; profile separately from final timing. Retain raw complete-call samples and selected best equivalent path. Save the selected Inductor tactic/template, generated source and autotune alternatives when exposed by the pinned version; these are body-search evidence, not assumed device-callable implementations.

**Pass:** each accepted baseline passes the same numerical/state oracle and has a reproducible config, operation trace, sample series and setup record. **Reject:** faster precision, stale replayed state, excluded real elementwise work or a path with different output ownership as a matched baseline. **Handoff:** best validated baseline ID per cell plus losses/failures of other modes.

### V3R-004 — Inventory exact hot shapes and removable work

**Depends / lane / review:** 002H, 003; GPU, R2. **Edit surface:** `frontend/inventory.py` or `benchmarks/v3/shape_inventory.py`, plus report tests.

**Implement:** for each contraction/attention in the full step record B/M/N/K, effective strides/transposes, dtype, accumulation, bias/cast/epilogue, call count, selected vendor/Inductor template path, weight preparation, cache condition and critical-path contribution. Record inspectable Inductor tile/layout/padding/fusion choices with their FX origin and selected target; mark opaque private-library details unknown. Separate semantic bytes, global-address-space bytes and measured HBM/L2 bytes. Identify candidate fused boundary savings only from traces, never by summing assumed launch costs.

**Pass:** inventory covers all live hot math including unrecognized FX regions and matches observed baseline operations; each shape has an origin FX region. **Reject:** a tensor-core microbenchmark under hot L2 cannot be compared as a whole-model HBM-bound body rate. **Handoff:** ranked hot-shape list and conservative body-quality budget for the first target.

### V3R-005 — Build a standalone and lean owner-entry body harness

**Depends / lane / review:** 001 and an exact shape descriptor; 004 is required before its result counts for G1; CT/GPU, R3 for entry. **Edit surface:** `benchmarks/v3/` body harness, minimal generated CUDA entry in `src/cuda/v3/`, target/profile records under `backends/cuda/`.

**Implement:** the identical body source must run standalone and as a callable tile inside an admitted minimal cooperative owner grid with realistic argument count, live scratch, CTA shape and repeated tile iteration. Time selected cuBLAS/cuBLASLt external operations with the same data/layout/precision; measure descriptor creation and any packing separately. Collect compiled registers, shared/local memory, spills, code size, occupancy, grid size and raw timings.

**Pass:** correctness and resource report for both placements, no hidden compute launch inside the owner entry, and vendor-relative raw samples for at least one projection shape. **Reject:** a standalone win whose body cannot run inside the resident entry is recorded as incompatible, not as a strict win. **Handoff:** harness source/artifact hashes and exact target/toolchain identity.

### V3R-006 — Generate K-parallel SIMT low-batch tactics

**Depends / lane / review:** 005 and an exact shape descriptor; 004 supplies real-model shapes for final G1; CT/GPU, R3. **Edit surface:** `backends/cuda/bodies/` or `src/cuda/v3/` body generator, body harness fixtures and GPU cases.

**Implement:** from indexed contraction semantics generate row tiles, lanes/warps per row, vector widths and reduction trees. Lanes cooperate across K; no thread serially owns an entire long K by default. Accept logical tile coordinates independent of `blockIdx`, handle K/N tails, declared accumulation/cast/epilogue and arbitrary guarded weight orientation. Search a bounded schedule set on exact shapes with compile-time feature/resource rejection.

**Pass:** `LINEAR_TINY` odd/tail case and representative hot model shapes are correct standalone and in lean entry; Pareto records include latency, resources, output tile count and selected schedule. **Reject:** wrong alpha/beta, weight transpose, out-of-bounds tail or underfilled historical V2 mapping. **Handoff:** selected and rejected schedules per target; no model-name dispatch.

### V3R-007 — Generate an output-channel-major tensor-core tactic family

**Depends / lane / review:** 005 and an exact shape descriptor; 004 supplies real-model shapes for final G1; CT/GPU, R3. **Edit surface:** target-qualified CUDA body generator and harness, not common semantics.

**Implement:** treat `Yᵀ = W Xᵀ` as a guarded algorithm choice for `Y=XWᵀ`; enumerate legal output/K tiles, warp/CTA roles, target MMA/copy primitive, mainloop depth, padding, optional split-K, output predication and epilogue. Include real global loads/stores and any layout preparation. A target may have no legal fast tensor-core variant and must retain SIMT.

**Pass:** correct FP16/BF16 policy and active stores for small N, large N and tails; standalone and lean-entry Pareto records expose wasted padded work, output tile count, resources and numerical differences. **Reject:** do not claim a nine-output-tile N=576 case is fast simply because MMA throughput is high. **Handoff:** exact shape/SM/toolchain guard and body-quality comparison to selected vendor path.

### V3R-008 — Probe optional cuBLASDx and CUTLASS body reuse

**Depends / lane / review:** 005; separate CT/GPU experiment, R3. **Edit surface:** isolated `backends/cuda/bodies/` provider prototype and experiment report; do not upgrade the core environment silently.

**Implement:** the pinned `torch==2.6.0+cu124` wheel and `nvidia-cutlass==3.8.0.0` do not themselves provide the CUDA 13.0+ toolkit and CUTLASS 4.4.1+ required by the current cuBLASDx release. Query the actual nvcc, then use a separate named toolkit/provider lane if needed. Probe the **pipelined global GEMM** API, host-created descriptor and `__grid_constant__` entry argument, `get_block_dim()`, divisibility/depth, `reset_tile`, accumulator/warp-specialization tradeoffs and coexistence with other bodies. Independently probe adapted CUTLASS/CuTe C++ collectives under the selected toolkit. Preserve source notices/licenses.

**Pass:** produce compatibility, correctness, standalone and lean-entry records or an explicit `rejected_candidate` reason. **Reject:** a cuBLAS host call, private `CUfunction` or launchable DSL kernel is not device-callable body reuse. **Handoff:** provider version/toolchain matrix and exact API/entry limitation; this card is not required to block 009 when another tensor-core tactic exists.

### V3R-009 — Compute the first body-quality budget and go/no-go

**Depends / lane / review:** 003–007; 008 optional, GPU/R2. **Edit surface:** `benchmarks/v3/` report generator and [performance model](MEGABAKE_V3_PERFORMANCE_MODEL.md) if a formula needs correction.

**Implement:** use measured baseline complete-call and hot-body rates to bound removable launch/handoff/traffic work and persistent coordination. Keep uncertainty and contention explicit. Compare body plus layout/packing alternatives, not just a GEMM instruction. Preserve a fused body with plausible paid savings even if its isolated timing loses. If every hot tactic fails the conservative budget, record the strict risk and return to body algorithms before pursuing a large queue/pipeline runtime.

**Pass:** one reviewed worksheet per first target with raw source samples, assumptions, plausible/not-plausible candidates and next experiment. **Reject:** no rigid per-op veto, no made-up overlap, no inference that a 4–6× hot-body gap will disappear from launch savings alone. **Handoff:** G1 result with a candidate Pareto set and strongest counterfactual.

## 6. Atomic cards: FX semantics without a FatOp coverage gate

### V3R-010 — Preserve capture while normalizing FX

**Depends / lane / review:** 002; CPU, R2. **Edit surface:** `frontend/normalize.py`, `frontend/capture.py` only for binding propagation, CPU normalization tests.

**Implement:** pin the normalization route per PyTorch version. For an `ExportedProgram`, use a copied `torch.export.default_decompositions()` table and `run_decompositions` for a selected inference subset while retaining guarded attention/recurrence forms where useful. A bare `GraphModule` keeps its explicit ABI and must not require hidden re-export. Apply any individually selected Inductor FX rewrite only through a versioned adapter on a copy; record exact pass/config, avoid global table mutation and do not run the whole `pre_grad_passes`/`post_grad_passes` sequence by default. Retain original executable FX, lifted bindings, guards, effect facts, output tree and origin map. Compare original and normalized behavior on representative pure and stateful inputs before downstream use. The current adapter handles replacement-returning passes; extend it so newly inserted/deleted nodes retain stable origin provenance and no state effect disappears.

**Pass:** pure and effectful fixtures agree, two normalization requests do not share mutable pass state, and CPU import does not query CUDA. **Reject:** a pass that changes cache state, output structure or cast boundary. **Handoff:** selected PyTorch version/path and origin mapping for every transformed node.

### V3R-011 — Collect proven tensor, alias and effect facts

**Depends / lane / review:** 010; CPU, R2. **Edit surface:** `frontend/facts.py`, `frontend/effects.py`, `semantics/` fact records and CPU guards.

**Implement:** collect shape and symbolic constraints, dtype, exact strides/offset, alignment only when proved, alias sets, mutability, producer/consumer uses, effect roots and numerical cast points. Consume export constraints and fake-tensor/SymInt metadata as candidate facts, then establish runtime guards for every body assumption; metadata alone cannot prove physical pointer alignment or storage aliasing. Replace string-substring view inference with operator-specific transfer functions. A `reshape` that copies and a view that aliases must differ. Tied lifted weights share storage identity. Guards must cover every fact later assumed by a body.

**Pass:** transpose, expand, noncontiguous reshape, tied weight and cache-write fixtures have correct facts; unknown alias or alignment remains `UNKNOWN`. **Reject:** expanded zero-stride write or in-place overlap without proof. **Handoff:** fact schema, guard coverage and an unsupported/unknown-fact inventory.

### V3R-012 — Define `IndexedTensorProgram` and local FX origins

**Depends / lane / review:** 011; CPU, R2. **Edit surface:** proposed `semantics/indexed.py`, `semantics/reference.py`, `semantics/verify.py`; adapt `frontend/semantic.py` without deleting its old tests.

**Implement:** introduce typed value IDs and operations for `Map`, `Broadcast/View`, `Reduce`, `Contraction`, `Gather`, `Scatter/StateWrite` and a guarded `Scan/Branch` extension point. Each op has iteration/reduction domains, exact input/output index maps, predicate/bounds, dtype/cast expression, alias/effect edges and FX-origin subgraph. Make a local reference evaluator for each region; the current `ReferenceRegion.reference=program.run_reference` is a whole-model callable and must not be mistaken for a local proof. Keep unsupported custom operators as precise diagnostics.

**Pass:** `LINEAR_TINY` and a map/reduce/view graph become indexed programs with every live FX origin and output mapped exactly once or explicitly pure-recomputed. **Reject:** unmatched live FX nodes cannot remain reference-only while `strict_supported=True`. **Handoff:** serialized example IR and a coverage report mapping every original FX node/effect to an indexed op or diagnostic.

### V3R-013 — Lower maps, broadcasts, views and reductions generically

**Depends / lane / review:** 012; CPU then CT, R2. **Edit surface:** `semantics/` lowering, generic CUDA emitter under `backends/cuda/bodies/`, focused CPU/toolchain/GPU tests as access permits.

**Implement:** support arithmetic/unary ops used by the chosen decoder, explicit scalar constants, `where`/mask, dtype conversion, broadcast, transpose/slice/reshape with proven semantics, and supported reductions with declared initial value, axis, accumulation and final cast. First run a bounded PyTorch-2.6-pinned bridge probe on one pure map/reduction fixture: inspect whether Inductor `Pointwise`/`Reduction` loop expressions can be translated into the same `IndexedOp` with exact FX origins, index maps, guards and casts. Record adoption or a precise rejection; do not make this private bridge a coverage prerequisite. Generate a slow but correct CUDA tile body from the indexed expression without calling a named `RMSNorm` or `SwiGLU` matcher. Preserve input stride and tail masks; reject unsupported numerical functions explicitly.

**Pass:** unfamiliar reorderings of supported map/reduce/view nodes reach generated source and match local FX reference on `GATE_TINY`/`NORM_VARIANTS`. **Reject:** moving epsilon across rsqrt, hoisting a half cast or writing through expanded storage. **Handoff:** supported primitive table, generated source sample and numerical policy coverage.

### V3R-014 — Lower contractions generically

**Depends / lane / review:** 012–013; CPU/CT/GPU, R2/R3. **Edit surface:** `semantics/` contraction lowering, `backends/cuda/bodies/` generic correctness body, fixture cases.

**Implement:** represent `mm`, `bmm`, `addmm` and supported einsum-like contractions as indexed loops with explicit index maps and reduction axes; preserve transpose/stride, alpha/beta, bias and cast order. First emit a conservative complete-reduction body for the supported subset. Connect V3R-006/007 tactic generators through the same semantic descriptor only after the generic reference path is correct. Specialization guards reject incompatible layouts rather than silently transposing data.

**Pass:** `LINEAR_TINY` [N,K] and [K,N] variants, nonunit alpha/beta and tail shapes share the correct semantic core and produce legal body source. **Reject:** `[K,N]` weight cannot be read with `[N,K]` strides; unsupported sparse/complex contraction yields a diagnostic. **Handoff:** exact contraction equation for each ATen spelling and generic CUDA/body comparison.

### V3R-015 — Make cache, indexing and state effects first class

**Depends / lane / review:** 012; CPU/CT/GPU, R2/R3. **Edit surface:** `frontend/match_state.py` as optional recognition, `semantics/` effects and generic scatter/gather, state fixture cases.

**Implement:** define old/new state values, append location, valid-length transition, write footprint, alias rule and ordering for a fixed-capacity cache. Functional state is the first path; an in-place state path needs a separately guarded alias contract. Support the required gather/slice/scatter primitive forms with bounded indices. State writes remain roots through DCE and retain untouched cache bytes. A cache read of the current token depends on the write's publication.

**Pass:** `ATTENTION_TINY` positions 0,1,15,16 and `STATE_POISON` agree for output, new state and untouched slots across consecutive steps. **Reject:** out-of-capacity write, negative/unknown index, duplicated effect or read-before-publish. **Handoff:** typed effect edges and state transition reference.

### V3R-016 — Recover executable repeated regions

**Depends / lane / review:** 012, 015; CPU, R2. **Edit surface:** `frontend/layers.py`, proposed `algorithms/repeat.py` and CPU structural tests.

**Implement:** use graph connectivity and per-layer bindings to promote matching structural spans into `RepeatRegion` with bounded iteration count, per-iteration weight/state table, carried values, branch predicates and entry/exit reference. `LayerSummary` may suggest spans but a matching hash is not sufficient proof. Keep exceptional layers separate, or record a guarded variant in the region. An expansion back to the original flat FX order is required.

**Pass:** repeated tiny blocks round-trip to flat reference, preserve distinct weights and cache effects, and show the exact layer whose norm/mask/recurrence differs. **Reject:** two similar operations with different cast or state semantics cannot be collapsed. **Handoff:** verified repeat mapping and flat-versus-loop reference record.

### V3R-017 — Enumerate guarded algorithm and layout choices

**Depends / lane / review:** 012–016; CPU, R2. **Edit surface:** proposed `algorithms/`, adapt existing `match_*.py` and `composites.py` as rule sources, CPU equivalence fixtures.

**Implement:** turn named matchers into `match + guards + reference relation + candidate expansion`, keeping the unfused indexed path. Initial choices: K-parallel versus transposed tensor-core projection, optional stable weight pack, separate/packed QKV, full/streamed gated MLP, exact RMSNorm/SDPA online-softmax choices and repeat unroll. Record preparation cost and numerical permission. Recognizer failure must not mark ordinary supported primitives unsupported.

**Pass:** overlapping candidates coexist; each has a machine-checkable guard, exact origin set, covered outputs/effects and reference expansion. `GATE_TINY` GELU and SwiGLU remain distinct. **Reject:** no semantic choice based solely on class name, graph adjacency or an unproved associativity. **Handoff:** candidate inventory for one real FX block with unfused control retained.

### V3R-018 — Verify semantic coverage and numerical policy

**Depends / lane / review:** 012–017; CPU, R2. **Edit surface:** `semantics/verify.py`, candidate-cover validation and CPU negative fixtures.

**Implement:** prove that every original live FX output and effect is implemented exactly once, except explicitly costed duplication of pure work. Check each candidate's guards, casts, alias reads/writes, state ordering and reference relation; differential checks are evidence under a declared tolerance, not an invented mathematical proof. The existing `validate_cover` only catches duplicate named operations/effects and must be strengthened for complete original-FX coverage.

**Pass:** valid unfused indexed path covers the supported graph; omitting residual, intermediate cast, cache write or final output fails with origin IDs. **Reject:** a `ReferenceRegion` without a device lowerer is not strict coverage. **Handoff:** machine-readable semantic verification report attached to every later plan.

## 7. Atomic cards: logical work and first CUDA entry

### V3R-019 — Build parametric tile domains and access maps

**Depends / lane / review:** 018; CPU, R2. **Edit surface:** proposed `logical/domains.py`, `logical/maps.py`, CPU enumerator fixtures.

**Implement:** lower one selected algorithm alternative into guarded tile families with output/reduction tile coordinates, exact read/write regions and logical tail predicates. Domain cardinality can depend on guarded B, valid length or routing result when bounded; it is independent of resident CTA count. A body tile change regenerates the affected domains. For a reduction, represent contributors and finalizer explicitly.

**Pass:** enumerate tiny domains and compare each region map to direct FX-index oracle; `LINEAR_TINY` covers all 17 outputs and 33 K elements exactly, with legal tails. **Reject:** no missing or double-written output; unknown data-dependent cardinality cannot be assumed static. **Handoff:** symbolic domain and concrete tiny enumeration.

### V3R-020 — Prove dependencies, lifetime and state order

**Depends / lane / review:** 019; CPU, R2. **Edit surface:** proposed `logical/dependence.py`, `logical/storage.py`, `logical/verify.py`, small interleaving fixtures.

**Implement:** derive producer/consumer relations from region overlap and effects, including reduction finalization and cache publication. Record address-known, data-ready, copy-issued, source-retired, destination-visible, compute-complete, published and reusable as distinct conditions. Build partial-order lifetimes; allow pure recomputation only when declared. A conservative materialized/barrier plan must always be derivable for supported graphs.

**Pass:** tiny task enumeration/interpretation agrees with reference, alternative legal schedules give the same output/state, and buffer overlay requires all old readers retired. **Reject:** a cycle, missing producer, premature reduction finalizer, source reuse before async retire or cache read before write. **Handoff:** logical verifier report and one counterexample trace per rejected case.

### V3R-021 — Query a target-qualified CUDA profile

**Depends / lane / review:** 005, 019; CT/GPU, R2. **Edit surface:** proposed `backends/cuda/profile.py`, target schema and toolchain/GPU cases.

**Implement:** query selected device/visible partition for compute capability, resource limits, cooperative support and selected feature attributes; record nvcc/runtime/driver/CUTLASS/provider versions and exact source of each fact. Keep architecture-specific `_a` and family-specific `_f` feature legality distinct. Do not use `SM>=90` as a proxy for WGMMA/TMA/tcgen05, or infer a full GPU's resource count from its name. Calibrated latency is separate from hard facts.

**Pass:** profile round-trips with provenance and reports unknown features explicitly; a synthetic profile can reject an unsupported body without importing CUDA in common IR. **Reject:** one target's cubin/tuning record cannot be relabelled as another's measurement. **Handoff:** profile key and feature legality matrix for the selected environment.

### V3R-022 — Define body ABI and schedule-variant registry

**Depends / lane / review:** 006–007, 019, 021; 008 optional; CT/GPU, R3. **Edit surface:** proposed `backends/cuda/bodies/` registry and `src/cuda/v3/` interface; CPU schema/toolchain/device cases.

**Implement:** a provider accepts indexed semantics, algorithm choice, numerical/layout guards and target; emits bounded variants with output/reduction footprint, logical tile coordinate, block/warp roles, movement/stage behavior, descriptor binding, scratch/accumulator lifetime, epilogue and publication contract. Register generic, SIMT, tensor-core and optional library tactics. Tactics needing a host-created descriptor declare entry ABI and lifetime; `ATOMIC_TILE`, `PRELOADABLE`, `STREAM_REDUCTION` and `EARLY_RELEASE` are separately proven capabilities. Cache by semantics/shape/target/toolchain, never model name.

**Pass:** same semantic contraction queries multiple legal tactics and rejects invalid shape/SM/block/toolchain combinations before codegen; selected tactic reports correct output tile maps. **Reject:** private cublasLt `CUfunction` or standalone launchable kernel is not a strict body. **Handoff:** example `BodyTacticSpec` records and compatibility matrix.

### V3R-023 — Search bounded whole-entry physical candidates

**Depends / lane / review:** 020–022; CPU/CT/GPU, R3. **Edit surface:** proposed `backends/cuda/plan.py`, `search.py`, resource/progress checks.

**Implement:** choose body tactics jointly with CTA block shape, worker count, tile ownership, memory placement, event protocol, scratch sharing and optional fusion/materialization. Start from a barrier control. Use a bounded beam or coordinate search: reject illegal/clearly dominated variants, compile finalists, retain measured alternatives. Search may request new logical tile granularity, which triggers V3R-019/020 re-verification. Penalize register/shared/code footprint and candidate-specific setup; do not add isolated speedups as if independent.

**Pass:** every candidate has semantic/logical hashes, tactic versions, resource estimates/unknowns, progress obligations and a control counterpart; incompatible block dimensions or unsafe waits are rejected. **Reject:** selecting each individually fastest body cannot bypass the common-entry resource and participant constraints. **Handoff:** candidate frontier with rejection reasons and search budget.

### V3R-024 — Emit, compile and admit the actual selected entry

**Depends / lane / review:** 021–023; CT/GPU, R3. **Edit surface:** proposed `backends/cuda/source.py`, `compile.py`, `admit.py`, `src/cuda/v3/` entry, artifact cache.

**Implement:** emit only selected tactics and worker program; no all-task source concatenation or inherited `--use_fast_math` when numerical policy forbids it. Compile for a named feature target; record actual register/shared/local/spill/code data, required launch attributes, block compatibility and kernel argument/descriptor footprint. Use CUDA occupancy/cooperative APIs on the **compiled function** to bound resident worker grid, then replan or reject. Artifact keys include semantics, guards, numerical policy, target, toolchain/provider, tactic mix and binding ABI.

**Pass:** compiled metadata belongs to the exact launched binary; illegal shared memory, block shape, cooperative grid or required instruction set yields a diagnostic before launch. **Reject:** standalone occupancy or a hard-coded SM count cannot admit the fused entry. **Handoff:** source hash, binary hash, command, resource report and admission decision.

### V3R-025 — Run a minimal one-grid resident worker program

**Depends / lane / review:** 020, 023–024; CT/GPU, R3. **Edit surface:** CUDA entry, physical schedule/progress verifier and GPU cases.

**Implement:** use an admitted cooperative grid with one low-overhead ordered/static dispatch and conservative uniform joins. Initialize all state/event generations before any wait; publish output after appropriate visibility fence; acquire before consuming; terminate all workers. `blockIdx` identifies a worker, not the semantic tile. All CTAs in a grid join must reach it, including idle workers. Use a watchdog in diagnostic runs, not as a correctness substitute.

**Pass:** a small multi-op graph executes in exactly one compute grid, has correct outputs and no deadlock over repeated invocations; trace shows no child/vendor grid. **Reject:** waiting for a producer that cannot become resident, stale event generation, partial collective participation or overwritten scratch. **Handoff:** progress proof sketch, trace and device correctness report.

### V3R-026 — Compile an unfamiliar supported FX block end to end

**Depends / lane / review:** 013–014, 018–025; CPU/CT/GPU, R2/R3. **Edit surface:** integration bridge under `src/megabake/v3/`, CPU/toolchain/GPU block fixtures and small benchmark.

**Implement:** feed the predeclared unfamiliar composition from §3 through capture → indexed lowering → algorithms (unfused choice valid) → logical tiles → body search → exact entry admission → one-grid execution. It must not call a checkpoint/family matcher. Compare every intermediate region against local FX reference in diagnostic mode and full output/state in production mode. Also run a variant differing only in cast or view semantics.

**Pass:** correct one-grid block and measured complete-block latency versus matched `torch.compile`, even if slower; generated source/plan shows generic primitive coverage. **Reject:** unsupported node reports its FX origin; a cast/alias variant cannot silently reuse a guard-incompatible body. **Handoff:** G2 coverage map, source/plan hashes, trace, result and explicit remaining full-step gaps.

## 8. Atomic cards: full cached step and optional mechanisms

### V3R-027 — Implement exact cached attention and KV transition

**Depends / lane / review:** 015, 018–026; CPU/CT/GPU, R3. **Edit surface:** indexed attention algorithm rules, CUDA attention body provider, cache effect protocol and GPU fixture cases.

**Implement:** start with the declared contiguous fixed-capacity KV layout and uniform valid length. Define Q head → KV head mapping for GQA/MQA, score scale, causal/window mask alignment, optional RoPE timing, current-token K/V publication, online max/sum/value accumulation and final cast. Retain a materialized reference algorithm; add split-context only with mathematically correct partial max/normalizer/value combine. Treat state write as a distinct effect even when attention output does not expose it.

**Pass:** `ATTENTION_TINY` and `STATE_POISON` at positions 0/1/15/16, long-context diagnostic, consecutive advancing steps and all masked-row edge cases agree with reference under policy. **Reject:** reading current-token cache before publish, including masked scores in denominator, wrong GQA mapping or `-inf - -inf` NaN. **Handoff:** attention algorithm guard/body resources, state proof and target-specific exact-shape comparison to strong external attention.

### V3R-028 — Deliver a correct complete strict cached-step session

**Depends / lane / review:** 001–003 including 002H, 016–027; GPU, R3. **Edit surface:** proposed `runtime/session.py`, V3 `compile_fx` entry, CUDA binding/launch and full-step tests/report.

**Implement:** bind stable weights, optional prepared layouts, old/new cache, token/positions/masks, scratch, output storage and any descriptors in a typed session. Validate guards on every call; choose a compiled/admitted `StrictGridPlan`, launch exactly one owner compute grid, retain output/state according to the manifest, and charge runtime-required preparation/copies/binding/return under the chosen timing view. Do not silently call legacy `compile_fx` or vendor math when strict is requested. A flat whole-model schedule is legal before repeat-loop optimization.

**Pass:** full output tree and cache state match the captured reference over several consecutive token steps; unfiltered trace contains one compute grid; exact entry resources and complete-call latency are recorded against V3R-003's strongest equivalent baseline. A measured loss is a valid G3 result. **Reject:** hidden child/vendor compute grid, stale cache position, output buffer reused while caller holds it, or an unsupported FX region quietly run outside the grid. **Handoff:** G3 status, source/plan/resource hashes, correctness report, raw samples and fallback path if any.

### V3R-029 — Search repeated-region lowering and code size

**Depends / lane / review:** 016, 020, 028; CPU/GPU, R2/R3. **Edit surface:** repeat lowering, codegen/search and model-wide diagnostic report.

**Implement:** compare flat specialization, device loop and partial unroll using the verified `RepeatRegion` and per-layer binding table. Preserve exceptional layer variants. Price code size, compile time, instruction-cache pressure, register envelope, pointer/descriptor binding and loss of cross-layer specialization. Layer N+1's weight address may be known early even if its activation is not; expose that in logical dependence, not as mandatory prefetch.

**Pass:** every candidate expands to the same layer-specific reference and complete state; whole-entry resources/latency select one option per target cell. **Reject:** grouping two different masks or recurrent branches because their `LayerSummary` hash looked similar; shorter source is not automatically faster. **Handoff:** selected lowering plus size/latency tradeoff.

### V3R-030 — Search fusion, materialization and ownership boundaries

**Depends / lane / review:** 017, 022–028; 029 optional; GPU, R3. **Edit surface:** algorithm/physical candidate search and mechanism ablation report.

**Implement:** enumerate guarded QKV, gate/up, norm+projection, epilogue/residual, local-forward versus global-materialize and pure recomputation choices. Derive exact producer/consumer tile regions after each body/tile change. Keep a matched unfused plan and vendor-strength `ExternalPlan` control. Compile each finalist as a whole entry; a fused body can change register count, CTA shape, output granularity and head readiness.

**Pass:** each promoted fusion passes original-FX output/state/numerical checks and improves measured complete call or remains a documented candidate; body and entry resources accompany timing. **Reject:** a launch saved by spilling a hot GEMM, duplicated state effect or cast moved across a nonlinear op. **Handoff:** per-mechanism gain/loss and selected target-qualified alternative.

### V3R-031 — Search head-ready attention

**Depends / lane / review:** 020, 027–028; 030 when its fusion candidate is compared; GPU, R3. **Edit surface:** CUDA physical schedule/events, QKV body publication metadata and ablation.

**Implement:** publish a Q/K/V head group only after its complete K reduction, required RoPE/cache write and memory visibility. Map each attention task to the exact groups/cache slots it needs. Compare with the same body mix behind a full-QKV barrier; account for flags, fences, changed projection tiles and consumer imbalance. Event storage has explicit initialization and epoch separation across calls.

**Pass:** numerical/state correctness and device progress over repeated steps; trace shows the claimed early start and complete-call samples show its net effect. **Reject:** publication from a partial reduction, stale epoch, unrelated-head wait or nonresident producer deadlock. **Handoff:** ablation and event/progress proof. This card may legitimately end `rejected_candidate`.

### V3R-032 — Search streamed MLP and low-batch GEMM orderings

**Depends / lane / review:** 022–028; 030 when its fusion candidate is compared; GPU, R3. **Edit surface:** MLP algorithm rules, body continuation interface, reduction ownership and ablation.

**Implement:** compare complete gate/up hidden vector plus strongest full-K down body, paired hidden chunks with output-owner `reduce_begin/update/finalize`, and independent split partials with combine. Evaluate K-parallel SIMT, output-channel-major MMA and padding/pack choices for each projection. Retain exact activation/cast order; account for long-lived accumulators, scratch, extra weight reads, event traffic and mixed-entry occupancy.

**Pass:** every down output incorporates all required hidden chunks once, then final cast/residual; standalone, lean-entry and complete-call records explain the selected plan for low/higher batch. **Reject:** cuBLASDx's internal K pipeline is not an externally continued reduction; underfilled MMA or repeated weight loads may make streaming lose. **Handoff:** per-tactic Pareto and full versus streamed score.

### V3R-033 — Search cross-task or cross-layer weight staging

**Depends / lane / review:** 023–028 and one preload-capable body from 022; 029–032 are optional candidate inputs; GPU, R3. **Edit surface:** target movement planner, async token/storage protocol and prefetch ablation.

**Implement:** from an address-known/free-slot relation, compare zero, one and two staging slots only where selected bodies expose a legal preload stage. Track source access retirement, destination visibility, final consumer retirement and slot reuse separately. Price descriptor setup, shared/TMEM allocation, occupancy, issue bandwidth and HBM/L2 contention. Do not assume a whole next matrix fits per CTA or that TPU VMEM behavior transfers to CUDA.

**Pass:** actual trace shows early movement and valid publication; cold and warm conditions are recorded; complete-call improvement survives a no-lookahead control. **Reject:** early loads that steal bandwidth or reduce residency without a net gain. **Handoff:** timeline, lifetime/progress proof and retained/rejected depth.

### V3R-034 — Add a bounded dynamic schedule only for measured irregularity

**Depends / lane / review:** 020, 025, 028 and a declared bounded irregular workload; 029–033 optional; GPU, R3. **Edit surface:** CUDA dispatcher/event storage and irregular-work benchmark.

**Implement:** when a declared workload has data-dependent but bounded expert/task count or measured static stragglers, add a ready queue or hybrid policy. Register producer counts before consumer readiness; prove capacity, no duplicate task, no stranded ready task, epoch safety and uniform required collectives. Compare to static/barrier on the same body mix. Do not add queue overhead to the dense default merely because another project used one.

**Pass:** routed tiny fixture and device stress interleavings terminate with correct output/effects; complete-call samples show gain or loss and dispatch overhead. **Reject:** a zero counter observed before registration, queue overflow or consumer CTA that blocks all producers. **Handoff:** dynamic-policy guard, proof and ablation.

### V3R-035 — Compile a held-out structural Transformers family

**Depends / lane / review:** 028; 029–030 optional; CPU/GPU, R2/R3. **Edit surface:** only generic semantic rules/providers as necessary, capture helper and scorecard; no checkpoint-name body registry.

**Implement:** choose and record a second decoder structure **before** extending the compiler for it. Inventory its differing norm/attention/gating/cast/layout or state properties. Compile its full FX through the same indexed core; add a reusable semantic primitive or guarded algorithm only when the graph demands it. If an opaque custom op is unsupported, report it and retain an external path separately.

**Pass:** origin-level coverage, output/state correctness and matched per-cell timings for the held-out family; explain any new generic rule and run it on the original family where applicable. **Reject:** hard-coded checkpoint/name switch to a whole-model kernel is not a breadth result. **Handoff:** exact coverage delta, diagnostics and strict win/loss cells.

### V3R-036 — Rebuild and tune on a second CUDA target

**Depends / lane / review:** 021–024, 028; execute when access exists, CT/GPU, R3. **Edit surface:** target provider variants, cache key and portability scorecard.

**Implement:** use the same indexed/logical input on a second SM generation or distinct toolkit feature set. Recompute legal instruction/descriptor choices, actual mixed-entry resources and cooperative worker grid; retune body and whole-entry candidates. A portable SIMT path is the correctness backstop. Do not reuse an `_a` cubin on a different compute capability or a CUDA13/cuBLASDx provider in the pinned CUDA12 environment.

**Pass:** named target compiles/admitted entry, correct outputs/state and matched full-call score; unavailable hardware is stated as unavailable, not failed or passed. **Reject:** source portability is not a cross-SM performance claim. **Handoff:** target delta and newly selected tactic mix.

### V3R-037 — Publish the complete predeclared scorecard

**Depends / lane / review:** 003, 028, 035; include 036 when available, GPU/R2. **Edit surface:** `benchmarks/v3/` report/validation code and published artifact index.

**Implement:** list every predeclared checkpoint/revision × batch × short/long context × supported dtype × target cell. For each, show capture/strict status, strongest equivalent baseline, strict latency distribution, numerical/state result, grid count, resources, setup and optional ExternalPlan result. Separate tuning samples from fresh validation samples. State the exact aggregation rule for “consistently beats” before validation; losses, unsupported cases and missing cells stay visible.

**Pass:** raw samples and unfiltered trace are reproducible from artifact hashes; a claim is generated only when all `must_win` cells meet the prespecified margin/confidence rule and no strict correctness failure is hidden. **Reject:** best-of-many point estimate, a favorable uncached case or a fast external fallback cannot become the headline strict result. **Handoff:** G4 scorecard and decision on the next bottleneck.

### V3R-038 — Stabilize the public V3 compiler and fallback surface

**Depends / lane / review:** 028, 035, 037; 036 when hardware exists; CPU/GPU, R2. **Edit surface:** V3 API/session package, optional HF capture helper, docs and public behavior tests; preserve legacy entry until a separately reviewed migration.

**Implement:** expose the [IR's public compiler signature](MEGABAKE_V3_IR_AND_REUSE_PLAN.md#14-first-strict-cached-step-abi-and-public-compiler-surface) with explicit `WorkloadSpec`, `NumericalPolicy` and `StepABI`, strict-only and external-allowed modes, artifact provenance and transparent fallback. The returned `ExecutionRecord` reports whether `StrictGridPlan` or `ExternalPlan` ran and why; guard mismatch either recompiles a legal specialization or returns a diagnostic/fallback according to requested mode. Cache compiled artifacts by semantic and target keys without storing tensor addresses as identity.

**Pass:** a user can inspect support, exact path, source/plan/resource hashes and measured cell; repeated calls preserve output ownership/state. **Reject:** fallback success cannot masquerade as a one-grid win; a graph-break fragment cannot be labelled full-step. **Handoff:** public API examples and final coverage/known-limit table.

## 9. Worked vertical trace that every agent can follow

Take `LINEAR_TINY` with `X[1,33]`, `W[17,33]`, optional bias and an explicit output cast, followed by a SiLU/map and a state write in the unfamiliar composition. Preserve the original FX callable and lifted bindings. Facts say X and W strides/aliases, numerical policy and which value is state. Indexed contraction is `Y[0,n] = cast_out(alpha * sum_{k=0..32} X[0,k]*W[n,k] + beta*bias[n])`; if the FX form uses `addmm`, the exact alpha/beta and bias semantics come from that node, not this example's defaults. `Map` applies SiLU with its declared cast placement. `StateWrite` targets one bounded slot.

Choose a conservative algorithm: K-parallel SIMT, no packing, full Y materialization, then map and state write. With output tile width eight, logical contraction domains cover `n=[0..7]`, `[8..15]`, `[16]`, each with K=33 and tail predicates. Map task `i` reads only the corresponding completed Y region; the state write reads its exact value and is an observable effect. A static CUDA owner grid assigns logical tiles to resident workers, joins before a global state read if required, and publishes output after the final writer. This gives a complete semantic/physical path even if it loses to `torch.compile`.

Now introduce alternatives: N tiles of four or sixteen, `Yᵀ=WXᵀ` with padded batch, fused map epilogue and a different owner assignment. Each change can alter tile dependencies, resource envelope or cast order. Regenerate logical maps and verify the original FX origin coverage. Compare bodies standalone, in the lean entry and in the complete mixed entry. If the generated tensor-core body is individually faster but makes the full entry spill, retain the SIMT plan. This example is small enough to inspect every read/write; its role is correctness and compiler generality, not a speedup claim.

The full decoder extends the same sequence: captured old KV and position → exact current K/V write → grouped attention over valid slots → output projection and MLP → repeated layers → logits/new cache. A head-ready or lookahead schedule is introduced only after the barrier/full-materialization path is correct and measured. [FX walkthrough](MEGABAKE_V3_FX_TO_MEGAKERNEL_PROPOSAL_2026-09-27.md) shows the whole-model choices in more detail.

## 10. Exit rules and failure response

| Observation | Required next action | Forbidden interpretation |
|---|---|---|
| Capture differs from original or omits state | Repair V3R-001/002 and replay reference | A faster graph is equivalent because logits look close |
| Known primitive left in `ReferenceRegion` | Implement indexed semantics/body in 012–015 | Named matcher or whole-model Python fallback counts as strict coverage |
| Indexed/logical verifier fails | Reject candidate and keep counterexample | Schedule can repair missing semantic effect |
| GPU compile succeeds, device protocol fails | Keep R3 gate failed; fix body/event/participation | CPU simulator proves CUDA memory ordering |
| Standalone body far behind vendor | Change orientation, tile/reduction, mainloop/provider | More launch savings are assumed to compensate |
| Standalone good, mixed entry poor | Inspect common CTA shape, register/shared/spills/code and residency | Standalone rank transfers to whole entry |
| Entry good, complete call poor | Inspect binding, output ownership, cache copies, handoffs and strongest baseline | Kernel duration equals user-visible latency |
| Prefetch/head-ready/streaming loses | Retain zero-depth/barrier/full-body control and record loss | Optional mechanism is mandatory by architecture |
| Small family wins, held-out family loses | Report both, extend generic semantics/provider if justified | Model-name branch establishes generality |
| Another SM unavailable | Mark target evidence unavailable and continue independent tasks | Source compilation proves cross-SM speed |

A task can be `patch_ready` and GPU-`not_run`; downstream GPU admission remains unsatisfied. A rejected tactic experiment can be reviewed and closed, while the gate requiring a working tactic remains open. The first strict win is one declared correct cached-step cell. The broader north star requires the predeclared matrix and fresh matched trials; it is not inferred from one lucky model/SM.

## 11. Traceability and scope control

| Owning document | Primary V3R hooks |
|---|---|
| [Architecture](MEGABAKE_V3_ARCHITECTURE.md) | 001/002 complete-step contract; 012–018 generic semantics; 022–030 strict body/plan; 035–038 claims |
| [IR](MEGABAKE_V3_IR_AND_REUSE_PLAN.md) | 010–020 origin/effect/indexed/algorithm/logical invariants; 022/023 target boundary |
| [Kernel reuse](MEGABAKE_V3_KERNEL_REUSE.md) | 005–009 body quality, 013/014 generic bodies, 022/024 composition |
| [Pipeline](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) | 019/020 readiness; 025/027 safe baseline; 031–034 optional mechanisms |
| [Hardware](MEGABAKE_V3_HARDWARE_MODEL.md) | 005/008 toolchains; 021–024 profile/admission; 036 retargeting |
| [Performance](MEGABAKE_V3_PERFORMANCE_MODEL.md) | 001/003/004 baseline and cells; 009 budget; 028/030–037 complete-call evidence |
| [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) | 004–009 historical bottleneck diagnosis; 028 correct cached workload |
| [Research/FX walkthrough](MEGABAKE_V3_RESEARCH_AND_DECISIONS.md), [worked graph](MEGABAKE_V3_FX_TO_MEGAKERNEL_PROPOSAL_2026-09-27.md) | Algorithm/schedule counterfactuals and source-derived caveats for 017/022/029–036 |

No V3R card authorizes importing a private cuBLAS binary, promising cuBLASLt parity from cuBLASDx, silently upgrading the core toolchain, building a TPU backend, adding training/backward, or claiming that a finite set of models proves a universal speedup. Future accelerator support uses the common indexed/dependence semantics and a separate physical backend only after CUDA's initial strict path has evidence. Preserve the legacy backend and historical measurements as controls until an explicit migration decision.

## 12. Handoff example and command discipline

The checked-in `TaskHandoff` schema is version 1. A CPU-only card can report completed source and CPU validation while leaving GPU fields unrun. The following is a **format example**, not evidence that V3R-012 has been implemented:

```json
{
  "schema_version": 1,
  "task_id": "V3R-012",
  "task_revision": "example-source-hash",
  "dependency_revisions": {"V3R-011": "example-fact-hash"},
  "changed_files": ["src/megabake/v3/semantics/indexed.py"],
  "commands": [
    {
      "schema_version": 1,
      "command": "python -m pytest tests/test_v3/cpu/test_indexed.py",
      "return_code": 0,
      "stdout": "example: see retained command log",
      "stderr": "",
      "duration_seconds": 1.0
    }
  ],
  "evidence": {
    "schema_version": 1,
    "implementation": "patch_ready",
    "cpu_validation": "pass",
    "cuda_compilation": "not_run",
    "gpu_correctness": "not_run",
    "gpu_performance": "not_measured",
    "disposition": "continue"
  },
  "remaining_risks": ["CUDA body and device semantics are not validated by this card"],
  "next_eligible_tasks": ["V3R-013", "V3R-014"],
  "diagnostics": [],
  "artifact_hashes": {"indexed_fixture": "example-content-hash"},
  "claim": "not_measured",
  "benchmark": null,
  "notes": "Illustrative handoff only"
}
```

Use exact test commands that exist **after** implementing a card. The current CPU lane has `tests/test_v3/cpu/`; proposed toolchain/GPU lanes use the existing `v3_toolchain` and `v3_gpu` pytest markers in `tests/test_v3/conftest.py`. A CPU card normally runs its focused case and relevant existing CPU regression suite. A toolchain card records selected `V3_NVCC`/headers/SM and the emitted nvcc command. A GPU card records device/profile, exact binary, profiler trace kept outside timing runs, correctness, raw unprofiled samples and full-call comparison. Do not turn an unavailable lane's skipped result into `pass`; store the reason in `remaining_risks` or diagnostics. No documentation task by itself runs any of these checks.
