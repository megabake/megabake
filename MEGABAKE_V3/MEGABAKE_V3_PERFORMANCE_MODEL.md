# MegaBake V3: body budgets, whole-entry cost and measurement

Status: proposed performance protocol, revised 2026-09-27. Equations and examples are conditional diagnostics, not V3 forecasts. No new V3 GPU latency is established here. The [architecture](MEGABAKE_V3_ARCHITECTURE.md) defines the strict result; the [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) labels historical V2 observations.

## 1. Claim hierarchy

Report each workload cell as: full-step captured or failed; strict one-grid compiled/correct or unsupported; and strict measured win/loss/inconclusive. A fallback may be fast but cannot count as a strict one-grid win. A universal win for every valid FX graph is impossible to promise: a graph already lowered to one excellent vendor kernel may have no boundary cost to recover, and a forced resident grid may add cost.

The first performance matrix is fixed before final tuning: checkpoint/revision and structural family, batch, valid context and cache layout, dtype/numerical policy, selected GPU/partition/SM, CUDA/compiler/library versions, input/output ownership and timed unit. Include at least two structurally different Hugging Face decoder families, short/long context and batch-one/small-batch cells. One GPU can support an initial result; cross-SM claims require measured cells on each target generation. Report all predeclared cells, including strict losses and unsupported operations.

The primary success condition is a correct complete-step one-grid artifact faster than the best validated equivalent `torch.compile` execution on the same target. Non-launch gains from fusion, handoff, prefetch or early readiness are important mechanism evidence and optimization opportunities, but a winning entry is not required to use all of them. Compilation cost, code size, setup/packing memory and fallback share are reported separately from steady-state latency.

## 2. Whole-entry objective

The target planner minimizes complete invocation latency under exact semantics and resource legality:

```text
T_strict = T_bind + T_entry + T_return
T_entry = T_init + makespan(tasks, dependencies, target resources) + T_drain
```

Put each cost in one place. Overlapped initialization or drain belongs in the makespan; host binding/output costs use the actual session contract. Tasks include global/on-chip movement, compute, publication, source retirement, reduction finalization and joins. Resources include resident workers, body participant roles, live scratch/accumulators, descriptor pressure, instruction footprint and contended HBM/L2/compute engines.

Two simultaneously active HBM-bound tasks do not each retain their isolated full bandwidth. Body times measured outside the persistent entry are priors; the mixed entry can alter registers, spills, shared-memory occupancy and tail waves. Use a resource-constrained critical path and measured interference for ranking, then the actual compiled entry and full invocation for acceptance.

## 3. Break-even accounting for the cuBLAS gap

For a matched baseline interval, an explanatory accounting is:

```text
T_baseline = C + O
T_strict   = C * (1 + delta) - R + H
```

`C` is retained baseline body work; `O` is boundary/dispatch work removed; `delta` is the relative retained-body change; `R` is additional supported fusion/overlap/work-removal saving; `H` is added persistent coordination, setup and residual overhead. Avoid double-counting one saving in `delta` and `R`. Define `f=O/T_baseline`, `r=R/T_baseline`, `h=H/T_baseline`:

```text
speedup = 1 / ((1-f)*(1+delta) - r + h)
strict win requires delta < (f + r - h)/(1-f)
```

For example, if only 10% of baseline time is removable, no additional saving is supported, and persistence adds 2%, retained body work can slow by less than 8.89% before the candidate loses. A fourfold body slowdown on dominant linears is unlikely to fit. This is a budget diagnostic, not a proof from summing profiler kernel durations. A fused QKV or epilogue candidate may remove work and justify an isolated-body deficit; evaluate its actual body-plus-dataflow and final entry rather than a rigid per-operator veto.

Historical V2 examples were 15.49 us against a 2.63 us vendor path for `(B,N,K)=(1,576,576)` and 95.78 us against 23.39 us for `(1,4096,4096)`. These are older, potentially L2-hot shape tests under a particular H200 MIG environment, not V3 or cached-decode numbers. [Evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md) They make competitive body generation the first research risk.

## 4. Body tactic and composition ledger

For each expensive indexed computation, record exact shapes/strides/transposes, dtype and accumulation, mask/cast/epilogue policy, stable weight packing, dynamic padding, call count, working-set/cache state, and the vendor/library path actually selected. Body candidates keep a Pareto record:

| Quantity | Why it matters |
|---|---|
| Standalone latency and achieved traffic | Basic math quality under matched inputs |
| Lean persistent-entry latency | Scheduler call, common block size, register/shared/code envelope |
| Output tile count and shape | Parallelism and final-wave waste on actual SM/partition |
| Packed/prepared layout cost | Session amortization, memory footprint and runtime conversion |
| Scratch/accumulator/descriptor lifecycle | Occupancy, legal composition and setup |
| Epilogue/stage capability | Possible fusion, early publication and preload |
| Numerical result | Reassociation, padding and split-K legality |

Do not compare an isolated hot-cache GEMV with a whole-model cold-weight baseline as if they measure the same regime. For target bottlenecks, inspect actual CUDA traces/counters in diagnostic runs. GPU global-address-space traffic is not automatically physical HBM traffic; activations may hit L2. No paper's peak bandwidth or another device's benchmark is a V3 forecast.

Body qualification has two filters. First, reject illegal/inaccurate tactics and record their reason. Second, use a **conservative plausibility bound** against the measured baseline and candidate saving opportunities to cap costly search. Preserve a fused or different-dataflow candidate if it can plausibly pay for a weaker standalone body. Final promotion requires measured whole-entry latency, not a predicted `delta` alone.

## 5. Baseline ladder and controls

Use the fastest validated equivalent baseline among legal `torch.compile` default, reduce-overhead, max-autotune and stable-address captured variants as applicable to the pinned version. Verify actual capture, graph breaks, vendor calls, recompiles and input/output/state behavior; a mode name is not evidence. Give baseline autotuning/warmup a fair budget. Eager execution is a numerical reference, not automatically the performance competitor. [PyTorch torch.compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile)

Additional controls isolate causes:

1. Vendor-preserving `ExternalPlan` or CUDA Graph under the same session contract.
2. Same owned bodies in multiple launches, where feasible, to isolate grid composition.
3. Same body mix in a simple barrier one-grid entry to isolate launch reduction.
4. Selected fused/pipelined variants with one mechanism changed at a time.
5. Strong attention and matmul external baselines at exact shapes and precision.

Programmatic dependent launch and other multi-kernel overlap can narrow the strict grid's advantage; they belong in suitable controls rather than being assumed impossible. [CUDA PDL documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) The full invocation, not kernel count, determines the winner.

## 6. Measurement contract

Record checkpoint/revision, same weights/inputs, model step, cache capacity and valid length, batch, dtype/numerical flags, target profile, CUDA/compiler/provider versions, output ownership, input origin and setup amortization. For fixed-state replay, restore or overwrite cache positions consistently under the same timing rule; separately validate advancing multi-step generation. Runtime-required copies, descriptor updates, counter resets and binding remain in the timed unit. Stable pointer addresses do not make token contents free.

Use three views:

- Unprofiled GPU interval with correctly ordered events on the relevant streams.
- CPU enqueue/framework duration without confusing submission with completion.
- Synchronized user-visible call or controlled repeated-run wall time with explicit synchronization.

Run traces, counters and instrumentation separately from final unprofiled timing. Warm compilation, capture and lazy allocation. Alternate or randomize paths in trial blocks to reduce order/clock bias. Retain raw samples, dispersion and a speedup confidence interval; reserve fresh validation trials after tuning. Define a practical margin before final runs, for example at least 5% lower median with a confidence interval above 1, while reporting smaller/inconclusive observations honestly. Do not claim p99 stability from a small correlated sample.

## 7. Artifact and result reporting

Each benchmark cell stores graph/plan/artifact hashes, exact target/profile, source and body-provider versions, unsupported/fallback reasons, numerical and state validation, actual compute-grid count, complete operation trace, body and full-entry resource reports, raw timing samples and selection/validation split. Include preparation/compile time, peak memory, code size and setup amortization. Ablations record causal mechanism, not merely a shorter kernel list.

Report status as `strict win`, `strict loss`, `inconclusive`, `incorrect`, `unsupported`, or `fallback-only`, with the matched baseline and complete-call timing. A strict loss on one SM or model is useful evidence for the next body or physical-plan candidate; it cannot be omitted from a “consistently beats” claim. The compiler may correctly compile a model yet fail performance admission.

## 8. Go/no-go interpretation

If no device-callable tactic for dominant matrix shapes approaches the vendor path within a plausible full-step savings budget, improve algorithmic orientation, K parallelism, mainloop/epilogue, layout or target provider before adding scheduler complexity. If standalone bodies are good but the entry loses, inspect common block shape, registers, spills, code size, live storage and cooperative residency. If body and resource quality are good but handoff dominates, search activation ownership, fusion and event granularity. If prefetch issues early loads but loses, inspect storage pressure and bandwidth contention and retain zero-lookahead.

This decision tree makes the north star falsifiable. It does not promise that strict one-grid execution wins on every Hugging Face family or CUDA target; it shows which layer of the compiler must improve when a declared cell loses.

## 9. Predeclared scorecard for the first claim

`WorkloadSpec` and its `BenchmarkCell`s are the comparison contract, not labels added after a favorable measurement. Before final tuning, choose two structurally different supported Hugging Face decoder checkpoints/revisions and record a matrix on one selected GPU: batch 1 and one small batch (initial candidate 4), one short and one long valid context (initial candidates 128 and 2048 if legal), and each declared FP16/BF16 numerical policy supported by that checkpoint/target. This gives eight cells **per declared dtype** across two families. Exact lengths, batch, cache capacity/layout and output tree go into the manifest; a model's unsupported shape is disclosed before tuning, not removed afterward. The existing `BenchmarkCell.baseline` string may name the prespecified selector `best_validated_equivalent_torch_compile`; resolve its exact mode from tuning data and freeze that choice before fresh validation.

The first G3 claim can be one correct measured cell. A claim that V3 **consistently beats** the strongest equivalent `torch.compile` on the declared initial regime requires every predeclared `must_win=true` primary cell to pass. Default the eight family×batch×context cells for a chosen supported dtype to `must_win=true`; if the project chooses a narrower primary regime, say so in the manifest before tuning and label the claim accordingly. An additional SM is a new target matrix; code compiling there does not extend the speed claim. Losses, `unsupported`, `incorrect` and `unavailable` remain visible and cannot be silently treated as wins.

## 10. Locked timing unit and validation split

For the first full-step campaign, set `timed_unit=cached_step` and use already GPU-resident token/weights/cache at the call boundary. The **primary latency** is one synchronized user-visible step from invocation start through required output/state readiness, including binding, per-call descriptor/counter updates, runtime copies and output ownership; use the same synchronization and session contract for V3 and baseline. Report an unprofiled GPU-event interval separately because it diagnoses device scheduling and may differ from wall time. If serving with queued asynchronous steps is later the target, predeclare that as a separate workload and measure its request-level latency; do not substitute it after observing results.

Each benchmark report states whether calls advance state or deliberately replay fixed state. For advancing calls, validate that state after step `t` is input to step `t+1`. For fixed replay, restore the same cache positions on both paths with the same timed or amortized reset rule. Stable addresses do not make token-content changes free. Warm compilation, autotuning, CUDA Graph capture, allocator and lazy library initialization before taking validation samples; record their setup time separately.

Select the best **equivalent and numerically valid** baseline mode using tuning trials. Freeze that mode and the strict candidate before fresh validation trials. A practical default validation rule is at least 30 randomized paired trial blocks per cell, median strict latency at least 5% lower than baseline, and a 95% block-bootstrap confidence interval for `baseline/strict` wholly above 1.0. Predeclare any different block count, statistic, margin or interval method before validation. Correlated calls within a block are not independent confidence samples; report raw block results and dispersion. A one-off best sample or profile-distorted trace is diagnostic only. Do not infer p99 from this p50-focused design.

## 11. Baseline completeness checklist

A baseline candidate is eligible only if it uses identical checkpoint weights, input tensors/positions, cache state transition, mask, dtype/accumulation and tolerance, output tree/ownership, and input origin. Record `torch.compile` mode, graph breaks, actual CUDA Graph capture/replay, vendor matmul/attention calls, recompiles, model preparation and unfiltered device operation trace. A mode name is not proof that it captured or used cuBLAS. If a faster path changes precision, state lifetime or output copying, keep it in a separate non-equivalent comparison.

The historical `benchmarks/bench_compare.py` and `benchmarks/test_harness.py` cannot directly furnish this scorecard: the HF wrapper calls `use_cache=False`, their filtered counts omit some real work, and they do not retain all latest raw real-model samples. [GPU audit](MEGABAKE_V3_GPU_REANALYSIS.md) V3R-003 creates a new evidence route while leaving historical results intact.

## 12. Diagnosis that determines the next implementation card

| Measured observation | First inspect | Next bounded work |
|---|---|---|
| Hot standalone SIMT/MMA body far slower than selected vendor path | K parallelism, output tile count, orientation, padding, global movement, epilogue | V3R-006/007 or alternative V3R-008 provider; do not start a queue project |
| Standalone competitive, lean entry poor | CTA shape, live registers/shared/TMEM, spills, descriptor role, code footprint | V3R-022–024 body/entry compatibility |
| Lean entry competitive, full step poor | Activation handoff, attention, cache effects, binding/copies, baseline graph capture | V3R-027–030 and targeted ablation |
| Head-ready or streaming starts work early but total time rises | Body degradation, event cost, accumulator lifetime, producer tile underfill | Reject/revise V3R-031/032 candidate |
| Prefetch reduces weight wait but total time rises | HBM/L2 contention, staging capacity, occupancy, fill/drain | Retain zero-lookahead and revise V3R-033 |
| One family/SM wins, held-out loses | Unsupported indexed primitive versus target provider/body ranking | V3R-035/036; report strict loss honestly |

Every decision is made from **whole-entry** and complete-call evidence with the same shape/cache condition. The break-even equation in §3 is a screening worksheet, not a substitute for this measurement. The selected `ExternalPlan` can still be a useful product path; its speed cannot satisfy a strict-grid `must_win` cell.
