# MegaBake V3: performance bounds, budgets and measurement

Status: analysis and proposed protocol, 2026-09-09. All illustrative values below are explicitly
conditional. No V3 GPU latency has been measured.

## 1. What can and cannot be proved

A universal strict speedup over torch.compile is impossible to promise. A valid input may already
compile to one excellent kernel with no removable launch or intermediate work. Empty/view-only
graphs can make a forced megakernel actively worse. Generic FX input support is not a universal
performance theorem.

There are three different claims:

1. **Correctness:** prove transformation, ownership and synchronization obligations, then validate
   implementation and numerical policy.
2. **Feasibility:** derive conditional lower bounds and break-even budgets from a stated model.
3. **Performance:** demonstrate speedup for specified workloads, target and invocation contract.

A lower bound can disprove an ambitious target if even the ideal cannot meet it. Being above a
lower bound does not prove that an implementation achieving it exists.

## 2. Model execution first, use break-even accounting second

The primary model is the resource-constrained makespan of ExecutionPlan's action graph:

```text
T_m = T_bind + T_entry + T_return
T_entry = T_initialize_and_join + makespan(actions, dependencies, resources) + T_final_drain
```

Put each cost in exactly one term. Initialization/drain that overlap scheduled work belong inside
the makespan instead of being charged again. Binding/output costs use the actual ownership contract.
The action graph includes load, compute/update, publication, retirement and retained joins.
Resources include worker/cohort capacity, compatible thread roles, live staging/accumulators,
and shared bandwidth/compute limits. Dependencies include address readiness, data readiness,
reduction order and resource release; different token types cannot be substituted.

There are two useful levels of estimates:

- With only opaque body timings, estimate whole-tile durations under the common entry and use
  measured interference for concurrent pairs. Do not invent internal load/compute stages.
- With staged-body measurements, model those actions and their actual overlap. Recalibrate the
  joint schedule when body shape, lookahead or cohort sizes change.

Effective bandwidth measured over a complete operation already includes its memory stalls/tails.
Do not then add a second complete set of pipeline bubbles to that same interval. Conversely,
dividing bytes by an optimistic bandwidth does not account for missing parallelism or dependencies.
The [pipeline design](MEGABAKE_V3_PIPELINING_AND_SCHEDULING.md) defines which early actions are legal.

### The break-even equation

For a matched baseline interval, define a conceptual non-overlapping accounting:

```text
T_b = C + O
T_m = C * (1 + delta) - R + H
```

`C` is baseline body work expressed as latency in this accounting. `O` is overhead eliminated by
the candidate. `delta` is the relative change in retained body latency before separately credited
work removal. `R` is latency saved by fusion/recomputation avoidance/overlap not already counted
elsewhere. `H` is added persistent coordination and invocation cost, including remaining overhead.

All terms refer to the same workload and measurement contract. For overlapping execution, use a
critical-path schedule model; do not sum concurrent events as if they were sequential. `C` and `O`
are not obtained exactly by subtracting a separate profiler's kernel-duration sum from a median.

Normalize by `T_b`: `f = O/T_b`, `r = R/T_b`, `h = H/T_b`. Then:

```text
speedup S = T_b/T_m = 1 / ((1-f)*(1+delta) - r + h)

win condition:       delta < (f + r - h)/(1-f)
target speedup X:    (1-f)*(1+delta) - r + h <= 1/X
```

Assume positive total latency and `f < 1`. Definitions must avoid double-crediting a fusion saving
as both lower `delta` and positive `r`. In practice, compare complete candidate measurements;
this equation explains budgets rather than replacing experiments.

### Illustrative sensitivity, not a forecast

Assume no extra fusion saving (`r=0`) and added persistent cost of 2% of baseline (`h=0.02`).

| Removable baseline fraction f | Same body, delta=0 | Body 10% slower | Body 25% slower | Largest delta allowing a win |
|---:|---:|---:|---:|---:|
| 5% | 1.031x | 0.939x | 0.828x | 3.16% |
| 10% | 1.087x | 0.990x | 0.873x | 8.89% |
| 20% | 1.220x | 1.111x | 0.980x | 22.50% |
| 40% | 1.613x | 1.471x | 1.299x | 63.33% |

This is why an ungraphed small-model comparison can tolerate poor math while a stronger baseline
or larger model cannot. It is an explanation consistent with the user's observation, not proof
that launch overhead is the only cause of the specific 2B result.

If only overhead is removed, with unchanged body and no new costs, the ideal speedup is
`1/(1-f)`. A 2x target then needs at least 50% removable baseline time. Real composites may remove
additional work; they need separately supported savings, not a larger guessed overhead fraction.

## 3. The actual historical target

From [the evidence audit](MEGABAKE_V3_GPU_REANALYSIS.md), the latest reported strong baseline is
1,781.7 us and MegaBake is 4,845.5 us for an uncached one-token SmolLM2 forward. Parity requires
approximately 2.72x improvement to the current invocation. This is not evidence that a new backend
can attain that factor; it is the size of the historical gap.

The cached-decode target must be measured anew. Its attention and state costs differ, and a newly
available GPU may not match the reported H200 partition. Never carry a 1,781.7 us target across
those changes as if it were a universal requirement.

## 4. Shape-level budgets and composition penalties

For an exact-shape body, define vendor-relative efficiency as:

```text
eta_i = vendor_latency_i / owned_latency_i
```

`eta=0.80` means 25% more latency, not 20%; `eta=0.95` means about 5.26% more latency. Do not
confuse throughput efficiency, percentage latency reduction and model speedup.

Estimate the weighted risk using invocation multiplicities, but include non-linear work and the
whole-entry penalty. A fast isolated GEMV can become slow after composition because of registers,
spills, shared storage, common block shape, instruction footprint or required synchronization.
Benchmark three versions where useful:

```text
standalone body -> same body in a persistent test entry -> full generated model
```

The difference diagnoses composition; it is not an automatically additive constant across models.
Do not mandate an arbitrary per-body efficiency percentage independent of the end-to-end savings
budget. A body slightly slower than the vendor can win with a useful epilogue; a nominally faster
microkernel can lose after packing or state-handling costs.

## 5. Bytes, FLOPs and conditional lower bounds

A roofline-style lower bound can take the form:

```text
T >= max(D_min / B_max, F_min / P_max, dependency_critical_path_lower_bound)
```

This requires valid lower bounds on necessary physical traffic/work and valid upper bounds on
available bandwidth/compute under the target contract. If the algorithm changes or data is cached,
the work/traffic assumptions may change. Compute and memory can overlap, so adding their times is
not a general lower bound. Sum only intervals known to be sequential.

For a cold stream of exactly two billion distinct 16-bit weights, four billion physical bytes
would take the following ideal transfer times, assuming the listed sustainable rates:

| Assumed rate | Conditional streaming time |
|---:|---:|
| 1 TB/s | 4.000 ms |
| 2.4 TB/s | 1.667 ms |
| 4.8 TB/s | 0.833 ms |

These are arithmetic scenarios, not predictions for Gemma or a MIG partition. Parameter count is
not automatically distinct physical bytes read: tied weights, embedding row selection, caches,
layouts and redundant reads all matter. The 2.4 TB/s historical estimate is qualified in the
[hardware model](MEGABAKE_V3_HARDWARE_MODEL.md#6-h200-mig-what-the-historical-profile-tells-us).

Track at least three separate metrics: semantic bytes per invocation, physical DRAM bytes from
counters, and useful outputs per complete latency. Similar numbers are not interchangeable proof.

### Context and state alter the regime

For an ordinary KV cache, a first storage estimate across L identical layers is:

```text
KV_bytes = 2 * B * L * H_kv * context_length * head_dim * bytes_per_element
```

Use sums for nonuniform layers. Ideal attention reads can reuse K/V across grouped query heads;
actual traffic can be larger due to tiling/redundancy or smaller at DRAM due to caching. A paged
cache adds indexing/layout effects. Long context can change the dominant cost from projection
weights to attention/state access.

A matrix-recurrent attention state instead has a term roughly
`B * H_value * key_dim * value_dim * state_element_bytes` per layer, plus convolution and other
state. That term need not grow with context, but its update is different mathematics. The two
regimes need different bodies and budgets.

Keeping a grid alive does not make multi-gigabyte model weights fit in on-chip memory. Causally
dependent next tokens also cannot all reuse one weight tile without respecting the intervening
layers/token dependency. Batching and quantization can change the problem, but are not free savings
for the requested batch-one reference-precision comparison.

## 6. Optimization ledger

| Candidate | Named saving | Cost/risk that can erase it | Required evidence |
|---|---|---|---|
| One persistent grid | Repeated launch/dispatch gaps | Body/resource penalties and grid barriers | Matched graph-capable baseline |
| Better GEMV mapping | More parallel reduction, useful lanes | Extra reductions and launch-envelope restrictions | Exact-shape and composed tests |
| Tensor-core tactic | Better math/data path for suitable shapes | Padded work, packing and staging | M=1 shape-specific full cost |
| Linear epilogue fusion | Intermediate write/read and separate phase | Register lifetime, reduced residency | Correct casts plus composed latency |
| Norm replication | Avoid materialization/phase boundary | Repeated reduction/vector work per tile | Replication multiplicity and full latency |
| Head-local pipeline | Earlier consumer readiness, reduced tail | Publication, scratch overlap, scheduling | Dependency proof and controlled A/B |
| Streamed gate/down reduction | Down K updates overlap later gate/up chunks | Long-lived accumulators or global partial/finalizer overhead | Ordered chunk coverage and full region latency |
| Cross-task weight prefetch | Hide exposed movement/fill stalls | Extra live storage, bandwidth contention | Matrix-data overlap, not merely descriptor prefetch |
| Consumer-aligned layout/tiles | Earlier complete regions, fewer conversions, useful coalescing | Worse producer mapping, extra weight packing | Physical traffic and complete producer/consumer pair |
| Tail-aware worker/cohort assignment | Ready useful work occupies otherwise idle capacity | Cohort reservation, smaller body tiles, imbalance | Matched tile/body controls and worker/event trace |
| Borrowed buffers | Avoid allocation/copies | Changed output lifetime contract | Equivalent session/ownership baseline |

Remove an optimization when its measured net saving is negative. That is a successful experiment,
not an architectural failure requiring another mandatory IR.

### Concrete arithmetic for non-launch opportunities

**Staging:** for n independent tiles, each with a load stage of duration a and compute stage b,
perfect two-stage overlap on nonconflicting resources gives:

```text
serial = n * (a + b)
pipelined_ideal = a + b + (n - 1) * max(a, b)
ideal_saved = (n - 1) * min(a, b)
```

For n=4, a=3 us and b=5 us, that is 32 -> 23 us, or 1.391x, **without changing launches**.
If the pipelined schedule adds 2 us total coordination, it becomes 25 us, or 1.280x. These are
synthetic action timings, not a GPU prediction. Staging capacity, address readiness and independent
resources are assumptions. If both stages contend for the same saturated bandwidth, recompute
the durations; this ideal formula is no longer an achieved schedule.

**Readiness/tails:** with two independent producer tiles and two consumer tiles, each taking 5 us,
one producer worker and one consumer worker can execute P0 at 0–5, P1 at 5–10, C0 at 5–10 and C1
at 10–15. An operator barrier with that same fixed assignment takes 20 us. However, a control
allowed to use both workers for both phases also takes 10 us under these assumptions and wins.
Therefore compare optimized assignments too: an appealing overlap diagram is not a speedup proof.

A positive tail example with the same two workers: producer tiles take 8, 2 and 2 us; their
respective consumers each take 2 us. An optimal operator-barrier assignment needs 8 us for
producers then 4 us for consumers, totaling 12 us. A legal ready-tile assignment runs the long
producer and its consumer on worker 0 in 10 us, while worker 1 runs the two short producer/consumer
pairs in 8 us. Makespan is 10 us, or 1.20x, with no launch change. This assumes these heterogeneous
durations, compatible bodies and no shared-resource slowdown; the 8+2 dependency path proves the
10-us lower bound for this toy case, not for a real model.

More generally, for N identical tiles, P workers and per-tile duration t, a coarse phase takes
`ceil(N/P)*t`. Its idle worker-time is `(ceil(N/P)*P - N)*t`. That quantity has units of
worker-time, not elapsed time saved. It can be exploited only by ready work with compatible
resources; a final tail with nothing else useful to execute cannot be eliminated by scheduling.

**Fusion bytes:** eliminating materialized gate and up vectors of length I avoids, at most,
two writes and two reads per element: `4 * I * element_bytes` of logical global traffic.
For I=8192 and FP16/BF16, this is 65,536 bytes. With hidden size H=2048, the three dense MLP
weight matrices contain `3*H*I*2 = 100,663,296` bytes. Thus this fusion does not remove most
weight traffic. Its gains can still include latency and cache/issue pressure; physical DRAM-byte
savings may be smaller than logical savings. Streamed gate/down execution additionally addresses
exposed stalls, but cannot claim that the h buffer vanishes when different CTAs exchange it.

**Reduction continuation:** owner-held accumulators avoid global partials, but consume live
register/storage capacity. If instead c full-output partial vectors of length H are written and
read by a finalizer, their traffic alone is approximately `2*c*H*partial_element_bytes`, excluding
the final output and metadata. Use the selected implementation, not a universal split-K surcharge.

### Ablations that establish the mechanism

| Control | Purpose |
|---|---|
| A: strongest equivalent torch.compile/captured baseline | Actual external performance target |
| B: selected owned bodies as separate kernels in a matched graph | Quality of reusable math without persistent composition |
| C: same supported bodies in barrier-control persistent plan | One-grid/common-resource effect relative to B |
| D: C plus selected local fusion/layout changes | Materialization/conversion benefit and body changes |
| E: D plus cross-task matrix-data lookahead | Data-movement overlap net of staging/resources |
| F: E plus tile readiness/streamed reductions/tail-aware assignment | Fine-grained overlap net of publication and ownership costs |

Use controls with matched numerical behavior, work, worker limits where appropriate and output
ownership. If a fused body or continuation cannot be run identically in B/C, document the changed
body rather than attributing its entire difference to scheduling. Keep optimized phase assignments,
not deliberately poor fixed-cohort controls.

Incremental effects depend on order. For the most important region, also measure the 2x2
lookahead-on/off and readiness-on/off matrix with the same fusion/body choices. A separate fusion
on/off comparison checks materialization. Do not add independently measured percentage gains or
multiply region speedups into a whole-model forecast. Report interactions and net final latency.

Evidence must include a correctness/state pass, a trace showing the claimed early work, complete
unprofiled latency, and actual resources. A larger prefetch counter or lower barrier count alone
does not establish a benefit. Report both `A/F` (external win) and `C/F` (combined non-launch
improvement), with the specific incremental controls that explain the latter. A launch-only win
remains a valid scoped result, but does not satisfy the revised mechanism objective.

## 7. Measurement protocol

### Contract first

Record identical checkpoint/revision and shared weights/inputs; batch, shapes, dtype, numerical
flags, context/mask and state layout; output ownership; whether inputs originate on CPU or GPU;
and whether the timed unit is one forward, one cached step or a full generation loop. Do not
compare a borrowed-output engine to an allocating baseline without identifying that distinction.

Use fixed-state one-step replay for controlled comparisons: restore or overwrite valid cache
positions consistently outside the timed region when restoration is benchmark setup, with the
same rule for both paths. Also test genuine advancing multi-step generation. Replaying a stateful
kernel repeatedly must not accidentally advance position, overflow a bucket or accumulate stale
state. Runtime-required resets/copies remain inside the measured contract.

### Baseline ladder

Test eager for reference, then legal torch.compile default, reduce-overhead and max-autotune
configurations, including the matched export route and a stable-address graph control where useful.
Use the fastest validated equivalent baseline, not whichever makes MegaBake look best.
PyTorch's [mode documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile)
describes graph-related modes, but capture suitability is workload-dependent. Inspect actual
capture, graph breaks, recompilations and operation timelines; do not infer them from the mode name.

Keep the PyTorch version and all relevant options fixed for a comparison. The historical pinned
2.6.0 environment and current documentation are not assumed to have identical compiler behavior.
Give the baseline its legal autotuning/warmup budget and report setup time separately.

### Three timing views

1. Unprofiled GPU interval with correctly ordered events on the relevant stream(s).
2. CPU enqueue/framework duration, measured on the host without confusing it with GPU completion.
3. Synchronized user-visible latency or controlled repeated-run wall time, with the synchronization
   contract explicitly stated.

Capture traces and hardware counters in separate diagnostic runs. Count all GPU operations,
including copies and resets. Report number of compute grids, graph nodes, host submissions and
bytes moved where known. Kernel count alone is not a performance metric.

### Sampling and selection

Warm all paths past compilation, graph capture and lazy allocation. Alternate or randomize backend
trial blocks to reduce clock/thermal/order bias. Use the real working set; label hot-cache isolated
shape tests distinctly from whole-model trials. Do not force cache flushes into a production claim
unless they match the intended contract.

Retain raw samples. Report median and dispersion; estimate a confidence interval on the speedup
using paired trial blocks or independent runs as appropriate. Avoid treating strongly correlated
within-run iterations as independent evidence. Reserve fresh runs for final validation after
candidate selection, so the winning noisy trial is not also the reported result.

Choose enough repetitions for the requested precision; hundreds of steady-state samples can be a
starting point for medians, not a guarantee. Tail claims require substantially more samples and
careful independence analysis. Do not advertise a stable p99 from a tiny sample set.

Define a practical margin before final timing, for example at least 5% lower median latency with
a speedup confidence interval above 1.0. This is a proposed engineering acceptance policy, not a
law or a guaranteed result. Report smaller improvements honestly if observed, including uncertainty.

### Artifact per benchmark cell

```text
source and model revisions; complete environment and target profile
graph/plan/artifact hashes; numerical and state validation results
baseline modes and capture evidence; input/output/timing contract
raw timing samples; trial order; summary statistics and selection/validation split
entry registers, stack, shared memory, block/grid and occupancy checks
all-operation trace; profiling settings; optional physical traffic counters
tile-ready/copy/retirement trace in diagnostic builds; ablation controls and interactions
setup/compile time, peak memory, fallback status and unsupported reasons
```

## 8. The honest result format

Report each supported workload as `strict win`, `strict loss`, `incorrect`, `unsupported`, or
`not measured`, with the exact baseline and contract. A fallback may make the product usable but
does not change a strict loss into a single-grid win. The first research milestone needs a small
and a roughly 2B cached-decode model on one GPU; it does not need an invented universal X multiplier.

The revised objective additionally demonstrates a non-launch mechanism against a matched phase
control. It does not require enabling every optimization in every cell: report which are profitable,
which are rejected and why. Every first-class pipeline receives a correct experiment, not a promise
that all three will improve the final model simultaneously.

## 9. Conscious model-size calculations, not promises

Model size is not enough to predict latency. Use active weight bytes, shape quality, layers,
context/KV traffic, baseline capture, precision and target resources. Two checkpoints called 2B
can have different vocabulary heads, head counts and nonlinear work. The current evidence does
not identify enough of these quantities to supply calibrated expected ranges.

The following deliberately simplified calculation isolates the user's central question: **can
better effective execution matter on larger models after launch savings become small?** Yes,
conditionally; the arithmetic shows exactly what efficiency change would have to be achieved.

Assumptions: batch-one short-context, GPU-resident inputs/state, two-byte exercised weights,
one read of each exercised weight from DRAM, and a chosen bandwidth reference of 1 TB/s. Let W
be exercised weight GB (decimal), L a hypothetical layer count, and e an aggregate effective
streaming rate in TB/s. The baseline rate is assumed 0.65; residual boundary cost is assumed
1 us/layer for baseline and 0.5 us/layer for V3; both have 10 us fixed cost:

```text
T_baseline_us = 1000*W/0.65 + L*1.0 + 10
T_V3_us(e)   = 1000*W/e    + L*0.5 + 10
S(e)         = T_baseline_us / T_V3_us(e)
```

e already incorporates weight-path stalls, tails and body efficiency. An improved e represents
the **net** benefit of better bodies/movement/scheduling, including their penalties; it is not
an extra multiplicative benefit applied after separately subtracting those same stalls. Other
nonlinear/state costs are omitted here; add their measured critical-path contribution for a real
forecast. The boundary costs are assumptions, not a claimed CUDA Graph or per-kernel latency.

| Exercised 16-bit weights, approximate count | W, GB | Assumed L | Baseline, ms | Same rate 0.65: boundary saving only | V3 rate 0.70 | V3 rate 0.80 | V3 regresses to 0.55 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 125M | 0.25 | 30 | 0.425 | 1.037x | 1.111x | 1.258x | 0.885x |
| 500M | 1 | 24 | 1.572 | 1.008x | 1.084x | 1.236x | 0.855x |
| 2B | 4 | 24 | 6.188 | 1.002x | 1.079x | 1.232x | 0.848x |
| 8B | 16 | 32 | 24.657 | 1.001x | 1.078x | 1.231x | 0.847x |
| 32B | 64 | 64 | 98.536 | 1.000x | 1.077x | 1.231x | 0.846x |

These are synthetic scale points, not actual SmolLM/Gemma/Qwen configurations, expected speedups,
confidence intervals or guaranteed bounds. The 64-GB row additionally requires adequate device
capacity for weights, state and workspace. No CPU offload or quantization is silently introduced.

Interpretation: in this weight-dominated scenario, a net rate change from 0.65 to 0.80 supports
about 1.23x even when boundary savings vanish; degrading to 0.55 loses despite one grid. For a
baseline already at 0.90 of an applicable bandwidth ceiling, a weight-only same-byte ideal is
just `1/0.90 = 1.111x` before other costs. A 2x weight-dominated claim would need enough initial
inefficiency, fewer required physical bytes, or a changed workload—not merely deeper prefetch.

### How to turn the calculation into a forecast when a GPU returns

Measure the baseline's exact shape/working-set costs and traffic, then measure composed stage
durations and joint interference for the candidate body/pipeline mixture. Build conservative,
central and optimistic schedule scenarios from those measurements and their uncertainty; propagate
them through the complete workload, including attention, state, invocation and common-entry costs.
Validate predicted versus observed block latency before extrapolating to repeated layers.

If a fraction p of baseline latency is improved by factor s while the rest stays unchanged,
`S_total = 1 / ((1-p) + p/s)` under that non-overlapping accounting. For example, improving a
30% region by 1.30x yields about 1.074x overall, not 1.30x. Dependent/overlapped regions need the
action makespan instead. Without measurements establishing p and s, these remain scenarios.

The disciplined commitment is a calculation with explicit inputs and rejection budgets, not a
promise that a chosen architecture must produce an X multiplier in a parameter-size range.
