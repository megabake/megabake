# MegaBake V3: audit of the GPU evidence

Audit date: 2026-09-09. No GPU available; no benchmarks rerun. Repository inspected at
`3695f06de14322fd8ac3e111c693612d53b56319`.

## 1. The conclusion supported today

The current implementation has strong source-level evidence of poor skinny-linear mapping and a
large whole-entry resource footprint. The latest V2 prose reports substantial losses against a
low-overhead compiled baseline. Neither more IR layers nor the GraCE argument-patching mechanism
addresses those problems directly.

The immediate experiment is competitive embedded math plus a correctly measured, generated
cached-decode step. Its eventual speedup is unknown.

## 2. Separate the evidence generations

| Evidence | Workload/environment | What is actually present |
|---|---|---|
| User recollection | About 1.3x small-model win; 0.7–0.8x near 2B | Useful context, not a reproducible benchmark artifact |
| [README](README.md) real-model table | H200 MIG 2g.35gb context; SmolLM2 and Gemma | Historical displayed latency table |
| [results.json](benchmarks/results.json), [post-Inductor results](benchmarks/results_post_inductor.json) | H200 MIG 2g.35gb, 32 SMs, small synthetic workloads | Tracked JSON, not latest real-model traces |
| [V2 GPU reanalysis](MEGABAKE_V2_GPU_REANALYSIS.md) | H200 MIG 3g.71gb, 60 SMs, SmolLM2-135M | Detailed historical prose reporting profiler and timing results |
| Current CUDA/Python source | Checked-in implementation | Directly inspectable mapping, launch and harness behavior |

The tracked tree contains no raw latest SmolLM/Gemma timing samples, Nsight reports, cubin or
saved full real-model trace substantiating the newest prose measurements. This does not make the
reports false; it prevents independently reproducing or reanalyzing them now. Do not merge numbers
from different GPU partitions or benchmark generations into one performance curve.

The older README lists SmolLM2 at 7,238 us versus 6,929 us, or 0.96x; Gemma at 28,190 us versus
7,580 us, or 0.27x. Those differ from the user's recollection and newer V2 measurements. The
available artifacts do not establish why, so V3 preserves the discrepancy instead of resolving
it by speculation.

## 3. What the latest V2 report says

Reported environment: H200 MIG `3g.71gb`, 60 visible SMs, about 29.4 MiB visible L2, driver
595.58.03, PyTorch 2.6.0+cu124, nvcc 12.8.93, Nsight Compute 2025.1.1, Transformers 5.16.1.
Model: `HuggingFaceTB/SmolLM2-135M`, FP16-like policy, batch one, sequence one. These version
strings are transcribed provenance, not a newly checked installed environment.

| Path | Reported invocation latency, us | Counted kernels | Separately reported summed kernel duration, us |
|---|---:|---:|---:|
| Eager | 11,350.7 | 674 | 1,491.0 |
| Ordinary torch.compile | 5,691.4 | 424 | 985.1 |
| MegaBake | 4,845.5 | 1 compute grid | 4,748.7 |
| Export + reduce-overhead | 1,781.7 median | Not specified here | 1,026.2 |

Source: [V2 GPU reanalysis §§1–3](MEGABAKE_V2_GPU_REANALYSIS.md). Its best low-overhead sample
was 1,737.9 us; compare medians to medians, not that best sample to another path's median.

Arithmetic on the reported invocation values:

```text
ordinary compiled / MegaBake       = 5691.4 / 4845.5 = 1.175x
low-overhead compiled / MegaBake   = 1781.7 / 4845.5 = 0.368x
MegaBake improvement for parity    = 4845.5 / 1781.7 = 2.720x
5% lower latency than baseline    = 0.95 * 1781.7    = 1692.615 us
improvement needed for that target = 4845.5 / 1692.615 = 2.863x
```

These are historical comparisons, not V3 predictions. The approximately 4.8x ratio between the
reported megakernel body duration and summed compiled-node durations is diagnostic, not an exact
causal decomposition or an automatically achievable improvement factor.

## 4. “Decode” currently means an uncached forward

In [_CausalLMWrapper.forward](src/megabake/integrations/transformers.py), the base model is called
with `use_cache=False`, then `lm_head` produces logits. The [test harness](benchmarks/test_harness.py)
defaults decode mode to sequence length one. There is no old/new KV-state argument in that wrapper.

This exercises the M=1 projection regime, which is useful. It does not exercise attention over a
growing KV cache, cache writes, context-position handling, or correctness across cached steps.
Do not extrapolate its timings to 2K/8K-context generation or to the recurrent attention in the
supplied Qwen configuration.

The new benchmark contract must specify checkpoint/revision, token batch, current context,
capacity/bucket, mask, KV layout, state dtype and returned logits. A real prefill can initialize
state outside a one-step timing interval; report that setup and test it for correctness separately.

## 5. Timing and counting limitations in the checked-in harnesses

Both [bench_compare.py](benchmarks/bench_compare.py) and [test_harness.py](benchmarks/test_harness.py)
use CUDA events around the callable and synchronize for each measured iteration. That is a GPU
timeline interval, not CPU wall-clock duration. It can include GPU idle time while Python enqueues
subsequent work after the start event; it is not merely the sum of kernel execution times either.

The printed `Host(us)` subtracts separately profiled, filtered kernel durations from the event
interval, clamping negative results. It is not a direct CPU-time measurement. Different trials,
profiling perturbation, omitted operations and possible overlap invalidate an exact interpretation
as host overhead. Measure CPU enqueue and synchronized user-visible latency independently.

The filters remove memcpy/memset, fill/zero functors and every `vectorized_elementwise_kernel`.
The latter can perform real tensor mathematics. Operation counts and body sums must not silently
discard these categories. Report all operations, and classify them by observed role afterward.

`bench_compare.run_single` constructs a model and inputs separately for each backend. Without
shared state and matched random inputs, its side-by-side trials need not use identical values.
Its correctness metric records maximum error but does not itself enforce a pass threshold.

The checked-in exported compile path calls default Inductor; it does not contain the reported
reduce-overhead experiment as a fully reproducible named benchmark path. Add that path in future
implementation work rather than assuming the current harness reproduces all V2 tables.

These issues qualify the evidence; they do not require throwing away the useful mapping/resource
findings. The [performance protocol](MEGABAKE_V3_PERFORMANCE_MODEL.md#7-measurement-protocol) specifies
the replacement measurement contract. No harness is modified in this documentation task.

## 6. Concrete implementation bottlenecks

| Source | Observed behavior | Consequence to test |
|---|---|---|
| [matmul.cu](src/cuda/tasks/matmul.cu) | Skinny path assigns an output column to a thread and serially loops over K | Too little K parallelism; poor useful-thread fraction for small N |
| [tiling.py](src/megabake/schedule_compiler/tiling.py) | Skinny work count tied to SM count | Logical work and residency are conflated |
| [launcher.py](src/megabake/runtime/launcher.py) | Fixed 256-thread blocks and architecture-threshold shared-memory reservation | Whole-grid resource cost not selected for the actual graph |
| [cuda_compiler.py](src/megabake/runtime/cuda_compiler.py) | Universal task-source composition and fast-math compilation | Unneeded reachable code/resources; numerical policy needs explicit control |
| [loader.py](src/megabake/runtime/loader.py) | Per-run dependency/tile counter clones and output cloning | Additional operations; mutable state and output ownership need explicit design |

For 60 workers, splitting N=576 gives roughly 10 output columns per worker; N=1536 gives roughly
26. This leaves few valid output-computing threads in a 256-thread block, though other threads
may participate in staging. Each valid output thread still traverses K serially.

The skinny implementation also contains a runtime-M `float acc[64]` array and very large staging
storage. Generate exact-M bodies and inspect their compiled code; source constructs suggest a
risk, not the exact number of executed spills.

### Reported compiled entry

| Property | Latest V2 report |
|---|---:|
| Grid/block | 60 CTAs x 256 threads |
| Registers per thread | 184 |
| Dynamic shared memory per CTA | 225,280 bytes = 220 KiB |
| Stack frame per thread | 1,072 bytes |
| Theoretical/achieved occupancy | 12.5% / 12.5% |
| DRAM / compute throughput metrics | 2.33% / 6.42% |
| Profiled body interval | Approximately 4.77 ms |

These resource and SASS observations are reported in V2, not re-read from a retained binary here.
A nonzero stack and local-load/store instructions do not establish how often every path executes.
Low occupancy is a diagnosis input, not a universal performance verdict: some good kernels
intentionally use many registers. The decisive test is latency and throughput of the actual entry.

Inlining and resource allocation also mean the composed register count is not necessarily the
maximum of isolated function counts. Scratch overlay is valid only for non-overlapping lifetimes.

## 7. The shape evidence argues for a small portfolio

Historical isolated measurements from [V2 §5](MEGABAKE_V2_GPU_REANALYSIS.md), potentially L2-hot:

| M,N,K | Reported selected vendor family | Vendor, us | Current MegaBake, us |
|---|---|---:|---:|
| 1,576,576 | GEMV | 2.63 | 15.49 |
| 1,1536,576 | GEMV | 3.17 | 15.97 |
| 1,576,1536 | GEMV | 2.94 | 33.22 |
| 1,49152,576 | WMMA-based | 34.98 | 77.76 |
| 1,4096,4096 | Tensor-core GEMM | 23.39 | 95.78 |

The evidence supports testing both K-parallel GEMV and a tensor-core tactic at M=1. It does not
support mandating four implementation libraries for every shape or claiming one family wins
all skinny matrices. Tune logical tile count separately from resident worker count.

## 8. Smaller overheads are real but not the first explanation

The V2 unfiltered invocation reports one compute grid plus five copies totaling about 5.5 us
of copy duration, in a profiled run with a roughly 4,772 us megakernel. Copies and associated host
work still count; removing them alone cannot explain a multi-millisecond body gap.

The reported scheduler comparison is 4.756 ms with queues versus 4.573 ms with a static loop:
about 3.85% lower body latency in that experiment. This shows a control-overhead difference for
those implementations; it does not measure head-ready attention, streamed gate/down reductions
or matrix-data lookahead. It neither establishes that fine-grained scheduling is worth only 4%
nor justifies excluding pipelines from the primary architecture. Scheduling simplification alone
also does not close the approximately 2.72x end-to-end gap.

Large reported logits differences, including 0.195312 for MegaBake, remain unresolved correctness
evidence. The fact that another compiled path differed more does not validate either policy.
No speedup is accepted until the agreed numerical and state tests pass.

## 9. What changes in V3

Keep the resource/mapping diagnosis. Strengthen the baseline and evidence contracts. Remove
unconditional claims that semantic parameter bytes divided by product bandwidth are a latency
floor; physical traffic depends on accessed tensors and cache conditions. Do not infer exact CPU
overhead by subtraction or exact attainable fusion savings from profiler sums.

The useful next result is not another architecture document with a promised multiplier. It is a
reproducible table containing correctness, complete latency, actual compiled resources and stateful
workload definitions for one generated design. See [implementation experiments](MEGABAKE_V3_IMPLEMENTATION.md).

### Revised implications for the pipeline-first architecture

| Historical evidence | What V3 changes | What still requires measurement |
|---|---|---|
| Serial per-output K work and few useful lanes | Shape-specific K-parallel/tensor-core bodies; logical tiling independent of workers | Quality after consumer-aligned tiling and embedding |
| Universal large shared allocation and register footprint | Selected bodies plus proven overlapping lifetimes, actual-entry resource gate | Whether lookahead/continuations fit without losing more residency than they save |
| Queue/static comparison without mechanism isolation | Keep barrier control; generate staged/ready-tile schedules too | Non-launch gains under matched optimized assignments |
| Uncached sequence-one wrapper | Real stateful decode with head/cache readiness | Attention overlap and long-context costs |
| Incomplete real-model raw artifacts | Matched external baseline and owned-body/fusion/pipeline ablations | Reproducible whole-model win and uncertainty |

The mapping defect is still an immediate priority, but treating scheduling as a small final polish
was too strong an inference. The revised [architecture](MEGABAKE_V3_ARCHITECTURE.md) jointly selects
bodies, tiles, movement and schedule. Good microkernel results do not excuse missing pipelines;
pipeline diagrams do not excuse weak math. Neither component's success alone establishes the goal.

No historical number above has been reinterpreted as a measured V3 benefit. In particular, the
new model-size/efficiency scenarios in the performance document are calculations, not an extension
of the H200/MIG benchmark record.
