# MegaBake: feasibility before compiler architecture

Status: CuTe DSL 4.8.0 feasibility pilot, optimized BF16 projection probe, and
a full-step **dummy-model megakernel** complete, 2026-09-30; six-model graph
atlas and native-BF16 checkpoint block remain. The target is a correct, complete cached decode step on one
NVIDIA GPU. A qualifying strict result executes the step in one resident CUDA
grid. An ordinary `torch.compile` path remains available for graphs that
cannot yet qualify; its performance is reported separately.

## Architecture verdict

The proposed path is technically possible. A manually written CuTe DSL kernel
now executes an entire two-layer dummy decoder step in one cooperative grid on
the available H200 MIG: both cached-attention blocks, their MLPs, final norm,
and logits. Its device epoch advances the cache slot across two calls without
a reset launch between them. A CUDA profiler records one GPU kernel for a
captured replay. Warp-cooperative attention and prepacked warp-reduction
projections also ran inside that grid, including a two-layer dummy with
SmolLM2-sized hidden/MLP/vocabulary dimensions. This proves full-step
composition and that the first scalar losses were not a CuTe ceiling. It does
not establish native-checkpoint performance or six-checkpoint coverage.

Change the drawing in these specific ways:

| Proposed box | Working interpretation |
| --- | --- |
| TorchDynamo → TorchInductor → MegaBake hook | Capture raw Dynamo FX for graph-break diagnosis and normalized `torch.export`/ATen FX for the semantic input. Use pinned Inductor internals as an optional source of comparison data. There is no stable handoff containing all of Inductor's decompositions, alias/layout decisions, and symbolic facts before scheduling. |
| MegaBake IR | Start with a semantic graph that preserves effects and state. Add a separate physical task graph only after profiling identifies useful tiles, CTA ownership, and data dependencies. Avoid a large generic IR before the math is competitive. |
| Kernel library | Store reusable **device-callable** CuTe `@cute.jit` bodies or shared algorithm templates with explicit shape, precision, layout and state contracts. A CuTe `@cute.kernel` cannot call another `@cute.kernel` on device. |
| Planner and stage barriers | Schedule dependencies at the smallest useful tile or head group. A whole-grid barrier after every FX operator would serialize work and repeat the launch-boundary problem inside the grid. Use a global barrier only where the data dependence requires it. |
| CuTe DSL → one persistent kernel | One cooperative, resident grid per complete cached decode step is the first strict target. CUDA Graph replay of that grid is compatible with the probe. A grid that remains alive across many generated tokens is a separate design with sampling and state-management implications. |

The one-grid cached decode step is a feasibility target, not a project-level
performance claim. A useful inference system must also handle longer contexts,
multiple active sequences, prefill and model-specific attention correctly.

The main first-principles constraint is a latency budget: read the relevant
weights and KV state, perform the low-batch projections and attention, and
complete synchronization in less time than the fastest equivalent captured
baseline. A new scheduler cannot recover time lost to a slow GEMV body. TMA,
tensor cores and warp specialization are tools for measured bottlenecks, not
assumptions about every batch-1 shape.

## What the CuTe pilot established

The project now has a `cute` extra pinned to `nvidia-cutlass-dsl[cu13]==4.8.0`,
an isolated `.venv-cute`, the [manual probe](benchmarks/cute/megakernel_probe.py),
[dummy decoder block](benchmarks/cute/decoder_block_probe.py),
[full-step dummy megakernel](benchmarks/cute/dummy_llm_megakernel.py),
[BF16 projection probe](benchmarks/cute/bf16_gemv_probe.py),
[run instructions](benchmarks/cute/README.md),
[MLP and handoff results](benchmarks/cute/results_h200_mig.json),
[decoder-block results](benchmarks/cute/results_decoder_blocks_h200_mig.json),
and full-step results at [small](benchmarks/cute/results_dummy_megakernel_small_h200_mig.json),
[medium](benchmarks/cute/results_dummy_megakernel_medium_h200_mig.json), and
[long-context](benchmarks/cute/results_dummy_megakernel_long_context_h200_mig.json)
shapes.
The environment is
PyTorch 2.14.0+cu130 on an H200 MIG 3g.71gb with 60 visible SMs. The probe uses
float32, random resident weights, and CUDA-event medians after compilation.

- **Execution:** Eight CTAs executed three stages with two global handoffs and
  exact output. An epoch counter in device memory made repeated launches and
  CUDA Graph replays work without a separate counter reset. A captured replay
  of the one-kernel handoff took about **3.7 µs**. A requested 4096-CTA
  cooperative grid failed with `cudaErrorCooperativeLaunchTooLarge` (720),
  confirming the residency limit is enforced.
- **Small composed MLP:** The one-CTA CuTe RMSNorm → SwiGLU → residual probe
  measured about **14.7 µs** at batch 1, hidden 128, intermediate 256, versus
  **15.8 µs** for its PyTorch CUDA Graph reference. At batch 8 it measured
  about **15.0 µs** versus **24.4 µs**. These are small float32 cells dominated
  by fixed overhead, and do not predict model-scale performance.
- **SmolLM2-360M MLP dimensions:** At batch 1, hidden 960 and intermediate
  2560, one CTA took about **924 µs**. A 60-CTA scalar-output cooperative
  version with counter reset captured in the graph took about **125 µs**.
  A warp-reduction version with prepacked transposed weights took about
  **70 µs**, while the equivalent PyTorch CUDA Graph took about **30.4 µs**.
  The best manual CuTe body is therefore roughly **2.3 times slower** in this
  pilot. Max absolute error against float32 PyTorch was below `7e-5` for that
  warp version. Profiling one baseline projection showed a cuBLAS GEMV kernel.
- **Dummy decoder block:** A separate one-CTA CuTe kernel composed RMSNorm,
  Q/K/V, a K/V cache write, full softmax attention, output projection,
  residual, second RMSNorm and SwiGLU. At hidden 64, MLP 128, two heads,
  context 32, it took **35.35 µs** versus **41.79 µs** for the PyTorch CUDA
  Graph. At hidden 128, MLP 256, four heads, context 128, it took
  **114.01 µs** versus **42.46 µs**. Both outputs and K/V writes matched the
  float32 reference (maximum output error below `1.5e-6`). The crossover
  exposes the scaling limit of this one-CTA, repeatedly recomputed-attention
  tactic well before real model dimensions.
- **Actual dummy-model megakernel:** A single cooperative CuTe grid with 4 or
  8 CTAs executes **two decoder layers plus final RMSNorm and vocabulary
  projection**. Each layer performs Q/K/V and a K/V append, stable full cached
  attention, output projection and residual, RMSNorm, SwiGLU and residual.
  Ten device-wide barriers connect dependent stages. Each replay takes the
  next preloaded token embedding, appends at the next cache slot, produces new
  logits, and increments a device epoch. State, logits, and the complete K/V
  cache match PyTorch after **two consecutive advancing steps** (max absolute
  error at most `4.6e-6`). CUDA profiling sees **one CuTe GPU kernel versus
  69 PyTorch GPU kernels per replay**. Fixed-state CUDA Graph medians on this
  H200 MIG were:

  | Layers / hidden / MLP / vocab / prefix / CTAs | CuTe grid | PyTorch graph | CuTe / PyTorch |
  | --- | ---: | ---: | ---: |
  | 2 / 64 / 128 / 128 / 8 / 4 | 49.23 µs | 89.95 µs | 0.55× |
  | 2 / 128 / 256 / 256 / 32 / 8 | 108.91 µs | 92.42 µs | 1.18× |
  | 2 / 128 / 256 / 256 / 128 / 8 | 248.96 µs | 94.21 µs | 2.64× |

  Reset copies and counter clears are outside each timed interval for both
  graphs. These single-replay CUDA-event intervals include the host enqueue
  gap, so the figures are comparative step latencies in this harness, not a
  pure device-only kernel duration. Compilation and initial weight creation
  are excluded. The rapidly rising CuTe time is consistent with its
  deliberately redundant scalar score calculation and whole-grid barriers;
  profiling individual stages is needed to allocate that cost precisely.

### Optimized-math follow-up

The [BF16 GEMV probe](benchmarks/cute/bf16_gemv_probe.py) gives each warp one
output channel, loads prepacked `[output, input]` weights contiguously across
lanes, accumulates in FP32, and reduces in the warp. At input 960/output 2560,
the first 60-CTA schedule took **15.96 µs** versus **4.59 µs** for captured
`torch.mv` using cuBLAS. Sweeping CTA count to 640 brought CuTe to **4.86 µs**
versus **4.62 µs** (5% slower). At the 480 CTAs that fit the full dummy grid,
CuTe took **5.13 µs** versus **4.57 µs** (12% slower). Results:
[60](benchmarks/cute/results_bf16_gemv_960x2560_h200_mig.json),
[480](benchmarks/cute/results_bf16_gemv_960x2560_grid480_h200_mig.json),
[640](benchmarks/cute/results_bf16_gemv_960x2560_grid640_h200_mig.json).
The outputs matched BF16 cuBLAS exactly in these 960→2560 runs. This is an
optimized SIMT tactic for batch-one GEMV, not a tensor-core or TMA kernel.
The same tactic was faster than the selected captured cuBLAS path for
[960→960](benchmarks/cute/results_bf16_gemv_960x960_grid240_h200_mig.json)
(**3.39 µs** versus **7.41 µs**, 240 CTAs) and
[2560→960](benchmarks/cute/results_bf16_gemv_2560x960_grid240_h200_mig.json)
(**5.75 µs** versus **15.34 µs**, 240 CTAs). These are separate GEMVs with
prepacked weights and do not imply that a fused QKV/MLP will retain the gains.

The vocabulary projection is harder: at 960→49152 BF16, CuTe took
**76.17 µs** with 480 CTAs versus **45.28 µs** for captured `torch.mv`.
Increasing to 960 CTAs improved CuTe to **54.94 µs** versus **45.20 µs**,
but such a grid cannot be carried into this cooperative full-step kernel.
See [480-CTA](benchmarks/cute/results_bf16_gemv_960x49152_grid480_h200_mig.json)
and [960-CTA](benchmarks/cute/results_bf16_gemv_960x49152_grid960_h200_mig.json)
results. Packing is excluded from these steady-state timings for both paths.

Inside the [full-step dummy megakernel](benchmarks/cute/dummy_llm_megakernel.py),
one warp now forms each Q·K score once per head and uses one-pass stable online
softmax. All Q/K/V, output, MLP and vocabulary projections can use the same
prepacked warp-reduction scheme. The optimized version still uses one
cooperative grid, ten barriers, a device epoch, and two-step correctness
checks. Its float32 fixed-state results were:

| Two layers: hidden / MLP / vocab / prefix / CTAs | CuTe optimized | PyTorch graph | Observation |
| --- | ---: | ---: | --- |
| 64 / 128 / 128 / 8 / 32 | 23.20 µs | 90.43 µs | scalar CuTe: 49.23 µs |
| 128 / 256 / 256 / 32 / 64 | 35.04 µs | 92.02 µs | scalar CuTe: 108.91 µs |
| 128 / 256 / 256 / 128 / 64 | 64.82 µs | 94.37 µs | scalar CuTe: 248.96 µs |
| 960 / 2560 / 49152 / 128 / 480 | 283.36 µs | 299.94 µs | synthetic model-size dimensions |
| 960 / 2560 / 49152 / 2048 / 480 | 1324.53 µs | 428.62 µs | serial per-head cache scan loses |

The [small](benchmarks/cute/results_dummy_megakernel_warp_all_small_h200_mig.json),
[medium](benchmarks/cute/results_dummy_megakernel_warp_all_medium_h200_mig.json),
[long dummy](benchmarks/cute/results_dummy_megakernel_warp_all_long_context_h200_mig.json),
[model-size prefix-128](benchmarks/cute/results_dummy_megakernel_warp_all_model_vocab_context128_h200_mig.json),
and [prefix-2048](benchmarks/cute/results_dummy_megakernel_warp_all_model_vocab_context2048_h200_mig.json)
JSON files include full correctness errors and profiler-confirmed one-kernel
CuTe replays. A 640-CTA cooperative full-step launch failed with
`cudaErrorCooperativeLaunchTooLarge`; 480 CTAs launched correctly. This is a
real integration constraint: the fastest standalone projection schedule may
not be legal when all stages share one resident grid.

The context-length failure has a concrete cause in the
[dummy attention loop](benchmarks/cute/dummy_llm_megakernel.py): one warp owns a
head and loops serially over every cached token, doing a warp Q·K reduction
and online-softmax update on each iteration. At hidden 960/head dimension 32,
there are 30 heads. With four warps per CTA, only eight of the 480 resident
CTAs do attention work; the others reach the following grid barrier. Increasing
the prefix from 128 to 2048 increases each active warp's loop from about 129
to 2049 iterations. The barrier's early arrivals poll a GPU-scope atomic
counter while waiting, which may further worsen the long-context case; its
share of time has not been profiled. The 283.36→1324.53 µs step increase is
therefore evidence of a serialized, under-occupied attention implementation,
rather than evidence of a CuTe DSL limit or KV traffic alone.

The next attention body should partition the context across CTAs and merge
stable partial softmax states. For the dummy at prefix 2048, 16 nearly equal
context partitions per head would provide 30 × 16 = 480 useful tasks. A real
scheduler must choose the split count from the work and hardware, cover
`(request, query head, context partition)` tasks with uneven lengths and GQA,
and account for the merge and synchronization cost.
[vLLM's Transformers integration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/transformers/base.py)
replaces the model's attention implementation with vLLM attention instances;
its [FlashAttention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)
can select split-KV execution and accounts for the additional partial-output storage.
The linked model directory is an integration layer, not the decode-attention
kernel itself.

These remain dummy models with only two layers, 32-dimensional full-attention
heads, random float32 model weights, a preloaded input embedding, and no RoPE,
GQA, sampling, native BF16/FP16 policy, or exact checkpoint behavior. Matching
hidden, MLP, and vocabulary dimensions does not make the dummy SmolLM2. The
full-step comparison is against a captured eager PyTorch graph with 69 or 71
GPU operations, not the best `torch.compile` or serving engine. The isolated
BF16 GEMV result is not yet a full BF16 decoder. The figures establish that
improved device math can reverse the tiny-model loss; they do not establish a
real-model win. Weight packing
was excluded from timed calls and must be included in cold-start economics.
The `torch.compile(max-autotune)` direct-call result was slower than the
captured PyTorch reference in the earlier MLP cell; that observation does not
make the new full-step PyTorch graph the strongest available baseline.

Correct stream handling mattered: CuTe's default launch stream initially
produced an empty PyTorch CUDA Graph and invalid timings. The final probe
passes `cutlass.torch.current_stream()` to every CuTe entry, verifies outputs
after replay, and records only those stream-correct measurements.

## How existing megakernels are built

[Hazy Research's Llama-1B megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
uses a GPU-resident interpreter with per-SM instruction schedules and a
common CUDA template for fused RMSNorm/QKV/RoPE, attention, projections and
MLP instructions. It coordinates dependent instructions with global counters,
including chunk-level MLP readiness, and pages shared memory so the next
instruction can prefetch weights while prior work finishes. Its control model
is more precise than uniform model-wide barriers.

[Mirage Persistent Kernel](https://arxiv.org/html/2512.22219v2) lowers tensor
operators to an SM-level task graph, generates CUDA task implementations, and
uses in-kernel workers and schedulers to dispatch ready tasks. Its published
design supports cross-operator pipelining and resource sharing. These systems
demonstrate full-model megakernels, while our CuTe probe demonstrates only the
smaller device-composition and synchronization primitives needed to attempt
one.

NVIDIA's [CuTe DSL calling convention](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
supports device `@cute.jit` helpers inside a `@cute.kernel` and cooperative
launches. Its [task-scheduling layer](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html)
targets asynchronous resources and warp/CTA pipelines. The current
[limitations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html)
say it cannot yet schedule existing CuTe kernels through FrontendNext and list
convolution as unsupported. It cannot be assumed to supply MegaBake's
whole-model task scheduler, especially for Muse2's convolution path.

## Why the order changes

The latest measured v3 cell is a warning about end-to-end performance. On the
H200 MIG 3g.71gb SmolLM2-135M batch-1,
length-128 cell, the strict one-grid call measured **36,842 µs** median against
**2,474 µs** for validated `torch.compile/max-autotune` (14.9 times slower).
This is one cell with fixed-state replay, not a six-model result. The source is
[`ART/tasks/V3R-028/full_step_performance.json`](ART/tasks/V3R-028/full_step_performance.json).

The optimized dummy corrects the original interpretation: its early loss came
largely from redundant attention work and an under-parallelized projection
schedule. A standalone BF16 960→2560 GEMV can approach cuBLAS with enough
CTAs. The full-grid occupancy limit and the 960→49152 vocabulary gap remain,
and one-warp-per-head attention scans 2048 cache positions serially. A new
graph IR, scheduler and broad kernel library remain proposals until those
bottlenecks are addressed against a strong real-model baseline.

## Workload and six-model census

Use GPU-resident weights, advancing KV/state, and each checkpoint's native
BF16/FP16 policy on the available H200 MIG (60 visible SMs). The batch-1
length-128/2048 pair is an initial diagnostic, not an acceptance gate. The
project scorecard must include prefill, sustained generation with growing
contexts, multiple concurrent sequences with both equal and unequal lengths,
and batch sizes 1, 4 and a larger device-feasible size. Use short, medium and
long valid contexts for each model. Report prefill latency, per-token decode
latency across the generation, aggregate throughput, peak memory and
correctness. For kernel comparisons, match logits and new state; for serving
comparisons, match sampling and observable outputs. Compare against the best
correct `torch.compile` path and the best correct supported vLLM model and
attention backend, including native model code where available. Show every
cell, including losses and unsupported cells. Treat
multi-token speculative verification as a separate workload. Record the exact
checkpoint revision, Transformers/source revision, attention implementation,
cache layout and capacity, and output ownership for every cell.

| Requested name | Initial checkpoint to pin | Relevant structure and capture route |
| --- | --- | --- |
| SmolLM2-360M | [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/blob/main/config.json) | Llama decoder; full attention with grouped KV heads. First conventional cached-step reference. |
| DeepSeek-V3-DRAFT-0.6B | [jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0](https://huggingface.co/jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0) | Qwen2-based draft for DeepSeek, not a DeepSeek-V3 architecture. |
| Kimi-K2-Instruct-DRAFT-0.6B | [jukofyork/Kimi-K2-Instruct-DRAFT-0.6B-v3.0](https://huggingface.co/jukofyork/Kimi-K2-Instruct-DRAFT-0.6B-v3.0) | Qwen2-based draft for Kimi, not a Kimi-K2 architecture. |
| Qwen3.5-0.8B | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json) | Text-only path of a multimodal model; 18 linear-attention and 6 full-attention layers in its current config. |
| Gemma-3-270M | [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | Gemma3 text decoder with local/global attention pattern; checkpoint access may require license acceptance. |
| Muse2-230M | [Muse-research/Muse2-230M](https://huggingface.co/Muse-research/Muse2-230M/blob/main/config.json) | Custom PyTorch loader, mixed convolution and full-attention layers, parallel MLP. |

The two draft checkpoints test Qwen2 reuse across different weights and
vocabularies; they do not add two new operator families. Qwen3.5 and Muse2
make a plain RMSNorm → QKV → softmax attention → SwiGLU demo insufficient as a
coverage test. The config and author/model sources above determine the initial
census; the captured FX graphs determine the actual operator census.

## Gate 1: inspect the real graphs

1. Reuse the v3 cached-step ABI and graph dump machinery, starting from
   [`benchmarks/v3/capture_hf_step.py`](benchmarks/v3/capture_hf_step.py).
   Capture complete advancing decode steps at both diagnostic lengths and
   batches 1 and 4 for all six checkpoints; capture representative prefills as
   a separate path. Run consecutive steps against the native model to expose
   cache mistakes.
   Use the text-only Qwen path and Muse's own loader. Record an explicit failure
   if a full step cannot be captured; a partial graph is not a full step.
2. Save the raw Dynamo FX graph through a custom `torch.compile` backend for
   graph-break diagnosis. Save both the initial `torch.export`/ATen graph
   (including mutation information) and a `run_decompositions()` version for
   functionalized operator grouping. Dump full `GraphModule.code`, graph
   signature, guards/shape constraints, node metadata, and input/output state
   maps; verify that functionalization preserves cache writes. PyTorch
   documents these capture contracts
   [here](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html)
   and [here](https://docs.pytorch.org/docs/stable/export.html).
3. Produce a one-page atlas per model: node counts by target, a representative
   block's actual FX, repeated-block differences, state reads/writes, shape and
   stride changes, opaque/custom ops, and graph breaks. Keep links to the full
   dumps. Read the atlas manually before assigning reusable block names.

Only then group equivalent regions by *semantic contract*, not checkpoint
name: reduction axis and epsilon for norm; shape/dtype/layout for linears;
RoPE convention; GQA and attention mask/window; cache update and ownership;
MLP gating and residual order. Linear attention and convolution require their
own contracts. Initially, any unsupported region sends the complete step to
the normal compiler path; partial-step partitioning needs separate evidence.

## Gate 2: prove CuTe DSL on model-shaped work

The isolated `.venv-cute` now runs CuTe DSL 4.8.0 with PyTorch 2.14.0+cu130;
the project dependency and exact setup are in
[`pyproject.toml`](pyproject.toml) and
[`benchmarks/cute/README.md`](benchmarks/cute/README.md). The existing v3
`.venv` has `nvidia-cutlass==3.8.0.0` under the same Python import name, so
keep it separate. NVIDIA's
[install guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
documents the CUDA 13 wheel route. The pilot compiled, launched, captured and
validated actual CuTe kernels; the toolchain-smoke question is closed.

Build one **manually written** CuTe DSL decoder block at the SmolLM2-360M
shapes (hidden 960, MLP 2560, 15 query heads, 5 KV heads, head dimension 64),
BF16, short and long contexts, and batches 1 and 4. The SIMT GEMV and
dummy-grid experiments above provide tactics and failure cases; neither
satisfies this checkpoint gate:

1. Close the remaining projection gap: the 960→960 and 2560→960 BF16 SIMT
   probes beat their selected cuBLAS kernels, and 960→2560 is within 12% at
   the 480-CTA legal full-grid size, but 960→49152 is 68% slower at that grid.
   Compare other SIMT layouts and tensor-core tactics
   where legal, then test the selected body inside the full grid. Measure
   against PyTorch's selected cuBLAS/cuBLASLt path and the exact
   Inductor-generated alternative. Include weight-packing cost separately.
2. Replace the one-warp-per-head serial cache scan with context tiles across
   CTAs, a stable reduction of partial softmax states, and only the required
   producer/consumer handoffs. Measure the attention body separately at short
   and long contexts, with uniform and nonuniform batches. The current dummy
   is 3.1 times slower than its captured eager reference at prefix 2048 even
   after the first optimization.
   Add RMSNorm, RoPE, GQA, cache update and attention with the same state ABI.
   Compare individual CuTe bodies with selected PyTorch/SDPA kernels, then
   compose one complete block and two consecutive blocks. Reuse device-callable
   `@cute.jit` helpers inside one `@cute.kernel`; a kernel cannot call another
   `@cute.kernel` as a device function per the
   [DSL calling convention](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
3. Measure steady-state GPU time and synchronized whole-call time, compiled
   registers, shared memory, spills, code size, active CTAs, and effective
   weight traffic. Keep compilation and weight-packing cost separate. Use
   multiple correct tactics where warranted; a faster standalone body can
   become slower when composed into a mixed-resource grid.
4. Use the optimized full-step dummy grid as the correctness scaffold for a
   real block. Its first redundant score computation and scalar projections
   have been replaced; add RoPE, GQA, native precision, masks and cache
   layout from the captured model contract. A global stage barrier needs a
   cooperative launch and a legal resident grid; CUDA's
   [cooperative-groups contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
   governs this. The dummy grid proves full-step feasibility on this MIG
   slice; occupancy, memory ordering, race checks and resource usage still
   need an audit for the real body. Use
   fine-grained producer/consumer counters where they shorten the critical
   path. Avoid a whole-grid barrier after every FX op.

The next technical gate is **correctness plus a measured route to a whole-step
win**: the composed block must be competitive with an equivalent captured
PyTorch block at the same shapes, and a conservative accounting of measured
body work, memory traffic and synchronization must fit below the best full-step
baseline. If it does not, improve the body or abandon the strict-grid route
before expanding compiler infrastructure. A small win in an isolated norm or
one batch-1 short-context cell is not a project-level pass. The broad,
real-checkpoint scorecard above decides any overall claim.

## Gate 3: choose the compiler boundary from evidence

The proposed "stop Inductor before normal kernel scheduling" point is not a
stable, complete optimized-FX handoff. In pinned PyTorch 2.14, Inductor has
post-grad graph passes and a prototype `_pre_fusion_custom_pass`, but the latter
operates on internal scheduler nodes after lowering; layout, aliasing and
memory choices are intertwined with lowering/scheduling. Read or instrument
that pass for an atlas comparison, with a version-pinned adapter, only if it
supplies facts missing from `torch.export`. Treat Inductor's generated kernels
and autotune choices as a performance teacher throughout. Relevant source:
[`torch/_inductor/config.py`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/config.py)
and [`torch/_inductor/graph.py`](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/graph.py).

If Gate 2 passes, introduce the smallest useful IR:

- **Semantic graph:** ATen-origin values, shape/stride/dtype, exact effects and
  alias/state rules, guards, and verified reusable regions. Preserve an FX
  origin and an eager/export oracle for every rewrite.
- **Physical plan:** selected device helper/body, tile and CTA ownership,
  scratch lifetime, stage dependence and legal synchronization. Materialize
  only the choices measured in Gate 2.
- **NVIDIA lowering:** generate one CuTe DSL entry for supported regions.
  A step containing an unsupported region uses ordinary `torch.compile`;
  mark that result as a fallback, not as a strict one-grid win.

NVIDIA's [experimental CuTe task-scheduling layer](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html)
may reduce warp pipeline boilerplate, but its own
[limitations](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html)
must be checked against cross-CTA composition before making it the IR or
runtime foundation.

## Measurement and stop rule

For every declared model/length/batch/prefill cell, use identical weights,
cache advance, precision, masks, and outputs. Validate sustained advancing
generation as well as two native steps.
Select the fastest *equivalent and correct* warmed `torch.compile` mode among
default, reduce-overhead, and max-autotune; verify whether CUDA Graph capture
actually occurred. PyTorch documents the modes
[here](https://docs.pytorch.org/docs/stable/generated/torch.compile).
Record median synchronized user-visible step latency, GPU event latency,
kernel/launch count, memory use, compile time, and any fallback. Also report
prefill latency and generation-latency distributions for the broader scorecard.
Profile separately from the final timing run. Freeze candidates before fresh paired
validation. Report strict win, strict loss, inconclusive, incorrect, or
unsupported for all six models; do not omit failing cells.

Next concrete artifacts: six complete-step FX graph atlases with advancing
state checks and prefill captures, and a native-BF16 SmolLM2 block using the
actual GQA/RoPE/cache contract. The 960→2560 standalone CuTe GEMV is already
near cuBLAS; focus on
the vocabulary path, legal cooperative-grid residency, and long-context
attention before composing a real full step. The new MegaBake planner starts
only after that block and its measured whole-step budget support a win.
