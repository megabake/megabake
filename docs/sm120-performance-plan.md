# SM120/SM121 performance plan

Prepared 2026-11-05 for SM120 commit 9e45cf6. This is a development plan, not a claim of new benchmark improvements. Primary measured hardware is GB10 (SM121); validate SM120 separately before generalizing results.

## Implementation progress, 2026-11-05

Implemented batch-one GEMV priority in the planner and JSON search reports under the compiler output directory (`generated/plan_search.json` for full-model benchmarks). Added direct coverage for narrow MMA and GEMV, plus regression checks for ordering and diagnostic output.

Root cause confirmed: table-launch compilation defaults to one measured plan; large graphs also seed enumeration with one combination when the Cartesian space exceeds the limit. MMA-first ordering therefore excluded the existing GEMV from the measured full-model plan. The revised priority changes that initial combination. General multi-plan exploration remains future work.

GB10 results, BF16 decode, capacity 128, 40 samples, five warmups, pipelined calls:

- SmolLM-135M unchanged baseline: 5.051 ms; GEMV-only change: 2.492 ms (50.7% lower latency). Same-session graphed Inductor: 2.481 ms.
- Qwen2-0.5B: 7.386 ms with GEMV priority (the exploratory activation-reuse body was also present). Same-session Inductor: 6.009 ms. Historical MegaBake baseline: approximately 15.20 ms; this is not a controlled same-session speedup comparison.
- SmolLM-1.7B GEMV-only: 18.656 ms; same-session Inductor: 18.075 ms. Historical MegaBake baseline: approximately 46.86 ms, again not a controlled same-session comparison.

The wide-MMA activation-reuse prototype passed GPU tests but was removed: it added accumulators without establishing an end-to-end benefit. Two initial experimental benchmark runs accidentally imported the original editable package and are excluded as evidence of modified-kernel performance. Corrected runs explicitly set `PYTHONPATH=src:.` in the isolated checkout `/tmp/megabake-sm120-perf-20261105` on DGX.

Reports: `/tmp/sm120-perf-baseline`, `/tmp/sm120-perf-gemv-only`, `/tmp/sm120-perf-qwen`, `/tmp/sm120-perf-17b` on DGX. These measurements establish a promising first improvement, not completion of this plan. Repeated interleaved trials, matched-context profiling, broader model/prefill validation, native SM120 testing, pipelining and fusion remain outstanding.

## Evidence and diagnosis

The existing GB10 SmolLM-135M report records 40 samples, capacity 128, and pipelined call times of 2.575 ms for graphed Inductor versus 5.152 ms for graphed MegaBake. Both finish at position 87. MegaBake already uses one launch; reducing launch count is not the primary opportunity.

The earlier iteration-1 phase profile attributes 3,888 us of 4,279 us of timed work (90.9%) to 210 narrow MMA phases. It records 48 workers, 398 phases, 181 barrier intervals, and three untimed trailing phases. Some slow GEMM intervals have only 18 tiles. This strongly prioritizes projections, but must be remeasured on the final candidate set: this profile is not the final 5.152 ms run and its category sums are not interchangeable with end-to-end timing.

Sources on DGX: `artifacts/sm120/iteration1-phase-profile.json` and `artifacts/sm120/final/HuggingFaceTB-SmolLM-135M.json/report.json`. The “Identify Laya optimisations” chat reports an exploratory 64-column candidate regression from about 5.00 to 7.36 ms; that candidate was removed. Treat the precise selection failure mechanism as unproven until reproduced.

### What Hopper does well

- `src/megabake/bodies/cuda/sm90/gemm.py` and `cutlass_gemm.py`: staged TMA/WGMMA GEMM and swapAB layouts put useful weight work into small-token tiles. The code documents bandwidth gains from swapAB; these are Hopper results, not GB10 forecasts.
- `docs/roadmap.md`, step 1: paired gate/up GEMM, residual and activation epilogues, and RoPE inside attention reduce materialization and phase overhead. The documented 135M phase count fell from 603 to 278 in that experiment.
- The roadmap's attention work reduces statistics exchanges and reuses KV data across grouped query heads. SM12x already inherits this attention implementation: audit actual dispatch and tune its parameters rather than assuming it needs a fresh port.
- Same-session comparisons, phase profiling, CUDA-graph baselines, and reverting unsuccessful experiments make the evidence useful.

### What limits the current SM12x path

1. **Small-M mismatch.** A 16-row MMA tile can do substantial padded work for one-token decode. A GEMV candidate exists, but adding it does not prove that search explores or selects it for the expensive projections.
2. **Wide tiling without full reuse.** `emit_sm12x_wide_mma` processes four 32-column segments, each with its own full K loop. It reloads A for each segment at source level. Cache/compiler effects need measurement; the implementation does not explicitly share an A fragment across four live accumulators.
3. **No explicit load/compute pipeline in this body.** The generated loop directly loads register fragments and executes synchronous warp MMA. It has no shared-memory staging or asynchronous copy pipeline.
4. **Too little parallel work for some shapes.** The 18-tile intervals cannot use all 48 persistent workers simultaneously. Wider tiles can worsen this even if they reduce loop overhead.
5. **Missing GEMM fusion coverage.** The inherited SM100 provider accepts unbiased 2-D GEMMs with a plain cast epilogue. It rejects richer post expressions, unlike Hopper's epilogue and paired-GEMM path.
6. **Selection uncertainty.** Search is bounded and measurements use limited samples. Candidate ordering, explored combinations, isolated timing, state reset, and whole-model replay can disagree. The 64-column regression is a reason to instrument this, not proof that one particular heuristic is broken.

## Ordered implementation work

### 1. Establish a reproducible baseline and selection trace

Files: `benchmarks/full_model.py`, `tools/phase_profile.py`, `src/megabake/compiler.py`, `planner.py`, and `plan.py`.

- Freeze commit, model/checkpoint, dtype, M/N/K, strides, GPU/software versions, workers, clocks/power conditions, KV capacity and positions.
- Reprofile current SM120 at matched contexts. Record projection shape, chosen provider/configuration, tiles per worker, registers, spills, shared memory, phase work and waits.
- Log every explored/rejected candidate and its measured score. Force narrow MMA, wide MMA, and GEMV individually to establish whether the search misses a faster choice.
- Compare the same shapes with device-resident PyTorch/cuBLAS and available compatible CUTLASS kernels. Measure both standalone and inside the persistent kernel.
- Use at least 40 samples and three interleaved baseline/candidate rounds, with identical state progression. Keep compilation out of steady-state timing. Report pipelined latency separately from synchronized calls and device time.

Exit: reproducible bottleneck attribution and an explanation of why the winning plan is selected. Keep the existing plan available as an explicit baseline candidate.

### 2. Optimize batch-one GEMV before broad GEMM tuning

File: `src/megabake/bodies/cuda/sm120/gemm.py`.

- Specialize for actual operand strides. The current warp dot product reads adjacent K values efficiently only for appropriate weight layout; inspect transaction efficiency before changing vector widths.
- Sweep output columns per CTA, warps per CTA, aligned vector loads, and activation reuse. Preserve tails and a strided fallback.
- For small output tile counts, test smaller output tiles first; then test limited K partitioning with deterministic FP32 reduction. Account for scratch, reduction work, synchronization and altered accumulation order.
- Compare with a transposed/swapAB-style MMA formulation where legal. Transfer Hopper's layout principle, not its instruction sequence.

Exit: improvement on the highest-cost projection shapes that survives integration into the full model; numerical/KV correctness maintained.

### 3. Add genuine reuse and pipelining for multi-row GEMM

- Prototype a K-outer loop with multiple live output accumulators so one activation fragment feeds several output segments. Check increased register pressure and spills before retaining it.
- Add a separate staged shared-memory/asynchronous-copy candidate using instructions supported by the installed SM12x toolchain. Sweep a small set of stage counts and tile shapes against occupancy and useful work.
- Query real target resources instead of inheriting Hopper shared-memory budgets. Keep simple synchronous bodies for shapes where pipeline setup costs more than it saves.

Exit: lower model latency, not merely higher isolated GEMM throughput. Retain a shape-specific choice between GEMV, synchronous MMA and pipelined MMA.

### 4. Port missing fusion contracts

Files: SM12x GEMM provider, `ops/gemm.py`, and generated epilogue support.

- Add bias/residual epilogues, then SiLU/GELU-times-up and paired gate/up where supported by the captured semantics.
- Reuse the existing recognition/contracts, preserving rounding points and resource accounting. Avoid materialized repacked weights unless their lifetime, invalidation and memory cost are justified.
- Measure fewer loads/stores and phase intervals alongside register allocation. Compare feature-off, feature-on, and compiled-but-inert controls if unrelated code changes move performance.

Exit: correctness over repeated graphed decode and a measurable reduction in full-model latency without prefill regressions.

### 5. Tune attention and scheduling from the new profile

- Audit inherited grouped-KV and one-exchange behavior on SM121. Tune chunking and worker allocation for contexts 128, 1024 and 4096, plus short contexts and sliding-window boundaries.
- Avoid creating a second wave for short contexts unless measured beneficial. Test head dimensions and GQA ratios separately.
- Address remaining barrier/phase overhead only after measuring it. Defer dynamic queues, cross-op producer roles, and broad runtime rewrites until the simpler body changes plateau.

## Acceptance and stopping rules

- Every change passes relevant unit/GPU tests, BF16/FP16 tolerances, tails, noncontiguous supported layouts, repeated binding, logits and KV mutation checks. Never relax correctness tolerances to manufacture speedups.
- Proposed promotion gate: repeatable >5% improvement in the targeted full-model workload, or a smaller gain clearly outside measured noise; no unexplained >3% regression on the comparison matrix. Treat these as experiment gates, not promised outcomes.
- Matrix: SmolLM-135M, SmolLM-1.7B and Qwen2-0.5B first; then all accessible models, decode contexts and representative prefill sizes. Explicitly mark gated/unsupported models and architecture-skipped tests.
- First milestone is reproducibly shrinking the approximately 2x 135M gap against graphed Inductor; eventual parity requires measurement. No reliable speedup forecast exists yet.
- If the historical 90.9% projection fraction persisted, halving that work would give roughly 1.83x total speedup by Amdahl's law. This is prioritization arithmetic, not a prediction for the final benchmark.
- Hopper's split-K and weight-streaming experiments sometimes regressed. Keep both opt-in until they win here. Do not copy Hopper WGMMA or SM100-specific kernels into SM12x: NVIDIA documents these as distinct architecture targets (https://github.com/NVIDIA/cutlass). The cp.async tutorial is a design reference, not proof of a ready-made SM121 body: https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu.

Deliver this as small changes: baseline/selection diagnostics; decode GEMV; pipelined MMA; epilogue fusion; attention tuning. Each should carry its own before/after evidence and rollback decision.


## Bounded-search increment

Added deterministic grouped alternatives for large SM12x graphs (beam width four), an eight-plan default for SM12x table launches, and 40-sample finalist verification with compiled-kernel reuse and a 3% incumbent retention window. Existing explicit budgets and the Hopper default remain intact. Benchmark provenance now rejects imports from another checkout and records resolved checkpoint/source/toolchain information. `benchmarks/model_matrix.py` orchestrates three interleaved rounds for all eight models at contexts 16/128/1024/4096, writes explicit pass/fail/gap cells, and pins the checkpoint revision after the first successful resolution. It does not establish prefill acceptance.

Measured SmolLM-135M at 2.499 ms/token with bounded search. One-/two-warp GEMV candidates passed direct correctness coverage but did not improve the complete workload (2.528 ms); removed them. Unit suite: 177 passed before candidate experiments. Experimental expanded suite: 215 passed, 24 skipped; final resource checks: 9 passed. The suite-wide matrix, prefill protection, pipeline/fusion work and the 0.90 latency-ratio objective remain uncompleted. No suite-wide performance claim is supported yet.
