# Short-decode GEMV experiments — 2026-11-06

## Outcome

Neither tested approach established the required 5% complete-model improvement over PR #22. Experimental provider, planner and kernel-test changes were removed from the shipping branch. The retained change adds explicit context selection to phase profiling. No new kernel optimization is promoted.

Baseline: signed commit `32cd8df965f65ada7c2537e42ce24b23acc97ddf`; GB10/SM121; BF16 SmolLM-135M checkpoint `1d461723eec654e65efdc40cf49301c89c0c92f4`; graphed execution, one cooperative kernel, capacity 128, five warmups and 40 measured calls. Compilation and profiling were outside timing windows. Older matrix jobs were paused during these experiments.

## Measured screening results

- Initial fresh baseline: 2.520 ms/token.
- Corrected combined candidate: 2.541 ms/token; no improvement.
- Adjacent baseline before the final unroll-only screen: 2.576 ms/token.
- Corrected unroll-only screen: 2.550 ms/token. Search selected the original GEMV for all 211 projections, so this approximately 1% difference is not an unrolling gain.
- Adjacent baseline before the final fusion-only screen: 2.546 ms/token.
- Corrected fusion-only screen: 2.567 ms/token. Search retained the unfused plan. The best fused isolated measurement was approximately 2.289 ms versus 2.342 ms for the incumbent, within the existing 3% retention window and below the 5% promotion requirement.

These are screening runs, not three successful matched promotion rounds. Since neither experiment passed screening, the six-model and prefill promotion matrix was not run for these prototypes. No suite-wide performance claim follows from these results.

## Implementation findings

The prototypes tested reduction-loop unroll factors two and four, plus bias/residual epilogues through the existing expression emitter. FP16/BF16 kernel tests covered tails, operand strides, strided residuals, rebinding and rank-three fused outputs. A prototype unit/GPU run passed 228 tests with 24 inapplicable combinations skipped. A subsequent additional CPU enumeration test failed to find a fused alternative for its captured graph; that experimental planner/test change is also not retained.

Two early comparisons are excluded from performance evidence:

- The first fused model failed compilation because its normalized 2-D GEMV output still bound to a physical rank-three tensor. A contiguous flat output view corrected this; rank-three tests were added to the prototype.
- The initial generator explicitly set `unroll=1` for the incumbent, changing the compiler's original default. Runs around 8.9–9.8 ms used that changed behavior and are invalid comparisons with the original loop default. Corrected runs omit the unroll setting on the incumbent and preserve its original output path.

The baseline and an early regressing candidate both reported 168 registers/thread and 32 bytes/thread of local memory. Those resource counts alone do not explain the early regression. Preserving the incumbent's generated loop behavior is essential when measuring alternatives.

## Profiling

`tools/phase_profile.py` accepts `--start-context`. It resets the position and valid-length tensors before both aggregate and per-CTA profiling, so both measurements start at the requested context. It validates available KV capacity and records the starting context in the JSON report. Existing checkpoint and hardware selection remain unchanged.

Short-context baseline attribution: 1,772 us of 1,992 us of timed phase work was GEMV (approximately 89%); attention accounted for 150 us. The separate CTA profile recorded 2,421 us interval span and 114 us barrier-release time. Aggregate phase attribution, per-CTA counters and benchmark latency are distinct measurements and must not be added together. The profile leaves three trailing phases untimed.

At context 128, the aggregate timed phase work was 2,125 us: GEMV remained 1,772 us while attention increased to 284 us. Barrier-release time in the separate CTA profile was 114 us.

The remaining opportunity is projection efficiency. Further work should measure load/address instruction cost and effective weight bandwidth on the unchanged GEMV before revisiting tile sizes, unrolling or fusion.

## Reproduction artifacts on DGX

- Frozen checkout: `/tmp/megabake-short-baseline`
- Combined prototype: `/tmp/megabake-short-candidate`
- Independent prototypes: `/tmp/megabake-short-unroll`, `/tmp/megabake-short-fusion`
- Experimental source patch: `/tmp/megabake-short-prototypes.patch`
- Baseline/candidate reports: `/tmp/megabake-short-round0-baseline`, `/tmp/megabake-short-corrected-round0`, `/tmp/megabake-short-final-unroll`, `/tmp/megabake-short-final-fusion`
- Adjacent comparison log, including both baseline measurements: `/tmp/megabake-short-final-comparison.log`
- Instrumented reports: `/tmp/megabake-short-profile0.json`, `/tmp/megabake-short-profile128.json`

Example profile invocation from the frozen checkout:

```sh
PYTHONPATH=src:. /home/prabhav/Desktop/repositories/megabake/.venv/bin/python tools/phase_profile.py \
  --model HuggingFaceTB/SmolLM-135M \
  --revision 1d461723eec654e65efdc40cf49301c89c0c92f4 \
  --mode decode --capacity 256 --calls 9 --start-context 128 --cta \
  --out /tmp/megabake-short-profile128.json
```
