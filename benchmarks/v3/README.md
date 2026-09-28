# V3 workload manifests

`manifests/tiny_cached_step.json` records the synthetic complete-step ABI and its environment. Its `StepManifest.contract_hash` is `ff2c4cd4d241550b262c7f8beeeea37cf815838ec35723904b438efe48c6a183`; the FX graph hash is `aa8614cee9bd602539b22797752086a11918d55dc6151a97eb55db9b5328c1d4`.

These CPU fixture cells check invocation semantics. They are not performance claims; the real Hugging Face scorecard is declared by later workload and baseline cards.

| Cell | State mode | Old valid length | Status |
|---|---|---:|---|
| `tiny-advance-L0` | advancing | 0 | not measured |
| `tiny-advance-L1` | advancing | 1 | not measured |

## Pinned Hugging Face cached-step cells

`smollm2-135m-fp16-b1-L128` and `smollm2-135m-fp16-b1-L2048` pin the
`HuggingFaceTB/SmolLM2-135M` checkpoint revision in their manifests. V3R-002H
captures and checks two consecutive state-advancing calls against the native
Transformers cache path. V3R-003 compares complete calls for `torch.compile`
default, `reduce-overhead`, and `max-autotune`; V3R-004 joins exact FX hot
shapes to the default compile's observed Aten operations and retains the
selected mode's separate unfiltered trace and kernel names. CUDA Graph traces
can hide those Aten shape events, so the selected kernel-to-FX mapping stays
unknown where the trace does not expose it. This inventory and its timings do
not claim a MegaBake kernel or a strict one-grid speedup.

Reproduce the capture and baseline artifacts with:

```sh
.venv/bin/python benchmarks/v3/capture_hf_step.py --lengths 128 2048 --capacity 2050 --artifacts-dir ART/tasks/V3R-002H/capture --manifests-dir benchmarks/v3/manifests
.venv/bin/python benchmarks/v3/run_hf_baselines.py benchmarks/v3/manifests/smollm2-135m-fp16-b1-L128.json benchmarks/v3/manifests/smollm2-135m-fp16-b1-L2048.json --output-dir ART/tasks/V3R-003/baselines --warmup 3 --samples 30 --modes default reduce-overhead max-autotune
```

Run `benchmarks/v3/run_shape_inventory.py CAPTURE_REPORT BASELINE_REPORT OUTPUT`
for each cell to write its V3R-004 inventory.
