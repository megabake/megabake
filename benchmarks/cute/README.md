# CuTe DSL feasibility probe

This directory contains manually written CuTe DSL experiments. They check
device-side composition, cooperative cross-CTA handoffs, PyTorch tensor/stream
interop, a model-shaped MLP, a small decoder block, and an actual two-layer
dummy-model decode step in one cooperative CUDA grid.

## Run

Use a separate environment because the v3 environment installs the older
`nvidia-cutlass` Python package under the same `cutlass` import name.

```sh
python3 -m venv .venv-cute
.venv-cute/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.venv-cute/bin/python -m pip install -e '.[cute]'
.venv-cute/bin/python benchmarks/cute/megakernel_probe.py --large --compile-baseline
.venv-cute/bin/python benchmarks/cute/decoder_block_probe.py
.venv-cute/bin/python benchmarks/cute/decoder_block_probe.py --hidden 128 --intermediate 256 --heads 4 --context 128
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py --hidden 128 --intermediate 256 --vocab 256 --prefix 32 --grid 8
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py --hidden 128 --intermediate 256 --vocab 256 --prefix 128 --grid 8
.venv-cute/bin/python benchmarks/cute/bf16_gemv_probe.py --grid 480
.venv-cute/bin/python benchmarks/cute/bf16_gemv_probe.py --input 2560 --output-dim 960 --grid 240
.venv-cute/bin/python benchmarks/cute/bf16_gemv_probe.py --output-dim 49152 --grid 480
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py --warp-online-attention --warp-projections --hidden 128 --intermediate 256 --vocab 256 --prefix 128 --grid 64
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py --warp-online-attention --warp-projections --hidden 960 --intermediate 2560 --vocab 49152 --prefix 128 --grid 480
.venv-cute/bin/python benchmarks/cute/dummy_llm_megakernel.py --warp-online-attention --warp-projections --hidden 960 --intermediate 2560 --vocab 49152 --prefix 2048 --grid 480
```

The optional `cute` dependency pins `nvidia-cutlass-dsl[cu13]==4.8.0`. The
tested device is an H200 MIG 3g.71gb with 60 visible SMs and driver 595.58.03.
The recorded results are in `results_h200_mig.json`,
`results_decoder_blocks_h200_mig.json`, and the
`results_dummy_megakernel_*_h200_mig.json` and
`results_bf16_gemv_*_h200_mig.json` files.

The script contains three related probes:

1. A one-CTA-per-row RMSNorm → gate/up projection → SiLU → down projection →
   residual kernel. `silu` is a device-callable `@cute.jit` helper.
2. A cooperative multi-CTA version of the same MLP with one global producer/
   consumer handoff. One version assigns one output to a thread; another uses
   warp reductions and weights packed in transposed order.
3. An eight-CTA, three-stage handoff with two global barriers. An on-device
   epoch counter lets one CUDA kernel run repeatedly without a reset launch.

`decoder_block_probe.py` implements a dummy one-CTA block with two RMSNorms,
Q/K/V projections, an append to K/V cache, full softmax attention, output
projection and residual, and SwiGLU MLP and residual. It checks both output
and K/V update against PyTorch. The input, weights and cache are synthetic;
the same cache slot is replayed for timing. Its attention uses a simple
unstabilized softmax and has no RoPE or grouped-query attention. This is a
composition check, not a numerically complete model implementation.

`dummy_llm_megakernel.py` is the actual megakernel feasibility test. One
cooperative grid owns a whole dummy decode step: two sequential RMSNorm →
Q/K/V → KV append → stable cached attention → output/residual → RMSNorm →
SwiGLU/residual layers, then final RMSNorm and a vocabulary projection to
logits. The original scalar path uses four or eight CTAs; the optimized path
uses 32–480 CTAs. Each path crosses ten device-side barriers. One
launch consumes one of two preloaded input embeddings, advances its own epoch,
and writes the next K/V slot. The script checks the entire state, logits,
K/V cache and barrier counters against PyTorch over two consecutive steps.
It captures that launch in a CUDA Graph and uses the PyTorch CUDA profiler to
check that one replay contains **one GPU kernel**; the equivalent captured
PyTorch graph has 69 GPU kernels. It must be reset after the two provided
input embeddings are consumed.

The captured one-kernel CuTe path took 49.23 µs versus 89.95 µs for the
equivalent captured PyTorch path at hidden 64, MLP 128, two heads and prefix
8. At hidden 128, MLP 256, four heads and prefix 32, it took 108.91 µs versus
92.42 µs; at prefix 128, 248.96 µs versus 94.21 µs. These are small float32
dummy architectures. The scalar attention recalculates scores per output
element, so the longer-context result is expected to degrade. The dummy
model has no RoPE, GQA, sampling or native checkpoint weights.

`--warp-online-attention` assigns one warp to each attention head. Its lanes
form each Q·K score together and use a one-pass stable online softmax, so a
score is no longer recalculated for each value channel. With
`--warp-projections`, each warp also computes a Q/K/V, output, MLP or
vocabulary channel from prepacked `[output, input]` weights, then reduces its
partial products. This is a CUDA-core SIMT tactic suited to batch-one GEMV;
it does not use tensor cores or TMA. The 128/256/256 dummy at prefix 128 fell
from **248.96 µs** to **64.82 µs** (PyTorch graph **94.37 µs**). At hidden
960, MLP 2560, vocabulary 49152 and prefix 128, it took **283.36 µs** versus
**299.94 µs** for the captured PyTorch graph. At prefix 2048 it took
**1324.53 µs** versus **428.62 µs**: attention still scans the cache serially
within each head. These full-step runs are float32, have two dummy layers and
32-dimensional full-attention heads, and use synthetic weights. They are not
SmolLM2 checkpoint results. The PyTorch graph is captured eager PyTorch, not
the fastest `torch.compile` path.

`bf16_gemv_probe.py` isolates the projection math with actual BF16 operands
and FP32 accumulation. At input 960/output 2560, the 480-CTA CuTe schedule
took **5.13 µs** versus **4.57 µs** for captured PyTorch `torch.mv` using
cuBLAS; 640 CTAs reduced CuTe to **4.86 µs** versus **4.62 µs**. A 640-CTA
cooperative full-step grid failed with `cudaErrorCooperativeLaunchTooLarge`,
while 480 CTAs worked. For a 960→49152 vocabulary projection, 480-CTA CuTe
took **76.17 µs** versus **45.28 µs** for cuBLAS; 960 CTAs reduced CuTe to
**54.94 µs**, still slower and not a legal grid size for this full-step body.
At 960→960, CuTe took **3.39 µs** versus **7.41 µs** for cuBLAS; at
2560→960, **5.75 µs** versus **15.34 µs**. Weight packing was outside timing.
The experiment shows that standalone math quality, full-grid residency, and
long-context scheduling have to be measured separately.

All CuTe calls use `cutlass.torch.current_stream()`. This is required for
ordering with PyTorch tensor operations and for valid PyTorch CUDA Graph
capture. If the stream is omitted, a CUDA Graph can be empty and its timing
meaningless. The script checks captured outputs after replay.

GPU timings are medians from CUDA events after compilation and warmup. Simple
calls use seven trials of 50 repetitions; cooperative direct calls use 30
single calls. CUDA Graph comparisons include the kernels in each captured
graph. The MLP uses float32 and random resident weights; weight packing and
compilation are outside the timed region. This is a feasibility and scheduling
probe, not a BF16 decoder-block or end-to-end model result.

The full-step script times 30 single graph replays. It restores each graph's
cache and the CuTe counters *before* the CUDA-event interval to compare the
same fixed-state step. Single-replay intervals include host enqueue delay.
They should be read as harness step latencies, not pure device kernel durations.

The JSON files are samples from this device. Rerun the scripts before
comparing changes to a new kernel or compiler version.
