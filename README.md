<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# megabake

`torch.export` to megakernel: compile an entire PyTorch model into a **single cooperative CUDA kernel**.

Instead of launching one kernel per operator (matmul, activation, normalization, ...),
megabake fuses the full forward pass into one persistent megakernel that runs all
operators back-to-back on the GPU, communicating through shared memory and
grid-wide barriers -- zero kernel launch overhead, zero host round-trips.

## How it works

```
torch.nn.Module
      |
      v
torch.export (FX graph)
      |
      v
Schedule Compiler          # graph_walker.py: ATen ops -> TaskDesc IR
  |-- Buffer Planner       # arena allocation, weight mapping
  |-- Tiling               # SM work distribution per op
  |-- Serializer           # binary schedule blob
      |
      v
CUDA Compiler              # nvcc: all .cu files -> single .cubin
      |
      v
Megakernel Launcher        # cuLaunchCooperativeKernel, 1 kernel, all SMs
```

**Architecture: BSP (Bulk Synchronous Parallel)**

All SMs execute the same task simultaneously. After each task completes, a
`cooperative_groups::this_grid().sync()` barrier synchronizes the entire grid
before the next task begins. This is the same execution model used by
[Mirage](https://github.com/mirage-project/mirage).

**Matmul: CuTe tensor-core GEMM**

The matmul kernel uses NVIDIA CuTe (CUTLASS) with SM80 MMA tensor-core
instructions (`SM80_16x8x16_F32F16F16F32_TN`), 128x128 output tiles,
swizzled shared memory (`Swizzle<3,3,3>`), `cp.async` pipelined loads, and
LDSM register fills. Runs on SM80+ (Ampere, Hopper).

## Benchmarks

Megabake vs `torch.compile` (inductor) on NVIDIA H200 MIG 2g.35gb (32 SMs).
Source of truth: `benchmarks/results.json` (regenerate with `python benchmarks/bench_compare.py --output benchmarks/results.json`).

```
Model                megabake     torch.compile   Speedup   Kernels
-----------------------------------------------------------------
linear_256x512         61.1us          63.5us       1.04x    1 vs 1
linear_512x1024        67.7us          68.8us       1.02x    1 vs 1
mlp_silu                FAIL         101.8us         —       — vs 3
mlp_gelu               62.5us          95.4us       1.53x    1 vs 3
mlp_3layer             69.5us         119.6us       1.72x    1 vs 5
rmsnorm_mlp             FAIL         111.8us         —       — vs 4
layernorm_mlp           FAIL         114.9us         —       — vs 4
llama_decoder         129.7us         437.9us       3.38x    1 vs 13
```

Three models fail due to weight-loading bugs (`mlp_silu`: missing `fc1.weight`, `rmsnorm_mlp`/`layernorm_mlp`: missing `norm.weight`).
On working models, megabake's single-kernel advantage compounds with complexity — the llama decoder layer runs **3.4x faster**.

### Real Model Benchmarks (decode, batch=1, seq=1)

```
Model                   megabake     torch.compile   Speedup   Kernels
----------------------------------------------------------------------
SmolLM2-135M             7238us          6929us       0.96x    1 vs 424
gemma-2b                28190us          7580us       0.27x    1 vs 256
```

SmolLM2-135M is near parity (0.96x). gemma-2b is 0.27x -- dominated by matmul (84.8% of megakernel time). Bandwidth utilization is 7.4% (SmolLM2) and 35.6% (gemma-2b) vs theoretical floor.

Per-task profile breakdown (run with `--task-profile`):

| Model | Tasks | MATMUL | MATMUL_SILU/GELU | ATTENTION | REDUCE | Other |
|-------|-------|--------|-----------------|-----------|--------|-------|
| SmolLM2-135M | 523 | 78.4% | 11.0% | 1.3% | 2.5% | 6.8% |
| gemma-2b | 320 | 84.8% | 12.4% | 0.4% | 0.7% | 1.7% |

Matmul dominates both models. Optimization priority: skinny matvec cp.async prefetch (bandwidth), then scheduler (barrier elimination), then attention (tensor cores).

## Quick start

```bash
pip install -e ".[dev]"
```

Requires: Python 3.10+, PyTorch 2.4+, CUDA 12+, CUTLASS headers (ships with PyTorch source or set `CUTLASS_PATH`).

```python
import torch
import megabake

model = MyModel().half().cuda()
x = torch.randn(4, 512, dtype=torch.float16, device="cuda")

compiled = megabake.compile(model, x)
output = megabake.run(compiled, model, x)
```

## Run benchmarks

```bash
python benchmarks/bench_compare.py
python benchmarks/bench_compare.py --models mlp_silu,llama_decoder
python benchmarks/bench_compare.py --backend megabake
```

## Per-task profiling

Profile cycle-level timing for every task inside the megakernel — see which ops dominate, SM utilization per task, and barrier overhead:

```bash
python benchmarks/test_harness.py mlp_silu --task-profile
python benchmarks/test_harness.py llama_decoder --task-profile
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --task-profile
```

Output shows per-task breakdown (median/max cycles, active/idle SMs, barrier wait) and a summary grouped by op type. Results auto-save to `benchmarks/baselines/<model>_profile.json`.

Profiling can also be used from Python directly:

```python
import megabake
from megabake.runtime.profiler import profile_model, analyze, print_report

compiled = megabake.compile(model, x)
tasks, cycles = profile_model(compiled, model, x)
print_report(analyze(tasks, cycles), num_sms=32)
```

## Run tests

```bash
pytest                          # all 69 tests
pytest tests/test_tasks/        # per-kernel correctness
pytest tests/test_e2e/          # end-to-end model tests
```

## Project structure

```
src/
  cuda/
    megakernel.cu              # BSP scheduler, dispatch, grid sync
    data_types.cuh             # TaskDesc struct, op codes
    tasks/
      matmul.cu                # CuTe tensor-core GEMM (SM80+)
      attention.cu             # Scaled dot-product attention
      elementwise.cu           # SiLU, GELU, ReLU, add, mul, ...
      reduce.cu                # RMSNorm, LayerNorm, softmax
      embedding.cu             # Token embedding lookup
      copy.cu                  # Strided copy, concat
      rope.cu                  # Rotary position embeddings
      index.cu                 # Gather, scatter, index_select
  megabake/
    __init__.py                # compile() and run() API
    data_types.py              # Python TaskDesc mirror
    schedule_compiler/
      graph_walker.py          # FX graph -> TaskDesc list
      buffer_planner.py        # Arena allocation
      tiling.py                # Per-op SM tile counts
      op_table.py              # ATen op -> megabake op mapping
      shape_ops.py             # View/reshape/transpose tracking
      serializer.py            # Binary schedule format
    runtime/
      cuda_compiler.py         # nvcc compilation pipeline
      launcher.py              # cuLaunchCooperativeKernel
      loader.py                # Schedule loading + cached execution
      profiler.py              # Per-task cycle-level profiling
benchmarks/
  bench_compare.py             # megabake vs torch.compile harness
  models.py                    # Benchmark workload definitions
tests/
  test_tasks/                  # Per-kernel unit tests
  test_schedule_compiler/      # Compiler pipeline tests
  test_e2e/                    # Full model end-to-end tests
```

## Supported operators

| Category | Operators |
|----------|-----------|
| Linear algebra | matmul (CuTe GEMM), batched matmul |
| Attention | scaled_dot_product_attention (causal / non-causal) |
| Activations | SiLU, GELU, ReLU, sigmoid, tanh, exp, log |
| Arithmetic | add, sub, mul, div, neg, abs, pow, clamp |
| Normalization | RMSNorm, LayerNorm, softmax |
| Embedding | token embedding, rotary position embeddings |
| Data movement | copy, concat, gather, scatter, index_select |
| Shape ops | reshape, transpose, permute, expand, squeeze, slice (zero-cost view tracking) |

## Design references

- [Mirage](https://github.com/mirage-project/mirage) -- Multi-level superoptimizer with persistent megakernels (BSP execution model, CuTe GEMM, WGMMA on Hopper)
- [Luminal](https://github.com/jafioti/luminal) -- ML compiler with cuBLASLt matmul and egglog-based fusion

## License

BSD-3-Clause
