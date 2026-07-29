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

Megabake vs `torch.compile` (inductor) on NVIDIA H200 MIG 2g.35gb (32 SMs):

```
Model                megabake     torch.compile   Speedup   Kernels
-----------------------------------------------------------------
linear_256x512         68.7us          62.0us       0.90x    1 vs 1
linear_512x1024        81.7us          69.8us       0.85x    1 vs 1
mlp_silu              104.7us          95.5us       0.91x    1 vs 3
mlp_gelu               90.4us         101.5us       1.12x    1 vs 3
mlp_3layer             94.2us         122.5us       1.30x    1 vs 5
rmsnorm_mlp           116.4us         113.2us       0.97x    1 vs 4
layernorm_mlp          95.3us         124.5us       1.31x    1 vs 4
llama_decoder         272.6us         441.8us       1.62x    1 vs 13
```

Megabake's single-kernel advantage compounds with model complexity.
The llama decoder layer (13 kernels in torch.compile) runs **1.6x faster**.

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
