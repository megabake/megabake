<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# megabake

**New here?** Read the [MegaBake launch post](blog/megabake-launch.md) for a quick introduction and an honest snapshot of the project's current state.

MegaBake compiles a **supported** `torch.export` graph into a task schedule
executed by one cooperative CUDA kernel. This reduces per-operator kernel
launches; it does not eliminate launch, allocation, or host overhead.

If the graph contains an unsupported ATen operation, compilation warns and
`megabake.run()` executes the original PyTorch module eagerly instead. Check
that warning and the profiler output before treating a measurement as MegaBake
performance.

## How it works

```text
PyTorch module
  ↓ torch.export
FX graph and supported-pattern matching
  ↓
Task schedule and reusable device-memory buffers
  ↓
Per-SM task queues and dependency tracking
  ↓
One cooperative CUDA-kernel launch
```

At runtime, blocks process assigned task tiles and signal dependent tasks when
they finish. Intermediate tensors live in a planned device-memory arena; shared
memory is used locally by individual kernels where needed. The current runtime
uses per-SM queues and dependency counters, not a grid-wide barrier after every
operation.

Matmul tasks use CuTe/CUTLASS-based kernels, including shape-specific paths.
Performance depends on the workload and GPU; a single kernel does not guarantee
that every model is faster.

## Benchmarks

Checked-in synthetic-workload results from
`benchmarks/results_post_inductor.json`:

| Workload | MegaBake | `torch.compile` | Speedup | Kernels (MegaBake / `torch.compile`) | MegaBake max abs diff vs eager |
|----------|---------:|----------------:|--------:|-------------------------------------:|-------------------------------:|
| `linear_256x512` | 51.5 µs | 55.8 µs | 1.08× | 1 / 1 | 0 |
| `linear_512x1024` | 65.7 µs | 64.4 µs | 0.98× | 1 / 1 | 0.000977 |
| `mlp_silu` | 68.2 µs | 103.5 µs | 1.52× | 1 / 3 | 0.000244 |
| `mlp_gelu` | 59.6 µs | 97.1 µs | 1.63× | 1 / 3 | 0.000366 |
| `mlp_3layer` | 60.1 µs | 122.3 µs | 2.04× | 1 / 5 | 0.000069 |
| `rmsnorm_mlp` | 73.3 µs | 108.8 µs | 1.48× | 1 / 4 | 0.000244 |
| `layernorm_mlp` | 62.9 µs | 111.9 µs | 1.78× | 1 / 4 | 0.000244 |
| `llama_decoder` | 133.8 µs | 422.3 µs | 3.16× | 1 / 13 | 0.001953 |

Speedup is `torch.compile` latency divided by MegaBake latency; kernel counts
are MegaBake / `torch.compile`.

The max-difference column compares MegaBake output with eager PyTorch.
These are median CUDA-event latencies for built-in workloads, with 10 warmups
and 100 measured iterations, on an NVIDIA H200 MIG 2g.35gb (32 SMs), CUDA 12.4,
and PyTorch 2.6.0+cu124. Seven of the eight workloads are faster in this
snapshot; `linear_512x1024` is slightly slower. These results are not full-model
Hugging Face decode benchmarks.

Run a Hugging Face model benchmark with:

```bash
python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode --profile
```

```
GPU: NVIDIA H200 MIG 3g.71gb  SMs: 60
Model: SmolLM2-135M (bs=1, seq=1)

-----------------------------------------------------------------------------------
Backend          Compile(ms) Latency(us) GPU Kern(us)  Host(us)  Kernels    MaxDiff
-----------------------------------------------------------------------------------
eager                      —     14272.3       1588.5   12683.8      674          —
torch.compile         4514.2      6024.0       1055.0    4969.0      402   0.632812
megabake              8042.6      4850.2       4723.5     126.7        1   0.195312
-----------------------------------------------------------------------------------
megabake vs eager:          2.94x  (1 vs 674 kernels)
megabake vs torch.compile:  1.24x  (1 vs 402 kernels)
  GPU kernel only:          0.22x  (megakernel 4723us vs TC kernels 1055us)

Model weights: 269.0 MB
Peak HBM bandwidth: 500 GB/s
Theoretical floor (weights/peak_bw): 538 us

BW utilization: eager 3.8%, torch.compile 8.9%, megabake 11.1%

  [eager] Top kernels (total GPU: 1588.5 us):
    nvjet_sm90_hsh_16x64_64x16_4x1_v_bz_TNN                                  x90       319.9 us ( 20.1%)
    nvjet_sm90_hsh_32x64_64x16_4x1_v_bz_TNN                                  x60       210.9 us ( 13.3%)
    nvjet_sm90_hsh_8x64_64x16_4x1_v_bz_TNN                                   x60       179.4 us ( 11.3%)
    void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 12...   x30       179.3 us ( 11.3%)
    void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_...   x120      176.9 us ( 11.1%)
    void at::native::unrolled_elementwise_kernel<at::native::direct_cop...   x62       125.2 us (  7.9%)
    void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, ...   x61       120.9 us (  7.6%)
    void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::nat...   x60        91.6 us (  5.8%)
    void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_...   x61        73.5 us (  4.6%)
    void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_...   x60        72.2 us (  4.5%)
    nvjet_sm90_hsh_64x8_64x16_2x1_v_bz_TNT                                   x1         28.7 us (  1.8%)
    void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::...   x1          1.5 us (  0.1%)
    void at_cuda_detail::cub::detail::scan::DeviceScanKernel<at_cuda_de...   x1          1.5 us (  0.1%)
    void at::native::unrolled_elementwise_kernel<at::native::direct_cop...   x1          1.2 us (  0.1%)
    void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectori...   x1          1.2 us (  0.1%)
    ... + 5 more                                                             x5          4.5 us

  [torch.compile] Top kernels (total GPU: 1055.0 us):
    nvjet_sm90_hsh_16x64_64x16_4x1_v_bz_TNN                                  x90       319.7 us ( 30.3%)
    nvjet_sm90_hsh_32x64_64x16_4x1_v_bz_TNN                                  x60       212.1 us ( 20.1%)
    nvjet_sm90_hsh_8x64_64x16_4x1_v_bz_TNN                                   x60       179.2 us ( 17.0%)
    fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::Att...   x30       155.8 us ( 14.8%)
    triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__...   x30        29.9 us (  2.8%)
    nvjet_sm90_hsh_64x8_64x16_2x1_v_bz_TNT                                   x1         27.1 us (  2.6%)
    triton_poi_fused__unsafe_view_mul_silu_7                                 x30        25.7 us (  2.4%)
    triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__...   x30        23.1 us (  2.2%)
    triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11         x14        17.7 us (  1.7%)
    triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12         x14        17.4 us (  1.6%)
    triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13         x14        17.4 us (  1.6%)
    triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10         x14        16.5 us (  1.6%)
    triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__...   x7          5.0 us (  0.5%)
    triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_r...   x1          1.5 us (  0.1%)
    triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_r...   x1          1.4 us (  0.1%)
    ... + 6 more                                                             x6          5.7 us

  [megabake] Top kernels (total GPU: 4723.5 us):
    megakernel                                                               x1       4723.5 us (100.0%)
```

The harness reports latency, kernel count, and max absolute difference. If it
reports unsupported ATen ops, the measured MegaBake call is eager fallback,
not a megakernel run.

## Quick start

```bash
git clone https://github.com/megabake/megabake.git
cd megabake
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Requires Python 3.10+, an NVIDIA GPU, and CUDA Toolkit with `nvcc`.
`requirements.txt` pins PyTorch 2.6.0+cu124 and installs CUTLASS/CuTe headers.
To use a separate CUTLASS checkout, set `CUTLASS_PATH` before starting Python:

```bash
export CUTLASS_PATH=/path/to/cutlass
```

```python
import torch
import megabake

model = MyModel().half().cuda()
x = torch.randn(4, 512, dtype=torch.float16, device="cuda")

compiled = megabake.compile(model, x)
output = megabake.run(compiled, model, x)
```

## Benchmark commands

```bash
python benchmarks/bench_compare.py
python benchmarks/bench_compare.py --models mlp_silu,llama_decoder
```

For per-task cycle diagnostics on a supported graph:

```bash
python benchmarks/test_harness.py llama_decoder --task-profile
```

The profiler reports per-task cycle statistics and saves JSON under
`benchmarks/baselines/`. This is diagnostic instrumentation, not a latency
benchmark; unsupported graphs fall back to eager execution.

## Run tests

```bash
pytest
pytest tests/test_tasks/
pytest tests/test_e2e/
```

## Coverage

The compiler supports a subset of ATen operations, including common matrix
multiplication, attention, elementwise, reduction/normalization, embedding,
indexing, copy, and shape operations. Support depends on the graph and tensor
metadata; one unsupported operation causes the whole call to fall back to
eager PyTorch. Partial graph acceleration is not implemented.

## License

MIT License ([LICENSE-MIT](LICENSE-MIT)).
