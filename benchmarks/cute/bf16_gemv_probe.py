"""CuTe batch-one BF16 GEMV versus cuBLAS at decoder projection sizes.

Weights are prepacked as [output, input] for contiguous warp K loads.
Packing and compilation are outside the steady-state measurement.
"""

from __future__ import annotations

import argparse
from datetime import date
import importlib.metadata
import json
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch

from megakernel_probe import cuda_event_us, warp_sum


@cute.kernel
def warp_gemv_kernel(x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = tid // 32
    lane = tid % 32
    n, k_size = weight.shape
    for j in range(block * 4 + warp, n, grid * 4):
        partial = cutlass.Float32(0.0)
        for k in range(lane, k_size, 32):
            partial += cutlass.Float32(x[k]) * cutlass.Float32(weight[j, k])
        total = warp_sum(partial)
        if lane == 0:
            out[j] = cutlass.BFloat16(total)


@cute.jit
def warp_gemv(x: cute.Tensor, weight: cute.Tensor, out: cute.Tensor, grid: cutlass.Constexpr[int], stream: cuda.CUstream):
    warp_gemv_kernel(x, weight, out).launch(grid=(grid, 1, 1), block=(128, 1, 1), stream=stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=int, default=960)
    parser.add_argument("--output-dim", type=int, default=2560)
    parser.add_argument("--grid", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.input % 32:
        parser.error("input dimension must be divisible by 32")
    torch.manual_seed(0)
    x = (torch.randn(args.input, device="cuda") * 0.1).to(torch.bfloat16)
    weight = (torch.randn(args.output_dim, args.input, device="cuda") * 0.1).to(torch.bfloat16)
    out = torch.empty(args.output_dim, device="cuda", dtype=torch.bfloat16)
    tensors = tuple(from_dlpack(t) for t in (x, weight, out))
    compiled = cute.compile(warp_gemv, *tensors, args.grid, cutlass_torch.current_stream())

    def cute_call():
        compiled(*tensors, cutlass_torch.current_stream())

    def reference():
        return torch.mv(weight, x)

    cute_call()
    expected = reference()
    torch.cuda.synchronize()
    max_abs = (out.float() - expected.float()).abs().max().item()
    if not torch.allclose(out.float(), expected.float(), atol=1.0e-2, rtol=1.0e-2):
        raise AssertionError(f"GEMV mismatch: max_abs={max_abs}")

    cute_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cute_graph):
        cute_call()
    torch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_graph):
        baseline_out = reference()
    cute_graph.replay()
    torch_graph.replay()
    if not torch.allclose(out.float(), baseline_out.float(), atol=1.0e-2, rtol=1.0e-2):
        raise AssertionError("captured GEMV mismatch")

    result = {
        "date_utc": date.today().isoformat(),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"input": args.input, "output": args.output_dim, "grid_ctas": args.grid},
        "dtype": "bfloat16",
        "weight_layout": "contiguous [output, input]",
        "max_abs_vs_cublas": max_abs,
        "cute_cudagraph_us": cuda_event_us(cute_graph.replay),
        "torch_mv_cudagraph_us": cuda_event_us(torch_graph.replay),
    }
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch_graph.replay()
        torch.cuda.synchronize()
    result["torch_cuda_kernel_names"] = [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
