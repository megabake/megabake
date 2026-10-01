"""Small CuTe DSL composition and cross-CTA feasibility probes on one GPU.

Run in a clean environment with `pip install -e '.[cute]'`.
"""

from __future__ import annotations

import argparse
from datetime import date
import importlib.metadata
import json
from statistics import median

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch


@cute.jit
def silu(x: cutlass.Float32) -> cutlass.Float32:
    return x / (1.0 + cute.exp(-x))


@cute.jit
def warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    value += cute.arch.shuffle_sync_down(value, 16)
    value += cute.arch.shuffle_sync_down(value, 8)
    value += cute.arch.shuffle_sync_down(value, 4)
    value += cute.arch.shuffle_sync_down(value, 2)
    value += cute.arch.shuffle_sync_down(value, 1)
    return value


@cute.kernel
def rmsnorm_swiglu_kernel(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    down: cute.Tensor,
    scale: cute.Tensor,
    norm: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
):
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    h = x.shape[1]
    intermediate = gate.shape[1]

    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[row, k])
            squares += value * value
        scale[row] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()

    for j in range(tid, h, 256):
        norm[row, j] = cutlass.Float32(x[row, j]) * scale[row] * cutlass.Float32(norm_weight[j])
    cute.arch.sync_threads()

    for j in range(tid, intermediate, 256):
        gate_sum = cutlass.Float32(0.0)
        up_sum = cutlass.Float32(0.0)
        for k in range(h):
            value = norm[row, k]
            gate_sum += value * cutlass.Float32(gate[k, j])
            up_sum += value * cutlass.Float32(up[k, j])
        hidden[row, j] = silu(gate_sum) * up_sum
    cute.arch.sync_threads()

    for j in range(tid, h, 256):
        result = cutlass.Float32(0.0)
        for k in range(intermediate):
            result += hidden[row, k] * cutlass.Float32(down[k, j])
        out[row, j] = result + cutlass.Float32(x[row, j])


@cute.jit
def rmsnorm_swiglu(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    down: cute.Tensor,
    scale: cute.Tensor,
    norm: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
):
    rmsnorm_swiglu_kernel(x, norm_weight, gate, up, down, scale, norm, hidden, out).launch(
        grid=(x.shape[0], 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def parallel_swiglu_kernel(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    down: cute.Tensor,
    scale: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    arrivals: cute.Tensor,
):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    grid, _, _ = cute.arch.grid_dim()
    h = x.shape[1]
    intermediate = gate.shape[1]

    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[0, k])
            squares += value * value
        scale[block] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()

    for j in range(block * 32 + tid, intermediate, grid * 32):
        gate_sum = cutlass.Float32(0.0)
        up_sum = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[0, k]) * scale[block] * cutlass.Float32(norm_weight[k])
            gate_sum += value * cutlass.Float32(gate[k, j])
            up_sum += value * cutlass.Float32(up[k, j])
        hidden[0, j] = silu(gate_sum) * up_sum
    cute.arch.sync_threads()

    if tid == 0:
        cute.arch.atomic_add(arrivals.iterator, 1, sem="release", scope="gpu")
        observed = cute.arch.atomic_add(arrivals.iterator, 0, sem="acquire", scope="gpu")
        while observed < grid:
            observed = cute.arch.atomic_add(arrivals.iterator, 0, sem="acquire", scope="gpu")
    cute.arch.sync_threads()

    for j in range(block * 32 + tid, h, grid * 32):
        result = cutlass.Float32(0.0)
        for k in range(intermediate):
            result += hidden[0, k] * cutlass.Float32(down[k, j])
        out[0, j] = result + cutlass.Float32(x[0, j])


@cute.jit
def parallel_swiglu(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    down: cute.Tensor,
    scale: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    arrivals: cute.Tensor,
    stream: cuda.CUstream,
):
    parallel_swiglu_kernel(x, norm_weight, gate, up, down, scale, hidden, out, arrivals).launch(
        grid=(scale.shape[0], 1, 1), block=(32, 1, 1), cooperative=True, stream=stream
    )


@cute.kernel
def warp_parallel_swiglu_kernel(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate_transposed: cute.Tensor,
    up_transposed: cute.Tensor,
    down_transposed: cute.Tensor,
    scale: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    arrivals: cute.Tensor,
):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = tid // 32
    lane = tid % 32
    h = x.shape[1]
    intermediate = gate_transposed.shape[0]

    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[0, k])
            squares += value * value
        scale[block] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()

    for j in range(block * 4 + warp, intermediate, grid * 4):
        gate_sum = cutlass.Float32(0.0)
        up_sum = cutlass.Float32(0.0)
        for k in range(lane, h, 32):
            value = cutlass.Float32(x[0, k]) * scale[block] * cutlass.Float32(norm_weight[k])
            gate_sum += value * cutlass.Float32(gate_transposed[j, k])
            up_sum += value * cutlass.Float32(up_transposed[j, k])
        gate_sum = warp_sum(gate_sum)
        up_sum = warp_sum(up_sum)
        if lane == 0:
            hidden[0, j] = silu(gate_sum) * up_sum
    cute.arch.sync_threads()

    if tid == 0:
        cute.arch.atomic_add(arrivals.iterator, 1, sem="release", scope="gpu")
        observed = cute.arch.atomic_add(arrivals.iterator, 0, sem="acquire", scope="gpu")
        while observed < grid:
            observed = cute.arch.atomic_add(arrivals.iterator, 0, sem="acquire", scope="gpu")
    cute.arch.sync_threads()

    for j in range(block * 4 + warp, h, grid * 4):
        result = cutlass.Float32(0.0)
        for k in range(lane, intermediate, 32):
            result += hidden[0, k] * cutlass.Float32(down_transposed[j, k])
        result = warp_sum(result)
        if lane == 0:
            out[0, j] = result + cutlass.Float32(x[0, j])


@cute.jit
def warp_parallel_swiglu(
    x: cute.Tensor,
    norm_weight: cute.Tensor,
    gate_transposed: cute.Tensor,
    up_transposed: cute.Tensor,
    down_transposed: cute.Tensor,
    scale: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    arrivals: cute.Tensor,
    stream: cuda.CUstream,
):
    warp_parallel_swiglu_kernel(
        x, norm_weight, gate_transposed, up_transposed, down_transposed, scale, hidden, out, arrivals
    ).launch(grid=(scale.shape[0], 1, 1), block=(128, 1, 1), cooperative=True, stream=stream)


@cute.kernel
def cross_cta_handoff_kernel(
    x: cute.Tensor,
    scratch: cute.Tensor,
    middle: cute.Tensor,
    out: cute.Tensor,
    first_arrivals: cute.Tensor,
    second_arrivals: cute.Tensor,
    epoch: cute.Tensor,
):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    blocks, width = x.shape
    step = epoch[0]
    target = (step + 1) * blocks
    scratch[block, tid] = x[block, tid] * 2.0
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.atomic_add(first_arrivals.iterator, 1, sem="release", scope="gpu")
        observed = cute.arch.atomic_add(first_arrivals.iterator, 0, sem="acquire", scope="gpu")
        while observed < target:
            observed = cute.arch.atomic_add(first_arrivals.iterator, 0, sem="acquire", scope="gpu")
    cute.arch.sync_threads()
    middle[block, tid] = scratch[(block + 1) % blocks, tid] + 1.0
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.atomic_add(second_arrivals.iterator, 1, sem="release", scope="gpu")
        observed = cute.arch.atomic_add(second_arrivals.iterator, 0, sem="acquire", scope="gpu")
        while observed < target:
            observed = cute.arch.atomic_add(second_arrivals.iterator, 0, sem="acquire", scope="gpu")
    cute.arch.sync_threads()
    out[block, tid] = middle[(block + 1) % blocks, tid] * 3.0
    if block == 0 and tid == 0:
        epoch[0] = step + 1


@cute.jit
def cross_cta_handoff(
    x: cute.Tensor,
    scratch: cute.Tensor,
    middle: cute.Tensor,
    out: cute.Tensor,
    first_arrivals: cute.Tensor,
    second_arrivals: cute.Tensor,
    epoch: cute.Tensor,
    stream: cuda.CUstream,
):
    cross_cta_handoff_kernel(x, scratch, middle, out, first_arrivals, second_arrivals, epoch).launch(
        grid=(x.shape[0], 1, 1), block=(x.shape[1], 1, 1), cooperative=True, stream=stream
    )


def cuda_event_us(fn, *, repetitions: int = 50, trials: int = 7) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repetitions)
    return median(samples)


def launch(compiled, tensors) -> None:
    compiled(*tensors, cutlass_torch.current_stream())


def cuda_event_single_us(fn, prepare, *, trials: int = 30) -> float:
    samples = []
    for _ in range(trials):
        prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return median(samples)


def run_mlp(
    hidden_size: int,
    intermediate_size: int,
    batch: int,
    parallel_grid: int = 0,
    warp_parallel: bool = False,
    compile_baseline: bool = False,
) -> dict:
    torch.manual_seed(0)
    x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float32) * 0.1
    norm_weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32) * 0.1 + 1
    gate = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float32) * 0.1
    up = torch.randn_like(gate) * 0.1
    down = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float32) * 0.1
    scale = torch.empty(batch, device="cuda", dtype=torch.float32)
    norm = torch.empty_like(x)
    hidden = torch.empty(batch, intermediate_size, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    tensors = tuple(from_dlpack(t) for t in (x, norm_weight, gate, up, down, scale, norm, hidden, out))
    compiled = cute.compile(rmsnorm_swiglu, *tensors, cutlass_torch.current_stream())

    def reference():
        normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5) * norm_weight
        return x + (torch.nn.functional.silu(normalized @ gate) * (normalized @ up)) @ down

    launch(compiled, tensors)
    torch.cuda.synchronize()
    expected = reference()
    max_abs = (out - expected).abs().max().item()
    if not torch.allclose(out, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError(f"CuTe MLP mismatch: max_abs={max_abs}")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed_out = reference()
    graph.replay()
    if not torch.allclose(graphed_out, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("CUDA Graph reference mismatch")
    cute_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cute_graph):
        launch(compiled, tensors)
    out.fill_(float("nan"))
    cute_graph.replay()
    if not torch.allclose(out, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured CuTe MLP did not execute correctly")
    result = {
        "shape": [batch, hidden_size, intermediate_size],
        "max_abs": max_abs,
        "cute_us": cuda_event_us(lambda: launch(compiled, tensors)),
        "cute_cudagraph_us": cuda_event_us(cute_graph.replay),
        "torch_eager_us": cuda_event_us(reference),
        "torch_cudagraph_us": cuda_event_us(graph.replay),
    }
    if compile_baseline:
        compiled_reference = torch.compile(reference, mode="max-autotune", fullgraph=True)
        optimized_out = compiled_reference()
        torch.cuda.synchronize()
        if not torch.allclose(optimized_out, expected, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("torch.compile reference mismatch")
        result["torch_compile_max_autotune_us"] = cuda_event_us(compiled_reference)
    if parallel_grid:
        parallel_scale = torch.empty(parallel_grid, device="cuda", dtype=torch.float32)
        parallel_hidden = torch.empty_like(hidden)
        parallel_out = torch.empty_like(out)
        arrivals = torch.zeros(1, device="cuda", dtype=torch.int32)
        parallel_tensors = tuple(from_dlpack(t) for t in (
            x, norm_weight, gate, up, down, parallel_scale, parallel_hidden, parallel_out, arrivals
        ))
        parallel_compiled = cute.compile(parallel_swiglu, *parallel_tensors, cutlass_torch.current_stream())

        def parallel_call():
            launch(parallel_compiled, parallel_tensors)

        arrivals.zero_()
        parallel_call()
        torch.cuda.synchronize()
        parallel_error = (parallel_out - expected).abs().max().item()
        if not torch.allclose(parallel_out, expected, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError(f"parallel CuTe MLP mismatch: max_abs={parallel_error}")
        result["parallel_grid"] = parallel_grid
        result["parallel_max_abs"] = parallel_error
        result["parallel_kernel_us"] = cuda_event_single_us(parallel_call, arrivals.zero_)
        result["parallel_with_reset_us"] = cuda_event_single_us(
            lambda: (arrivals.zero_(), parallel_call()), lambda: None
        )
        parallel_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(parallel_graph):
            arrivals.zero_()
            parallel_call()
        parallel_out.fill_(float("nan"))
        parallel_graph.replay()
        if not torch.allclose(parallel_out, expected, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured parallel CuTe MLP did not execute correctly")
        result["parallel_cudagraph_with_reset_us"] = cuda_event_us(parallel_graph.replay)
    if warp_parallel:
        warp_scale = torch.empty(parallel_grid, device="cuda", dtype=torch.float32)
        warp_hidden = torch.empty_like(hidden)
        warp_out = torch.empty_like(out)
        warp_arrivals = torch.zeros(1, device="cuda", dtype=torch.int32)
        warp_tensors = tuple(from_dlpack(t) for t in (
            x, norm_weight, gate.T.contiguous(), up.T.contiguous(), down.T.contiguous(),
            warp_scale, warp_hidden, warp_out, warp_arrivals
        ))
        warp_compiled = cute.compile(warp_parallel_swiglu, *warp_tensors, cutlass_torch.current_stream())

        def warp_call():
            launch(warp_compiled, warp_tensors)

        warp_call()
        torch.cuda.synchronize()
        warp_error = (warp_out - expected).abs().max().item()
        if not torch.allclose(warp_out, expected, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError(f"warp CuTe MLP mismatch: max_abs={warp_error}")
        result["warp_max_abs"] = warp_error
        result["warp_kernel_us"] = cuda_event_single_us(warp_call, warp_arrivals.zero_)
        result["warp_with_reset_us"] = cuda_event_single_us(
            lambda: (warp_arrivals.zero_(), warp_call()), lambda: None
        )
        warp_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warp_graph):
            warp_arrivals.zero_()
            warp_call()
        warp_out.fill_(float("nan"))
        warp_graph.replay()
        if not torch.allclose(warp_out, expected, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured warp CuTe MLP did not execute correctly")
        result["warp_cudagraph_with_reset_us"] = cuda_event_us(warp_graph.replay)
    return result


def run_cross_cta(blocks: int = 8, width: int = 128) -> dict:
    x = torch.arange(blocks * width, device="cuda", dtype=torch.float32).reshape(blocks, width)
    scratch = torch.empty_like(x)
    middle = torch.empty_like(x)
    out = torch.empty_like(x)
    first_arrivals = torch.zeros(1, device="cuda", dtype=torch.int32)
    second_arrivals = torch.zeros(1, device="cuda", dtype=torch.int32)
    epoch = torch.zeros(1, device="cuda", dtype=torch.int32)
    tensors = tuple(from_dlpack(t) for t in (x, scratch, middle, out, first_arrivals, second_arrivals, epoch))
    compiled = cute.compile(cross_cta_handoff, *tensors, cutlass_torch.current_stream())
    launch(compiled, tensors)
    torch.cuda.synchronize()
    expected = (x.roll(-2, dims=0) * 2.0 + 1.0) * 3.0
    if not torch.equal(out, expected) or first_arrivals.item() != blocks or second_arrivals.item() != blocks or epoch.item() != 1:
        raise AssertionError("cross-CTA handoff failed")

    def reset():
        first_arrivals.zero_()
        second_arrivals.zero_()
        epoch.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(compiled, tensors)
    out.fill_(float("nan"))
    graph.replay()
    if not torch.equal(out, expected) or epoch.item() != 2:
        raise AssertionError("captured cross-CTA handoff did not execute correctly")

    result = {
        "blocks": blocks,
        "width": width,
        "barriers": 2,
        "correct": True,
        "kernel_us": cuda_event_single_us(lambda: launch(compiled, tensors), reset),
        "with_reset_us": cuda_event_single_us(
            lambda: (reset(), launch(compiled, tensors)), lambda: None
        ),
        "cudagraph_single_kernel_us": cuda_event_us(graph.replay),
    }
    if not torch.equal(out, expected) or first_arrivals.item() != blocks * epoch.item() or second_arrivals.item() != blocks * epoch.item():
        raise AssertionError("repeated captured cross-CTA handoff failed")
    reset()
    launch(compiled, tensors)
    launch(compiled, tensors)
    if not torch.equal(out, expected) or first_arrivals.item() != 2 * blocks or second_arrivals.item() != 2 * blocks or epoch.item() != 2:
        raise AssertionError("repeated cross-CTA handoff without reset failed")
    result["epoch_repeated_correct"] = True
    result["epoch_kernel_us"] = cuda_event_single_us(lambda: launch(compiled, tensors), lambda: None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="include SmolLM2-360M MLP dimensions")
    parser.add_argument("--compile-baseline", action="store_true", help="compare against torch.compile max-autotune")
    args = parser.parse_args()
    results = {
        "date_utc": date.today().isoformat(),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "dtype": "float32",
        "timing": "median CUDA event elapsed time; 7x50 for simple calls, 30 single calls for cooperative grids",
    }
    results["cross_cta"] = run_cross_cta()
    results["mlp"] = [run_mlp(128, 256, 1), run_mlp(128, 256, 8)]
    if args.large:
        results["mlp"].append(run_mlp(960, 2560, 1, parallel_grid=16))
        results["mlp"].append(run_mlp(
            960, 2560, 1, parallel_grid=60, warp_parallel=True, compile_baseline=args.compile_baseline
        ))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
