#!/usr/bin/env python3
"""Benchmark harness: megabake vs torch.compile side-by-side comparison."""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

import torch
import torch.profiler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from models import get_workloads, BenchmarkWorkload


@dataclass
class BenchmarkResult:
    backend: str
    model_name: str
    compile_time_ms: float
    median_latency_us: float
    kernel_count: int
    peak_memory_mb: float
    max_abs_diff: float
    error: str | None = None


@dataclass
class Backend:
    name: str
    compile_fn: Callable
    run_fn: Callable


# ---------------------------------------------------------------------------
# Megabake backend
# ---------------------------------------------------------------------------

def _mb_compile(model, inputs):
    import megabake
    example = inputs[0] if len(inputs) == 1 else tuple(inputs)
    return megabake.compile(model, example)


def _mb_run(compiled, model, *inputs):
    import megabake
    return megabake.run(compiled, model, *inputs)


MEGABAKE = Backend("megabake", _mb_compile, _mb_run)


# ---------------------------------------------------------------------------
# torch.compile backend
# ---------------------------------------------------------------------------

def _tc_compile(model, inputs):
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        with torch.no_grad():
            compiled(*inputs)
    except Exception:
        torch._dynamo.reset()
        compiled = torch.compile(model, backend="inductor", fullgraph=False)
        with torch.no_grad():
            compiled(*inputs)
    return compiled


def _tc_run(compiled, _model, *inputs):
    with torch.no_grad():
        out = compiled(*inputs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


TORCH_COMPILE = Backend("torch.compile", _tc_compile, _tc_run)


# ---------------------------------------------------------------------------
# Measurement functions
# ---------------------------------------------------------------------------

def measure_compile_time(backend: Backend, model, inputs) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled = backend.compile_fn(model, inputs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return compiled, (t1 - t0) * 1000


def measure_latency(
    backend: Backend, compiled, model, inputs, warmup: int, iters: int,
) -> float:
    for _ in range(warmup):
        backend.run_fn(compiled, model, *inputs)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        backend.run_fn(compiled, model, *inputs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) * 1000)  # ms -> us
    return statistics.median(timings)


def count_kernels(backend: Backend, compiled, model, inputs) -> int:
    backend.run_fn(compiled, model, *inputs)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        backend.run_fn(compiled, model, *inputs)
        torch.cuda.synchronize()

    count = 0
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = evt.name
        if name.startswith("Memcpy") or name.startswith("Memset"):
            continue
        if "FillFunctor" in name or "ZeroFunctor" in name:
            continue
        if "vectorized_elementwise_kernel" in name:
            continue
        count += 1
    return count


def measure_peak_memory(backend: Backend, compiled, model, inputs) -> float:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    backend.run_fn(compiled, model, *inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def measure_correctness(backend: Backend, compiled, model, inputs) -> float:
    with torch.no_grad():
        ref = model(*inputs)
    if isinstance(ref, (tuple, list)):
        ref = ref[0]
    result = backend.run_fn(compiled, model, *inputs)
    return (result.float() - ref.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single(
    wl: BenchmarkWorkload, backend: Backend, warmup: int, iters: int,
) -> BenchmarkResult:
    model = wl.model_fn()
    inputs = wl.input_fn()

    try:
        compiled, compile_ms = measure_compile_time(backend, model, inputs)
        latency_us = measure_latency(backend, compiled, model, inputs, warmup, iters)
        kernels = count_kernels(backend, compiled, model, inputs)
        memory_mb = measure_peak_memory(backend, compiled, model, inputs)

        max_diff = measure_correctness(backend, compiled, model, inputs)

        return BenchmarkResult(
            backend=backend.name,
            model_name=wl.name,
            compile_time_ms=compile_ms,
            median_latency_us=latency_us,
            kernel_count=kernels,
            peak_memory_mb=memory_mb,
            max_abs_diff=max_diff,
        )
    except Exception as e:
        return BenchmarkResult(
            backend=backend.name,
            model_name=wl.name,
            compile_time_ms=-1, median_latency_us=-1,
            kernel_count=-1, peak_memory_mb=-1, max_abs_diff=-1,
            error=str(e),
        )
    finally:
        torch.cuda.empty_cache()


def run_benchmarks(
    workloads: list[BenchmarkWorkload],
    backends: list[Backend],
    warmup: int,
    iters: int,
) -> list[BenchmarkResult]:
    results = []
    for wl in workloads:
        print(f"\n{'='*70}")
        print(f"  {wl.name}")
        print(f"{'='*70}")
        for backend in backends:
            print(f"  [{backend.name}] ", end="", flush=True)
            r = run_single(wl, backend, warmup, iters)
            if r.error:
                print(f"FAIL: {r.error}")
            else:
                print(
                    f"compile={r.compile_time_ms:.0f}ms  "
                    f"latency={r.median_latency_us:.1f}us  "
                    f"kernels={r.kernel_count}  "
                    f"mem={r.peak_memory_mb:.1f}MB  "
                    f"diff={r.max_abs_diff:.6f}"
                )
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_table(results: list[BenchmarkResult]) -> None:
    hdr = (
        f"{'Model':<20s} {'Backend':<16s} {'Compile(ms)':>11s} "
        f"{'Latency(us)':>11s} {'Kernels':>8s} {'Mem(MB)':>8s} "
        f"{'MaxDiff':>10s}"
    )
    sep = "-" * len(hdr)

    print(f"\n{sep}")
    print(hdr)
    print(sep)

    by_model: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_model.setdefault(r.model_name, []).append(r)

    for model_name, runs in by_model.items():
        for i, r in enumerate(runs):
            name_col = model_name if i == 0 else ""
            if r.error:
                print(f"{name_col:<20s} {r.backend:<16s} {'FAIL':>11s}  {r.error}")
            else:
                print(
                    f"{name_col:<20s} {r.backend:<16s} "
                    f"{r.compile_time_ms:>11.1f} "
                    f"{r.median_latency_us:>11.1f} "
                    f"{r.kernel_count:>8d} "
                    f"{r.peak_memory_mb:>8.1f} "
                    f"{r.max_abs_diff:>10.6f}"
                )

        mb = [r for r in runs if r.backend == "megabake" and r.error is None]
        tc = [r for r in runs if r.backend == "torch.compile" and r.error is None]
        if mb and tc:
            speedup = tc[0].median_latency_us / mb[0].median_latency_us
            mb_k = mb[0].kernel_count
            tc_k = tc[0].kernel_count
            print(f"{'':>37s} Speedup: {speedup:.2f}x  ({mb_k} kernel vs {tc_k} kernels)")
        print(sep)


def write_json(results: list[BenchmarkResult], path: str, warmup: int, iters: int) -> None:
    props = torch.cuda.get_device_properties(0)
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "warmup_iters": warmup,
        "bench_iters": iters,
    }
    data = {"metadata": meta, "results": [asdict(r) for r in results]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Megabake vs torch.compile benchmark")
    parser.add_argument("--tag", action="append", help="Filter workloads by tag")
    parser.add_argument("--models", type=str, help="Comma-separated model names")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=str, help="JSON output file path")
    parser.add_argument(
        "--backend", type=str, choices=["megabake", "torch.compile"],
        help="Run only one backend",
    )
    args = parser.parse_args()

    names = args.models.split(",") if args.models else None
    workloads = get_workloads(tags=args.tag, names=names)
    if not workloads:
        print("No workloads matched filters.")
        return

    backends = [MEGABAKE, TORCH_COMPILE]
    if args.backend == "megabake":
        backends = [MEGABAKE]
    elif args.backend == "torch.compile":
        backends = [TORCH_COMPILE]

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SMs: {props.multi_processor_count}  CUDA: {torch.version.cuda}")
    print(f"Torch: {torch.__version__}")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")
    print(f"Workloads: {[w.name for w in workloads]}")
    print(f"Backends: {[b.name for b in backends]}")

    results = run_benchmarks(workloads, backends, args.warmup, args.iters)
    print_table(results)

    if args.output:
        write_json(results, args.output, args.warmup, args.iters)


if __name__ == "__main__":
    main()
