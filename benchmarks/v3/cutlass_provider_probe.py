"""Standalone V3R-008 CUTLASS device-callable body probe."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CUTLASS_ROOT = ROOT / ".venv/lib64/python3.13/site-packages/cutlass_library/source"
sys.path.insert(0, str(ROOT / "src"))


class Resources(ctypes.Structure):
    _fields_ = [("device_sms", ctypes.c_int), ("cooperative", ctypes.c_int),
                ("active_per_sm", ctypes.c_int), ("resident_ctas", ctypes.c_int),
                ("registers", ctypes.c_int), ("shared_bytes", ctypes.c_int),
                ("local_bytes", ctypes.c_int)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.cutlass_probe_profile.argtypes = [ctypes.POINTER(ctypes.c_int)] * 7
    lib.cutlass_probe_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_size_t,
    ]
    return lib


def _check(code: int, label: str) -> None:
    if code:
        raise RuntimeError(f"{label} returned CUDA status {code}")


def _launch(lib: Any, x: Any, weight: Any, bias: Any, output: Any,
            alpha: float, beta: float, owner: bool, stream: Any) -> None:
    _check(lib.cutlass_probe_launch(
        x.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None,
        output.data_ptr(), x.shape[0], weight.shape[0], x.shape[1],
        alpha, beta, int(owner), stream.cuda_stream,
    ), "CUTLASS body launch")


def _measure(torch_module: Any, lib: Any, x: Any, weight: Any, output: Any,
             *, samples: int, owner: bool) -> dict[str, Any]:
    stream = torch_module.cuda.Stream(device=torch_module.cuda.current_device())
    for _ in range(5):
        _launch(lib, x, weight, None, output, 1.0, 0.0, owner, stream)
    stream.synchronize()
    gpu, wall = [], []
    for _ in range(samples):
        begin = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch_module.cuda.stream(stream):
            begin.record(stream)
            _launch(lib, x, weight, None, output, 1.0, 0.0, owner, stream)
            end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1e6)
        gpu.append(begin.elapsed_time(end) * 1000.0)
    return {
        "placement": "cooperative_owner_body" if owner else "standalone_body",
        "samples": samples,
        "raw_gpu_event_us": gpu,
        "median_gpu_event_us": statistics.median(gpu),
        "raw_wrapper_wall_us": wall,
        "median_wrapper_wall_us": statistics.median(wall),
        "timing_note": "Direct launch; GPU event excludes host wrapper time, wall includes it.",
    }


def _measure_torch_addmm(torch_module: Any, x: Any, weight: Any,
                         *, samples: int) -> dict[str, Any]:
    stream = torch_module.cuda.Stream(device=torch_module.cuda.current_device())
    bias = torch_module.zeros((x.shape[0], weight.shape[0]),
                              device=x.device, dtype=x.dtype)
    output = torch_module.empty_like(bias)
    fn = lambda: torch_module.addmm(bias, x, weight.t(), out=output)
    for _ in range(5):
        with torch_module.cuda.stream(stream):
            fn()
    stream.synchronize()
    gpu, wall = [], []
    for _ in range(samples):
        begin = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch_module.cuda.stream(stream):
            begin.record(stream)
            fn()
            end.record(stream)
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1e6)
        gpu.append(begin.elapsed_time(end) * 1000.0)
    return {
        "path": "same-buffer eager torch.addmm with preallocated output",
        "samples": samples,
        "raw_gpu_event_us": gpu,
        "median_gpu_event_us": statistics.median(gpu),
        "raw_complete_callable_wall_us": wall,
        "median_complete_callable_wall_us": statistics.median(wall),
    }


def run(library_path: Path, output_path: Path, samples: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUTLASS provider probe requires an available CUDA GPU")
    lib = _load(library_path)
    resource_values = [ctypes.c_int() for _ in range(7)]
    _check(lib.cutlass_probe_profile(*[ctypes.byref(value) for value in resource_values]),
           "CUTLASS kernel resource query")
    names = ("device_sms", "cooperative_launch", "active_ctas_per_sm",
             "resident_ctas", "registers_per_thread", "static_shared_bytes",
             "local_bytes")
    resources = dict(zip(names, (int(value.value) for value in resource_values)))
    cases = []
    for m, n, k in ((1, 192, 576), (1, 576, 576), (1, 576, 1536),
                    (1, 1536, 576), (1, 49152, 576)):
        torch.manual_seed(8800 + n)
        x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.02
        weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
        # Keep sixteen half elements before/after output; 16-byte aligned output.
        storage = torch.full((m * n + 32,), -123.0, device="cuda", dtype=torch.float16)
        output = storage[16:16 + m * n].view(m, n)
        expected = torch.addmm(torch.zeros((m, n), device="cuda", dtype=torch.float16),
                               x, weight.t())
        tiles = ((m + 127) // 128) * ((n + 63) // 64)
        owner = None
        if tiles <= resources["resident_ctas"] and resources["cooperative_launch"]:
            owner = _measure(torch, lib, x, weight, output, samples=samples, owner=True)
        standalone = _measure(torch, lib, x, weight, output,
                              samples=samples, owner=False)
        torch_control = _measure_torch_addmm(torch, x, weight, samples=samples)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
        assert bool(torch.all(storage[:16] == -123.0))
        assert bool(torch.all(storage[16 + m * n:] == -123.0))
        max_abs = float((output.float() - expected.float()).abs().max().item())
        cases.append({
            "shape": [m, n, k], "dtype": "float16", "weight_layout": "row_major[N,K]",
            "output_tiles": tiles, "resources": resources,
            "standalone": standalone, "owner_entry": owner,
            "torch_addmm_control": torch_control,
            "owner_rejection": (None if owner is not None else {
                "reason": "tile grid exceeds measured resident cooperative grid or device lacks cooperative launch",
                "required_grid_ctas": tiles,
                "resident_ctas": resources["resident_ctas"],
            }),
            "max_abs_error": max_abs, "active_store_canaries_intact": True,
        })
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    report = {
        "schema_version": 1,
        "task_id": "V3R-008",
        "provider": "CUTLASS C++ device::Gemm body invoked from a custom __global__ wrapper",
        "cutlass_distribution": importlib.metadata.version("nvidia-cutlass"),
        "cutlass_include_root": str(CUTLASS_ROOT / "include"),
        "cutlass_headers_used": ["cutlass/gemm/device/gemm.h", "cutlass/gemm/kernel/gemm.h"],
        "toolchain": nvcc.stdout.splitlines()[-1].strip(),
        "target": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "visible_sms": int(torch.cuda.get_device_properties(0).multi_processor_count),
            "compiled_target": "sm_90a",
            "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        },
        "device_call_evidence": {
            "cutlass_arch_tag": "cutlass::arch::Sm80; the CUTLASS 3.8 kernel's tensor-op architecture tag",
            "compiled_target": "sm_90a",
            "host_descriptor_prepared": True,
            "kernel_params_passed_by_value_with_grid_constant": True,
            "custom_global_calls_kernel_operator": True,
            "host_launch_inside_body": False,
            "grid_shape": "logical ceil(M/128) x ceil(N/64) x 1; identity swizzle maps logical axes to CUDA x/y",
            "divisibility_and_depth": "tested K=576/1536, both divisible by the 32-element mainloop K tile; tested N values are multiples of 64 and M=1 uses the kernel's masked edge tile; no K-tail or N-tail candidate was tested",
            "accumulator_and_warp_specialization": "float accumulator with CUTLASS LinearCombination epilogue; one 128-thread, multistage kernel configuration; no warp-specialization tradeoff was tested",
            "cross_sm_legality": "independent CTAs with no grid-wide barrier or cross-SM data exchange; cooperative owner launches were measured only when the actual output tile grid was at most the queried resident CTA count",
            "reset_tile": "not present in CUTLASS 3.8 device::Gemm kernel API; no persistent CTA tile loop was adapted",
            "get_block_dim": "not present in this CUTLASS 3.8 API; block shape is fixed at the GEMM kernel's 128 threads",
            "warp_specialization": "CUTLASS 3.8 multistage kernel is used; this probe does not expose a warp-specialized device-call contract",
        },
        "compatibility_experiments": {
            "cuBLASDx": {
                "status": "rejected_candidate",
                "toolkit": "nvcc 12.8.93",
                "distribution": "no cublasdx.hpp found in /usr/local/cuda/include, /usr/include, or the installed CUTLASS include tree",
                "compile_probe": "nvcc -std=c++17 -arch=sm_90a -c ART/tasks/V3R-008/cublasdx_probe.cu -o /tmp/mb3_cublasdx_probe.o",
                "result": "rejected: fatal error: cublasdx.hpp: No such file or directory",
                "environment_changed": False,
            },
            "cutlass": {
                "status": "measured",
                "distribution": "nvidia-cutlass 3.8.0.0",
                "version_lane": "installed CUTLASS 3.8 headers with the selected nvcc 12.8.93 toolchain",
            },
        },
        "correctness": "FP16, beta=0, exact contiguous hot shapes, canary-guarded output",
        "cases": cases,
        "artifacts": {
            "source": "src/cuda/v3/cutlass_38_body_probe.cu",
            "source_sha256": _sha(ROOT / "src/cuda/v3/cutlass_38_body_probe.cu"),
            "library": str(library_path), "library_sha256": _sha(library_path),
            "library_bytes": library_path.stat().st_size,
        },
        "limitations": [
            "CUTLASS 3.8 tested only; no silent environment update to CUDA 13+/CUTLASS 4.4.1+.",
            "Cooperative owner placement is rejected when all CUTLASS output CTAs do not fit measured residency; no virtual tile remapping or composition with another body is proven.",
            "No BF16, split-K, layout conversion, tail alignment, or full-step call was measured.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    report = run(args.library, args.output, args.samples)
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
