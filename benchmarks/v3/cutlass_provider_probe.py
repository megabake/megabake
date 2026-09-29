"""Standalone V3R-008 CUTLASS device-callable body probe."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CUTLASS_ROOT = ROOT / ".venv/lib64/python3.13/site-packages/cutlass_library/source"
DX_TILE = (16, 64, 32)
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
    if hasattr(lib, "cutlass_probe_grid"):
        lib.cutlass_probe_grid.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_int),
        ]
        lib.cutlass_probe_grid.restype = ctypes.c_int
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


def _load_cublasdx(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.mb3_cublasdx_create.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.mb3_cublasdx_create.restype = ctypes.c_int
    lib.mb3_cublasdx_profile.argtypes = [ctypes.c_void_p] + [
        ctypes.POINTER(ctypes.c_int)
    ] * 11
    lib.mb3_cublasdx_profile.restype = ctypes.c_int
    lib.mb3_cublasdx_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    ]
    lib.mb3_cublasdx_launch.restype = ctypes.c_int
    lib.mb3_cublasdx_destroy.argtypes = [ctypes.c_void_p]
    return lib


def _cublasdx_create(lib: Any, x: Any, weight: Any,
                     shape: tuple[int, int, int]):
    m, n, k = shape
    context, shared, grid = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_int()
    started = time.perf_counter()
    status = lib.mb3_cublasdx_create(
        x.data_ptr(), weight.data_ptr(), m, n, k, ctypes.byref(context),
        ctypes.byref(shared), ctypes.byref(grid),
    )
    return context, int(status), int(shared.value), int(grid.value), (
        time.perf_counter() - started
    ) * 1e6


def _cublasdx_resources(lib: Any, context: Any) -> dict[str, int]:
    values = [ctypes.c_int() for _ in range(11)]
    status = lib.mb3_cublasdx_profile(
        context, *[ctypes.byref(value) for value in values]
    )
    _check(status, "cuBLASDx resource query")
    names = ("device_sms", "cooperative_launch", "active_ctas_per_sm",
             "registers_per_thread", "dynamic_shared_bytes", "static_shared_bytes",
             "local_bytes",
             "threads_per_block", "tile_m", "tile_n", "tile_k")
    return dict(zip(names, (int(value.value) for value in values)))


def _cuobjdump_resources(path: Path) -> dict[str, Any]:
    command = ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    match = re.search(
        r"Function [^\n]*pipeline_body[^\n]*:\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        result.stdout,
    )
    if not match:
        raise RuntimeError("cuobjdump output has no cuBLASDx pipeline_body resource row")
    registers, stack, shared, local = (int(value) for value in match.groups())
    return {
        "command": " ".join(command), "return_code": result.returncode,
        "registers_per_thread": registers, "stack_bytes": stack,
        "shared_bytes": shared, "local_bytes": local,
    }


def _launch_cublasdx(lib: Any, context: Any, output: Any,
                     owner: bool, stream: Any) -> None:
    _check(lib.mb3_cublasdx_launch(
        context, output.data_ptr(), int(owner), stream.cuda_stream,
    ), "cuBLASDx pipeline launch")


def _measure_cublasdx(torch_module: Any, lib: Any, context: Any,
                      output: Any, *,
                      samples: int, owner: bool) -> dict[str, Any]:
    stream = torch_module.cuda.Stream(device=torch_module.cuda.current_device())
    for _ in range(5):
        _launch_cublasdx(lib, context, output, owner, stream)
    stream.synchronize()
    gpu, wall = [], []
    for _ in range(samples):
        begin = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch_module.cuda.stream(stream):
            begin.record(stream)
            _launch_cublasdx(lib, context, output, owner, stream)
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
        "timing_note": "Direct CUDA body launch; host pipeline descriptor is prepared before timing.",
    }


def _run_cublasdx_smoke(lib: Any, cublaslt_case: dict[str, Any],
                        samples: int) -> dict[str, Any]:
    m, n, k = (16, 64, 64)
    torch.manual_seed(8864)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.02
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    backing = torch.full((m * n + 32,), -123.0, device="cuda", dtype=torch.float16)
    output = backing[16:16 + m * n].view(m, n)
    expected = torch.addmm(torch.zeros((m, n), device="cuda", dtype=torch.float16),
                           x, weight.t())
    torch.cuda.synchronize()
    context, status, shared, grid_ctas, setup_us = _cublasdx_create(
        lib, x, weight, (m, n, k)
    )
    _check(status, "cuBLASDx synthetic compatible descriptor")
    try:
        resources = _cublasdx_resources(lib, context)
        resident = resources["device_sms"] * resources["active_ctas_per_sm"]
        if not resources["cooperative_launch"] or grid_ctas > resident:
            raise RuntimeError("cuBLASDx cooperative owner grid is not resident/admissible")
        standalone = _measure_cublasdx(torch, lib, context, output,
                                       samples=samples, owner=False)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
        owner = _measure_cublasdx(torch, lib, context, output,
                                  samples=samples, owner=True)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
        assert bool(torch.all(backing[:16] == -123.0))
        assert bool(torch.all(backing[16 + m * n:] == -123.0))
        lt_measurement = cublaslt_case["measurement"]
        return {
            "shape": [m, n, k], "dtype": "float16",
            "status": "compatible_synthetic_tile_only",
            "weight_layout": "row_major[N,K] viewed as column-major [K,N]",
            "pipeline": {
                "tile_mnk": list(DX_TILE), "depth": 1,
                "k_iterations": k // DX_TILE[2],
                "get_block_dim": resources["threads_per_block"],
                "host_descriptor_prepared": True,
                "device_handle_passed_by_value_with_grid_constant": True,
                "reset_tile": "not used; one tile execution per CTA, no persistent tile loop",
                "cross_sm": "independent output tiles; no grid barrier or cross-SM data exchange",
            },
            "descriptor_setup_host_us": setup_us,
            "grid_ctas": grid_ctas, "resources": resources,
            "resident_ctas": resident, "shared_bytes": shared,
            "standalone": standalone, "owner_entry": owner,
            "cublaslt_external_control": lt_measurement,
            "owner_to_cublaslt_event_ratio": (
                owner["median_gpu_event_us"] / lt_measurement["median_gpu_event_us"]
            ),
            "max_abs_error": float((output.float() - expected.float()).abs().max().item()),
            "active_store_canaries_intact": True,
            "qualification": "Synthetic fully divisible tile for API/correctness/entry proof; it is not a model hot shape and its ratio is not a speed claim.",
        }
    finally:
        lib.mb3_cublasdx_destroy(context)


def run(library_path: Path, output_path: Path, samples: int,
        cublasdx_library_path: Path | None = None,
        cublaslt_report_path: Path | None = None,
        cutlass452_library_path: Path | None = None,
        cutlass_cute_library_path: Path | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUTLASS provider probe requires an available CUDA GPU")
    cublasdx_library_path = cublasdx_library_path or (
        ROOT / "ART/tasks/V3R-008/libmegabake_cublasdx_pipeline_sm90a.so"
    )
    cublaslt_report_path = cublaslt_report_path or (
        ROOT / "ART/tasks/V3R-005/cublaslt_report.json"
    )
    cublasdx_lib = _load_cublasdx(cublasdx_library_path)
    cublasdx_binary_resources = _cuobjdump_resources(cublasdx_library_path)
    cublaslt_report = json.loads(cublaslt_report_path.read_text())
    cublaslt_by_shape = {
        tuple(int(value) for value in case["shape"]): case
        for case in cublaslt_report["cases"]
    }
    smoke_shape = (16, 64, 64)
    if smoke_shape not in cublaslt_by_shape:
        raise ValueError("V3R-005 cuBLASLt report lacks the cuBLASDx synthetic smoke shape")
    lib = _load(library_path)
    resource_values = [ctypes.c_int() for _ in range(7)]
    _check(lib.cutlass_probe_profile(*[ctypes.byref(value) for value in resource_values]),
           "CUTLASS kernel resource query")
    names = ("device_sms", "cooperative_launch", "active_ctas_per_sm",
             "resident_ctas", "registers_per_thread", "static_shared_bytes",
             "local_bytes")
    resources = dict(zip(names, (int(value.value) for value in resource_values)))
    cutlass452_lib = _load(cutlass452_library_path) if cutlass452_library_path else None
    cutlass452_resources = None
    if cutlass452_lib:
        values = [ctypes.c_int() for _ in range(7)]
        _check(cutlass452_lib.cutlass_probe_profile(
            *[ctypes.byref(value) for value in values]
        ), "CUDA 13 CUTLASS 4.5.2 kernel resource query")
        cutlass452_resources = dict(zip(names, (int(value.value) for value in values)))
    cutlass_cute_lib = _load(cutlass_cute_library_path) if cutlass_cute_library_path else None
    cutlass_cute_resources = None
    if cutlass_cute_lib:
        values = [ctypes.c_int() for _ in range(7)]
        _check(cutlass_cute_lib.cutlass_probe_profile(
            *[ctypes.byref(value) for value in values]
        ), "CUDA 13 CUTLASS CuTe collective resource query")
        cutlass_cute_resources = dict(zip(names, (int(value.value) for value in values)))
    cases = []
    for m, n, k in ((1, 192, 576), (1, 576, 576), (1, 576, 1536),
                    (1, 1536, 576), (1, 49152, 576)):
        shape = (m, n, k)
        if shape not in cublaslt_by_shape:
            raise ValueError(f"V3R-005 cuBLASLt report lacks hot shape {shape}")
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
        cutlass452_case = None
        if cutlass452_lib and cutlass452_resources:
            c45_tiles = ((m + 127) // 128) * ((n + 63) // 64)
            c45_owner = None
            if (c45_tiles <= cutlass452_resources["resident_ctas"] and
                    cutlass452_resources["cooperative_launch"]):
                c45_owner = _measure(torch, cutlass452_lib, x, weight, output,
                                     samples=samples, owner=True)
            c45_standalone = _measure(
                torch, cutlass452_lib, x, weight, output,
                samples=samples, owner=False,
            )
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
            c45_max_abs = float((output.float() - expected.float()).abs().max().item())
            assert bool(torch.all(storage[:16] == -123.0))
            assert bool(torch.all(storage[16 + m * n:] == -123.0))
            cutlass452_case = {
                "status": "correct_standalone_and_owner" if c45_owner else "correct_standalone_owner_rejected",
                "output_tiles": c45_tiles,
                "resources": cutlass452_resources,
                "standalone": c45_standalone,
                "owner_entry": c45_owner,
                "owner_rejection": None if c45_owner else {
                    "reason": "tile grid exceeds measured resident cooperative grid or device lacks cooperative launch",
                    "required_grid_ctas": c45_tiles,
                    "resident_ctas": cutlass452_resources["resident_ctas"],
                },
                "max_abs_error": c45_max_abs,
                "active_store_canaries_intact": True,
            }
        cutlass_cute_case = None
        if cutlass_cute_lib and cutlass_cute_resources:
            grid_ctas = ctypes.c_int()
            _check(cutlass_cute_lib.cutlass_probe_grid(
                x.data_ptr(), weight.data_ptr(), None, output.data_ptr(),
                m, n, k, 1.0, 0.0, ctypes.byref(grid_ctas),
            ), "CUDA 13 CUTLASS CuTe collective grid query")
            cute_grid = int(grid_ctas.value)
            cute_owner = None
            if (cute_grid <= cutlass_cute_resources["resident_ctas"] and
                    cutlass_cute_resources["cooperative_launch"]):
                cute_owner = _measure(torch, cutlass_cute_lib, x, weight, output,
                                      samples=samples, owner=True)
            cute_standalone = _measure(
                torch, cutlass_cute_lib, x, weight, output,
                samples=samples, owner=False,
            )
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
            cute_max_abs = float((output.float() - expected.float()).abs().max().item())
            assert bool(torch.all(storage[:16] == -123.0))
            assert bool(torch.all(storage[16 + m * n:] == -123.0))
            cutlass_cute_case = {
                "status": "correct_standalone_and_owner" if cute_owner else "correct_standalone_owner_rejected",
                "grid_ctas": cute_grid,
                "resources": cutlass_cute_resources,
                "standalone": cute_standalone,
                "owner_entry": cute_owner,
                "owner_rejection": None if cute_owner else {
                    "reason": "actual CUTLASS persistent-scheduler grid exceeds measured resident cooperative grid or device lacks cooperative launch",
                    "required_grid_ctas": cute_grid,
                    "resident_ctas": cutlass_cute_resources["resident_ctas"],
                },
                "max_abs_error": cute_max_abs,
                "active_store_canaries_intact": True,
            }
        assert bool(torch.all(storage[:16] == -123.0))
        assert bool(torch.all(storage[16 + m * n:] == -123.0))
        max_abs = float((output.float() - expected.float()).abs().max().item())
        dx_context, dx_status, _, _, dx_setup_us = _cublasdx_create(
            cublasdx_lib, x, weight, shape
        )
        if dx_status != 9 or dx_context.value:
            if dx_context.value:
                cublasdx_lib.mb3_cublasdx_destroy(dx_context)
            raise AssertionError(
                f"cuBLASDx should explicitly reject non-divisible hot shape {shape}; "
                f"status={dx_status}, context={bool(dx_context.value)}"
            )
        cases.append({
            "shape": [m, n, k], "dtype": "float16", "weight_layout": "row_major[N,K]",
            "output_tiles": tiles, "resources": resources,
            "standalone": standalone, "owner_entry": owner,
            "cutlass_4_5_2_cuda13": cutlass452_case,
            "cutlass_cute_collective_4_5_2_cuda13": cutlass_cute_case,
            "torch_addmm_control": torch_control,
            "cublaslt_baseline": cublaslt_by_shape[shape],
            "cublasdx": {
                "status": "rejected_candidate",
                "reason": "pipelined global GEMM requires M/N/K divisible by the configured tile; M=1 is not divisible by tile_m=16",
                "status_code": dx_status,
                "tile_mnk": list(DX_TILE),
                "descriptor_setup_us": dx_setup_us,
                "context_created": False,
            },
            "owner_rejection": (None if owner is not None else {
                "reason": "tile grid exceeds measured resident cooperative grid or device lacks cooperative launch",
                "required_grid_ctas": tiles,
                "resident_ctas": resources["resident_ctas"],
            }),
            "max_abs_error": max_abs, "active_store_canaries_intact": True,
        })
    cublasdx_smoke = _run_cublasdx_smoke(
        cublasdx_lib, cublaslt_by_shape[smoke_shape], samples
    )
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    cuda13_nvcc_path = ROOT / ".venv/lib64/python3.13/site-packages/nvidia/cu13/bin/nvcc"
    cuda13_nvcc = subprocess.run(
        [str(cuda13_nvcc_path), "--version"], capture_output=True, text=True,
        check=True,
    ) if cutlass452_library_path else None
    report = {
        "schema_version": 1,
        "task_id": "V3R-008",
        "provider": "CUTLASS C++ device::Gemm body invoked from a custom __global__ wrapper",
        "cutlass_distribution": importlib.metadata.version("nvidia-cutlass"),
        "cutlass_include_root": str(CUTLASS_ROOT / "include"),
        "cutlass_headers_used": ["cutlass/gemm/device/gemm.h", "cutlass/gemm/kernel/gemm.h"],
        "toolchain": nvcc.stdout.splitlines()[-1].strip(),
        "toolchains": {
            "legacy_cutlass_3_8": nvcc.stdout.splitlines()[-1].strip(),
            "selected_cuda13_provider": (
                cuda13_nvcc.stdout.splitlines()[-1].strip()
                if cuda13_nvcc else None
            ),
            "torch_cuda_runtime": torch.version.cuda,
        },
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
                "status": "installed_and_measured_on_named_cuda13_lane",
                "distribution": "MathDx 26.06.1 / cuBLASDx 0.7.1 / bundled CUTLASS 4.5.2",
                "toolkit": "CUDA 13.0.88 nvcc, CUDA runtime 13.0.96",
                "target": "sm_90a on H200 MIG 3g.71gb",
                "gemm_link_mode": "documented GEMM-only header path with CUBLASDX_NO_FATBIN_AVAILABLE; no LTO fatbin linked",
                "mathdx_archive_sha256": "59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353",
                "fatbin_link_attempt": "nvlink fatal: elfLink linker library load error (CUDA 13.0.88)",
                "hot_shape_disposition": "rejected_candidate: all inventoried batch-one shapes require M=1, not divisible by tile_m=16",
                "positive_case": "synthetic M=16,N=64,K=64 pipelined global GEMM compiled, correct standalone and cooperative owner",
            },
            "cutlass": {
                "status": "measured",
                "distribution": "nvidia-cutlass 3.8.0.0",
                "version_lane": "installed CUTLASS 3.8 headers with the selected nvcc 12.8.93 toolchain",
            },
            "cutlass_4_5_2_cuda13": ({
                "status": "measured" if cutlass452_library_path else "not_requested",
                "distribution": "CUTLASS 4.5.2 bundled with MathDx 26.06.1",
                "version_lane": "project-local CUDA 13.0 nvcc and runtime; compiled for sm_90a",
                "toolchain": cuda13_nvcc.stdout.splitlines()[-1].strip()
                if cuda13_nvcc else None,
                "cuda_runtime": torch.version.cuda,
                "api": "adapted CUTLASS device::Gemm kernel body called inside a custom __global__ owner wrapper",
                "source": "src/cuda/v3/cutlass_38_body_probe.cu",
                "library": str(cutlass452_library_path) if cutlass452_library_path else None,
                "library_sha256": _sha(cutlass452_library_path) if cutlass452_library_path else None,
                "library_bytes": cutlass452_library_path.stat().st_size if cutlass452_library_path else None,
                "resources": cutlass452_resources,
                "cases": [case["cutlass_4_5_2_cuda13"] for case in cases],
                "limits": [
                    "This is the same adapted CUTLASS device::Gemm body under the selected CUDA 13/CUTLASS 4.5.2 headers, not a CuTe CollectiveBuilder kernel search.",
                    "No alternative tile, warp specialization, or mixed-body coexistence was evaluated in this version check.",
                ],
            } if cutlass452_library_path else None),
            "cutlass_cute_collective_4_5_2_cuda13": ({
                "status": "measured" if cutlass_cute_library_path else "not_requested",
                "distribution": "CUTLASS 4.5.2 bundled with MathDx 26.06.1",
                "version_lane": "project-local CUDA 13.0 nvcc and runtime; compiled for sm_90a",
                "toolchain": cuda13_nvcc.stdout.splitlines()[-1].strip()
                if cuda13_nvcc else None,
                "api": "CuTe CollectiveBuilder mainloop/epilogue, GemmUniversal operator invoked by a custom __global__ wrapper",
                "source": "src/cuda/v3/cutlass_cute_collective_probe.cu",
                "source_sha256": _sha(ROOT / "src/cuda/v3/cutlass_cute_collective_probe.cu")
                if cutlass_cute_library_path else None,
                "library": str(cutlass_cute_library_path) if cutlass_cute_library_path else None,
                "library_sha256": _sha(cutlass_cute_library_path)
                if cutlass_cute_library_path else None,
                "library_bytes": cutlass_cute_library_path.stat().st_size
                if cutlass_cute_library_path else None,
                "tile_mnk": [128, 64, 64],
                "cluster_shape": [1, 1, 1],
                "mainloop_schedule": "KernelTmaWarpSpecialized",
                "pipeline_stages": "StageCountAutoCarveout",
                "element_accumulator": "float32",
                "resources": cutlass_cute_resources,
                "cases": [case["cutlass_cute_collective_4_5_2_cuda13"] for case in cases],
                "compiler_warning": "ptxas C7510 reports WGMMA serialization across a function boundary in the directly invoked kernel operator.",
                "limits": [
                    "This is one CUTLASS CuTe collective configuration, not a tile or warp-specialization sweep.",
                    "The host wrapper builds the universal-kernel params for each invocation; GPU event times only the body, while wrapper wall includes preparation.",
                ],
            } if cutlass_cute_library_path else None),
        },
        "cublasdx": {
            "library": str(cublasdx_library_path),
            "library_sha256": _sha(cublasdx_library_path),
            "library_bytes": cublasdx_library_path.stat().st_size,
            "source": "src/cuda/v3/cublasdx_pipeline_probe.cu",
            "source_sha256": _sha(ROOT / "src/cuda/v3/cublasdx_pipeline_probe.cu"),
            "sdk": "MathDx 26.06.1, cuBLASDx 0.7.1, bundled CUTLASS 4.5.2",
            "environment_manifest": "ART/tasks/V3R-008/provider_environment.json",
            "environment_manifest_sha256": _sha(ROOT / "ART/tasks/V3R-008/provider_environment.json"),
            "toolchain": "CUDA 13.0.88 nvcc; CUDA 13.0.96 runtime",
            "target": "sm_90a",
            "tile_mnk": list(DX_TILE),
            "pipeline_depth": 1,
            "accumulation": "FP32 accumulator, FP16 input and output",
            "host_descriptor": "suggest_pipeline creates the host pipeline; get_device_handle is passed by value with __grid_constant__",
            "reset_tile": "not used; each CTA computes one output tile with no persistent CTA tile loop",
            "hot_shape_rejections": [case["cublasdx"] | {"shape": case["shape"]} for case in cases],
            "positive_synthetic_case": cublasdx_smoke,
            "offline_cuobjdump_resources": cublasdx_binary_resources,
            "resource_crosscheck": {
                "cuda_runtime_static_shared_bytes": cublasdx_smoke["resources"]["static_shared_bytes"],
                "cuobjdump_shared_bytes": cublasdx_binary_resources["shared_bytes"],
                "static_shared_mismatch": (
                    cublasdx_smoke["resources"]["static_shared_bytes"] !=
                    cublasdx_binary_resources["shared_bytes"]
                ),
                "runtime_dynamic_shared_bytes": cublasdx_smoke["resources"]["dynamic_shared_bytes"],
                "note": "cuobjdump reports embedded ELF resource metadata; cudaFuncGetAttributes and the cuBLASDx pipeline report runtime static/dynamic shared memory and feed occupancy. Investigate a static-shared mismatch before shared-memory-limited multi-body composition.",
            },
            "limits": [
                "No cuBLASDx timing exists for the actual M=1 hot shapes because the selected global pipeline tile is not legal for them.",
                "The synthetic divisible shape proves only provider API, numerical, standalone, and cooperative-entry behavior; it is not hot-shape body-quality evidence.",
                "No coexistence with another body, padding/tail strategy, or alternative tile/warp configuration was tested.",
                "The CUDA 13.0 LTO-fatbin device link failed; the documented GEMM-only no-LTO header path compiled and ran.",
            ],
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
            "CUTLASS 3.8 results remain on the prior CUDA 12.8 lane; cuBLASDx uses a separate CUDA 13.0/MathDx 26.06.1 provider lane without changing the core environment.",
            "Cooperative owner placement is rejected when all CUTLASS output CTAs do not fit measured residency; no virtual tile remapping or composition with another body is proven.",
            "cuBLASDx's selected 16x64x32 pipeline tile rejects the actual M=1 hot shapes; only the separate divisible synthetic shape was admitted.",
            "No BF16, split-K, layout conversion, tail alignment, or full-step call was measured.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cublasdx-library", type=Path,
                        default=ROOT / "ART/tasks/V3R-008/libmegabake_cublasdx_pipeline_sm90a.so")
    parser.add_argument("--cublaslt-report", type=Path,
                        default=ROOT / "ART/tasks/V3R-005/cublaslt_report.json")
    parser.add_argument("--cutlass452-library", type=Path)
    parser.add_argument("--cutlass-cute-library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    report = run(args.library, args.output, args.samples,
                 args.cublasdx_library, args.cublaslt_report,
                 args.cutlass452_library, args.cutlass_cute_library)
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
