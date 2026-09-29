"""Measure the direct cuBLASLt external operation required by V3R-005."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((1, 192, 576), (1, 576, 576), (1, 576, 1536),
          (1, 1536, 576), (1, 49152, 576), (16, 64, 64))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.mb3_cublaslt_create.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.mb3_cublaslt_create.restype = ctypes.c_int
    lib.mb3_cublaslt_launch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.mb3_cublaslt_launch.restype = ctypes.c_int
    lib.mb3_cublaslt_destroy.argtypes = [ctypes.c_void_p]
    return lib


def _create(lib: Any, shape: tuple[int, int, int]):
    m, n, k = shape
    context, algorithm, workspace = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_size_t()
    status = lib.mb3_cublaslt_create(
        m, n, k, ctypes.byref(context), ctypes.byref(algorithm), ctypes.byref(workspace)
    )
    if status:
        raise RuntimeError(f"cuBLASLt descriptor/heuristic setup returned status {status} for {shape}")
    return context, int(algorithm.value), int(workspace.value)


def _launch(lib: Any, context: Any, x: Any, weight: Any, output: Any, stream: Any) -> None:
    status = lib.mb3_cublaslt_launch(
        context, weight.data_ptr(), x.data_ptr(), output.data_ptr(), stream.cuda_stream
    )
    if status:
        raise RuntimeError(f"cublasLtMatmul returned status {status}")


def _measure(lib: Any, context: Any, x: Any, weight: Any, output: Any,
             *, samples: int) -> dict[str, Any]:
    stream = torch.cuda.Stream(device=torch.cuda.current_device())
    for _ in range(5):
        _launch(lib, context, x, weight, output, stream)
    stream.synchronize()
    gpu, submission, complete = [], [], []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            begin.record(stream)
            submitted = time.perf_counter()
            _launch(lib, context, x, weight, output, stream)
            submission.append((time.perf_counter() - submitted) * 1e6)
            end.record(stream)
        end.synchronize()
        complete.append((time.perf_counter() - started) * 1e6)
        gpu.append(begin.elapsed_time(end) * 1000.0)
    return {
        "path": "direct cublasLtMatmul; host descriptors/heuristic reused; preallocated output",
        "placement": "external_vendor_operation",
        "samples": samples,
        "raw_gpu_event_us": gpu,
        "median_gpu_event_us": statistics.median(gpu),
        "raw_host_submission_us": submission,
        "median_host_submission_us": statistics.median(submission),
        "raw_complete_callable_wall_us": complete,
        "median_complete_callable_wall_us": statistics.median(complete),
        "timing_note": "GPU event times the cuBLASLt operation. Submission is the host API call duration; complete wall includes event synchronization.",
    }


def _measure_setup(lib: Any, shape: tuple[int, int, int], *, samples: int) -> dict[str, Any]:
    raw, algorithms, workspaces = [], [], []
    for _ in range(samples):
        started = time.perf_counter()
        context, algorithm, workspace = _create(lib, shape)
        raw.append((time.perf_counter() - started) * 1e6)
        algorithms.append(algorithm)
        workspaces.append(workspace)
        lib.mb3_cublaslt_destroy(context)
    return {
        "operation": "create handle, matmul descriptor, matrix layouts, heuristic preference, and selected algorithm",
        "samples": samples,
        "raw_host_wall_us": raw,
        "median_host_wall_us": statistics.median(raw),
        "algorithm_ids": sorted(set(algorithms)),
        "workspace_bytes": sorted(set(workspaces)),
    }


def run(library_path: Path, output_path: Path, samples: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuBLASLt baseline probe requires an available CUDA GPU")
    lib = _load(library_path)
    invalid_context = ctypes.c_void_p()
    invalid_algorithm, invalid_workspace = ctypes.c_int(), ctypes.c_size_t()
    negative_status = lib.mb3_cublaslt_create(
        0, 192, 576, ctypes.byref(invalid_context), ctypes.byref(invalid_algorithm),
        ctypes.byref(invalid_workspace),
    )
    if negative_status != 7 or invalid_context.value:
        raise AssertionError(f"invalid M=0 descriptor should return CUBLAS_STATUS_INVALID_VALUE, got {negative_status}")

    cases = []
    for m, n, k in SHAPES:
        shape = (m, n, k)
        torch.manual_seed(77000 + n + k)
        x = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.02
        weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
        backing = torch.full((m * n + 16,), -123.0, device="cuda", dtype=torch.float16)
        output = backing[8:8 + m * n].view(m, n)
        expected = torch.addmm(torch.zeros((m, n), device="cuda", dtype=torch.float16),
                               x, weight.t())
        torch.cuda.synchronize()
        setup = _measure_setup(lib, shape, samples=samples)
        context, algorithm, workspace = _create(lib, shape)
        try:
            measurement = _measure(lib, context, x, weight, output, samples=samples)
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)
            assert bool(torch.all(backing[:8] == -123.0))
            assert bool(torch.all(backing[8 + m * n:] == -123.0))
            cases.append({
                "shape": list(shape), "dtype": "float16",
                "weight_layout": "row_major[N,K], interpreted as column-major [K,N] with opA=T",
                "x_layout": "row_major[M,K], interpreted as column-major [K,M] with opB=N",
                "output_layout": "row_major[M,N], interpreted as column-major [N,M]",
                "accumulation": "FP32", "alpha": 1.0, "beta": 0.0,
                "algorithm_id": algorithm, "workspace_bytes": workspace,
                "descriptor_setup": setup,
                "packing": {"performed": False, "measured_host_wall_us": 0.0},
                "measurement": measurement,
                "max_abs_error": float((output.float() - expected.float()).abs().max().item()),
                "active_store_canaries_intact": True,
            })
        finally:
            lib.mb3_cublaslt_destroy(context)

    nvcc = subprocess.run([
        str(ROOT / ".venv/lib64/python3.13/site-packages/nvidia/cu13/bin/nvcc"),
        "--version",
    ], capture_output=True, text=True, check=True)
    report = {
        "schema_version": 1,
        "task_id": "V3R-005",
        "provider": "direct cuBLASLt C API external operation",
        "toolchain": nvcc.stdout.splitlines()[-1].strip(),
        "target": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "visible_sms": int(torch.cuda.get_device_properties(0).multi_processor_count),
            "compiled_target": "sm_90a", "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cublas_package": importlib.metadata.version("nvidia-cublas"),
        },
        "descriptor_setup_scope": "handle, operation, layouts, preference, and heuristic setup measured separately; setup excluded from the reused-descriptor GPU event path",
        "negative_cases": [{
            "shape": [0, 192, 576], "expected": "CUBLAS_STATUS_INVALID_VALUE (7)",
            "status": negative_status, "context_created": False,
        }],
        "cases": cases,
        "artifacts": {
            "source": "src/cuda/v3/cublaslt_baseline_probe.cu",
            "source_sha256": _sha(ROOT / "src/cuda/v3/cublaslt_baseline_probe.cu"),
            "library": str(library_path), "library_sha256": _sha(library_path),
            "library_bytes": library_path.stat().st_size,
        },
        "qualification": "External standalone cuBLASLt operation baseline only. No custom owner-body reuse or one-grid/full-step result is claimed.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path,
                        default=ROOT / "ART/tasks/V3R-005/libmegabake_cublaslt_probe_sm90a.so")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ART/tasks/V3R-005/cublaslt_report.json")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    report = run(args.library, args.output, args.samples)
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
