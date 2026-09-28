"""Standalone and lean cooperative-entry CUDA body measurements for V3R-005/006."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from megabake.v3.backends.cuda.simt import (  # noqa: E402
    ContractionShape,
    SimtSchedule,
    enumerate_simt_schedules,
    owner_grid_candidates,
    output_tiles,
)


SERIAL = 0
SIMT = 1


class Problem(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("weight", ctypes.c_void_p),
        ("bias", ctypes.c_void_p), ("output", ctypes.c_void_p),
        ("m", ctypes.c_int64), ("n", ctypes.c_int64),
        ("k", ctypes.c_int64), ("x_m_stride", ctypes.c_int64),
        ("x_k_stride", ctypes.c_int64),
        ("weight_n_stride", ctypes.c_int64),
        ("weight_k_stride", ctypes.c_int64),
        ("alpha", ctypes.c_float), ("beta", ctypes.c_float),
    ]


class Resources(ctypes.Structure):
    _fields_ = [
        ("device_sms", ctypes.c_int), ("cooperative_launch", ctypes.c_int),
        ("active_ctas_per_sm", ctypes.c_int), ("resident_ctas", ctypes.c_int),
        ("registers_per_thread", ctypes.c_int),
        ("static_shared_bytes", ctypes.c_int), ("local_bytes", ctypes.c_int),
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_library(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "src/cuda/v3/body_harness.cu"
    library = output_dir / "libmegabake_v3_body_sm90a.so"
    command = [
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_90a",
        "--shared", "-Xcompiler", "-fPIC", "-Xptxas=-v", str(source),
        "-o", str(library),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"nvcc failed ({result.returncode})\n{result.stdout}\n{result.stderr}"
        )
    resource_dump = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(library)],
        cwd=ROOT, text=True, capture_output=True,
    )
    cubin = output_dir / "body_harness_sm90a.cubin"
    cubin_command = [
        "nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_90a",
        "-cubin", "-Xptxas=-v", str(source), "-o", str(cubin),
    ]
    cubin_result = subprocess.run(cubin_command, cwd=ROOT, text=True,
                                  capture_output=True)
    if cubin_result.returncode:
        raise RuntimeError(
            f"cubin build failed ({cubin_result.returncode})\n"
            f"{cubin_result.stdout}\n{cubin_result.stderr}"
        )
    sections = subprocess.run(["readelf", "-SW", str(cubin)], cwd=ROOT,
                              text=True, capture_output=True, check=True)
    code_sections = {}
    for line in sections.stdout.splitlines():
        match = re.match(
            r"\s*\[\s*\d+\]\s+(\.text\.\S+)\s+PROGBITS\s+\S+\s+\S+\s+([0-9a-fA-F]+)\b",
            line,
        )
        if match:
            code_sections[match.group(1)] = int(match.group(2), 16)
    return {
        "command": command, "stdout": result.stdout, "stderr": result.stderr,
        "library": str(library), "library_sha256": sha256(library),
        "library_bytes": library.stat().st_size,
        "resource_dump_command": ["cuobjdump", "--dump-resource-usage", str(library)],
        "resource_dump_return_code": resource_dump.returncode,
        "resource_dump_stdout": resource_dump.stdout,
        "resource_dump_stderr": resource_dump.stderr,
        "cubin_command": cubin_command,
        "cubin_stdout": cubin_result.stdout,
        "cubin_stderr": cubin_result.stderr,
        "cubin": str(cubin), "cubin_sha256": sha256(cubin),
        "cubin_bytes": cubin.stat().st_size,
        "device_code_sections_bytes": code_sections,
    }


def load_library(path: str | Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.mb3_owner_profile.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(Resources)]
    lib.mb3_standalone_profile.argtypes = list(lib.mb3_owner_profile.argtypes)
    lib.mb3_validate_owner_grid.argtypes = [ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int]
    lib.mb3_launch.argtypes = [ctypes.POINTER(Problem), ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_size_t]
    lib.mb3_capture.argtypes = [ctypes.POINTER(Problem), ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_void_p)]
    lib.mb3_graph_launch.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mb3_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.mb3_error_string.argtypes = [ctypes.c_int]
    lib.mb3_error_string.restype = ctypes.c_char_p
    return lib


def _check(lib: ctypes.CDLL, code: int, action: str) -> None:
    if code:
        detail = lib.mb3_error_string(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"{action}: CUDA status {code}: {detail}")


def _profile(lib: ctypes.CDLL, strategy: int, schedule: SimtSchedule,
             *, standalone: bool = False) -> dict[str, int]:
    profile = Resources()
    fn = lib.mb3_standalone_profile if standalone else lib.mb3_owner_profile
    _check(lib, fn(strategy, schedule.warps_per_cta,
                   schedule.vector_width, ctypes.byref(profile)), "function profile")
    return {name: int(getattr(profile, name)) for name, _ in profile._fields_}


def _problem(x: Any, weight: Any, bias: Any, output: Any, *, alpha: float,
             beta: float, w_n_stride: int, w_k_stride: int) -> Problem:
    return Problem(
        x.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else None,
        output.data_ptr(), int(x.shape[0]), int(output.shape[1]), int(x.shape[1]),
        int(x.stride(0)), int(x.stride(1)), int(w_n_stride), int(w_k_stride),
        float(alpha), float(beta),
    )


def _guarded_output(torch: Any, m: int, n: int, device: Any):
    storage = torch.full((m * n + 2,), -123.0, device=device, dtype=torch.float16)
    return storage, storage[1:-1].view(m, n)


def _assert_output(torch: Any, actual: Any, expected: Any, storage: Any,
                   label: str) -> float:
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005,
                               msg=f"{label} differs from torch reference")
    assert storage[0].item() == -123.0 and storage[-1].item() == -123.0, (
        f"{label} wrote outside its output region"
    )
    return float((actual.float() - expected.float()).abs().max().item())


def _body_launch(lib: ctypes.CDLL, problem: Problem, strategy: int,
                 schedule: SimtSchedule, grid: int, stream: Any) -> None:
    _check(lib, lib.mb3_launch(ctypes.byref(problem), strategy,
                               schedule.warps_per_cta, schedule.vector_width,
                               grid, stream.cuda_stream), "body launch")


def _capture_body(lib: ctypes.CDLL, problem: Problem, strategy: int,
                  schedule: SimtSchedule, grid: int, launches: int,
                  stream: Any) -> ctypes.c_void_p:
    graph = ctypes.c_void_p()
    _check(lib, lib.mb3_capture(ctypes.byref(problem), strategy,
                                schedule.warps_per_cta, schedule.vector_width,
                                grid, launches, stream.cuda_stream,
                                ctypes.byref(graph)), "body graph capture")
    return graph


def _measure_graph(torch: Any, graph: Any, launches_per_graph: int,
                   stream: Any, samples: int, replays: int,
                   body_graph: ctypes.c_void_p | None = None,
                   lib: ctypes.CDLL | None = None) -> list[float]:
    values = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record(stream)
        for _ in range(replays):
            if body_graph is not None:
                _check(lib, lib.mb3_graph_launch(body_graph, stream.cuda_stream),
                       "body graph replay")
            else:
                with torch.cuda.stream(stream):
                    graph.replay()
        end.record(stream)
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 /
                      (launches_per_graph * replays))
    return values


def _torch_graph(torch: Any, fn: Any, launches: int, stream: Any):
    graph = torch.cuda.CUDAGraph()
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(launches):
            fn()
    stream.synchronize()
    return graph


def _vendor_kernel_names(torch: Any, fn: Any, stream: Any) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    stream.synchronize()
    with torch.profiler.profile(activities=activities) as prof:
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
    return [event.name for event in prof.events()
            if "cuda" in str(event.device_type).lower()]


def _device_record(torch: Any) -> dict[str, Any]:
    prop = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()[0].strip()
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "visible_sms": int(prop.multi_processor_count),
        "total_memory_bytes": int(prop.total_memory),
        "cooperative_launch": bool(getattr(prop, "cooperative_launch", True)),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "driver_version": driver, "nvcc_path": shutil.which("nvcc"),
        "toolchain": subprocess.run(["nvcc", "--version"], text=True,
                                     capture_output=True, check=True).stdout.splitlines()[-1].strip(),
        "compiled_target": "sm_90a",
    }


def _make_tensors(torch: Any, shape: ContractionShape, *, seed: int,
                  transposed_weight: bool = False, with_bias: bool = False,
                  alpha: float = 1.0, beta: float = 0.0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    x = torch.randn((shape.m, shape.k), device=device, dtype=torch.float16) * 0.02
    if transposed_weight:
        weight_storage = torch.randn((shape.k, shape.n), device=device,
                                     dtype=torch.float16) * 0.02
        right = weight_storage
        w_n_stride, w_k_stride = 1, shape.n
    else:
        weight_storage = torch.randn((shape.n, shape.k), device=device,
                                     dtype=torch.float16) * 0.02
        right = weight_storage.t()
        w_n_stride, w_k_stride = shape.k, 1
    bias = (torch.randn((shape.m, shape.n), device=device, dtype=torch.float16) * 0.02
            if with_bias else None)
    bias_input = bias if bias is not None else torch.zeros(
        (shape.m, shape.n), device=device, dtype=torch.float16
    )
    output_storage, output = _guarded_output(torch, shape.m, shape.n, device)
    vendor_output = torch.empty((shape.m, shape.n), device=device, dtype=torch.float16)
    ref = torch.addmm(bias_input, x, right, beta=beta, alpha=alpha)
    problem = _problem(x, weight_storage, bias, output, alpha=alpha, beta=beta,
                       w_n_stride=w_n_stride, w_k_stride=w_k_stride)
    vendor = (lambda: torch.addmm(bias_input, x, right, beta=beta, alpha=alpha,
                                  out=vendor_output))
    return {
        "x": x, "weight": weight_storage, "bias": bias, "output": output,
        "output_storage": output_storage, "vendor_output": vendor_output,
        "bias_input": bias_input, "right": right, "reference": ref,
        "problem": problem, "vendor_fn": vendor,
        "weight_strides": [w_n_stride, w_k_stride],
    }


def _run_placement(torch: Any, lib: ctypes.CDLL, tensors: dict[str, Any],
                   strategy: int, schedule: SimtSchedule, grid: int,
                   *, samples: int, graph_launches: int, replays: int,
                   stream: Any, label: str) -> dict[str, Any]:
    problem = tensors["problem"]
    _body_launch(lib, problem, strategy, schedule, grid, stream)
    stream.synchronize()
    max_abs = _assert_output(torch, tensors["output"], tensors["reference"],
                             tensors["output_storage"], label)
    resources = _profile(lib, strategy, schedule, standalone=(grid == -1))
    graph = _capture_body(lib, problem, strategy, schedule, grid,
                          graph_launches, stream)
    raw = _measure_graph(torch, None, graph_launches, stream, samples, replays,
                         body_graph=graph, lib=lib)
    _check(lib, lib.mb3_graph_destroy(graph), "body graph destroy")
    return {
        "placement": label, "grid_ctas": grid,
        "resources": resources, "raw_gpu_event_us": raw,
        "median_gpu_event_us": statistics.median(raw),
        "max_abs_error": max_abs, "correct": True,
    }


def _run_case(torch: Any, lib: ctypes.CDLL, shape: ContractionShape,
              *, seed: int, schedules: tuple[SimtSchedule, ...], strategy: int,
              samples: int, graph_launches: int, replays: int,
              transposed_weight: bool = False, with_bias: bool = False,
              alpha: float = 1.0, beta: float = 0.0,
              origin: str = "fixture", call_count: int = 1) -> dict[str, Any]:
    shape.validate()
    tensors = _make_tensors(torch, shape, seed=seed,
                            transposed_weight=transposed_weight,
                            with_bias=with_bias, alpha=alpha, beta=beta)
    input_snapshots = {
        key: tensors[key].clone()
        for key in ("x", "weight", "bias") if tensors[key] is not None
    }
    source_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=torch.cuda.current_device())
    stream.wait_stream(source_stream)
    with torch.cuda.stream(stream):
        tensors["vendor_fn"]()
    stream.synchronize()
    torch.testing.assert_close(tensors["vendor_output"], tensors["reference"],
                               rtol=0.02, atol=0.005)
    vendor_graph = _torch_graph(torch, tensors["vendor_fn"], graph_launches, stream)
    vendor_kernel_names = _vendor_kernel_names(torch, tensors["vendor_fn"], stream)
    vendor_raw = _measure_graph(torch, vendor_graph, graph_launches, stream,
                                samples, replays)
    measurements = []
    for schedule in schedules:
        if strategy == SERIAL:
            standalone_grid = -1
            profile = _profile(lib, strategy, schedule, standalone=False)
            owner_grid = min(profile["device_sms"], math.ceil(shape.output_elements / 128))
            # The serial standalone wrapper has four warps (128 threads); reflect
            # its real function resources rather than the requested SIMT tile.
            standalone_resources = _profile(lib, strategy, schedule, standalone=True)
            serial_schedule = SimtSchedule(4, 1)
            result_s = _run_placement(
                torch, lib, tensors, strategy, serial_schedule, standalone_grid,
                samples=samples, graph_launches=graph_launches, replays=replays,
                stream=stream, label="standalone",
            )
            result_s["resources"] = standalone_resources
            _check(lib, lib.mb3_validate_owner_grid(strategy, 4, 1, owner_grid),
                   "serial owner-grid admission")
            result_e = _run_placement(
                torch, lib, tensors, strategy, serial_schedule, owner_grid,
                samples=samples, graph_launches=graph_launches, replays=replays,
                stream=stream, label="cooperative_owner_entry",
            )
            measurements.append({"schedule": {"strategy": "serial_control",
                                                "warps_per_cta": 4,
                                                "vector_width": 1},
                                 "output_tiles": math.ceil(shape.output_elements / 128),
                                 "standalone": result_s,
                                 "owner_entries": [result_e]})
            break
        standalone_grid = -1
        owner_profile = _profile(lib, strategy, schedule)
        owner_grids = owner_grid_candidates(
            shape, schedule, owner_profile["device_sms"],
            owner_profile["resident_ctas"],
        )
        result_s = _run_placement(
            torch, lib, tensors, strategy, schedule, standalone_grid,
            samples=samples, graph_launches=graph_launches, replays=replays,
            stream=stream, label="standalone",
        )
        owner_entries = []
        for owner_grid in owner_grids:
            if owner_grid > owner_profile["resident_ctas"]:
                raise RuntimeError("candidate owner grid exceeds actual-entry cooperative residency")
            _check(lib, lib.mb3_validate_owner_grid(
                strategy, schedule.warps_per_cta, schedule.vector_width, owner_grid
            ), "owner-grid admission")
            owner_entries.append(_run_placement(
                torch, lib, tensors, strategy, schedule, owner_grid,
                samples=samples, graph_launches=graph_launches, replays=replays,
                stream=stream, label=f"cooperative_owner_entry:{owner_grid}_ctas",
            ))
        selected_owner = min(owner_entries, key=lambda item: item["median_gpu_event_us"])
        measurements.append({
            "schedule": {"strategy": "k_parallel_simt",
                         "warps_per_cta": schedule.warps_per_cta,
                         "vector_width": schedule.vector_width,
                         "threads_per_cta": schedule.threads_per_cta},
            "output_tiles": output_tiles(shape, schedule),
            "standalone": result_s, "owner_entries": owner_entries,
            "selected_owner_grid_ctas": selected_owner["grid_ctas"],
        })
    # Recheck caller-visible inputs and bias after both placements.
    for key, snapshot in input_snapshots.items():
        assert torch.equal(tensors[key], snapshot), f"body modified input {key}"
    result = {
        "origin_fx": origin, "call_count": call_count,
        "seed": seed,
        "weight_data": "deterministic synthetic fp16 values; same tensors feed every tactic and local vendor control",
        "shape": {"M": shape.m, "N": shape.n, "K": shape.k,
                  "x_strides": [shape.x_m_stride, shape.x_k_stride],
                  "weight_strides": tensors["weight_strides"],
                  "dtype": shape.input_dtype,
                  "accumulation_dtype": shape.accumulation_dtype,
                  "alpha": alpha, "beta": beta,
                  "bias": with_bias, "working_set": "same resident tensors, repeated graph calls"},
        "vendor_control": {
            "path": "torch.addmm CUDA op on same buffers and stream",
            "profiled_cuda_kernel_names": vendor_kernel_names,
            "raw_gpu_event_us": vendor_raw,
            "median_gpu_event_us": statistics.median(vendor_raw),
            "correct": True,
        },
        "tactics": measurements,
    }
    return result


def _unique_shapes(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in inventory["hot_shapes"]:
        if "gemm" not in record:
            continue
        gemm = record["gemm"]
        key = (int(gemm["M"]), int(gemm["N"]), int(gemm["K"]))
        records.setdefault(key, record)
    return [records[key] for key in sorted(records)]


def run_card(card: str, library: ctypes.CDLL, *, samples: int = 15,
             graph_launches: int = 8, replays: int = 3,
             inventory_path: Path | None = None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("V3 GPU lane is unavailable")
    device = _device_record(torch)
    if card == "005":
        inventory_path = inventory_path or ROOT / "ART/tasks/V3R-004/inventory/smollm2-135m-fp16-b1-L128/inventory.json"
        inventory = json.loads(inventory_path.read_text())
        record = next(r for r in _unique_shapes(inventory)
                      if r["gemm"]["M"] == 1 and r["gemm"]["N"] == 576
                      and r["gemm"]["K"] == 576)
        shape = ContractionShape.from_inventory(record)
        result = _run_case(
            torch, library, shape, seed=5005, schedules=(SimtSchedule(4, 1),),
            strategy=SERIAL, samples=samples, graph_launches=graph_launches,
            replays=replays, origin=record["fx_origin"],
            call_count=record["call_count"],
        )
        # Negative admission: even a legal body must reject a grid larger than
        # the selected entry's measured cooperative residency.
        res = _profile(library, SERIAL, SimtSchedule(4, 1))
        rejected_grid = res["resident_ctas"] + 1
        rejection = library.mb3_validate_owner_grid(SERIAL, 4, 1, rejected_grid)
        if rejection == 0:
            raise AssertionError("over-resident owner grid was incorrectly admitted")
        result["negative_case"] = {
            "case": "owner_grid_exceeds_compiled_entry_residency",
            "requested_grid_ctas": rejected_grid,
            "resident_ctas": res["resident_ctas"], "rejected": True,
        }
        result["inventory_cell"] = inventory["cell_id"]
        result["selected_baseline_id"] = inventory["selected_baseline_id"]
        result["workload"] = {
            "checkpoint": inventory["checkpoint"],
            "checkpoint_revision": inventory["checkpoint_revision"],
            "manifest_hash": inventory["manifest_hash"],
            "cache_condition": inventory["cache_condition"],
            "shape_inventory": str(inventory_path),
        }
        return {"card": "V3R-005", "target": device, "cases": [result],
                "claim": "not_measured", "strict_one_grid_speed_claim": False}

    if card != "006":
        raise ValueError("card must be 005 or 006")
    inventory_path = inventory_path or ROOT / "ART/tasks/V3R-004/inventory/smollm2-135m-fp16-b1-L128/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    cases = []
    for index, record in enumerate(_unique_shapes(inventory)):
        shape = ContractionShape.from_inventory(record)
        cases.append(_run_case(
            torch, library, shape, seed=6000 + index,
            schedules=enumerate_simt_schedules(shape), strategy=SIMT,
            samples=samples, graph_launches=graph_launches, replays=replays,
            origin=record["fx_origin"], call_count=record["call_count"],
        ))
    # Odd N/K, transposed source weight, and non-unit alpha/beta exercise the
    # negative-index tails and arithmetic contract independently of hot shapes.
    tiny = ContractionShape(1, 17, 33, 33, 1, 1, 17,
                            "float16", "float16", "float32")
    tiny_case = _run_case(
        torch, library, tiny, seed=6060,
        schedules=enumerate_simt_schedules(tiny), strategy=SIMT,
        samples=samples, graph_launches=graph_launches, replays=replays,
        transposed_weight=True, with_bias=True, alpha=1.75, beta=-0.25,
        origin="LINEAR_TINY:transpose_weight/addmm_alpha_beta/tail",
    )
    cases.append(tiny_case)
    # Negative launch contract: nonzero beta cannot read an absent addmm input.
    problem = tiny_case_problem = _make_tensors(
        torch, tiny, seed=6061, transposed_weight=True,
        with_bias=False, alpha=1.0, beta=0.5,
    )
    tiny_case_problem["problem"].beta = 0.5
    tiny_case_problem["problem"].bias = None
    negative_stream = torch.cuda.current_stream()
    rejected = library.mb3_launch(
        ctypes.byref(tiny_case_problem["problem"]), SIMT, 1, 1, -1,
        negative_stream.cuda_stream,
    )
    if rejected == 0:
        raise AssertionError("beta without a bias input was incorrectly launched")
    return {
        "card": "V3R-006", "target": device,
        "inventory_cell": inventory["cell_id"],
        "selected_baseline_id": inventory["selected_baseline_id"],
        "workload": {
            "checkpoint": inventory["checkpoint"],
            "checkpoint_revision": inventory["checkpoint_revision"],
            "manifest_hash": inventory["manifest_hash"],
            "cache_condition": inventory["cache_condition"],
            "shape_inventory": str(inventory_path),
        },
        "cases": cases,
        "negative_case": {"case": "nonzero_beta_without_bias_input",
                          "rejected": True, "status": int(rejected)},
        "claim": "not_measured", "strict_one_grid_speed_claim": False,
    }


def run(output_dir: Path, card: str, *, samples: int, graph_launches: int,
        replays: int, inventory_path: Path | None = None) -> dict[str, Any]:
    build = compile_library(output_dir)
    lib = load_library(build["library"])
    report = run_card(card, lib, samples=samples, graph_launches=graph_launches,
                      replays=replays, inventory_path=inventory_path)
    code_sections = build["device_code_sections_bytes"]
    for case in report["cases"]:
        for tactic in case["tactics"]:
            schedule = tactic["schedule"]
            if schedule["strategy"] == "serial_control":
                standalone_tag = "serial_standaloneILi4EE"
                owner_tag = "serial_ownerILi4EE"
            else:
                suffix = (f"ILi{schedule['warps_per_cta']}E"
                          f"Li{schedule['vector_width']}EE")
                standalone_tag = "simt_standalone" + suffix
                owner_tag = "simt_owner" + suffix
            standalone_matches = [size for name, size in code_sections.items()
                                  if standalone_tag in name]
            owner_matches = [size for name, size in code_sections.items()
                             if owner_tag in name]
            if len(standalone_matches) != 1 or len(owner_matches) != 1:
                raise RuntimeError(
                    f"expected cubin text sections for {standalone_tag}/{owner_tag}"
                )
            tactic["standalone"]["compiled_code_bytes"] = standalone_matches[0]
            for owner_entry in tactic["owner_entries"]:
                owner_entry["compiled_code_bytes"] = owner_matches[0]
    report["build"] = build
    report["source_sha256"] = sha256(ROOT / "src/cuda/v3/body_harness.cu")
    report["header_sha256"] = sha256(ROOT / "src/cuda/v3/body_harness.h")
    report["driver_sha256"] = sha256(Path(__file__))
    report["schedule_generator_sha256"] = sha256(
        ROOT / "src/megabake/v3/backends/cuda/simt.py"
    )
    report["measurement"] = {
        "samples_per_path": samples, "body_launches_per_graph": graph_launches,
        "graph_replays_per_sample": replays,
        "timing": "CUDA event time per captured GPU operation; graph build, Python binding, and full-step call costs excluded",
        "cache_condition": "same resident input/weight/bias/output buffers, repeated calls",
        "preparation": "allocations and source-weight orientation are setup; no packing, counter binding, descriptor creation, or full-step binding is timed",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "body_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["report_path"] = str(report_path)
    report["report_sha256"] = sha256(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", choices=("005", "006"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--graph-launches", type=int, default=8)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    report = run(args.output_dir, args.card, samples=args.samples,
                 graph_launches=args.graph_launches, replays=args.replays,
                 inventory_path=args.inventory)
    print(json.dumps({"report_path": report["report_path"],
                      "report_sha256": report["report_sha256"],
                      "card": report["card"], "target": report["target"],
                      "case_count": len(report["cases"]),
                      "claim": report["claim"]}, sort_keys=True))


if __name__ == "__main__":
    main()
