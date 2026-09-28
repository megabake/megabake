"""Measure a diagnostic empty-kernel launch floor for V3R-009."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.mb3_launch_floor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_size_t]
    return lib


def _launch(lib: Any, stream: Any, cooperative: bool, grid: int,
            threads: int = 128) -> None:
    status = lib.mb3_launch_floor(int(cooperative), grid, threads,
                                  stream.cuda_stream)
    if status:
        raise RuntimeError(f"empty launch returned CUDA status {status}")


def _measure(torch_module: Any, lib: Any, *, cooperative: bool, grid: int,
             samples: int, launches_per_sample: int,
             graph_replays: int) -> dict[str, Any]:
    stream = torch_module.cuda.Stream(device=torch_module.cuda.current_device())
    for _ in range(launches_per_sample):
        _launch(lib, stream, cooperative, grid)
    stream.synchronize()
    direct_gpu, direct_wall = [], []
    for _ in range(samples):
        begin = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch_module.cuda.stream(stream):
            begin.record(stream)
            for _ in range(launches_per_sample):
                _launch(lib, stream, cooperative, grid)
            end.record(stream)
        end.synchronize()
        direct_wall.append((time.perf_counter() - started) * 1e6 / launches_per_sample)
        direct_gpu.append(begin.elapsed_time(end) * 1000.0 / launches_per_sample)

    graph_result: dict[str, Any]
    try:
        graph = torch_module.cuda.CUDAGraph()
        stream.synchronize()
        with torch_module.cuda.graph(graph, stream=stream):
            for _ in range(launches_per_sample):
                _launch(lib, stream, cooperative, grid)
        stream.synchronize()
        graph_gpu, graph_wall = [], []
        for _ in range(samples):
            begin = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            with torch_module.cuda.stream(stream):
                begin.record(stream)
                for _ in range(graph_replays):
                    graph.replay()
                end.record(stream)
            end.synchronize()
            count = launches_per_sample * graph_replays
            graph_wall.append((time.perf_counter() - started) * 1e6 / count)
            graph_gpu.append(begin.elapsed_time(end) * 1000.0 / count)
        graph_result = {
            "status": "measured",
            "raw_gpu_event_us_per_empty_launch": graph_gpu,
            "median_gpu_event_us_per_empty_launch": statistics.median(graph_gpu),
            "raw_replay_wall_us_per_empty_launch": graph_wall,
            "median_replay_wall_us_per_empty_launch": statistics.median(graph_wall),
        }
    except RuntimeError as error:
        graph_result = {"status": "capture_rejected", "error": str(error)}
    return {
        "kind": "cooperative" if cooperative else "regular",
        "grid_ctas": grid, "threads_per_cta": 128,
        "launches_per_sample": launches_per_sample,
        "samples": samples,
        "raw_direct_gpu_event_us_per_empty_launch": direct_gpu,
        "median_direct_gpu_event_us_per_empty_launch": statistics.median(direct_gpu),
        "raw_direct_wrapper_wall_us_per_empty_launch": direct_wall,
        "median_direct_wrapper_wall_us_per_empty_launch": statistics.median(direct_wall),
        "cuda_graph": graph_result,
    }


def run(library_path: Path, output_path: Path, samples: int,
        launches: int, replays: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("launch floor probe requires CUDA")
    lib = _load(library_path)
    prop = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "task_id": "V3R-009",
        "target": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "visible_sms": int(prop.multi_processor_count),
            "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        },
        "measurement": {
            "empty_kernel": "no memory operations and no arithmetic; one launch only",
            "interpretation": "diagnostic launch floor; not a removable fraction f and not a complete owner coordination cost h",
            "samples": samples, "launches_per_sample": launches,
            "graph_replays_per_sample": replays,
        },
        "cases": [
            _measure(torch, lib, cooperative=False, grid=1, samples=samples,
                     launches_per_sample=launches, graph_replays=replays),
            _measure(torch, lib, cooperative=False, grid=int(prop.multi_processor_count),
                     samples=samples, launches_per_sample=launches,
                     graph_replays=replays),
            _measure(torch, lib, cooperative=True, grid=int(prop.multi_processor_count),
                     samples=samples, launches_per_sample=launches,
                     graph_replays=replays),
        ],
        "artifacts": {
            "source": "src/cuda/v3/launch_floor_probe.cu",
            "source_sha256": _sha(ROOT / "src/cuda/v3/launch_floor_probe.cu"),
            "library": str(library_path), "library_sha256": _sha(library_path),
            "library_bytes": library_path.stat().st_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--launches", type=int, default=128)
    parser.add_argument("--replays", type=int, default=8)
    args = parser.parse_args()
    report = run(args.library, args.output, args.samples, args.launches, args.replays)
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
