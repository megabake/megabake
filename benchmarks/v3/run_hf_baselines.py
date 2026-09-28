#!/usr/bin/env python3
"""Measure the strongest equivalent torch.compile cached-step baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import random
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.profiler

from baseline_report import rejection_reasons, select_best_baseline
from hf_cached_step import (
    HuggingFaceCachedStep,
    assert_step_outputs,
    cuda_driver_version,
    load_hf_model,
    make_step_inputs,
    native_reference_step,
    write_json,
)
from megabake.v3.contracts import StepManifest


MODES = ("default", "reduce-overhead", "max-autotune")


def profile_call(fn, args, trace_path: Path) -> tuple[list[dict], list[str], bool]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profile:
        with torch.inference_mode():
            fn(*args)
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(trace_path))
    summary = []
    for event in profile.key_averages(group_by_input_shape=True):
        summary.append({
            "name": event.key,
            "count": event.count,
            "input_shapes": getattr(event, "input_shapes", None),
            "cpu_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "cpu_self_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            "device_total_us": float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
        })
    names = sorted({event.name for event in profile.events()})
    graph_launch = any("cudagraph" in name.lower() or "cuda_graph_launch" in name.lower() for name in names)
    return summary, names, graph_launch


def verify_output_ownership(fn, args) -> tuple[bool, bool]:
    state_before = args[1].clone()
    token_before = args[0].clone()
    with torch.inference_mode():
        first = fn(*args)
        first_copy = {key: value.clone() for key, value in first.items()}
        args[0].add_(1)
        try:
            second = fn(*args)
        finally:
            args[0].copy_(token_before)
    torch.cuda.synchronize()
    preserved = all(torch.equal(first[key], first_copy[key]) for key in first)
    inputs_unchanged = torch.equal(args[1], state_before)
    differs = not torch.equal(first_copy["logits"], second["logits"])
    return preserved and differs, inputs_unchanged


def profile_target() -> str:
    prop = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    driver = cuda_driver_version()
    return f"{prop.name}; cc={major}.{minor}; visible_sms={prop.multi_processor_count}; total_memory={prop.total_memory}; driver={driver}"


def run_cell(manifest_path: Path, output_root: Path, warmup: int, samples: int, modes: tuple[str, ...]) -> dict:
    manifest = StepManifest.from_json(manifest_path.read_text())
    workload = manifest.workload
    model = load_hf_model(workload.checkpoint_id, manifest.checkpoint_revision)
    inputs = make_step_inputs(
        model,
        workload.position,
        workload.capacity,
        workload.batch_size,
        seed=manifest.fixed_inputs["seed"],
    )
    step = HuggingFaceCachedStep(model, workload.position, workload.capacity).eval()
    args = (inputs.input_ids, inputs.old_cache)
    expected = native_reference_step(model, *args, workload.position)
    target = profile_target()
    cell_id = workload.benchmark_cells[0].cell_id
    cell_dir = output_root / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    candidate_records = []
    callable_by_id = {}

    for mode in modes:
        item = {
            "baseline_id": f"torch.compile/{mode}",
            "mode": mode,
            "cudagraph_requested": mode in {"reduce-overhead", "max-autotune"},
            "compiled": False,
            "equivalent_outputs": False,
            "equivalent_state": False,
            "old_state_unchanged": False,
            "output_ownership": False,
            "graph_break_count": None,
            "complete_call_wall_us": [],
            "gpu_event_us": [],
            "operation_trace": None,
            "unfiltered_trace": False,
            "errors": [],
        }
        started = time.perf_counter()
        try:
            torch._dynamo.reset()
            from torch._dynamo.utils import counters

            counters.clear()
            if mode in {"reduce-overhead", "max-autotune"}:
                torch._inductor.config.triton.cudagraph_trees_generation_cloning = "user_visible"
            compiled = torch.compile(step, mode=mode, fullgraph=True)

            def invoke(*call_args):
                if mode in {"reduce-overhead", "max-autotune"}:
                    torch.compiler.cudagraph_mark_step_begin()
                return compiled(*call_args)

            with torch.inference_mode():
                first = invoke(*args)
            torch.cuda.synchronize()
            item["first_call_setup_ms"] = (time.perf_counter() - started) * 1000.0
            item["compiled"] = True
            item["graph_break_count"] = 0  # fullgraph=True rejects graph breaks at compile time
            item["compiled_graph_count"] = int(counters["stats"].get("unique_graphs", 0))
            errors = assert_step_outputs(expected, first)
            item["equivalent_outputs"] = True
            item["equivalent_state"] = True
            item["max_abs_error"] = errors
            ownership, unchanged = verify_output_ownership(compiled, args)
            item["output_ownership"] = ownership
            item["old_state_unchanged"] = unchanged
            item["recompile_count_after_alternate_token"] = max(
                0, int(counters["stats"].get("unique_graphs", 0)) - item["compiled_graph_count"]
            )
            item["cudagraph_observed"] = False
            for _ in range(warmup):
                with torch.inference_mode():
                    invoke(*args)
            torch.cuda.synchronize()

            trace_path = cell_dir / f"{mode.replace('-', '_')}_trace.json"
            summary, event_names, graph_launch = profile_call(invoke, args, trace_path)
            item["cudagraph_observed"] = graph_launch
            item["operation_summary"] = summary
            item["runtime_event_names"] = event_names
            item["operation_trace"] = str(trace_path)
            item["unfiltered_trace"] = trace_path.is_file()
            item["profiled_call_is_separate_from_timing"] = True
            item["recompile_count"] = item["recompile_count_after_alternate_token"]
            callable_by_id[item["baseline_id"]] = invoke
        except Exception as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            item["first_call_setup_ms"] = (time.perf_counter() - started) * 1000.0
            item.setdefault("cudagraph_observed", False)
        item["rejection_reasons"] = rejection_reasons(item)
        candidate_records.append(item)

    eligible_ids = [
        item["baseline_id"] for item in candidate_records
        if not [reason for reason in item["rejection_reasons"] if reason != "complete-call sample series is empty"]
    ]
    rng = random.Random(f"{cell_id}:{manifest.contract_hash}")
    events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in eligible_ids}
    wall = {name: [] for name in eligible_ids}
    gpu = {name: [] for name in eligible_ids}
    for _ in range(samples):
        block = eligible_ids.copy()
        rng.shuffle(block)
        for name in block:
            start_event, end_event = events[name]
            start = time.perf_counter()
            start_event.record()
            with torch.inference_mode():
                callable_by_id[name](*args)
            end_event.record()
            torch.cuda.synchronize()
            wall[name].append((time.perf_counter() - start) * 1_000_000.0)
            gpu[name].append(start_event.elapsed_time(end_event) * 1000.0)
    for item in candidate_records:
        name = item["baseline_id"]
        if name in wall:
            item["complete_call_wall_us"] = wall[name]
            item["gpu_event_us"] = gpu[name]
        item["median_complete_call_wall_us"] = statistics.median(item["complete_call_wall_us"]) if item["complete_call_wall_us"] else None
        item["rejection_reasons"] = rejection_reasons(item)

    best = select_best_baseline(candidate_records)
    report = {
        "schema_version": 1,
        "task_id": "V3R-003",
        "cell_id": cell_id,
        "manifest": str(manifest_path),
        "manifest_hash": manifest.contract_hash,
        "graph_hash": manifest.graph_hash,
        "checkpoint": workload.checkpoint_id,
        "checkpoint_revision": manifest.checkpoint_revision,
        "target": target,
        "numerical_policy_hash": manifest.numerical_policy.contract_hash,
        "timed_unit": "synchronized complete cached-step call; fixed-state replay",
        "setup_is_separate": True,
        "warmup_calls_per_mode": warmup,
        "randomized_paired_blocks": samples,
        "candidate_modes": candidate_records,
        "best_baseline_id": best["baseline_id"] if best else None,
        "best_baseline_median_complete_call_wall_us": best["median_complete_call_wall_us"] if best else None,
        "claim": "baseline_only; no MegaBake strict artifact or one-grid claim",
    }
    report_path = cell_dir / "baseline_report.json"
    write_json(report_path, report)
    return {"cell_id": cell_id, "report": str(report_path), "best_baseline_id": report["best_baseline_id"], "candidate_count": len(candidate_records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("ART/tasks/V3R-003/baselines"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.warmup < 0 or args.samples <= 0:
        parser.error("warmup must be non-negative and samples positive")
    results = [run_cell(path, args.output_dir, args.warmup, args.samples, tuple(args.modes)) for path in args.manifests]
    print({"results": results})


if __name__ == "__main__":
    main()
