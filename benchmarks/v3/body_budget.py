"""Build the first target's conservative V3R-009 body-quality worksheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _samples(record: dict[str, Any], key: str, label: str) -> list[float]:
    values = [float(value) for value in record.get(key, ())]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{label} requires finite positive raw samples")
    return values


def _median(record: dict[str, Any], key: str, label: str) -> tuple[float, list[float]]:
    values = _samples(record, key, label)
    median = statistics.median(values)
    reported = record.get("median_gpu_event_us", record.get("median_complete_call_wall_us"))
    if reported is not None and not math.isclose(float(reported), median, rel_tol=1e-6):
        raise ValueError(f"{label} median does not match its raw samples")
    return median, values


def _shape_key(case: dict[str, Any]) -> tuple[int, int, int, str]:
    shape = case["shape"]
    return int(shape["M"]), int(shape["N"]), int(shape["K"]), str(shape["dtype"])


def _best_owner(case: dict[str, Any]) -> dict[str, Any]:
    entries = [entry for tactic in case["tactics"] for entry in tactic["owner_entries"]]
    if not entries:
        raise ValueError(f"{case.get('origin_fx')} has no cooperative owner measurement")
    return min(entries, key=lambda entry: float(entry["median_gpu_event_us"]))


def trace_diagnostic(path: Path) -> dict[str, Any]:
    """Summarize one selected CUDA-graph replay without treating duration sums as latency."""
    data = _read(path)
    events = [event for event in data.get("traceEvents", ())
              if event.get("ph") == "X"
              and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
              and event.get("dur", 0) > 0]
    by_graph: dict[Any, list[dict[str, Any]]] = {}
    for event in events:
        graph_id = event.get("args", {}).get("graph id")
        if graph_id not in (None, 0):
            by_graph.setdefault(graph_id, []).append(event)
    if by_graph:
        graph_id, events = max(
            by_graph.items(),
            key=lambda item: sum(event.get("cat") == "kernel" for event in item[1]),
        )
    else:
        graph_id = None
    if not events:
        raise ValueError(f"{path} has no GPU kernel/copy activities")
    intervals = sorted((float(event["ts"]),
                        float(event["ts"]) + float(event["dur"]))
                       for event in events)
    busy_union = 0.0
    union_end = intervals[0][0]
    for start, end in intervals:
        busy_union += max(0.0, end - max(start, union_end))
        union_end = max(union_end, end)
    span = intervals[-1][1] - intervals[0][0]
    names: dict[str, dict[str, float | int]] = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        row = names.setdefault(event["name"], {"count": 0, "duration_sum_us": 0.0})
        row["count"] = int(row["count"]) + 1
        row["duration_sum_us"] = float(row["duration_sum_us"]) + float(event["dur"])
    top_kernels = sorted(names.items(), key=lambda item: -float(item[1]["duration_sum_us"]))[:16]
    return {
        "trace_path": str(path), "trace_sha256": sha256(path),
        "trace_name": data.get("traceName"), "cuda_graph_id": graph_id,
        "gpu_activity_count": len(events),
        "kernel_count": sum(event.get("cat") == "kernel" for event in events),
        "stream_ids": sorted({event.get("args", {}).get("stream") for event in events}),
        "gpu_activity_duration_sum_us_diagnostic_only": sum(float(e["dur"]) for e in events),
        "kernel_duration_sum_us_diagnostic_only": sum(
            float(e["dur"]) for e in events if e.get("cat") == "kernel"
        ),
        "first_to_last_gpu_activity_span_us": span,
        "union_busy_interval_us": busy_union,
        "inter_event_idle_gap_us": max(0.0, span - busy_union),
        "inter_event_idle_fraction_of_span": max(0.0, span - busy_union) / span,
        "top_kernel_names_by_duration_sum": [
            {"name": name, **row} for name, row in top_kernels
        ],
        "interpretation": "One profiled replay's GPU event timeline; the activity duration sum is not complete-call latency and the idle gap is an optimistic boundary-removal ceiling, not measured savings.",
    }


def _empty_owner_floor(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    row = next((case for case in report.get("cases", ())
                if case.get("kind") == "cooperative"), None)
    if row is None:
        return None
    graph = row.get("cuda_graph", {})
    if graph.get("status") != "measured":
        return {"status": graph.get("status", "unavailable"), "kind": "cooperative"}
    return {
        "status": "measured_lower_bound_only",
        "grid_ctas": row["grid_ctas"],
        "gpu_event_us_per_empty_launch": graph["median_gpu_event_us_per_empty_launch"],
        "raw_gpu_event_us_per_empty_launch": graph["raw_gpu_event_us_per_empty_launch"],
        "source": report.get("artifacts", {}).get("library"),
        "interpretation": "Measured no-op cooperative launch floor; real owner initialization, scheduling, synchronization, body work, binding and return can only add cost.",
    }


def _candidate_budget_screen(delta: float, cell: dict[str, Any]) -> dict[str, Any] | None:
    bounds = cell["measured_budget_bounds"]
    f_ceiling = bounds["f_optimistic_trace_ceiling_fraction_of_complete_call"]
    h_floor = bounds["h_measured_lower_bound_fraction_of_complete_call"]
    if f_ceiling is None or h_floor is None:
        return None
    max_delta = (f_ceiling - h_floor) / (1.0 - f_ceiling)
    required_f = (delta + h_floor) / (1.0 + delta)
    return {
        "candidate_body_delta": delta,
        "r_assumption": 0.0,
        "h_lower_bound": h_floor,
        "f_optimistic_ceiling": f_ceiling,
        "maximum_delta_allowed_at_optimistic_bounds": max_delta,
        "minimum_f_required_for_this_delta": required_f,
        "passes_optimistic_bounds_only": delta < max_delta,
        "qualification": "Diagnostic body-rate screen using a local callable comparator; f is an upper ceiling and h is a lower bound, and neither is an exact complete-step attribution.",
    }


def build_budget(
    baseline_reports: dict[str, dict[str, Any]],
    inventories: dict[str, dict[str, Any]],
    simt_report: dict[str, Any],
    mma_report: dict[str, Any],
    *,
    trace_paths: dict[str, Path] | None = None,
    launch_floor_report: dict[str, Any] | None = None,
    provider_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mma_report.get("strict_one_grid_speed_claim") is not False:
        raise ValueError("isolated body evidence cannot carry a strict one-grid claim")
    if not baseline_reports or not inventories:
        raise ValueError("at least one matched baseline and shape inventory are required")
    target = mma_report["target"]
    if target["compiled_target"] != "sm_90a":
        raise ValueError("worksheet requires the measured sm_90a target")
    if simt_report.get("target", {}).get("device") != target["device"]:
        raise ValueError("SIMT and WMMA body reports must use the same target")
    if any(target["device"] not in baseline.get("target", "")
           for baseline in baseline_reports.values()):
        raise ValueError("complete-call baseline and body reports must use the same device")

    cells = []
    floor = _empty_owner_floor(launch_floor_report)
    for context, baseline in sorted(baseline_reports.items()):
        inventory = inventories[context]
        if baseline["manifest_hash"] != inventory["manifest_hash"]:
            raise ValueError(f"{context} baseline and inventory manifest hashes differ")
        best_id = baseline["best_baseline_id"]
        selected = next((mode for mode in baseline["candidate_modes"]
                         if mode["baseline_id"] == best_id), None)
        if selected is None or not all(selected.get(field) is True for field in (
            "equivalent_outputs", "equivalent_state", "old_state_unchanged",
            "output_ownership", "unfiltered_trace",
        )):
            raise ValueError(f"{context} selected baseline is not validated equivalent")
        baseline_median, baseline_samples = _median(
            selected, "complete_call_wall_us", f"{context} complete-call baseline"
        )
        if len(baseline_samples) != baseline.get("randomized_paired_blocks"):
            raise ValueError(f"{context} baseline sample count differs from paired blocks")

        call_count = sum(int(record.get("call_count", 0))
                         for record in inventory["hot_shapes"]
                         if "gemm" in record)
        state_writes = sum(1 for record in inventory.get("removable_work", ())
                           if record.get("kind") == "state_copy")
        trace = (trace_diagnostic(trace_paths[context])
                 if trace_paths and context in trace_paths else None)
        idle_ceiling_fraction = (
            trace["inter_event_idle_gap_us"] / baseline_median if trace else None
        )
        h_floor_fraction = (
            floor["gpu_event_us_per_empty_launch"] / baseline_median
            if floor and floor.get("status") == "measured_lower_bound_only" else None
        )
        cells.append({
            "cell_id": baseline["cell_id"],
            "selected_baseline_id": best_id,
            "baseline_mode_counterfactuals": [
                {
                    "baseline_id": mode["baseline_id"],
                    "mode": mode.get("mode"),
                    "compiled": mode.get("compiled"),
                    "equivalent_outputs": mode.get("equivalent_outputs"),
                    "equivalent_state": mode.get("equivalent_state"),
                    "output_ownership": mode.get("output_ownership"),
                    "cudagraph_observed": mode.get("cudagraph_observed"),
                    "median_complete_call_wall_us": mode.get("median_complete_call_wall_us"),
                    "raw_complete_call_wall_us": mode.get("complete_call_wall_us", []),
                    "selected": mode["baseline_id"] == best_id,
                }
                for mode in baseline["candidate_modes"]
            ],
            "complete_call_wall_median_us": baseline_median,
            "complete_call_wall_raw_samples_us": baseline_samples,
            "paired_blocks": len(baseline_samples),
            "setup_is_separate": baseline["setup_is_separate"],
            "hot_contraction_call_count": call_count,
            "state_copy_origins_without_direct_profile_events": state_writes,
            "selected_trace_diagnostic": trace,
            "measured_budget_bounds": {
                "f_actual": "unknown; selected kernel-to-FX and exact removable-origin attribution are unavailable",
                "f_optimistic_trace_ceiling_fraction_of_complete_call": idle_ceiling_fraction,
                "f_ceiling_is_not_savings": True,
                "h_actual": "unknown; no complete composed owner entry exists",
                "h_measured_lower_bound_fraction_of_complete_call": h_floor_fraction,
                "r_measured": 0.0,
                "r_status": "no paid fusion/layout/packing candidate was composed; zero is the conservative scenario, not evidence that fusion saves nothing",
            },
            "unmeasured": {
                "selected_kernel_to_fx_mapping": "unknown",
                "per_origin_critical_path_us": "unknown",
                "exact_removable_launch_handoff_fraction": "unknown within the trace-derived optimistic ceiling",
                "additional_fusion_savings": "unknown",
                "persistent_entry_coordination_cost": "unknown",
                "physical_hbm_l2_bytes": "unknown",
            },
        })

    hot_keys = {
        (int(record["gemm"]["M"]), int(record["gemm"]["N"]),
         int(record["gemm"]["K"]), str(record["dtypes"]["inputs"][0]))
        for inventory in inventories.values()
        for record in inventory["hot_shapes"] if "gemm" in record
    }
    simt_by_shape = {_shape_key(case): case for case in simt_report["cases"]}
    provider_by_shape = {
        tuple(int(dimension) for dimension in case["shape"]): case
        for case in (provider_report or {}).get("cases", ())
    }
    hot = [case for case in mma_report["cases"] if _shape_key(case) in hot_keys]
    comparisons = []
    for case in hot:
        key = _shape_key(case)
        simt = simt_by_shape.get(key)
        if simt is None:
            raise ValueError(f"no V3R-006 SIMT control for hot shape {key}")
        vendor_median, vendor_samples = _median(
            case["vendor_control"], "raw_gpu_event_us", f"{key} local addmm control"
        )
        mma_points = []
        for tactic in case["tactics"]:
            owner = min(tactic["owner_entries"],
                        key=lambda entry: float(entry["median_gpu_event_us"]))
            owner_median, owner_samples = _median(
                owner, "raw_gpu_event_us", f"{key} WMMA owner"
            )
            standalone_median, standalone_samples = _median(
                tactic["standalone"], "raw_gpu_event_us", f"{key} WMMA standalone"
            )
            mma_points.append({
                "schedule": tactic["schedule"],
                "standalone_median_us": standalone_median,
                "standalone_samples": len(standalone_samples),
                "standalone_raw_gpu_event_us": standalone_samples,
                "best_owner_median_us": owner_median,
                "owner_samples": len(owner_samples),
                "owner_raw_gpu_event_us": owner_samples,
                "owner_grid_ctas": owner["grid_ctas"],
                "owner_resources": owner["resources"],
                "work_estimate": tactic["work_estimate"],
            })
        best_mma = min(mma_points, key=lambda point: point["best_owner_median_us"])
        simt_owner = _best_owner(simt)
        simt_median, simt_samples = _median(
            simt_owner, "raw_gpu_event_us", f"{key} SIMT owner"
        )
        simt_points = []
        for tactic in simt["tactics"]:
            owner = min(tactic["owner_entries"],
                        key=lambda entry: float(entry["median_gpu_event_us"]))
            standalone_median, standalone_samples = _median(
                tactic["standalone"], "raw_gpu_event_us",
                f"{key} SIMT standalone",
            )
            owner_median, owner_samples = _median(
                owner, "raw_gpu_event_us", f"{key} SIMT owner schedule",
            )
            simt_points.append({
                "schedule": tactic["schedule"],
                "output_tiles": tactic["output_tiles"],
                "standalone_median_us": standalone_median,
                "standalone_raw_gpu_event_us": standalone_samples,
                "owner_median_us": owner_median,
                "owner_raw_gpu_event_us": owner_samples,
                "owner_grid_ctas": owner["grid_ctas"],
                "owner_resources": owner["resources"],
                "compiled_code_bytes": owner["compiled_code_bytes"],
            })
        best_simt_point = min(simt_points, key=lambda point: point["owner_median_us"])
        candidate_deltas = {
            "V3R-006 SIMT owner vs same-buffer graph addmm":
                simt_median / vendor_median - 1.0,
            "V3R-007 WMMA owner vs same-buffer graph addmm":
                best_mma["best_owner_median_us"] / vendor_median - 1.0,
        }
        provider_case = provider_by_shape.get(key[:3])
        provider_owner = provider_case.get("owner_entry") if provider_case else None
        if provider_owner is not None:
            provider_control_median, _ = _median(
                provider_case["torch_addmm_control"], "raw_gpu_event_us",
                f"{key} CUTLASS same-buffer direct addmm control",
            )
            provider_owner_median, _ = _median(
                provider_owner, "raw_gpu_event_us",
                f"{key} CUTLASS cooperative owner",
            )
            candidate_deltas[
                "V3R-008 CUTLASS cooperative owner vs same-buffer direct addmm"
            ] = provider_owner_median / provider_control_median - 1.0
        candidate_budget_screens = {
            label: {
                cell["cell_id"]: _candidate_budget_screen(delta, cell)
                for cell in cells
            }
            for label, delta in candidate_deltas.items()
        }
        compile_rows = case.get("torch_compile_controls", ())
        direct_compile_best = (min(
            compile_rows, key=lambda point: point["median_gpu_event_us"]
        ) if compile_rows else None)
        direct_body = case.get("selected_body_direct_invocation")
        direct_vendor = case.get("vendor_control", {}).get("direct_invocation")
        optimistic_budget = {}
        delta = max(0.0, best_mma["best_owner_median_us"] / vendor_median - 1.0)
        for cell in cells:
            bounds = cell["measured_budget_bounds"]
            f_ceiling = bounds["f_optimistic_trace_ceiling_fraction_of_complete_call"]
            h_floor_fraction = bounds["h_measured_lower_bound_fraction_of_complete_call"]
            if f_ceiling is None or h_floor_fraction is None:
                continue
            required_f = (delta + h_floor_fraction) / (1.0 + delta)
            allowed_delta = max(
                0.0, (f_ceiling - h_floor_fraction) / (1.0 - f_ceiling)
            ) if f_ceiling < 1.0 else math.inf
            optimistic_budget[cell["cell_id"]] = {
                "delta_against_same_buffer_graph_local_addmm": delta,
                "r_assumption": 0.0,
                "h_assumption": h_floor_fraction,
                "f_optimistic_trace_ceiling": f_ceiling,
                "maximum_delta_allowed_under_optimistic_ceiling": allowed_delta,
                "minimum_f_required_for_this_delta": required_f,
                "passes_even_optimistic_screen": required_f < f_ceiling,
                "qualification": "Diagnostic body-deficit scenario only; local addmm is not the selected complete-step kernel map.",
            }
        comparisons.append({
            "shape": {"M": key[0], "N": key[1], "K": key[2], "dtype": key[3]},
            "call_count": case["call_count"],
            "weight_strides": case["shape"]["weight_strides"],
            "local_torch_addmm_control": {
                "median_us": vendor_median, "samples": len(vendor_samples),
                "raw_gpu_event_us": vendor_samples,
                "kernel_names": case["vendor_control"]["profiled_cuda_kernel_names"],
            },
            "best_simt_owner": {
                "median_us": simt_median, "samples": len(simt_samples),
                "raw_gpu_event_us": simt_samples,
                "grid_ctas": simt_owner["grid_ctas"],
                "schedule": best_simt_point["schedule"],
                "resources": simt_owner["resources"],
                "compiled_code_bytes": simt_owner["compiled_code_bytes"],
            },
            "best_wmma_owner": best_mma,
            "wmma_over_local_control": best_mma["best_owner_median_us"] / vendor_median,
            "wmma_over_simt": best_mma["best_owner_median_us"] / simt_median,
            "wmma_minimum_per_region_deficit_to_local_control_us": max(
                0.0, best_mma["best_owner_median_us"] - vendor_median
            ),
            "direct_callable_comparison": {
                "selected_owner_body": direct_body,
                "same_buffer_eager_addmm": direct_vendor,
                "torch_compile_modes": compile_rows,
                "fastest_direct_torch_compile_mode": (
                    direct_compile_best.get("mode") if direct_compile_best else None
                ),
                "fastest_direct_torch_compile_gpu_event_us": (
                    direct_compile_best.get("median_gpu_event_us")
                    if direct_compile_best else None
                ),
                "selected_owner_to_fastest_direct_torch_compile_gpu_event_ratio": (
                    direct_body["median_gpu_event_us"] /
                    direct_compile_best["median_gpu_event_us"]
                    if direct_body and direct_compile_best else None
                ),
                "qualification": "One direct local contraction invocation on the same inputs; direct compiled result allocation and dispatch are included. This does not establish selected whole-step kernel-to-FX attribution or complete-step performance.",
            },
            "optimistic_break_even_scenarios": optimistic_budget,
            "candidate_budget_screens": candidate_budget_screens,
            "all_simt_schedule_points": simt_points,
            "all_wmma_schedule_points": mma_points,
        })

    if not comparisons:
        raise ValueError("the worksheet requires at least one inventoried hot shape")
    mma_loses = all(row["wmma_over_local_control"] > 1.0 for row in comparisons)
    simt_wins = all(row["best_simt_owner"]["median_us"] <
                    row["local_torch_addmm_control"]["median_us"]
                    for row in comparisons)
    provider_comparisons = []
    mma_cases_by_shape = {_shape_key(case): case for case in mma_report["cases"]}
    if provider_report:
        for case in provider_report.get("cases", ()):
            control = case.get("torch_addmm_control", {})
            provider_shape = tuple(int(dimension) for dimension in case["shape"])
            mma_case = mma_cases_by_shape.get((*provider_shape, "float16"))
            compile_rows = mma_case.get("torch_compile_controls", ()) if mma_case else ()
            fastest_compile = (min(
                compile_rows, key=lambda point: point["median_gpu_event_us"]
            ) if compile_rows else None)
            cutlass_owner = case.get("owner_entry")
            provider_comparisons.append({
                "shape": case.get("shape"),
                "resources": case.get("resources"),
                "torch_addmm_control": control,
                "standalone": case.get("standalone"),
                "owner_entry": case.get("owner_entry"),
                "owner_rejection": case.get("owner_rejection"),
                "standalone_to_control_gpu_event_ratio": (
                    case["standalone"]["median_gpu_event_us"] /
                    control["median_gpu_event_us"] if control else None
                ),
                "owner_to_control_gpu_event_ratio": (
                    case["owner_entry"]["median_gpu_event_us"] /
                    control["median_gpu_event_us"]
                    if case.get("owner_entry") and control else None
                ),
                "fastest_direct_torch_compile_control": (
                    {
                        "mode": fastest_compile["mode"],
                        "median_gpu_event_us": fastest_compile["median_gpu_event_us"],
                        "raw_gpu_event_us": fastest_compile["raw_gpu_event_us"],
                        "median_complete_callable_wall_us": fastest_compile[
                            "median_complete_callable_wall_us"
                        ],
                        "raw_complete_callable_wall_us": fastest_compile[
                            "raw_complete_callable_wall_us"
                        ],
                    }
                    if fastest_compile else None
                ),
                "owner_to_fastest_direct_torch_compile_gpu_event_ratio": (
                    cutlass_owner["median_gpu_event_us"] /
                    fastest_compile["median_gpu_event_us"]
                    if cutlass_owner and fastest_compile else None
                ),
                "torch_compile_comparison_qualification": "Same-shape one-operation control; torch.compile returns a fresh output while the CUTLASS entry writes caller-provided output, so this is diagnostic and not ownership-equivalent.",
                "measurement_scope": "direct same-buffer operation calls; not complete-step or cross-provider entry composition",
            })
    provider_owner_rows = [row for row in provider_comparisons if row["owner_entry"]]
    candidate_assessment = [
        {
            "candidate": "V3R-006 K-parallel SIMT owner",
            "status": "retain_as_first_strict_body_candidate",
            "evidence": "Wins the matched local addmm graph control on all inventoried hot shapes; lean full-step integration and selected-path delta remain unmeasured.",
        },
        {
            "candidate": "V3R-007 output-major WMMA batch-one family",
            "status": "no_go_for_more_tile_search_under_current_dataflow",
            "evidence": "All measured owner points lose to local addmm and pad useful batch work by at least 16x; optimistic trace-gap budget does not pay those measured body deficits when r=0.",
        },
        {
            "candidate": "V3R-008 CUTLASS 3.8 callable body",
            "status": "retain_for_resident_shape_subset",
            "evidence": f"{len(provider_owner_rows)} shapes have a correct cooperative entry whose whole tile grid fits measured residency; larger N is rejected by actual entry residency. Direct call is not a complete entry.",
        },
        {
            "candidate": "paid fusion or handoff reduction",
            "status": "preserve_as_plausible_unmeasured_candidate",
            "evidence": "No fusion candidate was composed, so r stays zero in the conservative screen; the trace does not justify a rigid per-operation veto.",
        },
        {
            "candidate": "stable weight packing/layout preparation",
            "status": "not_measured_no_promotion",
            "evidence": "All current body probes read the declared contiguous weight layout directly; no packing benefit or preparation cost was measured.",
        },
        {
            "candidate": "split-K WMMA",
            "status": "screened_out_unmeasured",
            "evidence": "Would require partial-output scratch, a cooperative grid rendezvous and final reduction; current unsplit batch-one WMMA is far outside the optimistic body-quality budget.",
        },
    ]
    return {
        "schema_version": 1,
        "task_id": "V3R-009",
        "target": target,
        "body_measurement_scope": "same-buffer isolated CUDA operation and cooperative owner entry; not complete-call latency",
        "complete_call_cells": cells,
        "hot_shape_comparisons": comparisons,
        "provider_body_comparisons": provider_comparisons,
        "candidate_assessment": candidate_assessment,
        "launch_floor": floor,
        "screening_model": {
            "equation": "strict win requires delta < (f + r - h) / (1 - f)",
            "delta": "measured for isolated WMMA owner vs local same-buffer addmm; selected kernel-to-FX mapping remains unavailable",
            "f": "exact fraction unknown; per selected CUDA graph trace, inter-event idle gap / complete-call median is an optimistic ceiling diagnostic",
            "r": "no paid fusion/layout/packing candidate was composed; use r=0 for the conservative scenario while leaving plausible fusion candidates open",
            "h": "actual complete-entry coordination unknown; measured empty cooperative launch is a lower bound only",
            "interpretation": "Trace idle gaps are an optimistic opportunity bound, not savings. No kernel duration sum is used as full-step latency; no per-origin amount or HBM traffic is inferred.",
        },
        "decision": {
            "wmma_for_measured_batch_one_hot_shapes": (
                "no_go_for_more_tile_search" if mma_loses else "retain_for_next_entry_trial"
            ),
            "simt_for_measured_hot_shapes": "retain_as_first_strict_body_candidate" if simt_wins else "revise_body",
            "strict_complete_step": "not_measured",
            "reason": (
                "Every measured WMMA owner point loses to the same-buffer local Torch graph control; "
                "the batch-one output-major path performs at least 16x padded MAC work. "
                "SIMT remains the measured body candidate, and CUTLASS 3.8 is retained only for shapes whose full tile grid fits the measured owner residency. Full-step savings and composition remain unmeasured."
                if mma_loses and simt_wins else
                "The body measurements do not justify a strict full-step conclusion."
            ),
            "next_experiment": "Use the retained SIMT tactic in a thin full-step owner entry; revisit WMMA only with a less padded batch/output mapping or a measured paid fusion opportunity.",
        },
        "raw_sample_sources": {
            "mma": "ART/tasks/V3R-007/body_report.json",
            "simt": "ART/tasks/V3R-006/body_report.json",
            "cutlass_provider": "ART/tasks/V3R-008/provider_report.json",
            "launch_floor": "ART/tasks/V3R-009/launch_floor.json",
            "selected_traces": {
                "L128": "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L128/max_autotune_trace.json",
                "L2048": "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L2048/reduce_overhead_trace.json",
            },
            "complete_step_baselines": [
                "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L128/baseline_report.json",
                "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L2048/baseline_report.json",
            ],
        },
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "ART/tasks/V3R-009/body_budget.json")
    args = parser.parse_args()
    contexts = ("L128", "L2048")
    baselines = {
        context: _read(ROOT / f"ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-{context}/baseline_report.json")
        for context in contexts
    }
    inventories = {
        context: _read(ROOT / f"ART/tasks/V3R-004/inventory/smollm2-135m-fp16-b1-{context}/inventory.json")
        for context in contexts
    }
    report = build_budget(
        baselines, inventories,
        _read(ROOT / "ART/tasks/V3R-006/body_report.json"),
        _read(ROOT / "ART/tasks/V3R-007/body_report.json"),
        trace_paths={
            "L128": ROOT / "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L128/max_autotune_trace.json",
            "L2048": ROOT / "ART/tasks/V3R-003/baselines/smollm2-135m-fp16-b1-L2048/reduce_overhead_trace.json",
        },
        launch_floor_report=(
            _read(ROOT / "ART/tasks/V3R-009/launch_floor.json")
            if (ROOT / "ART/tasks/V3R-009/launch_floor.json").exists() else None
        ),
        provider_report=(
            _read(ROOT / "ART/tasks/V3R-008/provider_report.json")
            if (ROOT / "ART/tasks/V3R-008/provider_report.json").exists() else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output), "sha256": sha256(args.output),
                      "hot_shapes": len(report["hot_shape_comparisons"]),
                      "decision": report["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
