from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/v3"))
from body_budget import _candidate_budget_screen, build_budget


def evidence():
    baseline = {
        "cell_id": "cell-L128", "manifest_hash": "manifest",
        "best_baseline_id": "torch.compile/default",
        "best_baseline_median_complete_call_wall_us": 100.0,
        "randomized_paired_blocks": 2, "setup_is_separate": True,
        "target": "NVIDIA H200 MIG 3g.71gb",
        "candidate_modes": [{
            "baseline_id": "torch.compile/default",
            "equivalent_outputs": True, "equivalent_state": True,
            "old_state_unchanged": True, "output_ownership": True,
            "unfiltered_trace": True, "median_complete_call_wall_us": 100.0,
            "complete_call_wall_us": [99.0, 101.0],
        }],
    }
    inventory = {
        "cell_id": "cell-L128", "manifest_hash": "manifest",
        "hot_shapes": [{
            "call_count": 3, "gemm": {"M": 1, "N": 17, "K": 33},
            "dtypes": {"inputs": ["float16"]},
        }],
        "removable_work": [{"kind": "state_copy"}],
    }
    shape = {"M": 1, "N": 17, "K": 33, "dtype": "float16",
             "weight_strides": [33, 1]}
    simt = {"target": {"device": "NVIDIA H200 MIG 3g.71gb"}, "cases": [{
        "shape": shape,
        "tactics": [{
            "schedule": {"warps_per_cta": 1}, "output_tiles": 2,
            "standalone": {
                "median_gpu_event_us": 1.1, "raw_gpu_event_us": [1.0, 1.2],
                "compiled_code_bytes": 100,
            },
            "owner_entries": [{
                "median_gpu_event_us": 1.0, "raw_gpu_event_us": [1.0, 1.0],
                "grid_ctas": 2, "resources": {"registers_per_thread": 32},
                "compiled_code_bytes": 100,
            }],
        }],
    }]}
    mma = {
        "target": {"device": "NVIDIA H200 MIG 3g.71gb",
                   "compiled_target": "sm_90a"},
        "strict_one_grid_speed_claim": False,
        "cases": [{
            "shape": shape, "call_count": 3,
            "vendor_control": {
                "median_gpu_event_us": 2.0, "raw_gpu_event_us": [2.0, 2.0],
                "profiled_cuda_kernel_names": ["local_control"],
            },
            "tactics": [{
                "schedule": {"warps_per_cta": 1},
                "work_estimate": {"padded_work_ratio": 16.0},
                "standalone": {
                    "median_gpu_event_us": 7.0, "raw_gpu_event_us": [7.0, 7.0],
                },
                "owner_entries": [{
                    "median_gpu_event_us": 6.0, "raw_gpu_event_us": [5.0, 7.0],
                    "grid_ctas": 2, "resources": {"registers_per_thread": 32},
                }],
            }],
        }],
    }
    return {"L128": baseline}, {"L128": inventory}, simt, mma


def test_v3r009_keeps_simt_and_rejects_unpaid_wmma_deficit():
    report = build_budget(*evidence())
    row = report["hot_shape_comparisons"][0]
    assert row["wmma_over_local_control"] == 3.0
    assert row["wmma_minimum_per_region_deficit_to_local_control_us"] == 4.0
    assert report["decision"]["wmma_for_measured_batch_one_hot_shapes"] == "no_go_for_more_tile_search"
    assert report["decision"]["simt_for_measured_hot_shapes"] == "retain_as_first_strict_body_candidate"
    assert report["decision"]["strict_complete_step"] == "not_measured"
    assert report["complete_call_cells"][0]["unmeasured"]["persistent_entry_coordination_cost"] == "unknown"


def test_v3r009_rejects_missing_samples_and_strict_speed_relabel():
    baseline, inventory, simt, mma = evidence()
    mma["cases"][0]["vendor_control"]["raw_gpu_event_us"] = []
    with pytest.raises(ValueError, match="raw samples"):
        build_budget(baseline, inventory, simt, mma)

    baseline, inventory, simt, mma = evidence()
    mma["strict_one_grid_speed_claim"] = True
    with pytest.raises(ValueError, match="cannot carry a strict one-grid claim"):
        build_budget(baseline, inventory, simt, mma)


def test_v3r009_trace_gap_is_only_a_ceiling_and_owner_floor_is_lower_bound(tmp_path):
    baseline, inventory, simt, mma = evidence()
    trace = tmp_path / "selected_trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "k0", "ts": 10,
         "dur": 2, "args": {"graph id": 11, "stream": 7}},
        {"ph": "X", "cat": "kernel", "name": "k1", "ts": 14,
         "dur": 2, "args": {"graph id": 11, "stream": 7}},
    ]}))
    floor = {"cases": [{"kind": "cooperative", "grid_ctas": 60,
                         "cuda_graph": {"status": "measured",
                                        "median_gpu_event_us_per_empty_launch": 1.5,
                                        "raw_gpu_event_us_per_empty_launch": [1.4, 1.6]}}]}
    report = build_budget(
        baseline, inventory, simt, mma,
        trace_paths={"L128": trace}, launch_floor_report=floor,
    )
    cell = report["complete_call_cells"][0]
    assert cell["selected_trace_diagnostic"]["inter_event_idle_gap_us"] == 2.0
    assert cell["measured_budget_bounds"]["f_optimistic_trace_ceiling_fraction_of_complete_call"] == 0.02
    assert cell["measured_budget_bounds"]["h_measured_lower_bound_fraction_of_complete_call"] == 0.015
    scenario = report["hot_shape_comparisons"][0]["optimistic_break_even_scenarios"]["cell-L128"]
    assert scenario["passes_even_optimistic_screen"] is False
    simt_screen = report["hot_shape_comparisons"][0]["candidate_budget_screens"][
        "V3R-006 SIMT owner vs same-buffer graph addmm"]["cell-L128"]
    assert simt_screen["passes_optimistic_bounds_only"] is True
    assert simt_screen["minimum_f_required_for_this_delta"] < 0


def test_v3r009_negative_body_delta_can_pay_owner_floor_when_f_is_smaller_than_h():
    cell = {"measured_budget_bounds": {
        "f_optimistic_trace_ceiling_fraction_of_complete_call": 0.01,
        "h_measured_lower_bound_fraction_of_complete_call": 0.015,
    }}
    screen = _candidate_budget_screen(-0.02, cell)
    assert screen["maximum_delta_allowed_at_optimistic_bounds"] < 0
    assert screen["passes_optimistic_bounds_only"] is True


def test_v3r009_provider_comparison_includes_fastest_same_shape_compile_control():
    baseline, inventory, simt, mma = evidence()
    mma["cases"][0]["torch_compile_controls"] = [{
        "mode": "default", "median_gpu_event_us": 1.5,
        "raw_gpu_event_us": [1.4, 1.6],
        "median_complete_callable_wall_us": 2.0,
        "raw_complete_callable_wall_us": [1.9, 2.1],
    }]
    provider = {"cases": [{
        "shape": [1, 17, 33], "resources": {"registers_per_thread": 40},
        "standalone": {"median_gpu_event_us": 0.9, "raw_gpu_event_us": [0.8, 1.0]},
        "owner_entry": {"median_gpu_event_us": 1.0, "raw_gpu_event_us": [0.9, 1.1]},
        "owner_rejection": None,
        "torch_addmm_control": {"median_gpu_event_us": 2.0, "raw_gpu_event_us": [1.9, 2.1]},
    }]}
    report = build_budget(baseline, inventory, simt, mma, provider_report=provider)
    row = report["provider_body_comparisons"][0]
    assert row["fastest_direct_torch_compile_control"]["mode"] == "default"
    assert row["fastest_direct_torch_compile_control"]["raw_gpu_event_us"] == [1.4, 1.6]
    assert row["owner_to_fastest_direct_torch_compile_gpu_event_ratio"] == pytest.approx(2 / 3)


def test_v3r009_rejects_trace_without_device_activity(tmp_path):
    from body_budget import trace_diagnostic

    trace = tmp_path / "empty_trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    with pytest.raises(ValueError, match="no GPU kernel/copy activities"):
        trace_diagnostic(trace)
