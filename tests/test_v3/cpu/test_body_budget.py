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
        "cutlass_4_5_2_cuda13": {
            "status": "correct_standalone_and_owner",
            "active_store_canaries_intact": True,
            "standalone": {"median_gpu_event_us": 0.8,
                           "raw_gpu_event_us": [0.7, 0.9]},
            "owner_entry": {"median_gpu_event_us": 0.9,
                            "raw_gpu_event_us": [0.8, 1.0]},
        },
        "cutlass_cute_collective_4_5_2_cuda13": {
            "status": "correct_standalone_and_owner",
            "active_store_canaries_intact": True,
            "standalone": {"median_gpu_event_us": 0.85,
                           "raw_gpu_event_us": [0.8, 0.9]},
            "owner_entry": {"median_gpu_event_us": 0.95,
                            "raw_gpu_event_us": [0.9, 1.0]},
        },
        "torch_addmm_control": {"median_gpu_event_us": 2.0, "raw_gpu_event_us": [1.9, 2.1]},
        "cublaslt_baseline": {
            "shape": [1, 17, 33], "dtype": "float16", "algorithm_id": 66,
            "measurement": {
                "median_gpu_event_us": 0.75, "raw_gpu_event_us": [0.7, 0.8],
            },
        },
        "cublasdx": {
            "status": "rejected_candidate", "status_code": 9,
            "reason": "M=1 is not divisible by tile_m=16",
        },
    }]}
    report = build_budget(baseline, inventory, simt, mma, provider_report=provider)
    row = report["provider_body_comparisons"][0]
    assert row["fastest_direct_torch_compile_control"]["mode"] == "default"
    assert row["fastest_direct_torch_compile_control"]["raw_gpu_event_us"] == [1.4, 1.6]
    assert row["owner_to_fastest_direct_torch_compile_gpu_event_ratio"] == pytest.approx(2 / 3)
    assert row["owner_to_cublaslt_gpu_event_ratio"] == pytest.approx(4 / 3)
    assert row["cutlass_4_5_2_owner_to_cublaslt_gpu_event_ratio"] == pytest.approx(1.2)
    assert row["cutlass_cute_collective_owner_to_cublaslt_gpu_event_ratio"] == pytest.approx(0.95 / 0.75)
    assert row["cublasdx_hot_shape"]["status_code"] == 9
    cutlass_decision = next(
        item for item in report["candidate_assessment"]
        if item["candidate"] == "V3R-008 CUTLASS 3.8 callable body"
    )
    assert cutlass_decision["status"] == "no_consistent_body_win_over_direct_cublasLt"
    cutlass452_decision = next(
        item for item in report["candidate_assessment"]
        if item["candidate"] == "V3R-008 CUTLASS 4.5.2 callable body under CUDA 13"
    )
    assert cutlass452_decision["status"] == "no_consistent_body_win_over_direct_cublasLt"
    cute_decision = next(
        item for item in report["candidate_assessment"]
        if item["candidate"] == "V3R-008 CUTLASS 4.5.2 CuTe collective under CUDA 13"
    )
    assert cute_decision["status"] == "no_consistent_body_win_over_direct_cublasLt"


def test_v3r009_rejects_bad_cuda13_cutlass_samples():
    baseline, inventory, simt, mma = evidence()
    provider = {"cases": [{
        "shape": [1, 17, 33], "owner_entry": None,
        "cutlass_4_5_2_cuda13": {
            "active_store_canaries_intact": True,
            "standalone": {"median_gpu_event_us": 1.0,
                           "raw_gpu_event_us": []},
            "owner_entry": None,
        },
    }]}
    with pytest.raises(ValueError, match="raw samples"):
        build_budget(baseline, inventory, simt, mma, provider_report=provider)


def test_v3r009_rejects_bad_cuda13_cute_collective_samples():
    baseline, inventory, simt, mma = evidence()
    provider = {"cases": [{
        "shape": [1, 17, 33], "owner_entry": None,
        "cutlass_cute_collective_4_5_2_cuda13": {
            "active_store_canaries_intact": True,
            "standalone": {"median_gpu_event_us": 1.0,
                           "raw_gpu_event_us": []},
            "owner_entry": None,
        },
    }]}
    with pytest.raises(ValueError, match="raw samples"):
        build_budget(baseline, inventory, simt, mma, provider_report=provider)


def test_v3r009_records_cublasdx_smoke_without_promoting_it_to_a_hot_shape():
    baseline, inventory, simt, mma = evidence()
    provider = {"cases": [], "cublasdx": {"positive_synthetic_case": {
        "shape": [16, 64, 64], "status": "compatible_synthetic_tile_only",
        "resources": {"registers_per_thread": 68, "local_bytes": 0},
        "grid_ctas": 1,
        "standalone": {"median_gpu_event_us": 8.0, "raw_gpu_event_us": [7.9, 8.1]},
        "owner_entry": {"median_gpu_event_us": 8.1, "raw_gpu_event_us": [8.0, 8.2]},
        "cublaslt_external_control": {
            "median_gpu_event_us": 10.0, "raw_gpu_event_us": [9.9, 10.1],
        },
        "qualification": "Synthetic case only.",
    }}}
    report = build_budget(baseline, inventory, simt, mma, provider_report=provider)
    assert report["provider_smoke_comparisons"][0]["shape"] == [16, 64, 64]
    assert report["provider_smoke_comparisons"][0]["owner_to_cublaslt_gpu_event_ratio"] == pytest.approx(0.81)
    assert report["provider_body_comparisons"] == []


def test_v3r009_rejects_bad_direct_cublaslt_samples():
    baseline, inventory, simt, mma = evidence()
    provider = {"cases": [{
        "shape": [1, 17, 33], "owner_entry": None,
        "cublaslt_baseline": {
            "shape": [1, 17, 33], "dtype": "float16",
            "measurement": {"median_gpu_event_us": 1.0, "raw_gpu_event_us": []},
        },
    }]}
    with pytest.raises(ValueError, match="raw samples"):
        build_budget(baseline, inventory, simt, mma, provider_report=provider)


def test_v3r009_rejects_trace_without_device_activity(tmp_path):
    from body_budget import trace_diagnostic

    trace = tmp_path / "empty_trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    with pytest.raises(ValueError, match="no GPU kernel/copy activities"):
        trace_diagnostic(trace)
