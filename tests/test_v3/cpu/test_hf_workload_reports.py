from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.v3.baseline_report import rejection_reasons, select_best_baseline
from benchmarks.v3.hf_cached_step import HuggingFaceCachedStep, assert_step_outputs, load_hf_model
from benchmarks.v3.shape_inventory import build_inventory, inventory_exported_graph


def _candidate(name: str, latency: float) -> dict:
    return {
        "baseline_id": name,
        "compiled": True,
        "equivalent_outputs": True,
        "equivalent_state": True,
        "old_state_unchanged": True,
        "output_ownership": True,
        "graph_break_count": 0,
        "cudagraph_requested": False,
        "cudagraph_observed": False,
        "unfiltered_trace": True,
        "complete_call_wall_us": [latency, latency + 0.2],
    }


def test_hf_step_and_revision_guards_accept_positive_cell_and_reject_capacity_overflow():
    step = HuggingFaceCachedStep(torch.nn.Identity(), valid_length=0, capacity=1)
    assert step.valid_length == 0
    with pytest.raises(ValueError, match="below the positive cache capacity"):
        HuggingFaceCachedStep(torch.nn.Identity(), valid_length=1, capacity=1)
    with pytest.raises(ValueError, match="full lowercase commit hash"):
        load_hf_model(revision="main")


def test_fx_inventory_records_linear_dtypes_bias_and_explicit_cast_epilogue():
    class Projection(torch.nn.Module):
        def __init__(self, with_bias):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float16))
            self.bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float16)) if with_bias else None

        def forward(self, x):
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            return y.to(torch.float32)

    for with_bias, expected_bias in ((False, "none"), (True, "present")):
        module = Projection(with_bias)
        graph = torch.export.export(module, (torch.ones(1, 4, dtype=torch.float16),))
        record = next(
            item for item in inventory_exported_graph(graph)["records"]
            if item["kind"] == "contraction" and ".linear" in item["operator"]
        )
        assert record["bias"] == expected_bias
        assert record["dtypes"] == {"inputs": ["float16"], "outputs": ["float16"]}
        assert any("aten.to" in item["operator"] for item in record["casts_and_epilogue"]["direct_output_casts"])


def test_cached_step_oracle_rejects_a_changed_output_tree():
    expected = {
        "logits": torch.zeros(1, 1),
        "cache": torch.zeros(1),
        "valid_length": torch.tensor(1),
    }
    with pytest.raises(AssertionError, match="output tree changed"):
        assert_step_outputs(expected, {"logits": expected["logits"], "valid_length": expected["valid_length"]})


def test_baseline_selection_keeps_fastest_equivalent_and_rejects_ownership_or_graph_breaks():
    valid = _candidate("default", 10.0)
    faster_but_reused = _candidate("reduce-overhead", 5.0)
    faster_but_reused["output_ownership"] = False
    broken = _candidate("max-autotune", 4.0)
    broken["graph_break_count"] = 1
    assert select_best_baseline([valid, faster_but_reused, broken]) is valid
    assert "returned outputs are reused across calls" in rejection_reasons(faster_but_reused)
    assert "torch.compile graph breaks are present or unknown" in rejection_reasons(broken)


def _capture_report(trace: str) -> tuple[dict, dict]:
    record = {
        "fx_origin": "linear_0",
        "kind": "contraction",
        "operator": "aten.linear.default",
        "inputs": [
            {"origin": "activation", "tensors": [{"shape": [1, 1, 576], "strides": [576, 576, 1]}]},
            {"origin": "weight", "tensors": [{"shape": [192, 576], "strides": [576, 1]}]},
        ],
        "outputs": [{"shape": [1, 1, 192]}],
        "dtypes": {"inputs": ["float16"], "outputs": ["float16"]},
        "bias": "none",
        "casts_and_epilogue": {"direct_input_casts": [], "direct_output_casts": [], "direct_output_users": []},
        "shape_complete": True,
        "gemm": {
            "B": 1, "M": 1, "N": 192, "K": 576,
            "effective_shapes": [[1, 576], [576, 192]],
            "effective_strides": [[576, 1], [1, 576]],
            "source_strides": [[576, 576, 1], [576, 1]],
            "rhs_storage": "weight[N,K]; effective operator uses transpose[K,N]",
        },
        "call_count": 1,
    }
    capture = {
        "cell_id": "cell-L128",
        "position": 128,
        "manifest_hash": "manifest-hash",
        "dtype": "float16",
        "cache": {"capacity": 2050, "layout": "layers,kv,batch,heads,capacity,head_dim"},
        "checkpoint": "model",
        "revision": "revision",
        "captures": [{
            "position": 128,
            "fx_hot_math": {
                "records": [record],
                "all_have_fx_origins": True,
                "all_shapes_complete": True,
            },
        }],
    }
    manifest_path = Path(trace).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "numerical_policy": {"accumulation_dtypes": {"projection": "float32"}},
        "step_abi": {"invocation_preparation": {
            "setup_amortization": "stable_model_weights_per_session", "actions": []
        }},
    }))
    capture["manifest"] = str(manifest_path)
    baseline = {
        "cell_id": "cell-L128",
        "manifest_hash": "manifest-hash",
        "best_baseline_id": "torch.compile/reduce-overhead",
        "best_baseline_median_complete_call_wall_us": 10.0,
        "target": "test-target",
        "candidate_modes": [
            {
                "baseline_id": "torch.compile/reduce-overhead",
                "compiled": True,
                "equivalent_outputs": True,
                "equivalent_state": True,
                "graph_break_count": 0,
                "unfiltered_trace": True,
                "operation_trace": trace,
                "runtime_event_names": ["triton_fused_gemm"],
                "operation_summary": [],
            },
            {
                "baseline_id": "torch.compile/default",
                "compiled": True,
                "equivalent_outputs": True,
                "equivalent_state": True,
                "graph_break_count": 0,
                "unfiltered_trace": True,
                "operation_trace": trace,
                "runtime_event_names": ["sm90_xmma_gemm"],
                "operation_summary": [{
                    "name": "aten::mm",
                    "count": 1,
                    "input_shapes": [[1, 576], [576, 192], [1, 192]],
                    "device_total_us": 1.25,
                }],
            },
        ],
    }
    return capture, baseline


def test_shape_inventory_joins_fx_origin_shape_to_measured_baseline_and_rejects_missing_origin(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    capture, baseline = _capture_report(str(trace))
    inventory = build_inventory(capture, baseline)
    assert inventory["selected_baseline_id"] == "torch.compile/reduce-overhead"
    assert inventory["hot_shapes"][0]["fx_origin"] == "linear_0"
    assert inventory["hot_shapes"][0]["gemm"]["effective_shapes"] == [[1, 576], [576, 192]]
    assert inventory["hot_shapes"][0]["gemm"]["effective_strides"] == [[576, 1], [1, 576]]
    assert inventory["hot_shapes"][0]["observed_profile_call_count"] == 1
    assert inventory["hot_shapes"][0]["shape_reference_device_time_us_per_fx_call"] == 1.25
    assert "not attributed to the selected baseline" in inventory["shape_reference_event_times"]
    assert inventory["hot_shapes"][0]["accumulation_dtype"] == "float32"
    assert inventory["cache_condition"]["capacity"] == 2050
    assert inventory["weight_preparation"]["setup_amortization"] == "stable_model_weights_per_session"
    assert inventory["shape_reference"]["baseline_id"] == "torch.compile/default"
    assert inventory["coverage"]["selected_kernel_to_fx_mapping"] == "unknown"

    capture["captures"][0]["fx_hot_math"]["all_have_fx_origins"] = False
    with pytest.raises(ValueError, match="lack an origin or exact shape"):
        build_inventory(capture, baseline)


def test_shape_inventory_rejects_a_linear_profile_with_the_untransposed_weight_shape(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    capture, baseline = _capture_report(str(trace))
    wrong_shape = copy.deepcopy(baseline)
    wrong_shape["candidate_modes"][1]["operation_summary"][0]["input_shapes"][1] = [192, 576]
    with pytest.raises(ValueError, match="no exact operation-shape match"):
        build_inventory(capture, wrong_shape)


def test_shape_inventory_rejects_baseline_without_a_selected_equivalent_path(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    capture, baseline = _capture_report(str(trace))
    baseline["best_baseline_id"] = None
    with pytest.raises(ValueError, match="validated V3R-003 baseline"):
        build_inventory(capture, baseline)
