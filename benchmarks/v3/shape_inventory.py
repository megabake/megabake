"""Join exact exported-FX hot shapes to an unfiltered matched baseline trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


_CONTRACTIONS = (".linear", ".mm", ".bmm", ".addmm", ".matmul", ".baddbmm")
_ATTENTION = ("scaled_dot_product_attention", "flash_attention", "efficient_attention")


def _tensor_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [{
            "shape": list(value.shape),
            "strides": list(value.stride()),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": value.device.type,
        }]
    if isinstance(value, (tuple, list)):
        return [record for item in value for record in _tensor_records(item)]
    return []


def _meta(node: Any) -> list[dict[str, Any]]:
    return _tensor_records(node.meta.get("val"))


def _tensor_bytes(record: dict[str, Any]) -> int | None:
    size = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int8": 1, "int16": 2, "int32": 4, "int64": 8, "bool": 1}.get(record["dtype"])
    if size is None:
        return None
    count = 1
    for dim in record["shape"]:
        if not isinstance(dim, int):
            return None
        count *= dim
    return count * size


def _gemm_shape(target: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = [item for group in inputs for item in group["tensors"]]
    matrices = [item for item in tensors if len(item["shape"]) >= 2]
    if len(matrices) < 2 or not outputs:
        return {"B": None, "M": None, "N": None, "K": None, "effective_strides": []}
    lhs, rhs = matrices[-2:] if ".addmm" in target else matrices[:2]
    left, right = lhs["shape"], rhs["shape"]
    if ".linear" in target:
        n, k = right[-2:]
        m = left[-2]
        batch = 1
        for dim in left[:-2]:
            batch *= dim
        lhs_stride = lhs["strides"]
        if len(left) > 2 and all(dim == 1 for dim in left[:-2]):
            lhs_stride = lhs_stride[-2:]
        elif len(left) > 2:
            lhs_stride = "unknown: linear flattens multiple leading dimensions; materialization depends on layout"
        return {
            "B": batch, "M": m, "N": n, "K": k,
            "effective_shapes": [[batch * m, k], [k, n]],
            "effective_strides": [lhs_stride, [rhs["strides"][-1], rhs["strides"][-2]]],
            "source_strides": [lhs["strides"], rhs["strides"]],
            "rhs_storage": "weight[N,K]; effective operator uses transpose[K,N]",
        }
    m, k = left[-2:]
    n = right[-1]
    batch = 1
    for dim in left[:-2]:
        batch *= dim
    return {"B": batch, "M": m, "N": n, "K": k, "effective_strides": [lhs["strides"], rhs["strides"]]}


def _attention_shape(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = [item for group in inputs for item in group["tensors"]]
    if len(tensors) < 3 or any(len(item["shape"]) < 4 for item in tensors[:3]):
        return {"B": None, "Hq": None, "Hkv": None, "Q": None, "K": None, "D": None}
    query, key, _value = (item["shape"] for item in tensors[:3])
    return {"B": query[0], "Hq": query[1], "Hkv": key[1], "Q": query[-2], "K": key[-2], "D": query[-1]}


def _operation_kind(target: str) -> str | None:
    lower = target.lower()
    if any(token in lower for token in _ATTENTION):
        return "attention"
    if any(token in lower for token in _CONTRACTIONS):
        return "contraction"
    if "index_copy" in lower or "index_put" in lower or "slice_scatter" in lower:
        return "state_copy"
    return None


def _is_cast(target: str) -> bool:
    lower = target.lower()
    return "to_copy" in lower or "convert_element_type" in lower or ".to." in lower or lower.endswith(".to")


def _linear_bias(node: Any) -> str:
    if "linear" not in str(node.target):
        return "not_applicable"
    if "bias" in node.kwargs:
        return "none" if node.kwargs["bias"] is None else "present"
    if len(node.args) < 3:
        return "none"  # aten.linear's omitted optional bias defaults to None.
    return "none" if node.args[2] is None else "present"


def inventory_exported_graph(exported_program: Any, model_config: Any = None) -> dict[str, Any]:
    records = []
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        kind = _operation_kind(target)
        if kind is None:
            continue
        inputs = [{"origin": item.name, "tensors": _meta(item)} for item in node.all_input_nodes]
        outputs = _meta(node)
        input_nodes = list(node.all_input_nodes)
        consumers = [
            {"fx_origin": user.name, "operator": str(user.target)}
            for user in node.users
        ]
        input_casts = [
            {"fx_origin": item.name, "operator": str(item.target)}
            for item in input_nodes if item.op == "call_function" and _is_cast(str(item.target))
        ]
        output_casts = [item for item in consumers if _is_cast(item["operator"])]
        input_bytes = [amount for item in inputs for tensor in item["tensors"] if (amount := _tensor_bytes(tensor)) is not None]
        output_bytes = [amount for tensor in outputs if (amount := _tensor_bytes(tensor)) is not None]
        record = {
            "fx_origin": node.name,
            "source_origin": (node.meta.get("stack_trace") or "").splitlines()[0] or None,
            "operator": target,
            "kind": kind,
            "inputs": inputs,
            "outputs": outputs,
            "dtypes": {
                "inputs": sorted({tensor["dtype"] for group in inputs for tensor in group["tensors"]}),
                "outputs": sorted({tensor["dtype"] for tensor in outputs}),
            },
            "bias": _linear_bias(node),
            "casts_and_epilogue": {
                "direct_input_casts": input_casts,
                "direct_output_casts": output_casts,
                "direct_output_users": consumers,
            },
            "shape_complete": bool(outputs) and all(bool(item["tensors"]) for item in inputs),
            "semantic_input_tensor_bytes": sum(input_bytes) if len(input_bytes) == sum(len(item["tensors"]) for item in inputs) else "unknown",
            "semantic_output_tensor_bytes": sum(output_bytes) if len(output_bytes) == len(outputs) else "unknown",
            "selected_template": "not determined by FX capture",
            "global_address_bytes": "unknown",
            "hbm_bytes": "unknown",
            "l2_bytes": "unknown",
            "critical_path_us": "unknown until baseline trace is joined",
            "call_count": 1,
        }
        if kind == "contraction":
            record["gemm"] = _gemm_shape(target, inputs, outputs)
        elif kind == "attention":
            record["attention"] = _attention_shape(inputs)
            if model_config is not None:
                record["attention"]["configured_Hkv"] = model_config.num_key_value_heads
                record["attention"]["gqa_group_size"] = model_config.num_attention_heads // model_config.num_key_value_heads
        records.append(record)
    return {
        "records": records,
        "contraction_count": sum(item["kind"] == "contraction" for item in records),
        "attention_count": sum(item["kind"] == "attention" for item in records),
        "state_copy_count": sum(item["kind"] == "state_copy" for item in records),
        "all_have_fx_origins": all(bool(item["fx_origin"]) for item in records),
        "all_shapes_complete": all(item["shape_complete"] for item in records),
    }


def _operator_family(target: str) -> str:
    value = target.removeprefix("aten.").split(".", 1)[0]
    if value == "linear":
        return "aten::mm"
    if "scaled_dot_product_attention" in value:
        return "aten::_scaled_dot_product"
    return f"aten::{value}"


def _shape_list(value: Any) -> list[tuple[int, ...]]:
    if isinstance(value, (tuple, list)):
        if all(isinstance(dim, int) for dim in value):
            return [tuple(value)]
        return [shape for item in value for shape in _shape_list(item)]
    return []


def _normalized_matrix_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    return shape


def _matching_profile_events(record: dict[str, Any], observed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family = _operator_family(record["operator"])
    expected = [
        tuple(tensor["shape"])
        for group in record["inputs"] for tensor in group["tensors"]
        if len(tensor["shape"]) >= 2
    ]
    if record["kind"] == "contraction":
        expected = expected[-2:] if ".addmm" in record["operator"] else expected[:2]
        expected = [_normalized_matrix_shape(shape) for shape in expected]
        if ".linear" in record["operator"]:
            # aten.linear stores weight as [N, K], while its observed mm
            # signature consumes the transposed [K, N] view.
            expected[1] = tuple(reversed(expected[1]))
    elif record["kind"] == "attention":
        expected = expected[:3]
    matches = []
    for event in observed:
        if not event.get("name", "").startswith(family):
            continue
        shapes = _shape_list(event.get("input_shapes"))
        if expected and len(shapes) >= len(expected) and shapes[:len(expected)] == expected:
            matches.append(event)
    return matches


def build_inventory(capture: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    if capture.get("cell_id") != baseline.get("cell_id"):
        raise ValueError("capture and baseline cells differ")
    if capture.get("manifest_hash") != baseline.get("manifest_hash"):
        raise ValueError("capture and baseline manifest hashes differ")
    if not baseline.get("best_baseline_id"):
        raise ValueError("V3R-004 requires a validated V3R-003 baseline")
    manifest_path = Path(capture.get("manifest", ""))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest_abi = manifest.get("step_abi", {})
    invocation_prep = manifest_abi.get("invocation_preparation", {})
    numerical_policy = manifest.get("numerical_policy", {})
    accumulation = numerical_policy.get("accumulation_dtypes", {})
    candidates = baseline.get("candidate_modes", [])
    selected = next((item for item in candidates if item.get("baseline_id") == baseline["best_baseline_id"]), None)
    if selected is None or not selected.get("unfiltered_trace"):
        raise ValueError("selected baseline has no retained unfiltered operation trace")
    shape_reference = next((item for item in candidates if item.get("baseline_id") == "torch.compile/default"), None)
    if (
        shape_reference is None
        or not shape_reference.get("compiled")
        or not shape_reference.get("equivalent_outputs")
        or not shape_reference.get("equivalent_state")
        or shape_reference.get("graph_break_count") != 0
        or not shape_reference.get("unfiltered_trace")
    ):
        raise ValueError("V3R-004 requires an equivalent, graph-break-free default compile trace for shape attribution")
    captures = capture.get("captures", [])
    current = next((item for item in captures if item.get("position") == capture.get("position", capture.get("cell_position"))), None)
    if current is None:
        current = captures[0] if captures else None
    if current is None or not current.get("fx_hot_math"):
        raise ValueError("capture is missing the FX hot-math inventory")
    records = current["fx_hot_math"]["records"]
    if not current["fx_hot_math"].get("all_have_fx_origins") or not current["fx_hot_math"].get("all_shapes_complete"):
        raise ValueError("one or more hot FX operations lack an origin or exact shape metadata")

    # CUDA Graph profiling retains compiled kernel events but can hide the Aten
    # operator shapes. Use the same-cell default compile trace for semantic
    # shape/call matching, and preserve the selected trace separately for its
    # actual kernel path. Never imply an FX-to-kernel mapping from this join.
    observed = shape_reference.get("operation_summary", [])
    origins_per_shape: dict[str, int] = {}
    for record in records:
        key = json.dumps((record["operator"], record.get("gemm") or record.get("attention")), sort_keys=True)
        origins_per_shape[key] = origins_per_shape.get(key, 0) + 1
    for record in records:
        if record["kind"] == "contraction":
            record["gemm"] = _gemm_shape(record["operator"], record["inputs"], record["outputs"])
        if record["kind"] == "attention":
            record["accumulation_dtype"] = accumulation.get("attention", "unknown: not declared in manifest")
        elif record["kind"] == "contraction":
            category = "projection" if ".linear" in record["operator"] else None
            record["accumulation_dtype"] = accumulation.get(category, "unknown: not declared for this contraction") if category else "unknown: non-linear contraction policy"
        else:
            record["accumulation_dtype"] = "not_applicable"
        matches = _matching_profile_events(record, observed)
        record["baseline_operation_matches"] = matches
        key = json.dumps((record["operator"], record.get("gemm") or record.get("attention")), sort_keys=True)
        observed_us = sum(item.get("device_total_us", 0.0) for item in matches)
        observed_calls = sum(item.get("count", 0) for item in matches)
        record["shape_reference_device_time_us_per_fx_call"] = observed_us / max(observed_calls, 1) if matches else "unknown"
        record["critical_path_us"] = "unknown: profiler aggregation does not expose per-origin span" 
        record["profiled_shape_reference_mode"] = shape_reference["baseline_id"]
        record["selected_template"] = "unknown: selected compiled path does not expose an FX-to-kernel mapping"
        record["observed_reference_operator_names"] = sorted({item["name"] for item in matches})
        record["observed_profile_call_count"] = observed_calls
        record["matched_fx_origin_count_for_shape"] = origins_per_shape[key]
    unmatched_hot = [
        item["fx_origin"] for item in records
        if item["kind"] in {"contraction", "attention"} and not item["baseline_operation_matches"]
    ]
    if unmatched_hot:
        raise ValueError(f"baseline trace has no exact operation-shape match for FX origins {unmatched_hot[:8]!r}")
    trace_path = Path(selected["operation_trace"])
    if not trace_path.is_file():
        raise ValueError("selected baseline trace artifact is missing")
    trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    shape_trace_path = Path(shape_reference["operation_trace"])
    if not shape_trace_path.is_file():
        raise ValueError("default compile shape-reference trace artifact is missing")
    shape_trace_hash = hashlib.sha256(shape_trace_path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "task_id": "V3R-004",
        "cell_id": capture["cell_id"],
        "manifest_hash": capture["manifest_hash"],
        "checkpoint": capture["checkpoint"],
        "checkpoint_revision": capture["revision"],
        "target": baseline["target"],
        "cache_condition": {
            "position": capture.get("position", capture.get("cell_position")),
            "capacity": capture.get("cache", {}).get("capacity"),
            "layout": capture.get("cache", {}).get("layout"),
            "dtype": capture.get("dtype"),
            "state_mode": "one fixed-state replay call for baseline profiling; advancing state verified by V3R-002H",
        },
        "weight_preparation": {
            "setup_amortization": invocation_prep.get("setup_amortization", "unknown"),
            "actions": invocation_prep.get("actions", []),
            "body_specific_packing": "unknown; no selected MegaBake or standalone body exists",
        },
        "selected_baseline_id": baseline["best_baseline_id"],
        "median_complete_call_wall_us": baseline["best_baseline_median_complete_call_wall_us"],
        "unfiltered_trace": {"path": str(trace_path), "sha256": trace_hash, "kernel_names": selected.get("runtime_event_names", [])},
        "shape_reference": {
            "baseline_id": shape_reference["baseline_id"],
            "trace_path": str(shape_trace_path),
            "trace_sha256": shape_trace_hash,
            "purpose": "FX-to-observed-operation matching; any event times below belong only to this mode, and selected kernel attribution is unknown",
        },
        "shape_reference_event_times": "per-event measurements belong only to the named shape-reference mode; not attributed to the selected baseline",
        "observed_math_kernel_names": sorted({
            name for name in selected.get("runtime_event_names", [])
            if any(token in name.lower() for token in ("gemm", "gemv", "cublas", "xmma", "cutlass", "flash", "attention"))
        }),
        "vendor_template_details": "provider/kernel names are observed above; private tile and descriptor details remain unknown",
        "hot_shapes": records,
        "observed_baseline_operations": observed,
        "traffic": {
            "semantic_bytes": "per-record logical input/output tensor bytes; excludes reuse and padding",
            "global_address_space_bytes": "unknown until a body and physical plan are selected",
            "hbm_bytes": "unknown; profiler trace does not provide validated hardware counters",
            "l2_bytes": "unknown; profiler trace does not provide validated hardware counters",
        },
        "removable_work": [item for item in records if item["kind"] == "state_copy"],
        "coverage": {
            "all_fx_hot_regions_have_origins": True,
            "all_shapes_complete": True,
            "all_records_have_dtype_and_epilogue_metadata": all(
                "dtypes" in item and "casts_and_epilogue" in item for item in records
            ),
            "accumulation_dtype_unknown_count": sum(
                isinstance(item["accumulation_dtype"], str) and item["accumulation_dtype"].startswith("unknown")
                for item in records
            ),
            "baseline_trace_unfiltered": True,
            "all_contraction_and_attention_shapes_match_profiled_operations": True,
            "selected_kernel_to_fx_mapping": "unknown",
            "opaque_vendor_details_marked_unknown": True,
            "unmatched_contraction_attention_shapes": len(unmatched_hot),
            "state_copy_origins_without_direct_profile_event": sum(
                item["kind"] == "state_copy" and not item["baseline_operation_matches"] for item in records
            ),
        },
    }


def write_inventory(path: str | Path, inventory: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
