"""Deterministic normalized-graph inventory with explicitly unknown costs."""

from __future__ import annotations

import json
from typing import Any

from .semantic import SemanticGraph


def build_inventory(graph: SemanticGraph) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    multiplicity: dict[str, int] = {}
    seen_parameter_aliases: set[str] = set()
    parameter_storage = 0
    semantic_bytes = 0
    for operation in graph.operations:
        attrs = dict(operation.attributes)
        values = [graph.facts.facts[value_id] for value_id in operation.inputs + operation.outputs
                  if value_id in graph.facts.facts]
        weight_values = [fact for fact in values if fact.role == "weight"]
        record = {
            "op_id": operation.op_id, "name": operation.name, "origins": list(operation.origin_nodes),
            "inputs": list(operation.inputs), "outputs": list(operation.outputs), "attributes": attrs,
            "live_boundaries": list(operation.live_boundaries), "effects": list(operation.effects),
            "layout": [{"value_id": fact.value_id, "shape": list(fact.shape), "dtype": fact.dtype,
                        "strides": list(fact.strides), "storage_offset": fact.storage_offset}
                       for fact in values],
            "weights": [{"value_id": fact.value_id, "identity": fact.alias_set} for fact in weight_values],
            "epilogue": attrs.get("expression") or attrs.get("activation"),
        }
        for value_id in operation.inputs + operation.outputs:
            fact = graph.facts.facts.get(value_id)
            if not fact:
                continue
            count = _numel(fact.shape)
            if count is not None and fact.dtype_bytes is not None:
                semantic_bytes += count * fact.dtype_bytes
            if fact.role == "weight" and fact.alias_set and fact.alias_set not in seen_parameter_aliases:
                seen_parameter_aliases.add(fact.alias_set)
                if count is not None and fact.dtype_bytes is not None:
                    parameter_storage += count * fact.dtype_bytes
        entries.append(record)
        key = f"{operation.name}:{attrs.get('M')}:{attrs.get('N')}:{attrs.get('K')}:{attrs.get('output_dtype')}"
        multiplicity[key] = multiplicity.get(key, 0) + 1
    return {
        "graph_hash": graph.structural_hash,
        "operations": entries,
        "unsupported": [{"node_id": region.node_id, "target": region.target} for region in graph.reference_regions],
        "values": {value_id: _fact_dict(fact) for value_id, fact in sorted(graph.facts.facts.items())},
        "multiplicity": dict(sorted(multiplicity.items())),
        "bytes": {"semantic_accessed": semantic_bytes, "parameter_storage": parameter_storage, "physical_dram": None},
    }


def inventory_json(graph: SemanticGraph) -> str:
    return json.dumps(build_inventory(graph), sort_keys=True, separators=(",", ":"), default=str)


def inventory_text(graph: SemanticGraph) -> str:
    inventory = build_inventory(graph)
    lines = [f"graph {inventory['graph_hash']}"]
    for entry in inventory["operations"]:
        attrs = entry["attributes"]
        shape = " ".join(f"{name}={attrs[name]}" for name in ("M", "N", "K") if name in attrs)
        lines.append(f"{entry['op_id']} {entry['name']} {shape}".rstrip())
    lines.append(f"unsupported={len(inventory['unsupported'])} physical_dram=unknown")
    return "\n".join(lines)


def _numel(shape: tuple[Any, ...]) -> int | None:
    result = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        result *= dim
    return result


def _fact_dict(fact: Any) -> dict[str, Any]:
    return {"shape": list(fact.shape), "dtype": fact.dtype, "strides": list(fact.strides),
            "storage_offset": fact.storage_offset, "alias_set": fact.alias_set,
            "role": fact.role, "mutability": fact.mutability,
            "symbolic_constraints": list(fact.symbolic_constraints), "provenance": list(fact.provenance)}
