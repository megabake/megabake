"""Structural layer summaries are analysis records, never opaque graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from .semantic import SemanticGraph, SemanticNode


@dataclass(frozen=True)
class LayerSummary:
    layer_index: int
    operation_ids: tuple[str, ...]
    fingerprint: str
    parameter_bindings: tuple[str, ...]
    state_bindings: tuple[str, ...]
    boundary_values: tuple[str, ...]


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))
                if "address" not in str(key).lower() and "binding" not in str(key).lower()
                and str(key) not in {"weight", "old_state", "input", "gate", "up"}}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def fingerprint_operations(operations: Iterable[SemanticNode], *, facts: Any = None, policy: Any = None) -> str:
    operations = tuple(operations)
    producer_index = {value: index for index, operation in enumerate(operations) for value in operation.outputs}
    payload = [(operation.name, operation.version, _canonical(operation.attributes),
                tuple("external" if value not in producer_index else producer_index[value] for value in operation.inputs),
                tuple((fact.shape, fact.strides, fact.dtype) for value in operation.inputs + operation.outputs
                      if facts is not None and (fact := facts.facts.get(value)) is not None), operation.effects)
               for operation in operations]
    payload.append(getattr(policy, "contract_hash", _canonical(policy)))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize_layers(graph: SemanticGraph, groups: Iterable[Iterable[str]] | None = None) -> tuple[LayerSummary, ...]:
    by_id = {operation.op_id: operation for operation in graph.operations}
    group_ids = tuple(tuple(group) for group in groups) if groups is not None else ((operation.op_id,) for operation in graph.operations)
    summaries: list[LayerSummary] = []
    for index, ids in enumerate(group_ids):
        operations = tuple(by_id[op_id] for op_id in ids)
        boundaries = tuple(value for operation in operations for value in operation.live_boundaries)
        params = tuple(sorted({str(operation.attributes.get("weight")) for operation in operations if operation.attributes.get("weight") is not None}))
        state = tuple(sorted({str(operation.attributes.get("old_state")) for operation in operations if operation.attributes.get("old_state") is not None}))
        summaries.append(LayerSummary(index, ids, fingerprint_operations(operations, facts=graph.facts, policy=graph.program.policy), params, state, boundaries))
    return tuple(summaries)


def group_repeated_layers(graph: SemanticGraph, groups: Iterable[Iterable[str]]) -> dict[str, tuple[LayerSummary, ...]]:
    """Group caller-proven layer boundaries without hiding their FX regions."""
    summaries = summarize_layers(graph, groups)
    grouped: dict[str, list[LayerSummary]] = {}
    for summary in summaries:
        grouped.setdefault(summary.fingerprint, []).append(summary)
    return {fingerprint: tuple(items) for fingerprint, items in grouped.items()}
