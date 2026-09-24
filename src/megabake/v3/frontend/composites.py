"""Bounded overlapping semantic composites; selection is intentionally deferred."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .semantic import SemanticGraph, SemanticNode


@dataclass(frozen=True)
class CompositeCandidate:
    kind: str
    covered_ops: tuple[str, ...]
    live_boundaries: tuple[str, ...]
    effects: tuple[str, ...]
    recomputation_count: int = 0
    reference_ops: tuple[str, ...] = ()


def _candidate(kind: str, operations: Iterable[SemanticNode], *, recomputation_count: int = 0) -> CompositeCandidate:
    operations = tuple(operations)
    return CompositeCandidate(
        kind, tuple(operation.op_id for operation in operations),
        tuple(dict.fromkeys(value for operation in operations for value in operation.live_boundaries)),
        tuple(dict.fromkeys(effect for operation in operations for effect in operation.effects)),
        recomputation_count=recomputation_count,
        reference_ops=tuple(origin for operation in operations for origin in operation.origin_nodes),
    )


def _feeds(left: SemanticNode, right: SemanticNode) -> bool:
    return bool(set(left.outputs).intersection(right.inputs))


def enumerate_composites(graph: SemanticGraph) -> tuple[CompositeCandidate, ...]:
    """Enumerate candidates by value connectivity, never by graph adjacency.

    The returned candidates do not alter ``graph.operations``.  Consequently the
    unfused semantic operations remain available to every later cover/search.
    """
    ops = tuple(graph.operations)
    candidates: list[CompositeCandidate] = []
    for left in ops:
        for right in ops:
            if left is right or not _feeds(left, right):
                continue
            if left.name == "Linear" and right.name in {"Pointwise", "SwiGLU"}:
                candidates.append(_candidate("linear_epilogue", (left, right)))
            if left.name == "RMSNorm" and right.name == "Linear":
                candidates.append(_candidate("norm_linear", (left, right)))
            if left.name == "RoPE" and right.name == "CacheUpdate":
                candidates.append(_candidate("rope_cache", (left, right)))

    linears = [operation for operation in ops if operation.name == "Linear" and not operation.effects]
    by_input: dict[str, list[SemanticNode]] = {}
    for operation in linears:
        if operation.inputs:
            by_input.setdefault(operation.inputs[0], []).append(operation)
    for group in by_input.values():
        if len(group) >= 3:
            candidates.append(_candidate("qkv_group", tuple(group[:3])))

    for gate in (operation for operation in ops if operation.name == "SwiGLU"):
        sources = [linear for linear in linears if set(linear.outputs).intersection(gate.inputs)]
        if len(sources) >= 2:
            candidates.append(_candidate("gate_up_swiglu", (sources[0], sources[1], gate)))
            if not gate.effects:
                candidates.append(_candidate("gate_up_swiglu_recompute", (sources[0], sources[1], gate),
                                             recomputation_count=2))
    # Deduplicate equivalent overlapping alternatives while preserving a stable
    # order for inventories and tests.
    unique: dict[tuple[str, tuple[str, ...], int], CompositeCandidate] = {}
    for candidate in candidates:
        unique.setdefault((candidate.kind, candidate.covered_ops, candidate.recomputation_count), candidate)
    return tuple(unique.values())


def validate_cover(cover: Iterable[CompositeCandidate | SemanticNode]) -> None:
    covered: set[str] = set()
    effects: set[str] = set()
    for item in cover:
        ids = item.covered_ops if isinstance(item, CompositeCandidate) else (item.op_id,)
        item_effects = item.effects
        duplicate = covered.intersection(ids)
        if duplicate:
            raise ValueError(f"semantic cover duplicates operations: {sorted(duplicate)}")
        if item_effects and effects.intersection(item_effects):
            raise ValueError("semantic cover duplicates a state effect")
        covered.update(ids)
        effects.update(item_effects)
