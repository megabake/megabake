"""Logical dependency, state-order, and lifetime verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, islice
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..semantics.indexed import IndexedTensorProgram
from .dependence import (DependencyRelation, Readiness, TaskDependency, derive_dependencies,
                         materialize_dependencies)
from .plan import LogicalExecutionPlan
from .storage import StorageLifetime, build_storage_lifetimes, can_overlay


@dataclass(frozen=True)
class LogicalVerificationReport:
    valid: bool
    diagnostics: tuple[DiagnosticRecord, ...]
    topological_order: tuple[str, ...]
    task_dependencies: tuple[TaskDependency, ...]
    safe_overlay_pairs: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "valid": self.valid,
                "diagnostics": [item.to_dict() for item in self.diagnostics],
                "topological_order": list(self.topological_order),
                "task_dependencies": [item.to_dict() for item in self.task_dependencies],
                "safe_overlay_pairs": [list(item) for item in self.safe_overlay_pairs]}


def _relations_topology(plan: LogicalExecutionPlan, relations: tuple[DependencyRelation, ...]
                        ) -> tuple[tuple[str, ...], set[tuple[str, str]]]:
    family_ids = {item.family_id for item in plan.families}
    outgoing = {item: set() for item in family_ids}
    indegree = {item: 0 for item in family_ids}
    for relation in relations:
        before, after = relation.producer_family_id, relation.consumer_family_id
        if before not in family_ids or after not in family_ids or after in outgoing[before]:
            continue
        outgoing[before].add(after)
        indegree[after] += 1
    ready = sorted(item for item, degree in indegree.items() if degree == 0)
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for after in sorted(outgoing[current]):
            indegree[after] -= 1
            if indegree[after] == 0:
                ready.append(after)
                ready.sort()
    if len(order) != len(family_ids):
        return (), set()
    closure: set[tuple[str, str]] = set()
    for before in order:
        stack = list(outgoing[before])
        seen = set()
        while stack:
            after = stack.pop()
            if after in seen:
                continue
            seen.add(after)
            closure.add((before, after))
            stack.extend(outgoing[after])
    return tuple(order), closure


def _missing_writer_diagnostics(plan: LogicalExecutionPlan, program: IndexedTensorProgram
                                ) -> list[DiagnosticRecord]:
    nodes = {node.name: node for node in program.source_program.graph_module.graph.nodes}
    writes: dict[str, list[str]] = {}
    for family in plan.families:
        for region in family.writes:
            writes.setdefault(region.value_id, []).append(family.family_id)
    diagnostics = []
    for output in program.outputs:
        if output.value_id is None:
            continue
        value = next((item for item in program.values if item.value_id == output.value_id), None)
        node = nodes.get(value.fx_node) if value else None
        if node is not None and node.op in {"placeholder", "get_attr"}:
            continue
        owners = writes.get(output.value_id, ())
        if len(owners) != 1:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"logical output {output.value_id} has {len(owners)} writers",
                DiagnosticSeverity.ERROR, node_id=output.origin_id,
                details={"negative_case": "missing or duplicated output writer", "writers": list(owners)},
            ))
    return diagnostics


def verify_logical_plan(plan: LogicalExecutionPlan, program: IndexedTensorProgram
                        ) -> LogicalVerificationReport:
    expected, derivation_diagnostics = derive_dependencies(plan, program)
    diagnostics = list(derivation_diagnostics)
    expected_by_id = {item.relation_id: item for item in expected}
    actual_ids = [item.relation_id for item in plan.dependencies]
    if len(actual_ids) != len(set(actual_ids)):
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.INVALID_EVENT, "logical dependency relation is duplicated",
            DiagnosticSeverity.ERROR, details={"negative_case": "duplicate relation"},
        ))
    actual_by_id = {item.relation_id: item for item in plan.dependencies}
    for relation_id, relation in expected_by_id.items():
        if relation_id not in actual_by_id:
            negative_case = {
                "region_overlap": "missing producer",
                "reduction_finalize": "premature reduction finalizer",
                "state_publish": "cache read before publication",
                "state_order": "missing state order",
            }.get(relation.kind, "missing logical dependency")
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"required {relation.kind} relation {relation_id} is missing",
                DiagnosticSeverity.ERROR, action_id=relation_id,
                details={"negative_case": negative_case,
                         "producer_family": relation.producer_family_id,
                         "consumer_family": relation.consumer_family_id,
                         "value_id": relation.value_id},
            ))
        elif actual_by_id[relation_id] != relation:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.INVALID_EVENT, f"dependency {relation_id} differs from its indexed regions",
                DiagnosticSeverity.ERROR, action_id=relation_id,
                details={"negative_case": "altered producer/read region"},
            ))
    for relation_id in set(actual_by_id) - set(expected_by_id):
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.INVALID_EVENT, f"dependency {relation_id} has no indexed relation",
            DiagnosticSeverity.ERROR, action_id=relation_id,
            details={"negative_case": "invented dependency"},
        ))
    family_ids = {item.family_id for item in plan.families}
    for relation in plan.dependencies:
        if relation.producer_family_id not in family_ids or relation.consumer_family_id not in family_ids:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"dependency {relation.relation_id} names a missing family",
                DiagnosticSeverity.ERROR, action_id=relation.relation_id,
                details={"negative_case": "missing producer family"},
            ))
        if relation.kind == "reduction_finalize" and (
                relation.producer_readiness != Readiness.COMPUTE_COMPLETE
                or relation.consumer_requires != Readiness.REDUCTION_FINAL):
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.INVALID_EVENT, "reduction finalizer readiness is incomplete",
                DiagnosticSeverity.ERROR, action_id=relation.relation_id,
                details={"negative_case": "premature reduction finalizer"},
            ))
        if relation.kind == "state_publish" and (
                relation.producer_readiness != Readiness.OUTPUT_PUBLISHED
                or relation.consumer_requires != Readiness.INPUT_DATA_READY):
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.INVALID_EVENT, "state read is not ordered after output publication",
                DiagnosticSeverity.ERROR, action_id=relation.relation_id,
                details={"negative_case": "cache read before publication"},
            ))
    diagnostics.extend(_missing_writer_diagnostics(plan, program))
    order, closure = _relations_topology(plan, plan.dependencies)
    if not order and plan.families:
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.INVALID_PROGRESS, "logical dependency graph contains a cycle",
            DiagnosticSeverity.ERROR, details={"negative_case": "dependency cycle"},
        ))
    try:
        task_edges = materialize_dependencies(plan, plan.dependencies)
    except ValueError:
        # Symbolic domains are preserved in the plan; tiny concrete expansion is optional evidence.
        task_edges = ()
    lifetimes = plan.storage_lifetimes
    safe_overlays = tuple((first.value_id, second.value_id)
                          for first, second in combinations(lifetimes, 2)
                          if can_overlay(first, second, closure))
    return LogicalVerificationReport(not diagnostics, tuple(diagnostics), order, task_edges, safe_overlays)


def analyze_logical_plan(plan: LogicalExecutionPlan, program: IndexedTensorProgram
                         ) -> tuple[LogicalExecutionPlan, LogicalVerificationReport]:
    relations, _ = derive_dependencies(plan, program)
    lifetimes = build_storage_lifetimes(plan, program)
    analyzed = replace(
        plan, dependencies=relations,
        readiness_states=tuple(item.value for item in Readiness),
        storage_lifetimes=lifetimes,
    )
    return analyzed, verify_logical_plan(analyzed, program)


def topological_orders(plan: LogicalExecutionPlan, *, limit: int = 8) -> tuple[tuple[str, ...], ...]:
    """Enumerate a few legal family orders for tiny CPU schedule checks."""
    family_ids = tuple(sorted(item.family_id for item in plan.families))
    predecessors = {item: set() for item in family_ids}
    for relation in plan.dependencies:
        predecessors[relation.consumer_family_id].add(relation.producer_family_id)

    def visit(done: tuple[str, ...]):
        if len(done) == len(family_ids):
            yield done
            return
        ready = [item for item in family_ids if item not in done and predecessors[item].issubset(done)]
        for item in ready:
            yield from visit(done + (item,))

    return tuple(islice(visit(()), limit))


def execute_schedule(program: IndexedTensorProgram, plan: LogicalExecutionPlan,
                     family_order: tuple[str, ...], inputs: Mapping[str, Any]) -> Mapping[tuple[Any, ...], Any]:
    """Interpret whole indexed operations when their logical completion family runs."""
    if set(family_order) != {item.family_id for item in plan.families}:
        raise ValueError("schedule must run every logical family exactly once")
    source = program.source_program
    nodes = {node.name: node for node in source.graph_module.graph.nodes}
    values = {item.value_id: item for item in program.values}
    environment: dict[str, Any] = {}
    source_graph = source.source_graph_module or source.graph_module
    for value in program.values:
        node = nodes.get(value.fx_node)
        if node is None:
            continue
        if node.op == "placeholder":
            if node.name in inputs:
                environment[value.value_id] = inputs[node.name]
            elif value.value_id in inputs:
                environment[value.value_id] = inputs[value.value_id]
            else:
                raise KeyError(f"missing schedule input {node.name!r}")
        elif node.op == "get_attr":
            binding = source.lifted_bindings.get(node.name)
            if binding is not None:
                environment[value.value_id] = source.binding_values[binding.target]
            else:
                result = source_graph
                for component in str(node.target).split("."):
                    result = getattr(result, component)
                environment[value.value_id] = result

    operations = {item.op_id: item for item in program.operations}
    family_by_id = {item.family_id: item for item in plan.families}
    executed = set()
    for family_id in family_order:
        family = family_by_id[family_id]
        if family.phase == "reduction_contributor" or family.operation_id in executed:
            continue
        operation = operations[family.operation_id]
        result = operation.local_reference.evaluate(environment)
        if operation.outputs:
            environment[operation.outputs[0]] = result
        executed.add(operation.op_id)
    missing = set(operations) - executed
    if missing:
        raise ValueError(f"schedule omitted indexed operations: {sorted(missing)}")
    return {output.path: environment[output.value_id] if output.value_id else output.literal
            for output in program.outputs}


__all__ = ["LogicalVerificationReport", "analyze_logical_plan", "can_overlay", "execute_schedule",
           "topological_orders", "verify_logical_plan"]
