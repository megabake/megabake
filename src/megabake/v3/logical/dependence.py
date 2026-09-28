"""Target-neutral producer, reduction, and effect-order relations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..semantics.indexed import IndexedTensorProgram
from .domains import TileInstance
from .maps import AccessMapError, RegionMap, enumerate_region


class Readiness(str, Enum):
    ADDRESS_KNOWN = "address_known"
    INPUT_DATA_READY = "input_data_ready"
    COPY_ISSUED = "copy_issued"
    SOURCE_RETIRED = "source_retired"
    DESTINATION_VISIBLE = "destination_visible"
    COMPUTE_COMPLETE = "compute_complete"
    OUTPUT_PUBLISHED = "output_published"
    REDUCTION_FINAL = "reduction_final"
    STORAGE_REUSABLE = "storage_reusable"


@dataclass(frozen=True)
class DependencyRelation:
    relation_id: str
    producer_family_id: str
    consumer_family_id: str
    kind: str
    value_id: str
    producer_region: RegionMap | None
    consumer_region: RegionMap | None
    producer_readiness: Readiness
    consumer_requires: Readiness

    def to_dict(self) -> dict[str, Any]:
        return {"relation_id": self.relation_id, "producer_family_id": self.producer_family_id,
                "consumer_family_id": self.consumer_family_id, "kind": self.kind,
                "value_id": self.value_id,
                "producer_region": self.producer_region.to_dict() if self.producer_region else None,
                "consumer_region": self.consumer_region.to_dict() if self.consumer_region else None,
                "producer_readiness": self.producer_readiness.value,
                "consumer_requires": self.consumer_requires.value}


@dataclass(frozen=True)
class TaskDependency:
    relation_id: str
    producer_task_id: str
    consumer_task_id: str
    producer_readiness: Readiness
    consumer_requires: Readiness
    scope: str = "region_overlap"

    def to_dict(self) -> dict[str, Any]:
        return {"relation_id": self.relation_id, "producer_task_id": self.producer_task_id,
                "consumer_task_id": self.consumer_task_id,
                "producer_readiness": self.producer_readiness.value,
                "consumer_requires": self.consumer_requires.value, "scope": self.scope}


def _family_for_operation(families: tuple[Any, ...], operation_id: str) -> Any | None:
    return next((family for family in families if family.operation_id == operation_id
                 and family.phase not in {"reduction_finalizer"}), None)


def _writer_for_value(families: tuple[Any, ...], value_id: str) -> tuple[Any, RegionMap] | None:
    matches = [(family, region) for family in families for region in family.writes
               if region.value_id == value_id]
    return matches[0] if len(matches) == 1 else None


def derive_dependencies(plan: Any, program: IndexedTensorProgram) -> tuple[
        tuple[DependencyRelation, ...], tuple[DiagnosticRecord, ...]]:
    """Compress equal tile edges into exact family-level map relations."""
    relations: list[DependencyRelation] = []
    diagnostics: list[DiagnosticRecord] = []
    families = plan.families
    values = {value.value_id: value for value in program.values}
    graph_nodes = {node.name: node for node in program.source_program.graph_module.graph.nodes}

    for family in families:
        if family.finalizes_family:
            producer = next((item for item in families if item.family_id == family.finalizes_family), None)
            if producer is None or producer.contributes_to_family != family.family_id:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS, "reduction finalizer has no matching contributor family",
                    DiagnosticSeverity.ERROR, action_id=family.family_id,
                    details={"negative_case": "missing reduction producer"},
                ))
            else:
                relations.append(DependencyRelation(
                    f"reduction:{producer.family_id}:{family.family_id}", producer.family_id,
                    family.family_id, "reduction_finalize", family.partial_value_id or "",
                    producer.writes[0] if producer.writes else None,
                    family.reads[0] if family.reads else None,
                    Readiness.COMPUTE_COMPLETE, Readiness.REDUCTION_FINAL,
                ))

    for family in families:
        if family.phase == "reduction_finalizer":
            continue
        for access_index, read in enumerate(family.reads):
            if read.value_id.startswith("partial:"):
                continue
            writer = _writer_for_value(families, read.value_id)
            if writer is None:
                value = values.get(read.value_id)
                node = graph_nodes.get(value.fx_node) if value else None
                if node is not None and node.op not in {"placeholder", "get_attr"}:
                    diagnostics.append(DiagnosticRecord(
                        DiagnosticCode.MISSING_FACTS,
                        f"read of {read.value_id} has no unique logical producer",
                        DiagnosticSeverity.ERROR, node_id=family.operation_id,
                        action_id=family.family_id,
                        details={"negative_case": "missing producer", "value_id": read.value_id},
                    ))
                continue
            producer, write = writer
            if producer.family_id == family.family_id:
                continue
            relations.append(DependencyRelation(
                f"data:{producer.family_id}:{family.family_id}:{read.value_id}:{access_index}",
                producer.family_id, family.family_id, "region_overlap", read.value_id, write, read,
                Readiness.OUTPUT_PUBLISHED, Readiness.INPUT_DATA_READY,
            ))

    transition_by_effect = {item.effect_id: item for item in program.state_transitions}
    for operation in program.operations:
        consumer = _family_for_operation(families, operation.op_id)
        if consumer is None:
            continue
        for edge in operation.effect_edges:
            if not edge.required or edge.kind != "state_read_after_publish":
                continue
            for effect_id in edge.depends_on:
                transition = transition_by_effect.get(effect_id)
                writer = _writer_for_value(families, transition.new_value) if transition else None
                if transition is None or writer is None:
                    diagnostics.append(DiagnosticRecord(
                        DiagnosticCode.MISSING_FACTS,
                        f"state read in {operation.op_id} has no logical publication producer",
                        DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
                        action_id=consumer.family_id,
                        details={"negative_case": "missing state publication", "effect_id": effect_id},
                    ))
                    continue
                producer, write = writer
                relations.append(DependencyRelation(
                    f"state_publish:{effect_id}:{consumer.family_id}", producer.family_id,
                    consumer.family_id, "state_publish", transition.new_value, write,
                    next((read for read in consumer.reads if read.value_id == transition.new_value), None),
                    Readiness.OUTPUT_PUBLISHED, Readiness.INPUT_DATA_READY,
                ))

    transitions = sorted(program.state_transitions, key=lambda item: (item.state_id, item.order))
    for before, after in zip(transitions, transitions[1:]):
        if before.state_id != after.state_id:
            continue
        first = _writer_for_value(families, before.new_value)
        second = _writer_for_value(families, after.new_value)
        if first and second:
            relations.append(DependencyRelation(
                f"state_order:{before.effect_id}:{after.effect_id}", first[0].family_id,
                second[0].family_id, "state_order", after.new_value, first[1], second[1],
                Readiness.OUTPUT_PUBLISHED, Readiness.INPUT_DATA_READY,
            ))
    return tuple(relations), tuple(diagnostics)


def materialize_dependencies(plan: Any, relations: tuple[DependencyRelation, ...], *,
                             dynamic_inputs: Mapping[str, Any] = (), limit: int = 100_000
                             ) -> tuple[TaskDependency, ...]:
    """Instantiate tiny domains for verifier fixtures; the plan stores relations symbolically."""
    dynamic_inputs = {} if dynamic_inputs == () else dynamic_inputs
    families = {family.family_id: family for family in plan.families}
    edges: list[TaskDependency] = []
    for relation in relations:
        if relation.producer_family_id not in families or relation.consumer_family_id not in families:
            continue
        producer, consumer = (families[relation.producer_family_id],
                             families[relation.consumer_family_id])
        source_tiles, target_tiles = producer.enumerate(limit=limit), consumer.enumerate(limit=limit)
        if relation.kind == "reduction_finalize":
            for source in source_tiles:
                for target in target_tiles:
                    if source.coordinate.output == target.coordinate.output:
                        edges.append(TaskDependency(relation.relation_id, source.task_id, target.task_id,
                                                    relation.producer_readiness, relation.consumer_requires,
                                                    "all_reduction_contributors"))
            continue
        if relation.kind in {"state_publish", "state_order"}:
            edges.extend(TaskDependency(relation.relation_id, source.task_id, target.task_id,
                                        relation.producer_readiness, relation.consumer_requires,
                                        "publication_barrier")
                         for source, target in product(source_tiles, target_tiles))
            continue
        if relation.producer_region is None or relation.consumer_region is None:
            continue
        try:
            writes = [(tile, set(enumerate_region(relation.producer_region, tile, limit=limit)))
                      for tile in source_tiles]
            for target in target_tiles:
                reads = set(enumerate_region(relation.consumer_region, target,
                                             dynamic_inputs=dynamic_inputs, limit=limit))
                for source, points in writes:
                    if points.intersection(reads):
                        edges.append(TaskDependency(relation.relation_id, source.task_id, target.task_id,
                                                    relation.producer_readiness,
                                                    relation.consumer_requires, "region_overlap"))
        except AccessMapError:
            # A bounded indirect read remains exact in the symbolic relation; until runtime values
            # exist, requiring its whole producer family is the safe logical materialization.
            edges.extend(TaskDependency(relation.relation_id, source.task_id, target.task_id,
                                        relation.producer_readiness, relation.consumer_requires,
                                        "whole_producer_family_until_runtime_index")
                         for source, target in product(source_tiles, target_tiles))
    return tuple(edges)


__all__ = ["DependencyRelation", "Readiness", "TaskDependency", "derive_dependencies",
           "materialize_dependencies"]
