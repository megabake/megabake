"""Parametric logical task families derived from a verified indexed cover."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from ..algorithms.choices import AlgorithmChoice
from ..semantics.indexed import IndexedTensorProgram
from ..semantics.verify import verify_algorithm_choice_cover
from .domains import TileDomain, TileInstance, tile_domain
from .maps import AccessMapError, RegionMap, index_expressions_supported, region_for_input, region_for_output


@dataclass(frozen=True)
class LogicalTaskFamily:
    family_id: str
    operation_id: str
    choice_id: str
    phase: str
    domain: TileDomain
    reads: tuple[RegionMap, ...]
    writes: tuple[RegionMap, ...]
    partial_value_id: str | None = None
    finalizes_family: str | None = None
    contributes_to_family: str | None = None
    required_effects: tuple[str, ...] = ()

    def enumerate(self, *, limit: int = 100_000) -> tuple[TileInstance, ...]:
        return self.domain.enumerate(self.family_id, limit=limit)

    def to_dict(self) -> dict[str, Any]:
        return {"family_id": self.family_id, "operation_id": self.operation_id,
                "choice_id": self.choice_id, "phase": self.phase,
                "domain": self.domain.to_dict(),
                "reads": [item.to_dict() for item in self.reads],
                "writes": [item.to_dict() for item in self.writes],
                "partial_value_id": self.partial_value_id,
                "finalizes_family": self.finalizes_family,
                "contributes_to_family": self.contributes_to_family,
                "required_effects": list(self.required_effects)}


@dataclass(frozen=True)
class LogicalExecutionPlan:
    indexed_program_hash: str
    selected_choice_ids: tuple[str, ...]
    guards: tuple[str, ...]
    families: tuple[LogicalTaskFamily, ...]
    semantic_coverage: tuple[Mapping[str, Any], ...]
    dependencies: tuple[Any, ...] = ()
    readiness_states: tuple[str, ...] = ()
    storage_lifetimes: tuple[Any, ...] = ()
    recompute_values: tuple[str, ...] = ()

    @property
    def structural_hash(self) -> str:
        return hashlib.sha256(json.dumps(self._payload(), sort_keys=True,
                                         separators=(",", ":"), default=str).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["logical_plan_hash"] = self.structural_hash
        return result

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "indexed_program_hash": self.indexed_program_hash,
                  "selected_choice_ids": list(self.selected_choice_ids), "guards": list(self.guards),
                  "families": [item.to_dict() for item in self.families],
                  "semantic_coverage": [dict(item) for item in self.semantic_coverage],
                  "dependencies": [_record(item) for item in self.dependencies],
                  "readiness_states": list(self.readiness_states),
                  "storage_lifetimes": [_record(item) for item in self.storage_lifetimes],
                  "recompute_values": list(self.recompute_values)}


def _record(item: Any) -> Any:
    return item.to_dict() if hasattr(item, "to_dict") else item


class LogicalPlanError(ValueError):
    pass


def lower_logical_plan(program: IndexedTensorProgram, choices: tuple[AlgorithmChoice, ...],
                       selected_choice_ids: tuple[str, ...], *,
                       tile_sizes: Mapping[str, int] = ()) -> LogicalExecutionPlan:
    """Lower an indexed reference expansion without assigning physical workers."""
    if not program.strict_supported:
        raise LogicalPlanError("indexed program has unsupported semantics or missing facts")
    coverage, diagnostics = verify_algorithm_choice_cover(program, choices, selected_choice_ids)
    if diagnostics or not coverage["valid"]:
        raise LogicalPlanError("selected algorithm choices do not cover the indexed program")

    selected = [choice for choice in choices if choice.choice_id in set(selected_choice_ids)]
    if any(choice.algorithm != "indexed" for choice in selected):
        raise LogicalPlanError("logical tile derivation currently requires the selected indexed expansion")
    choice_for_op = {op_id: choice for choice in selected for op_id in choice.operation_ids}
    operation_ids = tuple(dict.fromkeys(op_id for choice in selected for op_id in choice.operation_ids))
    operations = {operation.op_id: operation for operation in program.operations}
    values = {value.value_id: value for value in program.values}
    valid_axis_names = {axis.name for operation in program.operations
                        for axis in operation.iteration_domain + operation.reduction_domain}
    tile_sizes = {} if tile_sizes == () else tile_sizes
    unknown_axes = set(tile_sizes) - valid_axis_names
    if unknown_axes:
        raise LogicalPlanError(f"tile sizes name unknown axes: {sorted(unknown_axes)}")

    families = []
    for op_id in operation_ids:
        operation = operations[op_id]
        choice = choice_for_op[op_id]
        if not index_expressions_supported(operation):
            raise LogicalPlanError(f"{op_id} has an access map that cannot be proved")
        selected_axes = {axis.name for axis in
                         operation.iteration_domain + operation.reduction_domain}
        op_tiles = {name: size for name, size in tile_sizes.items() if name in selected_axes}
        output_tiles = {name: size for name, size in op_tiles.items()
                        if name in {axis.name for axis in operation.iteration_domain}}
        output_domain = tile_domain(operation, output_tiles, include_reduction=False)
        full_domain = tile_domain(operation, op_tiles)
        reads = tuple(region_for_input(operation, item, values)
                      for item in operation.input_index_maps)
        effects = tuple(edge.effect_id for edge in operation.effect_edges
                        if edge.required and edge.effect_id)

        if operation.kind == "Guard":
            families.append(LogicalTaskFamily(
                f"{op_id}:guard", op_id, choice.choice_id, "guard", TileDomain(()), reads, (),
                required_effects=effects,
            ))
            continue
        if not operation.outputs:
            raise LogicalPlanError(f"{op_id} has no logical output")
        output_value = operation.outputs[0]
        if operation.reduction_domain:
            partial_value = f"partial:{output_value}"
            compute_id, finalize_id = f"{op_id}:contribute", f"{op_id}:finalize"
            partial_write = RegionMap(
                partial_value, ("output_tile", "reduction_tile"), "partial_tile",
                tuple(axis.name for axis in operation.iteration_domain + operation.reduction_domain),
                (), tuple(axis.extent for axis in operation.iteration_domain),
            )
            families.append(LogicalTaskFamily(
                compute_id, op_id, choice.choice_id, "reduction_contributor", full_domain,
                reads, (partial_write,), partial_value_id=partial_value,
                contributes_to_family=finalize_id, required_effects=effects,
            ))
            partial_read = RegionMap(
                partial_value, ("all_contributors",), "partial_reduction", (), (),
                tuple(axis.extent for axis in operation.iteration_domain),
            )
            output = region_for_output(operation, output_value, values,
                                       predicate=operation.predicate_bounds + output_domain.guards)
            families.append(LogicalTaskFamily(
                finalize_id, op_id, choice.choice_id, "reduction_finalizer", output_domain,
                (partial_read,), (output,), partial_value_id=partial_value,
                finalizes_family=compute_id,
            ))
        else:
            output = region_for_output(operation, output_value, values,
                                       predicate=operation.predicate_bounds + full_domain.guards)
            families.append(LogicalTaskFamily(
                f"{op_id}:compute", op_id, choice.choice_id, "compute", full_domain,
                reads, (output,), required_effects=effects,
            ))

    recompute_values = []
    for choice in selected:
        for action in choice.preparation_actions:
            if action.get("kind") != "recompute":
                continue
            value_id = action.get("value_id")
            producers = [operation for operation in program.operations if value_id in operation.outputs]
            if (not value_id or len(producers) != 1 or producers[0].kind == "Scatter/StateWrite"
                    or any(edge.required for edge in producers[0].effect_edges)):
                raise LogicalPlanError("recomputation requires one pure, effect-free indexed producer")
            recompute_values.append(value_id)

    return LogicalExecutionPlan(
        program.structural_hash, tuple(selected_choice_ids),
        tuple(dict.fromkeys(guard for family in families for guard in family.domain.guards)),
        tuple(families), tuple({"op_id": op_id, "coverage_count": count}
                               for op_id, count in coverage.get("operation_coverage", {}).items()),
        recompute_values=tuple(dict.fromkeys(recompute_values)),
    )


__all__ = ["LogicalExecutionPlan", "LogicalPlanError", "LogicalTaskFamily", "lower_logical_plan"]
