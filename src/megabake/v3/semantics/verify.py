"""Origin, output and effect coverage checks for the indexed frontend."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..frontend.capture import NormalizedProgram, _ORIGIN_META
from .indexed import IndexedOp, OutputLeaf


def _live_nodes(graph_module: Any, effects: set[str]) -> set[Any]:
    nodes = list(graph_module.graph.nodes)
    stack = [node for node in nodes if node.op == "output" or node.name in effects]
    live: set[Any] = set()
    while stack:
        node = stack.pop()
        if node in live:
            continue
        live.add(node)
        stack.extend(node.all_input_nodes)
    return live


def verify_indexed_program(program: NormalizedProgram, operations: tuple[IndexedOp, ...],
                           node_to_value: Mapping[str, str],
                           outputs: tuple[OutputLeaf, ...] = ()) -> tuple[
                               tuple[Mapping[str, Any], ...], tuple[DiagnosticRecord, ...]
                           ]:
    diagnostics: list[DiagnosticRecord] = []
    operation_by_node = {operation.local_reference.node_name: operation for operation in operations}
    effect_nodes = {effect.node_id for effect in program.effects if effect.required}
    live = _live_nodes(program.graph_module, effect_nodes)
    coverage = []
    origin_owner: dict[str, str] = {}
    for node in program.graph_module.graph.nodes:
        if node not in live:
            continue
        origin_ids = tuple(program.origin_map.get(node.name, ()))
        if not origin_ids:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"live FX node {node.name} has no stable origin",
                DiagnosticSeverity.ERROR, node_id=node.name,
            ))
        for origin_id in origin_ids:
            prior = origin_owner.get(origin_id)
            if prior is not None and prior != node.name:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live FX origin {origin_id} is represented by multiple nodes",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"first_node": prior, "second_node": node.name},
                ))
            origin_owner[origin_id] = node.name

        if node.op == "placeholder":
            status = "input_value"
            operation = None
        elif node.op == "get_attr":
            status = "bound_value" if node.name in program.lifted_bindings else "constant_value"
            operation = None
        elif node.op == "output":
            status = "output_tree"
            operation = None
        else:
            operation = operation_by_node.get(node.name)
            status = "indexed_op" if operation is not None and operation.kind != "Unsupported" else "diagnostic"
            if operation is None:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live FX node {node.name} has no indexed operation or diagnostic",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                ))
        coverage.append({"fx_node": node.name, "origin_ids": list(origin_ids),
                         "value_id": node_to_value.get(node.name), "status": status,
                         "op_id": operation.op_id if operation else None})

    source = program.source_graph_module or program.graph_module
    source_live = _live_nodes(source, effect_nodes)
    current_live_names = {node.name for node in live}
    for node in source.graph.nodes:
        if node not in source_live:
            continue
        for origin_id in getattr(node, "meta", {}).get(_ORIGIN_META, ()):
            descendants = tuple(name for name in program.origin_history.get(origin_id, ())
                                if name in current_live_names)
            if not descendants:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS,
                    f"live source FX origin {origin_id} was removed without replacement provenance",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                ))
                continue
            descendant_ops = tuple(operation_by_node[name].op_id for name in descendants
                                   if name in operation_by_node)
            if node.op not in {"placeholder", "get_attr", "output"} and len(descendant_ops) != len(descendants):
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live source FX origin {origin_id} expands to an unindexed node",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"normalized_nodes": list(descendants)},
                ))
            coverage.append({"source_fx_node": node.name, "source_origin_id": origin_id,
                             "normalized_nodes": list(descendants), "indexed_ops": list(descendant_ops),
                             "status": "indexed_origin_region" if descendant_ops else "source_value_or_output"})

    for effect in program.effects:
        if not effect.required:
            continue
        effect_origins = set(program.origin_map.get(effect.node_id, ()))
        covered = any(effect.origin_id in effect_origins for operation in operations for effect in operation.effect_edges)
        if not covered:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"required {effect.kind} effect at {effect.node_id} has no indexed owner",
                DiagnosticSeverity.ERROR, node_id=effect.node_id,
            ))

    output_keys = set(program.output_origins)
    seen_output_origins = set()
    for output in outputs:
        path_key = json.dumps(output.path, separators=(",", ":"), default=str)
        if output.origin_id is None or path_key not in output_keys:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"output path {path_key} has no stable origin",
                DiagnosticSeverity.ERROR, node_id="output",
            ))
        elif output.origin_id in seen_output_origins:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS, f"output origin {output.origin_id} is duplicated",
                DiagnosticSeverity.ERROR, node_id="output",
            ))
        seen_output_origins.add(output.origin_id)
        coverage.append({"fx_node": "output", "origin_ids": [output.origin_id] if output.origin_id else [],
                         "value_id": output.value_id, "path": list(output.path), "status": "output_leaf"})
    return tuple(coverage), tuple(diagnostics)
