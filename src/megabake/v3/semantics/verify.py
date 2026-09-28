"""Origin, output and effect coverage checks for the indexed frontend."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..frontend.capture import NormalizedProgram, _ORIGIN_META
from .indexed import IndexedOp, OutputLeaf, StateTransition


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


def _output_references(value: Any, path: tuple[Any, ...] = ()):
    if hasattr(value, "op") and hasattr(value, "name"):
        yield path, value.name, None
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _output_references(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _output_references(item, path + (index,))
    else:
        yield path, None, value


def verify_indexed_program(program: NormalizedProgram, operations: tuple[IndexedOp, ...],
                           node_to_value: Mapping[str, str],
                           outputs: tuple[OutputLeaf, ...] = (),
                           state_transitions: tuple[StateTransition, ...] = ()) -> tuple[
                               tuple[Mapping[str, Any], ...], tuple[DiagnosticRecord, ...]
                           ]:
    diagnostics: list[DiagnosticRecord] = []
    operations_by_node: dict[str, list[IndexedOp]] = {}
    operation_ids: set[str] = set()
    for operation in operations:
        operations_by_node.setdefault(operation.local_reference.node_name, []).append(operation)
        if operation.op_id in operation_ids:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"indexed operation ID {operation.op_id} is duplicated",
                DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
            ))
        operation_ids.add(operation.op_id)
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
            owners = operations_by_node.get(node.name, ())
            operation = owners[0] if len(owners) == 1 else None
            status = "indexed_op" if operation is not None and operation.kind != "Unsupported" else "diagnostic"
            if len(owners) > 1:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live FX node {node.name} has multiple indexed owners",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"op_ids": [item.op_id for item in owners]},
                ))
            elif operation is None:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live FX node {node.name} has no indexed operation or diagnostic",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"negative_case": "omitted live operation", "origin_ids": list(origin_ids)},
                ))
            elif operation.kind == "Unsupported":
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"live FX node {node.name} has no strict lowerer",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"target": operation.target, "negative_case": "reference-only region"},
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
            descendant_ops = tuple(
                owner.op_id for name in descendants
                for owner in operations_by_node.get(name, ())
            )
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
        owners = [(operation, edge) for operation in operations for edge in operation.effect_edges
                  if edge.origin_id in effect_origins and edge.kind == effect.kind and edge.target == effect.target]
        if not owners:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"required {effect.kind} effect at {effect.node_id} has no indexed owner",
                DiagnosticSeverity.ERROR, node_id=effect.node_id,
                details={"origin_ids": sorted(effect_origins), "target": effect.target},
            ))
        elif len(owners) != 1:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"required {effect.kind} effect at {effect.node_id} has duplicated indexed ownership",
                DiagnosticSeverity.ERROR, node_id=effect.node_id,
                details={"owner_count": len(owners)},
            ))

    transitions_by_effect: dict[str, StateTransition] = {}
    for transition in state_transitions:
        if transition.effect_id in transitions_by_effect:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"state effect {transition.effect_id} has duplicate transitions",
                DiagnosticSeverity.ERROR, node_id=transition.new_value,
            ))
        transitions_by_effect[transition.effect_id] = transition
        writers = [operation for operation in operations
                   if any(edge.effect_id == transition.effect_id and edge.writes == (transition.new_value,)
                          for edge in operation.effect_edges)]
        if len(writers) != 1 or transition.alias_rule != "functional_new_value":
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"state transition {transition.effect_id} has no unique functional writer",
                DiagnosticSeverity.ERROR, node_id=transition.new_value,
                details={"writer_count": len(writers), "alias_rule": transition.alias_rule},
            ))
    for operation in operations:
        for edge in operation.effect_edges:
            if edge.kind != "state_read_after_publish":
                continue
            for dependency in edge.depends_on:
                transition = transitions_by_effect.get(dependency)
                if transition is None or edge.reads != (transition.new_value,):
                    diagnostics.append(DiagnosticRecord(
                        DiagnosticCode.UNSUPPORTED_SEMANTICS,
                        f"state read at {operation.local_reference.node_name} is not ordered after its writer publication",
                        DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
                        details={"dependency": dependency, "negative_case": "read before publish"},
                    ))

    for operation in operations:
        for transition in state_transitions:
            if transition.new_value in operation.inputs and not any(
                    edge.kind == "state_read_after_publish" and transition.effect_id in edge.depends_on
                    for edge in operation.effect_edges):
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"state read at {operation.local_reference.node_name} is missing its writer publication dependency",
                    DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
                    details={"dependency": transition.effect_id, "negative_case": "read before publish"},
                ))

    output_keys = set(program.output_origins)
    seen_output_origins = set()
    output_paths: Counter[str] = Counter()
    for output in outputs:
        path_key = json.dumps(output.path, separators=(",", ":"), default=str)
        output_paths[path_key] += 1
        if output.origin_id is None or path_key not in output_keys:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"output path {path_key} has no stable origin",
                DiagnosticSeverity.ERROR, node_id="output",
            ))
        elif output.origin_id != program.output_origins[path_key]:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"output path {path_key} is bound to the wrong original FX origin",
                DiagnosticSeverity.ERROR, node_id="output",
                details={"expected_origin": program.output_origins[path_key],
                         "actual_origin": output.origin_id},
            ))
        elif output.origin_id in seen_output_origins:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS, f"output origin {output.origin_id} is duplicated",
                DiagnosticSeverity.ERROR, node_id="output",
            ))
        seen_output_origins.add(output.origin_id)
        coverage.append({"fx_node": "output", "origin_ids": [output.origin_id] if output.origin_id else [],
                         "value_id": output.value_id, "path": list(output.path), "status": "output_leaf"})
    for path_key in sorted(output_keys - set(output_paths)):
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            f"original output path {path_key} has no indexed output leaf",
            DiagnosticSeverity.ERROR, node_id="output",
            details={"negative_case": "omitted output leaf", "path": json.loads(path_key),
                     "origin_ids": [program.output_origins[path_key]]},
        ))
    for path_key, count in output_paths.items():
        if count > 1:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"output path {path_key} is covered {count} times",
                DiagnosticSeverity.ERROR, node_id="output",
                details={"negative_case": "duplicate output leaf", "count": count},
            ))
    return tuple(coverage), tuple(diagnostics)


def verify_algorithm_choice_cover(program: Any, choices: tuple[Any, ...],
                                  selected_choice_ids: tuple[str, ...], *,
                                  numerical_policy: Any = None) -> tuple[
                                      Mapping[str, Any], tuple[DiagnosticRecord, ...]
                                  ]:
    """Verify a selected guarded choice set covers indexed work, outputs and effects."""
    from ..algorithms.choices import choice_guard_failures

    diagnostics: list[DiagnosticRecord] = []
    choices_by_id: dict[str, Any] = {}
    for choice in choices:
        if choice.choice_id in choices_by_id:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"algorithm choice ID {choice.choice_id} is duplicated",
                DiagnosticSeverity.ERROR, action_id=choice.choice_id,
            ))
        choices_by_id[choice.choice_id] = choice

    if not selected_choice_ids:
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            "algorithm cover selects no choices",
            DiagnosticSeverity.ERROR, details={"negative_case": "empty selected cover"},
        ))
    if len(set(selected_choice_ids)) != len(selected_choice_ids):
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            "algorithm cover selects a choice more than once",
            DiagnosticSeverity.ERROR, details={"selected_choice_ids": list(selected_choice_ids)},
        ))

    selected = []
    for choice_id in selected_choice_ids:
        choice = choices_by_id.get(choice_id)
        if choice is None:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"selected algorithm choice {choice_id} does not exist",
                DiagnosticSeverity.ERROR, action_id=choice_id,
            ))
            continue
        failures = choice_guard_failures(program, choice, numerical_policy=numerical_policy)
        for failure in failures:
            code = (DiagnosticCode.FAILED_NUMERICAL_GATE if "numerical policy" in failure
                    or "reassociation" in failure else DiagnosticCode.MISSING_FACTS)
            diagnostics.append(DiagnosticRecord(
                code, f"algorithm choice {choice_id} rejected: {failure}",
                DiagnosticSeverity.ERROR, action_id=choice_id,
                details={"choice_id": choice_id, "guard_failure": failure},
            ))
        selected.append(choice)

    op_counts = Counter(op_id for choice in selected for op_id in choice.reference_expansion)
    expected_ops = {operation.op_id for operation in program.operations if operation.kind != "Unsupported"}
    for op_id in sorted(expected_ops - set(op_counts)):
        operation = next(item for item in program.operations if item.op_id == op_id)
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"live indexed operation {op_id} is not covered by the selected algorithm cover",
            DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
            details={"negative_case": "omitted residual or cast", "origin_ids": list(operation.origin_ids)},
        ))
    for op_id, count in op_counts.items():
        if op_id in expected_ops and count != 1:
            operation = next(item for item in program.operations if item.op_id == op_id)
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"live indexed operation {op_id} is covered {count} times",
                DiagnosticSeverity.ERROR, node_id=operation.local_reference.node_name,
                details={"negative_case": "duplicate operation coverage", "count": count,
                         "origin_ids": list(operation.origin_ids)},
            ))
        elif op_id not in expected_ops:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"algorithm cover references unknown or unsupported operation {op_id}",
                DiagnosticSeverity.ERROR, action_id=op_id,
            ))

    for item in program.diagnostics:
        diagnostics.append(item)

    output_paths = Counter(json.dumps(output.path, separators=(",", ":"), default=str)
                           for output in program.outputs)
    source_graph = program.source_program.graph_module
    source_output_node = next((node for node in reversed(tuple(source_graph.graph.nodes))
                               if node.op == "output"), None)
    source_output_refs = ({json.dumps(path, separators=(",", ":"), default=str): (node_name, literal)
                          for path, node_name, literal in _output_references(source_output_node.args[0])}
                         if source_output_node is not None else {})
    expected_output_paths = set(program.source_program.output_origins)
    if set(source_output_refs) != expected_output_paths:
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            "normalized source output tree differs from its recorded output paths",
            DiagnosticSeverity.ERROR, node_id="output",
            details={"source_paths": sorted(source_output_refs),
                     "recorded_paths": sorted(expected_output_paths)},
        ))
    expected_value_ids = program.source_program.value_ids
    for output in program.outputs:
        path_key = json.dumps(output.path, separators=(",", ":"), default=str)
        expected_origin = program.source_program.output_origins.get(path_key)
        if output.origin_id != expected_origin:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"output path {path_key} is bound to the wrong original FX origin",
                DiagnosticSeverity.ERROR, node_id="output",
                details={"path": list(output.path), "expected_origin": expected_origin,
                         "actual_origin": output.origin_id,
                         "origin_ids": [expected_origin] if expected_origin else []},
            ))
        source_ref = source_output_refs.get(path_key)
        if source_ref is not None:
            source_node_name, source_literal = source_ref
            expected_value = expected_value_ids.get(source_node_name) if source_node_name is not None else None
            if output.value_id != expected_value or (
                    source_node_name is None and output.literal != source_literal):
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS,
                    f"output path {path_key} references the wrong normalized source value",
                    DiagnosticSeverity.ERROR, node_id="output",
                    details={"path": list(output.path), "expected_value_id": expected_value,
                             "actual_value_id": output.value_id,
                             "origin_ids": [expected_origin] if expected_origin else []},
                ))
    for path in sorted(expected_output_paths - set(output_paths)):
        diagnostics.append(DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            f"original output path {path} is absent from the indexed program",
            DiagnosticSeverity.ERROR, node_id="output",
            details={"negative_case": "omitted final output", "origin_ids": [
                program.source_program.output_origins[path]
            ]},
        ))
    for path, count in output_paths.items():
        if count != 1 or path not in expected_output_paths:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"indexed output path {path} has invalid coverage count {count}",
                DiagnosticSeverity.ERROR, node_id="output",
                details={"count": count, "expected": path in expected_output_paths},
            ))
    values_with_writers = {value for operation in program.operations for value in operation.outputs}
    for output in program.outputs:
        if output.value_id in values_with_writers and output.value_id not in {
                value for choice in selected for value in choice.output_values}:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"output value {output.value_id} has no selected writer",
                DiagnosticSeverity.ERROR, node_id="output",
                details={"path": list(output.path), "negative_case": "omitted final output",
                         "origin_ids": [program.source_program.output_origins.get(
                             json.dumps(output.path, separators=(",", ":"), default=str), "")
                                        ]},
            ))

    selected_ops = {op_id for choice in selected for op_id in choice.reference_expansion}
    expected_effects = Counter(
        (origin_id, effect.kind, effect.target)
        for effect in program.source_program.effects if effect.required
        for origin_id in program.source_program.origin_map.get(effect.node_id, ())
    )
    actual_effects = Counter(
        (edge.origin_id, edge.kind, edge.target)
        for operation in program.operations if operation.op_id in selected_ops
        for edge in operation.effect_edges
        if edge.required and edge.kind != "state_read_after_publish"
    )
    for key, count in expected_effects.items():
        if actual_effects[key] != count or count != 1:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"required effect {key[1]} at origin {key[0]} has invalid selected ownership",
                DiagnosticSeverity.ERROR, node_id=next(
                    (effect.node_id for effect in program.source_program.effects
                     if key[0] in program.source_program.origin_map.get(effect.node_id, ())), None
                ), details={"origin_id": key[0], "expected_count": count,
                            "actual_count": actual_effects[key],
                            "negative_case": "omitted or duplicated effect"},
            ))
    for key, count in actual_effects.items():
        if key not in expected_effects:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"algorithm cover adds an undeclared observable effect at origin {key[0]}",
                DiagnosticSeverity.ERROR, details={"effect": list(key), "count": count},
            ))

    report = {
        "schema_version": 1,
        "indexed_program_hash": program.structural_hash,
        "selected_choice_ids": list(selected_choice_ids),
        "operation_coverage": {op_id: count for op_id, count in sorted(op_counts.items())},
        "output_paths": {path: count for path, count in sorted(output_paths.items())},
        "effect_coverage": {"/".join(str(item) for item in key): count
                            for key, count in sorted(actual_effects.items(), key=lambda item: str(item[0]))},
        "valid": not diagnostics,
        "diagnostics": [item.to_dict() for item in diagnostics],
    }
    return report, tuple(diagnostics)
