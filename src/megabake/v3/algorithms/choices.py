"""Backend-neutral, guarded algorithm alternatives for indexed FX work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import ContractError
from ..frontend.capture import graph_hash
from ..frontend.facts import collect_facts
from ..frontend.semantic import SemanticNode, recognize
from ..semantics.indexed import IndexedOp, IndexedTensorProgram


def _stable(value: Any) -> Any:
    if hasattr(value, "op") and hasattr(value, "name"):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


@dataclass(frozen=True)
class AlgorithmChoice:
    """One guarded implementation idea with an executable indexed expansion."""

    choice_id: str
    algorithm: str
    operation_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    effect_owners: tuple[Mapping[str, Any], ...]
    guards: Mapping[str, Any]
    reference_expansion: tuple[str, ...]
    preparation_actions: tuple[Mapping[str, Any], ...] = ()
    numerical_requirements: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice_id": self.choice_id,
            "algorithm": self.algorithm,
            "operation_ids": list(self.operation_ids),
            "origin_ids": list(self.origin_ids),
            "input_values": list(self.input_values),
            "output_values": list(self.output_values),
            "effect_owners": [_stable(item) for item in self.effect_owners],
            "guards": _stable(self.guards),
            "reference_expansion": list(self.reference_expansion),
            "preparation_actions": [_stable(item) for item in self.preparation_actions],
            "numerical_requirements": _stable(self.numerical_requirements),
        }


def _semantic_snapshot(operation: SemanticNode) -> dict[str, Any]:
    return {
        "name": operation.name,
        "version": operation.version,
        "origin_nodes": list(operation.origin_nodes),
        "attributes": _stable(operation.attributes),
        "effects": list(operation.effects),
    }


def _policy(program: IndexedTensorProgram, explicit: Any) -> Any:
    return explicit if explicit is not None else program.source_program.policy


def _make_choice(program: IndexedTensorProgram, algorithm: str, operations: Sequence[IndexedOp], *,
                 semantic: SemanticNode | None = None,
                 guard_conditions: Sequence[Mapping[str, Any]] = (),
                 preparation_actions: Sequence[Mapping[str, Any]] = (),
                 numerical_requirements: Mapping[str, Any] | None = None,
                 policy: Any = None) -> AlgorithmChoice:
    positions = {operation.op_id: index for index, operation in enumerate(program.operations)}
    ordered = tuple(sorted(operations, key=lambda item: positions[item.op_id]))
    op_ids = tuple(operation.op_id for operation in ordered)
    input_values = tuple(dict.fromkeys(
        value for operation in ordered for value in operation.inputs
        if not any(value in producer.outputs for producer in ordered)
    ))
    output_values = tuple(dict.fromkeys(value for operation in ordered for value in operation.outputs))
    origins = tuple(dict.fromkeys(origin for operation in ordered for origin in operation.origin_ids))
    effects = tuple(
        {"origin_id": edge.origin_id, "kind": edge.kind, "target": edge.target,
         "required": edge.required, "effect_id": edge.effect_id, "reads": list(edge.reads),
         "writes": list(edge.writes), "depends_on": list(edge.depends_on),
         "alias_rule": edge.alias_rule}
        for operation in ordered for edge in operation.effect_edges
    )
    values = {value.value_id: value for value in program.values}
    guard_values = tuple(dict.fromkeys(input_values + output_values))
    policy_hash = getattr(policy, "contract_hash", None)
    requirements = dict(numerical_requirements or {})
    preparations = tuple(_stable(item) for item in preparation_actions)
    guards = {
        "choice_id": f"{algorithm}:{'+'.join(op_ids)}",
        "algorithm": algorithm,
        "indexed_program_hash": program.structural_hash,
        "source_graph_hash": graph_hash(program.source_program),
        "reference_expansion": list(op_ids),
        "operations": [operation.to_dict() for operation in ordered],
        "values": [values[value_id].to_dict() for value_id in guard_values if value_id in values],
        "semantic_match": _semantic_snapshot(semantic) if semantic is not None else None,
        "preparation_actions": list(preparations),
        "numerical_requirements": _stable(requirements),
        "numerical_policy_hash": policy_hash,
        "conditions": [
            {"kind": "indexed_kind", "op_id": operation.op_id, "equals": operation.kind}
            for operation in ordered
        ] + [_stable(item) for item in guard_conditions],
    }
    return AlgorithmChoice(
        choice_id=f"{algorithm}:{'+'.join(op_ids)}", algorithm=algorithm,
        operation_ids=op_ids, origin_ids=origins, input_values=input_values,
        output_values=output_values, effect_owners=effects, guards=guards,
        reference_expansion=op_ids, preparation_actions=preparations,
        numerical_requirements=_stable(requirements),
    )


def _region_operations(program: IndexedTensorProgram, node_names: Sequence[str], *,
                       include_unsupported: bool = False) -> tuple[IndexedOp, ...]:
    wanted = set(node_names)
    return tuple(operation for operation in program.operations
                 if operation.local_reference.node_name in wanted
                 and (include_unsupported or operation.kind != "Unsupported"))


def _add_semantic_choice(program: IndexedTensorProgram, result: list[AlgorithmChoice],
                         semantic: SemanticNode, algorithm: str, *, policy: Any = None,
                         numerical_requirements: Mapping[str, Any] | None = None,
                         include_unsupported: bool = False) -> None:
    operations = _region_operations(program, semantic.origin_nodes, include_unsupported=include_unsupported)
    if operations and len({operation.local_reference.node_name for operation in operations}) == len(
            set(semantic.origin_nodes)):
        result.append(_make_choice(program, algorithm, operations, semantic=semantic,
                                   numerical_requirements=numerical_requirements, policy=policy))


def _linear_for_value(semantic_ops: Sequence[SemanticNode], value_id: str) -> SemanticNode | None:
    return next((operation for operation in semantic_ops
                 if operation.name == "Linear" and value_id in operation.outputs), None)


def _stable_weight(program: IndexedTensorProgram, value_id: str) -> bool:
    value = next((item for item in program.values if item.value_id == value_id), None)
    return bool(value is not None and value.role in {"weight", "parameter", "buffer", "constant"}
                and value.alias_kind == "tied_binding" and not value.effects)


def enumerate_algorithm_choices(program: IndexedTensorProgram, *, numerical_policy: Any = None,
                                repeat_regions: Sequence[Any] = ()) -> tuple[AlgorithmChoice, ...]:
    """Return overlapping guarded choices and the per-op unfused control.

    This pass consumes only FX/indexed facts. Target legality and tactic cost are
    deliberately deferred to backend providers.
    """
    policy = _policy(program, numerical_policy)
    choices: list[AlgorithmChoice] = []
    by_id = {operation.op_id: operation for operation in program.operations}
    # Every lowerable operation keeps an exact, one-op fallback.
    for operation in program.operations:
        if operation.kind == "Attention":
            choices.append(_make_choice(program, "materialized_attention", (operation,), policy=policy))
            if (policy is not None and callable(getattr(policy, "tolerance_for", None))
                    and policy.reassociation_allowed("scaled_dot_product_attention")
                    and operation.attributes.get("attention")):
                try:
                    dtype = next(value.dtype for value in program.values
                                 if value.value_id == operation.outputs[0])
                    policy.tolerance_for("scaled_dot_product_attention", dtype)
                except (StopIteration, ContractError):
                    pass
                else:
                    choices.append(_make_choice(
                        program, "online_softmax", (operation,), policy=policy,
                        numerical_requirements={"reassociation": "required",
                                                "operation": "scaled_dot_product_attention"},
                    ))
        elif operation.kind != "Unsupported":
            choices.append(_make_choice(program, "indexed", (operation,), policy=policy))

    for operation in program.operations:
        if operation.kind != "Contraction":
            continue
        base = (operation,)
        if len(operation.iteration_domain) >= 2 and operation.reduction_domain:
            choices.append(_make_choice(
                program, "projection_output_major", base, policy=policy,
                guard_conditions=({"kind": "contraction_rank_at_least", "op_id": operation.op_id,
                                   "rank": 2, "has_reduction": True},),
                numerical_requirements={"reassociation": "preserved", "orientation": "output_major"},
            ))
            reassociation_key = str(operation.attributes.get("operator_name", "contraction"))
            choices.append(_make_choice(
                program, "projection_k_parallel", base, policy=policy,
                guard_conditions=({"kind": "contraction_rank_at_least", "op_id": operation.op_id,
                                   "rank": 2, "has_reduction": True},),
                numerical_requirements={"reassociation": "required", "operation": reassociation_key},
            ))
            weight = (next((value for value in program.values
                            if value.value_id == operation.inputs[1]), None)
                      if len(operation.inputs) > 1 else None)
            if weight is not None and _stable_weight(program, weight.value_id):
                choices.append(_make_choice(
                    program, "projection_transposed_tensorcore", base, policy=policy,
                    guard_conditions=({"kind": "contraction_rank_at_least", "op_id": operation.op_id,
                                       "rank": 2, "has_reduction": True},
                                      {"kind": "stable_binding", "value_id": weight.value_id}),
                    preparation_actions=({"kind": "pack_weight", "value_id": weight.value_id,
                                          "layout": "transposed_tensorcore", "lifetime": "session",
                                          "cost": "UNKNOWN", "extra_storage": "UNKNOWN"},),
                    numerical_requirements={"reassociation": "required", "operation": reassociation_key,
                                            "orientation": "transposed_weight"},
                ))
                choices.append(_make_choice(
                    program, "stable_weight_pack", base, policy=policy,
                    guard_conditions=({"kind": "stable_binding", "value_id": weight.value_id},),
                    preparation_actions=({"kind": "pack_weight", "value_id": weight.value_id,
                                          "lifetime": "session", "cost": "UNKNOWN",
                                          "extra_storage": "UNKNOWN"},),
                    numerical_requirements={"reassociation": "preserved", "layout_copy": "value_exact"},
                ))

    semantic_graph = recognize(program.source_program, facts=collect_facts(program.source_program))
    semantic_ops = semantic_graph.operations
    for semantic in semantic_ops:
        if semantic.name == "RMSNorm":
            _add_semantic_choice(program, choices, semantic, "rmsnorm_exact", policy=policy,
                                 numerical_requirements={"reassociation": "preserved",
                                                         "cast_order": "reference_exact"})
        elif semantic.name == "SDPA":
            _add_semantic_choice(program, choices, semantic, "attention_online_softmax", policy=policy,
                                 numerical_requirements={"reassociation": "required",
                                                         "operation": "softmax"},
                                 include_unsupported=True)
        elif semantic.name == "RoPE":
            _add_semantic_choice(program, choices, semantic, "rope_exact", policy=policy,
                                 numerical_requirements={"reassociation": "preserved"})
        elif semantic.name == "SwiGLU":
            _add_semantic_choice(program, choices, semantic, "swiglu_exact", policy=policy,
                                 numerical_requirements={"activation": semantic.attributes.get("activation"),
                                                         "cast_order": "reference_exact"})

    # Keep separate Q/K/V operations and add their packed alternative.
    from ..frontend.composites import enumerate_composites
    for composite in enumerate_composites(semantic_graph):
        if composite.kind == "qkv_group":
            semantic_group = tuple(item for item in semantic_ops if item.op_id in composite.covered_ops)
            operations = _region_operations(program, tuple(
                node for item in semantic_group for node in item.origin_nodes
            ))
            if operations:
                weights = tuple(item.inputs[1] for item in semantic_group
                                if item.name == "Linear" and len(item.inputs) > 1)
                stable_weights = bool(weights) and all(_stable_weight(program, value_id) for value_id in weights)
                choices.append(_make_choice(program, "qkv_separate", operations, policy=policy,
                                            guard_conditions=({"kind": "same_input", "operation_ids":
                                                               [item.op_id for item in operations],
                                                               "input_index": 0},
                                                              {"kind": "semantic_group", "matches":
                                                               [_semantic_snapshot(item) for item in semantic_group]}),
                                            numerical_requirements={"output_partition": "reference_exact"}))
                if stable_weights:
                    choices.append(_make_choice(
                        program, "qkv_packed", operations, policy=policy,
                        guard_conditions=({"kind": "same_input", "operation_ids":
                                           [item.op_id for item in operations], "input_index": 0},
                                          {"kind": "stable_binding", "value_ids": list(weights)},
                                          {"kind": "semantic_group", "matches":
                                           [_semantic_snapshot(item) for item in semantic_group]}),
                        preparation_actions=({"kind": "pack_qkv_weights", "lifetime": "session",
                                              "cost": "UNKNOWN", "extra_storage": "UNKNOWN"},),
                        numerical_requirements={"output_partition": "reference_exact",
                                                "reassociation": "preserved"},
                    ))
        elif composite.kind in {"linear_epilogue", "norm_linear", "rope_cache"}:
            semantic_group = tuple(item for item in semantic_ops if item.op_id in composite.covered_ops)
            operations = _region_operations(program, tuple(
                node for item in semantic_group for node in item.origin_nodes
            ))
            if operations:
                choices.append(_make_choice(program, composite.kind, operations, policy=policy,
                                            guard_conditions=({"kind": "semantic_group", "matches":
                                                               [_semantic_snapshot(item) for item in semantic_group]},),
                                            numerical_requirements={"reference_relation": "exact"}))

    # The gate input is inside the matched SiLU region, so resolve its two
    # projection producers from the match attributes rather than graph adjacency.
    value_by_node = {value.fx_node: value.value_id for value in program.values}
    for swiglu in (item for item in semantic_ops if item.name == "SwiGLU"):
        gate = _linear_for_value(semantic_ops, value_by_node.get(swiglu.attributes.get("gate"), ""))
        up = _linear_for_value(semantic_ops, value_by_node.get(swiglu.attributes.get("up"), ""))
        if gate is None or up is None:
            continue
        down = next((item for item in semantic_ops if item.name == "Linear"
                     and any(output in item.inputs for output in swiglu.outputs)), None)
        region = [gate, up, swiglu] + ([down] if down is not None else [])
        operations = _region_operations(program, tuple(
            node for item in region for node in item.origin_nodes
        ))
        if not operations:
            continue
        choices.append(_make_choice(program, "gated_mlp_full", operations, policy=policy,
                                    semantic=swiglu,
                                    numerical_requirements={"activation": "silu",
                                                            "hidden_materialization": "full",
                                                            "cast_order": "reference_exact"}))
        if down is not None:
            down_op = next((item for item in operations if item.op_id in {
                candidate.op_id for candidate in program.operations
                if candidate.local_reference.node_name == down.origin_nodes[-1]
            }), None)
            choices.append(_make_choice(
                program, "gated_mlp_streamed", operations, policy=policy,
                semantic=swiglu,
                numerical_requirements={"activation": "silu", "hidden_materialization": "streamed",
                                        "reassociation": "required",
                                        "operation": (down_op.attributes.get("operator_name")
                                                      if down_op is not None else "contraction")},
            ))

    for repeat in repeat_regions:
        flat_nodes = tuple(repeat.expand_to_flat())
        operations = tuple(by_id[op_id] for op_id in flat_nodes if op_id in by_id)
        if operations and len(operations) == len(flat_nodes):
            choices.append(_make_choice(
                program, "repeat_unroll", operations, policy=policy,
                numerical_requirements={"flat_fx_order": list(flat_nodes), "reassociation": "preserved"},
            ))

    # Choice IDs are deterministic; exact duplicates can arise when a matcher
    # and composite describe the same rule.
    unique: dict[str, AlgorithmChoice] = {}
    for choice in choices:
        unique.setdefault(choice.choice_id, choice)
    return tuple(unique.values())


def choice_guard_failures(program: IndexedTensorProgram, choice: AlgorithmChoice, *,
                          numerical_policy: Any = None) -> tuple[str, ...]:
    """Check the stored structural/fact/numerical guards without backend queries."""
    failures: list[str] = []
    if choice.choice_id != choice.guards.get("choice_id"):
        failures.append("choice identity changed")
    if choice.algorithm != choice.guards.get("algorithm"):
        failures.append("algorithm choice changed")
    if choice.guards.get("indexed_program_hash") != program.structural_hash:
        failures.append("indexed program hash changed")
    if choice.guards.get("source_graph_hash") != graph_hash(program.source_program):
        failures.append("source graph hash changed")
    current_ops = {operation.op_id: operation for operation in program.operations}
    guarded_operation_records = tuple(choice.guards.get("operations", ()))
    guarded_operation_order = tuple(item.get("op_id") for item in guarded_operation_records)
    expected_ops = {item.get("op_id"): item for item in guarded_operation_records}
    if tuple(choice.operation_ids) != tuple(choice.reference_expansion):
        failures.append("reference expansion differs from covered operation order")
    if tuple(choice.reference_expansion) != tuple(choice.guards.get("reference_expansion", ())):
        failures.append("guarded reference expansion changed")
    if tuple(choice.operation_ids) != guarded_operation_order:
        failures.append("guarded indexed operation order changed")
    program_order = tuple(operation.op_id for operation in program.operations
                           if operation.op_id in set(choice.operation_ids))
    if tuple(choice.operation_ids) != program_order:
        failures.append("reference expansion differs from indexed program order")
    if len(set(choice.operation_ids)) != len(choice.operation_ids):
        failures.append("choice repeats an indexed operation")
    for op_id in choice.operation_ids:
        operation = current_ops.get(op_id)
        expected = expected_ops.get(op_id)
        if operation is None or expected is None or operation.to_dict() != expected:
            failures.append(f"indexed operation guard failed: {op_id}")
    if set(expected_ops) != set(choice.operation_ids):
        failures.append("guard operation set differs from reference expansion")
    ordered = tuple(current_ops[op_id] for op_id in choice.operation_ids if op_id in current_ops)
    expected_origins = tuple(dict.fromkeys(origin for operation in ordered for origin in operation.origin_ids))
    expected_inputs = tuple(dict.fromkeys(
        value for operation in ordered for value in operation.inputs
        if not any(value in producer.outputs for producer in ordered)
    ))
    expected_outputs = tuple(dict.fromkeys(value for operation in ordered for value in operation.outputs))
    expected_effects = tuple(
        {"origin_id": edge.origin_id, "kind": edge.kind, "target": edge.target,
         "required": edge.required, "effect_id": edge.effect_id, "reads": list(edge.reads),
         "writes": list(edge.writes), "depends_on": list(edge.depends_on),
         "alias_rule": edge.alias_rule}
        for operation in ordered for edge in operation.effect_edges
    )
    if choice.origin_ids != expected_origins:
        failures.append("candidate origin set differs from its reference expansion")
    if choice.input_values != expected_inputs:
        failures.append("candidate input boundaries differ from its reference expansion")
    if choice.output_values != expected_outputs:
        failures.append("candidate outputs differ from its reference expansion")
    if tuple(_stable(item) for item in choice.effect_owners) != tuple(_stable(item) for item in expected_effects):
        failures.append("candidate effect ownership differs from its reference expansion")
    if list(_stable(item) for item in choice.preparation_actions) != choice.guards.get("preparation_actions"):
        failures.append("preparation action or cost record changed")
    if _stable(choice.numerical_requirements) != choice.guards.get("numerical_requirements"):
        failures.append("candidate numerical requirements changed")
    current_values = {value.value_id: value for value in program.values}
    expected_values = {item.get("value_id"): item for item in choice.guards.get("values", ())}
    for value_id, expected in expected_values.items():
        if value_id not in current_values or current_values[value_id].to_dict() != expected:
            failures.append(f"value fact guard failed: {value_id}")
    if set(expected_values) != set(choice.input_values + choice.output_values):
        failures.append("guard value set differs from candidate inputs and outputs")

    for condition in choice.guards.get("conditions", ()):
        kind = condition.get("kind")
        if kind == "indexed_kind":
            operation = current_ops.get(condition.get("op_id"))
            if operation is None or operation.kind != condition.get("equals"):
                failures.append(f"indexed kind guard failed: {condition.get('op_id')}")
        elif kind == "contraction_rank_at_least":
            operation = current_ops.get(condition.get("op_id"))
            if (operation is None or operation.kind != "Contraction"
                    or len(operation.iteration_domain) < condition.get("rank", 0)
                    or condition.get("has_reduction") and not operation.reduction_domain):
                failures.append(f"contraction rank guard failed: {condition.get('op_id')}")
        elif kind == "stable_binding":
            value_ids = condition.get("value_ids", [condition.get("value_id")])
            for value_id in value_ids:
                value = current_values.get(value_id)
                if (value is None or value.role not in {"weight", "parameter", "buffer", "constant"}
                        or value.alias_kind != "tied_binding" or value.effects):
                    failures.append(f"stable binding guard failed: {value_id}")
        elif kind == "same_input":
            operations = [current_ops.get(op_id) for op_id in condition.get("operation_ids", ())]
            input_index = condition.get("input_index", 0)
            if (not operations or any(operation is None or len(operation.inputs) <= input_index
                                      for operation in operations)
                    or len({operation.inputs[input_index] for operation in operations if operation}) != 1):
                failures.append("shared-input projection guard failed")
        elif kind == "semantic_group":
            graph = recognize(program.source_program, facts=collect_facts(program.source_program))
            actual = tuple(_semantic_snapshot(item) for item in graph.operations)
            if any(not any(expected == found for found in actual)
                   for expected in condition.get("matches", ())):
                failures.append("named matcher group no longer proves the recorded semantic region")
        else:
            failures.append(f"unknown machine guard condition: {kind}")

    policy = _policy(program, numerical_policy)
    expected_policy_hash = choice.guards.get("numerical_policy_hash")
    if expected_policy_hash is not None and getattr(policy, "contract_hash", None) != expected_policy_hash:
        failures.append("numerical policy hash changed")
    if choice.numerical_requirements.get("reassociation") == "required":
        key = str(choice.numerical_requirements.get("operation", "*"))
        if policy is None or expected_policy_hash is None or not policy.reassociation_allowed(key):
            failures.append(f"numerical policy does not permit reassociation for {key}")

    semantic = choice.guards.get("semantic_match")
    if semantic is not None:
        graph = recognize(program.source_program, facts=collect_facts(program.source_program))
        expected = semantic
        found = any(_semantic_snapshot(item) == expected for item in graph.operations)
        if not found:
            failures.append("named matcher no longer proves the recorded semantic region")
    return tuple(failures)


__all__ = ["AlgorithmChoice", "choice_guard_failures", "enumerate_algorithm_choices"]
