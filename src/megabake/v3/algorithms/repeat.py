"""Verified executable repeats over indexed FX spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..semantics.indexed import IndexedOp, IndexedTensorProgram


class RepeatRegionError(ValueError):
    def __init__(self, message: str, *, iteration: int | None = None,
                 operation_id: str | None = None, field: str | None = None):
        self.iteration = iteration
        self.operation_id = operation_id
        self.field = field
        location = f" layer={iteration}" if iteration is not None else ""
        location += f" operation={operation_id}" if operation_id is not None else ""
        location += f" field={field}" if field is not None else ""
        super().__init__(message + location)


@dataclass(frozen=True)
class CarriedValue:
    producer_iteration: int
    consumer_iteration: int
    value_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"producer_iteration": self.producer_iteration,
                "consumer_iteration": self.consumer_iteration, "value_id": self.value_id}


@dataclass(frozen=True)
class RepeatRegion:
    indexed_program_hash: str
    iteration_count: int
    iteration_variable: str
    per_iteration_operation_ids: tuple[tuple[str, ...], ...]
    per_iteration_parameter_bindings: tuple[Mapping[str, str], ...]
    per_iteration_state_bindings: tuple[Mapping[str, str], ...]
    entry_values: tuple[tuple[str, ...], ...]
    exit_values: tuple[tuple[str, ...], ...]
    result_values: tuple[tuple[tuple[Any, ...], str], ...]
    carried_values: tuple[CarriedValue, ...]
    branch_predicates: tuple[tuple[str, ...], ...]
    body_reference: tuple[str, ...]
    flat_fx_order: tuple[str, ...]

    def expand_to_flat(self) -> tuple[str, ...]:
        return self.flat_fx_order

    def execute(self, program: IndexedTensorProgram, inputs: Mapping[str, Any]) -> dict[tuple[Any, ...], Any]:
        """Run the captured local FX references in the verified per-layer loop."""
        if program.structural_hash != self.indexed_program_hash:
            raise RepeatRegionError("repeat region belongs to a different indexed program")
        by_value = {value.value_id: value for value in program.values}
        node_to_value = {value.fx_node: value.value_id for value in program.values}
        nodes = {node.name: node for node in program.source_program.graph_module.graph.nodes}
        environment: dict[str, Any] = {}
        for key, value in inputs.items():
            value_id = key if key in by_value else node_to_value.get(key)
            if value_id is None:
                raise KeyError(f"repeat input {key!r} is not an indexed value")
            environment[value_id] = value

        source = program.source_program
        attribute_graph = source.source_graph_module or source.graph_module
        for value in program.values:
            node = nodes[value.fx_node]
            binding = source.lifted_bindings.get(node.name)
            if node.op == "placeholder" and binding is not None and value.value_id not in environment:
                environment[value.value_id] = source.binding_values[binding.target]
            elif node.op == "get_attr" and value.value_id not in environment:
                if binding is not None:
                    environment[value.value_id] = source.binding_values[binding.target]
                else:
                    attribute = attribute_graph
                    for component in str(node.target).split("."):
                        attribute = getattr(attribute, component)
                    environment[value.value_id] = attribute

        operations = {operation.op_id: operation for operation in program.operations}
        for iteration in self.per_iteration_operation_ids:
            for op_id in iteration:
                operation = operations[op_id]
                result = operation.local_reference.evaluate(environment)
                if len(operation.outputs) != 1:
                    raise RepeatRegionError("repeat executor requires one indexed output per operation",
                                            operation_id=op_id, field="outputs")
                environment[operation.outputs[0]] = result
        return {path: environment[value_id] for path, value_id in self.result_values
                if value_id in environment}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "indexed_program_hash": self.indexed_program_hash,
            "iteration_count": self.iteration_count,
            "iteration_variable": self.iteration_variable,
            "per_iteration_operation_ids": [list(items) for items in self.per_iteration_operation_ids],
            "per_iteration_parameter_bindings": [dict(items) for items in self.per_iteration_parameter_bindings],
            "per_iteration_state_bindings": [dict(items) for items in self.per_iteration_state_bindings],
            "entry_values": [list(items) for items in self.entry_values],
            "exit_values": [list(items) for items in self.exit_values],
            "result_values": [[list(path), value_id] for path, value_id in self.result_values],
            "carried_values": [item.to_dict() for item in self.carried_values],
            "branch_predicates": [list(items) for items in self.branch_predicates],
            "body_reference": list(self.body_reference),
            "flat_fx_order": list(self.flat_fx_order),
        }


def _external_role(program: IndexedTensorProgram, value_id: str) -> tuple[str, str | None]:
    value = next(item for item in program.values if item.value_id == value_id)
    source = program.source_program
    state_id = source.state_bindings.get(value.fx_node)
    if state_id is None and value.alias_set and value.alias_set.startswith("state:"):
        state_id = value.alias_set.removeprefix("state:")
    if state_id is not None:
        return "state", state_id
    position_name = None
    if source.step_abi is not None:
        position_name = source.step_abi.position_and_valid_length.get("position_source")
    if value.fx_node == position_name:
        return "control", None
    binding = source.lifted_bindings.get(value.fx_node)
    if binding is not None:
        if binding.role in {"weight", "parameter"}:
            return "parameter", binding.target
        if binding.role == "state":
            return "state", binding.target
        return binding.role, binding.target
    node = next(node for node in source.graph_module.graph.nodes if node.name == value.fx_node)
    return ("activation" if node.op in {"placeholder", "call_function", "call_module"} else "external"), None


def _region_signature(program: IndexedTensorProgram, operations: tuple[IndexedOp, ...],
                      entries: tuple[str, ...]) -> tuple[Any, ...]:
    internal: dict[str, tuple[int, int]] = {}
    for op_index, operation in enumerate(operations):
        for output_index, value_id in enumerate(operation.outputs):
            internal[value_id] = (op_index, output_index)
    entry_index = {value_id: index for index, value_id in enumerate(entries)}
    entry_roles = {value_id: _external_role(program, value_id)[0] for value_id in entries}

    def label(value_id: str) -> tuple[Any, ...]:
        if value_id in internal:
            return ("body",) + internal[value_id]
        if value_id in entry_index:
            return ("arg", entry_roles[value_id], entry_index[value_id])
        raise RepeatRegionError("repeat operation uses an unbound boundary value", field=value_id)

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"value"}:
                return {"value": label(str(value["value"]))}
            return {str(key): canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)

    def operation_signature(operation: IndexedOp) -> tuple[Any, ...]:
        dtype = dict(operation.dtype_expression)
        dtype["cast_origin"] = bool(dtype.get("cast_origin"))
        attrs = dict(operation.attributes)
        transition = attrs.pop("state_transition", None)
        if transition is not None:
            attrs["state_transition"] = {
                "old_value": label(transition["old_value"]),
                "new_value": label(transition["new_value"]),
                "index_value": label(transition["index_value"]),
                "valid_length_after": transition["valid_length_after"],
                "capacity": transition["capacity"], "axis": transition["axis"],
                "write_footprint": transition["write_footprint"],
                "alias_rule": transition["alias_rule"],
            }
        if "index_bounds" in attrs:
            attrs["index_bounds"] = {key: item for key, item in attrs["index_bounds"].items()
                                      if key != "value_id"}
        effects = tuple((edge.kind, edge.target is not None, edge.required,
                         tuple(label(value) for value in edge.reads),
                         tuple(label(value) for value in edge.writes),
                         len(edge.depends_on), edge.alias_rule)
                        for edge in operation.effect_edges)
        aliases = tuple((label(edge.value_id),
                         "activation_boundary" if edge.value_id in entry_roles and
                         entry_roles[edge.value_id] == "activation" else edge.relation,
                         tuple(label(value) for value in edge.sources))
                        for edge in operation.alias_edges)
        return (
            operation.kind, operation.target,
            tuple((axis.name, axis.extent) for axis in operation.iteration_domain),
            tuple((axis.name, axis.extent) for axis in operation.reduction_domain),
            tuple((item.mode, tuple(item.expressions), label(item.value_id)) for item in operation.input_index_maps),
            operation.output_index_map, operation.predicate_bounds,
            canonical(dtype), canonical(attrs), aliases, effects,
        )

    return tuple(operation_signature(operation) for operation in operations)


def _span_data(program: IndexedTensorProgram, op_ids: tuple[str, ...]):
    by_id = {operation.op_id: operation for operation in program.operations}
    operations = tuple(by_id[op_id] for op_id in op_ids)
    internal = {value for operation in operations for value in operation.outputs}
    entries = tuple(dict.fromkeys(value for operation in operations for value in operation.inputs
                                  if value not in internal))
    region_nodes = {operation.local_reference.node_name for operation in operations}
    outputs = {leaf.value_id for leaf in program.outputs if leaf.value_id is not None}
    value_info = {value.value_id: value for value in program.values}
    exits = tuple(value for operation in operations for value in operation.outputs
                  if value in outputs or any(user not in region_nodes
                                             for user in value_info[value].consumers))
    exits = tuple(dict.fromkeys(exits))
    for operation in operations:
        if operation.kind in {"Unsupported", "Scan/Branch"}:
            raise RepeatRegionError("unsupported indexed operation cannot enter a repeat region",
                                    operation_id=operation.op_id, field="kind")
    # Require a connected layer graph; common boundary inputs also connect parallel branches.
    adjacency = {index: set() for index in range(len(operations))}
    for left in range(len(operations)):
        for right in range(left + 1, len(operations)):
            first, second = operations[left], operations[right]
            if (set(first.outputs).intersection(second.inputs) or
                    set(second.outputs).intersection(first.inputs) or
                    set(first.inputs).intersection(second.inputs)):
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached = {0}
    pending = [0]
    while pending:
        for neighbor in adjacency[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if len(reached) != len(operations):
        raise RepeatRegionError("candidate layer span is not graph-connected", field="connectivity")

    roles = [_external_role(program, value_id) for value_id in entries]
    parameters: dict[str, str] = {}
    states: dict[str, str] = {}
    for index, (value_id, (role, binding)) in enumerate(zip(entries, roles)):
        key = f"arg{index}"
        if role == "parameter" and binding is not None:
            parameters[key] = binding
        elif role == "state" and binding is not None:
            states[key] = binding
    predicates = tuple(f"{operation.op_id}:{operation.attributes.get('guard_kind')}"
                       for operation in operations if operation.kind == "Guard")
    return operations, entries, exits, parameters, states, predicates


def recover_repeat_region(program: IndexedTensorProgram,
                          layer_spans: Sequence[Sequence[str]], *,
                          iteration_variable: str = "layer") -> RepeatRegion:
    """Verify caller-suggested layer spans against indexed graph structure and bindings."""
    if not program.strict_supported:
        raise RepeatRegionError("repeat recovery requires a strict-supported indexed program",
                                field="strict_supported")
    if len(layer_spans) < 2:
        raise RepeatRegionError("a repeat region requires at least two layer spans")
    by_id = {operation.op_id: operation for operation in program.operations}
    by_node = {operation.local_reference.node_name: operation.op_id for operation in program.operations}
    spans: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for iteration, candidate in enumerate(layer_spans):
        resolved = []
        for name in candidate:
            op_id = name if name in by_id else by_node.get(name)
            if op_id is None:
                raise RepeatRegionError("layer span refers to an unknown indexed operation",
                                        iteration=iteration, operation_id=name, field="operation_ids")
            if op_id in seen:
                raise RepeatRegionError("operation appears in more than one repeat iteration",
                                        iteration=iteration, operation_id=op_id, field="operation_ids")
            seen.add(op_id)
            resolved.append(op_id)
        if not resolved:
            raise RepeatRegionError("repeat iteration has no operations", iteration=iteration)
        spans.append(tuple(resolved))

    positions = {operation.op_id: index for index, operation in enumerate(program.operations)}
    flat = tuple(sorted(seen, key=positions.__getitem__))
    if tuple(op_id for span in spans for op_id in span) != flat:
        raise RepeatRegionError("layer spans do not preserve the original flat FX order", field="flat_order")
    flat_positions = [positions[op_id] for op_id in flat]
    if flat_positions != list(range(flat_positions[0], flat_positions[-1] + 1)):
        raise RepeatRegionError("repeat region contains a gap between its layer spans",
                                operation_id=flat[0], field="flat_order")
    layer_data = [_span_data(program, span) for span in spans]
    for iteration, (operations, *_rest) in enumerate(layer_data):
        indexes = [positions[operation.op_id] for operation in operations]
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise RepeatRegionError("layer span omits or interleaves indexed FX operations",
                                    iteration=iteration, operation_id=operations[0].op_id, field="connectivity")

    reference_operations = layer_data[0][0]
    reference_entries = layer_data[0][1]
    reference = _region_signature(program, reference_operations, reference_entries)
    for iteration, (operations, entries, *_rest) in enumerate(layer_data[1:], 1):
        candidate = _region_signature(program, operations, entries)
        if len(candidate) != len(reference):
            raise RepeatRegionError("repeated layer operation counts differ", iteration=iteration,
                                    operation_id=operations[0].op_id, field="operation_count")
        for index, (expected, actual) in enumerate(zip(reference, candidate)):
            if expected != actual:
                raise RepeatRegionError("repeated layer semantics differ", iteration=iteration,
                                        operation_id=operations[index].op_id, field="semantic_signature")

    carried = []
    for producer_iteration, (operations, *_rest) in enumerate(layer_data[:-1]):
        produced = {value for operation in operations for value in operation.outputs}
        for consumer_iteration, (_, entries, *_tail) in enumerate(layer_data[producer_iteration + 1:],
                                                                  producer_iteration + 1):
            carried.extend(CarriedValue(producer_iteration, consumer_iteration, value)
                           for value in entries if value in produced)

    region_values = {value for span, *_rest in layer_data
                     for operation in span for value in operation.outputs}
    result_values = tuple((output.path, output.value_id) for output in program.outputs
                          if output.value_id is not None and output.value_id in region_values)

    return RepeatRegion(
        indexed_program_hash=program.structural_hash,
        iteration_count=len(spans), iteration_variable=iteration_variable,
        per_iteration_operation_ids=tuple(spans),
        per_iteration_parameter_bindings=tuple(item[3] for item in layer_data),
        per_iteration_state_bindings=tuple(item[4] for item in layer_data),
        entry_values=tuple(item[1] for item in layer_data),
        exit_values=tuple(item[2] for item in layer_data),
        result_values=result_values,
        carried_values=tuple(carried),
        branch_predicates=tuple(item[5] for item in layer_data),
        body_reference=tuple(reference_operation.op_id for reference_operation in reference_operations),
        flat_fx_order=flat,
    )


__all__ = ["CarriedValue", "RepeatRegion", "RepeatRegionError", "recover_repeat_region"]
