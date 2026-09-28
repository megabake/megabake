"""Small local FX-region evaluator used as an indexed-op oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _is_node(value: Any) -> bool:
    return hasattr(value, "op") and hasattr(value, "name")


def _target_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return f"{schema.name}.{schema.overload_name or 'default'}"
    return ".".join(filter(None, (getattr(target, "__module__", None),
                                  getattr(target, "__qualname__", getattr(target, "__name__", None))))) or type(target).__name__


def _json_arg(value: Any, node_to_value: Mapping[str, str]) -> Any:
    if _is_node(value):
        return {"value": node_to_value.get(value.name, value.name)}
    if isinstance(value, Mapping):
        return {str(key): _json_arg(item, node_to_value) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_arg(item, node_to_value) for item in value]
    if isinstance(value, slice):
        return {"slice": [_json_arg(value.start, node_to_value), _json_arg(value.stop, node_to_value),
                          _json_arg(value.step, node_to_value)]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value_type = type(value)
    if value_type.__module__.startswith("torch"):
        return {"type": value_type.__name__, "value": str(value)}
    return {"type": f"{value_type.__module__}.{value_type.__qualname__}"}


def resolve_fx(value: Any, environment: Mapping[str, Any], node_to_value: Mapping[str, str]) -> Any:
    if _is_node(value):
        value_id = node_to_value.get(value.name)
        if value_id is None or value_id not in environment:
            raise KeyError(f"local FX region is missing boundary value {value.name!r}")
        return environment[value_id]
    if isinstance(value, tuple):
        return tuple(resolve_fx(item, environment, node_to_value) for item in value)
    if isinstance(value, list):
        return [resolve_fx(item, environment, node_to_value) for item in value]
    if isinstance(value, dict):
        return {key: resolve_fx(item, environment, node_to_value) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class LocalFXReference:
    """One FX node with only its explicit boundary values and static arguments."""

    node_name: str
    node_op: str
    target_name: str
    target: Any = field(compare=False, repr=False)
    args: Any = field(compare=False, repr=False)
    kwargs: Any = field(compare=False, repr=False)
    node_to_value: Mapping[str, str] = field(compare=False, repr=False)
    module: Any = field(default=None, compare=False, repr=False)

    def evaluate(self, environment: Mapping[str, Any]) -> Any:
        args = resolve_fx(self.args, environment, self.node_to_value)
        kwargs = resolve_fx(self.kwargs, environment, self.node_to_value)
        if self.node_op == "call_function":
            return self.target(*args, **kwargs)
        if self.node_op == "call_method":
            return getattr(args[0], str(self.target))(*args[1:], **kwargs)
        if self.node_op == "call_module":
            return self.module(*args, **kwargs)
        raise TypeError(f"{self.node_name}: {self.node_op} is not a local call region")

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node_name, "node_op": self.node_op, "target": self.target_name,
                "args": _json_arg(self.args, self.node_to_value),
                "kwargs": _json_arg(self.kwargs, self.node_to_value)}


def target_name(target: Any) -> str:
    return _target_name(target)


def evaluate_program(indexed_program: Any, inputs: Mapping[str, Any]) -> Any:
    """Evaluate each operation's local FX reference in graph order."""
    program = indexed_program.source_program
    graph_module = program.graph_module
    values = indexed_program.values
    value_by_node = {value.fx_node: value.value_id for value in values}
    environment: dict[str, Any] = {}
    operations = {operation.local_reference.node_name: operation for operation in indexed_program.operations}
    for node in graph_module.graph.nodes:
        value_id = value_by_node[node.name]
        if node.op == "placeholder":
            if node.name in inputs:
                environment[value_id] = inputs[node.name]
            elif value_id in inputs:
                environment[value_id] = inputs[value_id]
            else:
                raise KeyError(f"missing indexed input {node.name!r} ({value_id})")
        elif node.op == "get_attr":
            binding = program.lifted_bindings.get(node.name)
            if binding is not None:
                environment[value_id] = program.binding_values[binding.target]
            else:
                attribute = program.source_graph_module or graph_module
                for component in str(node.target).split("."):
                    attribute = getattr(attribute, component)
                environment[value_id] = attribute
        elif node.op == "output":
            result = resolve_fx(node.args[0], environment, value_by_node)
            if program.source_kind == "exported_program" and program.output_tree_spec is not None:
                try:
                    import torch.utils._pytree as pytree
                    leaves, _ = pytree.tree_flatten(result)
                    return pytree.tree_unflatten(leaves, program.output_tree_spec)
                except (ImportError, TypeError, ValueError):
                    pass
            return result
        else:
            operation = operations.get(node.name)
            if operation is None:
                raise RuntimeError(f"{node.name}: indexed program has no local reference region")
            environment[value_id] = operation.local_reference.evaluate(environment)
    raise RuntimeError("indexed FX program has no output node")
