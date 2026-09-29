"""Origin-complete, backend-neutral indexed FX semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..frontend.capture import NormalizedProgram, graph_hash
from ..frontend.facts import FactTable, TensorFacts, collect_facts
from .reference import LocalFXReference, target_name


@dataclass(frozen=True)
class IterationAxis:
    name: str
    extent: Any

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "extent": _stable(self.extent)}


@dataclass(frozen=True)
class InputIndexMap:
    value_id: str
    expressions: tuple[str, ...]
    mode: str = "affine"

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "expressions": list(self.expressions), "mode": self.mode}


@dataclass(frozen=True)
class AliasEdge:
    value_id: str
    alias_set: str | None
    relation: str
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "alias_set": self.alias_set,
                "relation": self.relation, "sources": list(self.sources)}


@dataclass(frozen=True)
class EffectEdge:
    origin_id: str
    kind: str
    target: str | None
    required: bool
    effect_id: str | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    alias_rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"origin_id": self.origin_id, "kind": self.kind,
                "target": self.target, "required": self.required,
                "effect_id": self.effect_id, "reads": list(self.reads),
                "writes": list(self.writes), "depends_on": list(self.depends_on),
                "alias_rule": self.alias_rule}


@dataclass(frozen=True)
class StateTransition:
    """Functional, bounded append from one fixed-capacity state value to another."""

    effect_id: str
    state_id: str
    old_value: str
    new_value: str
    index_value: str
    valid_length_before: str
    valid_length_after: str
    capacity: int
    axis: int
    order: int
    write_footprint: Mapping[str, Any]
    alias_rule: str = "functional_new_value"

    def validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.capacity:
            raise ValueError(f"append index {index!r} is outside [0, {self.capacity})")

    def to_dict(self) -> dict[str, Any]:
        return {"effect_id": self.effect_id, "state_id": self.state_id,
                "old_value": self.old_value, "new_value": self.new_value,
                "index_value": self.index_value,
                "valid_length_before": self.valid_length_before,
                "valid_length_after": self.valid_length_after,
                "capacity": self.capacity, "axis": self.axis, "order": self.order,
                "write_footprint": dict(self.write_footprint), "alias_rule": self.alias_rule}


@dataclass(frozen=True)
class IndexedValue:
    value_id: str
    fx_node: str
    origin_ids: tuple[str, ...]
    value_kind: str
    shape: tuple[Any, ...] = ()
    dtype: str | None = None
    strides: tuple[int, ...] = ()
    storage_offset: int | None = None
    alignment_bytes: int | None = None
    alias_set: str | None = None
    alias_kind: str = "unknown"
    alias_sources: tuple[str, ...] = ()
    non_overlapping: bool | None = None
    role: str = "intermediate"
    producer: str | None = None
    consumers: tuple[str, ...] = ()
    cast_origin: str | None = None
    effects: tuple[str, ...] = ()
    layout: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "fx_node": self.fx_node, "origin_ids": list(self.origin_ids),
                "value_kind": self.value_kind, "shape": [_stable(x) for x in self.shape],
                "dtype": self.dtype, "strides": list(self.strides), "storage_offset": self.storage_offset,
                "alignment_bytes": self.alignment_bytes, "alias_set": self.alias_set,
                "alias_kind": self.alias_kind, "alias_sources": list(self.alias_sources),
                "non_overlapping": self.non_overlapping,
                "role": self.role, "producer": self.producer, "consumers": list(self.consumers),
                "cast_origin": self.cast_origin, "effects": list(self.effects), "layout": self.layout}


@dataclass(frozen=True)
class IndexedOp:
    op_id: str
    kind: str
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    iteration_domain: tuple[IterationAxis, ...]
    reduction_domain: tuple[IterationAxis, ...]
    input_index_maps: tuple[InputIndexMap, ...]
    output_index_map: tuple[str, ...]
    predicate_bounds: tuple[str, ...]
    dtype_expression: Mapping[str, Any]
    attributes: Mapping[str, Any]
    alias_edges: tuple[AliasEdge, ...]
    effect_edges: tuple[EffectEdge, ...]
    origin_ids: tuple[str, ...]
    local_reference: LocalFXReference = field(compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "kind": self.kind, "target": self.target,
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "iteration_domain": [axis.to_dict() for axis in self.iteration_domain],
                "reduction_domain": [axis.to_dict() for axis in self.reduction_domain],
                "input_index_maps": [item.to_dict() for item in self.input_index_maps],
                "output_index_map": list(self.output_index_map),
                "predicate_bounds": list(self.predicate_bounds),
                "dtype_expression": dict(self.dtype_expression), "attributes": dict(self.attributes),
                "alias_edges": [edge.to_dict() for edge in self.alias_edges],
                "effect_edges": [edge.to_dict() for edge in self.effect_edges],
                "origin_ids": list(self.origin_ids), "local_reference": self.local_reference.to_dict()}


@dataclass(frozen=True)
class OutputLeaf:
    path: tuple[Any, ...]
    value_id: str | None
    origin_id: str | None
    literal: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": [_stable(item) for item in self.path], "value_id": self.value_id,
                "origin_id": self.origin_id, "literal": _stable(self.literal)}


@dataclass
class IndexedTensorProgram:
    source_program: NormalizedProgram = field(compare=False, repr=False)
    values: tuple[IndexedValue, ...]
    operations: tuple[IndexedOp, ...]
    outputs: tuple[OutputLeaf, ...]
    coverage: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[DiagnosticRecord, ...]
    state_transitions: tuple[StateTransition, ...] = ()

    @property
    def strict_supported(self) -> bool:
        return not self.diagnostics

    @property
    def structural_hash(self) -> str:
        encoded = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def evaluate_local(self, op_id: str, boundary_values: Mapping[str, Any]) -> Any:
        operation = next(item for item in self.operations if item.op_id == op_id)
        return operation.local_reference.evaluate(boundary_values)

    def evaluate(self, inputs: Mapping[str, Any]) -> Any:
        """Interpret the indexed node sequence through its one-node FX references."""
        from .reference import evaluate_program
        return evaluate_program(self, inputs)

    def to_dict(self) -> dict[str, Any]:
        result = self._payload()
        result["indexed_program_hash"] = self.structural_hash
        return result

    def _payload(self) -> dict[str, Any]:
        return {"schema_version": 1,
                "source_graph_hash": graph_hash(self.source_program),
                "numerical_policy_hash": getattr(self.source_program.policy, "contract_hash", None),
                "strict_supported": self.strict_supported,
                "values": [value.to_dict() for value in self.values],
                "operations": [operation.to_dict() for operation in self.operations],
                "outputs": [output.to_dict() for output in self.outputs],
                "state_transitions": [item.to_dict() for item in self.state_transitions],
                "coverage": [dict(item) for item in self.coverage],
                "diagnostics": [item.to_dict() for item in self.diagnostics]}


_KIND_BY_OPERATOR = {
    **{name: "Map" for name in (
        "add", "sub", "mul", "div", "neg", "exp", "rsqrt", "sqrt", "square", "sigmoid",
        "tanh", "relu", "silu", "gelu", "where", "eq", "ne", "gt", "ge", "lt", "le", "to",
        "_to_copy", "convert_element_type", "maximum", "minimum", "pow",
    )},
    **{name: "Broadcast/View" for name in (
        "view", "reshape", "transpose", "permute", "t", "slice", "select", "squeeze", "unsqueeze",
        "expand", "detach", "alias", "flatten", "as_strided",
    )},
    "contiguous": "Map",
    **{name: "Reduce" for name in ("sum", "mean", "amax", "max")},
    **{name: "Contraction" for name in ("mm", "bmm", "addmm", "matmul", "linear")},
    **{name: "Gather" for name in ("index_select", "gather", "take")},
    **{name: "Scatter/StateWrite" for name in ("index_copy", "index_put", "slice_scatter", "scatter", "copy")},
    "scaled_dot_product_attention": "Attention",
    "_assert_tensor_metadata": "Guard",
}


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _operator(node: Any) -> tuple[str, str]:
    target = node.target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return schema.name.split("::")[-1], target_name(target)
    if node.op == "call_method":
        return str(target), f"method:{target}"
    rendered = target_name(target)
    return rendered.rsplit(".", 1)[-1], rendered


def _shape(fact: TensorFacts | None, node: Any) -> tuple[Any, ...]:
    if fact is not None:
        return fact.shape
    metadata = node.meta.get("val", node.meta.get("tensor_meta"))
    return tuple(getattr(metadata, "shape", ()))


def _kind(node: Any, operator: str) -> str:
    schema = getattr(node.target, "_schema", None)
    if operator == "max" and schema is not None and schema.overload_name not in {"", "default"}:
        return "Unsupported"
    if operator in _KIND_BY_OPERATOR:
        return _KIND_BY_OPERATOR[operator]
    name = str(node.target).lower()
    if any(part in name for part in ("cond", "while_loop", "scan", "higher_order")):
        return "Scan/Branch"
    return "Unsupported"


def _ranked_axes(shape: tuple[Any, ...], prefix: str = "i") -> tuple[IterationAxis, ...]:
    return tuple(IterationAxis(f"{prefix}{axis}", extent) for axis, extent in enumerate(shape))


def _literal_ints(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return None


def _argument(node: Any, position: int, keyword: str, default: Any = None) -> Any:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(keyword, default)


def _reduction_axes(node: Any, rank: int) -> tuple[int, ...] | None:
    dimensions = _literal_ints(_argument(node, 1, "dim"))
    if dimensions is None:
        dimensions = tuple(range(rank))
    if any(axis < -rank or axis >= rank for axis in dimensions):
        return None
    normalized = tuple(axis + rank if axis < 0 else axis for axis in dimensions)
    return normalized if len(set(normalized)) == len(normalized) else None


def _broadcast_map(input_shape: tuple[Any, ...], output_shape: tuple[Any, ...]) -> tuple[str, ...]:
    if len(input_shape) > len(output_shape):
        return ("UNKNOWN",)
    pad = len(output_shape) - len(input_shape)
    expressions = ["0"] * pad
    for axis, dim in enumerate(input_shape):
        output_axis = pad + axis
        if dim == 1:
            expressions.append("0")
        elif dim == output_shape[output_axis]:
            expressions.append(f"i{output_axis}")
        else:
            expressions.append("UNKNOWN")
    return tuple(expressions)


def _input_maps(node: Any, operator: str, kind: str, inputs: tuple[str, ...],
                node_to_value: Mapping[str, str], facts: FactTable,
                input_nodes: tuple[Any, ...], output_shape: tuple[Any, ...]) -> tuple[InputIndexMap, ...]:
    maps = []
    rank = len(output_shape)
    input_shapes = [_shape(facts.for_node(item), item) for item in input_nodes]
    if kind == "Attention" and len(inputs) == 4:
        heads_per_kv = (input_shapes[0][1] // input_shapes[1][1]
                        if len(input_shapes[0]) > 1 and len(input_shapes[1]) > 1 and
                        isinstance(input_shapes[0][1], int) and
                        isinstance(input_shapes[1][1], int) and input_shapes[1][1] else "UNKNOWN")
        return (
            InputIndexMap(inputs[0], ("i0", "i1", "0", "0:D"), "attention_row"),
            InputIndexMap(inputs[1], ("i0", f"i1//{heads_per_kv}", "0:C", "0:D"), "attention_row"),
            InputIndexMap(inputs[2], ("i0", f"i1//{heads_per_kv}", "0:C", "0:D"), "attention_row"),
            InputIndexMap(inputs[3], ("i0", "0" if len(input_shapes[3]) > 1 and
                                      input_shapes[3][1] == 1 else "i1",
                                      "0", "0:C"), "attention_row"),
        )
    if kind == "Map":
        return tuple(InputIndexMap(value, _broadcast_map(shape, output_shape), "broadcast")
                     for value, shape in zip(inputs, input_shapes))
    if kind == "Reduce" and input_nodes:
        input_shape = input_shapes[0]
        dims = _reduction_axes(node, len(input_shape))
        if dims is None:
            return (InputIndexMap(inputs[0], ("UNKNOWN",), "reduction"),)
        keepdim = _argument(node, 2, "keepdim", False)
        if not isinstance(keepdim, bool):
            return (InputIndexMap(inputs[0], ("UNKNOWN",), "reduction"),)
        reduction_axes = {axis: reduction for reduction, axis in enumerate(dims)}
        expressions = []
        output_axis = 0
        for axis in range(len(input_shape)):
            if axis in reduction_axes:
                expressions.append(f"r{reduction_axes[axis]}")
            elif keepdim:
                expressions.append(f"i{axis}")
            else:
                expressions.append(f"i{output_axis}")
                output_axis += 1
        maps.append(InputIndexMap(inputs[0], tuple(expressions), "reduction"))
        return tuple(maps)
    if kind == "Contraction" and len(inputs) >= 2:
        if operator == "addmm" and len(inputs) >= 3:
            if len(input_shapes[1]) != 2 or len(input_shapes[2]) != 2 or rank != 2:
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "broadcast"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "contraction"),
                        InputIndexMap(inputs[2], ("UNKNOWN",), "contraction"))
            maps.append(InputIndexMap(inputs[0], _broadcast_map(input_shapes[0], output_shape), "broadcast"))
            maps.append(InputIndexMap(inputs[1], ("i0", "k"), "contraction"))
            maps.append(InputIndexMap(inputs[2], ("k", "i1"), "contraction"))
        elif operator == "linear":
            if len(input_shapes[1]) != 2 or not input_shapes[0] or rank != len(input_shapes[0]):
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "contraction"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "contraction"))
            maps.append(InputIndexMap(inputs[0], tuple(f"i{axis}" for axis in range(max(rank - 1, 0))) + ("k",), "contraction"))
            maps.append(InputIndexMap(inputs[1], (f"i{max(rank - 1, 0)}", "k"), "contraction"))
            if len(inputs) > 2:
                maps.append(InputIndexMap(inputs[2], _broadcast_map(input_shapes[2], output_shape), "broadcast"))
        elif operator == "bmm":
            if len(input_shapes[0]) != 3 or len(input_shapes[1]) != 3 or rank != 3:
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "contraction"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "contraction"))
            maps.append(InputIndexMap(inputs[0], ("i0", "i1", "k"), "contraction"))
            maps.append(InputIndexMap(inputs[1], ("i0", "k", "i2"), "contraction"))
        else:
            if len(input_shapes[0]) != 2 or len(input_shapes[1]) != 2 or rank != 2:
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "contraction"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "contraction"))
            maps.append(InputIndexMap(inputs[0], ("i0", "k"), "contraction"))
            maps.append(InputIndexMap(inputs[1], ("k", "i1"), "contraction"))
        return tuple(maps)
    if kind == "Broadcast/View" and inputs:
        shape = input_shapes[0]
        if operator in {"transpose", "permute", "t"}:
            permutation = list(range(len(shape)))
            if operator == "t" and len(shape) > 2:
                maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "permutation"))
                return tuple(maps)
            elif operator == "t" and len(shape) == 2:
                permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
            elif operator == "transpose":
                dims = node.args[1:3]
                if len(dims) != 2:
                    return (InputIndexMap(inputs[0], ("UNKNOWN",), "permutation"),)
                first, second = (dim + len(shape) if dim < 0 else dim for dim in dims)
                if first == second or not (0 <= first < len(shape) and 0 <= second < len(shape)):
                    return (InputIndexMap(inputs[0], ("UNKNOWN",), "permutation"),)
                permutation[first], permutation[second] = permutation[second], permutation[first]
            elif operator == "permute":
                dims = node.args[1] if len(node.args) > 1 else None
                if isinstance(dims, (tuple, list)) and len(dims) == len(shape):
                    permutation = [dim + len(shape) if dim < 0 else dim for dim in dims]
                    if sorted(permutation) != list(range(len(shape))):
                        return (InputIndexMap(inputs[0], ("UNKNOWN",), "permutation"),)
                else:
                    return (InputIndexMap(inputs[0], ("UNKNOWN",), "permutation"),)
            inverse = {source_axis: out_axis for out_axis, source_axis in enumerate(permutation)}
            maps.append(InputIndexMap(inputs[0], tuple(f"i{inverse[axis]}" for axis in range(len(shape))), "permutation"))
        elif operator == "expand":
            maps.append(InputIndexMap(inputs[0], _broadcast_map(shape, output_shape), "zero_stride_broadcast"))
        elif operator == "as_strided":
            # Storage bounds are not represented in TensorFacts yet.  Even a
            # syntactically affine map can escape the input's allocation.
            maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "strided_view"))
        elif operator in {"reshape", "view", "flatten"}:
            maps.append(InputIndexMap(inputs[0], (f"unravel(ravel(i0..i{max(rank - 1, 0)}),{_stable(shape)})",), "reshape"))
        elif operator == "select":
            dim = node.args[1] if len(node.args) > 1 else 0
            selected = node.args[2] if len(node.args) > 2 else "index"
            axes = [f"i{i}" for i in range(rank)]
            dim = dim + len(shape) if dim < 0 else dim
            if (not isinstance(dim, int) or not 0 <= dim < len(shape) or
                    not isinstance(selected, int) or not isinstance(shape[dim], int)):
                maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "select"))
                return tuple(maps)
            selected = selected + shape[dim] if selected < 0 else selected
            if not 0 <= selected < shape[dim]:
                maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "select"))
                return tuple(maps)
            axes.insert(dim, str(selected))
            maps.append(InputIndexMap(inputs[0], tuple(axes), "select"))
        elif operator == "squeeze":
            requested = _literal_ints(node.args[1]) if len(node.args) > 1 else None
            squeezed = ({dim + len(shape) if dim < 0 else dim for dim in requested}
                        if requested is not None else {axis for axis, extent in enumerate(shape) if extent == 1})
            next_axis = 0
            expressions = []
            for axis in range(len(shape)):
                if axis in squeezed and shape[axis] == 1:
                    expressions.append("0")
                else:
                    expressions.append(f"i{next_axis}")
                    next_axis += 1
            maps.append(InputIndexMap(inputs[0], tuple(expressions), "squeeze"))
        elif operator == "unsqueeze":
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + rank if dim < 0 else dim
            maps.append(InputIndexMap(inputs[0], tuple(f"i{axis if axis < dim else axis + 1}"
                                                        for axis in range(len(shape))), "unsqueeze"))
        elif operator == "slice":
            dim = node.args[1] if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else 0
            step = node.args[4] if len(node.args) > 4 else 1
            dim = dim + len(shape) if dim < 0 else dim
            start = 0 if start is None else start
            step = 1 if step is None else step
            if (not isinstance(dim, int) or not 0 <= dim < len(shape) or
                    not isinstance(start, int) or not isinstance(step, int) or step <= 0 or
                    not isinstance(shape[dim], int)):
                maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "slice"))
                return tuple(maps)
            start = max(start + shape[dim], 0) if start < 0 else min(start, shape[dim])
            axes = [f"i{i}" for i in range(len(shape))]
            axes[dim] = f"{start}+i{dim}*{step}"
            maps.append(InputIndexMap(inputs[0], tuple(axes), "slice"))
        else:
            maps.append(InputIndexMap(inputs[0], tuple(f"i{i}" for i in range(len(shape))), "view"))
        return tuple(maps)
    if kind == "Gather" and inputs:
        if operator == "index_select" and len(inputs) == 2:
            shape = input_shapes[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + len(shape) if isinstance(dim, int) and dim < 0 else dim
            if not isinstance(dim, int) or not 0 <= dim < len(shape) or len(input_shapes[1]) != 1:
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "indirect_select"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "index"))
            source_map = [f"i{i}" for i in range(len(shape))]
            source_map[dim] = f"index[i{dim}]"
            return (InputIndexMap(inputs[0], tuple(source_map), "indirect_select"),
                    InputIndexMap(inputs[1], (f"i{dim}",), "index"))
        maps.append(InputIndexMap(inputs[0], ("input[index_map]",), "indirect"))
        for value in inputs[1:]:
            maps.append(InputIndexMap(value, ("index_map",), "index"))
        return tuple(maps)
    if kind == "Scatter/StateWrite" and inputs:
        shape = input_shapes[0]
        maps.append(InputIndexMap(inputs[0], tuple(f"i{i}" for i in range(len(shape))), "read_modify_write"))
        if operator == "index_copy" and len(inputs) == 3:
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + len(shape) if isinstance(dim, int) and dim < 0 else dim
            source_shape = input_shapes[2]
            if not isinstance(dim, int) or not 0 <= dim < len(shape) or len(source_shape) != len(shape):
                return (InputIndexMap(inputs[0], ("UNKNOWN",), "read_modify_write"),
                        InputIndexMap(inputs[1], ("UNKNOWN",), "index"),
                        InputIndexMap(inputs[2], ("UNKNOWN",), "write"))
            source_map = [f"i{i}" for i in range(len(source_shape))]
            source_map[dim] = "0"
            maps.append(InputIndexMap(inputs[1], ("0",), "index"))
            maps.append(InputIndexMap(inputs[2], tuple(source_map), "write"))
        else:
            for value in inputs[1:]:
                maps.append(InputIndexMap(value, ("bounded_write_region",), "write"))
    return tuple(maps)


def _reduction_domain(node: Any, operator: str, kind: str,
                      input_nodes: tuple[Any, ...], facts: FactTable) -> tuple[IterationAxis, ...]:
    if kind == "Contraction" and input_nodes:
        matrix_index = 1 if operator == "addmm" and len(input_nodes) >= 3 else 0
        shape = _shape(facts.for_node(input_nodes[matrix_index]), input_nodes[matrix_index])
        return (IterationAxis("k", shape[-1] if shape else "UNKNOWN"),)
    if kind != "Reduce" or not input_nodes:
        return ()
    shape = _shape(facts.for_node(input_nodes[0]), input_nodes[0])
    dims = _reduction_axes(node, len(shape))
    if dims is None:
        return (IterationAxis("r0", "UNKNOWN"),)
    axes = []
    for index, dim in enumerate(dims):
        axes.append(IterationAxis(f"r{index}", shape[dim] if dim < len(shape) else "UNKNOWN"))
    return tuple(axes)


def _cache_axis(layout: str, rank: int) -> int | None:
    axes = tuple(part.strip() for part in layout.split(","))
    if len(axes) != rank or "capacity" not in axes:
        return None
    return axes.index("capacity")


def _attention_contract(input_facts: list[TensorFacts | None], output_fact: TensorFacts | None,
                        node: Any, inputs: tuple[str, ...],
                        transitions: Mapping[str, StateTransition]) -> Mapping[str, Any] | None:
    if len(inputs) != 4 or any(fact is None for fact in input_facts) or output_fact is None:
        return None
    q, k, v, mask = input_facts
    assert q is not None and k is not None and v is not None and mask is not None
    if any(len(fact.shape) != 4 for fact in (q, k, v, mask, output_fact)):
        return None
    batch, heads_q, query_length, depth = q.shape
    _, heads_kv, capacity, _ = k.shape
    if (not all(isinstance(dim, int) and dim > 0 for dim in (batch, heads_q, heads_kv, capacity, depth))
            or query_length != 1 or k.shape != v.shape or output_fact.shape != q.shape
            or k.shape[0] != batch or k.shape[-1] != depth or heads_q % heads_kv
            or mask.shape != (batch, mask.shape[1], 1, capacity)
            or mask.shape[1] not in (1, heads_q) or mask.dtype != "bool"
            or q.dtype not in {"float16", "bfloat16", "float32"}
            or any(fact.dtype != q.dtype for fact in (k, v, output_fact))
            or any(tuple(fact.strides) != tuple(math.prod(fact.shape[i + 1:])
                                                     for i in range(4)) for fact in (k, v))):
        return None
    k_state, v_state = transitions.get(inputs[1]), transitions.get(inputs[2])
    if (k_state is None or v_state is None or k_state.state_id == v_state.state_id
            or k_state.index_value != v_state.index_value
            or any(item.capacity != capacity or item.axis != 2 for item in (k_state, v_state))):
        return None
    dropout = _argument(node, 4, "dropout_p", 0.0)
    causal = _argument(node, 5, "is_causal", False)
    scale = _argument(node, 6, "scale")
    gqa = _argument(node, 7, "enable_gqa", False)
    if (dropout not in (0, 0.0) or causal is not False or
            gqa is not (heads_q != heads_kv) or
            scale is not None and (not isinstance(scale, (int, float)) or
                                   not math.isfinite(scale))):
        return None
    return {"batch": batch, "heads_q": heads_q, "heads_kv": heads_kv,
            "capacity": capacity, "head_dim": depth, "mask_heads": mask.shape[1],
            "scale": float(scale) if scale is not None else depth ** -0.5,
            "state_effects": (k_state.effect_id, v_state.effect_id),
            "mask_rule": "boolean_true_attends; caller supplies causal/window alignment; runtime rejects true beyond L"}


def _cache_transition(program: NormalizedProgram, node: Any, facts: FactTable,
                      inputs: tuple[str, ...], output_id: str) -> tuple[StateTransition | None, DiagnosticRecord | None]:
    effects = tuple(effect for effect in program.effects if effect.node_id == node.name and effect.required)
    if not effects:
        return None, None
    if len(effects) != 1:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} has duplicated effect ownership",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"effect_count": len(effects)},
        )
    effect = effects[0]
    abi = program.step_abi
    if abi is None:
        return None, DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            f"state write {node.name} has no StepABI bounds and alias contract",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"negative_case": "dynamic index has no proved bounds"},
        )
    if effect.kind != "functional_state_update" or not effect.target:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} is not a functional update",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"effect_kind": effect.kind},
        )
    if len(inputs) != 3 or len(node.args) < 4:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} is not a four-argument index_copy",
            DiagnosticSeverity.ERROR, node_id=node.name,
        )

    state_specs = [item for item in abi.old_state_inputs if item["state_id"] == effect.target]
    state_effects = [item for item in abi.state_effects if item["state_id"] == effect.target]
    output_specs = [item for item in abi.new_state_outputs if item["state_id"] == effect.target]
    if len(state_specs) != 1 or len(state_effects) != 1 or len(output_specs) != 1:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} does not have one matching old/effect/new ABI edge",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"old_states": len(state_specs), "state_effects": len(state_effects),
                     "new_outputs": len(output_specs)},
        )

    state_spec, effect_spec, output_spec = state_specs[0], state_effects[0], output_specs[0]
    old_node, index_node = node.args[0], node.args[2]
    old_fact, index_fact = facts.for_node(old_node), facts.for_node(index_node)
    update_fact, output_fact = facts.facts.get(inputs[2]), facts.facts.get(output_id)
    if getattr(old_node, "name", None) != state_spec["placeholder"]:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} does not consume its declared old state",
            DiagnosticSeverity.ERROR, node_id=node.name,
        )
    if output_spec.get("source_id") not in {node.name, output_id}:
        return None, DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            f"state write {node.name} differs from the StepABI new-state source",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"declared_source": output_spec.get("source_id")},
        )
    if old_fact is None or index_fact is None or update_fact is None or output_fact is None:
        return None, DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            f"state write {node.name} lacks exact state/index/update facts",
            DiagnosticSeverity.ERROR, node_id=node.name,
        )
    dim = node.args[1]
    dim = dim + len(old_fact.shape) if isinstance(dim, int) and dim < 0 else dim
    axis = _cache_axis(state_spec["layout"], len(old_fact.shape))
    capacity = state_spec["capacity"]
    position = abi.position_and_valid_length
    write_ranges = {item.get("range") for item in effect_spec.get("writes", ()) if isinstance(item, Mapping)}
    if (not isinstance(dim, int) or dim != axis or axis is None or
            old_fact.shape != output_fact.shape or old_fact.shape[axis] != capacity or
            output_fact.alias_kind != "fresh" or
            len(index_fact.shape) != 1 or index_fact.shape != (1,) or
            index_fact.dtype not in {"int32", "int64"} or
            update_fact.shape != old_fact.shape[:axis] + (1,) + old_fact.shape[axis + 1:] or
            getattr(index_node, "name", None) != position.get("position_source") or
            position.get("old_valid_length_source") != position.get("position_source") or
            position.get("append_position_expression") != "L" or
            position.get("new_valid_length_expression") != "L+1" or
            "[L,L+1)" not in write_ranges):
        return None, DiagnosticRecord(
            DiagnosticCode.MISSING_FACTS,
            f"state write {node.name} is outside the proved fixed-capacity append contract",
            DiagnosticSeverity.ERROR, node_id=node.name,
            details={"capacity": capacity, "axis": dim,
                     "negative_case": "index, alias, footprint, or valid-length guard mismatch"},
        )
    guard_capacity = abi.guard_set.get("capacity", {})
    if isinstance(guard_capacity, Mapping) and state_spec["placeholder"] in guard_capacity:
        if guard_capacity[state_spec["placeholder"]] != capacity:
            return None, DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"state write {node.name} disagrees with the StepABI capacity guard",
                DiagnosticSeverity.ERROR, node_id=node.name,
            )
    effect_id = str(effect_spec["effect_id"])
    index_id = facts.node_to_value.get(index_node.name, inputs[1])
    return StateTransition(
        effect_id=effect_id, state_id=effect.target,
        old_value=inputs[0], new_value=output_id, index_value=index_id,
        valid_length_before=index_id, valid_length_after="L+1",
        capacity=capacity, axis=axis, order=effect_spec["order"],
        write_footprint={"axis": axis, "range": "[L,L+1)", "other_axes": "full"},
    ), None


def _bounded_state_index(program: NormalizedProgram, index_node: Any, state_id: str,
                         source_shape: tuple[Any, ...], axis: int,
                         index_fact: TensorFacts | None) -> Mapping[str, Any] | None:
    abi = program.step_abi
    if abi is None or index_fact is None or index_fact.shape != (1,) or index_fact.dtype not in {"int32", "int64"}:
        return None
    state_specs = [item for item in abi.old_state_inputs if item["state_id"] == state_id]
    if len(state_specs) != 1:
        return None
    state = state_specs[0]
    expected_axis = _cache_axis(state["layout"], len(source_shape))
    position = abi.position_and_valid_length
    if (getattr(index_node, "name", None) != position.get("position_source") or
            position.get("old_valid_length_source") != position.get("position_source") or
            position.get("append_position_expression") != "L" or
            expected_axis != axis or source_shape[axis] != state["capacity"]):
        return None
    return {"value_id": index_fact.value_id, "lower": 0,
            "upper_exclusive": state["capacity"], "guard": "StepABI/v1:L<capacity"}


def _bounded_declared_index(program: NormalizedProgram, index_node: Any,
                            source_shape: tuple[Any, ...], axis: int,
                            index_fact: TensorFacts | None) -> Mapping[str, Any] | None:
    abi = program.step_abi
    if abi is None or index_fact is None or index_fact.dtype not in {"int32", "int64"}:
        return None
    declared = abi.guard_set.get("index_bounds")
    if not isinstance(declared, Mapping):
        return None
    bounds = declared.get(getattr(index_node, "name", None))
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or
            tuple(bounds) != (0, source_shape[axis])):
        return None
    return {"value_id": index_fact.value_id, "lower": 0,
            "upper_exclusive": source_shape[axis], "guard": "StepABI/v1:index_bounds"}


def _output_leaves(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    if hasattr(value, "op") and hasattr(value, "name"):
        return [(path, value)]
    if isinstance(value, Mapping):
        return [leaf for key, item in value.items() for leaf in _output_leaves(item, path + (str(key),))]
    if isinstance(value, (tuple, list)):
        return [leaf for index, item in enumerate(value) for leaf in _output_leaves(item, path + (index,))]
    return [(path, value)]


def lower_indexed_program(program: NormalizedProgram, *, facts: FactTable | None = None) -> IndexedTensorProgram:
    """Lower one FX node per local indexed operation; unknown nodes stay diagnostic."""
    facts = facts or collect_facts(program)
    graph_nodes = list(program.graph_module.graph.nodes)
    node_to_value = dict(facts.node_to_value)
    values = []
    operations = []
    state_transitions: list[StateTransition] = []
    transitions_by_output: dict[str, StateTransition] = {}
    diagnostics: list[DiagnosticRecord] = []
    module_lookup = getattr(program.graph_module, "get_submodule", lambda _name: None)
    for node in graph_nodes:
        value_id = node_to_value[node.name]
        fact = facts.facts.get(value_id)
        metadata = node.meta.get("val", node.meta.get("tensor_meta"))
        origins = tuple(program.origin_map.get(node.name, ()))
        value_kind = "tensor" if fact is not None else type(metadata).__name__ if metadata is not None else "unknown"
        values.append(IndexedValue(
            value_id, node.name, origins, value_kind,
            fact.shape if fact else tuple(getattr(metadata, "shape", ())),
            fact.dtype if fact else str(getattr(metadata, "dtype", "" )).removeprefix("torch.") or None,
            fact.strides if fact else (), fact.storage_offset if fact else None,
            fact.alignment_bytes if fact else None, fact.alias_set if fact else None,
            fact.alias_kind if fact else "unknown", fact.alias_sources if fact else (),
            fact.non_overlapping if fact else None,
            fact.role if fact else ("input" if node.op == "placeholder" else "intermediate"),
            fact.producer if fact else None, fact.consumers if fact else tuple(user.name for user in node.users),
            fact.cast_origin if fact else None, fact.write_effects if fact else (),
            fact.layout if fact else None,
        ))
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        operator, rendered_target = _operator(node)
        kind = _kind(node, operator)
        input_nodes = tuple(node.all_input_nodes)
        inputs = tuple(node_to_value[item.name] for item in input_nodes)
        output = (value_id,)
        refs = LocalFXReference(
            node.name, node.op, rendered_target, node.target, node.args, node.kwargs,
            node_to_value, module_lookup(str(node.target)) if node.op == "call_module" else None,
        )
        if kind in {"Unsupported", "Scan/Branch"}:
            message = (f"no indexed primitive for FX target {rendered_target}" if kind == "Unsupported"
                       else f"guarded {kind} extension is not implemented for {rendered_target}")
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                message,
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"target": rendered_target, "negative_case": "unsupported live FX operation"},
            ))
        if node.op != "call_function" and kind != "Unsupported":
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.UNSUPPORTED_SEMANTICS,
                f"indexed primitive requires a functional call node; got {node.op}",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"target": rendered_target, "negative_case": "non-functional FX call"},
            ))
            kind = "Unsupported"
        output_fact = facts.facts.get(value_id)
        output_shape = _shape(output_fact, node)
        input_facts = [facts.facts.get(item) for item in inputs]
        if any(item is None for item in input_facts) and kind != "Unsupported":
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"indexed operation {node.name} lacks a tensor input fact",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"inputs_without_facts": [value for value, item in zip(inputs, input_facts) if item is None]},
            ))
        domain = _ranked_axes(output_shape)
        reductions = _reduction_domain(node, operator, kind, input_nodes, facts)
        maps = _input_maps(node, operator, kind, inputs, node_to_value, facts, input_nodes, output_shape)
        transition = None
        index_bounds = None
        transition_diagnostic = None
        if kind == "Scatter/StateWrite" and operator == "index_copy":
            transition, transition_diagnostic = _cache_transition(program, node, facts, inputs, value_id)
            if transition is not None:
                index_bounds = {"value_id": transition.index_value, "lower": 0,
                                "upper_exclusive": transition.capacity, "guard": "StepABI/v1:L<capacity"}
        elif kind == "Gather" and operator == "index_select" and input_nodes:
            source_transition = transitions_by_output.get(inputs[0])
            source_state = program.state_bindings.get(input_nodes[0].name)
            if source_transition is not None:
                source_state = source_transition.state_id
            source_fact = facts.for_node(input_nodes[0])
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + len(source_fact.shape) if source_fact and isinstance(dim, int) and dim < 0 else dim
            if source_fact is not None and isinstance(dim, int) and 0 <= dim < len(source_fact.shape):
                index_bounds = _bounded_state_index(
                    program, input_nodes[1], source_state or "", source_fact.shape, dim,
                    facts.for_node(input_nodes[1]),
                )
                if index_bounds is None:
                    index_bounds = _bounded_declared_index(
                        program, input_nodes[1], source_fact.shape, dim,
                        facts.for_node(input_nodes[1]),
                    )
        attention = None
        if kind == "Attention":
            attention = _attention_contract(input_facts, output_fact, node, inputs, transitions_by_output)
            if attention is None:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    f"cached attention {node.name} lacks a proved one-token boolean-mask/GQA/KV contract",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"negative_case": "mask, state publication, shape, dropout or causal alignment is unsupported"},
                ))
        if transition_diagnostic is not None:
            diagnostics.append(transition_diagnostic)
        if any("UNKNOWN" in item.expressions for item in maps) and kind != "Unsupported":
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"indexed operation {node.name} has an unproved index map",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"index_maps": [item.to_dict() for item in maps]},
            ))
        if kind in {"Gather", "Scatter/StateWrite"} and operator in {
            "index_select", "gather", "take", "index_copy", "index_put", "slice_scatter", "scatter"
        } and index_bounds is None:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"indexed operation {node.name} has dynamic indices without a proved bounds guard",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"negative_case": "runtime index bounds are unknown", "operator": operator},
            ))
        aliases = tuple(
            AliasEdge(item.value_id, item.alias_set, item.alias_kind, item.alias_sources)
            for item in [facts.facts.get(value) for value in inputs + output]
            if item is not None
        )
        source_origins = set(origins)
        effect_edges = tuple(
            EffectEdge(
                origin_id, effect.kind, effect.target, effect.required,
                effect_id=transition.effect_id if transition and effect.target == transition.state_id else None,
                reads=(transition.old_value,) if transition and effect.target == transition.state_id else (),
                writes=(transition.new_value,) if transition and effect.target == transition.state_id else (),
                alias_rule=transition.alias_rule if transition and effect.target == transition.state_id else None,
            )
            for effect in program.effects
            for origin_id in program.origin_map.get(effect.node_id, ())
            if effect.node_id == node.name or origin_id in source_origins
        )
        read_transitions = tuple(transitions_by_output[value] for value in inputs if value in transitions_by_output)
        if read_transitions and origins:
            effect_edges += tuple(
                EffectEdge(origins[0], "state_read_after_publish", item.state_id, True,
                           effect_id=f"read:{node.name}:{item.effect_id}",
                           reads=(item.new_value,), depends_on=(item.effect_id,),
                           alias_rule="acquire_published_value")
                for item in read_transitions
            )
        attrs = {"arguments": refs.to_dict()["args"], "keywords": refs.to_dict()["kwargs"],
                 "operator_name": operator}
        if index_bounds is not None:
            attrs["index_bounds"] = dict(index_bounds)
        if transition is not None:
            attrs["state_transition"] = transition.to_dict()
        if attention is not None:
            attrs["attention"] = dict(attention)
        if kind == "Guard":
            attrs["guard_kind"] = "tensor_metadata"
        if kind == "Reduce" and input_nodes:
            reduction_axes = _reduction_axes(node, len(_shape(facts.for_node(input_nodes[0]), input_nodes[0])))
            operation_name = operator
            attrs["reduction"] = {
                "axes": list(reduction_axes) if reduction_axes is not None else None,
                "keepdim": _argument(node, 2, "keepdim", False),
                "initial": float("-inf") if operation_name in {"amax", "max"} else 0.0,
                "accumulation_dtype": "float32",
                "final_dtype": output_fact.dtype if output_fact else None,
            }
        if kind == "Contraction":
            attrs["contraction"] = {
                "operator": operator,
                "alpha": _argument(node, 4, "alpha", 1.0) if operator == "addmm" else 1.0,
                "beta": _argument(node, 3, "beta", 1.0) if operator == "addmm" else 0.0,
            }
        dtype_expr = {
            "operator": rendered_target,
            "input_dtypes": [fact.dtype if fact else None for fact in input_facts],
            "output_dtype": output_fact.dtype if output_fact else None,
            "cast_origin": output_fact.cast_origin if output_fact else None,
        }
        bounds = [f"0 <= i{axis} < {_stable(extent)}" for axis, extent in enumerate(output_shape)]
        if kind in {"Gather", "Scatter/StateWrite"} and input_nodes:
            source_shape = _shape(facts.for_node(input_nodes[0]), input_nodes[0])
            dimension = node.args[1] if len(node.args) > 1 and isinstance(node.args[1], int) else 0
            dimension = dimension + len(source_shape) if dimension < 0 else dimension
            if 0 <= dimension < len(source_shape):
                bounds.append(f"0 <= index < {_stable(source_shape[dimension])} on axis {dimension}")
        operation = IndexedOp(
            f"iop:{node.name}", kind, rendered_target, inputs, output, domain, reductions, maps,
            tuple(f"i{i}" for i in range(len(output_shape))),
            tuple(bounds),
            dtype_expr, attrs, aliases, effect_edges, origins, refs,
        )
        operations.append(operation)
        if transition is not None:
            state_transitions.append(transition)
            transitions_by_output[transition.new_value] = transition

    outputs: list[OutputLeaf] = []
    output_node = next((node for node in reversed(graph_nodes) if node.op == "output"), None)
    if output_node is not None:
        import json
        for path, value in _output_leaves(output_node.args[0]):
            key = json.dumps(path, separators=(",", ":"), default=str)
            output_id = program.output_origins.get(key)
            if hasattr(value, "op") and hasattr(value, "name"):
                outputs.append(OutputLeaf(path, node_to_value[value.name], output_id))
            else:
                outputs.append(OutputLeaf(path, None, output_id, value))
    from .verify import verify_indexed_program
    coverage, coverage_diagnostics = verify_indexed_program(
        program, tuple(operations), node_to_value, tuple(outputs), tuple(state_transitions)
    )
    diagnostics.extend(coverage_diagnostics)
    return IndexedTensorProgram(program, tuple(values), tuple(operations), tuple(outputs),
                                tuple(coverage), tuple(diagnostics), tuple(state_transitions))


__all__ = ["AliasEdge", "EffectEdge", "IndexedOp", "IndexedTensorProgram", "IndexedValue",
           "InputIndexMap", "IterationAxis", "OutputLeaf", "StateTransition", "lower_indexed_program"]
