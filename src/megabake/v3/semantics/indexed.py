"""Origin-complete, backend-neutral indexed FX semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
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

    def to_dict(self) -> dict[str, Any]:
        return {"origin_id": self.origin_id, "kind": self.kind,
                "target": self.target, "required": self.required}


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

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "fx_node": self.fx_node, "origin_ids": list(self.origin_ids),
                "value_kind": self.value_kind, "shape": [_stable(x) for x in self.shape],
                "dtype": self.dtype, "strides": list(self.strides), "storage_offset": self.storage_offset,
                "alignment_bytes": self.alignment_bytes, "alias_set": self.alias_set,
                "alias_kind": self.alias_kind, "alias_sources": list(self.alias_sources),
                "non_overlapping": self.non_overlapping,
                "role": self.role, "producer": self.producer, "consumers": list(self.consumers),
                "cast_origin": self.cast_origin, "effects": list(self.effects)}


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
                "strict_supported": self.strict_supported,
                "values": [value.to_dict() for value in self.values],
                "operations": [operation.to_dict() for operation in self.operations],
                "outputs": [output.to_dict() for output in self.outputs],
                "coverage": [dict(item) for item in self.coverage],
                "diagnostics": [item.to_dict() for item in self.diagnostics]}


_KIND_BY_OPERATOR = {
    **{name: "Map" for name in (
        "add", "sub", "mul", "div", "neg", "exp", "rsqrt", "sqrt", "square", "sigmoid",
        "tanh", "relu", "silu", "gelu", "where", "eq", "ne", "gt", "ge", "lt", "le", "to",
        "_to_copy", "convert_element_type", "maximum", "minimum", "pow",
    )},
    **{name: "Broadcast/View" for name in (
        "view", "reshape", "transpose", "permute", "slice", "select", "squeeze", "unsqueeze",
        "expand", "detach", "alias", "flatten", "contiguous", "as_strided",
    )},
    **{name: "Reduce" for name in ("sum", "mean", "amax", "max")},
    **{name: "Contraction" for name in ("mm", "bmm", "addmm", "matmul", "linear")},
    **{name: "Gather" for name in ("index_select", "gather", "take")},
    **{name: "Scatter/StateWrite" for name in ("index_copy", "index_put", "slice_scatter", "scatter", "copy")},
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
    if kind == "Map":
        return tuple(InputIndexMap(value, _broadcast_map(shape, output_shape), "broadcast")
                     for value, shape in zip(inputs, input_shapes))
    if kind == "Reduce" and input_nodes:
        input_shape = input_shapes[0]
        dims = _literal_ints(_argument(node, 1, "dim"))
        if dims is None:
            dims = tuple(range(len(input_shape)))
        if any(dim < -len(input_shape) or dim >= len(input_shape) for dim in dims):
            return (InputIndexMap(inputs[0], ("UNKNOWN",), "reduction"),)
        dims = tuple(dim + len(input_shape) if dim < 0 else dim for dim in dims)
        keepdim = _argument(node, 2, "keepdim", False)
        if not isinstance(keepdim, bool):
            return (InputIndexMap(inputs[0], ("UNKNOWN",), "reduction"),)
        expressions = tuple(f"r{axis}" if axis in dims else f"i{axis if keepdim else axis - sum(red < axis for red in dims)}"
                           for axis in range(len(input_shape)))
        maps.append(InputIndexMap(inputs[0], expressions, "reduction"))
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
        if operator in {"transpose", "permute"}:
            dims = (node.args[1:3] if operator == "transpose" else node.args[1])
            permutation = list(range(len(shape)))
            if operator == "transpose" and len(dims) == 2:
                first, second = (dim + len(shape) if dim < 0 else dim for dim in dims)
                permutation[first], permutation[second] = permutation[second], permutation[first]
            elif isinstance(dims, (tuple, list)) and len(dims) == len(shape):
                permutation = [dim + len(shape) if dim < 0 else dim for dim in dims]
            inverse = {source_axis: out_axis for out_axis, source_axis in enumerate(permutation)}
            maps.append(InputIndexMap(inputs[0], tuple(f"i{inverse[axis]}" for axis in range(len(shape))), "permutation"))
        elif operator == "expand":
            maps.append(InputIndexMap(inputs[0], _broadcast_map(shape, output_shape), "zero_stride_broadcast"))
        elif operator == "as_strided":
            strides = _literal_ints(node.args[2]) if len(node.args) > 2 else None
            offset = node.args[3] if len(node.args) > 3 else None
            if strides is None or len(strides) != len(output_shape):
                maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "strided_view"))
            elif offset is not None:
                # The indexed ABI uses logical tensor pointers; translating an
                # absolute storage offset therefore needs a proved base offset.
                source_fact = facts.for_node(input_nodes[0]) if input_nodes else None
                if not isinstance(offset, int) or source_fact is None or source_fact.storage_offset is None:
                    maps.append(InputIndexMap(inputs[0], ("UNKNOWN",), "strided_view"))
                else:
                    relative_offset = offset - source_fact.storage_offset
                    expression = " + ".join([str(relative_offset)] + [f"i{axis}*{stride}" for axis, stride in enumerate(strides)])
                    maps.append(InputIndexMap(inputs[0], (expression,), "strided_view"))
            else:
                expression = " + ".join(["0"] + [f"i{axis}*{stride}" for axis, stride in enumerate(strides)])
                maps.append(InputIndexMap(inputs[0], (expression,), "strided_view"))
        elif operator in {"reshape", "view", "flatten", "contiguous"}:
            maps.append(InputIndexMap(inputs[0], (f"unravel(ravel(i0..i{max(rank - 1, 0)}),{_stable(shape)})",), "reshape"))
        elif operator == "select":
            dim = node.args[1] if len(node.args) > 1 else 0
            selected = node.args[2] if len(node.args) > 2 else "index"
            axes = [f"i{i}" for i in range(rank)]
            dim = dim + len(shape) if dim < 0 else dim
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
            axes = [f"i{i}" for i in range(len(shape))]
            axes[dim] = f"{start}+i{dim}*{step}"
            maps.append(InputIndexMap(inputs[0], tuple(axes), "slice"))
        else:
            maps.append(InputIndexMap(inputs[0], tuple(f"i{i}" for i in range(len(shape))), "view"))
        return tuple(maps)
    if kind == "Gather" and inputs:
        if operator == "index_select" and len(inputs) > 1:
            shape = input_shapes[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + len(shape) if dim < 0 else dim
            base_map = [f"i{i}" for i in range(len(shape))]
            index_map = [f"i{dim}"]
            if 0 <= dim < len(base_map):
                base_map[dim] = "index[i" + str(dim) + "]"
            maps.append(InputIndexMap(inputs[0], tuple(base_map), "indirect"))
            maps.append(InputIndexMap(inputs[1], tuple(index_map), "index"))
        else:
            maps.append(InputIndexMap(inputs[0], ("input[index_map]",), "indirect"))
            for value in inputs[1:]:
                maps.append(InputIndexMap(value, ("index_map",), "index"))
        return tuple(maps)
    if kind == "Scatter/StateWrite" and inputs:
        maps.append(InputIndexMap(inputs[0], tuple(f"i{i}" for i in range(len(input_shapes[0]))), "read_modify_write"))
        if operator == "index_copy" and len(inputs) == 3:
            shape = input_shapes[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            dim = dim + len(shape) if dim < 0 else dim
            index_map = [f"i{i}" for i in range(len(shape))]
            source_shape = input_shapes[2]
            source_map = [f"i{i}" for i in range(len(source_shape))]
            if 0 <= dim < len(shape):
                index_map = [f"i{dim}"]
            maps.append(InputIndexMap(inputs[1], tuple(index_map), "index"))
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
    dims = _literal_ints(_argument(node, 1, "dim"))
    if dims is None:
        dims = tuple(range(len(shape)))
    if any(dim < -len(shape) or dim >= len(shape) for dim in dims):
        return (IterationAxis("r0", "UNKNOWN"),)
    axes = []
    for index, dim in enumerate(dims):
        axis = dim + len(shape) if dim < 0 else dim
        axes.append(IterationAxis(f"r{index}", shape[axis] if axis < len(shape) else "UNKNOWN"))
    return tuple(axes)


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
        if any("UNKNOWN" in item.expressions for item in maps) and kind != "Unsupported":
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS, f"indexed operation {node.name} has an unproved index map",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"index_maps": [item.to_dict() for item in maps]},
            ))
        if kind in {"Gather", "Scatter/StateWrite"} and operator in {
            "index_select", "gather", "take", "index_copy", "index_put", "slice_scatter", "scatter"
        }:
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
            EffectEdge(origin_id, effect.kind, effect.target, effect.required)
            for effect in program.effects
            for origin_id in program.origin_map.get(effect.node_id, ())
            if effect.node_id == node.name or origin_id in source_origins
        )
        attrs = {"arguments": refs.to_dict()["args"], "keywords": refs.to_dict()["kwargs"]}
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
    coverage, coverage_diagnostics = verify_indexed_program(program, tuple(operations), node_to_value, tuple(outputs))
    diagnostics.extend(coverage_diagnostics)
    return IndexedTensorProgram(program, tuple(values), tuple(operations), tuple(outputs),
                                tuple(coverage), tuple(diagnostics))


__all__ = ["AliasEdge", "EffectEdge", "IndexedOp", "IndexedTensorProgram", "IndexedValue",
           "InputIndexMap", "IterationAxis", "OutputLeaf", "lower_indexed_program"]
