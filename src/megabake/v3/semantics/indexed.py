"""Origin-complete, backend-neutral indexed FX semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from ..diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ..frontend.capture import NormalizedProgram, canonical_output_leaves, graph_hash
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
    writer_values: tuple[str, ...] = ()

    def validate_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self.capacity:
            raise ValueError(f"append index {index!r} is outside [0, {self.capacity})")

    def to_dict(self) -> dict[str, Any]:
        result = {"effect_id": self.effect_id, "state_id": self.state_id,
                "old_value": self.old_value, "new_value": self.new_value,
                "index_value": self.index_value,
                "valid_length_before": self.valid_length_before,
                "valid_length_after": self.valid_length_after,
                "capacity": self.capacity, "axis": self.axis, "order": self.order,
                "write_footprint": dict(self.write_footprint), "alias_rule": self.alias_rule}
        if self.writer_values:
            result["writer_values"] = list(self.writer_values)
        return result


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
                "step_abi_hash": getattr(self.source_program.step_abi, "contract_hash", None),
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
        "tanh", "relu", "silu", "gelu", "cos", "sin", "where", "eq", "ne", "gt", "ge", "lt", "le", "__and__", "to",
        "_to_copy", "convert_element_type", "maximum", "minimum", "pow",
    )},
    **{name: "Broadcast/View" for name in (
        "view", "reshape", "transpose", "permute", "t", "slice", "select", "squeeze", "unsqueeze",
        "expand", "detach", "detach_", "alias", "flatten", "as_strided",
    )},
    "contiguous": "Map",
    **{name: "Reduce" for name in ("sum", "mean", "amax", "max")},
    **{name: "Contraction" for name in ("mm", "bmm", "addmm", "matmul", "linear")},
    **{name: "Cat/Stack" for name in ("cat", "stack")},
    **{name: "Gather" for name in ("index_select", "gather", "take")},
    "embedding": "Gather",
    "index": "Gather",
    **{name: "Constructor" for name in ("arange", "ones", "new_ones")},
    "lift_fresh_copy": "Copy",
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


def _cat_stack_contract(node: Any, operator: str, inputs: tuple[str, ...],
                        input_nodes: tuple[Any, ...], facts: FactTable,
                        output_fact: TensorFacts | None) -> tuple[
                            Mapping[str, Any] | None, tuple[InputIndexMap, ...], str | None
                        ]:
    if operator not in {"cat", "stack"}:
        return None, (), None
    fail = lambda reason: (None, tuple(InputIndexMap(value, ("UNKNOWN",), "cat_stack")
                                        for value in inputs), reason)
    if (not inputs or len(inputs) != len(input_nodes) or output_fact is None or
            not isinstance(node.args[0] if node.args else None, (tuple, list)) or
            len(node.args[0]) != len(inputs)):
        return fail("tensor list or output facts are missing")
    facts_in = [facts.for_node(item) for item in input_nodes]
    if any(item is None for item in facts_in):
        return fail("an input has no tensor facts")
    shapes = [tuple(item.shape) for item in facts_in if item is not None]
    output_shape = tuple(output_fact.shape)
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0
           for shape in shapes + [output_shape] for size in shape):
        return fail("dynamic or invalid tensor dimensions are unsupported")
    if (output_fact.layout != "strided" or
            any(item.layout != "strided" for item in facts_in if item is not None)):
        return fail("non-strided tensors are unsupported")
    if any(item.dtype != output_fact.dtype for item in facts_in if item is not None):
        return fail("input and output dtypes differ")

    dimension = _argument(node, 1, "dim", 0)
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        return fail("dimension is not a static integer")
    rank = len(output_shape)
    if operator == "cat":
        if rank == 0 or dimension < -rank or dimension >= rank:
            return fail("cat dimension is outside the output rank")
        dimension %= rank
        normalized_shapes = []
        for shape in shapes:
            if shape == (0,) and len(shape) != rank:
                normalized_shapes.append(None)
            elif len(shape) == rank:
                normalized_shapes.append(shape)
            else:
                return fail("cat inputs must have equal rank, except a 1-D empty input")
        positive_shapes = [shape for shape in normalized_shapes if shape is not None]
        if not positive_shapes:
            if output_shape != (0,) or rank != 1:
                return fail("all-empty cat output shape is inconsistent")
        elif any(any(shape[axis] != positive_shapes[0][axis]
                     for axis in range(rank) if axis != dimension)
                 for shape in positive_shapes[1:]):
            return fail("cat non-concatenated dimensions differ")
        segments = [0 if shape is None else shape[dimension] for shape in normalized_shapes]
        expected = list(positive_shapes[0] if positive_shapes else (0,))
        expected[dimension] = sum(segments)
        if tuple(expected) != output_shape:
            return fail("cat output shape does not equal its input segments")
        maps = []
        offset = 0
        for value, shape, extent in zip(inputs, normalized_shapes, segments):
            if shape is None:
                maps.append(InputIndexMap(value, ("0",), "empty_cat"))
            else:
                expressions = [f"i{axis}" for axis in range(rank)]
                expressions[dimension] = f"i{dimension}-{offset}" if offset else f"i{dimension}"
                maps.append(InputIndexMap(value, tuple(expressions), "cat"))
            offset += extent
        return ({"operator": "cat", "dimension": dimension, "segments": segments}, tuple(maps), None)

    if rank == 0 or dimension < -(rank) or dimension >= rank:
        return fail("stack dimension is outside the output rank")
    dimension %= rank
    if any(len(shape) != rank - 1 for shape in shapes) or any(shape != shapes[0] for shape in shapes[1:]):
        return fail("stack inputs must have identical shapes one rank below the output")
    expected = list(shapes[0])
    expected.insert(dimension, len(inputs))
    if tuple(expected) != output_shape:
        return fail("stack output shape does not match its input count and dimension")
    maps = tuple(InputIndexMap(
        value, tuple(f"i{axis if axis < dimension else axis + 1}" for axis in range(rank - 1)), "stack"
    ) for value in inputs)
    return ({"operator": "stack", "dimension": dimension, "segments": [1] * len(inputs)}, maps, None)


def _tensor_constructor_contract(node: Any, operator: str,
                                 output_fact: TensorFacts | None) -> tuple[Mapping[str, Any] | None, str | None]:
    if output_fact is None or output_fact.layout != "strided":
        return None, "constructor output facts or dense layout are missing"
    output_shape = tuple(output_fact.shape)
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in output_shape):
        return None, "constructor output shape is dynamic or invalid"
    if operator == "arange":
        args = node.args
        if len(args) == 1:
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            start, end, step = args
        else:
            return None, "arange requires one to three static integer bounds"
        if (any(not isinstance(value, int) or isinstance(value, bool)
                for value in (start, end, step)) or step == 0 or len(output_shape) != 1 or
                output_fact.dtype not in {"int32", "int64"}):
            return None, "arange bounds, rank, dtype, or step are unsupported"
        if len(range(start, end, step)) != output_shape[0]:
            return None, "arange output extent differs from its exact range"
        return {"operator": "arange", "start": start, "stop": end, "step": step}, None
    if operator == "ones":
        shape = _literal_ints(node.args[0]) if node.args else None
        fill = 1
    else:
        shape = _literal_ints(node.args[1]) if len(node.args) > 1 else None
        fill = 1
    if shape is None or any(size < 0 for size in shape) or tuple(shape) != output_shape:
        return None, f"{operator} needs an exact static output shape"
    if output_fact.dtype not in {"bool", "int32", "int64", "float16", "bfloat16", "float32"}:
        return None, f"{operator} output dtype is unsupported"
    return {"operator": operator, "shape": list(shape), "fill": fill}, None


def _arange_pattern(node: Any, facts: FactTable, depth: int = 0) -> tuple[int, int, int] | None:
    """Recognize exact integer ranges through zero-add and shape-only wrappers."""
    if depth > 8 or getattr(node, "op", None) != "call_function":
        return None
    operator, _ = _operator(node)
    if operator == "arange":
        args = node.args
        if len(args) == 1:
            result = (0, args[0], 1)
        elif len(args) == 2:
            result = (args[0], args[1], 1)
        elif len(args) == 3:
            result = tuple(args)
        else:
            return None
        return result if all(isinstance(value, int) and not isinstance(value, bool) for value in result) and result[2] != 0 else None
    if operator == "add" and len(node.args) >= 2:
        left, right = node.args[:2]
        scalar = right if isinstance(right, int) and not isinstance(right, bool) else left
        source = left if scalar is right else right
        alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
        if scalar == 0 and alpha == 1 and hasattr(source, "op"):
            return _arange_pattern(source, facts, depth + 1)
        return None
    if operator in {"unsqueeze", "to", "_to_copy", "detach", "detach_", "alias"} and node.args:
        return _arange_pattern(node.args[0], facts, depth + 1)
    return None


def _broadcast_shape(shapes: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...] | None:
    rank = max((len(shape) for shape in shapes), default=0)
    result = [1] * rank
    for shape in shapes:
        padded = (1,) * (rank - len(shape)) + shape
        for axis, size in enumerate(padded):
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                return None
            if result[axis] not in (1, size) and size != 1:
                return None
            result[axis] = max(result[axis], size)
    return tuple(result)


def _advanced_index_contract(node: Any, inputs: tuple[str, ...], input_nodes: tuple[Any, ...],
                             facts: FactTable, output_fact: TensorFacts | None) -> tuple[
                                 Mapping[str, Any] | None, tuple[InputIndexMap, ...], str | None
                             ]:
    """Prove dense two-axis indexing by zero and ascending unit-step ranges."""
    fail = lambda reason: (None, tuple(InputIndexMap(value, ("UNKNOWN",), "advanced_index")
                                        for value in inputs), reason)
    index_args = node.args[1] if len(node.args) > 1 else None
    if (not isinstance(index_args, (tuple, list)) or len(index_args) != 2 or
            output_fact is None or len(inputs) != 3 or len(input_nodes) != 3 or
            any(not hasattr(item, "op") for item in index_args)):
        return fail("only two tensor indices over a rank-two base are supported")
    base_fact, first_fact, second_fact = (facts.for_node(item) for item in input_nodes)
    if any(item is None for item in (base_fact, first_fact, second_fact)):
        return fail("base or index tensors lack exact facts")
    assert base_fact is not None and first_fact is not None and second_fact is not None
    base_shape = tuple(base_fact.shape)
    index_shapes = (tuple(first_fact.shape), tuple(second_fact.shape))
    output_shape = tuple(output_fact.shape)
    dimensions = base_shape + output_shape + tuple(size for shape in index_shapes for size in shape)
    if (len(base_shape) != 2 or base_fact.layout != "strided" or
            any(fact.dtype not in {"int32", "int64"} for fact in (first_fact, second_fact)) or
            any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in dimensions)):
        return fail("advanced index requires a dense rank-two base and static integer index tensors")
    if _broadcast_shape(index_shapes) != output_shape:
        return fail("advanced index tensor shapes do not broadcast to the exact output shape")
    expressions: list[str] = []
    ranges = []
    for axis, (index_node, index_shape, extent) in enumerate(zip(index_args, index_shapes, base_shape)):
        pattern = _arange_pattern(index_node, facts)
        if pattern is None:
            return fail("advanced index values are not statically bounded arange expressions")
        start, stop, step = pattern
        try:
            sequence = range(start, stop, step)
        except ValueError:
            return fail("advanced index has an invalid range step")
        if not sequence or sequence[0] < 0 or sequence[-1] >= extent:
            return fail("advanced index range falls outside the base dimension")
        padded = (1,) * (len(output_shape) - len(index_shape)) + index_shape
        varying_axes = [i for i, size in enumerate(padded) if size != 1]
        if len(sequence) == 1 and sequence[0] == 0:
            expressions.append("0")
        elif (start == 0 and step == 1 and len(sequence) == extent and len(varying_axes) == 1
              and output_shape[varying_axes[0]] == extent):
            expressions.append(f"i{varying_axes[0]}")
        else:
            return fail("advanced index is not a zero or identity range on one output axis")
        ranges.append({"base_axis": axis, "lower": min(sequence),
                       "upper_exclusive": max(sequence) + 1,
                       "output_axis": varying_axes[0] if len(varying_axes) == 1 else None})
    maps = [InputIndexMap(inputs[0], tuple(expressions), "advanced_index")]
    for value, shape in zip(inputs[1:], index_shapes):
        padded = (1,) * (len(output_shape) - len(shape)) + shape
        maps.append(InputIndexMap(value, tuple("0" if size == 1 else f"i{axis}"
                                                for axis, size in enumerate(padded)),
                                  "advanced_index_value"))
    return ({"base_shape": list(base_shape), "output_shape": list(output_shape),
             "base_axis_expressions": expressions, "index_ranges": ranges}, tuple(maps), None)


def _embedding_contract(node: Any, input_facts: list[TensorFacts | None],
                        output_fact: TensorFacts | None) -> Mapping[str, Any] | None:
    if len(input_facts) != 2 or any(item is None for item in input_facts) or output_fact is None:
        return None
    weight, indices = input_facts
    assert weight is not None and indices is not None
    padding_idx = _argument(node, 2, "padding_idx", -1)
    scale_grad = _argument(node, 3, "scale_grad_by_freq", False)
    sparse = _argument(node, 4, "sparse", False)
    if (len(weight.shape) != 2 or not indices.shape or indices.dtype not in {"int32", "int64"} or
            tuple(output_fact.shape) != tuple(indices.shape) + (weight.shape[1],) or
            output_fact.dtype != weight.dtype or weight.layout != "strided" or
            not isinstance(padding_idx, int) or isinstance(padding_idx, bool) or
            not isinstance(scale_grad, bool) or not isinstance(sparse, bool)):
        return None
    if padding_idx < -weight.shape[0] or padding_idx >= weight.shape[0]:
        return None
    return {"vocabulary": weight.shape[0], "embedding_dim": weight.shape[1],
            "padding_idx": padding_idx, "scale_grad_by_freq": scale_grad, "sparse": sparse,
            "index_bounds": {"value_id": "input_indices", "lower": 0,
                             "upper_exclusive": weight.shape[0], "guard": "per_invocation"}}


def _metadata_guard_contract(node: Any, input_nodes: tuple[Any, ...],
                             facts: FactTable) -> Mapping[str, Any] | None:
    """Prove a metadata assertion from exact captured tensor facts and ABI guards."""
    if _operator(node)[0] != "_assert_tensor_metadata" or len(input_nodes) != 1:
        return None
    fact = facts.for_node(input_nodes[0])
    if fact is None:
        return None
    kwargs = dict(node.kwargs)
    provided = {}
    if len(node.args) > 1 and node.args[1] is not None:
        provided["shape"] = tuple(node.args[1])
    if "sizes" in kwargs and kwargs["sizes"] is not None:
        provided["shape"] = tuple(kwargs["sizes"])
    if "strides" in kwargs and kwargs["strides"] is not None:
        provided["strides"] = tuple(kwargs["strides"])
    if kwargs.get("dtype") is not None:
        provided["dtype"] = str(kwargs["dtype"]).removeprefix("torch.")
    if kwargs.get("device") is not None:
        provided["device"] = str(kwargs["device"]).removeprefix("torch.")
    if kwargs.get("layout") is not None:
        provided["layout"] = str(kwargs["layout"]).removeprefix("torch.")
    exact = {"shape": tuple(fact.shape), "strides": tuple(fact.strides),
             "dtype": fact.dtype, "device": fact.device, "layout": fact.layout}
    runtime_asserted = {name: value for name, value in provided.items()
                        if name == "device" and exact[name] == "unknown"}
    if any(exact.get(name) != value for name, value in provided.items()
           if name not in runtime_asserted):
        return None
    return {"statically_proven": True, "value_id": fact.value_id,
            "metadata": {name: list(value) if isinstance(value, tuple) else value
                         for name, value in exact.items()},
            "asserted": {name: list(value) if isinstance(value, tuple) else value
                         for name, value in provided.items()},
            "runtime_asserted": runtime_asserted,
            "proof": "assertion metadata equals exact captured fact; invocation binding guards its source"}


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
    if kind == "Constructor":
        return ()
    if kind == "Copy" and len(inputs) == 1 and input_shapes[0] == output_shape:
        return (InputIndexMap(inputs[0], tuple(f"i{axis}" for axis in range(rank)), "copy"),)
    if kind == "Gather" and operator == "index":
        _, advanced_maps, _ = _advanced_index_contract(
            node, inputs, input_nodes, facts, facts.for_node(node)
        )
        return advanced_maps
    if kind == "Gather" and operator == "embedding" and len(inputs) == 2:
        weight_shape, index_shape = input_shapes
        if (len(weight_shape) != 2 or not index_shape or len(output_shape) != len(index_shape) + 1 or
                output_shape != tuple(index_shape) + (weight_shape[1],)):
            return (InputIndexMap(inputs[0], ("UNKNOWN",), "embedding"),
                    InputIndexMap(inputs[1], ("UNKNOWN",), "embedding_index"))
        return (
            InputIndexMap(inputs[0], ("index[" + ",".join(f"i{axis}" for axis in range(len(index_shape))) + "]",
                                      f"i{len(index_shape)}"), "embedding"),
            InputIndexMap(inputs[1], tuple(f"i{axis}" for axis in range(len(index_shape))),
                          "embedding_index"),
        )
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
        elif operator == "matmul":
            left, right = input_shapes
            if not left or not right:
                return tuple(InputIndexMap(value, ("UNKNOWN",), "contraction") for value in inputs)
            left_vector, right_vector = len(left) == 1, len(right) == 1
            left_k = left[-1]
            right_k = right[-1] if right_vector else right[-2]
            if left_k != right_k:
                return tuple(InputIndexMap(value, ("UNKNOWN",), "contraction") for value in inputs)
            left_batch, right_batch = left[:-2], right[:-2]
            batch_rank = max(len(left_batch), len(right_batch))
            batch_shape = output_shape[:batch_rank]
            expected_shape = batch_shape + (() if left_vector and right_vector else
                ((right[-1],) if left_vector else ()))
            if not left_vector and not right_vector:
                expected_shape = batch_shape + (left[-2], right[-1])
            elif right_vector and not left_vector:
                expected_shape = batch_shape + (left[-2],)
            if tuple(expected_shape) != tuple(output_shape):
                return tuple(InputIndexMap(value, ("UNKNOWN",), "contraction") for value in inputs)
            left_expressions = list(_broadcast_map(left_batch, batch_shape))
            right_expressions = list(_broadcast_map(right_batch, batch_shape))
            if left_vector:
                left_expressions = ["k"]
            else:
                left_expressions.extend((f"i{len(batch_shape)}", "k"))
            if right_vector:
                right_expressions = ["k"]
            else:
                right_expressions.extend(("k", f"i{len(batch_shape) + 1}"))
            maps.append(InputIndexMap(inputs[0], tuple(left_expressions), "contraction"))
            maps.append(InputIndexMap(inputs[1], tuple(right_expressions), "contraction"))
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


def _graph_ancestors(node: Any) -> tuple[Any, ...]:
    pending = list(getattr(node, "all_input_nodes", ()))
    seen = set()
    result = []
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        pending.extend(getattr(item, "all_input_nodes", ()))
    return tuple(result)


def _transport_source(node: Any) -> Any:
    while getattr(node, "op", None) == "call_function":
        operator, _ = _operator(node)
        if operator not in {"to", "_to_copy", "unsqueeze", "squeeze", "view", "reshape", "expand", "detach", "detach_", "alias"}:
            break
        if not node.args or not hasattr(node.args[0], "op"):
            break
        node = node.args[0]
    return node


def _cache_prefix_coordinates(program: NormalizedProgram, facts: FactTable, prefix: Any,
                              valid_length: int, value_index: int) -> tuple[int, int] | None:
    """Prove one layer's old KV prefix is the first input to a length-L cat."""
    abi = program.step_abi
    if abi is None or len(abi.old_state_inputs) != 1:
        return None
    state = abi.old_state_inputs[0]
    state_name = state["placeholder"]
    prefix_fact = facts.for_node(prefix)
    if prefix_fact is None or len(prefix_fact.shape) != 4 or prefix_fact.shape[2] != valid_length:
        return None
    ancestors = _graph_ancestors(prefix)
    slices = []
    for item in ancestors:
        if item.op != "call_function" or _operator(item)[0] != "slice" or len(item.args) < 4:
            continue
        dim, start, stop = item.args[1:4]
        step = item.args[4] if len(item.args) > 4 else 1
        if (dim, start, stop, step) != (2, 0, valid_length, 1):
            continue
        current = item.args[0]
        selections = []
        while getattr(current, "op", None) == "call_function" and _operator(current)[0] == "select":
            if len(current.args) < 3:
                break
            selections.append(current)
            current = current.args[0]
        if (getattr(current, "op", None) != "placeholder" or current.name != state_name or
                len(selections) != 2):
            continue
        layer_select, kv_select = reversed(selections)
        if (layer_select.args[1] != 0 or kv_select.args[1] != 0 or
                not isinstance(layer_select.args[2], int) or
                kv_select.args[2] != value_index):
            continue
        layer = layer_select.args[2]
        layers = tuple(abi.guard_set.get("shapes", {}).get(state_name, ()))
        if len(layers) != 6 or not 0 <= layer < layers[0]:
            continue
        slices.append((item, layer))
    if len(slices) != 1:
        return None
    slice_node, layer = slices[0]
    # The empty tensor is the DynamicCache empty-prefix identity used at L=0;
    # at nonzero L it remains the same exact cat form around the old-cache view.
    prefix_cats = [item for item in (prefix,) + ancestors if item.op == "call_function" and
                   _operator(item)[0] == "cat" and
                   facts.for_node(item) is not None and facts.for_node(item).shape == prefix_fact.shape]
    found = False
    for cat in prefix_cats:
        sources = _ordered_nodes(cat.args[0]) if cat.args else ()
        if len(sources) != 2 or len(cat.args) < 2:
            continue
        dim = cat.args[1] + 4 if isinstance(cat.args[1], int) and cat.args[1] < 0 else cat.args[1]
        source_facts = tuple(facts.for_node(item) for item in sources)
        if dim != 2 or any(item is None for item in source_facts):
            continue
        if (any(item is slice_node for item in sources) and
                any(tuple(item.shape) == (0,) and not any(
                    ancestor.name == state_name for ancestor in _graph_ancestors(source)
                ) for source, item in zip(sources, source_facts))):
            found = True
            break
    return (layer, value_index) if found else None


def _attention_sequence_source(program: NormalizedProgram, facts: FactTable,
                               root: Any, valid_length: int,
                               value_index: int, batch: int, depth: int) -> tuple[int, int] | None:
    sequence_cat = _transport_source(root)
    if (getattr(sequence_cat, "op", None) != "call_function" or
            _operator(sequence_cat)[0] != "cat" or facts.for_node(sequence_cat) is None or
            len(facts.for_node(sequence_cat).shape) != 4 or
            facts.for_node(sequence_cat).shape[2] != valid_length + 1):
        return None
    ancestors = _graph_ancestors(root)
    cat_fact = facts.for_node(sequence_cat)
    sources = _ordered_nodes(sequence_cat.args[0]) if sequence_cat.args else ()
    dim = sequence_cat.args[1] if len(sequence_cat.args) > 1 else None
    dim = dim + 4 if isinstance(dim, int) and dim < 0 else dim
    if (dim != 2 or len(sources) != 2 or cat_fact.shape[0] != batch or
            cat_fact.shape[-1] != depth):
        return None
    prefix_fact, current_fact = (facts.for_node(item) for item in sources)
    if (prefix_fact is None or current_fact is None or len(prefix_fact.shape) != 4 or
            len(current_fact.shape) != 4 or prefix_fact.shape[0] != batch or
            prefix_fact.shape[2:] != (valid_length, depth) or
            current_fact.shape != prefix_fact.shape[:2] + (1, depth)):
        return None
    # The prior tokens must come from this layer's matching K or V cache view;
    # the new token must come from that same layer's projection path.
    coordinates = _cache_prefix_coordinates(program, facts, sources[0], valid_length, value_index)
    if coordinates is None:
        return None
    layer, _ = coordinates
    projection = "k" if value_index == 0 else "v"
    expected_suffix = f".layers.{layer}.self_attn.{projection}_proj.weight"
    ancestors = _graph_ancestors(sources[1])
    matching_weights = {
        item for item in ancestors
        if item.op == "placeholder" and item.name in program.lifted_bindings and
        program.lifted_bindings[item.name].role == "weight" and
        str(program.lifted_bindings[item.name].target).endswith(expected_suffix)
    }
    projections = [item for item in ancestors if item.op == "call_function" and
                   _operator(item)[0] == "linear" and len(item.args) > 1 and
                   item.args[1] in matching_weights and facts.for_node(item) is not None and
                   facts.for_node(item).shape == (batch, 1, prefix_fact.shape[1] * depth)]
    if len(matching_weights) != 1 or len(projections) != 1:
        return None
    return layer, prefix_fact.shape[1]


def _mask_is_exact_valid_prefix(program: NormalizedProgram, facts: FactTable,
                                mask_node: Any, valid_length: int) -> bool:
    def leaves(node: Any, depth: int = 0) -> list[Any] | None:
        if depth > 16:
            return None
        node = _transport_source(node)
        if getattr(node, "op", None) != "call_function":
            return [node]
        operator, _ = _operator(node)
        if operator == "__and__" and len(node.args) >= 2:
            left, right = leaves(node.args[0], depth + 1), leaves(node.args[1], depth + 1)
            return None if left is None or right is None else left + right
        return [node]

    leaf_nodes = leaves(mask_node)
    if leaf_nodes is None or len(leaf_nodes) != 3:
        return False
    operators = [_operator(item)[0] if getattr(item, "op", None) == "call_function" else ""
                 for item in leaf_nodes]
    if sorted(operators) != ["index", "le", "new_ones"]:
        return False
    by_operator = dict(zip(operators, leaf_nodes))
    true_scalar = by_operator["new_ones"]
    true_contract, error = _tensor_constructor_contract(true_scalar, "new_ones", facts.for_node(true_scalar))
    if error or not true_contract or tuple(true_contract.get("shape", ())) or true_contract.get("fill") != 1:
        return False
    comparison = by_operator["le"]
    if (len(comparison.args) < 2 or
            _arange_pattern(comparison.args[0], facts) != (0, valid_length + 1, 1) or
            _constant_ints(program, comparison.args[1]) != (valid_length,)):
        return False
    index = by_operator["index"]
    contract, _, error = _advanced_index_contract(
        index, tuple(facts.node_to_value[item.name] for item in index.all_input_nodes),
        tuple(index.all_input_nodes), facts, facts.for_node(index),
    )
    if error or contract is None:
        return False
    base = index.args[0]
    base_nodes = (base,) + _graph_ancestors(base)
    ones = [item for item in base_nodes if item.op == "call_function" and _operator(item)[0] == "ones"]
    if len(ones) != 1:
        return False
    ones_contract, error = _tensor_constructor_contract(ones[0], "ones", facts.for_node(ones[0]))
    if error or not ones_contract or tuple(ones_contract.get("shape", ())) != (1, valid_length + 1):
        return False
    ranges = contract.get("index_ranges", ())
    return (tuple(contract.get("output_shape", ())) == (1, 1, 1, valid_length + 1) and
            len(ranges) == 2 and ranges[0].get("lower") == ranges[0].get("upper_exclusive") - 1 == 0 and
            ranges[1].get("lower") == 0 and ranges[1].get("upper_exclusive") == valid_length + 1)


def _attention_contract(program: NormalizedProgram, facts: FactTable,
                        input_facts: list[TensorFacts | None], output_fact: TensorFacts | None,
                        node: Any, inputs: tuple[str, ...],
                        transitions: Mapping[str, StateTransition]) -> Mapping[str, Any] | None:
    if len(inputs) != 4 or any(fact is None for fact in input_facts) or output_fact is None:
        return None
    q, k, v, mask = input_facts
    assert q is not None and k is not None and v is not None and mask is not None
    if any(len(fact.shape) != 4 for fact in (q, k, v, mask, output_fact)):
        return None
    abi = program.step_abi
    position = abi.guard_set.get("position", {}) if abi is not None else {}
    valid_length = position.get("specialized") if isinstance(position, Mapping) else None
    if not isinstance(valid_length, int) or valid_length < 0:
        return None
    batch, heads_q, query_length, depth = q.shape
    _, heads_kv, sequence_length, _ = k.shape
    if (not all(isinstance(dim, int) and dim > 0 for dim in
                (batch, heads_q, heads_kv, sequence_length, depth))
            or query_length != 1 or
            k.shape != v.shape or output_fact.shape != q.shape
            or k.shape[0] != batch or k.shape[-1] != depth or heads_q % heads_kv
            or mask.shape != (batch, mask.shape[1], 1, sequence_length)
            or mask.shape[1] not in (1, heads_q) or mask.dtype != "bool"
            or q.dtype not in {"float16", "bfloat16", "float32"}
            or any(fact.dtype != q.dtype for fact in (k, v, output_fact))
            or any(tuple(fact.strides) != tuple(math.prod(fact.shape[i + 1:])
                                                     for i in range(4)) for fact in (k, v))):
        return None
    cache_capacity = abi.old_state_inputs[0]["capacity"]
    if valid_length >= cache_capacity:
        return None
    key_source = _attention_sequence_source(program, facts, node.args[1], valid_length, 0, batch, depth)
    value_source = _attention_sequence_source(program, facts, node.args[2], valid_length, 1, batch, depth)
    position_value_id = None
    mask_value_id = None
    if (sequence_length == valid_length + 1 and key_source is not None and value_source is not None and
            key_source[0] == value_source[0] and key_source[1] == value_source[1] and
            _mask_is_exact_valid_prefix(program, facts, node.args[3], valid_length)):
        source_kv_heads = key_source[1]
        mask_rule = "captured_boolean_true_prefix_[0,L+1); no unwritten slot is in the key/value tensors"
    else:
        key_transition = transitions.get(inputs[1])
        value_transition = transitions.get(inputs[2])
        key_state = next((item for item in abi.old_state_inputs
                          if key_transition is not None and item["state_id"] == key_transition.state_id), None)
        value_state = next((item for item in abi.old_state_inputs
                            if value_transition is not None and item["state_id"] == value_transition.state_id), None)
        if (sequence_length != cache_capacity or mask.role != "input" or
                key_transition is None or value_transition is None or key_state is None or value_state is None or
                key_transition.index_value != value_transition.index_value or
                key_transition.capacity != cache_capacity or value_transition.capacity != cache_capacity or
                key_transition.axis != 2 or value_transition.axis != 2 or
                key_state["capacity"] != cache_capacity or value_state["capacity"] != cache_capacity):
            return None
        source_kv_heads = heads_kv
        position_value_id = key_transition.index_value
        mask_value_id = inputs[3]
        mask_rule = "caller_boolean_mask; runtime rejects true entries at or beyond L+1"
    dropout = _argument(node, 4, "dropout_p", 0.0)
    causal = _argument(node, 5, "is_causal", False)
    scale = _argument(node, 6, "scale")
    gqa = _argument(node, 7, "enable_gqa", False)
    if (dropout not in (0, 0.0) or causal is not False or
            gqa is not (heads_q != heads_kv) or
            scale is not None and (not isinstance(scale, (int, float)) or
                                   not math.isfinite(scale))):
        return None
    result = {"batch": batch, "heads_q": heads_q, "heads_kv": heads_kv,
              "capacity": sequence_length, "cache_capacity": cache_capacity,
              "position": valid_length, "head_dim": depth, "mask_heads": mask.shape[1],
              "scale": float(scale) if scale is not None else depth ** -0.5,
              "state_effect": abi.state_effects[0]["effect_id"],
              "source_kv_heads": source_kv_heads,
              "gqa_repeat": heads_q // source_kv_heads,
              "mask_rule": mask_rule}
    if position_value_id is not None:
        result["position_value_id"] = position_value_id
    if mask_value_id is not None:
        result["mask_value_id"] = mask_value_id
    return result


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


def _constant_ints(program: NormalizedProgram, node: Any, seen: set[str] | None = None) -> tuple[int, ...] | None:
    """Read an immutable captured integer constant through copy/detach wrappers."""
    if isinstance(node, int) and not isinstance(node, bool):
        return (node,)
    if getattr(node, "op", None) not in {"placeholder", "get_attr", "call_function"}:
        return None
    seen = set() if seen is None else seen
    if getattr(node, "name", None) in seen:
        return None
    if getattr(node, "name", None):
        seen.add(node.name)
    value = None
    if node.op == "placeholder":
        binding = program.lifted_bindings.get(node.name)
        if binding is None or binding.role != "constant":
            return None
        try:
            value = program.binding_for(node.name)
        except (KeyError, ValueError):
            return None
    elif node.op == "get_attr":
        value = program.graph_module
        try:
            for component in str(node.target).split("."):
                value = getattr(value, component)
        except AttributeError:
            return None
    elif node.op == "call_function":
        operator, _ = _operator(node)
        if operator == "arange":
            args = node.args
            if len(args) == 1:
                start, stop, step = 0, args[0], 1
            elif len(args) == 2:
                start, stop, step = args[0], args[1], 1
            elif len(args) == 3:
                start, stop, step = args
            else:
                return None
            if (any(not isinstance(item, int) or isinstance(item, bool)
                    for item in (start, stop, step)) or step == 0):
                return None
            result = tuple(range(start, stop, step))
            return result if len(result) <= 1_000_000 else None
        if operator in {"add", "sub"} and len(node.args) >= 2:
            left, right = (_constant_ints(program, item, seen.copy()) for item in node.args[:2])
            alpha = _argument(node, 2, "alpha", 1)
            if not isinstance(alpha, int) or alpha != 1 or left is None or right is None:
                return None
            if len(left) == 1 and len(right) > 1:
                left = left * len(right)
            elif len(right) == 1 and len(left) > 1:
                right = right * len(left)
            if len(left) != len(right):
                return None
            return tuple(a + b if operator == "add" else a - b for a, b in zip(left, right))
        if operator not in {"detach", "detach_", "lift_fresh_copy", "alias", "to", "_to_copy",
                            "unsqueeze", "squeeze", "view", "reshape"} or not node.args:
            return None
        return _constant_ints(program, node.args[0], seen)
    else:
        return None
    try:
        values = value.detach().cpu().reshape(-1).tolist()
    except (AttributeError, RuntimeError, TypeError):
        return None
    if not isinstance(values, list) or any(not isinstance(item, int) or isinstance(item, bool)
                                           for item in values):
        return None
    return tuple(values)


def _selected_state_view(program: NormalizedProgram, node: Any, state_name: str,
                         rank: int) -> tuple[dict[int, int], tuple[int, ...]] | None:
    """Resolve only static ``select`` views back to one declared state input."""
    chain = []
    current = node
    while getattr(current, "name", None) != state_name:
        if (getattr(current, "op", None) != "call_function" or
                _operator(current)[0] != "select" or len(current.args) < 3 or len(chain) >= rank):
            return None
        chain.append(current)
        current = current.args[0]
    axes = list(range(rank))
    coordinates: dict[int, int] = {}
    for current in reversed(chain):
        dim, index = current.args[1:3]
        if not isinstance(dim, int):
            return None
        dim = dim + len(axes) if dim < 0 else dim
        if not 0 <= dim < len(axes):
            return None
        selected = _constant_ints(program, index) if hasattr(index, "op") else (index,)
        if selected is None or len(selected) != 1 or selected[0] < 0:
            return None
        coordinates[axes.pop(dim)] = selected[0]
    return coordinates, tuple(axes)


def _stack_writer_paths(node: Any, writer_names: set[str]) -> tuple[tuple[tuple[int, ...], str], ...] | None:
    """Return writer leaves only for a nested, dimension-zero stack assembly."""
    if node.name in writer_names:
        return (((), node.name),)
    if getattr(node, "op", None) != "call_function" or _operator(node)[0] != "stack":
        return None
    values = node.args[0] if node.args else None
    dim = node.args[1] if len(node.args) > 1 else 0
    if not isinstance(values, (tuple, list)) or not values or dim != 0:
        return None
    leaves = []
    for index, value in enumerate(values):
        if not hasattr(value, "op"):
            return None
        child = _stack_writer_paths(value, writer_names)
        if child is None:
            return None
        leaves.extend(((index,) + path, name) for path, name in child)
    return tuple(leaves)


def _grouped_cache_transition(program: NormalizedProgram, facts: FactTable) -> tuple[
        StateTransition | None, Mapping[str, StateTransition], frozenset[str],
        tuple[DiagnosticRecord, ...]]:
    """Prove a full grouped append from selected state views through one output stack."""
    abi = program.step_abi
    if abi is None or len(abi.state_effects) != 1 or len(abi.old_state_inputs) != 1 or len(abi.new_state_outputs) != 1:
        return None, {}, frozenset(), ()
    effect_spec, state_spec, output_spec = abi.state_effects[0], abi.old_state_inputs[0], abi.new_state_outputs[0]
    effect_id = str(effect_spec["effect_id"])
    state_id = str(effect_spec["state_id"])
    if state_spec["state_id"] != state_id or output_spec["state_id"] != state_id:
        return None, {}, frozenset(), ()
    effects = tuple(effect for effect in program.effects
                    if effect.required and effect.kind == "functional_state_update" and effect.target == state_id)
    writer_nodes_by_name = {node.name: node for node in program.graph_module.graph.nodes}
    writer_names = frozenset(effect.node_id for effect in effects)
    if len(writer_names) <= 1:
        return None, {}, frozenset(), ()
    fail = lambda message, **details: (
        None, {}, writer_names,
        (DiagnosticRecord(DiagnosticCode.MISSING_FACTS, message, DiagnosticSeverity.ERROR,
                          node_id=state_id, details={"effect_id": effect_id, **details}),)
    )
    nodes = [writer_nodes_by_name.get(name) for name in writer_names]
    if any(node is None or node.op != "call_function" or _operator(node)[0] != "index_copy"
           for node in nodes):
        return fail("grouped cache effect contains a non-index_copy writer",
                    negative_case="unknown grouped state writer")
    old_node = writer_nodes_by_name.get(state_spec["placeholder"])
    old_fact = facts.for_node(old_node) if old_node is not None else None
    if old_fact is None or len(old_fact.shape) != len(tuple(state_spec["layout"].split(","))):
        return fail("grouped cache state has no exact layout facts",
                    negative_case="missing old-state shape")
    layout = tuple(part.strip() for part in state_spec["layout"].split(","))
    capacity_axis = _cache_axis(state_spec["layout"], len(old_fact.shape))
    if capacity_axis is None or old_fact.shape[capacity_axis] != state_spec["capacity"]:
        return fail("grouped cache state capacity differs from its declared layout",
                    negative_case="capacity mismatch")
    position = abi.position_and_valid_length
    source = position.get("position_source", "")
    prefix = "specialized_valid_length:"
    try:
        append_position = int(source.removeprefix(prefix)) if source.startswith(prefix) else None
    except ValueError:
        append_position = None
    write_ranges = {item.get("range") for item in effect_spec.get("writes", ())
                    if isinstance(item, Mapping)}
    if (append_position is None or append_position < 0 or append_position >= state_spec["capacity"] or
            position.get("old_valid_length_source") != source or
            position.get("append_position_expression") != "L" or
            position.get("new_valid_length_expression") != "L+1" or
            not any(str(value).startswith("[L,L+1)") for value in write_ranges)):
        return fail("grouped cache append has no bounded specialized position proof",
                    negative_case="index is dynamic or valid length disagrees")
    guard_capacity = abi.guard_set.get("capacity", {})
    if state_spec["placeholder"] in guard_capacity and guard_capacity[state_spec["placeholder"]] != state_spec["capacity"]:
        return fail("grouped cache capacity differs from the invocation guard",
                    negative_case="capacity guard mismatch")

    writer_info: dict[str, tuple[dict[int, int], tuple[int, ...], str, str]] = {}
    index_values: set[str] = set()
    selected_axes: tuple[int, ...] | None = None
    for node in nodes:
        if len(node.args) < 4:
            return fail("grouped cache writer lacks index_copy operands",
                        negative_case="malformed writer", writer=node.name)
        old_view, dim, index_node, _update_node = node.args[:4]
        selected = _selected_state_view(program, old_view, state_spec["placeholder"], len(old_fact.shape))
        old_view_fact = facts.for_node(old_view)
        output_fact = facts.for_node(node)
        index_fact = facts.for_node(index_node)
        update_fact = facts.for_node(node.args[3])
        if selected is None or old_view_fact is None or output_fact is None or index_fact is None or update_fact is None:
            return fail("grouped cache writer lacks a proved static state view or tensor facts",
                        negative_case="unknown view ancestry or facts", writer=node.name)
        coordinates, remaining_axes = selected
        axes = tuple(sorted(coordinates))
        if selected_axes is None:
            selected_axes = axes
        if (axes != selected_axes or axes != tuple(range(len(axes))) or
                any(axis >= capacity_axis for axis in axes) or remaining_axes != tuple(
                    axis for axis in range(len(old_fact.shape)) if axis not in axes)):
            return fail("grouped writers select different or non-prefix state partitions",
                        negative_case="state writer partitions do not form one regular grid", writer=node.name)
        if any(coordinates[axis] >= old_fact.shape[axis] for axis in axes):
            return fail("grouped state view selects outside the declared state shape",
                        negative_case="selected coordinate out of bounds", writer=node.name)
        expected_view_shape = tuple(old_fact.shape[axis] for axis in remaining_axes)
        local_capacity_axis = capacity_axis - sum(axis < capacity_axis for axis in axes)
        write_dim = dim + len(old_view_fact.shape) if isinstance(dim, int) and dim < 0 else dim
        if (old_view_fact.shape != expected_view_shape or not isinstance(write_dim, int) or
                write_dim != local_capacity_axis or output_fact.shape != expected_view_shape or
                output_fact.alias_kind != "fresh" or
                update_fact.shape != expected_view_shape[:local_capacity_axis] + (1,) +
                expected_view_shape[local_capacity_axis + 1:]):
            return fail("grouped index_copy does not write one exact cache position into a fresh value",
                        negative_case="writer axis, shape, or alias mismatch", writer=node.name)
        index_shape = tuple(index_fact.shape)
        constant_index = _constant_ints(program, index_node) if hasattr(index_node, "op") else (index_node,)
        if (index_shape != (1,) or index_fact.dtype not in {"int32", "int64"} or
                constant_index != (append_position,)):
            return fail("grouped cache writer index differs from specialized valid length",
                        negative_case="dynamic, malformed, or wrong-position index", writer=node.name)
        index_values.add(facts.node_to_value.get(index_node.name, ""))
        writer_info[node.name] = (coordinates, remaining_axes,
                                  facts.node_to_value[node.name], facts.node_to_value[index_node.name])
    if selected_axes is None or not selected_axes or len(index_values) != 1:
        return fail("grouped cache writers do not share one append index",
                    negative_case="partition set empty or writer indexes differ")
    # Keep the Cartesian construction explicit; all declared partition coordinates must occur once.
    from itertools import product
    expected_coordinates = set(product(*(range(old_fact.shape[axis]) for axis in selected_axes)))
    if {tuple(info[0][axis] for axis in selected_axes) for info in writer_info.values()} != expected_coordinates:
        return fail("grouped cache writers do not cover every state partition exactly once",
                    negative_case="writer coordinates are missing, repeated, or overlapping",
                    expected_count=len(expected_coordinates), actual_count=len(writer_info))

    output_expr = next((node.args[0] for node in reversed(tuple(program.graph_module.graph.nodes))
                        if node.op == "output"), None)
    output_leaves = _output_leaves(output_expr) if output_expr is not None else []
    public_path = tuple(output_spec["path"])
    matching_leaf = next((i for i, item in enumerate(abi.user_output_tree["leaves"])
                          if tuple(item["path"]) == public_path), None)
    if matching_leaf is None or matching_leaf >= len(output_leaves):
        return fail("StepABI state output path has no corresponding captured output leaf",
                    negative_case="aggregate state output path missing")
    direct = next((value for path, value in output_leaves if tuple(path) == public_path), None)
    aggregate_node = direct if hasattr(direct, "op") else output_leaves[matching_leaf][1]
    if not hasattr(aggregate_node, "op"):
        return fail("StepABI cache output is not a captured tensor value",
                    negative_case="aggregate state output is not a tensor")
    aggregate_fact = facts.for_node(aggregate_node)
    if aggregate_fact is None or aggregate_fact.shape != old_fact.shape or aggregate_fact.alias_kind != "fresh":
        return fail("assembled cache output does not preserve full state shape and ownership",
                    negative_case="aggregate output shape or alias mismatch")
    tree = _stack_writer_paths(aggregate_node, set(writer_names))
    if tree is None or {name for _, name in tree} != set(writer_names) or len(tree) != len(writer_names):
        return fail("returned cache does not assemble each grouped writer exactly once",
                    negative_case="aggregate stack has missing, duplicate, or unrelated leaves")
    for path, name in tree:
        coordinates = writer_info[name][0]
        if path != tuple(coordinates[axis] for axis in selected_axes):
            return fail("returned cache stack order differs from selected state coordinates",
                        negative_case="aggregate stack reorders key/value or layer writers", writer=name)

    writer_values = tuple(writer_info[name][2] for _, name in tree)
    index_value = next(iter(index_values))
    transition = StateTransition(
        effect_id=effect_id, state_id=state_id,
        old_value=facts.node_to_value[state_spec["placeholder"]],
        new_value=facts.node_to_value[aggregate_node.name], index_value=index_value,
        valid_length_before=index_value, valid_length_after="L+1",
        capacity=state_spec["capacity"], axis=capacity_axis, order=effect_spec["order"],
        write_footprint={"axis": capacity_axis, "range": "[L,L+1)", "other_axes": "full",
                         "partition_axes": list(selected_axes),
                         "partition_shape": [old_fact.shape[axis] for axis in selected_axes],
                         "writers": [{"value": writer_info[name][2], "coordinates": list(path)}
                                     for path, name in tree]},
        alias_rule="functional_grouped_new_value", writer_values=writer_values,
    )
    return transition, {name: transition for name in writer_names}, writer_names, ()


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


def _ordered_nodes(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "op") and hasattr(value, "name"):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(node for item in value.values() for node in _ordered_nodes(item))
    if isinstance(value, (tuple, list)):
        return tuple(node for item in value for node in _ordered_nodes(item))
    return ()


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
    grouped_transition, grouped_writers, grouped_candidate_names, grouped_diagnostics = \
        _grouped_cache_transition(program, facts)
    diagnostics.extend(grouped_diagnostics)
    if grouped_transition is not None:
        state_transitions.append(grouped_transition)
        transitions_by_output[grouped_transition.new_value] = grouped_transition
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
        input_nodes = (_ordered_nodes(node.args[0]) if kind == "Cat/Stack" and node.args
                       else tuple(node.all_input_nodes))
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
        tensor_construct = None
        constructor = None
        advanced_index = None
        if kind == "Cat/Stack":
            tensor_construct, maps, construct_error = _cat_stack_contract(
                node, operator, inputs, input_nodes, facts, output_fact
            )
            if construct_error:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS,
                    f"tensor construction {node.name} has no exact cat/stack map",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"reason": construct_error, "negative_case": "unproved tensor construction"},
                ))
        elif kind == "Constructor":
            constructor, construct_error = _tensor_constructor_contract(node, operator, output_fact)
            maps = _input_maps(node, operator, kind, inputs, node_to_value, facts, input_nodes, output_shape)
            if construct_error:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS,
                    f"tensor constructor {node.name} has no exact static contract",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"reason": construct_error, "negative_case": "dynamic constructor arguments"},
                ))
        elif kind == "Gather" and operator == "index":
            advanced_index, maps, construct_error = _advanced_index_contract(
                node, inputs, input_nodes, facts, output_fact
            )
            if construct_error:
                diagnostics.append(DiagnosticRecord(
                    DiagnosticCode.MISSING_FACTS,
                    f"advanced index {node.name} has no exact bounded index map",
                    DiagnosticSeverity.ERROR, node_id=node.name,
                    details={"reason": construct_error, "negative_case": "unbounded or unsupported index expression"},
                ))
        elif kind == "Copy":
            maps = _input_maps(node, operator, kind, inputs, node_to_value, facts, input_nodes, output_shape)
        else:
            maps = _input_maps(node, operator, kind, inputs, node_to_value, facts, input_nodes, output_shape)
        embedding = _embedding_contract(node, input_facts, output_fact) if kind == "Gather" and operator == "embedding" else None
        if kind == "Gather" and operator == "embedding" and embedding is None:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"embedding {node.name} has no exact weight, index, and output contract",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"negative_case": "embedding shape or integer index facts are unsupported"},
            ))
        metadata_guard = _metadata_guard_contract(node, input_nodes, facts) if kind == "Guard" else None
        if kind == "Guard" and metadata_guard is None:
            diagnostics.append(DiagnosticRecord(
                DiagnosticCode.MISSING_FACTS,
                f"metadata guard {node.name} is not implied by exact captured facts",
                DiagnosticSeverity.ERROR, node_id=node.name,
                details={"negative_case": "metadata assertion differs from its exact input fact"},
            ))
        transition = None
        index_bounds = None
        transition_diagnostic = None
        if kind == "Scatter/StateWrite" and operator == "index_copy":
            if node.name in grouped_candidate_names:
                transition = grouped_writers.get(node.name)
            else:
                transition, transition_diagnostic = _cache_transition(program, node, facts, inputs, value_id)
            if transition is not None:
                index_bounds = {"value_id": transition.index_value, "lower": 0,
                                "upper_exclusive": transition.capacity,
                                "guard": "StepABI/v1:L<capacity;captured_index==L" if
                                node.name in grouped_candidate_names else "StepABI/v1:L<capacity"}
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
        elif kind == "Gather" and operator == "index" and advanced_index is not None:
            index_bounds = {"statically_proved": True,
                            "ranges": list(advanced_index["index_ranges"]),
                            "guard": "ascending_constant_arange"}
        elif kind == "Gather" and operator == "embedding" and embedding is not None:
            index_bounds = {**dict(embedding["index_bounds"]), "value_id": inputs[1]}
        attention = None
        if kind == "Attention":
            attention = _attention_contract(
                program, facts, input_facts, output_fact, node, inputs, transitions_by_output
            )
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
            "index_select", "gather", "take", "embedding", "index", "index_copy", "index_put",
            "slice_scatter", "scatter"
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
                reads=((inputs[0] if node.name in grouped_candidate_names else transition.old_value),)
                if transition and effect.target == transition.state_id else (),
                writes=((value_id if node.name in grouped_candidate_names else transition.new_value),)
                if transition and effect.target == transition.state_id else (),
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
        if tensor_construct is not None:
            attrs["tensor_construct"] = dict(tensor_construct)
        if constructor is not None:
            attrs["constructor"] = dict(constructor)
        if advanced_index is not None:
            attrs["advanced_index"] = dict(advanced_index)
        if embedding is not None:
            attrs["embedding"] = dict(embedding)
        if index_bounds is not None:
            attrs["index_bounds"] = dict(index_bounds)
        if transition is not None:
            attrs["state_transition"] = transition.to_dict()
        if attention is not None:
            attrs["attention"] = dict(attention)
        if kind == "Guard":
            attrs["guard_kind"] = "tensor_metadata"
            if metadata_guard is not None:
                attrs["metadata_guard"] = dict(metadata_guard)
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
        if transition is not None and node.name not in grouped_candidate_names:
            state_transitions.append(transition)
            transitions_by_output[transition.new_value] = transition

    outputs: list[OutputLeaf] = []
    output_node = next((node for node in reversed(graph_nodes) if node.op == "output"), None)
    if output_node is not None:
        import json
        for path, value in canonical_output_leaves(program, output_node.args[0]):
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
