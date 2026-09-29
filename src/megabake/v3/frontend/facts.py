"""Conservative tensor/view facts and specialization guards for normalized FX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .capture import FrontendError, NormalizedProgram


_DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "complex64": 8, "complex128": 16,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "uint8": 1,
    "int16": 2, "int32": 4, "int64": 8, "bool": 1,
}
_ALIAS_VIEWS = {"view", "as_strided", "transpose", "permute", "t", "slice", "select", "squeeze", "unsqueeze", "expand", "detach", "detach_", "alias"}


class FactError(FrontendError):
    pass


@dataclass(frozen=True)
class TensorFacts:
    value_id: str
    shape: tuple[Any, ...]
    dtype: str
    dtype_bytes: int | None
    device: str
    strides: tuple[int, ...]
    storage_offset: int | None
    alignment_bytes: int | None
    alias_set: str | None
    role: str
    mutability: str
    producer: str | None
    consumers: tuple[str, ...]
    provenance: tuple[str, ...]
    symbolic_constraints: tuple[str, ...] = ()
    alias_kind: str = "unknown"
    alias_sources: tuple[str, ...] = ()
    cast_origin: str | None = None
    write_effects: tuple[str, ...] = ()
    non_overlapping: bool | None = None
    layout: str = "strided"

    @property
    def has_zero_stride(self) -> bool:
        return any(stride == 0 for stride in self.strides)


@dataclass(frozen=True)
class SpecializationGuard:
    value_id: str
    shape: tuple[Any, ...]
    strides: tuple[int, ...]
    dtype: str
    storage_offset: int | None

    def check(self, tensor: Any) -> bool:
        return (tuple(getattr(tensor, "shape", ())) == self.shape and
                tuple(getattr(tensor, "stride", lambda: ())()) == self.strides and
                str(getattr(tensor, "dtype", "")).removeprefix("torch.") == self.dtype and
                (not callable(getattr(tensor, "storage_offset", None)) or
                 tensor.storage_offset() == self.storage_offset))


@dataclass(frozen=True)
class FactTable:
    facts: Mapping[str, TensorFacts]
    node_to_value: Mapping[str, str]
    guards: tuple[SpecializationGuard, ...]
    cast_points: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    effect_nodes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def for_node(self, node_or_name: Any) -> TensorFacts | None:
        name = getattr(node_or_name, "name", node_or_name)
        value_id = self.node_to_value.get(name)
        return self.facts.get(value_id) if value_id else None

    def validate(self, values: Mapping[str, Any]) -> None:
        for guard in self.guards:
            value = values.get(guard.value_id)
            if value is not None and not guard.check(value):
                raise FactError(f"specialization guard failed for {guard.value_id}")


def _tensor_fact(value: Any, *, value_id: str, node: Any, alias_set: str | None,
                 alias_kind: str, alias_sources: tuple[str, ...], role: str,
                 mutability: str, constraints: tuple[str, ...], cast_origin: str | None,
                 write_effects: tuple[str, ...]) -> TensorFacts | None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    dtype = str(value.dtype).removeprefix("torch.")
    if dtype not in _DTYPE_BYTES:
        raise FactError(f"{node.name}: unknown dtype {dtype}")
    shape = tuple(value.shape)
    layout = str(getattr(value, "layout", "torch.strided")).removeprefix("torch.")
    stride = getattr(value, "stride", None)
    strides = (tuple(stride()) if callable(stride) else tuple(getattr(value, "stride", ()) or ())) if layout == "strided" else ()
    offset = value.storage_offset() if layout == "strided" and callable(getattr(value, "storage_offset", None)) else None
    # TensorMetadata from direct GraphModule ShapeProp has no device/storage identity.
    device = str(getattr(value, "device", "unknown"))
    return TensorFacts(value_id, shape, dtype, _DTYPE_BYTES[dtype], device, strides, offset,
                       None, alias_set, role, mutability, node.name if node.op != "placeholder" else None,
                       tuple(user.name for user in node.users), (node.name,), constraints,
                       alias_kind, alias_sources, cast_origin, write_effects,
                       _non_overlapping(shape, strides), layout)


def _first_node(value: Any) -> Any | None:
    if hasattr(value, "op") and hasattr(value, "name"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_node(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_node(item)
            if found is not None:
                return found
    return None


def collect_facts(program: NormalizedProgram) -> FactTable:
    facts: dict[str, TensorFacts] = {}
    node_to_value: dict[str, str] = {}
    aliases: dict[str, tuple[str | None, str]] = {}
    guards: list[SpecializationGuard] = []
    cast_points: dict[str, tuple[str, ...]] = {}
    effect_nodes: dict[str, list[str]] = {}
    for effect in program.effects:
        origin_ids = program.origin_map.get(effect.node_id, ())
        effect_nodes[effect.node_id] = list(origin_ids)
    # Raw data pointers are useful only for discovering ties.  They are never
    # retained: facts and JSON inventories must be reproducible across runs.
    storage_groups: dict[object, list[str]] = {}
    for target, value in program.binding_values.items():
        try:
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            storage_key = (str(value.device), pointer, storage.nbytes()) if pointer else id(value)
        except (AttributeError, RuntimeError):
            storage_key = id(value)
        storage_groups.setdefault(storage_key, []).append(target)
    binding_aliases = {
        target: f"binding:{min(targets)}"
        for targets in storage_groups.values()
        for target in targets
    }
    constraints = tuple(str(item) for item in program.constraints)
    for index, node in enumerate(program.graph_module.graph.nodes):
        value_id = program.value_ids.get(node.name, f"v{index}")
        node_to_value[node.name] = value_id
        source = _first_node(node.args)
        operator = _operator_name(node)
        binding = program.lifted_bindings.get(node.name)
        role = binding.role if binding is not None else "input" if node.op == "placeholder" else "intermediate"
        if node.op in {"placeholder", "get_attr"} and binding is not None:
            alias, alias_kind = binding_aliases.get(binding.target), "tied_binding"
            alias_sources = ()
        elif node.op == "placeholder":
            if node.name in program.state_bindings:
                alias, alias_kind = f"state:{program.state_bindings[node.name]}", "state_input"
            else:
                # Distinct user arguments may alias at runtime; a name is not an alias proof.
                alias, alias_kind = None, "unknown"
            alias_sources = ()
        elif operator in _ALIAS_VIEWS and source is not None:
            source_alias, _ = aliases.get(source.name, (None, "unknown"))
            alias, alias_kind, alias_sources = source_alias, "view", (node_to_value[source.name],)
        elif operator in {"reshape", "flatten"} and source is not None:
            source_fact = facts.get(node_to_value.get(source.name, ""))
            meta = node.meta.get("val", node.meta.get("tensor_meta")) if hasattr(node, "meta") else None
            new_shape = tuple(getattr(meta, "shape", ()))
            new_strides = (source_fact.strides if source_fact and source_fact.shape == new_shape
                           else _reshape_strides(source_fact.shape, source_fact.strides, new_shape) if source_fact else None)
            if new_strides is not None:
                alias, alias_kind = aliases.get(source.name, (None, "unknown"))[0], "view"
            elif (source_fact is not None and _concrete_shape(source_fact.shape) and _concrete_shape(new_shape)
                  and _numel(source_fact.shape) and _numel(new_shape)):
                alias, alias_kind = f"storage:{value_id}", "copy"
            else:
                alias, alias_kind = None, "unknown"
            alias_sources = (node_to_value[source.name],)
        elif _writes_input(node):
            alias, alias_kind = aliases.get(source.name, (None, "unknown"))[0] if source is not None else None, "write"
            alias_sources = (node_to_value[source.name],) if source is not None else ()
        else:
            alias, alias_kind, alias_sources = f"storage:{value_id}", "fresh", ()
        aliases[node.name] = alias, alias_kind
        meta = node.meta.get("val", node.meta.get("tensor_meta")) if hasattr(node, "meta") else None
        cast_origin = node.name if operator in {"to", "_to_copy", "convert_element_type"} else None
        node_origins = set(program.origin_map.get(node.name, ()))
        write_effects = tuple(
            effect.node_id for effect in program.effects
            if effect.node_id == node.name or node_origins.intersection(program.origin_map.get(effect.node_id, ()))
        )
        mutability = "state" if role == "state" else "input" if role == "input" else "immutable"
        fact = _tensor_fact(meta, value_id=value_id, node=node, alias_set=alias, role=role,
                            alias_kind=alias_kind, alias_sources=alias_sources,
                            mutability="mutated" if write_effects else mutability,
                            constraints=constraints, cast_origin=cast_origin, write_effects=write_effects)
        if fact is None:
            continue
        facts[value_id] = fact
        if cast_origin:
            cast_points[value_id] = (cast_origin,)
        if node.op == "placeholder":
            guards.append(SpecializationGuard(value_id, fact.shape, fact.strides, fact.dtype, fact.storage_offset))
    return FactTable(facts, node_to_value, tuple(guards), cast_points,
                     {key: tuple(value) for key, value in effect_nodes.items()})


def require_safe_write(fact: TensorFacts) -> None:
    if fact.has_zero_stride:
        raise FactError(f"{fact.value_id}: zero-stride expanded view cannot be written")
    if fact.alias_set is None:
        raise FactError(f"{fact.value_id}: unknown alias relationship cannot prove a safe write")
    if fact.non_overlapping is not True:
        raise FactError(f"{fact.value_id}: overlapping or unknown storage layout cannot prove a safe write")


def _operator_name(node: Any) -> str:
    schema = getattr(getattr(node, "target", None), "_schema", None)
    if schema is not None:
        return schema.name.split("::")[-1]
    return str(getattr(node, "target", "")).split(".")[-2:-1][0] if "." in str(getattr(node, "target", "")) else ""


def _writes_input(node: Any) -> bool:
    schema = getattr(getattr(node, "target", None), "_schema", None)
    return any(
        getattr(getattr(argument, "alias_info", None), "is_write", False)
        for argument in getattr(schema, "arguments", ())
    )


def _concrete_shape(shape: tuple[Any, ...]) -> bool:
    return all(isinstance(dim, int) and dim >= 0 for dim in shape)


def _non_overlapping(shape: tuple[Any, ...], strides: tuple[int, ...]) -> bool | None:
    if not _concrete_shape(shape) or len(shape) != len(strides):
        return None
    required = 1
    for stride, size in sorted((stride, size) for size, stride in zip(shape, strides) if size > 1):
        if stride <= 0 or stride < required:
            return False
        required = stride * size
    return True


def _reshape_strides(old_shape: tuple[Any, ...], old_strides: tuple[int, ...],
                     new_shape: tuple[Any, ...]) -> tuple[int, ...] | None:
    """Return view strides when concrete old/new shapes fit storage-contiguous chunks."""
    if not _concrete_shape(old_shape) or not _concrete_shape(new_shape) or len(old_shape) != len(old_strides):
        return None
    old_numel = _numel(old_shape)
    if old_numel != _numel(new_shape):
        return None
    if not old_shape:
        if old_numel != 1:
            return None
        stride = 1
        result = []
        for dim in reversed(new_shape):
            result.append(stride)
            stride *= dim
        return tuple(reversed(result))
    if old_numel == 0:
        return None
    chunks: list[tuple[int, int]] = []
    index = len(old_shape) - 1
    while index >= 0:
        size = old_shape[index]
        base_stride = old_strides[index]
        while index > 0 and (old_shape[index - 1] == 1 or old_strides[index - 1] == old_strides[index] * old_shape[index]):
            index -= 1
            size *= old_shape[index]
        chunks.append((size, base_stride))
        index -= 1
    chunks.reverse()
    result = [0] * len(new_shape)
    new_index = 0
    for chunk_size, base_stride in chunks:
        start = new_index
        product = 1
        while new_index < len(new_shape) and (product < chunk_size or new_shape[new_index] == 1):
            product *= new_shape[new_index]
            new_index += 1
        if product != chunk_size:
            return None
        stride = base_stride
        for axis in range(new_index - 1, start - 1, -1):
            result[axis] = stride
            stride *= new_shape[axis]
    if new_index != len(new_shape):
        return None
    return tuple(result)


def _numel(shape: tuple[Any, ...]) -> int | None:
    result = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        result *= dim
    return result
