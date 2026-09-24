"""Conservative tensor/view facts and specialization guards for normalized FX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .capture import FrontendError, NormalizedProgram


_DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1, "uint8": 1,
    "int16": 2, "int32": 4, "int64": 8, "bool": 1,
}
_VIEW_NAMES = ("view", "reshape", "transpose", "permute", "slice", "select", "squeeze", "unsqueeze", "expand", "detach", "alias")


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
                 role: str, mutability: str, constraints: tuple[str, ...]) -> TensorFacts | None:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        return None
    dtype = str(value.dtype).removeprefix("torch.")
    if dtype not in _DTYPE_BYTES:
        raise FactError(f"{node.name}: unknown dtype {dtype}")
    shape = tuple(value.shape)
    strides = tuple(value.stride()) if callable(getattr(value, "stride", None)) else ()
    offset = value.storage_offset() if callable(getattr(value, "storage_offset", None)) else None
    # Alignment cannot be inferred from fake tensors or a generic FX graph.
    return TensorFacts(value_id, shape, dtype, _DTYPE_BYTES[dtype], str(value.device), strides, offset,
                       None, alias_set, role, mutability, node.name if node.op != "placeholder" else None,
                       tuple(user.name for user in node.users), (node.name,), constraints)


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
    aliases: dict[str, str | None] = {}
    guards: list[SpecializationGuard] = []
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
        target_name = str(getattr(node, "target", ""))
        view = node.op == "call_function" and any(name in target_name for name in _VIEW_NAMES)
        binding = program.lifted_bindings.get(node.name)
        role = binding.role if binding is not None else "input" if node.op == "placeholder" else "intermediate"
        if node.op == "placeholder":
            alias = binding_aliases.get(binding.target, f"binding:{binding.target}") if binding is not None else f"input:{node.name}"
        elif view and source is not None:
            source_fact = facts.get(node_to_value.get(source.name, ""))
            # A reshape of a non-contiguous source is not assumed to preserve a
            # simple affine alias; a later pass must insert/prove a copy.
            reshape = "reshape" in target_name
            source_contiguous = bool(source_fact and source_fact.strides == _contiguous_strides(source_fact.shape))
            alias = aliases.get(source.name) if not reshape or source_contiguous else None
        else:
            alias = None
        aliases[node.name] = alias
        meta = node.meta.get("val") if hasattr(node, "meta") else None
        mutability = "state" if role == "state" else "input" if role == "input" else "immutable"
        fact = _tensor_fact(meta, value_id=value_id, node=node, alias_set=alias, role=role,
                            mutability=mutability, constraints=constraints)
        if fact is None:
            continue
        facts[value_id] = fact
        if node.op == "placeholder":
            guards.append(SpecializationGuard(value_id, fact.shape, fact.strides, fact.dtype, fact.storage_offset))
    return FactTable(facts, node_to_value, tuple(guards))


def require_safe_write(fact: TensorFacts) -> None:
    if fact.has_zero_stride:
        raise FactError(f"{fact.value_id}: zero-stride expanded view cannot be written")
    if fact.alias_set is None:
        raise FactError(f"{fact.value_id}: unknown alias relationship cannot prove a safe write")


def _contiguous_strides(shape: tuple[Any, ...]) -> tuple[int, ...] | None:
    """Return contiguous strides only for concrete dimensions."""
    stride = 1
    result: list[int] = []
    for dim in reversed(shape):
        if not isinstance(dim, int):
            return None
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))
