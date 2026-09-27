"""Immutable, target-neutral V3 workload and numerical contracts.

These records are intentionally small.  They are the contract that travels with a
future normalized graph, execution plan, or benchmark cell; they do not contain
CUDA objects, tensors, callables, or device queries.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CONTRACT_SCHEMA_VERSION = 1


class ContractError(ValueError):
    """Raised when a workload or numerical contract is incomplete or invalid."""


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_value(value: Any) -> Any:
    """Convert supported values to deterministic JSON data.

    Rejecting arbitrary objects here is important: contract hashes must never
    depend on repr(), object addresses, or executable Python state.
    """

    value = _enum_value(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("contract mapping keys must be strings")
            result[key] = _json_value(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("contract JSON cannot contain non-finite floats")
        return value
    raise ContractError(
        f"unsupported contract value {type(value).__name__}; use JSON-safe data"
    )


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for contract hashes."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _contract_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _dtype_name(value: Any, field_name: str = "dtype") -> str:
    # Accept torch.dtype-like values without importing torch in the contract
    # module.  This keeps CPU planning/import independent from CUDA packages.
    value = str(value)
    if value.startswith("torch."):
        value = value[6:]
    return _nonempty(value, field_name)


def _mapping_proxy(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be a mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ContractError(f"{field_name} keys must be non-empty strings")
        result[key.strip()] = item
    return MappingProxyType({key: result[key] for key in sorted(result)})


def _contains_pointer_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower().replace("_", "")
            if "pointer" in name or name.endswith("ptr") or name in {"dataptr", "address", "deviceaddress"}:
                return True
            if _contains_pointer_key(item):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_pointer_key(item) for item in value)
    return False


def _freeze_json(value: Any, field_name: str) -> Any:
    """Validate JSON data and freeze nested mappings/sequences for contracts."""
    canonical_json(value)
    if _contains_pointer_key(value):
        raise ContractError(f"{field_name} cannot contain raw tensor pointers or addresses")
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item, field_name) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, field_name) for item in value)
    return value


def _record_tuple(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (tuple, list)):
        raise ContractError(f"{field_name} must be a sequence of JSON objects")
    records = tuple(_freeze_json(item, field_name) for item in value)
    if any(not isinstance(item, Mapping) for item in records):
        raise ContractError(f"{field_name} must contain only JSON objects")
    return records


def _path(value: Any, field_name: str) -> tuple[str | int, ...]:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, (str, int)) or isinstance(item, bool) for item in value
    ):
        raise ContractError(f"{field_name} must be a sequence of string/integer path components")
    return tuple(value)


class InputOrigin(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class OutputOwnership(str, Enum):
    OWNED = "owned"
    BORROWED = "borrowed"
    CALLER_OWNED = "caller_owned"


class TimedUnit(str, Enum):
    FORWARD = "forward"
    CACHED_STEP = "cached_step"
    GENERATION = "generation"


def _canonical_enum(value: Any, enum_type: type[Enum], field_name: str) -> str:
    raw = _enum_value(value)
    if not isinstance(raw, str):
        raise ContractError(f"{field_name} must be a string or {enum_type.__name__}")
    aliases = {
        "device": "gpu",
        "host": "cpu",
        "cached_decode_step": "cached_step",
        "full_generation": "generation",
        "caller-provided": "caller_owned",
        "caller_provided": "caller_owned",
    }
    raw = aliases.get(raw, raw)
    try:
        return enum_type(raw).value
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ContractError(
            f"{field_name}={raw!r} is invalid; expected one of {choices}"
        ) from exc


@dataclass(frozen=True)
class BenchmarkCell:
    """A pre-registered comparison cell and its acceptance requirement."""

    cell_id: str
    baseline: str
    must_win: bool
    context_length: int | None = None
    notes: str = ""
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _nonempty(self.cell_id, "cell_id"))
        object.__setattr__(self, "baseline", _nonempty(self.baseline, "baseline"))
        if not isinstance(self.must_win, bool):
            raise ContractError("must_win must be a boolean")
        if self.context_length is not None and (
            not isinstance(self.context_length, int) or self.context_length < 0
        ):
            raise ContractError("benchmark context_length must be a non-negative integer")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported BenchmarkCell schema_version={self.schema_version}; "
                f"expected {CONTRACT_SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cell_id": self.cell_id,
            "baseline": self.baseline,
            "must_win": self.must_win,
            "context_length": self.context_length,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkCell":
        _check_schema(value, cls.__name__)
        return cls(**{key: value[key] for key in (
            "cell_id", "baseline", "must_win", "context_length", "notes",
        )}, schema_version=value["schema_version"])


@dataclass(frozen=True)
class ExceptionalValuePolicy:
    """Explicit handling for NaN and signed infinity in comparisons."""

    allow_nan: bool
    allow_pos_inf: bool
    allow_neg_inf: bool
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("allow_nan", "allow_pos_inf", "allow_neg_inf"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"{name} must be a boolean")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported ExceptionalValuePolicy schema_version={self.schema_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "allow_nan": self.allow_nan,
            "allow_pos_inf": self.allow_pos_inf,
            "allow_neg_inf": self.allow_neg_inf,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExceptionalValuePolicy":
        _check_schema(value, cls.__name__)
        return cls(
            allow_nan=value["allow_nan"],
            allow_pos_inf=value["allow_pos_inf"],
            allow_neg_inf=value["allow_neg_inf"],
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class ToleranceSpec:
    """Reference-based tolerance for one operation/dtype comparison."""

    atol: float
    rtol: float
    equal_nan: bool = False
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.atol, (int, float)) or not math.isfinite(self.atol) or self.atol < 0:
            raise ContractError("tolerance atol must be a finite non-negative number")
        if not isinstance(self.rtol, (int, float)) or not math.isfinite(self.rtol) or self.rtol < 0:
            raise ContractError("tolerance rtol must be a finite non-negative number")
        if not isinstance(self.equal_nan, bool):
            raise ContractError("tolerance equal_nan must be a boolean")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported ToleranceSpec schema_version={self.schema_version}"
            )
        object.__setattr__(self, "atol", float(self.atol))
        object.__setattr__(self, "rtol", float(self.rtol))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atol": self.atol,
            "rtol": self.rtol,
            "equal_nan": self.equal_nan,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToleranceSpec":
        _check_schema(value, cls.__name__)
        return cls(
            atol=value["atol"],
            rtol=value["rtol"],
            equal_nan=value.get("equal_nan", False),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class WorkloadSpec:
    """Immutable invocation and benchmark semantics for one V3 workload."""

    batch_size: int
    context_length: int
    capacity: int
    dtype: str
    state_semantics: str
    mask_semantics: str
    cache_layout: str
    input_origin: InputOrigin | str
    output_ownership: OutputOwnership | str
    timed_unit: TimedUnit | str
    checkpoint_id: str | None = None
    config_id: str | None = None
    position: int | None = None
    shape_buckets: tuple[str, ...] = ()
    dropout: bool = False
    benchmark_cells: tuple[BenchmarkCell, ...] = ()
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ContractError("batch_size must be a positive integer")
        if not isinstance(self.context_length, int) or self.context_length < 0:
            raise ContractError("context_length must be a non-negative integer")
        if not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ContractError("capacity must be a positive integer")
        if self.context_length > self.capacity:
            raise ContractError("context_length cannot exceed capacity")
        if self.position is not None and (
            not isinstance(self.position, int) or not 0 <= self.position < self.capacity
        ):
            raise ContractError("position must be within [0, capacity)")
        object.__setattr__(self, "dtype", _dtype_name(self.dtype))
        for name in ("state_semantics", "mask_semantics", "cache_layout"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self, "input_origin", _canonical_enum(self.input_origin, InputOrigin, "input_origin")
        )
        object.__setattr__(
            self,
            "output_ownership",
            _canonical_enum(self.output_ownership, OutputOwnership, "output_ownership"),
        )
        object.__setattr__(
            self, "timed_unit", _canonical_enum(self.timed_unit, TimedUnit, "timed_unit")
        )
        for name in ("checkpoint_id", "config_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        if not isinstance(self.dropout, bool):
            raise ContractError("dropout must be a boolean")
        buckets = tuple(_nonempty(item, "shape_buckets item") for item in self.shape_buckets)
        object.__setattr__(self, "shape_buckets", buckets)
        cells = tuple(
            item if isinstance(item, BenchmarkCell) else BenchmarkCell(**item)
            for item in self.benchmark_cells
        )
        if len({cell.cell_id for cell in cells}) != len(cells):
            raise ContractError("benchmark cell IDs must be unique")
        object.__setattr__(self, "benchmark_cells", cells)
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported WorkloadSpec schema_version={self.schema_version}; "
                f"expected {CONTRACT_SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "capacity": self.capacity,
            "dtype": self.dtype,
            "state_semantics": self.state_semantics,
            "mask_semantics": self.mask_semantics,
            "cache_layout": self.cache_layout,
            "input_origin": self.input_origin,
            "output_ownership": self.output_ownership,
            "timed_unit": self.timed_unit,
            "checkpoint_id": self.checkpoint_id,
            "config_id": self.config_id,
            "position": self.position,
            "shape_buckets": list(self.shape_buckets),
            "dropout": self.dropout,
            "benchmark_cells": [cell.to_dict() for cell in self.benchmark_cells],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkloadSpec":
        _check_schema(value, cls.__name__)
        fields_to_load = (
            "batch_size", "context_length", "capacity", "dtype", "state_semantics",
            "mask_semantics", "cache_layout", "input_origin", "output_ownership",
            "timed_unit", "checkpoint_id", "config_id", "position", "shape_buckets",
            "dropout",
        )
        cells = tuple(BenchmarkCell.from_dict(item) for item in value.get("benchmark_cells", ()))
        return cls(
            **{key: value[key] for key in fields_to_load},
            benchmark_cells=cells,
            schema_version=value["schema_version"],
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def contract_hash(self) -> str:
        return _contract_hash(self.to_dict())


@dataclass(frozen=True)
class NumericalPolicy:
    """Immutable numerical semantics used by reference and implementation paths."""

    reference_expansion: str
    intermediate_casts: tuple[str, ...]
    accumulation_dtypes: Mapping[str, str]
    output_casts: Mapping[str, str]
    permitted_reassociation: Mapping[str, bool]
    tolerances: Mapping[str, Mapping[str, ToleranceSpec | Mapping[str, Any]]]
    exceptional_value_policy: ExceptionalValuePolicy
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_expansion", _nonempty(self.reference_expansion, "reference_expansion")
        )
        casts = tuple(_nonempty(item, "intermediate_casts item") for item in self.intermediate_casts)
        object.__setattr__(self, "intermediate_casts", casts)
        accum = {
            key: _dtype_name(item, f"accumulation_dtypes[{key!r}]")
            for key, item in dict(_mapping_proxy(self.accumulation_dtypes, "accumulation_dtypes")).items()
        }
        outputs = {
            key: _dtype_name(item, f"output_casts[{key!r}]")
            for key, item in dict(_mapping_proxy(self.output_casts, "output_casts")).items()
        }
        reassociation = dict(_mapping_proxy(self.permitted_reassociation, "permitted_reassociation"))
        if any(not isinstance(item, bool) for item in reassociation.values()):
            raise ContractError("permitted_reassociation values must be booleans")
        object.__setattr__(self, "accumulation_dtypes", MappingProxyType(dict(sorted(accum.items()))))
        object.__setattr__(self, "output_casts", MappingProxyType(dict(sorted(outputs.items()))))
        object.__setattr__(
            self,
            "permitted_reassociation",
            MappingProxyType(dict(sorted(reassociation.items()))),
        )
        tolerance_table = {}
        for operation, dtype_table in dict(_mapping_proxy(self.tolerances, "tolerances")).items():
            if not isinstance(dtype_table, Mapping):
                raise ContractError(f"tolerances[{operation!r}] must be a mapping")
            normalized = {}
            for dtype, item in dtype_table.items():
                if isinstance(item, ToleranceSpec):
                    tolerance = item
                elif isinstance(item, Mapping):
                    tolerance = ToleranceSpec.from_dict(
                        {"schema_version": CONTRACT_SCHEMA_VERSION, **dict(item)}
                    )
                elif isinstance(item, Sequence) and len(item) == 2:
                    tolerance = ToleranceSpec(atol=item[0], rtol=item[1])
                else:
                    raise ContractError(
                        f"tolerances[{operation!r}][{dtype!r}] must be ToleranceSpec or {{atol, rtol}}"
                    )
                normalized[_dtype_name(dtype, "tolerance dtype")] = tolerance
            tolerance_table[operation] = MappingProxyType(dict(sorted(normalized.items())))
        object.__setattr__(self, "tolerances", MappingProxyType(dict(sorted(tolerance_table.items()))))
        if isinstance(self.exceptional_value_policy, Mapping):
            policy = ExceptionalValuePolicy.from_dict(
                {"schema_version": CONTRACT_SCHEMA_VERSION, **dict(self.exceptional_value_policy)}
            )
        elif isinstance(self.exceptional_value_policy, ExceptionalValuePolicy):
            policy = self.exceptional_value_policy
        else:
            raise ContractError("exceptional_value_policy must be a policy record")
        object.__setattr__(self, "exceptional_value_policy", policy)
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(
                f"unsupported NumericalPolicy schema_version={self.schema_version}; "
                f"expected {CONTRACT_SCHEMA_VERSION}"
            )

    def tolerance_for(self, operation: str, dtype: Any) -> ToleranceSpec:
        operation = _nonempty(operation, "operation")
        dtype = _dtype_name(dtype)
        table = self.tolerances.get(operation) or self.tolerances.get("*")
        if table is None:
            raise ContractError(f"no tolerance registered for operation {operation!r}")
        try:
            return table[dtype]
        except KeyError as exc:
            try:
                return table["*"]
            except KeyError:
                raise ContractError(
                    f"no tolerance registered for operation={operation!r}, dtype={dtype!r}"
                ) from exc

    def reassociation_allowed(self, operation: str) -> bool:
        return bool(self.permitted_reassociation.get(operation, self.permitted_reassociation.get("*", False)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_expansion": self.reference_expansion,
            "intermediate_casts": list(self.intermediate_casts),
            "accumulation_dtypes": dict(self.accumulation_dtypes),
            "output_casts": dict(self.output_casts),
            "permitted_reassociation": dict(self.permitted_reassociation),
            "tolerances": {
                operation: {
                    dtype: tolerance.to_dict()
                    for dtype, tolerance in dtype_table.items()
                }
                for operation, dtype_table in self.tolerances.items()
            },
            "exceptional_value_policy": self.exceptional_value_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NumericalPolicy":
        _check_schema(value, cls.__name__)
        return cls(
            reference_expansion=value["reference_expansion"],
            intermediate_casts=tuple(value["intermediate_casts"]),
            accumulation_dtypes=value["accumulation_dtypes"],
            output_casts=value["output_casts"],
            permitted_reassociation=value["permitted_reassociation"],
            tolerances=value["tolerances"],
            exceptional_value_policy=ExceptionalValuePolicy.from_dict(
                value["exceptional_value_policy"]
            ),
            schema_version=value["schema_version"],
        )

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def contract_hash(self) -> str:
        return _contract_hash(self.to_dict())


@dataclass(frozen=True)
class StepABI:
    """Versioned caller/state/output ABI for one complete cached decode step."""

    ordered_user_inputs: tuple[Mapping[str, Any], ...]
    lifted_bindings: Mapping[str, Mapping[str, Any]]
    old_state_inputs: tuple[Mapping[str, Any], ...]
    state_effects: tuple[Mapping[str, Any], ...]
    new_state_outputs: tuple[Mapping[str, Any], ...]
    user_output_tree: Mapping[str, Any]
    position_and_valid_length: Mapping[str, Any]
    batch_rule: str
    invocation_preparation: Mapping[str, Any]
    guard_set: Mapping[str, Any]
    state_mode: str
    cache_update_mode: str
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        input_records = _record_tuple(self.ordered_user_inputs, "ordered_user_inputs")
        state_records = _record_tuple(self.old_state_inputs, "old_state_inputs")
        effect_records = _record_tuple(self.state_effects, "state_effects")
        output_records = _record_tuple(self.new_state_outputs, "new_state_outputs")
        if not input_records:
            raise ContractError("StepABI requires ordered user inputs")
        input_ids = []
        for item in input_records:
            input_ids.append(_nonempty(item.get("placeholder"), "input placeholder"))
            _path(item.get("path"), "input path")
        if len(set(input_ids)) != len(input_ids):
            raise ContractError("StepABI user input placeholders must be unique")
        object.__setattr__(self, "ordered_user_inputs", input_records)
        state_mode = _nonempty(self.state_mode, "state_mode")
        if state_mode not in {"advancing", "fixed_replay"}:
            raise ContractError("state_mode must be 'advancing' or 'fixed_replay'")
        object.__setattr__(self, "state_mode", state_mode)
        if self.cache_update_mode != "functional_append":
            raise ContractError("StepABI/v1 requires functional_append cache updates")

        bindings = _freeze_json(self.lifted_bindings, "lifted_bindings")
        if not isinstance(bindings, Mapping):
            raise ContractError("lifted_bindings must be a mapping")
        for placeholder, binding in bindings.items():
            _nonempty(placeholder, "lifted binding placeholder")
            if not isinstance(binding, Mapping):
                raise ContractError(f"lifted_bindings[{placeholder!r}] must be an object")
            _nonempty(binding.get("identity"), f"lifted_bindings[{placeholder!r}].identity")
            if binding.get("role") not in {"weight", "parameter", "buffer", "constant"}:
                raise ContractError(f"lifted_bindings[{placeholder!r}].role is invalid")
            _nonempty(binding.get("lifetime"), f"lifted_bindings[{placeholder!r}].lifetime")
        object.__setattr__(self, "lifted_bindings", bindings)

        if not state_records:
            raise ContractError("cached-step ABI requires old state inputs")
        state_ids: set[str] = set()
        for item in state_records:
            placeholder = _nonempty(item.get("placeholder"), "state placeholder")
            if placeholder not in input_ids:
                raise ContractError(f"state placeholder {placeholder!r} is not an ordered user input")
            state_id = _nonempty(item.get("state_id"), "state_id")
            if state_id in state_ids:
                raise ContractError(f"duplicate old state identity {state_id!r}")
            state_ids.add(state_id)
            _path(item.get("path"), "state input path")
            _nonempty(item.get("layout"), "state layout")
            _nonempty(item.get("alias_set"), "state alias_set")
            capacity = item.get("capacity")
            if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
                raise ContractError("state capacity must be a positive integer")
        object.__setattr__(self, "old_state_inputs", state_records)

        if not effect_records:
            raise ContractError("cached-step ABI requires explicit state effects")
        effect_ids: set[str] = set()
        orders: set[int] = set()
        for item in effect_records:
            effect_id = _nonempty(item.get("effect_id"), "effect_id")
            if effect_id in effect_ids:
                raise ContractError(f"duplicate state effect {effect_id!r}")
            effect_ids.add(effect_id)
            if item.get("state_id") not in state_ids:
                raise ContractError(f"state effect {effect_id!r} refers to unknown state")
            order = item.get("order")
            if not isinstance(order, int) or isinstance(order, bool) or order < 0 or order in orders:
                raise ContractError("state effect order must be a unique non-negative integer")
            orders.add(order)
            for field_name in ("reads", "writes"):
                if not isinstance(item.get(field_name), (tuple, list, Mapping)) or not item[field_name]:
                    raise ContractError(f"state effect {effect_id!r} needs explicit {field_name} regions")
        object.__setattr__(self, "state_effects", effect_records)

        if not output_records:
            raise ContractError("cached-step ABI requires new state outputs")
        output_state_ids: set[str] = set()
        for item in output_records:
            state_id = _nonempty(item.get("state_id"), "new state output state_id")
            if state_id not in state_ids:
                raise ContractError(f"new state output refers to unknown state {state_id!r}")
            output_state_ids.add(state_id)
            _path(item.get("path"), "new state output path")
            if not item.get("source_id") and not item.get("alias_of"):
                raise ContractError("new state output requires source_id or alias_of")
        if output_state_ids != state_ids:
            raise ContractError("every old state input must have a declared new state output")
        object.__setattr__(self, "new_state_outputs", output_records)

        output_tree = _freeze_json(self.user_output_tree, "user_output_tree")
        if not isinstance(output_tree, Mapping) or "structure" not in output_tree:
            raise ContractError("user_output_tree requires a structure")
        leaves = output_tree.get("leaves")
        if not isinstance(leaves, (tuple, list)) or not leaves:
            raise ContractError("user_output_tree requires typed leaves")
        output_paths: set[tuple[str | int, ...]] = set()
        for leaf in leaves:
            if not isinstance(leaf, Mapping):
                raise ContractError("user_output_tree leaves must be objects")
            path = _path(leaf.get("path"), "output leaf path")
            if path in output_paths:
                raise ContractError(f"duplicate output leaf path {path!r}")
            output_paths.add(path)
            if leaf.get("ownership") not in {item.value for item in OutputOwnership}:
                raise ContractError(f"output ownership at {path!r} is invalid")
            _nonempty(leaf.get("lifetime"), f"output lifetime at {path!r}")
        if any(_path(item["path"], "new state output path") not in output_paths for item in output_records):
            raise ContractError("new state outputs must be leaves of user_output_tree")
        object.__setattr__(self, "user_output_tree", output_tree)

        position = _freeze_json(self.position_and_valid_length, "position_and_valid_length")
        if not isinstance(position, Mapping):
            raise ContractError("position_and_valid_length must be an object")
        for name in ("position_source", "old_valid_length_source", "attend_range"):
            _nonempty(position.get(name), f"position_and_valid_length.{name}")
        if position.get("append_position_expression") != "L":
            raise ContractError("cached-step append position must be the old valid length L")
        if position.get("new_valid_length_expression") != "L+1":
            raise ContractError("cached-step new valid length must be L+1")
        object.__setattr__(self, "position_and_valid_length", position)

        batch_rule = _nonempty(self.batch_rule, "batch_rule")
        if batch_rule != "uniform_valid_length":
            raise ContractError("the first cached-step ABI requires uniform_valid_length")
        object.__setattr__(self, "batch_rule", batch_rule)

        preparation = _freeze_json(self.invocation_preparation, "invocation_preparation")
        if not isinstance(preparation, Mapping):
            raise ContractError("invocation_preparation must be an object")
        _nonempty(preparation.get("setup_amortization"), "setup_amortization")
        actions = preparation.get("actions")
        if not isinstance(actions, (tuple, list)) or any(not isinstance(action, Mapping) for action in actions):
            raise ContractError("invocation_preparation.actions must be a sequence of objects")
        if state_mode == "fixed_replay" and not preparation.get("fixed_state_reset"):
            raise ContractError("fixed_replay requires an explicit fixed_state_reset policy")
        object.__setattr__(self, "invocation_preparation", preparation)

        guards = _freeze_json(self.guard_set, "guard_set")
        if not isinstance(guards, Mapping):
            raise ContractError("guard_set must be an object")
        required_guards = {"shapes", "strides", "dtypes", "capacity", "features", "numerical_policy_hash"}
        if not required_guards.issubset(guards):
            raise ContractError(f"guard_set is missing {sorted(required_guards.difference(guards))}")
        for key in required_guards:
            if guards[key] in (None, "", (), [], {}):
                raise ContractError(f"guard_set.{key} cannot be empty")
        object.__setattr__(self, "guard_set", guards)

        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported StepABI schema_version={self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return _json_value({
            "schema_version": self.schema_version,
            "ordered_user_inputs": self.ordered_user_inputs,
            "lifted_bindings": self.lifted_bindings,
            "old_state_inputs": self.old_state_inputs,
            "state_effects": self.state_effects,
            "new_state_outputs": self.new_state_outputs,
            "user_output_tree": self.user_output_tree,
            "position_and_valid_length": self.position_and_valid_length,
            "batch_rule": self.batch_rule,
            "invocation_preparation": self.invocation_preparation,
            "guard_set": self.guard_set,
            "state_mode": self.state_mode,
            "cache_update_mode": self.cache_update_mode,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepABI":
        _check_schema(value, cls.__name__)
        fields_to_load = (
            "ordered_user_inputs", "lifted_bindings", "old_state_inputs", "state_effects",
            "new_state_outputs", "user_output_tree", "position_and_valid_length", "batch_rule",
            "invocation_preparation", "guard_set", "state_mode", "cache_update_mode",
        )
        return cls(**{key: value[key] for key in fields_to_load}, schema_version=value["schema_version"])

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def contract_hash(self) -> str:
        return _contract_hash(self.to_dict())


@dataclass(frozen=True)
class StepManifest:
    """Reproducible, hashed workload + numerical + state ABI manifest."""

    workload: WorkloadSpec
    numerical_policy: NumericalPolicy
    step_abi: StepABI
    checkpoint_revision: str
    graph_hash: str
    versions: Mapping[str, str]
    fixed_inputs: Mapping[str, Any]
    cell_status: Mapping[str, str]
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.workload, WorkloadSpec):
            object.__setattr__(self, "workload", WorkloadSpec.from_dict(self.workload))
        if not isinstance(self.numerical_policy, NumericalPolicy):
            object.__setattr__(self, "numerical_policy", NumericalPolicy.from_dict(self.numerical_policy))
        if not isinstance(self.step_abi, StepABI):
            object.__setattr__(self, "step_abi", StepABI.from_dict(self.step_abi))
        if self.workload.timed_unit != TimedUnit.CACHED_STEP.value:
            raise ContractError("a StepManifest must use timed_unit=cached_step")
        if not self.workload.benchmark_cells:
            raise ContractError("a StepManifest must declare at least one benchmark cell")
        for state in self.step_abi.old_state_inputs:
            if state["capacity"] != self.workload.capacity:
                raise ContractError("WorkloadSpec and StepABI cache capacities differ")
            if state["layout"] != self.workload.cache_layout:
                raise ContractError("WorkloadSpec and StepABI cache layouts differ")
        object.__setattr__(self, "checkpoint_revision", _nonempty(self.checkpoint_revision, "checkpoint_revision"))
        graph_hash = _nonempty(self.graph_hash, "graph_hash")
        if len(graph_hash) != 64 or any(char not in "0123456789abcdef" for char in graph_hash.lower()):
            raise ContractError("graph_hash must be a 64-character SHA-256 hex digest")
        object.__setattr__(self, "graph_hash", graph_hash.lower())

        versions = _freeze_json(self.versions, "versions")
        if not isinstance(versions, Mapping):
            raise ContractError("versions must be a mapping")
        required_versions = {"python", "pytorch", "cuda_runtime", "cuda_toolkit", "transformers"}
        if not required_versions.issubset(versions):
            raise ContractError(f"versions is missing {sorted(required_versions.difference(versions))}")
        for name in required_versions:
            _nonempty(versions[name], f"versions.{name}")
        object.__setattr__(self, "versions", versions)

        fixed_inputs = _freeze_json(self.fixed_inputs, "fixed_inputs")
        if not isinstance(fixed_inputs, Mapping) or not fixed_inputs:
            raise ContractError("fixed_inputs must describe seeded input/state values")
        object.__setattr__(self, "fixed_inputs", fixed_inputs)
        if self.step_abi.guard_set["numerical_policy_hash"] != self.numerical_policy.contract_hash:
            raise ContractError("StepABI numerical guard does not match NumericalPolicy")

        statuses = _freeze_json(self.cell_status, "cell_status")
        if not isinstance(statuses, Mapping):
            raise ContractError("cell_status must be a mapping")
        declared = {cell.cell_id for cell in self.workload.benchmark_cells}
        if set(statuses) != declared:
            raise ContractError("cell_status must report every declared benchmark cell exactly once")
        allowed_statuses = {"not_measured", "measured", "unsupported", "unavailable", "incorrect", "strict_win", "strict_loss", "inconclusive"}
        if any(value not in allowed_statuses for value in statuses.values()):
            raise ContractError("cell_status contains an unknown result status")
        object.__setattr__(self, "cell_status", statuses)
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ContractError(f"unsupported StepManifest schema_version={self.schema_version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workload": self.workload.to_dict(),
            "numerical_policy": self.numerical_policy.to_dict(),
            "step_abi": self.step_abi.to_dict(),
            "checkpoint_revision": self.checkpoint_revision,
            "graph_hash": self.graph_hash,
            "versions": _json_value(self.versions),
            "fixed_inputs": _json_value(self.fixed_inputs),
            "cell_status": _json_value(self.cell_status),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StepManifest":
        _check_schema(value, cls.__name__)
        return cls(
            workload=WorkloadSpec.from_dict(value["workload"]),
            numerical_policy=NumericalPolicy.from_dict(value["numerical_policy"]),
            step_abi=StepABI.from_dict(value["step_abi"]),
            checkpoint_revision=value["checkpoint_revision"],
            graph_hash=value["graph_hash"],
            versions=value["versions"],
            fixed_inputs=value["fixed_inputs"],
            cell_status=value["cell_status"],
            schema_version=value["schema_version"],
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        value = self.to_dict()
        if indent is None:
            return canonical_json(value)
        return json.dumps(value, sort_keys=True, indent=indent, ensure_ascii=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> "StepManifest":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid step manifest JSON: {exc}") from exc
        try:
            return cls.from_dict(value)
        except (KeyError, TypeError, ContractError) as exc:
            raise ContractError(f"invalid step manifest: {exc}") from exc

    @property
    def contract_hash(self) -> str:
        return _contract_hash(self.to_dict())

    @property
    def not_measured_cells(self) -> tuple[str, ...]:
        return tuple(cell_id for cell_id, status in self.cell_status.items() if status == "not_measured")


def _check_schema(value: Mapping[str, Any], record_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{record_name} requires a JSON object")
    version = value.get("schema_version")
    if version != CONTRACT_SCHEMA_VERSION:
        raise ContractError(
            f"unsupported {record_name} schema_version={version!r}; "
            f"expected {CONTRACT_SCHEMA_VERSION}"
        )


__all__ = [
    "BenchmarkCell",
    "CONTRACT_SCHEMA_VERSION",
    "ContractError",
    "ExceptionalValuePolicy",
    "InputOrigin",
    "NumericalPolicy",
    "OutputOwnership",
    "StepABI",
    "StepManifest",
    "TimedUnit",
    "ToleranceSpec",
    "WorkloadSpec",
    "canonical_json",
]
