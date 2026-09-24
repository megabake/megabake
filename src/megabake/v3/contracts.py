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
    "TimedUnit",
    "ToleranceSpec",
    "WorkloadSpec",
    "canonical_json",
]
