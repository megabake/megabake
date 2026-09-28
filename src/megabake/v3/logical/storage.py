"""Partial-order storage lifetimes independent of physical memory spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..semantics.indexed import IndexedTensorProgram


@dataclass(frozen=True)
class StorageLifetime:
    value_id: str
    producer_family_id: str
    reader_family_ids: tuple[str, ...]
    shape: tuple[Any, ...]
    dtype: str | None
    alias_set: str | None
    alias_kind: str
    externally_owned: bool
    allocatable: bool
    recompute_declared: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "producer_family_id": self.producer_family_id,
                "reader_family_ids": list(self.reader_family_ids),
                "release_after": list(self.reader_family_ids), "shape": [str(item) for item in self.shape],
                "dtype": self.dtype, "alias_set": self.alias_set, "alias_kind": self.alias_kind,
                "externally_owned": self.externally_owned, "allocatable": self.allocatable,
                "recompute_declared": self.recompute_declared,
                "release_condition": "source_retired"}


def build_storage_lifetimes(plan: Any, program: IndexedTensorProgram) -> tuple[StorageLifetime, ...]:
    values = {item.value_id: item for item in program.values}
    output_values = {item.value_id for item in program.outputs if item.value_id is not None}
    output_values.update(item.new_value for item in program.state_transitions)
    writers = {region.value_id: family for family in plan.families for region in family.writes
               if region.value_id in values}
    result = []
    for value_id, family in writers.items():
        value = values[value_id]
        readers = tuple(dict.fromkeys(candidate.family_id for candidate in plan.families
                                       if any(access.value_id == value_id for access in candidate.reads)))
        declared = value_id in plan.recompute_values
        result.append(StorageLifetime(
            value_id, family.family_id, readers, tuple(value.shape), value.dtype,
            value.alias_set, value.alias_kind, value_id in output_values,
            value.alias_kind == "fresh", declared,
        ))
    for family in plan.families:
        if not family.partial_value_id or not family.contributes_to_family:
            continue
        finalizer = next((item for item in plan.families
                          if item.family_id == family.contributes_to_family), None)
        result.append(StorageLifetime(
            family.partial_value_id, family.family_id,
            (finalizer.family_id,) if finalizer else (), (), None, None, "fresh", False, True, False,
        ))
    return tuple(result)


def can_overlay(old: StorageLifetime, new: StorageLifetime,
                happens_before: set[tuple[str, str]]) -> bool:
    """Reuse old storage only after every reader retires before new writes begin."""
    if old.externally_owned or not old.allocatable or not new.allocatable:
        return False
    if old.alias_set is not None and old.alias_set == new.alias_set:
        return False
    release_points = old.reader_family_ids or (old.producer_family_id,)
    return all((reader, new.producer_family_id) in happens_before for reader in release_points)


__all__ = ["StorageLifetime", "build_storage_lifetimes", "can_overlay"]
