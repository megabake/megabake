"""Target-neutral parametric tile domains for indexed operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Mapping


def _extent(value: Any) -> int | str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"axis extent must be non-negative, got {value}")
        return value
    return str(value)


@dataclass(frozen=True)
class TileAxis:
    name: str
    extent: int | str
    tile_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "extent", _extent(self.extent))
        if not isinstance(self.tile_size, int) or isinstance(self.tile_size, bool) or self.tile_size <= 0:
            raise ValueError(f"tile size for {self.name} must be a positive integer")

    @property
    def tile_count(self) -> int | str:
        if isinstance(self.extent, int):
            return (self.extent + self.tile_size - 1) // self.tile_size
        return f"ceil_div({self.extent},{self.tile_size})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "extent": self.extent, "tile_size": self.tile_size,
                "tile_count": self.tile_count}


@dataclass(frozen=True)
class TileCoordinate:
    output: tuple[int, ...]
    reduction: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, list[int]]:
        return {"output": list(self.output), "reduction": list(self.reduction)}


@dataclass(frozen=True)
class TileInstance:
    family_id: str
    domain: "TileDomain"
    coordinate: TileCoordinate

    def axis_bounds(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        for axes, coordinates in ((self.domain.output_axes, self.coordinate.output),
                                  (self.domain.reduction_axes, self.coordinate.reduction)):
            for axis, tile in zip(axes, coordinates):
                assert isinstance(axis.extent, int)
                start = tile * axis.tile_size
                result[axis.name] = (start, min(start + axis.tile_size, axis.extent))
        return result

    @property
    def task_id(self) -> str:
        values = [*(f"{axis.name}={tile}" for axis, tile in
                    zip(self.domain.output_axes, self.coordinate.output)),
                  *(f"{axis.name}={tile}" for axis, tile in
                    zip(self.domain.reduction_axes, self.coordinate.reduction))]
        return f"{self.family_id}[{','.join(values)}]"

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "coordinate": self.coordinate.to_dict(),
                "axis_bounds": {name: list(bounds) for name, bounds in self.axis_bounds().items()}}


@dataclass(frozen=True)
class TileDomain:
    output_axes: tuple[TileAxis, ...]
    reduction_axes: tuple[TileAxis, ...] = ()
    guards: tuple[str, ...] = ()

    @property
    def cardinality(self) -> int | str:
        axes = self.output_axes + self.reduction_axes
        if all(isinstance(axis.tile_count, int) for axis in axes):
            return prod(axis.tile_count for axis in axes)
        return " * ".join(str(axis.tile_count) for axis in axes) or "1"

    @property
    def has_symbolic_extent(self) -> bool:
        return any(not isinstance(axis.extent, int)
                   for axis in self.output_axes + self.reduction_axes)

    def enumerate(self, family_id: str, *, limit: int = 100_000) -> tuple[TileInstance, ...]:
        if self.has_symbolic_extent:
            raise ValueError("cannot enumerate a tile domain with symbolic extents")
        count = self.cardinality
        if not isinstance(count, int) or count > limit:
            raise ValueError(f"tile domain cardinality {count} exceeds enumeration limit {limit}")
        ranges = [range(axis.tile_count) for axis in self.output_axes + self.reduction_axes]
        coordinates = product(*ranges) if ranges else [()]
        result = []
        output_rank = len(self.output_axes)
        for values in coordinates:
            result.append(TileInstance(
                family_id, self,
                TileCoordinate(tuple(values[:output_rank]), tuple(values[output_rank:])),
            ))
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {"output_axes": [axis.to_dict() for axis in self.output_axes],
                "reduction_axes": [axis.to_dict() for axis in self.reduction_axes],
                "cardinality": self.cardinality, "guards": list(self.guards)}


def tile_domain(operation: Any, tile_sizes: Mapping[str, int] = (), *,
                include_reduction: bool = True) -> TileDomain:
    """Create output and optional reduction tile axes from an IndexedOp."""
    tile_sizes = {} if tile_sizes == () else tile_sizes
    output_names = [axis.name for axis in operation.iteration_domain]
    reduction_axes = operation.reduction_domain if include_reduction else ()
    all_axes = tuple(operation.iteration_domain) + tuple(reduction_axes)
    unknown = set(tile_sizes) - {axis.name for axis in all_axes}
    if unknown:
        raise ValueError(f"tile sizes name unknown axes: {sorted(unknown)}")

    def make_axis(axis: Any) -> TileAxis:
        extent = _extent(axis.extent)
        default = max(1, extent) if isinstance(extent, int) else 1
        return TileAxis(axis.name, extent, tile_sizes.get(axis.name, default))

    output = tuple(make_axis(axis) for axis in operation.iteration_domain)
    reduction = tuple(make_axis(axis) for axis in reduction_axes)
    guards = tuple(guard for axis in output + reduction for guard in (
        f"0 <= {axis.name} < {axis.extent}",
        f"tile_{axis.name}*{axis.tile_size} <= {axis.name} < "
        f"min((tile_{axis.name}+1)*{axis.tile_size},{axis.extent})",
    ))
    return TileDomain(output, reduction, guards)


__all__ = ["TileAxis", "TileCoordinate", "TileDomain", "TileInstance", "tile_domain"]
