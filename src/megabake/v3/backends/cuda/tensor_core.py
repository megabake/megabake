"""A small, target-qualified output-channel-major WMMA schedule family."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .simt import ContractionShape


TARGET = "sm_90a"
_TILE = 16


@dataclass(frozen=True)
class TensorCoreSchedule:
    warps_per_cta: int
    mainloop_depth: int = 1

    def __post_init__(self) -> None:
        if self.warps_per_cta not in (1, 2, 4):
            raise ValueError("warps_per_cta must be one of 1, 2, or 4")
        if self.mainloop_depth not in (1, 2, 4):
            raise ValueError("mainloop_depth must be one of 1, 2, or 4")

    @property
    def vector_width(self) -> int:
        return 1

    @property
    def threads_per_cta(self) -> int:
        return 32 * self.warps_per_cta

    @property
    def output_channels_per_cta(self) -> int:
        return _TILE * self.warps_per_cta


def enumerate_tensor_core_schedules(
    shape: ContractionShape, *, target: str = TARGET,
) -> tuple[TensorCoreSchedule, ...]:
    """Keep this first pass exact-target and limited to one padded batch tile."""
    shape.validate()
    if target != TARGET or shape.m > _TILE:
        return ()
    return tuple(
        TensorCoreSchedule(warps, depth)
        for warps in (1, 2, 4)
        for depth in (1, 2, 4)
    )


def output_tiles(shape: ContractionShape) -> int:
    shape.validate()
    return ceil(shape.n / _TILE)


def worker_groups(shape: ContractionShape, schedule: TensorCoreSchedule) -> int:
    return ceil(output_tiles(shape) / schedule.warps_per_cta)


def owner_grid_candidates(
    shape: ContractionShape, schedule: TensorCoreSchedule,
    visible_sms: int, resident_ctas: int,
) -> tuple[int, ...]:
    if visible_sms <= 0 or resident_ctas <= 0:
        raise ValueError("visible_sms and resident_ctas must be positive")
    groups = worker_groups(shape, schedule)
    return tuple(dict.fromkeys((
        min(visible_sms, groups, resident_ctas),
        min(4 * visible_sms, groups, resident_ctas),
        min(resident_ctas, groups),
    )))


def work_estimate(shape: ContractionShape, schedule: TensorCoreSchedule) -> dict[str, int | float]:
    """Count padded MMA work, including inactive warps in the final CTA group."""
    tiles = output_tiles(shape)
    mma_tiles = worker_groups(shape, schedule) * schedule.warps_per_cta
    padded_k = ceil(shape.k / _TILE) * _TILE
    issued_macs = mma_tiles * _TILE * _TILE * padded_k
    useful_macs = shape.m * shape.n * shape.k
    return {
        "output_tiles": tiles,
        "worker_groups": worker_groups(shape, schedule),
        "mma_tiles_including_inactive_warps": mma_tiles,
        "padded_batch": _TILE,
        "padded_channels": mma_tiles * _TILE,
        "padded_k": padded_k,
        "k_tile_per_mainloop": _TILE * schedule.mainloop_depth,
        "mainloop_iterations": ceil(shape.k / (_TILE * schedule.mainloop_depth)),
        "useful_macs": useful_macs,
        "issued_macs": issued_macs,
        "padded_work_ratio": issued_macs / useful_macs,
    }
