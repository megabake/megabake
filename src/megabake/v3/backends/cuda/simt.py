"""Bounded CUDA schedules for the V3 low-batch contraction probe.

These descriptors are provisional benchmark inputs from V3R-004 inventories.
They stay in the CUDA backend and do not become common FX semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping


@dataclass(frozen=True)
class ContractionShape:
    m: int
    n: int
    k: int
    x_m_stride: int
    x_k_stride: int
    w_n_stride: int
    w_k_stride: int
    input_dtype: str
    output_dtype: str
    accumulation_dtype: str
    call_count: int = 1

    @property
    def output_elements(self) -> int:
        return self.m * self.n

    @classmethod
    def from_inventory(cls, record: Mapping[str, Any]) -> "ContractionShape":
        gemm = record["gemm"]
        x_strides, rhs_strides = gemm["effective_strides"]
        dtypes = record["dtypes"]
        # Inventory strides describe X[M,K] @ rhs[K,N]. The body indexes the
        # source weight as logical W[N,K], so its two strides are transposed.
        return cls(
            m=int(gemm["M"]), n=int(gemm["N"]), k=int(gemm["K"]),
            x_m_stride=int(x_strides[0]), x_k_stride=int(x_strides[1]),
            w_n_stride=int(rhs_strides[1]), w_k_stride=int(rhs_strides[0]),
            input_dtype=str(dtypes["inputs"][0]),
            output_dtype=str(dtypes["outputs"][0]),
            accumulation_dtype=str(record["accumulation_dtype"]),
            call_count=int(record["call_count"]),
        )

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.call_count) <= 0:
            raise ValueError("contraction dimensions and call_count must be positive")
        if min(self.x_m_stride, self.x_k_stride, self.w_n_stride, self.w_k_stride) <= 0:
            raise ValueError("contraction strides must be positive")
        if (self.input_dtype, self.output_dtype, self.accumulation_dtype) != (
            "float16", "float16", "float32"
        ):
            raise ValueError("the V3R-006 probe supports fp16 inputs/output with fp32 accumulation")


@dataclass(frozen=True)
class SimtSchedule:
    warps_per_cta: int
    vector_width: int

    def __post_init__(self) -> None:
        if self.warps_per_cta not in (1, 2, 4):
            raise ValueError("warps_per_cta must be one of 1, 2, or 4")
        if self.vector_width not in (1, 4):
            raise ValueError("vector_width must be one of 1 or 4")

    @property
    def threads_per_cta(self) -> int:
        return self.warps_per_cta * 32


def enumerate_simt_schedules(shape: ContractionShape) -> tuple[SimtSchedule, ...]:
    """Return a small, legal K-parallel menu; no Cartesian search explosion."""
    shape.validate()
    widths = (1, 4) if shape.x_k_stride == shape.w_k_stride == 1 else (1,)
    return tuple(
        SimtSchedule(warps, width)
        for warps in (1, 2, 4)
        for width in widths
    )


def output_tiles(shape: ContractionShape, schedule: SimtSchedule) -> int:
    shape.validate()
    return ceil(shape.output_elements / schedule.warps_per_cta)


def owner_grid_size(shape: ContractionShape, schedule: SimtSchedule, visible_sms: int) -> int:
    if visible_sms <= 0:
        raise ValueError("visible_sms must be positive")
    return min(visible_sms, output_tiles(shape, schedule))


def owner_grid_candidates(shape: ContractionShape, schedule: SimtSchedule,
                          visible_sms: int, resident_ctas: int) -> tuple[int, ...]:
    """Bound persistent-worker counts by actual entry residency and tile count."""
    if visible_sms <= 0 or resident_ctas <= 0:
        raise ValueError("visible_sms and resident_ctas must be positive")
    tiles = output_tiles(shape, schedule)
    return tuple(dict.fromkeys((
        min(visible_sms, tiles, resident_ctas),
        min(4 * visible_sms, tiles, resident_ctas),
        min(resident_ctas, tiles),
    )))
