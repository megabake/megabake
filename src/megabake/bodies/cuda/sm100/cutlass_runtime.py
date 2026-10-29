from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch


_COMPILED: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True)
class CutlassSm120Config:
    tile_shape: tuple[int, int, int] = (128, 128, 64)
    cluster_shape: tuple[int, int, int] = (1, 1, 1)
    pingpong: bool = False


class CutlassSm120Gemm:
    def __init__(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        config: CutlassSm120Config = CutlassSm120Config(),
    ) -> None:
        _validate_tensors(source, weight, output)
        self.config = config
        self._compiled = _compile(source, weight, output, config)

    def __call__(
        self,
        source: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        _validate_tensors(source, weight, output)
        self._compiled(*_arguments(source, weight, output, self.config))
        return output


def _validate_tensors(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not all(tensor.is_cuda for tensor in (source, weight, output)):
        raise ValueError("CUTLASS SM120 GEMM requires CUDA tensors")
    if torch.cuda.get_device_capability(source.device)[0] != 12:
        raise ValueError("CUTLASS SM120 GEMM requires an SM12x device")
    if source.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("CUTLASS SM120 GEMM requires FP16 or BF16 operands")
    if weight.dtype != source.dtype or output.dtype != source.dtype:
        raise ValueError("CUTLASS SM120 GEMM requires matching operand and output dtypes")
    if source.ndim != 2 or weight.ndim != 2 or output.ndim != 2:
        raise ValueError("CUTLASS SM120 GEMM requires rank-two tensors")
    if source.shape[1] != weight.shape[1]:
        raise ValueError("CUTLASS SM120 GEMM requires source and weight K dimensions to match")
    if output.shape != (source.shape[0], weight.shape[0]):
        raise ValueError("CUTLASS SM120 GEMM output shape does not match source @ weight.T")
    if source.stride(1) != 1 or weight.stride(1) != 1 or output.stride(1) != 1:
        raise ValueError("CUTLASS SM120 GEMM requires contiguous K-major operands and row-major output")
    if any(tensor.data_ptr() % 16 for tensor in (source, weight, output)):
        raise ValueError("CUTLASS SM120 GEMM requires 16-byte aligned tensors")


def _view(tensor: torch.Tensor) -> Any:
    from cutlass.cute.runtime import from_dlpack

    view = from_dlpack(tensor.unsqueeze(-1), assumed_align=16)
    return view.mark_layout_dynamic(leading_dim=1)


def _arguments(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: CutlassSm120Config,
) -> tuple[Any, ...]:
    import cutlass
    from cuda.bindings import driver as cuda
    from torch._vendor.quack.gemm_default_epi import GemmDefaultSm120
    from torch._vendor.quack.tile_scheduler import TileSchedulerOptions

    epilogue = GemmDefaultSm120.EpilogueArguments()
    scheduler = TileSchedulerOptions(
        max_active_clusters=cutlass.Int32(
            torch.cuda.get_device_properties(source.device).multi_processor_count
        ),
        max_swizzle_size=8,
        tile_count_semaphore=None,
        batch_idx_permute=None,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream)
    return (
        _view(source),
        _view(weight),
        _view(output),
        None,
        epilogue,
        scheduler,
        None,
        stream,
        None,
    )


@lru_cache(maxsize=32)
def _kernel(
    dtype: torch.dtype,
    config: CutlassSm120Config,
) -> Any:
    import cutlass
    from torch._vendor.quack.gemm_default_epi import GemmDefaultSm120

    cutlass_dtype = (
        cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    )
    return GemmDefaultSm120(
        cutlass.Float32,
        cutlass_dtype,
        config.tile_shape,
        config.cluster_shape,
        pingpong=config.pingpong,
        is_persistent=True,
    )


def _compile(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: CutlassSm120Config,
) -> Any:
    import cutlass.cute as cute

    key = (
        source.device.index,
        source.dtype,
        tuple(source.shape),
        tuple(source.stride()),
        tuple(weight.shape),
        tuple(weight.stride()),
        tuple(output.stride()),
        config,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _kernel(source.dtype, config),
            *_arguments(source, weight, output, config),
        )
        _COMPILED[key] = compiled
    return compiled
