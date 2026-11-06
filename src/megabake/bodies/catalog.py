"""The body providers a target runs: the generic ones on every target, plus
those of the GPU generation its architecture belongs to."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..graph import ComputeGraph, Region
from .cuda import sm90, sm100
from .cuda.generic import GenericBodyProvider
from .interface import (
    BodyConfig,
    BodyProvider,
    CandidateSet,
    Generation,
    InvalidBodyConfig,
    ProviderRejection,
    SelectionPolicy,
    TableGemmProvider,
    TargetFacts,
    validate_body_config,
)

GENERIC: tuple[BodyProvider, ...] = (GenericBodyProvider(),)
GENERATIONS: tuple[Generation, ...] = (
    replace(
        sm90.GENERATION,
        target_aliases={"sm90": "sm90a"},
        supports_wgmma=True,
    ),
    sm100.GENERATION,
)


def generation(target: TargetFacts) -> Generation | None:
    return next(
        (candidate for candidate in GENERATIONS if target.architecture in candidate.architectures),
        None,
    )


def providers(target: TargetFacts) -> tuple[BodyProvider, ...]:
    found = generation(target)
    return GENERIC + (found.providers if found is not None else ())


def provider(name: str) -> BodyProvider:
    """The provider that emits bodies configured with ``BodyConfig.provider``."""
    for candidate in GENERIC + tuple(item for found in GENERATIONS for item in found.providers):
        if candidate.provider_name == name:
            return candidate
    raise KeyError(f"no body provider named {name!r}")


def table_gemm(name: str) -> TableGemmProvider | None:
    """The table GEMM implementation for a body, if it uses that contract."""
    return next(
        (item for found in GENERATIONS for item in found.table_gemms if item.provider_name == name),
        None,
    )


def is_table_gemm(name: str) -> bool:
    """Recognize the original Hopper GEMM and registered table implementations."""
    return name == sm90.G0RangeBodyProvider.provider_name or table_gemm(name) is not None


def supported_architecture(architecture: str) -> bool:
    return any(architecture in found.architectures for found in GENERATIONS)


def tuner_for_device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("tuning requires a CUDA device")
    target = cuda_target(torch.cuda.get_device_properties(torch.cuda.current_device()))
    found = generation(target)
    if found is None or found.tuner is None:
        raise RuntimeError(f"no tuning provider for {target.architecture}")
    return found.tuner


def cuda_target(properties: Any) -> TargetFacts:
    """Translate CUDA device properties to the registered generation's facts."""
    architecture = f"sm{properties.major}{properties.minor}"
    for found in GENERATIONS:
        normalized = found.target_aliases.get(architecture, architecture)
        if normalized in found.architectures:
            return TargetFacts(
                architecture=normalized,
                threads_per_warp=properties.warp_size,
                max_threads_per_cta=properties.max_threads_per_block,
                max_shared_memory_per_cta=properties.shared_memory_per_block_optin,
                max_registers_per_thread=255,
                multiprocessor_count=properties.multi_processor_count,
                supports_tma=True,
                supports_wgmma=found.supports_wgmma,
            )
    raise ValueError(f"unsupported CUDA architecture {architecture}")


def launch_options(graph: ComputeGraph, target: TargetFacts, table_launch: bool) -> Any:
    """The ``SelectionPolicy.target_options`` of a launch of ``graph``."""
    found = generation(target)
    if found is None or found.launch_options is None:
        return None
    return found.launch_options(graph, table_launch)


def get_candidates(region: Region, target: TargetFacts, policy: SelectionPolicy) -> CandidateSet:
    configurations: list[BodyConfig] = []
    rejections: list[ProviderRejection] = []
    for candidate in providers(target):
        result = candidate.get_candidates(region, target, policy)
        for config in result.configurations:
            try:
                validate_body_config(config, target, policy)
            except InvalidBodyConfig as error:
                rejections.append(ProviderRejection(config.provider, str(error)))
            else:
                configurations.append(config)
        rejections.extend(result.rejections)
    return CandidateSet(tuple(configurations), tuple(rejections))
