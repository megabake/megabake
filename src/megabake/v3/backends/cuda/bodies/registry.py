"""Target-checked device-body ABI and bounded contraction tactic menu."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ....algorithms import AlgorithmChoice, choice_guard_failures
from ....contracts import ContractError, canonical_json
from ....semantics.indexed import IndexedOp, IndexedTensorProgram
from ..profile import CudaTargetProfile
from ..simt import ContractionShape, SimtSchedule, enumerate_simt_schedules
from ..tensor_core import TensorCoreSchedule, enumerate_tensor_core_schedules
from .generic import CudaBodyError, emit_cuda_body


BODY_ABI_VERSION = 1
_ROOT = Path(__file__).resolve().parents[6]
_SIMT_SOURCE = _ROOT / "src/cuda/v3/simt_body.cuh"
_SIMT_HEADER = _ROOT / "src/cuda/v3/body_harness.h"
_WMMA_SOURCE = _ROOT / "src/cuda/v3/body_harness.cu"
_CUTLASS_SOURCE = _ROOT / "src/cuda/v3/cutlass_38_body_probe.cu"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_source(path: Path) -> tuple[str, str]:
    source = path.read_text()
    return source, _sha256(source.encode())


def _output_map(operation: IndexedOp, program: IndexedTensorProgram) -> dict[str, Any]:
    values = {value.value_id: value for value in program.values}
    output = values[operation.outputs[0]]
    shape = list(output.shape)
    axes = list(operation.output_index_map)
    extent = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            extent = None
            break
        extent *= dim
    return {
        "value_id": output.value_id,
        "shape": shape,
        "dtype": output.dtype,
        "strides": list(output.strides),
        "index_map": axes,
        "logical_tile_coordinate": "flat_output_index",
        "linearization": "row_major",
        "elements_per_coordinate": 1,
        "output_elements": extent,
        "tail_predicate": (f"0 <= flat_output_index < {extent}" if extent is not None
                           else "guarded output extent"),
    }


def _reduction(operation: IndexedOp, program: IndexedTensorProgram) -> dict[str, Any]:
    values = {value.value_id: value for value in program.values}
    output = values[operation.outputs[0]]
    return {
        "axes": [{"name": axis.name, "extent": axis.extent}
                 for axis in operation.reduction_domain],
        "complete_reduction": True,
        "accumulator_dtype": operation.dtype_expression.get("accumulator", "unknown"),
        "output_dtype": output.dtype,
        "numerical_requirements": dict(operation.attributes.get("numerical_requirements", {})),
    }


def _contract_scalars(operation: IndexedOp) -> dict[str, Any]:
    contraction = operation.attributes.get("contraction", {})
    return {name: contraction.get(name, default)
            for name, default in (("alpha", 1.0), ("beta", 0.0))}


@dataclass(frozen=True)
class BodyTacticSpec:
    """Serializable body ABI record; source text is retained only in-process."""

    tactic_id: str
    provider: str
    provider_version: str
    algorithm_id: str
    indexed_program_hash: str
    operation_id: str
    origin_ids: tuple[str, ...]
    target_set: str
    target: str | None
    source_ref: str
    source_hash: str | None
    source_kind: str
    source_symbol: str | None
    device_callable: bool
    compatible: bool
    rejection_reasons: tuple[str, ...]
    shape_layout_guard: Mapping[str, Any]
    numerical_guard: Mapping[str, Any]
    output_tile_map: Mapping[str, Any]
    reduction_footprint: Mapping[str, Any]
    block_contract: Mapping[str, Any]
    movement_contract: Mapping[str, Any]
    descriptor_contract: Mapping[str, Any]
    scratch_contract: Mapping[str, Any]
    epilogue_contract: Mapping[str, Any]
    publication_contract: Mapping[str, Any]
    stage_capabilities: tuple[str, ...]
    local_measurements: Mapping[str, Any]
    source_text: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("tactic_id", "provider", "provider_version", "algorithm_id",
                     "indexed_program_hash", "operation_id", "target_set", "source_ref",
                     "source_kind"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractError(f"BodyTacticSpec {name} must be non-empty")
        if self.compatible and (not self.device_callable or self.rejection_reasons):
            raise ContractError("a compatible tactic must be callable and have no rejection reason")
        if not self.compatible and not self.rejection_reasons:
            raise ContractError("an incompatible tactic requires an explicit rejection reason")
        for name in ("shape_layout_guard", "numerical_guard", "output_tile_map",
                     "reduction_footprint", "block_contract", "movement_contract",
                     "descriptor_contract", "scratch_contract", "epilogue_contract",
                     "publication_contract", "local_measurements"):
            canonical_json(getattr(self, name))
        if self.source_text is not None and self.source_hash != _sha256(self.source_text.encode()):
            raise ContractError("BodyTacticSpec source hash does not match retained source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BODY_ABI_VERSION,
            "tactic_id": self.tactic_id,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "algorithm_id": self.algorithm_id,
            "indexed_program_hash": self.indexed_program_hash,
            "operation_id": self.operation_id,
            "origin_ids": list(self.origin_ids),
            "target_set": self.target_set,
            "target": self.target,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "source_kind": self.source_kind,
            "source_symbol": self.source_symbol,
            "device_callable": self.device_callable,
            "compatible": self.compatible,
            "rejection_reasons": list(self.rejection_reasons),
            "shape_layout_guard": dict(self.shape_layout_guard),
            "numerical_guard": dict(self.numerical_guard),
            "output_tile_map": dict(self.output_tile_map),
            "reduction_footprint": dict(self.reduction_footprint),
            "block_contract": dict(self.block_contract),
            "movement_contract": dict(self.movement_contract),
            "descriptor_contract": dict(self.descriptor_contract),
            "scratch_contract": dict(self.scratch_contract),
            "epilogue_contract": dict(self.epilogue_contract),
            "publication_contract": dict(self.publication_contract),
            "stage_capabilities": list(self.stage_capabilities),
            "local_measurements": dict(self.local_measurements),
        }


@dataclass(frozen=True)
class BodyTacticRegistry:
    indexed_program_hash: str
    choice_id: str
    profile_key: str
    compatible_tactics: tuple[BodyTacticSpec, ...]
    rejected_tactics: tuple[BodyTacticSpec, ...]
    provider_rejections: tuple[Mapping[str, str], ...]

    @property
    def registry_key(self) -> str:
        payload = {
            "indexed_program_hash": self.indexed_program_hash,
            "choice_id": self.choice_id,
            "profile_key": self.profile_key,
            "compatible": [item.to_dict() for item in self.compatible_tactics],
            "rejected": [item.to_dict() for item in self.rejected_tactics],
            "provider_rejections": [dict(item) for item in self.provider_rejections],
        }
        return _sha256(canonical_json(payload).encode())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BODY_ABI_VERSION,
            "registry_key": self.registry_key,
            "indexed_program_hash": self.indexed_program_hash,
            "choice_id": self.choice_id,
            "target_profile_key": self.profile_key,
            "compatible_tactics": [item.to_dict() for item in self.compatible_tactics],
            "rejected_tactics": [item.to_dict() for item in self.rejected_tactics],
            "provider_rejections": [dict(item) for item in self.provider_rejections],
            "claim": "not_measured",
        }


def _base_spec(*, program: IndexedTensorProgram, operation: IndexedOp,
               choice: AlgorithmChoice, target_set: str, target: str | None,
               tactic_id: str, provider: str, provider_version: str,
               source_ref: str, source_hash: str | None, source_kind: str,
               source_symbol: str | None, device_callable: bool, compatible: bool,
               reasons: tuple[str, ...], block: Mapping[str, Any],
               movement: Mapping[str, Any], descriptor: Mapping[str, Any],
               scratch: Mapping[str, Any], stages: tuple[str, ...],
               source_text: str | None = None) -> BodyTacticSpec:
    values = {value.value_id: value for value in program.values}
    inputs = [values[value_id] for value_id in operation.inputs]
    output = values[operation.outputs[0]]
    return BodyTacticSpec(
        tactic_id=tactic_id, provider=provider, provider_version=provider_version,
        algorithm_id=choice.algorithm, indexed_program_hash=program.structural_hash,
        operation_id=operation.op_id, origin_ids=operation.origin_ids,
        target_set=target_set, target=target, source_ref=source_ref,
        source_hash=source_hash, source_kind=source_kind, source_symbol=source_symbol,
        device_callable=device_callable, compatible=compatible,
        rejection_reasons=reasons,
        shape_layout_guard={"inputs": [
            {"value_id": value.value_id, "shape": list(value.shape),
             "strides": list(value.strides), "dtype": value.dtype,
             "layout": value.layout, "alignment_bytes": value.alignment_bytes}
            for value in inputs],
            "output": {"value_id": output.value_id, "shape": list(output.shape),
                       "strides": list(output.strides), "dtype": output.dtype,
                       "layout": output.layout, "alignment_bytes": output.alignment_bytes}},
        numerical_guard={"choice_requirements": dict(choice.numerical_requirements),
                         "dtype_expression": dict(operation.dtype_expression)},
        output_tile_map=_output_map(operation, program),
        reduction_footprint=_reduction(operation, program),
        block_contract=dict(block), movement_contract=dict(movement),
        descriptor_contract=dict(descriptor), scratch_contract=dict(scratch),
        epilogue_contract={"owner": "body", "operation": dict(operation.attributes.get("contraction", {})),
                           "output_cast_dtype": output.dtype},
        publication_contract={"owner": "lane_zero_or_all_output_threads",
                              "scope": "global_output_store", "requires_completion_before_return": True},
        stage_capabilities=stages,
        local_measurements={"standalone_us": None, "lean_entry_us": None,
                            "status": "not_measured"},
        source_text=source_text,
    )


def _target(profile: CudaTargetProfile, name: str) -> str | None:
    value = profile.target_sets[name].value
    return value[0] if isinstance(value, (list, tuple)) and value else None


def _target_reason(profile: CudaTargetProfile, target_set: str, target: str | None) -> str | None:
    if target is None:
        return f"{target_set} target is unknown or unavailable in this profile"
    status = profile.supports_target(target)
    if status is True:
        return None
    if status is False:
        return f"exact target {target} is not both device-legal and supported by the selected compiler"
    return f"exact target {target} legality is unknown"


def _generic_tactic(program: IndexedTensorProgram, operation: IndexedOp,
                    choice: AlgorithmChoice, profile: CudaTargetProfile,
                    target: str | None) -> BodyTacticSpec:
    body = emit_cuda_body(program, operation)
    reasons = []
    if target_reason := _target_reason(profile, "baseline", target):
        reasons.append(target_reason)
    limit = profile.resource_limits.get("max_threads_per_block")
    maximum = limit.value if limit and isinstance(limit.value, int) else None
    if maximum is None:
        reasons.append("maximum threads per block is unknown")
    return _base_spec(
        program=program, operation=operation, choice=choice, target_set="baseline", target=target,
        tactic_id="generic.indexed", provider="generic", provider_version="indexed-body-v1",
        source_ref=f"generated:{operation.op_id}:{body.entry_point or body.body_kind}",
        source_hash=_sha256(body.source.encode()), source_kind="generated_device_function",
        source_symbol=body.entry_point, device_callable=body.entry_point is not None,
        compatible=not reasons and body.entry_point is not None,
        reasons=tuple(reasons or (["operation has no callable generated body"] if body.entry_point is None else [])),
        block={"dimensions": ["x"], "threads_min": 1, "threads_max": maximum,
               "collective": "none", "uniform_call": True},
        movement={"reads": "synchronous_global", "writes": "synchronous_global",
                  "async_operations": []},
        descriptor={"host_descriptors": [], "binding": "pointer_arguments", "lifetime": "call"},
        scratch={"shared_bytes": 0, "registers": "compiler_measured_on_composed_entry",
                 "lifetime": "one_output_tile"},
        stages=("ATOMIC_TILE",) if not reasons and body.entry_point else (),
        source_text=body.source,
    )


def _simt_tactics(program: IndexedTensorProgram, operation: IndexedOp,
                  choice: AlgorithmChoice, profile: CudaTargetProfile,
                  shape: ContractionShape, target: str | None) -> tuple[list[BodyTacticSpec], list[dict[str, str]]]:
    try:
        schedules = enumerate_simt_schedules(shape)
    except ValueError as exc:
        return [], [{"provider": "simt", "reason": str(exc)}]
    body_text, _ = _file_source(_SIMT_SOURCE)
    header_text, _ = _file_source(_SIMT_HEADER)
    # Include the struct definition in the source hash: it is part of the body ABI.
    source_text = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cstdint>\n" + header_text + "\n" + body_text
    source_hash = _sha256(source_text.encode())
    reasons = []
    if target_reason := _target_reason(profile, "baseline", target):
        reasons.append(target_reason)
    warp = profile.resource_limits.get("warp_size")
    if not warp or warp.value != 32:
        reasons.append("SIMT shuffle body requires a queried 32-lane warp")
    max_threads = profile.resource_limits.get("max_threads_per_block")
    result = []
    for schedule in schedules:
        local_reasons = list(reasons)
        if not max_threads or not isinstance(max_threads.value, int):
            local_reasons.append("maximum threads per block is unknown")
        elif schedule.threads_per_cta > max_threads.value:
            local_reasons.append(f"block needs {schedule.threads_per_cta} threads; target limit is {max_threads.value}")
        result.append(_base_spec(
            program=program, operation=operation, choice=choice, target_set="baseline", target=target,
            tactic_id=f"simt.k_parallel.w{schedule.warps_per_cta}.v{schedule.vector_width}",
            provider="simt", provider_version="V3R-006-schedule-family-v1",
            source_ref="src/cuda/v3/simt_body.cuh#simt_output",
            source_hash=source_hash, source_kind="device_function_fragment",
            source_symbol=f"simt_output<{schedule.vector_width}>", device_callable=True,
            compatible=not local_reasons, reasons=tuple(local_reasons),
            block={"dimensions": [schedule.threads_per_cta, 1, 1],
                   "warps": schedule.warps_per_cta, "warp_size": 32,
                   "roles": ["one_logical_output_per_warp", "all_32_lanes_reduce_k"],
                   "collective": "warp_shuffle", "uniform_call": True},
            movement={"reads": "synchronous_global", "writes": "lane_zero_global_store",
                      "async_operations": []},
            descriptor={"host_descriptors": [], "binding": "Mb3Contraction_by_value",
                        "lifetime": "entry_call", "entry_argument_bytes": 104},
            scratch={"shared_bytes": 0, "registers": "compiler_measured_on_composed_entry",
                     "accumulator": "fp32_per_lane", "lifetime": "full_k_reduction"},
            stages=("ATOMIC_TILE",) if not local_reasons else (), source_text=source_text,
        ))
    return result, []


def _tensor_core_rejections(program: IndexedTensorProgram, operation: IndexedOp,
                            choice: AlgorithmChoice, profile: CudaTargetProfile,
                            shape: ContractionShape, target: str | None) -> tuple[list[BodyTacticSpec], list[dict[str, str]]]:
    try:
        schedules = enumerate_tensor_core_schedules(shape, target=target or "")
    except ValueError as exc:
        return [], [{"provider": "tensor_core", "reason": str(exc)}]
    if not schedules:
        reason = (_target_reason(profile, "architecture_specific", target) or
                  "WMMA provider has no schedule for this exact shape")
        return [], [{"provider": "tensor_core", "reason": reason}]
    source_text, source_hash = _file_source(_WMMA_SOURCE)
    reason = ("output_major_mma is a __global__ entry and uses blockIdx.x as its tile owner; "
              "the provider has no device-callable logical-tile adapter")
    target_reason = _target_reason(profile, "architecture_specific", target)
    reasons = (reason,) + ((target_reason,) if target_reason else ())
    result = []
    for schedule in schedules:
        result.append(_base_spec(
            program=program, operation=operation, choice=choice,
            target_set="architecture_specific", target=target,
            tactic_id=f"tensor_core.wmma.w{schedule.warps_per_cta}.d{schedule.mainloop_depth}",
            provider="tensor_core", provider_version="V3R-007-output-major-wmma",
            source_ref="src/cuda/v3/body_harness.cu#output_major_mma",
            source_hash=source_hash, source_kind="global_entry", source_symbol="output_major_mma",
            device_callable=False, compatible=False, reasons=reasons,
            block={"dimensions": [schedule.threads_per_cta, 1, 1],
                   "warps": schedule.warps_per_cta, "roles": ["wmma_warp_tiles"],
                   "collective": "warp_mma", "uniform_call": True},
            movement={"reads": "global_to_shared_synchronous", "writes": "global_store",
                      "async_operations": []},
            descriptor={"host_descriptors": [], "binding": "global_kernel_arguments",
                        "lifetime": "entry_call"},
            scratch={"shared_bytes": (schedule.mainloop_depth * schedule.warps_per_cta * 256 * 2 +
                      schedule.mainloop_depth * 256 * 2 + schedule.warps_per_cta * 256 * 4),
                     "registers": "compiler_measured_on_composed_entry",
                     "accumulator": "wmma_fp32", "lifetime": "full_k_reduction"},
            stages=(), source_text=source_text,
        ))
    return result, []


def _cutlass_rejection(program: IndexedTensorProgram, operation: IndexedOp,
                       choice: AlgorithmChoice, profile: CudaTargetProfile,
                       shape: ContractionShape, target: str | None) -> BodyTacticSpec | None:
    if not _CUTLASS_SOURCE.is_file() or not profile.versions.get("cutlass") or not profile.versions["cutlass"].value:
        return None
    source_text, source_hash = _file_source(_CUTLASS_SOURCE)
    target_reason = _target_reason(profile, "architecture_specific", target)
    reasons = ["CUTLASS probe is a host-launched __global__ entry with a grid-bound Params object; "
               "it has no arbitrary logical-tile body adapter"]
    if target_reason:
        reasons.append(target_reason)
    return _base_spec(
        program=program, operation=operation, choice=choice,
        target_set="architecture_specific", target=target,
        tactic_id="cutlass3x.probe128x64x32", provider="cutlass3x",
        provider_version=str(profile.versions["cutlass"].value),
        source_ref="src/cuda/v3/cutlass_38_body_probe.cu#cutlass_device_callable",
        source_hash=source_hash, source_kind="global_entry", source_symbol="cutlass_device_callable",
        device_callable=False, compatible=False,
        reasons=tuple(reasons),
        block={"dimensions": [128, 1, 1], "roles": ["CUTLASS_threadblock"],
               "collective": "CTA_collective", "uniform_call": True},
        movement={"reads": "CUTLASS_global_to_shared", "writes": "CUTLASS_epilogue",
                  "async_operations": ["provider_internal"]},
        descriptor={"host_descriptors": ["Kernel::Params"],
                    "binding": "__grid_constant__ entry_argument", "lifetime": "entry_call"},
        scratch={"shared_bytes": None, "status": "requires_exact_composed_compile",
                 "registers": "compiler_measured_on_composed_entry",
                 "accumulator": "CUTLASS_fp32", "lifetime": "full_k_reduction"},
        stages=(), source_text=source_text,
    )


def query_body_tactics(program: IndexedTensorProgram, operation: IndexedOp | str,
                       choice: AlgorithmChoice, profile: CudaTargetProfile, *,
                       numerical_policy: Any = None) -> BodyTacticRegistry:
    """Return compatible body specs and explicit rejected-provider candidates."""
    if not program.strict_supported:
        raise ValueError("body registry requires a strict-supported IndexedTensorProgram")
    if isinstance(operation, str):
        try:
            operation = next(item for item in program.operations if item.op_id == operation)
        except StopIteration as exc:
            raise ValueError(f"unknown indexed operation {operation!r}") from exc
    if operation.kind != "Contraction":
        raise ValueError("V3R-022 first registry slice supports indexed contractions")
    if operation.op_id not in choice.operation_ids:
        raise ValueError("AlgorithmChoice does not cover the requested operation")
    if choice.guards.get("indexed_program_hash") != program.structural_hash:
        raise ValueError("AlgorithmChoice was built for a different IndexedTensorProgram")
    guard_failures = choice_guard_failures(program, choice, numerical_policy=numerical_policy)
    if guard_failures:
        raise ValueError("AlgorithmChoice guard failed: " + "; ".join(guard_failures))

    target = _target(profile, "baseline")
    compatible: list[BodyTacticSpec] = []
    rejected: list[BodyTacticSpec] = []
    provider_rejections: list[dict[str, str]] = []
    try:
        generic = _generic_tactic(program, operation, choice, profile, target)
        (compatible if generic.compatible else rejected).append(generic)
    except CudaBodyError as exc:
        provider_rejections.append({"provider": "generic", "reason": str(exc)})

    try:
        shape = ContractionShape.from_indexed(program, operation)
    except ValueError as exc:
        provider_rejections.append({"provider": "contraction_schedules", "reason": str(exc)})
        shape = None
    if shape is not None:
        simt, failures = _simt_tactics(program, operation, choice, profile, shape, target)
        provider_rejections.extend(failures)
        for tactic in simt:
            (compatible if tactic.compatible else rejected).append(tactic)
        architecture_target = _target(profile, "architecture_specific")
        tensor_core, failures = _tensor_core_rejections(
            program, operation, choice, profile, shape, architecture_target)
        rejected.extend(tensor_core)
        provider_rejections.extend(failures)
        if cutlass := _cutlass_rejection(program, operation, choice, profile,
                                         shape, architecture_target):
            rejected.append(cutlass)
        cublasdx = profile.versions.get("cublasdx")
        if cublasdx is None or cublasdx.value is None:
            provider_rejections.append({
                "provider": "cublasdx",
                "reason": (cublasdx.unknown_reason if cublasdx else
                           "optional provider version was not queried"),
            })

    return BodyTacticRegistry(
        indexed_program_hash=program.structural_hash, choice_id=choice.choice_id,
        profile_key=profile.profile_key, compatible_tactics=tuple(compatible),
        rejected_tactics=tuple(rejected), provider_rejections=tuple(provider_rejections),
    )


__all__ = ["BODY_ABI_VERSION", "BodyTacticRegistry", "BodyTacticSpec", "query_body_tactics"]
