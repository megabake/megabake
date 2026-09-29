"""Ordered static worker entries for small, fully resident indexed programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from ...contracts import canonical_json
from ...algorithms import enumerate_algorithm_choices
from ...logical.plan import LogicalExecutionPlan
from ...semantics.indexed import IndexedTensorProgram
from .admit import EntryAdmissionError, _EntryResources, _entry_code_size, _runtime_device
from .bodies.registry import query_body_tactics
from .bodies.generic import CudaBodyError, emit_cuda_body
from .compile import CompileResult, _compiler_version, _ptxas
from .profile import CudaTargetProfile


class WorkerProgramError(ValueError):
    """A worker program cannot prove its static phase or binding contract."""


@dataclass(frozen=True)
class WorkerStage:
    operation_id: str
    origin_id: str
    entry_point: str
    input_value_ids: tuple[str, ...]
    output_value_id: str
    input_types: tuple[str, ...]
    output_type: str
    output_elements: int
    body_source: str
    body_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "origin_id": self.origin_id,
            "entry_point": self.entry_point,
            "input_value_ids": list(self.input_value_ids),
            "output_value_id": self.output_value_id,
            "input_types": list(self.input_types),
            "output_type": self.output_type,
            "output_elements": self.output_elements,
            "body_kind": self.body_kind,
        }


@dataclass(frozen=True)
class WorkerProgram:
    indexed_program_hash: str
    logical_plan_hash: str
    target_profile_key: str
    target: str
    block_threads: int
    grid_ctas: int
    value_ids: tuple[str, ...]
    stages: tuple[WorkerStage, ...]
    body_search: tuple[Mapping[str, Any], ...]
    runtime_guards: tuple[Mapping[str, Any], ...]
    source: str
    program_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "indexed_program_hash": self.indexed_program_hash,
            "logical_plan_hash": self.logical_plan_hash,
            "target_profile_key": self.target_profile_key,
            "target": self.target,
            "block_threads": self.block_threads,
            "grid_ctas": self.grid_ctas,
            "value_ids": list(self.value_ids),
            "stages": [stage.to_dict() for stage in self.stages],
            "body_search": [dict(item) for item in self.body_search],
            "runtime_guards": [dict(item) for item in self.runtime_guards],
            "source_hash": hashlib.sha256(self.source.encode()).hexdigest(),
            "program_hash": self.program_hash,
            "progress_proof": {
                "dispatch": "blockIdx.x is a worker id; each worker owns one contiguous output interval per stage",
                "phase_order": [stage.operation_id for stage in self.stages],
                "join": "uniform cooperative grid.sync after every non-final stage",
                "idle_workers_join": True,
                "wait_for_graph": "acyclic static stage order; every grid worker is resident before launch",
                "body_async_operations": [],
            },
        }


def _selected_target(profile: CudaTargetProfile) -> str:
    baseline = profile.target_sets.get("baseline")
    choices = baseline.value if baseline else None
    if not isinstance(choices, (tuple, list)) or not choices:
        raise WorkerProgramError("CUDA baseline target is unknown")
    target = choices[0]
    if profile.supports_target(target) is not True:
        raise WorkerProgramError(f"exact CUDA target {target!r} is not proven legal")
    cooperative = profile.feature_attributes.get("cooperative_launch")
    if cooperative is None or cooperative.value is not True:
        raise WorkerProgramError("cooperative launch support is not proven for this target")
    return target


def _value_elements(value: Any, operation_id: str) -> int:
    if not value.shape or any(not isinstance(dim, int) or dim < 0 for dim in value.shape):
        raise WorkerProgramError(f"{operation_id} has a scalar or dynamic output shape")
    return math.prod(value.shape)


def _body_search(program: IndexedTensorProgram, logical_plan: LogicalExecutionPlan,
                 profile: CudaTargetProfile, operation: Any, body: Any) -> Mapping[str, Any]:
    if body.body_kind == "view_map":
        output = next(value for value in program.values if value.value_id == operation.outputs[0])
        return {"operation_id": operation.op_id, "candidates": [
            {"id": "view.alias", "provider": "generic", "status": "compatible"}],
            "selected": "view.alias" if output.alias_kind == "view" else None,
            "reason": "the proved view is an address map and needs no device stage"}
    candidates: list[Mapping[str, Any]] = [{
        "id": "generic.indexed", "provider": "generic", "status": "compatible",
        "source_hash": hashlib.sha256(body.source.encode()).hexdigest(),
    }]
    rejections: list[Mapping[str, Any]] = []
    if operation.kind == "Contraction":
        choices = enumerate_algorithm_choices(program)
        choice = next((item for item in choices
                       if item.choice_id in logical_plan.selected_choice_ids and
                       operation.op_id in item.operation_ids), None)
        if choice is None:
            raise WorkerProgramError(f"{operation.op_id} has no selected logical algorithm choice")
        registry = query_body_tactics(program, operation, choice, profile)
        generic = next((item for item in registry.compatible_tactics
                        if item.provider == "generic" and item.tactic_id == "generic.indexed"), None)
        if generic is None or generic.source_hash != candidates[0]["source_hash"]:
            raise WorkerProgramError(f"{operation.op_id} generic registry body differs from the emitted worker body")
        candidates = [
            {"id": item.tactic_id, "provider": item.provider,
             "status": "compatible" if item.compatible else "rejected",
             "source_hash": item.source_hash,
             "reasons": list(item.rejection_reasons)}
            for item in (*registry.compatible_tactics, *registry.rejected_tactics)
        ]
        rejections = [dict(item) for item in registry.provider_rejections]
        return {"operation_id": operation.op_id, "candidates": candidates,
                "provider_rejections": rejections, "selected": generic.tactic_id,
                "reason": "selected the generic body with the worker tile ABI; specialized contraction tactics remain unadapted alternatives"}
    return {"operation_id": operation.op_id, "candidates": candidates,
            "provider_rejections": rejections, "selected": "generic.indexed",
            "reason": "generic indexed body is the available callable tactic for this primitive"}


def _emit_source(stages: Sequence[WorkerStage], value_ids: Sequence[str],
                 block_threads: int) -> str:
    value_index = {value_id: index for index, value_id in enumerate(value_ids)}
    sources = ["""#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
"""]
    sources.extend(stage.body_source for stage in stages)
    source = "\n".join(sources)
    source += f"""
struct V3WorkerBindings {{ void *values[{len(value_ids)}]; }};

extern \"C\" __global__ __launch_bounds__({block_threads})
void megabake_v3_entry(V3WorkerBindings p) {{
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int64_t worker = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_begin = worker * static_cast<int64_t>(blockDim.x);
"""
    for index, stage in enumerate(stages):
        source += (f"  const int64_t tile_end_{index} = tile_begin + blockDim.x < {stage.output_elements} "
                   f"? tile_begin + blockDim.x : {stage.output_elements};\n")
        source += f"  const int64_t stage_begin_{index} = tile_begin;\n"
        source += f"  const int64_t stage_end_{index} = tile_end_{index};\n"
        args = [f"static_cast<const {dtype} *>(p.values[{value_index[value_id]}])"
                for dtype, value_id in zip(stage.input_types, stage.input_value_ids)]
        args.extend((f"static_cast<{stage.output_type} *>(p.values[{value_index[stage.output_value_id]}])",
                     f"stage_begin_{index}", f"stage_end_{index}"))
        source += f"  {stage.entry_point}({', '.join(args)});\n"
        if index + 1 < len(stages):
            source += "  grid.sync();\n"
    source += "}\n"
    return source


def lower_worker_program(program: IndexedTensorProgram, logical_plan: LogicalExecutionPlan,
                         profile: CudaTargetProfile, *, block_threads: int = 128) -> WorkerProgram:
    """Turn a verified indexed expansion into ordered, uniformly joined CUDA stages."""
    if not program.strict_supported:
        raise WorkerProgramError("worker lowering requires strict-supported indexed semantics")
    if logical_plan.indexed_program_hash != program.structural_hash:
        raise WorkerProgramError("logical plan belongs to different indexed semantics")
    expected_coverage = {operation.op_id for operation in program.operations}
    selected_coverage = {item.get("op_id") for item in logical_plan.semantic_coverage}
    if selected_coverage != expected_coverage:
        raise WorkerProgramError("logical plan does not cover every original indexed operation")
    if not isinstance(block_threads, int) or block_threads <= 0:
        raise WorkerProgramError("worker block size must be a positive integer")
    maximum = profile.resource_limits.get("max_threads_per_block")
    if maximum is None or not isinstance(maximum.value, int) or block_threads > maximum.value:
        raise WorkerProgramError("worker block size exceeds or lacks a proven target limit")
    target = _selected_target(profile)
    values = {item.value_id: item for item in program.values}
    producer_order: dict[str, int] = {}
    stages: list[WorkerStage] = []
    body_search: list[Mapping[str, Any]] = []
    runtime_guards: list[Mapping[str, Any]] = []
    transitions_by_output = {item.new_value: item for item in program.state_transitions}
    transition_orders = []
    effect_producers = {}
    for order, operation in enumerate(program.operations):
        for output_id in operation.outputs:
            transition = transitions_by_output.get(output_id)
            if transition is not None:
                transition_orders.append(transition.order)
                effect_producers[transition.effect_id] = order
    if transition_orders != sorted(transition_orders):
        raise WorkerProgramError("indexed state effects differ from their declared source order")
    for order, operation in enumerate(program.operations):
        for value_id in operation.inputs:
            if value_id in producer_order and producer_order[value_id] >= order:
                raise WorkerProgramError("indexed dependency is not in static producer-before-consumer order")
        for edge in operation.effect_edges:
            if not edge.required or edge.kind != "state_read_after_publish":
                continue
            for effect_id in edge.depends_on:
                producer = effect_producers.get(effect_id)
                if producer is None or producer >= order:
                    raise WorkerProgramError("state read is not ordered after its publishing write")
        try:
            body = emit_cuda_body(program, operation)
        except CudaBodyError:
            raise
        body_search.append(_body_search(program, logical_plan, profile, operation, body))
        if operation.kind == "Guard":
            raise WorkerProgramError(f"runtime guard {operation.op_id} has no worker-entry implementation")
        if body.body_kind == "view_map":
            output = values[operation.outputs[0]]
            if output.alias_kind != "view":
                raise WorkerProgramError(f"{operation.op_id} has an unproved view alias")
            producer_order[operation.outputs[0]] = order
            continue
        if (not body.entry_point or not body.output_type or len(operation.outputs) != 1 or
                len(body.input_types) != len(operation.inputs)):
            raise WorkerProgramError(f"{operation.op_id} has no complete one-output body ABI")
        output_id = operation.outputs[0]
        output_elements = _value_elements(values[output_id], operation.op_id)
        stages.append(WorkerStage(
            operation_id=operation.op_id,
            origin_id=operation.origin_ids[0] if operation.origin_ids else operation.op_id,
            entry_point=body.entry_point,
            input_value_ids=operation.inputs,
            output_value_id=output_id,
            input_types=body.input_types,
            output_type=body.output_type,
            output_elements=output_elements,
            body_source=body.source,
            body_kind=body.body_kind,
        ))
        producer_order[output_id] = order
        if operation.kind == "Scatter/StateWrite":
            transition = transitions_by_output.get(output_id)
            bounds = operation.attributes.get("index_bounds")
            if transition is None or not isinstance(bounds, Mapping):
                raise WorkerProgramError(f"state write {operation.op_id} lacks a bounded StepABI transition")
            runtime_guards.append({
                "operation_id": operation.op_id,
                "value_id": transition.index_value,
                "lower": int(bounds["lower"]),
                "upper_exclusive": int(bounds["upper_exclusive"]),
            })
    if not stages:
        raise WorkerProgramError("indexed program has no executable worker stages")
    max_outputs = max(stage.output_elements for stage in stages)
    grid_ctas = math.ceil(max_outputs / block_threads)
    value_ids = tuple(value.value_id for value in program.values)
    source = _emit_source(stages, value_ids, block_threads)
    payload = {
        "indexed_program_hash": program.structural_hash,
        "logical_plan_hash": logical_plan.structural_hash,
        "profile_key": profile.profile_key,
        "target": target,
        "block_threads": block_threads,
        "grid_ctas": grid_ctas,
        "value_ids": value_ids,
        "stages": [stage.to_dict() for stage in stages],
        "body_search": [dict(item) for item in body_search],
        "runtime_guards": runtime_guards,
        "source_hash": hashlib.sha256(source.encode()).hexdigest(),
    }
    return WorkerProgram(
        program.structural_hash, logical_plan.structural_hash, profile.profile_key,
        target, block_threads, grid_ctas, value_ids, tuple(stages),
        tuple(body_search), tuple(runtime_guards), source,
        hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    )


def compile_worker_entry(worker: WorkerProgram, profile: CudaTargetProfile,
                         output: str | Path, *, nvcc: str = "nvcc") -> CompileResult:
    """Compile the exact static worker entry as a shared library for admission and launch."""
    if worker.target_profile_key != profile.profile_key:
        raise WorkerProgramError("worker program belongs to another target profile")
    if worker.target != _selected_target(profile) or profile.supports_target(worker.target) is not True:
        raise WorkerProgramError("worker target is not exact-target legal")
    executable = nvcc if "/" in nvcc else shutil.which(nvcc)
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError(f"CUDA compiler {nvcc!r} is unavailable")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = output.with_suffix(".cu")
    source_path.write_text(worker.source + _worker_runtime_source(worker))
    source_bytes = source_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    compiler_version = _compiler_version(executable)
    artifact_key = hashlib.sha256(canonical_json({
        "worker_program_hash": worker.program_hash,
        "source_hash": source_hash,
        "target": worker.target,
        "profile_key": profile.profile_key,
        "compiler_version": compiler_version,
        "binding_abi": f"V3WorkerBindings/v1/{len(worker.value_ids)}",
    }).encode()).hexdigest()
    command = [executable, "-std=c++17", f"-arch={worker.target}", "-Xptxas=-v",
               "-shared", "-Xcompiler", "-fPIC", str(source_path), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    artifact_hash = (hashlib.sha256(output.read_bytes()).hexdigest()
                     if result.returncode == 0 and output.is_file() else None)
    return CompileResult(
        tuple(command), result.returncode, result.stdout, result.stderr,
        compiler_version, worker.target, source_hash, str(output), artifact_hash,
        output.stat().st_size if artifact_hash else None, _ptxas(result.stderr), artifact_key,
    )


def _worker_runtime_source(worker: WorkerProgram) -> str:
    return f"""
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
struct V3WorkerResources {{
  int device_sms; int cooperative_launch; int block_threads; int requested_ctas;
  int active_ctas_per_sm; int resident_ctas; int registers_per_thread;
  size_t static_shared_bytes; size_t local_bytes; int function_max_threads;
  int max_dynamic_shared_bytes; size_t constant_bytes; int binary_version;
  int ptx_version; size_t parameter_bytes; int admitted;
}};
extern \"C\" const char *mb3_worker_program_hash() {{ return \"{worker.program_hash}\"; }}
extern \"C\" int mb3_worker_value_count() {{ return {len(worker.value_ids)}; }}
extern \"C\" int mb3_entry_resources(int requested_ctas, V3WorkerResources *out) {{
  if (!out || requested_ctas <= 0) return -1;
  int device = 0; cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaDeviceProp properties{{}};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  int cooperative = 0;
  status = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaFuncAttributes attributes{{}};
  status = cudaFuncGetAttributes(&attributes, megabake_v3_entry);
  if (status != cudaSuccess) return static_cast<int>(status);
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, megabake_v3_entry, {worker.block_threads}, 0);
  if (status != cudaSuccess) return static_cast<int>(status);
  const int resident = active * properties.multiProcessorCount;
  *out = V3WorkerResources{{properties.multiProcessorCount, cooperative,
      {worker.block_threads}, requested_ctas, active, resident, attributes.numRegs,
      attributes.sharedSizeBytes, attributes.localSizeBytes,
      attributes.maxThreadsPerBlock, attributes.maxDynamicSharedSizeBytes,
      attributes.constSizeBytes, attributes.binaryVersion, attributes.ptxVersion,
      sizeof(V3WorkerBindings), cooperative && requested_ctas <= resident}};
  return 0;
}}
extern \"C\" int mb3_launch_worker_entry(const V3WorkerBindings *bindings,
                                            int requested_ctas, uintptr_t stream_value) {{
  if (!bindings) return -1;
  if (requested_ctas != {worker.grid_ctas}) return -3;
  void *arguments[] = {{const_cast<V3WorkerBindings *>(bindings)}};
  return static_cast<int>(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(megabake_v3_entry), dim3(requested_ctas),
      dim3({worker.block_threads}), arguments, 0,
      reinterpret_cast<cudaStream_t>(stream_value)));
}}
"""


def _worker_library(path: str | Path, value_count: int):
    import ctypes
    library = ctypes.CDLL(str(path))
    library.mb3_entry_resources.argtypes = [ctypes.c_int, ctypes.POINTER(_EntryResources)]
    library.mb3_entry_resources.restype = ctypes.c_int
    fields = [("values", ctypes.c_void_p * value_count)]
    bindings_type = type("V3WorkerBindings", (ctypes.Structure,), {"_fields_": fields})
    library.mb3_launch_worker_entry.argtypes = [ctypes.POINTER(bindings_type), ctypes.c_int,
                                                ctypes.c_size_t]
    library.mb3_launch_worker_entry.restype = ctypes.c_int
    return library, bindings_type


class WorkerEntrySession:
    """Hold one admitted function and its by-value pointer binding record."""

    def __init__(self, library_path: str | Path, worker: WorkerProgram,
                 profile: CudaTargetProfile, admission: Mapping[str, Any]):
        import ctypes

        if (admission.get("program_hash") != worker.program_hash or
                admission.get("target_profile_key") != profile.profile_key or
                not admission.get("launch_contract", {}).get("admitted")):
            raise EntryAdmissionError("worker admission does not match the selected program and target")
        self.runtime_target = _runtime_device(profile)
        self.worker = worker
        self.profile = profile
        self.admission = admission
        self.library, self.bindings_type = _worker_library(library_path, len(worker.value_ids))
        self.library.mb3_worker_program_hash.argtypes = []
        self.library.mb3_worker_program_hash.restype = ctypes.c_char_p
        self.library.mb3_worker_value_count.argtypes = []
        self.library.mb3_worker_value_count.restype = ctypes.c_int
        if self.library.mb3_worker_program_hash().decode() != worker.program_hash:
            raise EntryAdmissionError("compiled worker library belongs to a different program hash")
        if self.library.mb3_worker_value_count() != len(worker.value_ids):
            raise EntryAdmissionError("compiled worker library has a different binding ABI")
        self.resident_ctas = admission["compiled_function"]["cooperative_resident_ctas"]

    def launch(self, values: Sequence[Any], *, grid_ctas: int | None = None,
               stream: int = 0) -> dict[str, Any]:
        if len(values) != len(self.worker.value_ids):
            raise ValueError(f"worker ABI expects {len(self.worker.value_ids)} values, got {len(values)}")
        import torch
        if torch.cuda.current_device() != self.profile.device_index:
            return {"launched": False, "return_code": -9,
                    "reason": "current CUDA device differs from the admitted worker target"}
        by_id = dict(zip(self.worker.value_ids, values))
        for guard in self.worker.runtime_guards:
            value = by_id.get(guard["value_id"])
            if value is None or not hasattr(value, "numel") or value.numel() != 1:
                return {"launched": False, "return_code": -4,
                        "reason": f"state index for {guard['operation_id']} is not a scalar tensor"}
            index = int(value.reshape(-1)[0].item())
            if not guard["lower"] <= index < guard["upper_exclusive"]:
                return {"launched": False, "return_code": -5,
                        "reason": f"state index {index} is outside [{guard['lower']}, {guard['upper_exclusive']})"}
        requested = self.worker.grid_ctas if grid_ctas is None else grid_ctas
        if not isinstance(requested, int) or requested <= 0:
            return {"launched": False, "return_code": -1, "reason": "worker grid must be positive"}
        if requested > self.resident_ctas:
            return {"launched": False, "return_code": -2,
                    "reason": "worker grid exceeds exact-entry cooperative residency",
                    "requested_ctas": requested, "resident_ctas": self.resident_ctas,
                    "runtime_target": self.runtime_target}
        if requested != self.worker.grid_ctas:
            return {"launched": False, "return_code": -3,
                    "reason": "worker grid differs from the selected static ownership plan",
                    "requested_ctas": requested, "selected_ctas": self.worker.grid_ctas,
                    "resident_ctas": self.resident_ctas,
                    "runtime_target": self.runtime_target}
        bindings = self.bindings_type()
        for index, value in enumerate(values):
            bindings.values[index] = int(value.data_ptr()) if hasattr(value, "data_ptr") else 0
        import ctypes
        code = self.library.mb3_launch_worker_entry(
            ctypes.byref(bindings), requested, stream
        )
        return {"launched": code == 0, "return_code": code,
                "reason": None if code == 0 else "cooperative worker launch failed",
                "requested_ctas": requested, "resident_ctas": self.resident_ctas,
                "runtime_target": self.runtime_target}


def open_worker_session(library_path: str | Path, worker: WorkerProgram,
                        profile: CudaTargetProfile,
                        admission: Mapping[str, Any]) -> WorkerEntrySession:
    """Check target identity once and retain the exact admitted entry for repeated calls."""
    return WorkerEntrySession(library_path, worker, profile, admission)


def inspect_worker_entry(library_path: str | Path, worker: WorkerProgram,
                         profile: CudaTargetProfile, compiled: CompileResult) -> dict[str, Any]:
    """Admit the actual compiled cooperative function and its complete worker ABI."""
    if compiled.return_code or not compiled.artifact_hash:
        raise EntryAdmissionError("worker entry has no successful compiled artifact")
    if worker.target_profile_key != profile.profile_key or worker.target != compiled.target:
        raise EntryAdmissionError("worker entry target/profile differs from the selected program")
    path = Path(library_path)
    binary_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if binary_hash != compiled.artifact_hash:
        raise EntryAdmissionError("compiled worker binary changed before admission")
    runtime = _runtime_device(profile)
    library, _ = _worker_library(path, len(worker.value_ids))
    import ctypes
    library.mb3_worker_program_hash.argtypes = []
    library.mb3_worker_program_hash.restype = ctypes.c_char_p
    library.mb3_worker_value_count.argtypes = []
    library.mb3_worker_value_count.restype = ctypes.c_int
    if library.mb3_worker_program_hash().decode() != worker.program_hash:
        raise EntryAdmissionError("compiled worker library belongs to a different program hash")
    if library.mb3_worker_value_count() != len(worker.value_ids):
        raise EntryAdmissionError("compiled worker library has a different binding ABI")
    expected_source_hash = hashlib.sha256(
        (worker.source + _worker_runtime_source(worker)).encode()
    ).hexdigest()
    if compiled.source_hash != expected_source_hash:
        raise EntryAdmissionError("compile record belongs to a different worker source")
    expected_artifact_key = hashlib.sha256(canonical_json({
        "worker_program_hash": worker.program_hash,
        "source_hash": compiled.source_hash,
        "target": worker.target,
        "profile_key": profile.profile_key,
        "compiler_version": compiled.compiler_version,
        "binding_abi": f"V3WorkerBindings/v1/{len(worker.value_ids)}",
    }).encode()).hexdigest()
    if compiled.artifact_key != expected_artifact_key:
        raise EntryAdmissionError("compile artifact key does not cover this worker program")
    resources = _EntryResources()
    status = library.mb3_entry_resources(worker.grid_ctas, ctypes.byref(resources))
    if status:
        raise EntryAdmissionError(f"CUDA worker resource query failed with code {status}")
    code_bytes, code_command, code_source = _entry_code_size(path, worker.target)
    expected_binary = int(str(profile.device_facts["compute_capability"].value).replace(".", ""))
    ptxas_regs = compiled.ptxas.get("registers_per_thread")
    registers_match = ptxas_regs is None or ptxas_regs == resources.registers_per_thread
    admitted = bool(resources.admitted and resources.binary_version == expected_binary and
                    resources.requested_ctas == worker.grid_ctas and
                    resources.block_threads == worker.block_threads and
                    resources.function_max_threads >= worker.block_threads and
                    registers_match and code_bytes is not None)
    reasons = []
    if not resources.cooperative_launch:
        reasons.append("selected device does not support cooperative launch")
    if resources.requested_ctas > resources.resident_ctas:
        reasons.append("worker grid exceeds compiled-entry cooperative residency")
    if resources.binary_version != expected_binary:
        reasons.append("compiled worker binary differs from the exact target")
    if not registers_match:
        reasons.append("ptxas and CUDA function register counts differ")
    if code_bytes is None:
        reasons.append(code_source)
    return {
        "schema_version": 1,
        "program_hash": worker.program_hash,
        "indexed_program_hash": worker.indexed_program_hash,
        "logical_plan_hash": worker.logical_plan_hash,
        "target_profile_key": profile.profile_key,
        "runtime_target": runtime,
        "compiled_artifact": {"sha256": binary_hash, "source_hash": compiled.source_hash,
                              "artifact_key": compiled.artifact_key,
                              "compile_command": list(compiled.command),
                              "compiler_version": compiled.compiler_version,
                              "artifact_bytes": compiled.artifact_bytes},
        "compiled_function": {
            "block_threads": resources.block_threads,
            "function_max_threads": resources.function_max_threads,
            "requested_grid_ctas": resources.requested_ctas,
            "active_ctas_per_sm": resources.active_ctas_per_sm,
            "cooperative_resident_ctas": resources.resident_ctas,
            "registers_per_thread": resources.registers_per_thread,
            "ptxas_registers_per_thread": ptxas_regs,
            "static_shared_bytes": resources.static_shared_bytes,
            "dynamic_shared_bytes": 0,
            "local_bytes": resources.local_bytes,
            "spill_store_bytes": compiled.ptxas.get("spill_store_bytes"),
            "spill_load_bytes": compiled.ptxas.get("spill_load_bytes"),
            "entry_code_bytes": code_bytes,
            "entry_code_query_command": code_command,
            "kernel_parameter_bytes": resources.parameter_bytes,
            "binary_version": resources.binary_version,
            "ptx_version": resources.ptx_version,
        },
        "launch_contract": {"api": "cudaLaunchCooperativeKernel",
                            "admitted": admitted,
                            "rejection_reasons": reasons},
        "progress_proof": worker.to_dict()["progress_proof"],
        "cross_cta": {"communication": len(worker.stages) > 1,
                       "protocol": "cooperative grid.sync between ordered stages"},
        "claim": "not_measured",
    }


def launch_worker_entry(library_path: str | Path, worker: WorkerProgram,
                        profile: CudaTargetProfile, values: Sequence[Any], *,
                        grid_ctas: int | None = None,
                        admission: Mapping[str, Any] | None = None,
                        session: WorkerEntrySession | None = None,
                        stream: int = 0) -> dict[str, Any]:
    """Validate state-index guards and launch only the admitted static grid."""
    if session is not None:
        if session.worker.program_hash != worker.program_hash or session.profile.profile_key != profile.profile_key:
            return {"launched": False, "return_code": -8,
                    "reason": "worker session does not match the selected program and target"}
        return session.launch(values, grid_ctas=grid_ctas, stream=stream)
    import ctypes
    if worker.target_profile_key != profile.profile_key:
        raise EntryAdmissionError("worker program belongs to a different target profile")
    if len(values) != len(worker.value_ids):
        raise ValueError(f"worker ABI expects {len(worker.value_ids)} values, got {len(values)}")
    by_id = dict(zip(worker.value_ids, values))
    for guard in worker.runtime_guards:
        value = by_id.get(guard["value_id"])
        if value is None or not hasattr(value, "numel") or value.numel() != 1:
            return {"launched": False, "return_code": -4,
                    "reason": f"state index for {guard['operation_id']} is not a scalar tensor"}
        index = int(value.reshape(-1)[0].item())
        if not guard["lower"] <= index < guard["upper_exclusive"]:
            return {"launched": False, "return_code": -5,
                    "reason": f"state index {index} is outside [{guard['lower']}, {guard['upper_exclusive']})"}
    runtime = _runtime_device(profile)
    library, bindings_type = _worker_library(library_path, len(worker.value_ids))
    library.mb3_worker_program_hash.argtypes = []
    library.mb3_worker_program_hash.restype = ctypes.c_char_p
    library.mb3_worker_value_count.argtypes = []
    library.mb3_worker_value_count.restype = ctypes.c_int
    if library.mb3_worker_program_hash().decode() != worker.program_hash:
        return {"launched": False, "return_code": -6,
                "reason": "compiled worker library belongs to a different program hash"}
    if library.mb3_worker_value_count() != len(worker.value_ids):
        return {"launched": False, "return_code": -7,
                "reason": "compiled worker library has a different binding ABI"}
    requested = worker.grid_ctas if grid_ctas is None else grid_ctas
    if not isinstance(requested, int) or requested <= 0:
        return {"launched": False, "return_code": -1, "reason": "worker grid must be positive"}
    if admission is not None:
        if (admission.get("program_hash") != worker.program_hash or
                admission.get("target_profile_key") != profile.profile_key or
                not admission.get("launch_contract", {}).get("admitted")):
            return {"launched": False, "return_code": -8,
                    "reason": "worker admission does not match the selected program and target"}
        resident_ctas = admission["compiled_function"]["cooperative_resident_ctas"]
        admitted = requested <= resident_ctas
    else:
        resources = _EntryResources()
        status = library.mb3_entry_resources(requested, ctypes.byref(resources))
        if status:
            raise EntryAdmissionError(f"CUDA worker resource query failed with code {status}")
        resident_ctas = resources.resident_ctas
        admitted = bool(resources.admitted)
    if not admitted:
        return {"launched": False, "return_code": -2,
                "reason": "worker grid exceeds exact-entry cooperative residency",
                "requested_ctas": requested,
                "resident_ctas": resident_ctas,
                "runtime_target": runtime}
    if requested != worker.grid_ctas:
        return {"launched": False, "return_code": -3,
                "reason": "worker grid differs from the selected static ownership plan",
                "requested_ctas": requested,
                "selected_ctas": worker.grid_ctas,
                "resident_ctas": resident_ctas,
                "runtime_target": runtime}
    pointers = (ctypes.c_void_p * len(values))(*[
        int(value.data_ptr()) if hasattr(value, "data_ptr") else 0 for value in values
    ])
    bindings = bindings_type(pointers)
    code = library.mb3_launch_worker_entry(ctypes.byref(bindings), requested, stream)
    return {"launched": code == 0, "return_code": code,
            "reason": None if code == 0 else "cooperative worker launch failed",
            "requested_ctas": requested,
            "resident_ctas": resident_ctas,
            "runtime_target": runtime}


def bind_worker_values(program: IndexedTensorProgram,
                       user_args: Sequence[Any]) -> tuple[Any, ...]:
    """Bind inputs and allocate fresh intermediates without executing the reference."""
    import torch

    source = program.source_program
    nodes = {node.name: node for node in source.graph_module.graph.nodes}
    placeholders = [node.name for node in source.graph_module.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(user_args):
        raise ValueError(f"worker graph expects {len(placeholders)} inputs, got {len(user_args)}")
    by_node = dict(zip(placeholders, user_args))
    operations_by_output = {output: operation for operation in program.operations
                            for output in operation.outputs}
    dtype_by_name = {
        "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
        "bool": torch.bool, "int32": torch.int32, "int64": torch.int64,
    }
    bound: dict[str, Any] = {}
    for value in program.values:
        node = nodes.get(value.fx_node)
        if node is None:
            bound[value.value_id] = None
        elif node.op == "placeholder":
            bound[value.value_id] = by_node[value.fx_node]
        elif node.op == "get_attr":
            attr: Any = source.graph_module
            for part in str(node.target).split("."):
                attr = getattr(attr, part)
            bound[value.value_id] = attr
        elif value.shape and value.dtype:
            operation = operations_by_output.get(value.value_id)
            if value.alias_kind == "view" and operation and len(operation.inputs) == 1:
                base = bound.get(operation.inputs[0])
                if not isinstance(base, torch.Tensor):
                    raise WorkerProgramError(f"view {value.fx_node} has no bound tensor base")
                offset = value.storage_offset
                bound[value.value_id] = torch.as_strided(
                    base, tuple(value.shape), tuple(value.strides),
                    storage_offset=base.storage_offset() if offset is None else offset,
                )
            else:
                if value.dtype not in dtype_by_name:
                    raise WorkerProgramError(f"{value.fx_node} has unsupported binding dtype {value.dtype!r}")
                tensor_input = next((item for item in user_args if isinstance(item, torch.Tensor)), None)
                if tensor_input is None:
                    raise WorkerProgramError("worker binding has no tensor input device")
                bound[value.value_id] = torch.empty_strided(
                    tuple(value.shape), tuple(value.strides), dtype=dtype_by_name[value.dtype],
                    device=tensor_input.device,
                )
        else:
            bound[value.value_id] = None
    return tuple(bound[value.value_id] for value in program.values)


__all__ = ["WorkerEntrySession", "WorkerProgram", "WorkerProgramError", "WorkerStage", "bind_worker_values",
           "compile_worker_entry", "inspect_worker_entry", "launch_worker_entry",
           "lower_worker_program", "open_worker_session"]
