"""Inspect and admit the exact compiled CUDA entry for one target profile."""

from __future__ import annotations

import ctypes
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .compile import CompileResult
from .physical import PhysicalCandidate
from .profile import CudaTargetProfile


class EntryAdmissionError(RuntimeError):
    pass


class Mb3Contraction(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_void_p), ("weight", ctypes.c_void_p),
        ("bias", ctypes.c_void_p), ("output", ctypes.c_void_p),
        ("m", ctypes.c_int64), ("n", ctypes.c_int64), ("k", ctypes.c_int64),
        ("x_m_stride", ctypes.c_int64), ("x_k_stride", ctypes.c_int64),
        ("weight_n_stride", ctypes.c_int64), ("weight_k_stride", ctypes.c_int64),
        ("alpha", ctypes.c_float), ("beta", ctypes.c_float),
        ("input_dtype", ctypes.c_int),
    ]


class _EntryResources(ctypes.Structure):
    _fields_ = [
        ("device_sms", ctypes.c_int), ("cooperative_launch", ctypes.c_int),
        ("block_threads", ctypes.c_int), ("requested_ctas", ctypes.c_int),
        ("active_ctas_per_sm", ctypes.c_int), ("resident_ctas", ctypes.c_int),
        ("registers_per_thread", ctypes.c_int), ("static_shared_bytes", ctypes.c_size_t),
        ("local_bytes", ctypes.c_size_t), ("function_max_threads", ctypes.c_int),
        ("max_dynamic_shared_bytes", ctypes.c_int), ("constant_bytes", ctypes.c_size_t),
        ("binary_version", ctypes.c_int), ("ptx_version", ctypes.c_int),
        ("parameter_bytes", ctypes.c_size_t), ("admitted", ctypes.c_int),
    ]


def _runtime_device(profile: CudaTargetProfile) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise EntryAdmissionError("PyTorch is required to verify the selected runtime device") from exc
    if not torch.cuda.is_available():
        raise EntryAdmissionError("no CUDA device is available for entry admission")
    try:
        torch.cuda.set_device(profile.device_index)
        properties = torch.cuda.get_device_properties(profile.device_index)
    except Exception as exc:
        raise EntryAdmissionError(f"cannot select profiled CUDA device: {exc}") from exc
    actual = {
        "device_index": profile.device_index,
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "")),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "visible_sms": properties.multi_processor_count,
    }
    expected = {
        "device_name": profile.device_name,
        "device_uuid": profile.device_uuid,
        "compute_capability": profile.device_facts["compute_capability"].value,
        "visible_sms": profile.device_facts["visible_sms"].value,
    }
    mismatches = [f"{key}: profile={value!r}, runtime={actual[key]!r}"
                  for key, value in expected.items() if value is not None and actual[key] != value]
    if not profile.device_uuid or not actual["device_uuid"]:
        mismatches.append("device UUID is unknown")
    if mismatches:
        raise EntryAdmissionError("runtime target does not match profile: " + "; ".join(mismatches))
    return actual


def _library(path: str | Path) -> ctypes.CDLL:
    result = ctypes.CDLL(str(path))
    result.mb3_entry_resources.argtypes = [ctypes.c_int, ctypes.POINTER(_EntryResources)]
    result.mb3_entry_resources.restype = ctypes.c_int
    result.mb3_launch_selected_entry.argtypes = [ctypes.POINTER(Mb3Contraction), ctypes.c_int,
                                                 ctypes.c_size_t]
    result.mb3_launch_selected_entry.restype = ctypes.c_int
    return result


def _function_resources(library: ctypes.CDLL, grid_ctas: int) -> _EntryResources:
    result = _EntryResources()
    status = library.mb3_entry_resources(grid_ctas, ctypes.byref(result))
    if status:
        raise EntryAdmissionError(f"CUDA function resource query failed with code {status}")
    return result


def _entry_code_size(path: str | Path, target: str) -> tuple[int | None, list[str], str]:
    executable = shutil.which("cuobjdump")
    if executable is None:
        return None, [], "cuobjdump is unavailable"
    command = [executable, "--dump-elf", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        return None, command, result.stderr.strip() or f"cuobjdump exited {result.returncode}"
    for chunk in result.stdout.split("Fatbin elf code:"):
        if not re.search(rf"\barch\s*=\s*{re.escape(target)}\b", chunk):
            continue
        for line in chunk.splitlines():
            if "megabake_v3_entry" not in line:
                continue
            fields = line.split()
            if len(fields) >= 7 and fields[0].startswith("0x") and fields[2].startswith("0x"):
                return int(fields[2], 16), command, "cuobjdump ELF symbol size for megabake_v3_entry"
    return None, command, f"no {target} ELF symbol size for megabake_v3_entry"


def inspect_entry(library_path: str | Path, candidate: PhysicalCandidate,
                  profile: CudaTargetProfile, compiled: CompileResult) -> dict[str, Any]:
    """Query attributes and cooperative capacity from the compiled function itself."""
    if compiled.return_code or not compiled.artifact_hash:
        raise EntryAdmissionError("selected entry has no successful compiled artifact")
    artifact = Path(library_path)
    binary_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if binary_hash != compiled.artifact_hash:
        raise EntryAdmissionError("compiled binary hash changed before admission")
    if candidate.target_profile_key != profile.profile_key:
        raise EntryAdmissionError("candidate profile key differs from selected target")
    runtime = _runtime_device(profile)
    library = _library(artifact)
    resources = _function_resources(library, candidate.grid_ctas)
    code_bytes, code_command, code_source = _entry_code_size(artifact, candidate.target)
    expected_cc = str(profile.device_facts["compute_capability"].value).replace(".", "")
    expected_binary_version = int(expected_cc)
    spills = compiled.ptxas
    register_match = (spills.get("registers_per_thread") is None or
                      spills["registers_per_thread"] == resources.registers_per_thread)
    target_match = resources.binary_version == expected_binary_version
    grid_fits = resources.requested_ctas <= resources.resident_ctas
    block_fits = (resources.block_threads == candidate.block_threads and
                  resources.function_max_threads >= candidate.block_threads)
    code_known = code_bytes is not None
    admitted = bool(resources.admitted and target_match and grid_fits and block_fits and
                    register_match and code_known)
    reasons = []
    if not resources.cooperative_launch:
        reasons.append("selected device does not support cooperative launch")
    if not grid_fits:
        reasons.append("selected grid exceeds compiled-entry cooperative residency")
    if not block_fits:
        reasons.append("compiled function block limit does not admit the selected block")
    if not target_match:
        reasons.append("compiled function binary version differs from the exact target")
    if not register_match:
        reasons.append("ptxas and cudaFuncGetAttributes register counts differ")
    if not code_known:
        reasons.append(code_source)
    return {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "indexed_program_hash": candidate.indexed_program_hash,
        "logical_plan_hash": candidate.logical_plan_hash,
        "target_profile_key": profile.profile_key,
        "target": candidate.target,
        "runtime_target": runtime,
        "compiled_artifact": {
            "path": str(artifact), "sha256": binary_hash,
            "artifact_key": compiled.artifact_key,
            "source_hash": compiled.source_hash,
            "compiler_version": compiled.compiler_version,
            "compile_command": list(compiled.command),
            "binary_bytes": compiled.artifact_bytes,
        },
        "compiled_function": {
            "block_threads": resources.block_threads,
            "function_max_threads": resources.function_max_threads,
            "requested_grid_ctas": resources.requested_ctas,
            "active_ctas_per_sm": resources.active_ctas_per_sm,
            "cooperative_resident_ctas": resources.resident_ctas,
            "registers_per_thread": resources.registers_per_thread,
            "ptxas_registers_per_thread": spills.get("registers_per_thread"),
            "static_shared_bytes": resources.static_shared_bytes,
            "dynamic_shared_bytes": 0,
            "max_dynamic_shared_bytes": resources.max_dynamic_shared_bytes,
            "local_bytes": resources.local_bytes,
            "stack_frame_bytes": spills.get("stack_frame_bytes"),
            "spill_store_bytes": spills.get("spill_store_bytes"),
            "spill_load_bytes": spills.get("spill_load_bytes"),
            "entry_code_bytes": code_bytes,
            "entry_code_bytes_source": code_source,
            "entry_code_query_command": code_command,
            "constant_bytes": resources.constant_bytes,
            "kernel_parameter_bytes": resources.parameter_bytes,
            "binary_version": resources.binary_version,
            "ptx_version": resources.ptx_version,
        },
        "launch_contract": {
            "api": "cudaLaunchCooperativeKernel",
            "required_launch_attributes": [],
            "cooperative_launch_supported": bool(resources.cooperative_launch),
            "admitted": admitted,
            "rejection_reasons": reasons,
        },
        "cross_sm": {
            "inter_cta_communication": False,
            "ownership": "disjoint flat output intervals",
            "status": "no cross-SM publication protocol is required by this candidate",
        },
        "claim": "not_measured",
    }


def launch_admitted(library_path: str | Path, candidate: PhysicalCandidate,
                    profile: CudaTargetProfile, problem: Mb3Contraction, *,
                    grid_ctas: int | None = None, stream: int = 0) -> dict[str, Any]:
    """Launch only the exact admitted candidate; return a prelaunch reject otherwise."""
    runtime = _runtime_device(profile)
    library = _library(library_path)
    requested = candidate.grid_ctas if grid_ctas is None else grid_ctas
    if requested <= 0:
        return {"launched": False, "return_code": -1, "reason": "grid must be positive"}
    resources = _function_resources(library, requested)
    if requested != candidate.grid_ctas and resources.admitted:
        return {"launched": False, "return_code": -3,
                "reason": "launch grid differs from the selected ownership plan",
                "runtime_target": runtime}
    if not resources.admitted:
        return {"launched": False, "return_code": -2,
                "reason": "grid exceeds the compiled entry's cooperative resident capacity",
                "resident_ctas": resources.resident_ctas,
                "requested_ctas": requested, "runtime_target": runtime}
    status = library.mb3_launch_selected_entry(ctypes.byref(problem), requested, stream)
    return {"launched": status == 0, "return_code": status,
            "reason": None if status == 0 else "cooperative CUDA launch failed",
            "requested_ctas": requested, "resident_ctas": resources.resident_ctas,
            "runtime_target": runtime}


def contraction_problem(x: Any, weight: Any, bias: Any, output: Any, *,
                        alpha: float, beta: float) -> Mb3Contraction:
    """Bind the exact contiguous LINEAR_TINY addmm pointer ABI."""
    if tuple(x.shape) != (1, 33) or tuple(weight.shape) != (17, 33):
        raise ValueError("initial entry binding supports LINEAR_TINY shapes only")
    if tuple(bias.shape) != (17,) or tuple(output.shape) != (1, 17):
        raise ValueError("initial entry binding requires the exact AddMM output/bias shapes")
    if any(not tensor.is_cuda for tensor in (x, weight, bias, output)):
        raise ValueError("entry bindings must be CUDA tensors")
    return Mb3Contraction(
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
        1, 17, 33, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        alpha, beta, 0,
    )


__all__ = ["EntryAdmissionError", "Mb3Contraction", "contraction_problem",
           "inspect_entry", "launch_admitted"]
