"""Selected-source CUDA compilation with explicit target and resource evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ...contracts import canonical_json
from .bodies.registry import BodyTacticSpec
from .physical import PhysicalCandidate
from .profile import CudaTargetProfile
from .source import emit_entry_source


@dataclass(frozen=True)
class CompileResult:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    compiler_version: str
    target: str
    source_hash: str
    artifact_path: str
    artifact_hash: str | None
    artifact_bytes: int | None
    ptxas: dict[str, int | None]
    artifact_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command), "return_code": self.return_code,
            "stdout": self.stdout, "stderr": self.stderr,
            "compiler_version": self.compiler_version, "target": self.target,
            "source_hash": self.source_hash, "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash, "artifact_bytes": self.artifact_bytes,
            "ptxas": dict(self.ptxas), "artifact_key": self.artifact_key,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compiler_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"cannot query nvcc version: {result.stderr.strip()}")
    return (result.stdout or result.stderr).strip()


def _ptxas(stderr: str) -> dict[str, int | None]:
    frame = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
        stderr,
    )
    registers = re.search(r"Used (\d+) registers", stderr)
    return {
        "registers_per_thread": int(registers.group(1)) if registers else None,
        "stack_frame_bytes": int(frame.group(1)) if frame else None,
        "spill_store_bytes": int(frame.group(2)) if frame else None,
        "spill_load_bytes": int(frame.group(3)) if frame else None,
    }


def artifact_key_for(candidate: PhysicalCandidate, tactic: BodyTacticSpec,
                     profile: CudaTargetProfile, source_hash: str,
                     compiler_version: str) -> str:
    return _sha256(canonical_json({
        "candidate": candidate.to_dict(), "body_tactic": tactic.to_dict(),
        "entry_source_hash": source_hash, "target": candidate.target,
        "profile_key": profile.profile_key, "compiler_version": compiler_version,
        "provider": tactic.provider, "provider_version": tactic.provider_version,
        "body_abi": 1, "binding_abi": "Mb3Contraction-by-value-v1",
    }).encode())


def compile_entry(candidate: PhysicalCandidate, tactic: BodyTacticSpec,
                  profile: CudaTargetProfile, output: str | Path, *,
                  nvcc: str = "nvcc", shared_library: bool = False) -> CompileResult:
    """Compile the exact selected entry for the profile's declared baseline target."""
    if profile.profile_key != candidate.target_profile_key:
        raise ValueError("candidate belongs to a different CUDA target profile")
    if candidate.indexed_program_hash != tactic.indexed_program_hash:
        raise ValueError("candidate and tactic indexed semantics differ")
    if candidate.target != tactic.target or profile.supports_target(candidate.target) is not True:
        raise ValueError("candidate target is not exact-target legal and compiler-supported")
    executable = shutil.which(nvcc) if "/" not in nvcc else nvcc
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError(f"CUDA compiler {nvcc!r} is unavailable")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = output.with_suffix(".cu")
    source = emit_entry_source(candidate, tactic, runtime_api=shared_library)
    source_bytes = source.encode()
    source_path.write_bytes(source_bytes)
    source_hash = _sha256(source_bytes)
    compiler_version = _compiler_version(executable)
    artifact_key = artifact_key_for(candidate, tactic, profile, source_hash,
                                    compiler_version)
    command = [executable, "-std=c++17", f"-arch={candidate.target}", "-Xptxas=-v",
               "-I", str(Path(__file__).resolve().parents[5] / "src/cuda/v3")]
    if shared_library:
        command += ["-shared", "-Xcompiler", "-fPIC"]
    else:
        command += ["-cubin"]
    command += [str(source_path), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    artifact_path = str(output)
    artifact_hash = _sha256(output.read_bytes()) if result.returncode == 0 and output.is_file() else None
    artifact_bytes = output.stat().st_size if artifact_hash else None
    return CompileResult(
        tuple(command), result.returncode, result.stdout, result.stderr,
        compiler_version, candidate.target, source_hash, artifact_path,
        artifact_hash, artifact_bytes, _ptxas(result.stderr), artifact_key,
    )


__all__ = ["CompileResult", "artifact_key_for", "compile_entry"]
