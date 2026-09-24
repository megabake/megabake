"""Stable diagnostics and JSON-safe task evidence for V3.

The records in this module are deliberately data-only.  They can be written to
an artifact directory and reviewed by another process without importing CUDA,
executing a callable, or unpickling arbitrary Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping

from .contracts import CONTRACT_SCHEMA_VERSION, ContractError, canonical_json


class DiagnosticCode(str, Enum):
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"
    MISSING_FACTS = "missing_facts"
    BODY_INCOMPATIBILITY = "body_incompatibility"
    INVALID_EVENT = "invalid_event"
    INVALID_STORAGE = "invalid_storage"
    INVALID_PROGRESS = "invalid_progress"
    MISSING_TOOLCHAIN = "missing_toolchain"
    MISSING_DEVICE = "missing_device"
    FAILED_NUMERICAL_GATE = "failed_numerical_gate"
    INVALID_SCHEMA = "invalid_schema"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ImplementationStatus(str, Enum):
    NOT_STARTED = "not_started"
    PARTIAL = "partial"
    PATCH_READY = "patch_ready"
    REVIEWED = "reviewed"


class CPUValidation(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CudaCompilation(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class GPUCorrectness(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class GPUPerformance(str, Enum):
    NOT_MEASURED = "not_measured"
    MEASURED_WIN = "measured_win"
    MEASURED_LOSS = "measured_loss"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class TaskDisposition(str, Enum):
    CONTINUE = "continue"
    REVISE = "revise"
    REJECTED_CANDIDATE = "rejected_candidate"
    BLOCKED_EXTERNAL = "blocked_external"


class ClaimStatus(str, Enum):
    NOT_MEASURED = "not_measured"
    STRICT_WIN = "strict_win"
    STRICT_LOSS = "strict_loss"
    INCORRECT = "incorrect"
    UNSUPPORTED = "unsupported"


def _enum_value(value: Any, enum_type: type[Enum], field_name: str) -> str:
    value = value.value if isinstance(value, Enum) else value
    if not isinstance(value, str):
        raise ContractError(f"{field_name} must be a string or {enum_type.__name__}")
    try:
        return enum_type(value).value
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ContractError(
            f"{field_name}={value!r} is invalid; expected one of {choices}"
        ) from exc


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _schema(value: Mapping[str, Any], record_name: str) -> int:
    if not isinstance(value, Mapping):
        raise ContractError(f"{record_name} requires a JSON object")
    version = value.get("schema_version")
    if version != CONTRACT_SCHEMA_VERSION:
        raise ContractError(
            f"unsupported {record_name} schema_version={version!r}; "
            f"expected {CONTRACT_SCHEMA_VERSION}"
        )
    return version


@dataclass(frozen=True)
class DiagnosticRecord:
    code: DiagnosticCode | str
    message: str
    severity: DiagnosticSeverity | str = DiagnosticSeverity.ERROR
    node_id: str | None = None
    action_id: str | None = None
    artifact_hash: str | None = None
    details: Mapping[str, Any] = ()
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum_value(self.code, DiagnosticCode, "code"))
        object.__setattr__(self, "severity", _enum_value(self.severity, DiagnosticSeverity, "severity"))
        object.__setattr__(self, "message", _text(self.message, "message"))
        for name in ("node_id", "action_id", "artifact_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        details = {} if self.details == () else self.details
        if not isinstance(details, Mapping):
            raise ContractError("diagnostic details must be a JSON mapping")
        # Validate that details are JSON-safe at construction time.
        canonical_json(details)
        object.__setattr__(self, "details", dict(details))
        _schema({"schema_version": self.schema_version}, type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "node_id": self.node_id,
            "action_id": self.action_id,
            "artifact_hash": self.artifact_hash,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiagnosticRecord":
        _schema(value, cls.__name__)
        return cls(**{key: value.get(key) for key in (
            "code", "message", "severity", "node_id", "action_id",
            "artifact_hash", "details",
        )}, schema_version=value["schema_version"])


@dataclass(frozen=True)
class CommandResult:
    command: str
    return_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float | None = None
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _text(self.command, "command"))
        if self.return_code is not None and not isinstance(self.return_code, int):
            raise ContractError("return_code must be an integer or null")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ContractError("command stdout/stderr must be strings")
        if self.duration_seconds is not None and (
            not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ContractError("duration_seconds must be finite and non-negative")
        _schema({"schema_version": self.schema_version}, type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommandResult":
        _schema(value, cls.__name__)
        return cls(
            command=value["command"],
            return_code=value.get("return_code"),
            stdout=value.get("stdout", ""),
            stderr=value.get("stderr", ""),
            duration_seconds=value.get("duration_seconds"),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class EvidenceVector:
    implementation: ImplementationStatus | str = ImplementationStatus.NOT_STARTED
    cpu_validation: CPUValidation | str = CPUValidation.NOT_RUN
    cuda_compilation: CudaCompilation | str = CudaCompilation.NOT_RUN
    gpu_correctness: GPUCorrectness | str = GPUCorrectness.NOT_RUN
    gpu_performance: GPUPerformance | str = GPUPerformance.NOT_MEASURED
    disposition: TaskDisposition | str = TaskDisposition.CONTINUE
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        enum_fields = (
            ("implementation", ImplementationStatus),
            ("cpu_validation", CPUValidation),
            ("cuda_compilation", CudaCompilation),
            ("gpu_correctness", GPUCorrectness),
            ("gpu_performance", GPUPerformance),
            ("disposition", TaskDisposition),
        )
        for name, enum_type in enum_fields:
            object.__setattr__(self, name, _enum_value(getattr(self, name), enum_type, name))
        _schema({"schema_version": self.schema_version}, type(self).__name__)
        if self.gpu_performance == GPUPerformance.MEASURED_WIN.value:
            if self.cuda_compilation != CudaCompilation.PASS.value:
                raise ContractError("a measured GPU win requires cuda_compilation=pass")
            if self.gpu_correctness != GPUCorrectness.PASS.value:
                raise ContractError("a measured GPU win requires gpu_correctness=pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "implementation": self.implementation,
            "cpu_validation": self.cpu_validation,
            "cuda_compilation": self.cuda_compilation,
            "gpu_correctness": self.gpu_correctness,
            "gpu_performance": self.gpu_performance,
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceVector":
        _schema(value, cls.__name__)
        return cls(
            **{key: value[key] for key in (
                "implementation", "cpu_validation", "cuda_compilation",
                "gpu_correctness", "gpu_performance", "disposition",
            )},
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Evidence needed before a benchmark claim can be called strict."""

    baseline_id: str
    target_id: str
    raw_samples: tuple[float, ...]
    target_profile_key: str | None = None
    selection_validation_split: str = "unreported"
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_id", _text(self.baseline_id, "baseline_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        samples = tuple(float(item) for item in self.raw_samples)
        if any(not math.isfinite(item) or item <= 0 for item in samples):
            raise ContractError("raw timing samples must be finite and positive")
        if self.target_profile_key is not None:
            object.__setattr__(self, "target_profile_key", _text(self.target_profile_key, "target_profile_key"))
        object.__setattr__(self, "raw_samples", samples)
        object.__setattr__(
            self, "selection_validation_split", _text(self.selection_validation_split, "selection_validation_split")
        )
        _schema({"schema_version": self.schema_version}, type(self).__name__)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_id": self.baseline_id,
            "target_id": self.target_id,
            "raw_samples": list(self.raw_samples),
            "target_profile_key": self.target_profile_key,
            "selection_validation_split": self.selection_validation_split,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkEvidence":
        _schema(value, cls.__name__)
        return cls(
            baseline_id=value["baseline_id"],
            target_id=value["target_id"],
            raw_samples=tuple(value["raw_samples"]),
            target_profile_key=value.get("target_profile_key"),
            selection_validation_split=value.get("selection_validation_split", "unreported"),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class TaskHandoff:
    """Portable completion/evidence record for one atomic implementation card."""

    task_id: str
    task_revision: str
    dependency_revisions: Mapping[str, str]
    changed_files: tuple[str, ...]
    commands: tuple[CommandResult, ...]
    evidence: EvidenceVector
    remaining_risks: tuple[str, ...] = ()
    next_eligible_tasks: tuple[str, ...] = ()
    diagnostics: tuple[DiagnosticRecord, ...] = ()
    artifact_hashes: Mapping[str, str] = ()
    claim: ClaimStatus | str = ClaimStatus.NOT_MEASURED
    benchmark: BenchmarkEvidence | None = None
    notes: str = ""
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "task_revision", _text(self.task_revision, "task_revision"))
        dependencies = {} if self.dependency_revisions == () else self.dependency_revisions
        artifacts = {} if self.artifact_hashes == () else self.artifact_hashes
        if not isinstance(dependencies, Mapping) or not isinstance(artifacts, Mapping):
            raise ContractError("dependency_revisions and artifact_hashes must be mappings")
        object.__setattr__(self, "dependency_revisions", dict(dependencies))
        object.__setattr__(self, "artifact_hashes", dict(artifacts))
        object.__setattr__(self, "changed_files", tuple(_text(item, "changed_files item") for item in self.changed_files))
        object.__setattr__(self, "remaining_risks", tuple(_text(item, "remaining_risks item") for item in self.remaining_risks))
        object.__setattr__(self, "next_eligible_tasks", tuple(_text(item, "next_eligible_tasks item") for item in self.next_eligible_tasks))
        object.__setattr__(self, "commands", tuple(
            item if isinstance(item, CommandResult) else CommandResult.from_dict(item)
            for item in self.commands
        ))
        object.__setattr__(self, "diagnostics", tuple(
            item if isinstance(item, DiagnosticRecord) else DiagnosticRecord.from_dict(item)
            for item in self.diagnostics
        ))
        if not isinstance(self.evidence, EvidenceVector):
            object.__setattr__(self, "evidence", EvidenceVector.from_dict(self.evidence))
        object.__setattr__(self, "claim", _enum_value(self.claim, ClaimStatus, "claim"))
        if self.benchmark is not None and not isinstance(self.benchmark, BenchmarkEvidence):
            object.__setattr__(self, "benchmark", BenchmarkEvidence.from_dict(self.benchmark))
        for key, value in self.dependency_revisions.items():
            _text(key, "dependency revision key")
            _text(value, "dependency revision")
        for key, value in self.artifact_hashes.items():
            _text(key, "artifact key")
            _text(value, "artifact hash")
        object.__setattr__(self, "notes", self.notes if isinstance(self.notes, str) else str(self.notes))
        _schema({"schema_version": self.schema_version}, type(self).__name__)
        if self.claim == ClaimStatus.STRICT_WIN.value:
            if self.benchmark is None or not self.benchmark.raw_samples:
                raise ContractError("a strict win requires raw benchmark samples")
            if self.evidence.gpu_performance != GPUPerformance.MEASURED_WIN.value:
                raise ContractError("a strict win requires measured GPU performance evidence")
            if self.evidence.gpu_correctness != GPUCorrectness.PASS.value:
                raise ContractError("a strict win requires passed GPU correctness evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "task_revision": self.task_revision,
            "dependency_revisions": dict(self.dependency_revisions),
            "changed_files": list(self.changed_files),
            "commands": [command.to_dict() for command in self.commands],
            "evidence": self.evidence.to_dict(),
            "remaining_risks": list(self.remaining_risks),
            "next_eligible_tasks": list(self.next_eligible_tasks),
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "artifact_hashes": dict(self.artifact_hashes),
            "claim": self.claim,
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
            "notes": self.notes,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        # canonical_json performs the strict JSON-safe validation and preserves
        # the invariant that a missing GPU record cannot be emitted as a win.
        result = canonical_json(self.to_dict())
        if indent is None:
            return result
        return json.dumps(json.loads(result), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskHandoff":
        _schema(value, cls.__name__)
        return cls(
            task_id=value["task_id"],
            task_revision=value["task_revision"],
            dependency_revisions=value["dependency_revisions"],
            changed_files=tuple(value["changed_files"]),
            commands=tuple(CommandResult.from_dict(item) for item in value["commands"]),
            evidence=EvidenceVector.from_dict(value["evidence"]),
            remaining_risks=tuple(value.get("remaining_risks", ())),
            next_eligible_tasks=tuple(value.get("next_eligible_tasks", ())),
            diagnostics=tuple(DiagnosticRecord.from_dict(item) for item in value.get("diagnostics", ())),
            artifact_hashes=value.get("artifact_hashes", {}),
            claim=value.get("claim", ClaimStatus.NOT_MEASURED.value),
            benchmark=BenchmarkEvidence.from_dict(value["benchmark"]) if value.get("benchmark") else None,
            notes=value.get("notes", ""),
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "TaskHandoff":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid task handoff JSON: {exc}") from exc
        try:
            return cls.from_dict(value)
        except (KeyError, TypeError, ContractError) as exc:
            raise ContractError(f"invalid task handoff record: {exc}") from exc

    def render(self) -> str:
        lines = [
            f"{self.task_id} ({self.task_revision})",
            f"implementation={self.evidence.implementation}; cpu={self.evidence.cpu_validation}; "
            f"cuda={self.evidence.cuda_compilation}; gpu_correctness={self.evidence.gpu_correctness}; "
            f"gpu_performance={self.evidence.gpu_performance}; disposition={self.evidence.disposition}",
        ]
        if self.diagnostics:
            lines.append("diagnostics:")
            lines.extend(
                f"  - {item.code}: {item.message}" for item in self.diagnostics
            )
        if self.remaining_risks:
            lines.append("remaining risks:")
            lines.extend(f"  - {item}" for item in self.remaining_risks)
        return "\n".join(lines)


__all__ = [
    "BenchmarkEvidence",
    "CPUValidation",
    "ClaimStatus",
    "CommandResult",
    "CudaCompilation",
    "DiagnosticCode",
    "DiagnosticRecord",
    "DiagnosticSeverity",
    "EvidenceVector",
    "GPUCorrectness",
    "GPUPerformance",
    "ImplementationStatus",
    "TaskDisposition",
    "TaskHandoff",
]
