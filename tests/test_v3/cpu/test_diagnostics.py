"""MB3-006: diagnostics distinguish unsupported, failed, and unmeasured work."""

from __future__ import annotations

import pytest

from megabake.v3.diagnostics import (
    CPUValidation,
    ClaimStatus,
    CudaCompilation,
    DiagnosticCode,
    DiagnosticRecord,
    EvidenceVector,
    GPUCorrectness,
    GPUPerformance,
    ImplementationStatus,
    TaskDisposition,
    TaskHandoff,
)


def _handoff() -> TaskHandoff:
    return TaskHandoff(
        task_id="MB3-006",
        task_revision="test-revision",
        dependency_revisions={"MB3-004": "contract-hash"},
        changed_files=("src/megabake/v3/diagnostics.py",),
        commands=(),
        evidence=EvidenceVector(
            implementation=ImplementationStatus.PATCH_READY,
            cpu_validation=CPUValidation.PASS,
            cuda_compilation=CudaCompilation.NOT_APPLICABLE,
            gpu_correctness=GPUCorrectness.NOT_RUN,
            gpu_performance=GPUPerformance.NOT_MEASURED,
            disposition=TaskDisposition.CONTINUE,
        ),
        diagnostics=(DiagnosticRecord(
            code=DiagnosticCode.MISSING_DEVICE,
            message="GPU gate was not run on this CPU lane",
            node_id="node-3",
        ),),
        remaining_risks=("GPU evidence remains unavailable",),
        next_eligible_tasks=("MB3-007",),
        claim=ClaimStatus.NOT_MEASURED,
    )


def test_handoff_round_trip_and_human_rendering() -> None:
    handoff = _handoff()
    restored = TaskHandoff.from_json(handoff.to_json())
    assert restored.to_dict() == handoff.to_dict()
    assert "MB3-006" in restored.render()
    assert "missing_device" in restored.render()


def test_measured_win_requires_all_gpu_evidence() -> None:
    with pytest.raises(ValueError, match="measured GPU win"):
        EvidenceVector(gpu_performance=GPUPerformance.MEASURED_WIN)


def test_strict_win_requires_raw_samples_and_gpu_pass() -> None:
    base = _handoff().to_dict()
    base["claim"] = "strict_win"
    with pytest.raises(ValueError, match="strict win"):
        TaskHandoff.from_dict(base)


def test_corrupt_schema_is_actionable() -> None:
    payload = _handoff().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        TaskHandoff.from_dict(payload)
