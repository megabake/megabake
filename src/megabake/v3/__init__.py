"""Target-neutral V3 contracts and evidence utilities.

The V3 package is deliberately independent from the legacy CUDA runtime.  Importing
these records must be safe on a CPU-only machine and must not discover a device or a
toolchain.
"""

from .contracts import (
    BenchmarkCell,
    ExceptionalValuePolicy,
    NumericalPolicy,
    ToleranceSpec,
    WorkloadSpec,
)

from .diagnostics import (
    BenchmarkEvidence,
    ClaimStatus,
    DiagnosticCode,
    DiagnosticRecord,
    EvidenceVector,
    TaskHandoff,
)

__all__ = [
    "BenchmarkCell",
    "ExceptionalValuePolicy",
    "NumericalPolicy",
    "ToleranceSpec",
    "WorkloadSpec",
    "BenchmarkEvidence",
    "ClaimStatus",
    "DiagnosticCode",
    "DiagnosticRecord",
    "EvidenceVector",
    "TaskHandoff",
]
