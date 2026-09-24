"""MB3-003: optional lanes skip visibly while CPU remains unconditional."""

from __future__ import annotations

import os
import subprocess
import sys


def _probe(env_name: str) -> subprocess.CompletedProcess[str]:
    code = (
        "from tests.test_v3.conftest import lane_skip_reason; "
        "assert lane_skip_reason('cpu') is None; "
        f"reason = lane_skip_reason('{ 'gpu' if env_name == 'V3_FORCE_DEVICE_RAISE' else 'toolchain' }'); "
        "assert reason is not None; print(reason)"
    )
    env = os.environ.copy()
    env[env_name] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
        env=env,
        text=True,
        capture_output=True,
    )


def test_cpu_baseline_always_runs() -> None:
    from tests.test_v3.conftest import lane_skip_reason

    assert lane_skip_reason("cpu") is None


def test_missing_cuda_skips_not_errors() -> None:
    result = _probe("V3_FORCE_DEVICE_RAISE")
    assert result.returncode == 0, result.stderr
    assert "forced" in result.stdout


def test_missing_toolchain_skips_not_errors() -> None:
    result = _probe("V3_FORCE_TOOLCHAIN_RAISE")
    assert result.returncode == 0, result.stderr
    assert "forced" in result.stdout
