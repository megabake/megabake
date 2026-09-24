"""MB3-002: importing the legacy package must not resolve CUTLASS headers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_clean_import() -> subprocess.CompletedProcess[str]:
    source_root = Path(__file__).resolve().parents[3] / "src"
    code = """
import importlib.util
import os

real_find_spec = importlib.util.find_spec
def blocked(name, *args, **kwargs):
    if name == 'cutlass_library':
        return None
    return real_find_spec(name, *args, **kwargs)
importlib.util.find_spec = blocked
os.environ['CUTLASS_PATH'] = '/nonexistent/cutlass'
import megabake.runtime.cuda_compiler as compiler
assert not hasattr(compiler, '_CUTLASS_INCLUDE')
try:
    compiler._compile_megakernel(force=True)
except RuntimeError as exc:
    assert 'CUTLASS/CuTe headers were not found' in str(exc)
else:
    raise AssertionError('compile request unexpectedly succeeded without headers')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=source_root.parent,
        env=env,
        text=True,
        capture_output=True,
    )


def test_import_defers_header_lookup_until_compile() -> None:
    result = _run_clean_import()
    assert result.returncode == 0, result.stderr or result.stdout


