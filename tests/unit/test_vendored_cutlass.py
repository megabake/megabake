"""The pinned CUTLASS upstream content next to the GEMM bodies is hash-checked."""

from __future__ import annotations

import ast
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cutlass_upstream_copy_is_unmodified():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "vendor_cutlass.py"), "--check"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_sm100_persistent_fork_preserves_upstream_kernel():
    directory = ROOT / "src" / "megabake" / "bodies" / "cuda" / "sm100"
    original = (directory / "cutlass_upstream" / "dense_gemm_persistent.py").read_text()
    fork = (directory / "cutlass_gemm.py").read_text()

    def method(source, name):
        module = ast.parse(source)
        cls = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "PersistentDenseGemmKernel"
        )
        return next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
        )

    assert ast.dump(method(original, "kernel")) == ast.dump(method(fork, "kernel"))
    assert method(fork, "table_kernel")


@pytest.mark.parametrize("modify_header", [False, True])
@pytest.mark.parametrize("filename", ["fp16_gemm_0.py", "dense_gemm_persistent.py"])
def test_sm100_upstream_hash_check_rejects_modified_content(
    monkeypatch, tmp_path, modify_header, filename
):
    spec = spec_from_file_location("vendor_cutlass", ROOT / "tools" / "vendor_cutlass.py")
    vendor = module_from_spec(spec)
    spec.loader.exec_module(vendor)
    modified = tmp_path / filename
    reference = next(item for item in vendor.REFERENCES if item["original"].name == filename)
    source = reference["original"].read_bytes()
    modified.write_bytes(b"# changed\n" + source if modify_header else source + b"\n# changed\n")
    monkeypatch.setitem(reference, "original", modified)
    with pytest.raises(SystemExit, match="upstream .* modified"):
        vendor.check_upstream()
