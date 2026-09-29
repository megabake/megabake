from dataclasses import replace
from pathlib import Path

import pytest

from megabake.v3.backends.cuda.admit import _entry_code_size, inspect_entry
from megabake.v3.backends.cuda.compile import compile_entry
from megabake.v3.backends.cuda.physical import search_physical_candidates
from tests.test_v3.cpu.test_physical_search import _case


def _selected():
    plan, registry, profile = _case()
    search = search_physical_candidates(plan, registry, profile)
    candidate = next(item for item in search.candidates
                     if item.candidate_id == search.selected_candidate_id)
    tactic = next(item for item in registry.compatible_tactics
                  if item.tactic_id == candidate.tactic_id)
    return candidate, tactic, profile


@pytest.mark.v3_toolchain
def test_v3r024_shared_entry_build_has_exact_function_code_and_resources(tmp_path):
    candidate, tactic, profile = _selected()
    result = compile_entry(candidate, tactic, profile, tmp_path / "entry.so",
                           shared_library=True)
    assert result.return_code == 0, result.stderr
    assert result.target == "sm_90"
    assert result.artifact_hash and result.artifact_bytes > 0
    assert result.ptxas["registers_per_thread"] is not None
    code_bytes, command, source = _entry_code_size(result.artifact_path, candidate.target)
    assert code_bytes and command[-1] == result.artifact_path, source


def test_v3r024_rejects_changed_binary_and_target_before_device_admission(tmp_path):
    candidate, tactic, profile = _selected()
    result = compile_entry(candidate, tactic, profile, tmp_path / "entry.so",
                           shared_library=True)
    assert result.return_code == 0, result.stderr
    with pytest.raises(ValueError, match="exact-target legal"):
        compile_entry(replace(candidate, target="sm_90a"), tactic, profile,
                      tmp_path / "wrong-target.so", shared_library=True)
    with pytest.raises(RuntimeError, match="binary hash changed"):
        inspect_entry(result.artifact_path, candidate, profile,
                      replace(result, artifact_hash="0" * 64))
