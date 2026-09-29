from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from pathlib import Path

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.backends.cuda.bodies.registry import query_body_tactics
from megabake.v3.backends.cuda.compile import compile_entry
from megabake.v3.backends.cuda.physical import search_physical_candidates
from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.logical import lower_logical_plan
from tests.test_v3.body_codegen import index_module
from tests.test_v3.cpu.test_physical_search import AddMM


def _case():
    root = Path(__file__).resolve().parents[3]
    profile = CudaTargetProfile.from_json(
        (root / "ART/tasks/V3R-021/target_profile.json").read_text()
    )
    torch.manual_seed(41)
    inputs = (torch.randn(1, 33, dtype=torch.float16),
              torch.randn(17, 33, dtype=torch.float16),
              torch.randn(17, dtype=torch.float16))
    _, indexed = index_module(AddMM(), *inputs)
    choices = enumerate_algorithm_choices(indexed)
    logical_choices = tuple(item.choice_id for item in choices if item.algorithm == "indexed")
    plan = lower_logical_plan(indexed, choices, logical_choices)
    operation = next(item for item in indexed.operations if item.kind == "Contraction")
    policy = SimpleNamespace(contract_hash="reassociation-allowed",
                             reassociation_allowed=lambda _name: True)
    body_choices = enumerate_algorithm_choices(indexed, numerical_policy=policy)
    body_choice = next(item for item in body_choices if item.algorithm == "projection_k_parallel")
    registry = query_body_tactics(indexed, operation, body_choice, profile,
                                  numerical_policy=policy)
    search = search_physical_candidates(plan, registry, profile)
    candidate = next(item for item in search.candidates
                     if item.candidate_id == search.selected_candidate_id)
    tactic = next(item for item in registry.compatible_tactics
                  if item.tactic_id == candidate.tactic_id)
    return candidate, tactic, profile


@pytest.mark.v3_toolchain
def test_v3r023_compiles_selected_and_control_finalists_for_exact_target(tmp_path):
    candidate, tactic, profile = _case()
    result = compile_entry(candidate, tactic, profile, tmp_path / "selected.cubin")
    assert result.return_code == 0, result.stderr
    assert result.target == "sm_90"
    assert "-arch=sm_90" in result.command
    assert "--use_fast_math" not in result.command
    assert result.artifact_hash and result.artifact_bytes > 0
    assert result.ptxas["registers_per_thread"] is not None


def test_v3r023_rejects_target_or_block_tampering_before_nvcc(tmp_path):
    candidate, tactic, profile = _case()
    with pytest.raises(ValueError, match="exact-target legal"):
        compile_entry(replace(candidate, target="sm_90a"), tactic, profile,
                      tmp_path / "wrong-target.cubin")
    with pytest.raises(ValueError, match="one output per warp"):
        compile_entry(replace(candidate, block_threads=64), tactic, profile,
                      tmp_path / "wrong-block.cubin")
