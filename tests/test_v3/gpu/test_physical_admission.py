from pathlib import Path

import pytest
import torch

from megabake.v3.backends.cuda.admit import (
    contraction_problem,
    inspect_entry,
    launch_admitted,
)
from megabake.v3.backends.cuda.compile import compile_entry
from megabake.v3.backends.cuda.profile import CudaTargetProfile
from tests.test_v3.cpu.test_physical_search import _case


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def admitted_entry(tmp_path_factory):
    profile = CudaTargetProfile.from_json(
        (ROOT / "ART/tasks/V3R-021/target_profile.json").read_text()
    )
    from megabake.v3.backends.cuda.physical import search_physical_candidates

    plan, registry, profile = _case(profile)
    search = search_physical_candidates(plan, registry, profile)
    candidate = next(item for item in search.candidates
                     if item.candidate_id == search.selected_candidate_id)
    tactic = next(item for item in registry.compatible_tactics
                  if item.tactic_id == candidate.tactic_id)
    output = tmp_path_factory.mktemp("v3-entry") / "selected.so"
    compiled = compile_entry(candidate, tactic, profile, output, shared_library=True)
    assert compiled.return_code == 0, compiled.stderr
    report = inspect_entry(output, candidate, profile, compiled)
    assert report["launch_contract"]["admitted"], report["launch_contract"]
    return candidate, tactic, profile, output, compiled, report


@pytest.mark.v3_gpu
def test_v3r024_admits_exact_compiled_entry_resources(admitted_entry):
    candidate, _, profile, _, compiled, report = admitted_entry
    entry = report["compiled_function"]
    assert report["target_profile_key"] == profile.profile_key
    assert report["compiled_artifact"]["sha256"] == compiled.artifact_hash
    assert entry["block_threads"] == candidate.block_threads == 32
    assert entry["function_max_threads"] >= candidate.block_threads
    assert entry["requested_grid_ctas"] == candidate.grid_ctas == 17
    assert entry["cooperative_resident_ctas"] >= candidate.grid_ctas
    assert entry["registers_per_thread"] == entry["ptxas_registers_per_thread"]
    assert entry["static_shared_bytes"] >= 0 and entry["local_bytes"] >= 0
    assert entry["spill_store_bytes"] == entry["spill_load_bytes"] == 0
    assert entry["entry_code_bytes"] > 0
    assert entry["kernel_parameter_bytes"] == 104
    assert report["cross_sm"]["inter_cta_communication"] is False


@pytest.mark.v3_gpu
def test_v3r024_runs_disjoint_outputs_and_rejects_overresident_grid(admitted_entry):
    candidate, _, profile, library, _, report = admitted_entry
    torch.manual_seed(41)
    x = torch.randn(1, 33, dtype=torch.float16, device="cuda")
    weight = torch.randn(17, 33, dtype=torch.float16, device="cuda")
    bias = torch.randn(17, dtype=torch.float16, device="cuda")
    output = torch.empty(1, 17, dtype=torch.float16, device="cuda")
    problem = contraction_problem(x, weight, bias, output, alpha=1.75, beta=-0.25)
    launched = launch_admitted(library, candidate, profile, problem)
    assert launched["launched"] and launched["return_code"] == 0, launched
    torch.cuda.synchronize()
    expected = torch.addmm(bias, x, weight.transpose(0, 1), beta=-0.25, alpha=1.75)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.005)

    overresident = report["compiled_function"]["cooperative_resident_ctas"] + 1
    rejected = launch_admitted(library, candidate, profile, problem,
                               grid_ctas=overresident)
    assert not rejected["launched"] and rejected["return_code"] == -2
    assert rejected["requested_ctas"] > rejected["resident_ctas"]
