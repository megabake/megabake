from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.backends.cuda.bodies.registry import query_body_tactics
from megabake.v3.backends.cuda.physical import search_physical_candidates
from megabake.v3.backends.cuda.profile import CudaTargetProfile, ProfileFact
from megabake.v3.logical import lower_logical_plan
from tests.test_v3.body_codegen import index_module


class AddMM(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.addmm(bias, x, weight.transpose(0, 1), beta=-0.25, alpha=1.75)


def _inputs():
    torch.manual_seed(41)
    return (torch.randn(1, 33, dtype=torch.float16),
            torch.randn(17, 33, dtype=torch.float16),
            torch.randn(17, dtype=torch.float16))


def _target(cooperative=True):
    fact = lambda value: ProfileFact(value, "physical-search CPU fixture")
    unknown = ProfileFact(None, "physical-search CPU fixture", "not declared")
    return CudaTargetProfile(
        device_index=0, device_name="fixture H200", device_uuid="fixture",
        device_facts={"compute_capability": fact("9.0"), "visible_sms": fact(60),
                      "visible_memory_bytes": fact(1_000_000)},
        resource_limits={"warp_size": fact(32), "max_threads_per_block": fact(1024)},
        feature_attributes={"cooperative_launch": fact(cooperative)},
        target_sets={"baseline": fact(["sm_90"]),
                     "architecture_specific": fact(["sm_90a"]),
                     "family_specific": unknown,
                     "compiler_targets": fact(["sm_90", "sm_90a"])},
        versions={"nvcc": fact("12.8.93")},
    )


def _case(profile=None):
    module = AddMM()
    args = _inputs()
    _, indexed = index_module(module, *args)
    choices = enumerate_algorithm_choices(indexed)
    logical_choices = tuple(item.choice_id for item in choices if item.algorithm == "indexed")
    body_choice = next(item for item in choices if item.algorithm == "projection_k_parallel")
    plan = lower_logical_plan(indexed, choices, logical_choices)
    operation = next(item for item in indexed.operations if item.kind == "Contraction")
    profile = profile or _target()
    policy = SimpleNamespace(contract_hash="reassociation-allowed",
                             reassociation_allowed=lambda _name: True)
    body_choices = enumerate_algorithm_choices(indexed, numerical_policy=policy)
    body_choice = next(item for item in body_choices if item.algorithm == "projection_k_parallel")
    registry = query_body_tactics(indexed, operation, body_choice, profile,
                                  numerical_policy=policy)
    return plan, registry, profile


def test_v3r023_searches_joint_body_block_and_tile_menu_with_control():
    plan, registry, profile = _case()
    report = search_physical_candidates(plan, registry, profile)
    selected = next(item for item in report.candidates
                    if item.candidate_id == report.selected_candidate_id)
    control = next(item for item in report.candidates
                   if item.candidate_id == report.control_candidate_id)
    assert len(report.candidates) == 9
    assert selected.provider == "simt" and selected.grid_ctas == 17
    assert selected.tactic_id == "simt.k_parallel.w1.v1"
    assert selected.block_threads == 32 and selected.outputs_per_cta == 1
    assert selected.indexed_program_hash == registry.indexed_program_hash
    assert selected.logical_plan_hash == plan.structural_hash
    assert selected.target_profile_key == profile.profile_key
    assert selected.control_candidate_id == control.candidate_id
    assert control.provider == "generic" and control.grid_ctas == 1
    assert selected.resource_estimates["registers_per_thread"].startswith("UNKNOWN")
    assert selected.to_dict()["latency_us"] is None
    assert report.rejections  # tensor-core and library entries retain explicit reasons
    for item in report.candidates:
        assert item.grid_ctas * item.outputs_per_cta >= item.output_elements
        assert item.event_protocol["kind"] == "none"


def test_v3r023_rejects_unproved_target_or_semantics_and_bad_simt_block():
    plan, registry, profile = _case(_target(cooperative=False))
    with pytest.raises(ValueError, match="cooperative launch support"):
        search_physical_candidates(plan, registry, profile)

    plan, registry, profile = _case()
    with pytest.raises(ValueError, match="different indexed semantics"):
        search_physical_candidates(replace(plan, indexed_program_hash="tampered"),
                                   registry, profile)
    bad_tactics = tuple(
        replace(item, block_contract={**item.block_contract, "dimensions": [16, 1, 1]})
        if item.provider == "simt" else item
        for item in registry.compatible_tactics
    )
    rejected = search_physical_candidates(plan, replace(registry, compatible_tactics=bad_tactics),
                                          profile)
    assert len([item for item in rejected.candidates if item.provider == "simt"]) == 0
    assert any("SIMT block" in item.get("reason", "") for item in rejected.rejections)

