from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.backends.cuda.bodies.registry import query_body_tactics
from megabake.v3.backends.cuda.profile import CudaTargetProfile, ProfileFact
from tests.test_v3.body_codegen import index_module


def _policy(allow_reassociation=True):
    return SimpleNamespace(
        contract_hash=f"reassociation-{allow_reassociation}",
        reassociation_allowed=lambda _operation: allow_reassociation,
    )


def _target(*, baseline=("sm_90",), architecture=("sm_90a",), max_threads=1024,
            compiler=None):
    fact = lambda value: ProfileFact(value, "synthetic CUDA target fixture")
    unknown = ProfileFact(None, "synthetic CUDA target fixture", "not declared by fixture")
    return CudaTargetProfile(
        device_index=0, device_name="synthetic H200", device_uuid="fixture-uuid",
        device_facts={"compute_capability": fact("9.0"), "visible_sms": fact(60),
                      "visible_memory_bytes": fact(1_000_000)},
        resource_limits={"warp_size": fact(32), "max_threads_per_block": fact(max_threads)},
        feature_attributes={"cooperative_launch": fact(True)},
        target_sets={"baseline": fact(list(baseline)),
                     "architecture_specific": fact(list(architecture)),
                     "family_specific": unknown,
                     "compiler_targets": fact(list((*baseline, *architecture) if compiler is None else compiler))},
        versions={"nvcc": fact("12.8.93"), "cutlass": fact("3.8.0.0"),
                  "cublasdx": ProfileFact(None, "fixture", "provider is unavailable")},
    )


def _linear(dtype=torch.float16):
    class AddMM(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.addmm(bias, x, weight.transpose(0, 1), beta=-0.25, alpha=1.75)

    x = torch.randn(1, 33, dtype=dtype)
    weight = torch.randn(17, 33, dtype=dtype)
    bias = torch.randn(17, dtype=dtype)
    program, indexed = index_module(AddMM(), x, weight, bias)
    policy = _policy()
    choices = enumerate_algorithm_choices(indexed, numerical_policy=policy)
    choice = next(item for item in choices if item.algorithm == "projection_k_parallel")
    operation = next(item for item in indexed.operations if item.kind == "Contraction")
    return program, indexed, operation, choice, policy


def test_v3r022_indexes_generic_and_simt_bodies_with_exact_output_ownership():
    _, indexed, operation, choice, policy = _linear()
    result = query_body_tactics(indexed, operation, choice, _target(), numerical_policy=policy)
    by_id = {item.tactic_id: item for item in result.compatible_tactics}
    assert set(by_id) == {
        "generic.indexed",
        "simt.k_parallel.w1.v1", "simt.k_parallel.w1.v4",
        "simt.k_parallel.w2.v1", "simt.k_parallel.w2.v4",
        "simt.k_parallel.w4.v1", "simt.k_parallel.w4.v4",
    }
    for tactic in result.compatible_tactics:
        assert tactic.device_callable and tactic.target == "sm_90"
        assert tactic.output_tile_map["shape"] == [1, 17]
        assert tactic.output_tile_map["index_map"] == ["i0", "i1"]
        assert tactic.output_tile_map["logical_tile_coordinate"] == "flat_output_index"
        assert tactic.reduction_footprint["complete_reduction"]
        assert tactic.stage_capabilities == ("ATOMIC_TILE",)
        assert tactic.source_hash
    simt = by_id["simt.k_parallel.w4.v4"]
    assert simt.block_contract["dimensions"] == [128, 1, 1]
    assert simt.block_contract["roles"] == ["one_logical_output_per_warp", "all_32_lanes_reduce_k"]
    assert "simt_output" in simt.source_text
    assert len([item for item in result.rejected_tactics if item.provider == "tensor_core"]) == 9
    assert all(not item.device_callable for item in result.rejected_tactics)
    assert any("blockIdx.x" in item.rejection_reasons[0] for item in result.rejected_tactics)
    assert any(item.provider == "cutlass3x" for item in result.rejected_tactics)
    assert any(item["provider"] == "cublasdx" for item in result.provider_rejections)
    assert result.to_dict()["claim"] == "not_measured"
    assert result.registry_key == result.registry_key


def test_v3r022_rejects_unlisted_target_and_incompatible_cta_size():
    _, indexed, operation, choice, policy = _linear()
    target = _target(baseline=("sm_90",), architecture=(), compiler=())
    no_baseline = query_body_tactics(indexed, operation, choice, target, numerical_policy=policy)
    assert not no_baseline.compatible_tactics
    assert any("exact target sm_90" in reason for tactic in no_baseline.rejected_tactics
               for reason in tactic.rejection_reasons)

    small_block = query_body_tactics(indexed, operation, choice, _target(max_threads=16),
                                     numerical_policy=policy)
    simt = [item for item in small_block.rejected_tactics if item.provider == "simt"]
    assert len(simt) == 6
    assert all("block needs" in item.rejection_reasons[0]
               and "target limit is 16" in item.rejection_reasons[0] for item in simt)
    assert "generic.indexed" in {item.tactic_id for item in small_block.compatible_tactics}


def test_v3r022_keeps_bf16_and_tampered_algorithm_guards_out_of_simt():
    _, indexed, operation, choice, policy = _linear(torch.bfloat16)
    result = query_body_tactics(indexed, operation, choice, _target(), numerical_policy=policy)
    assert not any(item.provider == "simt" for item in result.compatible_tactics)
    assert any(item["provider"] == "simt" and "fp16-only" in item["reason"]
               for item in result.provider_rejections)

    tampered = replace(choice, operation_ids=())
    with pytest.raises(ValueError, match="does not cover"):
        query_body_tactics(indexed, operation, tampered, _target(), numerical_policy=policy)
