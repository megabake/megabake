from pathlib import Path

import pytest
import torch

from megabake.v3.algorithms import enumerate_algorithm_choices, materialized_cached_attention
from megabake.v3.backends.cuda.bodies.generic import emit_cuda_body
from megabake.v3.backends.cuda.bodies.generic import CudaBodyError
from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import WorkerProgramError, lower_worker_program
from megabake.v3.frontend.capture import UnsupportedGraphError
from megabake.v3.logical import derive_dependencies
from tests.test_v3.attention_fixtures import (
    attention_inputs, capture_attention, oracle_inputs, rotary_attention_inputs,
)
from tests.test_v3.fixtures import ATTENTION_TINY, STATE_POISON


@pytest.mark.parametrize("position", [0, 1, 15, 16])
@pytest.mark.parametrize("mask_kind", ["causal", "window", "empty"])
def test_cached_attention_has_exact_reference_state_and_published_inputs(position, mask_kind):
    inputs = attention_inputs(position, mask_kind=mask_kind)
    captured, indexed, plan = capture_attention(inputs)
    assert indexed.strict_supported and plan is not None
    expected = captured.run_reference(*inputs)
    actual = indexed.evaluate(dict(zip(
        ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask"), inputs)))
    torch.testing.assert_close(actual, expected)
    assert len(indexed.state_transitions) == 2
    operation = indexed.operations[-1]
    assert operation.kind == "Attention"
    assert {effect for edge in operation.effect_edges for effect in edge.depends_on} == {
        "append-k", "append-v"}
    assert emit_cuda_body(indexed, operation).body_kind == "online_cached_attention_warp32"
    torch.testing.assert_close(
        materialized_cached_attention(inputs[0], expected["cache_k"],
                                      expected["cache_v"], inputs[6]),
        expected["output"], rtol=1e-5, atol=1e-5)
    relations, diagnostics = derive_dependencies(plan, indexed)
    assert not diagnostics
    assert len([item for item in relations if item.kind == "state_publish"]) == 2
    if mask_kind == "empty":
        assert torch.count_nonzero(expected["output"]) == 0
    for name, source in (("cache_k", inputs[3]), ("cache_v", inputs[4])):
        torch.testing.assert_close(expected[name][..., :position, :], source[..., :position, :], rtol=0, atol=0)
        torch.testing.assert_close(expected[name][..., position + 1:, :],
                                   source[..., position + 1:, :], rtol=0, atol=0)


def test_cached_attention_rejects_unpublished_current_k_and_unsupported_masks():
    inputs = attention_inputs(1)
    _, unpublished, plan = capture_attention(inputs, publish_current=False)
    assert not unpublished.strict_supported and plan is None
    assert any("cached attention" in item.message for item in unpublished.diagnostics)

    float_mask = (*inputs[:-1], inputs[-1].float())
    _, unsupported, plan = capture_attention(float_mask)
    assert not unsupported.strict_supported and plan is None
    assert any("cached attention" in item.message for item in unsupported.diagnostics)

    rank_one_mask = (*inputs[:-1], inputs[-1].reshape(-1))
    with pytest.raises(UnsupportedGraphError, match="ShapeProp"):
        capture_attention(rank_one_mask)

    _, dropout, plan = capture_attention(inputs, dropout=0.1)
    assert not dropout.strict_supported and plan is None

    _, no_policy, _ = capture_attention(inputs)
    no_policy.source_program.policy = None
    assert not any(choice.algorithm == "online_softmax"
                   for choice in enumerate_algorithm_choices(no_policy))
    with pytest.raises(CudaBodyError, match="reassociation policy"):
        emit_cuda_body(no_policy, no_policy.operations[-1])


def test_cached_attention_long_context_and_final_capacity():
    for position in (128, 256):
        inputs = attention_inputs(position, capacity=257)
        _, indexed, plan = capture_attention(inputs)
        assert indexed.strict_supported and plan is not None
        assert indexed.operations[-1].attributes["attention"]["capacity"] == 257


@pytest.mark.parametrize("oracle", [ATTENTION_TINY, STATE_POISON])
@pytest.mark.parametrize("position", [0, 1, 15, 16])
def test_named_cached_attention_oracles(oracle, position):
    case = oracle(position=position)
    inputs = oracle_inputs(case)
    captured, indexed, plan = capture_attention(inputs)
    assert indexed.strict_supported and plan is not None
    for result in (captured.run_reference(*inputs), indexed.evaluate(dict(zip(
            ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask"), inputs)))):
        torch.testing.assert_close(result["output"], case.expected["output"], rtol=1e-5, atol=1e-5)
        for key in ("cache_k", "cache_v"):
            torch.testing.assert_close(result[key], case.expected[key], rtol=0, atol=0)


def test_mqa_explicit_scale_and_individual_empty_head():
    inputs = list(attention_inputs(1, heads_kv=1, mask_heads=4))
    inputs[6][0, 2] = False
    captured, indexed, plan = capture_attention(tuple(inputs), scale=0.125)
    assert indexed.strict_supported and plan is not None
    expected = captured.run_reference(*inputs)
    actual = indexed.evaluate(dict(zip(
        ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask"), inputs)))
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual["output"][0, 2]) == 0


@pytest.mark.parametrize("scale", [0.0, -0.5])
def test_attention_accepts_finite_nonpositive_explicit_scale(scale):
    inputs = attention_inputs(1)
    captured, indexed, plan = capture_attention(inputs, scale=scale)
    assert indexed.strict_supported and plan is not None
    expected = captured.run_reference(*inputs)
    actual = indexed.evaluate(dict(zip(
        ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask"), inputs)))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("position", [0, 16])
def test_rope_producer_precedes_current_k_publication(position):
    inputs = rotary_attention_inputs(position)
    captured, indexed, plan = capture_attention(inputs, rotary=True)
    assert indexed.strict_supported and plan is not None
    operations = {op.local_reference.node_name: index for index, op in enumerate(indexed.operations)}
    assert operations["rotated_k"] < operations["write_k"] < operations["attention"]
    q, current_k, _, _, _, _, _, permutation, cosine, signed_sine = inputs
    rotated_q = q * cosine + q.index_select(-1, permutation) * signed_sine
    rotated_k = current_k * cosine + current_k.index_select(-1, permutation) * signed_sine
    expected = captured.run_reference(*inputs)
    torch.testing.assert_close(expected["cache_k"][..., position:position + 1, :], rotated_k)
    torch.testing.assert_close(expected["cache_k"][..., position + 1:, :], inputs[3][..., position + 1:, :], rtol=0, atol=0)
    actual = indexed.evaluate(dict(zip(
        ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask",
         "rope_index", "rope_cos", "rope_signed_sin"), inputs)))
    torch.testing.assert_close(actual, expected)
    assert rotated_q.shape == expected["output"].shape


def test_rope_permutation_needs_bounds_and_warp_tactic_needs_full_warps():
    inputs = rotary_attention_inputs()
    _, unsupported, plan = capture_attention(inputs, rotary=True, declare_rotary_bounds=False)
    assert not unsupported.strict_supported and plan is None
    assert any("dynamic indices" in item.message for item in unsupported.diagnostics)

    _, indexed, plan = capture_attention(inputs, rotary=True)
    profile = CudaTargetProfile.from_json((
        Path(__file__).resolve().parents[3] /
        "ART/tasks/V3R-021/target_profile.json").read_text())
    with pytest.raises(WorkerProgramError, match="divisible by 32"):
        lower_worker_program(indexed, plan, profile, block_threads=16)
