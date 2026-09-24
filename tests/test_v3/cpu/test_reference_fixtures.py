"""MB3-005: reference fixtures catch semantic, state, and policy mistakes."""

from __future__ import annotations

import torch

from megabake.v3.contracts import ExceptionalValuePolicy
from tests.test_v3.fixtures import (
    GATE_TINY,
    ATTENTION_TINY,
    LINEAR_TINY,
    NORM_VARIANTS,
    StateRegion,
    assert_reference_match,
    compare_outputs,
)


def test_seeds_replay_and_linear_views_are_distinct() -> None:
    first = LINEAR_TINY(seed=7)
    second = LINEAR_TINY(seed=7)
    assert torch.equal(first.inputs["x"], second.inputs["x"])
    assert first.metadata["weight_kn_stride"] != first.metadata["weight_nk_stride"]
    assert_reference_match(first.expected, second.expected)


def test_gate_variants_and_tail_chunks_are_explicit() -> None:
    case = GATE_TINY(seed=11)
    assert len(case.expected["silu_chunks"]) == 5
    assert case.expected["silu_chunks"][-1].shape[-1] == 1
    assert not torch.equal(case.expected["silu"], case.expected["gelu"])
    swapped = torch.nn.functional.silu(case.inputs["up"]) * case.inputs["gate"]
    report = compare_outputs(case.expected["silu"], swapped)
    assert not report.ok


def test_norm_keeps_cast_and_weight_expression_variants() -> None:
    case = NORM_VARIANTS(seed=13)
    assert case.metadata["contains_zero"]
    assert not torch.equal(case.expected["weight"], case.expected["one_plus_weight"])
    assert not torch.equal(
        case.expected["cast_before_weight"], case.expected["cast_after_weight"]
    )
    policy = ExceptionalValuePolicy(
        allow_nan=True, allow_pos_inf=True, allow_neg_inf=True
    )
    assert_reference_match(
        case.expected["eps_inside"],
        case.expected["eps_inside"].clone(),
        exceptional_value_policy=policy,
    )


def test_state_region_is_checked_separately_and_wrong_state_fails() -> None:
    expected = {"logits": torch.ones(1, 2), "state": torch.arange(8, dtype=torch.float32)}
    actual = {"logits": expected["logits"].clone(), "state": expected["state"].clone()}
    region = StateRegion(path=("state",), slices=(slice(2, 5),))
    assert compare_outputs(expected, actual, state_regions=(region,)).state_regions_ok
    actual["state"][3] = 99
    report = compare_outputs(expected, actual, state_regions=(region,))
    assert not report.ok
    assert not report.state_regions_ok


def test_attention_tiny_keeps_append_only_gqa_cache_semantics() -> None:
    case = ATTENTION_TINY(seed=17, position=16)
    assert case.expected["valid_length"] == 17
    assert torch.equal(case.state_before["cache_k"][:, :, :16], case.state_after["cache_k"][:, :, :16])
