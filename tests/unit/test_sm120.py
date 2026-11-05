from __future__ import annotations

import pytest

from megabake import select_bodies
from megabake.bodies import SelectionPolicy, validate_body_config
from megabake.bodies.catalog import generation, get_candidates, provider
from megabake.bodies.cuda.sm120.gemm import SM12xGemmProvider
from megabake.fx_handler.transformers import StatefulAttention
from tests.unit.test_blackwell import _region
from tests.unit.test_bodies import _target
from tests.unit.test_task28_kv_state import _graph, _inputs


@pytest.mark.parametrize("architecture", ["sm120", "sm121"])
def test_sm12x_generation_owns_exact_architectures(architecture):
    target = _target(architecture=architecture, supports_wgmma=False)

    selected = generation(target)

    assert selected is not None
    assert selected.architectures == frozenset({"sm120", "sm121"})
    assert [item.provider_name for item in selected.providers] == [
        "sm12x_mma",
        "sm12x_attention",
    ]


@pytest.mark.parametrize("architecture", ["sm120", "sm121"])
def test_sm12x_gemm_selection_and_fallback(architecture):
    target = _target(architecture=architecture, supports_wgmma=False)
    result = get_candidates(_region(), target, SelectionPolicy())
    config = next(item for item in result.configurations if item.provider == "sm12x_mma")

    assert config.tiles.mma_family == "mma.sync"
    assert config.evidence.source.startswith("megabake SM12x")
    assert {item.name for item in result.configurations if item.provider == "sm12x_mma"} >= {
        "sm12x_mma_16x32x16",
        "sm12x_mma_16x128x16",
    }
    assert any(item.provider == "generic" for item in result.configurations)
    assert not any(item.provider == "blackwell_mma" for item in result.configurations)
    validate_body_config(config, target, SelectionPolicy())
    assert provider(config.provider).provider_name == "sm12x_mma"


def test_sm12x_gemm_rejects_non_sm12x():
    result = SM12xGemmProvider().get_candidates(_region(), _target(), SelectionPolicy())
    assert not result.configurations
    assert "SM12x" in result.rejections[0].reason


@pytest.mark.parametrize("architecture", ["sm120", "sm121"])
def test_sm12x_stateful_decode_attention_selection(architecture):
    graph = _graph(StatefulAttention(), _inputs())
    target = _target(architecture=architecture, supports_wgmma=False)

    selection = select_bodies(graph, target, SelectionPolicy(mode="decode"))

    config = next(
        item.body for item in selection.selections if item.body.provider == "sm12x_attention"
    )
    assert config.name == "attention_stateful_decode_rows"
    validate_body_config(config, target, SelectionPolicy(mode="decode"))
    assert provider(config.provider).provider_name == "sm12x_attention"


def test_batch_one_gemv_is_reachable_before_mma_in_bounded_search():
    import torch

    from megabake.planner import _contraction_body_order
    from tests.unit.test_blackwell import Gemm
    from tests.unit.test_bodies import _regions

    region = next(
        region
        for region in _regions(
            Gemm(),
            torch.randn(1, 70, dtype=torch.bfloat16),
            torch.randn(33, 70, dtype=torch.bfloat16),
        )
        if region.tensor_programs[0].family == "contraction"
    )
    policy = SelectionPolicy(mode="decode")
    candidates = SM12xGemmProvider().get_candidates(
        region, _target(architecture="sm121", supports_wgmma=False), policy
    )
    for body in candidates.configurations:
        validate_body_config(body, _target(architecture="sm121", supports_wgmma=False), policy)
    ordered = sorted(
        candidates.configurations,
        key=lambda body: _contraction_body_order(body, region.tensor_programs[0], policy),
    )
    assert ordered[0].name == "sm12x_gemv_warp4"


def test_grouped_search_reaches_alternatives_without_cartesian_expansion():
    from dataclasses import replace

    from megabake.planner import _grouped_combinations

    region = _region()
    result = SM12xGemmProvider().get_candidates(
        region, _target(architecture="sm121", supports_wgmma=False), SelectionPolicy()
    )
    options = []
    for index in range(20):
        current = replace(region, id=f"projection{index}")
        options.append(
            [(current, current.tensor_programs[0], body) for body in result.configurations]
        )
    first = [items[0] for items in options]
    plans = list(_grouped_combinations(first, options, 8))
    assert plans[0] == tuple(first)
    assert 1 < len(plans) <= 8
    assert list(_grouped_combinations(first, options, 8)) == plans
    assert all(len({item[2].name for item in plan}) == 1 for plan in plans)
    assert len(list(_grouped_combinations(first, options, 1))) == 1
