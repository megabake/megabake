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
