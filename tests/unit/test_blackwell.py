from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from megabake.bodies.cuda.sm100.gemm import BlackwellGemmProvider
from megabake.bodies import SelectionPolicy, validate_body_config
from megabake.bodies.catalog import get_candidates
from megabake.fx_handler.transformers import StatefulAttention
from megabake.lowering import Constant
from megabake.planner import select_bodies
from tests.unit.test_bodies import _regions, _target
from tests.unit.test_task28_kv_state import _graph, _inputs
from benchmarks.isolated import _graph_ordered_inputs


class Gemm(torch.nn.Module):
    def forward(self, source, weight):
        return source @ weight.T


def _region():
    return next(r for r in _regions(Gemm(), torch.randn(5, 70, dtype=torch.bfloat16),
                                    torch.randn(33, 70, dtype=torch.bfloat16))
                if r.tensor_programs[0].family == "contraction")


@pytest.mark.parametrize("architecture", ["sm100", "sm103", "sm120", "sm121"])
def test_blackwell_provider_is_tensor_core_without_hopper_resources(architecture):
    target = _target(architecture=architecture, supports_wgmma=False,
                     max_shared_memory_per_cta=99_328)
    result = get_candidates(_region(), target, SelectionPolicy())
    config = next(c for c in result.configurations if c.provider == "blackwell_mma")
    assert not any(c.provider == "g0_range" for c in result.configurations)
    assert config.tiles.mma_family == "mma.sync"
    assert config.scratch.size_bytes == 0
    assert not config.pipeline.async_operations
    validate_body_config(config, target, SelectionPolicy())


def test_blackwell_provider_rejects_other_targets_policy_and_changed_semantics():
    provider = BlackwellGemmProvider()
    region = _region()
    target = _target(architecture="sm121", supports_wgmma=False)
    assert not provider.get_candidates(region, _target(), SelectionPolicy()).configurations
    assert not provider.get_candidates(region, target, SelectionPolicy(allow_tuned=False)).configurations
    program = region.tensor_programs[0]
    for changed in (replace(program, init=Constant(1, "torch.float32")),
                    replace(program, post=Constant(0, "torch.bfloat16"))):
        result = provider.get_candidates(replace(region, tensor_programs=(changed,)), target, SelectionPolicy())
        assert not result.configurations


def test_sm100_single_row_contiguous_k_has_matvec_candidate(monkeypatch):
    monkeypatch.setenv("MEGABAKE_SM100_MATVEC", "1")
    region = next(
        r for r in _regions(Gemm(), torch.randn(1, 70, dtype=torch.bfloat16),
                            torch.randn(129, 70, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    provider = BlackwellGemmProvider()
    sm100 = _target(architecture="sm100", supports_wgmma=False)
    candidates = provider.get_candidates(region, sm100, SelectionPolicy(mode="decode"))
    matvec = next(c for c in candidates.configurations if c.name == "blackwell_matvec_1x16")
    assert matvec.tiles.cta_shape == (1, 16)
    validate_body_config(matvec, sm100, SelectionPolicy(mode="decode"))
    assert any(c.name == "blackwell_mma_16x32x16" for c in candidates.configurations)
    sm120 = _target(architecture="sm120", supports_wgmma=False)
    assert not any(c.name == "blackwell_matvec_1x16"
                   for c in provider.get_candidates(region, sm120, SelectionPolicy()).configurations)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5])
def test_sm100_short_row_candidates_respect_rows_and_architecture(monkeypatch, rows):
    monkeypatch.setenv("MEGABAKE_SM100_MATVEC", "1")
    region = next(
        r for r in _regions(Gemm(), torch.randn(rows, 128, dtype=torch.bfloat16),
                            torch.randn(256, 128, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    provider = BlackwellGemmProvider()
    sm100 = _target(architecture="sm100", supports_wgmma=False)
    names = {c.name for c in provider.get_candidates(region, sm100, SelectionPolicy()).configurations}
    expected = "blackwell_matvec_1x16" if rows == 1 else f"blackwell_smallrows_{rows}x16"
    assert (expected in names) == (rows <= 4)
    assert "blackwell_mma_16x32x16" in names
    sm120 = _target(architecture="sm120", supports_wgmma=False)
    assert not any(c.name.startswith(("blackwell_matvec_", "blackwell_smallrows_"))
                   for c in provider.get_candidates(region, sm120, SelectionPolicy()).configurations)


@pytest.mark.parametrize("tile", ["64", "128"])
def test_sm100_tcgen_is_opt_in_and_respects_policy(monkeypatch, tile):
    region = next(
        r for r in _regions(Gemm(), torch.randn(4, 128, dtype=torch.bfloat16),
                            torch.randn(256, 128, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    provider = BlackwellGemmProvider()
    target = _target(architecture="sm100", supports_wgmma=False)
    policy = SelectionPolicy(mode="prefill", table_launch=True)
    monkeypatch.delenv("MEGABAKE_SM100_TCGEN", raising=False)
    assert not any(c.provider == "sm100_tma"
                   for c in provider.get_candidates(region, target, policy).configurations)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", tile)
    configs = provider.get_candidates(region, target, policy).configurations
    tma = next(c for c in configs if c.provider == "sm100_tma")
    assert tma.tiles.cta_shape == (8, int(tile))
    assert all(layout.alignment_bytes == 16 for layout in tma.layouts.inputs.values())
    validate_body_config(tma, target, policy)
    assert any(c.provider == "sm100_tma" for c in provider.get_candidates(
        region, target, replace(policy, mode="decode")).configurations)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_DECODE", "0")
    for blocked in (replace(policy, mode="decode"), replace(policy, table_launch=False),
                    replace(policy, allow_tma_gemm=False)):
        assert not any(c.provider == "sm100_tma"
                       for c in provider.get_candidates(region, target, blocked).configurations)
    assert not any(c.provider == "sm100_tma" for c in provider.get_candidates(
        region, replace(target, supports_tma=False), policy).configurations)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_DECODE", "1")
    assert any(c.provider == "sm100_tma" for c in provider.get_candidates(
        region, target, replace(policy, mode="decode")).configurations)


@pytest.mark.parametrize("columns, expected", [(192, 64), (256, 128), (224, None)])
def test_sm100_auto_tile_is_selected_per_shape(monkeypatch, columns, expected):
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "auto")
    region = next(
        r for r in _regions(Gemm(), torch.randn(1, 128, dtype=torch.bfloat16),
                            torch.randn(columns, 128, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    configs = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="decode", table_launch=True),
    ).configurations
    tma = [c for c in configs if c.provider == "sm100_tma"]
    if expected is None:
        assert not tma
    else:
        assert [c.tiles.cta_shape for c in tma] == [(8, expected)]


def test_sm100_tuning_file_validates_device_and_shape(monkeypatch, tmp_path):
    from megabake.bodies.cuda.sm100 import tuning

    fingerprint = {"multiprocessor_count": 60, "name": "test device"}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(tuning, "device_fingerprint", lambda index: fingerprint)
    path = tmp_path / "tuning.json"
    path.write_text(json.dumps({"schema": 1, "fingerprint": fingerprint,
                                "choices": {"1:192:512": {"tile": 64, "splits": 4}}}))
    monkeypatch.setenv("MEGABAKE_SM100_TUNING_FILE", str(path))
    target = _target(architecture="sm100", supports_wgmma=False)
    assert tuning.choice_for_shape(target, 1, 192, 512) == (64, 4)
    assert tuning.choice_for_shape(target, 1, 256, 512) is None
    with pytest.raises(ValueError, match="compilation target"):
        tuning.choice_for_shape(replace(target, multiprocessor_count=40), 1, 192, 512)
    monkeypatch.setattr(tuning, "device_fingerprint", lambda index: {"multiprocessor_count": 60})
    with pytest.raises(ValueError, match="does not match this device"):
        tuning.choice_for_shape(target, 1, 192, 512)


def test_sm100_tuning_fingerprint_tracks_experimental_settings(monkeypatch):
    from megabake.bodies.cuda.sm100 import tuning

    monkeypatch.setattr(tuning, "_device_details", lambda index: {"name": "test device"})
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TOKEN_TILE", "32")
    monkeypatch.setenv("MEGABAKE_PREFILL_VALUE_BLOCK", "8")
    fingerprint = tuning.device_fingerprint(0)
    assert fingerprint["fused_epilogues"] is True
    assert fingerprint["token_tile"] == 32
    assert fingerprint["prefill_value_block"] == 8
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "0")
    monkeypatch.setenv("MEGABAKE_SM100_TOKEN_TILE", "8")
    monkeypatch.setenv("MEGABAKE_PREFILL_VALUE_BLOCK", "0")
    changed = tuning.device_fingerprint(0)
    assert changed["fused_epilogues"] is False
    assert changed["token_tile"] == 8
    assert changed["prefill_value_block"] == 0


def test_sm100_measured_choice_controls_tile_and_split(monkeypatch):
    from megabake.bodies.cuda.sm100 import tuning
    from megabake.bodies.cuda.sm90.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setattr(tuning, "choice_for_shape", lambda target, rows, columns, reduction: (64, 4))
    region = next(
        r for r in _regions(Gemm(), torch.randn(1, 512, dtype=torch.bfloat16),
                            torch.randn(192, 512, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    configs = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="decode", table_launch=True),
    ).configurations
    config = next(c for c in configs if c.provider == "sm100_tma")
    assert config.tiles.cta_shape == (8, 64)
    assert split_k(config) == 4


@pytest.mark.parametrize("columns, expected", [(2048, 2), (8192, 1)])
def test_sm100_split_limit_avoids_second_wave(monkeypatch, columns, expected):
    from megabake.bodies.cuda.sm90.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", "8")
    region = next(
        r for r in _regions(Gemm(), torch.randn(1, 2048, dtype=torch.bfloat16),
                            torch.randn(columns, 2048, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    configs = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="decode", table_launch=True),
    ).configurations
    assert split_k(next(c for c in configs if c.provider == "sm100_tma")) == expected


def test_sm100_warp_reduction_is_opt_in(monkeypatch):
    monkeypatch.delenv("MEGABAKE_SM100_MATVEC", raising=False)
    monkeypatch.delenv("MEGABAKE_SM100_TCGEN", raising=False)
    region = next(
        r for r in _regions(Gemm(), torch.randn(1, 128, dtype=torch.bfloat16),
                            torch.randn(256, 128, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    configs = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="decode"),
    ).configurations
    assert [c.name for c in configs] == ["blackwell_mma_16x32x16"]


@pytest.mark.parametrize("rows, columns, reduction, dtype", [
    (33, 256, 128, torch.bfloat16), (4, 129, 128, torch.bfloat16),
    (4, 256, 70, torch.bfloat16), (4, 256, 128, torch.float16),
])
def test_sm100_tcgen_falls_back_for_unsupported_shapes(monkeypatch, rows, columns, reduction, dtype):
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    region = next(
        r for r in _regions(Gemm(), torch.randn(rows, reduction, dtype=dtype),
                            torch.randn(columns, reduction, dtype=dtype))
        if r.tensor_programs[0].family == "contraction"
    )
    result = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="prefill", table_launch=True),
    )
    assert result.configurations
    assert not any(c.provider == "sm100_tma" for c in result.configurations)


@pytest.mark.parametrize("rows", [5, 8, 9, 17, 32])
def test_sm100_tcgen_prefill_accepts_multiple_token_tiles(monkeypatch, rows):
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    region = next(
        r for r in _regions(Gemm(), torch.randn(rows, 128, dtype=torch.bfloat16),
                            torch.randn(256, 128, dtype=torch.bfloat16))
        if r.tensor_programs[0].family == "contraction"
    )
    result = BlackwellGemmProvider().get_candidates(
        region, _target(architecture="sm100", supports_wgmma=False),
        SelectionPolicy(mode="prefill", table_launch=True),
    )
    assert any(c.provider == "sm100_tma" for c in result.configurations)


def test_sm100_selects_stateful_attention_without_hopper_wgmma():
    graph = _graph(StatefulAttention(), _inputs())
    target = _target(architecture="sm100", supports_wgmma=False)
    selection = select_bodies(graph, target, SelectionPolicy(mode="decode"))

    assert selection.selections
    assert selection.report.complete
    attention = next(s.body for s in selection.selections if s.body.provider == "attention")
    assert attention.name == "attention_stateful_decode_rows"
    assert not attention.evidence.measured
    assert not attention.evidence.measurements
    assert "active_blocks_per_sm" not in attention.evidence.resources


@pytest.mark.parametrize(
    ("architecture", "supports_wgmma", "threads_per_warp"),
    [("sm103", False, 32), ("sm90a", False, 32), ("sm100", False, 64)],
)
def test_attention_rejects_unvalidated_targets(architecture, supports_wgmma, threads_per_warp):
    graph = _graph(StatefulAttention(), _inputs())
    target = _target(
        architecture=architecture,
        supports_wgmma=supports_wgmma,
        threads_per_warp=threads_per_warp,
    )
    selection = select_bodies(graph, target, SelectionPolicy(mode="decode"))

    assert not selection.selections
    assert any(rejection.provider == "attention" for rejection in selection.rejections)


def test_isolated_benchmark_maps_non_self_inverse_input_permutation():
    inputs = tuple(object() for _ in range(4))
    compiled = SimpleNamespace(input_order=("l_gate_weight_", "l_up_weight_", "l_source_", "l_down_weight_"))
    assert _graph_ordered_inputs(compiled, inputs) == (inputs[1], inputs[2], inputs[0], inputs[3])
