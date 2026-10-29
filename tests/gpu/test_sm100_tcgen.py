"""SM100 TMA/tcgen05 inside a cooperative table launch."""
import torch
import pytest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="SM100 CUDA device required",
)


@pytest.fixture(autouse=True)
def reset_dynamo():
    # Every parameter represents a distinct graph and backend configuration.
    torch._dynamo.reset()


def test_graphed_baseline_copies_new_tokens():
    from benchmarks.full_model import GraphedStep

    token = torch.tensor([[1]], device="cuda")
    step = GraphedStep(lambda token, position: token + position, token, [])
    for token_value, position_value in ((3, 2), (8, 5)):
        result = step(torch.tensor([[token_value]], device="cuda"),
                      torch.tensor([[position_value]], device="cuda"))
        assert result.item() == token_value + position_value


def test_graphed_prefill_copies_new_inputs(monkeypatch, tmp_path):
    import megabake.compiler as compiler
    from benchmarks.full_model import GraphedPrefill
    from megabake import make_backend

    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")

    class Model(torch.nn.Module):
        def forward(self, tokens, attention_mask):
            return tokens @ attention_mask.T

    torch.manual_seed(821)
    source = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) / 16
    mask = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16
    model = Model().eval()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    function = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        graph = GraphedPrefill(function, source, mask)
        for x, y in ((source, mask), (source + 0.125, mask - 0.125)):
            torch.testing.assert_close(graph(x, attention_mask=y), model(x, attention_mask=y),
                                       atol=0.01, rtol=0.03)


def test_sm100_auto_tile_mixes_tiles_in_one_kernel(monkeypatch, tmp_path):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "auto")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class MixedGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(407)
    args = (
        torch.randn(1, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(192, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 192, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = MixedGemm().eval()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args)):
            actual = compiled(*current)
            expected = model(*current)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.05)
    phases = [phase for phase in backend.megabake_last_compiled.plan.phases
              if phase.body.provider == "sm100_tma"]
    assert [phase.body.tiles.cta_shape for phase in phases] == [(8, 64), (8, 128)]
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1


@pytest.mark.parametrize("tile,splits,rows,columns", [
    (64, 2, 1, 192), (128, 4, 1, 256), (128, 2, 9, 256),
])
def test_sm100_split_k_repeats_and_rebinds(monkeypatch, tmp_path, tile, splits, rows, columns):
    import megabake.compiler as compiler
    from megabake import make_backend
    from megabake.bodies.cuda.sm90.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(678)
    args = (
        torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(columns, 512, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        rebound = tuple(tensor + 0.125 for tensor in args)
        for index in range(20):
            current = args if index % 2 == 0 else rebound
            actual = compiled(*current)
            expected = model(*current)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.03)
    phases = [phase for phase in backend.megabake_last_compiled.plan.phases
              if phase.body.provider == "sm100_tma"]
    assert [split_k(phase.body) for phase in phases] == [splits]


@pytest.mark.parametrize("splits", [1, 4])
def test_sm100_fused_residual_and_plain_share_variant(monkeypatch, tmp_path, splits):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", "128")
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Residual(torch.nn.Module):
        def forward(self, source, first, second, residual):
            return (source @ first.T + residual) @ second.T

    torch.manual_seed(193)
    args = (
        torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Residual().eval()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    cases = (args, tuple(tensor + 0.125 for tensor in args), args)
    fused_outputs = []
    with torch.no_grad():
        for current in cases:
            actual = compiled(*current)
            fused_outputs.append(actual.clone())
            torch.testing.assert_close(actual, model(*current), atol=0.02, rtol=0.05)
    names = [phase.body.name for phase in backend.megabake_last_compiled.plan.phases]
    assert sum(name.endswith("_epilogue") for name in names) == 1
    assert sum(name.startswith("sm100_tma") for name in names) == 2
    monkeypatch.setenv("MEGABAKE_SM100_EPILOGUE", "0")
    torch._dynamo.reset()
    reference_backend = make_backend(mode="decode", compile_kernel=True, cache=False,
                                     dump_dir=tmp_path / "unfused", scheduler="phase")
    unfused = torch.compile(model, backend=reference_backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current, expected in zip(cases, fused_outputs, strict=True):
            torch.testing.assert_close(unfused(*current), expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [64, 128])
def test_sm100_tcgen_extended_prefill_rows(monkeypatch, tmp_path, rows):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_MAX_ROWS", "128")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(472)
    source = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) / 16
    weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    model = Gemm().eval()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        torch.testing.assert_close(compiled(source, weight), model(source, weight), atol=0.01, rtol=0.03)
    assert any(phase.body.provider == "sm100_tma" for phase in backend.megabake_last_compiled.plan.phases)


@pytest.mark.parametrize("rows,token_tile,feature_tile,splits", [
    (9, 16, 64, 1), (17, 16, 128, 2), (9, 32, 64, 8),
    (17, 32, 128, 2), (32, 32, 64, 1), (128, 32, 128, 1),
])
def test_sm100_wider_token_tiles(monkeypatch, tmp_path, rows, token_tile, feature_tile, splits):
    import megabake.compiler as compiler
    from megabake import make_backend
    from megabake.bodies.cuda.sm90.gemm import split_k

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(feature_tile))
    monkeypatch.setenv("MEGABAKE_SM100_TOKEN_TILE", str(token_tile))
    monkeypatch.setenv("MEGABAKE_SM100_SPLIT_K", str(splits))
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_MAX_ROWS", "128")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    torch.manual_seed(551)
    args = (torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16) / 16,
            torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 16)
    model = Gemm().eval()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args), args):
            torch.testing.assert_close(compiled(*current), model(*current), atol=0.02, rtol=0.04)
    phase = next(phase for phase in backend.megabake_last_compiled.plan.phases
                 if phase.body.provider == "sm100_tma")
    assert phase.body.tiles.cta_shape == (token_tile, feature_tile)
    assert split_k(phase.body) == splits


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("features", [256, 19456])
def test_sm100_tcgen_two_dependent_gemms_and_rebinding(monkeypatch, tmp_path, tile, features):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class TwoGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(143)
    args = (
        torch.randn(4, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(features, 256, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(128, features, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = TwoGemm().eval()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for current in (args, tuple(tensor + 0.125 for tensor in args)):
            actual = compiled_model(*current)
            expected = model(*current)
            relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
            assert relative < 0.02
        compiled = backend.megabake_last_compiled
        assert compiled.generated.launch.launch_policy == "megabake_cooperative_table"
        assert sum(phase.body.provider == "sm100_tma" for phase in compiled.plan.phases) == 2
        assert backend.megabake_last_trace[-1].workload_kernel_launches == 1
        if features == 256 and tile == 128:
            unaligned = torch.empty(4 * 256 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(4, 256)
            unaligned.copy_(args[0])
            assert unaligned.data_ptr() % 16 != 0
            with pytest.raises(Exception, match="16-byte aligned"):
                compiled_model(unaligned, args[1], args[2])


@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("rows", [1, 6, 9, 17, 32])
def test_sm100_tcgen_multiple_token_tiles(monkeypatch, tmp_path, tile, rows):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_TILE", str(tile))
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class TwoGemm(torch.nn.Module):
        def forward(self, source, first, second):
            return (source @ first.T) @ second.T

    torch.manual_seed(243)
    args = (
        torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = TwoGemm().eval()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual = compiled_model(*args)
        expected = model(*args)
    relative = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
    assert relative < 0.02
    assert sum(phase.body.provider == "sm100_tma"
               for phase in backend.megabake_last_compiled.plan.phases) == 2
    assert backend.megabake_last_trace[-1].workload_kernel_launches == 1


def test_sm100_tcgen_one_token_decode(monkeypatch, tmp_path):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setenv("MEGABAKE_SM100_TCGEN_DECODE", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    args = (
        torch.randn(1, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(mode="decode", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="phase")
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual, expected = compiled_model(*args), model(*args)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    assert any(phase.body.provider == "sm100_tma"
               for phase in backend.megabake_last_compiled.plan.phases)


def test_sm100_event_scheduler_uses_fallback_gemm(monkeypatch, tmp_path):
    import megabake.compiler as compiler
    from megabake import make_backend

    monkeypatch.setenv("MEGABAKE_SM100_TCGEN", "1")
    monkeypatch.setattr(compiler, "_TABLE_LAUNCH_OPS", 0)

    class Gemm(torch.nn.Module):
        def forward(self, source, weight):
            return source @ weight.T

    args = (
        torch.randn(4, 128, device="cuda", dtype=torch.bfloat16) / 16,
        torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) / 16,
    )
    model = Gemm().eval()
    backend = make_backend(mode="prefill", compile_kernel=True, cache=False,
                           dump_dir=tmp_path, scheduler="region-events")
    compiled_model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        actual, expected = compiled_model(*args), model(*args)
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
    assert all(phase.body.provider != "sm100_tma"
               for phase in backend.megabake_last_compiled.plan.phases)
