from __future__ import annotations

import pytest
import torch

from megabake import make_backend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class LayerNorm(torch.nn.Module):
    def forward(self, source, weight, bias):
        centered = source - source.mean(dim=-1, keepdim=True)
        variance = (centered * centered).sum(dim=-1, keepdim=True) / source.shape[-1]
        return ((centered * torch.rsqrt(variance + 1e-5)) * weight + bias).bfloat16()


@pytest.mark.parametrize("width", [7, 128, 1024])
@pytest.mark.parametrize("strided", [False, True])
def test_layer_norm_body(tmp_path, width, strided):
    torch.manual_seed(312)
    source = torch.randn(2, 3, width * (2 if strided else 1), device="cuda")
    if strided:
        source = source[..., ::2]
    weight = torch.randn(width, device="cuda")
    bias = torch.randn(width, device="cuda")
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path)
    model = LayerNorm()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for offset in (0.0, 0.25):
            source.add_(offset)
            actual = compiled(source, weight, bias)
            torch.testing.assert_close(actual, model(source, weight, bias), atol=0.015625, rtol=0.008)
    selections = backend.megabake_last_compiled.plan.selection.selections
    assert any(selection.program.source_target == "layer_norm" for selection in selections)


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] not in {10, 12},
                    reason="Blackwell required")
def test_blackwell_bias_residual_epilogue(tmp_path):
    def model(source, weight, bias, residual):
        return torch.nn.functional.gelu(source @ weight.T + bias).float() + residual

    torch.manual_seed(313)
    inputs = (torch.randn(5, 70, device="cuda", dtype=torch.bfloat16) / 8,
              torch.randn(33, 70, device="cuda", dtype=torch.bfloat16) / 8,
              torch.randn(33, device="cuda", dtype=torch.bfloat16) / 8,
              torch.randn(5, 33, device="cuda"))
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, model(*inputs), atol=0.004, rtol=0.01)
    candidates = backend.megabake_last_compiled.search.candidates
    assert any("blackwell_mma_32x32x16_epilogue" in candidate.description for candidate in candidates)


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] not in {10, 12},
                    reason="Blackwell required")
def test_blackwell_epilogue_supports_rank_three_output(tmp_path):
    def model(source, weight, residual):
        return residual + (source @ weight.T).view(1, 4, 32)

    torch.manual_seed(314)
    inputs = (torch.randn(4, 8, device="cuda", dtype=torch.bfloat16),
              torch.randn(32, 8, device="cuda", dtype=torch.bfloat16),
              torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16))
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, model(*inputs))
    candidates = backend.megabake_last_compiled.search.candidates
    assert any(candidate.status == "measured" and
               "blackwell_mma_32x32x16_epilogue" in candidate.description
               for candidate in candidates)


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] not in {10, 12},
                    reason="Blackwell required")
def test_blackwell_epilogue_rejects_narrow_output(tmp_path):
    def model(source, weight, bias):
        return torch.nn.functional.linear(source, weight, bias).float()

    torch.manual_seed(316)
    inputs = (torch.randn(9, 1024, device="cuda", dtype=torch.bfloat16),
              torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16),
              torch.randn(1, device="cuda", dtype=torch.bfloat16))
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, model(*inputs))
    selections = backend.megabake_last_compiled.plan.selection.selections
    assert not any(selection.body.name == "blackwell_mma_32x32x16_epilogue" for selection in selections)


def test_parameter_cast_cache_refreshes_after_mutation(tmp_path, monkeypatch):
    import megabake.compiler as compiler_module

    monkeypatch.setattr(compiler_module, "_TABLE_LAUNCH_OPS", 0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))

        def forward(self, source):
            return source @ self.weight.to(source.dtype)

    torch.manual_seed(317)
    model = Model().eval()
    source = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    backend = make_backend(compile_kernel=True, cache=False, dump_dir=tmp_path, scheduler="phase")
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        first = compiled(source)
        torch.testing.assert_close(first, model(source), atol=0.04, rtol=0.01)
        state = backend.megabake_last_callable.megabake_runtime_state
        assert state._cached_parameter_phases
        torch.cuda.synchronize()
        column = -2 if state.generated.launch.stream_table else -1
        assert all(
            state._phase_table[0][phase, column].item() == 0
            for phase in state._cached_parameter_phases
        )
        model.weight.add_(0.25)
        second = compiled(source)
        torch.testing.assert_close(second, model(source), atol=0.04, rtol=0.01)
        assert not torch.equal(first, second)
