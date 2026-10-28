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
    assert any("blackwell_mma_16x32x16_epilogue" in candidate.description for candidate in candidates)

