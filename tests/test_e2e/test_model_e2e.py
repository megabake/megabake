"""Level 3 tests: end-to-end model execution through the megakernel.

Compile a PyTorch model → schedule → megakernel → compare with torch reference.
"""

import pytest
import torch

from megabake.schedule_compiler.graph_walker import compile_model
from megabake.runtime.loader import execute_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
SM_VERSION = 90


def _reference(model, x):
    with torch.no_grad():
        return model(x)


class TestLinearE2E:
    def test_linear_no_bias(self):
        """Single Linear(256, 512, bias=False) end-to-end."""
        model = torch.nn.Linear(256, 512, bias=False).cuda().half().eval()
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape, f"Shape mismatch: {result.shape} vs {ref.shape}"
        assert torch.allclose(result, ref, atol=1e-2, rtol=1e-2), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    def test_linear_larger(self):
        """Linear(512, 1024, bias=False) with batch=4."""
        model = torch.nn.Linear(512, 1024, bias=False).cuda().half().eval()
        x = torch.randn(4, 512, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=1e-2, rtol=1e-2), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )


class TestMLPE2E:
    def test_mlp_silu(self):
        """MLP: Linear + SiLU + Linear end-to-end."""

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 512, bias=False)
                self.fc2 = torch.nn.Linear(512, 256, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = MLP().cuda().half().eval()
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=5e-2, rtol=5e-2), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    def test_mlp_gelu(self):
        """MLP with GELU activation."""

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128, 256, bias=False)
                self.fc2 = torch.nn.Linear(256, 128, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

        model = MLP().cuda().half().eval()
        x = torch.randn(2, 128, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=5e-2, rtol=5e-2), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    def test_two_layer_mlp(self):
        """Two hidden layers: Linear + ReLU + Linear + SiLU + Linear."""

        class TwoLayerMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128, bias=False)
                self.fc2 = torch.nn.Linear(128, 128, bias=False)
                self.fc3 = torch.nn.Linear(128, 64, bias=False)

            def forward(self, x):
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.silu(self.fc2(x))
                return self.fc3(x)

        model = TwoLayerMLP().cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=5e-2, rtol=5e-2), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )


class TestTransformerBlockE2E:
    def test_rmsnorm_mlp(self):
        """RMSNorm + MLP block (decomposed RMSNorm, silu activation)."""

        class RMSNormMLP(torch.nn.Module):
            def __init__(self, dim=256, hidden=512):
                super().__init__()
                self.norm = torch.nn.RMSNorm(dim)
                self.fc1 = torch.nn.Linear(dim, hidden, bias=False)
                self.fc2 = torch.nn.Linear(hidden, dim, bias=False)

            def forward(self, x):
                x = self.norm(x)
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = RMSNormMLP().cuda().half().eval()
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=0.1, rtol=0.1), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    def test_layernorm_mlp(self):
        """LayerNorm + MLP block."""

        class LNBlock(torch.nn.Module):
            def __init__(self, dim=128, hidden=256):
                super().__init__()
                self.norm = torch.nn.LayerNorm(dim)
                self.fc1 = torch.nn.Linear(dim, hidden, bias=False)
                self.fc2 = torch.nn.Linear(hidden, dim, bias=False)

            def forward(self, x):
                x = self.norm(x)
                return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

        model = LNBlock().cuda().half().eval()
        x = torch.randn(2, 128, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        result = execute_model(compiled, model.state_dict(), [x])
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=0.1, rtol=0.1), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )


class TestTopLevelAPI:
    def test_compile_and_run(self):
        """Top-level megabake.compile() + megabake.run() API."""
        import megabake

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128, bias=False)
                self.fc2 = torch.nn.Linear(128, 64, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = MLP().cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)

        compiled = megabake.compile(model, x)
        result = megabake.run(compiled, model, x)
        ref = _reference(model, x)

        assert result.shape == ref.shape
        assert torch.allclose(result, ref, atol=5e-2, rtol=5e-2)


class TestCompileModelMetadata:
    def test_input_output_buffer_ids(self):
        """compile_model tracks input and output buffer IDs."""
        model = torch.nn.Linear(64, 128, bias=False).cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        assert len(compiled.input_buffer_ids) == 1
        assert compiled.output_buffer_id >= 0
        assert compiled.output_shape == [1, 128]
        assert compiled.num_buffers >= 2

    def test_mlp_buffer_counts(self):
        """MLP has correct number of buffers."""

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128, bias=False)
                self.fc2 = torch.nn.Linear(128, 64, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = MLP().cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)

        compiled = compile_model(model, x, sm_version=SM_VERSION)
        # input + 2 weights + mm_out + silu_out + mm_out2 = 6 minimum
        # (t.default nodes share buffers via views, don't allocate new ones)
        assert compiled.num_buffers >= 5
        assert compiled.output_shape == [1, 64]
