"""Level 3 tests: LLaMA decoder layer end-to-end through the megakernel."""

import pytest
import torch

from megabake.schedule_compiler.graph_walker import compile_model
from megabake.runtime.loader import execute_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

SM_VERSION = 90


class LlamaLayerWrapper(torch.nn.Module):
    """Wraps HF LlamaDecoderLayer to take positional args for torch.export."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin):
        return self.layer(
            hidden_states, position_embeddings=(cos, sin)
        )


def _make_llama_layer():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        vocab_size=256,
        rms_norm_eps=1e-5,
        attn_implementation="sdpa",
    )
    layer = LlamaDecoderLayer(config, layer_idx=0).cuda().half().eval()
    return LlamaLayerWrapper(layer)


class TestLlamaLayerE2E:
    def test_single_decoder_layer(self):
        """Single LLaMA decoder layer: RMSNorm -> SDPA -> residual -> RMSNorm -> gated SiLU MLP -> residual."""
        model = _make_llama_layer()

        B, S, D = 1, 8, 64
        head_dim = 32
        hidden = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
        cos = torch.ones(B, S, head_dim, device="cuda", dtype=torch.float16)
        sin = torch.zeros(B, S, head_dim, device="cuda", dtype=torch.float16)

        compiled = compile_model(model, (hidden, cos, sin), sm_version=SM_VERSION)

        with torch.no_grad():
            ref = model(hidden, cos, sin)

        result = execute_model(
            compiled, model.state_dict(), [hidden, cos, sin]
        )

        assert result.shape == ref.shape, f"Shape: {result.shape} vs {ref.shape}"
        assert torch.allclose(result, ref, atol=0.15, rtol=0.15), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )
