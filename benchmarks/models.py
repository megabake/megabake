"""Benchmark workload registry: model + input factories for perf comparison."""

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn


@dataclass
class BenchmarkWorkload:
    name: str
    model_fn: Callable[[], nn.Module]
    input_fn: Callable[[], tuple[torch.Tensor, ...]]
    tags: list[str] = field(default_factory=lambda: ["baseline"])


# ---------------------------------------------------------------------------
# Model definitions (replicated from tests/test_e2e/)
# ---------------------------------------------------------------------------

class _MLPSiLU(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class _MLPGeLU(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        return self.fc3(x)


class _RMSNormMLP(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class _LNBlock(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _LlamaLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin):
        return self.layer(hidden_states, position_embeddings=(cos, sin))


def _make_llama_decoder():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, vocab_size=256, rms_norm_eps=1e-5,
        attn_implementation="sdpa",
    )
    layer = LlamaDecoderLayer(config, layer_idx=0).cuda().half().eval()
    return _LlamaLayerWrapper(layer)


def _cuda_half(shape):
    return torch.randn(shape, device="cuda", dtype=torch.float16)


# ---------------------------------------------------------------------------
# Workload registry
# ---------------------------------------------------------------------------

WORKLOADS: list[BenchmarkWorkload] = [
    BenchmarkWorkload(
        name="linear_256x512",
        model_fn=lambda: nn.Linear(256, 512, bias=False).cuda().half().eval(),
        input_fn=lambda: (_cuda_half((1, 256)),),
    ),
    BenchmarkWorkload(
        name="linear_512x1024",
        model_fn=lambda: nn.Linear(512, 1024, bias=False).cuda().half().eval(),
        input_fn=lambda: (_cuda_half((4, 512)),),
    ),
    BenchmarkWorkload(
        name="mlp_silu",
        model_fn=lambda: _MLPSiLU().cuda().half().eval(),
        input_fn=lambda: (_cuda_half((1, 256)),),
    ),
    BenchmarkWorkload(
        name="mlp_gelu",
        model_fn=lambda: _MLPGeLU().cuda().half().eval(),
        input_fn=lambda: (_cuda_half((2, 128)),),
    ),
    BenchmarkWorkload(
        name="mlp_3layer",
        model_fn=lambda: _TwoLayerMLP().cuda().half().eval(),
        input_fn=lambda: (_cuda_half((1, 64)),),
    ),
    BenchmarkWorkload(
        name="rmsnorm_mlp",
        model_fn=lambda: _RMSNormMLP().cuda().half().eval(),
        input_fn=lambda: (_cuda_half((1, 256)),),
    ),
    BenchmarkWorkload(
        name="layernorm_mlp",
        model_fn=lambda: _LNBlock().cuda().half().eval(),
        input_fn=lambda: (_cuda_half((2, 128)),),
    ),
    BenchmarkWorkload(
        name="llama_decoder",
        model_fn=_make_llama_decoder,
        input_fn=lambda: (
            _cuda_half((1, 8, 64)),
            torch.ones(1, 8, 32, device="cuda", dtype=torch.float16),
            torch.zeros(1, 8, 32, device="cuda", dtype=torch.float16),
        ),
        tags=["baseline", "llama"],
    ),
]


def get_workloads(
    tags: list[str] | None = None,
    names: list[str] | None = None,
) -> list[BenchmarkWorkload]:
    result = WORKLOADS
    if tags:
        result = [w for w in result if any(t in w.tags for t in tags)]
    if names:
        result = [w for w in result if w.name in names]
    return result
