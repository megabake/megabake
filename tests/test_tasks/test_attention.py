"""Level 1 tests: attention task."""

import pytest
import torch

from megabake.data_types import OpType, TaskDesc
from megabake.runtime.launcher import run_single_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16


class TestAttention:
    def test_small(self):
        B, H, S, D = 1, 4, 16, 64
        Q = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        K = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        V = torch.randn(B, H, S, D, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).half()
        task = TaskDesc(
            op_type=OpType.ATTENTION, op_code=0,
            dimensions=[B, H, S, D, S] + [0] * 3,
        )
        result = run_single_task(task, [Q, K, V], list(ref.shape), DTYPE)
        assert torch.allclose(ref, result, atol=5e-2, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_decode_single_token(self):
        """Single query token attending to 128 KV tokens."""
        B, H, Sq, D, Sk = 1, 32, 1, 128, 128
        Q = torch.randn(B, H, Sq, D, device=DEVICE, dtype=DTYPE)
        K = torch.randn(B, H, Sk, D, device=DEVICE, dtype=DTYPE)
        V = torch.randn(B, H, Sk, D, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.scaled_dot_product_attention(
            Q.float(), K.float(), V.float()
        ).half()
        task = TaskDesc(
            op_type=OpType.ATTENTION, op_code=0,
            dimensions=[B, H, Sq, D, Sk] + [0] * 3,
        )
        result = run_single_task(task, [Q, K, V], list(ref.shape), DTYPE)
        assert torch.allclose(ref, result, atol=5e-2, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"
