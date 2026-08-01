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


def _run_attention(B, H, Sq, D, Sk, num_kv_heads=0, is_causal=False):
    """Helper: run megabake attention and compare to PyTorch reference."""
    Q = torch.randn(B, H, Sq, D, device=DEVICE, dtype=DTYPE)
    Hk = num_kv_heads if num_kv_heads else H
    K = torch.randn(B, Hk, Sk, D, device=DEVICE, dtype=DTYPE)
    V = torch.randn(B, Hk, Sk, D, device=DEVICE, dtype=DTYPE)

    # Expand KV for GQA reference
    if Hk != H:
        rep = H // Hk
        K_ref = K.repeat_interleave(rep, dim=1)
        V_ref = V.repeat_interleave(rep, dim=1)
    else:
        K_ref, V_ref = K, V

    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_ref.float(), V_ref.float(), is_causal=is_causal
    ).half()

    task = TaskDesc(
        op_type=OpType.ATTENTION, op_code=0,
        dimensions=[B, H, Sq, D, Sk, num_kv_heads] + [0] * 2,
        strides=[1 if is_causal else 0] + [0] * 7,
    )
    result = run_single_task(task, [Q, K, V], list(ref.shape), DTYPE)
    max_diff = (ref.float() - result.float()).abs().max().item()
    return ref, result, max_diff


class TestAttention:
    def test_small(self):
        B, H, S, D = 1, 4, 16, 64
        _, _, diff = _run_attention(B, H, S, D, S)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_decode_single_token(self):
        B, H, Sq, D, Sk = 1, 32, 1, 128, 128
        _, _, diff = _run_attention(B, H, Sq, D, Sk)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_causal(self):
        B, H, S, D = 1, 4, 32, 64
        _, _, diff = _run_attention(B, H, S, D, S, is_causal=True)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_causal_decode(self):
        B, H, Sq, D, Sk = 1, 8, 1, 64, 64
        _, _, diff = _run_attention(B, H, Sq, D, Sk, is_causal=True)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_gqa(self):
        """Grouped-query attention: 8 query heads, 2 KV heads."""
        B, H, S, D, Hk = 1, 8, 16, 64, 2
        _, _, diff = _run_attention(B, H, S, D, S, num_kv_heads=Hk)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_gqa_causal(self):
        B, H, S, D, Hk = 1, 8, 32, 64, 2
        _, _, diff = _run_attention(B, H, S, D, S, num_kv_heads=Hk, is_causal=True)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_long_sequence_k_tiling(self):
        """seq_k > ATTN_BK (64): exercises K-block tiling."""
        B, H, Sq, D, Sk = 1, 4, 1, 64, 256
        _, _, diff = _run_attention(B, H, Sq, D, Sk)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_long_causal_k_tiling(self):
        """Causal with seq > BK: tests causal early exit across K-blocks."""
        B, H, S, D = 1, 4, 128, 64
        _, _, diff = _run_attention(B, H, S, D, S, is_causal=True)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_batch2(self):
        B, H, S, D = 2, 4, 16, 64
        _, _, diff = _run_attention(B, H, S, D, S)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_head_dim_128(self):
        B, H, S, D = 1, 4, 16, 128
        _, _, diff = _run_attention(B, H, S, D, S)
        assert diff < 5e-2, f"max diff: {diff}"

    def test_seq_q_gt_seq_k(self):
        """More queries than keys (unusual but valid)."""
        B, H, Sq, D, Sk = 1, 4, 32, 64, 8
        _, _, diff = _run_attention(B, H, Sq, D, Sk)
        assert diff < 5e-2, f"max diff: {diff}"
