"""Level 1 tests: matmul task."""

import pytest
import torch

from megabake.data_types import OpType, TaskDesc
from megabake.runtime.launcher import run_single_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16


class TestMatmul:
    def test_square_small(self):
        M, N, K = 256, 256, 256
        A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        B = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
        ref = A @ B
        task = TaskDesc(
            op_type=OpType.MATMUL, op_code=0,
            dimensions=[M, N, K] + [0] * 5,
        )
        result = run_single_task(task, [A, B], [M, N], DTYPE)
        assert torch.allclose(ref, result, atol=1.0, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_decode_shape(self):
        """batch=1 token through a 4096x4096 weight -- the critical decode path."""
        M, N, K = 1, 4096, 4096
        A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        B = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
        ref = A @ B
        task = TaskDesc(
            op_type=OpType.MATMUL, op_code=0,
            dimensions=[M, N, K] + [0] * 5,
        )
        result = run_single_task(task, [A, B], [M, N], DTYPE)
        assert torch.allclose(ref, result, atol=0.5, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    @pytest.mark.parametrize("M,N,K", [(1, 4096, 4096), (4, 8192, 4096), (2, 2048, 2048)])
    def test_skinny_transposed(self, M, N, K):
        """Skinny matmul with transposed B (the real decode hot path)."""
        A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        W = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        ref = A @ W.t()
        task = TaskDesc(
            op_type=OpType.MATMUL, op_code=0,
            dimensions=[M, N, K] + [0] * 5,
            strides=[1] + [0] * 7,
        )
        result = run_single_task(task, [A, W], [M, N], DTYPE)
        assert torch.allclose(ref, result, atol=0.5, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_rectangular(self):
        M, N, K = 128, 11008, 4096
        A = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        B = torch.randn(K, N, device=DEVICE, dtype=DTYPE)
        ref = A @ B
        task = TaskDesc(
            op_type=OpType.MATMUL, op_code=0,
            dimensions=[M, N, K] + [0] * 5,
        )
        result = run_single_task(task, [A, B], [M, N], DTYPE)
        assert torch.allclose(ref, result, atol=5.0, rtol=5e-2), \
            f"max diff: {(ref - result).abs().max().item()}"
