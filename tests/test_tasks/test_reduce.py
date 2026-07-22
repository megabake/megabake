"""Level 1 tests: reduce task operations."""

import pytest
import torch

from megabake.data_types import OpType, ReduceCode, TaskDesc
from megabake.runtime.launcher import run_single_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16


class TestRMSNorm:
    def test_basic(self):
        rows, cols = 8, 4096
        x = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE)
        w = torch.ones(cols, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.rms_norm(x, (cols,), w, 1e-5)
        task = TaskDesc(
            op_type=OpType.REDUCE, op_code=ReduceCode.RMSNORM,
            dimensions=[rows, cols] + [0] * 6,
        )
        result = run_single_task(task, [x, w], [rows, cols], DTYPE)
        assert torch.allclose(ref, result, atol=5e-2, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_with_weight(self):
        rows, cols = 4, 2048
        x = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE)
        w = torch.randn(cols, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.rms_norm(x, (cols,), w, 1e-5)
        task = TaskDesc(
            op_type=OpType.REDUCE, op_code=ReduceCode.RMSNORM,
            dimensions=[rows, cols] + [0] * 6,
        )
        result = run_single_task(task, [x, w], [rows, cols], DTYPE)
        assert torch.allclose(ref, result, atol=5e-2, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestSoftmax:
    def test_basic(self):
        rows, cols = 8, 128
        x = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE)
        ref = torch.softmax(x.float(), dim=-1).half()
        task = TaskDesc(
            op_type=OpType.REDUCE, op_code=ReduceCode.SOFTMAX,
            dimensions=[rows, cols] + [0] * 6,
        )
        result = run_single_task(task, [x], [rows, cols], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_large_row(self):
        rows, cols = 4, 4096
        x = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE)
        ref = torch.softmax(x.float(), dim=-1).half()
        task = TaskDesc(
            op_type=OpType.REDUCE, op_code=ReduceCode.SOFTMAX,
            dimensions=[rows, cols] + [0] * 6,
        )
        result = run_single_task(task, [x], [rows, cols], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2)


class TestLayerNorm:
    def test_basic(self):
        rows, cols = 8, 1024
        x = torch.randn(rows, cols, device=DEVICE, dtype=DTYPE)
        w = torch.ones(cols, device=DEVICE, dtype=DTYPE)
        b = torch.zeros(cols, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.layer_norm(x.float(), (cols,), w.float(), b.float()).half()
        task = TaskDesc(
            op_type=OpType.REDUCE, op_code=ReduceCode.LAYERNORM,
            dimensions=[rows, cols] + [0] * 6,
        )
        result = run_single_task(task, [x, w, b], [rows, cols], DTYPE)
        assert torch.allclose(ref, result, atol=5e-2, rtol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"
