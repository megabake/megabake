"""Level 1 tests: elementwise task operations."""

import pytest
import torch

from megabake.data_types import OpType, ElemCode, TaskDesc
from megabake.runtime.launcher import run_single_task


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16
N = 4096


class TestElementwiseAdd:
    def test_basic(self):
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = a + b
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.ADD,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [a, b], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-3), \
            f"max diff: {(ref - result).abs().max().item()}"

    def test_large(self):
        n = 1_000_000
        a = torch.randn(n, device=DEVICE, dtype=DTYPE)
        b = torch.randn(n, device=DEVICE, dtype=DTYPE)
        ref = a + b
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.ADD,
            dimensions=[n] + [0] * 7,
        )
        result = run_single_task(task, [a, b], [n], DTYPE)
        assert torch.allclose(ref, result, atol=1e-3)


class TestElementwiseMul:
    def test_basic(self):
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = a * b
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.MUL,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [a, b], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-3)


class TestElementwiseSub:
    def test_basic(self):
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = a - b
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.SUB,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [a, b], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-3)


class TestElementwiseSiLU:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.silu(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.SILU,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestElementwiseGELU:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.gelu(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.GELU,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=2e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestElementwiseReLU:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.relu(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.RELU,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-5)


class TestElementwiseSigmoid:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.sigmoid(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.SIGMOID,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-3)


class TestElementwiseExp:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE) * 0.5
        ref = torch.exp(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.EXP,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2)


class TestElementwiseNeg:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = -x
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.NEG,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-5)


class TestElementwiseAbs:
    def test_basic(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.abs(x)
        task = TaskDesc(
            op_type=OpType.ELEMENTWISE,
            op_code=ElemCode.ABS,
            dimensions=[N] + [0] * 7,
        )
        result = run_single_task(task, [x], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-5)
