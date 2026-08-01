"""Level 1 tests: elementwise task operations."""

import pytest
import torch

from megabake.data_types import OpType, ElemCode, UopCode, TaskDesc, pack_uop
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


class TestFusedElementwiseBasic:
    """Fused chain within 8-uop limit: add → silu (4 uops: 2 LOAD + ADD + SILU + STORE = 5)."""

    def test_add_silu(self):
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.nn.functional.silu(a + b)
        # r0=LOAD slot1, r1=LOAD slot2, r2=ADD r0,r1, r3=SILU r2, STORE slot0,r3
        uops = [
            pack_uop(UopCode.LOAD, 0, 1),
            pack_uop(UopCode.LOAD, 1, 2),
            pack_uop(UopCode.ADD, 2, 0, 1),
            pack_uop(UopCode.SILU, 3, 2),
            pack_uop(UopCode.STORE, 0, 3),
        ]
        task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            dimensions=[N, len(uops)] + [0] * 6,
            strides=uops + [0] * (8 - len(uops)),
        )
        result = run_single_task(task, [a, b], [N], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestFusedElementwiseExpanded:
    """Chain exceeding 8-uop limit: uses program buffer for overflow."""

    def test_long_chain(self):
        a = torch.randn(N, device=DEVICE, dtype=DTYPE)
        b = torch.randn(N, device=DEVICE, dtype=DTYPE)
        c = torch.randn(N, device=DEVICE, dtype=DTYPE)
        # ref: silu(neg(exp(sigmoid(mul(add(a, b), c)))))
        ref = torch.nn.functional.silu(
            -torch.exp(torch.sigmoid((a + b) * c)))
        # r0=a, r1=b, r2=a+b, r3=c, r4=(a+b)*c, r5=sigmoid, r6=exp, r7=neg, r8=silu
        # 3 LOADs + 5 ops + STORE = 9 uops (exceeds 8)
        uops = [
            pack_uop(UopCode.LOAD, 0, 1),       # r0 = a
            pack_uop(UopCode.LOAD, 1, 2),        # r1 = b
            pack_uop(UopCode.ADD, 2, 0, 1),      # r2 = a+b
            pack_uop(UopCode.LOAD, 3, 3),         # r3 = c
            pack_uop(UopCode.MUL, 4, 2, 3),      # r4 = (a+b)*c
            pack_uop(UopCode.SIGMOID, 5, 4),     # r5 = sigmoid(r4)
            pack_uop(UopCode.EXP, 6, 5),          # r6 = exp(r5)
            pack_uop(UopCode.NEG, 7, 6),          # r7 = -r6
            pack_uop(UopCode.SILU, 8, 7),         # r8 = silu(r7)
            pack_uop(UopCode.STORE, 0, 8),        # out = r8
        ]
        # First 8 in strides, overflow (uops[8:]) in program buffer
        prog_buf = torch.tensor(uops[8:], dtype=torch.int32, device=DEVICE)
        # dimensions[2] = buffer index of program buffer (index 4 in ptrs array)
        task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            dimensions=[N, len(uops), 4] + [0] * 5,
            strides=list(uops[:8]),
        )
        result = run_single_task(task, [a, b, c, prog_buf], [N], DTYPE)
        assert torch.allclose(ref, result, atol=5e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestFusedElementwiseBroadcast:
    """Fused chain with broadcast LOAD: add(tensor[N], bias[M]) where M < N."""

    def test_add_broadcast_silu(self):
        rows, cols = 32, 128
        n = rows * cols
        x = torch.randn(n, device=DEVICE, dtype=DTYPE)
        bias = torch.randn(cols, device=DEVICE, dtype=DTYPE)
        # Broadcast: bias repeats every `cols` elements, `rows` times
        ref = torch.nn.functional.silu(
            x + bias.repeat(rows))
        # LOAD r0 slot1 (x), LOAD_BROADCAST r1 slot2 (bias), ADD r2, SILU r3, STORE
        bc_info = (cols << 16) | 1  # numel=cols, repeat=1 (cyclic)
        uops = [
            pack_uop(UopCode.LOAD, 0, 1),
            pack_uop(UopCode.LOAD_BROADCAST, 1, 2, 0),  # s2=0 → dims[3]
            pack_uop(UopCode.ADD, 2, 0, 1),
            pack_uop(UopCode.SILU, 3, 2),
            pack_uop(UopCode.STORE, 0, 3),
        ]
        task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            dimensions=[n, len(uops), 0, bc_info] + [0] * 4,
            strides=uops + [0] * (8 - len(uops)),
        )
        result = run_single_task(task, [x, bias], [n], DTYPE)
        assert torch.allclose(ref, result, atol=1e-2), \
            f"max diff: {(ref - result).abs().max().item()}"


class TestFusedElementwiseReduceSum:
    """Reduction uop: exp(x) → sum."""

    def test_exp_reduce_sum(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = torch.exp(x.float()).sum()
        uops = [
            pack_uop(UopCode.LOAD, 0, 1),
            pack_uop(UopCode.EXP, 1, 0),
            pack_uop(UopCode.REDUCE_SUM, 2, 1),
        ]
        # ponytail: num_tiles=1 for reduce — cross-tile atomics deferred
        task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            num_tiles=1,
            dimensions=[N, len(uops)] + [0] * 6,
            strides=uops + [0] * (8 - len(uops)),
        )
        result = run_single_task(task, [x], [1], DTYPE)
        ref_half = ref.half()
        assert torch.allclose(ref_half, result, rtol=0.05), \
            f"ref={ref_half.item():.4f} got={result.item():.4f}"


class TestFusedElementwiseReduceMean:
    """Reduction uop: mean(x)."""

    def test_reduce_mean(self):
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)
        ref = x.float().mean()
        uops = [
            pack_uop(UopCode.LOAD, 0, 1),
            pack_uop(UopCode.REDUCE_MEAN, 1, 0),
        ]
        task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            num_tiles=1,
            dimensions=[N, len(uops)] + [0] * 6,
            strides=uops + [0] * (8 - len(uops)),
        )
        result = run_single_task(task, [x], [1], DTYPE)
        ref_half = ref.half()
        assert torch.allclose(ref_half, result, rtol=0.1, atol=1e-2), \
            f"ref={ref_half.item():.4f} got={result.item():.4f}"
