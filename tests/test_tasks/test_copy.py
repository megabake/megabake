"""Level 1 tests: copy task."""

import pytest
import torch

from megabake.data_types import OpType, TaskDesc
from megabake.runtime.launcher import run_single_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16


class TestCopy:
    def test_basic(self):
        n = 4096
        x = torch.randn(n, device=DEVICE, dtype=DTYPE)
        task = TaskDesc(
            op_type=OpType.COPY, op_code=0,
            dimensions=[n] + [0] * 7,
        )
        result = run_single_task(task, [x], [n], DTYPE)
        assert torch.equal(x, result)

    def test_large(self):
        n = 1_000_000
        x = torch.randn(n, device=DEVICE, dtype=DTYPE)
        task = TaskDesc(
            op_type=OpType.COPY, op_code=0,
            dimensions=[n] + [0] * 7,
        )
        result = run_single_task(task, [x], [n], DTYPE)
        assert torch.equal(x, result)
