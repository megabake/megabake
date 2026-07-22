"""Level 1 tests: embedding task."""

import pytest
import torch

from megabake.data_types import OpType, TaskDesc
from megabake.runtime.launcher import run_single_task

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"
DTYPE = torch.float16


class TestEmbedding:
    def test_basic(self):
        vocab_size, embed_dim = 32000, 4096
        num_indices = 128
        table = torch.randn(vocab_size, embed_dim, device=DEVICE, dtype=DTYPE)
        indices = torch.randint(0, vocab_size, (num_indices,), device=DEVICE, dtype=torch.int64)
        ref = table[indices]
        task = TaskDesc(
            op_type=OpType.EMBEDDING, op_code=0,
            dimensions=[num_indices, embed_dim, vocab_size] + [0] * 5,
        )
        result = run_single_task(task, [indices, table], [num_indices, embed_dim], DTYPE)
        assert torch.equal(ref, result)

    def test_single_token(self):
        vocab_size, embed_dim = 32000, 4096
        table = torch.randn(vocab_size, embed_dim, device=DEVICE, dtype=DTYPE)
        indices = torch.tensor([42], device=DEVICE, dtype=torch.int64)
        ref = table[indices]
        task = TaskDesc(
            op_type=OpType.EMBEDDING, op_code=0,
            dimensions=[1, embed_dim, vocab_size] + [0] * 5,
        )
        result = run_single_task(task, [indices, table], [1, embed_dim], DTYPE)
        assert torch.equal(ref, result)
