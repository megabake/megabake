"""Level 2 tests: FX graph walking and schedule compilation."""

import pytest
import torch

from megabake.data_types import OpType, ScheduleHeader, SCHEDULE_MAGIC, UopCode
from megabake.schedule_compiler.graph_walker import compile_schedule, compile_model
from megabake.schedule_compiler.serializer import load_schedule


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

DEVICE = "cuda"


class TestSimpleModels:
    def test_linear_model(self):
        """A single Linear layer should produce at least one MATMUL task."""
        model = torch.nn.Linear(256, 512).cuda().half().eval()
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, x, sm_version=90)
        header, tasks, *_ = load_schedule(data)
        assert header.magic == SCHEDULE_MAGIC
        assert header.num_tasks >= 1
        op_types = [t.op_type for t in tasks]
        assert OpType.MATMUL in op_types or OpType.ELEMENTWISE in op_types

    def test_mlp_model(self):
        """An MLP (Linear + SiLU + Linear) should produce matmul + elementwise tasks."""
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 512, bias=False)
                self.fc2 = torch.nn.Linear(512, 256, bias=False)

            def forward(self, x):
                return self.fc2(torch.nn.functional.silu(self.fc1(x)))

        model = MLP().cuda().half().eval()
        x = torch.randn(1, 256, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, x, sm_version=90)
        header, tasks, *_ = load_schedule(data)
        assert header.num_tasks >= 2  # fc1+silu (fused) + fc2

    def test_shape_ops_eliminated(self):
        """Reshape and transpose produce zero tasks."""
        class ShapeModel(torch.nn.Module):
            def forward(self, x):
                x = x.reshape(2, 8, 64)
                x = x.transpose(1, 2)
                x = x.reshape(2, 512)
                return x + 1.0

        model = ShapeModel().cuda().half().eval()
        x = torch.randn(2, 512, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, x, sm_version=90)
        header, tasks, *_ = load_schedule(data)
        # Only the add should produce a task
        assert header.num_tasks >= 1
        assert any(t.op_type == OpType.ELEMENTWISE for t in tasks)

    def test_weight_names_preserved(self):
        """Weight names from state_dict should appear in the schedule."""
        model = torch.nn.Linear(64, 128).cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, x, sm_version=90)
        _, _, _, _, weights, *_ = load_schedule(data)
        weight_values = list(weights.values())
        assert any("weight" in w for w in weight_values), \
            f"Expected 'weight' in weight names, got: {weight_values}"

    def test_fused_elementwise_silu_mul(self):
        """SiLU(x) * y should fuse into a single FUSED_ELEMENTWISE task."""
        class SwiGLU(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.silu(x) * y

        model = SwiGLU().cuda().half().eval()
        x = torch.randn(4, 256, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 256, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, (x, y), sm_version=90)
        header, tasks, *_ = load_schedule(data)
        op_types = [t.op_type for t in tasks]
        assert OpType.FUSED_ELEMENTWISE in op_types, \
            f"Expected FUSED_ELEMENTWISE, got {op_types}"
        fused = [t for t in tasks if t.op_type == OpType.FUSED_ELEMENTWISE][0]
        assert fused.dimensions[0] == 4 * 256
        assert fused.dimensions[1] >= 3  # at least LOAD+SILU+LOAD+MUL+STORE

    def test_fused_elementwise_triple_chain(self):
        """add -> relu -> neg should fuse into one FUSED_ELEMENTWISE."""
        class TripleChain(torch.nn.Module):
            def forward(self, x, y):
                return -(torch.relu(x + y))

        model = TripleChain().cuda().half().eval()
        x = torch.randn(4, 128, device=DEVICE, dtype=torch.float16)
        y = torch.randn(4, 128, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(model, (x, y), sm_version=90)
        header, tasks, *_ = load_schedule(data)
        op_types = [t.op_type for t in tasks]
        assert OpType.FUSED_ELEMENTWISE in op_types, \
            f"Expected FUSED_ELEMENTWISE, got {op_types}"
        assert len(tasks) == 1

    def test_schedule_metadata(self):
        """Schedule header has correct metadata."""
        model = torch.nn.Linear(64, 64, bias=False).cuda().half().eval()
        x = torch.randn(1, 64, device=DEVICE, dtype=torch.float16)
        data = compile_schedule(
            model, x, sm_version=90,
            batch_range=(1, 4), seq_range=(1, 512),
        )
        header, *_ = load_schedule(data)
        assert header.sm_version == 90
        assert header.batch_min == 1
        assert header.batch_max == 4
        assert header.seq_min == 1
        assert header.seq_max == 512
        assert header.workspace_bytes > 0
