"""Level 2 tests: schedule serialization round-trip."""

import pytest
from megabake.data_types import (
    TaskDesc, ScheduleHeader, OpType, ElemCode,
    SCHEDULE_MAGIC, SCHEDULE_VERSION,
)
from megabake.schedule_compiler.buffer_planner import Placement
from megabake.schedule_compiler.serializer import write_schedule, load_schedule


class TestSerializerRoundTrip:
    def test_empty_schedule(self):
        data = write_schedule(
            tasks=[], placements=[], weight_names={},
            workspace_bytes=0, sm_version=90, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 128),
        )
        header, tasks, buffers, wm, weights = load_schedule(data)
        assert header.magic == SCHEDULE_MAGIC
        assert header.version == SCHEDULE_VERSION
        assert header.num_tasks == 0
        assert header.sm_version == 90

    def test_single_task_round_trip(self):
        t = TaskDesc(
            op_type=OpType.ELEMENTWISE, op_code=ElemCode.ADD,
            num_tiles=32,
            buffer_indices=[0, 1, 2] + [0xFFFFFFFF] * 5,
            dimensions=[4096] + [0] * 7,
        )
        placements = [
            Placement(buffer_id=0, offset=0, size=8192),
            Placement(buffer_id=1, offset=8192, size=8192),
            Placement(buffer_id=2, offset=16384, size=8192),
        ]
        data = write_schedule(
            tasks=[t], placements=placements,
            weight_names={1: "model.weight"},
            workspace_bytes=24576, sm_version=90, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 128),
        )
        header, tasks, buffers, wm, weights = load_schedule(data)
        assert header.num_tasks == 1
        assert header.workspace_bytes == 24576
        assert tasks[0].op_type == OpType.ELEMENTWISE
        assert tasks[0].op_code == ElemCode.ADD
        assert tasks[0].num_tiles == 32
        assert tasks[0].dimensions[0] == 4096
        assert len(buffers) == 3
        assert weights[1] == "model.weight"

    def test_multiple_tasks(self):
        tasks_in = [
            TaskDesc(op_type=OpType.EMBEDDING, num_tiles=1,
                     dimensions=[128, 4096, 32000] + [0] * 5),
            TaskDesc(op_type=OpType.MATMUL, num_tiles=32,
                     dimensions=[128, 4096, 4096] + [0] * 5),
            TaskDesc(op_type=OpType.ELEMENTWISE, op_code=ElemCode.SILU,
                     num_tiles=8, dimensions=[128 * 4096] + [0] * 7),
        ]
        placements = [Placement(i, i * 256, 256) for i in range(5)]
        data = write_schedule(
            tasks=tasks_in, placements=placements, weight_names={},
            workspace_bytes=4096, sm_version=90, compute_dtype=0,
            batch_range=(1, 4), seq_range=(1, 2048),
        )
        header, tasks_out, *_ = load_schedule(data)
        assert header.num_tasks == 3
        assert tasks_out[0].op_type == OpType.EMBEDDING
        assert tasks_out[1].op_type == OpType.MATMUL
        assert tasks_out[2].op_type == OpType.ELEMENTWISE
        assert header.batch_max == 4
        assert header.seq_max == 2048


class TestStructSizes:
    def test_task_desc_size(self):
        t = TaskDesc(op_type=1)
        assert len(t.to_bytes()) == TaskDesc.STRUCT_SIZE

    def test_header_size(self):
        h = ScheduleHeader()
        assert len(h.to_bytes()) == ScheduleHeader.STRUCT_SIZE

    def test_task_round_trip(self):
        t = TaskDesc(
            op_type=OpType.MATMUL, op_code=0, num_tiles=132,
            buffer_indices=[0, 1, 2, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
            dimensions=[4096, 4096, 4096, 0, 0, 0, 0, 0],
            strides=[-1, 2, 3, 0, 0, 0, 0, 0],
        )
        data = t.to_bytes()
        t2 = TaskDesc.from_bytes(data)
        assert t2.op_type == t.op_type
        assert t2.num_tiles == t.num_tiles
        assert t2.buffer_indices == t.buffer_indices
        assert t2.dimensions == t.dimensions
        assert t2.strides == t.strides
