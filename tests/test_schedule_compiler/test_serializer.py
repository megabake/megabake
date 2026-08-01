"""Level 2 tests: schedule serialization round-trip."""

from megabake.data_types import (
    TaskDesc, ScheduleHeader, SMQueueEntry, OpType, ElemCode,
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
        header, tasks, buffers, wm, weights, *_ = load_schedule(data)
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
        header, tasks, buffers, wm, weights, *_ = load_schedule(data)
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

    def test_sm_queue_entry_round_trip(self):
        e = SMQueueEntry(task_id=42, tile_id=7)
        data = e.to_bytes()
        assert len(data) == SMQueueEntry.STRUCT_SIZE
        e2 = SMQueueEntry.from_bytes(data)
        assert e2.task_id == 42
        assert e2.tile_id == 7

    def test_header_scheduler_fields(self):
        h = ScheduleHeader(num_sms=32, max_queue_len=10, num_edges=5, scheduler_type=1)
        data = h.to_bytes()
        h2 = ScheduleHeader.from_bytes(data)
        assert h2.num_sms == 32
        assert h2.max_queue_len == 10
        assert h2.num_edges == 5
        assert h2.scheduler_type == 1


class TestSchedulerSerialization:
    def _make_tasks_and_placements(self):
        tasks = [
            TaskDesc(op_type=OpType.MATMUL, num_tiles=4,
                     buffer_indices=[0, 1, 2] + [0xFFFFFFFF] * 5,
                     dimensions=[64, 128, 256] + [0] * 5),
            TaskDesc(op_type=OpType.ELEMENTWISE, op_code=ElemCode.SILU,
                     num_tiles=1,
                     buffer_indices=[3, 0] + [0xFFFFFFFF] * 6,
                     dimensions=[64 * 128] + [0] * 7),
            TaskDesc(op_type=OpType.MATMUL, num_tiles=4,
                     buffer_indices=[4, 3, 5] + [0xFFFFFFFF] * 5,
                     dimensions=[64, 64, 128] + [0] * 5),
        ]
        placements = [Placement(i, i * 1024, 1024) for i in range(6)]
        return tasks, placements

    def test_scheduler_round_trip(self):
        tasks, placements = self._make_tasks_and_placements()
        sm_queues = [
            [(0, 0), (0, 1), (1, 0), (2, 0)],
            [(0, 2), (0, 3), (2, 1)],
            [(2, 2), (2, 3)],
        ]
        dep_count = [0, 1, 1]
        successors = [[1], [2], []]

        data = write_schedule(
            tasks=tasks, placements=placements, weight_names={},
            workspace_bytes=6144, sm_version=90, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 128),
            sm_queues=sm_queues, dep_count=dep_count,
            successors=successors, scheduler_type=1,
        )
        (header, tasks_out, _, _, _,
         sq_out, dc_out, tr_out, so_out, sl_out) = load_schedule(data)

        assert header.num_sms == 3
        assert header.max_queue_len == 4
        assert header.num_edges == 2
        assert header.scheduler_type == 1

        assert len(sq_out) == 3
        assert len(sq_out[0]) == 4
        assert sq_out[0][0].task_id == 0 and sq_out[0][0].tile_id == 0
        assert sq_out[0][2].task_id == 1 and sq_out[0][2].tile_id == 0
        assert len(sq_out[1]) == 3
        assert sq_out[1][2].task_id == 2 and sq_out[1][2].tile_id == 1
        assert len(sq_out[2]) == 2

        assert dc_out == [0, 1, 1]
        assert tr_out == [4, 1, 4]
        assert so_out == [0, 1, 2, 2]
        assert sl_out == [1, 2]

    def test_scheduler_with_weights(self):
        tasks, placements = self._make_tasks_and_placements()
        sm_queues = [[(0, 0), (1, 0), (2, 0)], [(0, 1), (2, 1)]]
        dep_count = [0, 1, 1]
        successors = [[1], [2], []]

        data = write_schedule(
            tasks=tasks, placements=placements,
            weight_names={2: "layer.weight", 5: "layer2.weight"},
            workspace_bytes=6144, sm_version=90, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 128),
            sm_queues=sm_queues, dep_count=dep_count,
            successors=successors, scheduler_type=1,
        )
        (header, _, _, _, weights,
         sq_out, dc_out, *_) = load_schedule(data)

        assert weights[2] == "layer.weight"
        assert weights[5] == "layer2.weight"
        assert header.num_sms == 2
        assert len(sq_out) == 2
        assert dc_out == [0, 1, 1]

    def test_no_scheduler_data(self):
        tasks, placements = self._make_tasks_and_placements()
        data = write_schedule(
            tasks=tasks, placements=placements, weight_names={},
            workspace_bytes=6144, sm_version=90, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 128),
        )
        (header, _, _, _, _,
         sq_out, dc_out, tr_out, so_out, sl_out) = load_schedule(data)

        assert header.num_sms == 0
        assert header.scheduler_type == 0
        assert sq_out is None
        assert dc_out is None
        assert tr_out is None
        assert so_out is None
        assert sl_out is None

    def test_no_edges(self):
        tasks = [
            TaskDesc(op_type=OpType.ELEMENTWISE, num_tiles=2,
                     buffer_indices=[0, 1] + [0xFFFFFFFF] * 6,
                     dimensions=[1024] + [0] * 7),
            TaskDesc(op_type=OpType.ELEMENTWISE, num_tiles=2,
                     buffer_indices=[2, 3] + [0xFFFFFFFF] * 6,
                     dimensions=[1024] + [0] * 7),
        ]
        placements = [Placement(i, i * 256, 256) for i in range(4)]
        sm_queues = [[(0, 0), (1, 0)], [(0, 1), (1, 1)]]
        dep_count = [0, 0]
        successors = [[], []]

        data = write_schedule(
            tasks=tasks, placements=placements, weight_names={},
            workspace_bytes=1024, sm_version=80, compute_dtype=0,
            batch_range=(1, 1), seq_range=(1, 64),
            sm_queues=sm_queues, dep_count=dep_count,
            successors=successors, scheduler_type=1,
        )
        (header, _, _, _, _,
         sq_out, dc_out, _, so_out, sl_out) = load_schedule(data)

        assert header.num_edges == 0
        assert dc_out == [0, 0]
        assert so_out == [0, 0, 0]
        assert sl_out == []
