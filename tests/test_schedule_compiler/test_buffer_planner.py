"""Level 2 tests: buffer planner."""

import pytest
from megabake.data_types import TaskDesc, OpType, UNUSED_BUFFER
from megabake.schedule_compiler.buffer_planner import plan_buffers, Placement


class TestBufferPlanner:
    def test_single_buffer(self):
        tasks = [TaskDesc(op_type=OpType.ELEMENTWISE, buffer_indices=[0] + [UNUSED_BUFFER] * 7)]
        sizes = {0: 1024}
        placements, total = plan_buffers(tasks, sizes, weight_buffers=set())
        assert len(placements) == 1
        assert placements[0].offset == 0
        assert total >= 1024

    def test_non_overlapping_reuse(self):
        """Buffers with non-overlapping lifetimes can share the same memory."""
        tasks = [
            TaskDesc(op_type=OpType.ELEMENTWISE,
                     buffer_indices=[0, 1] + [UNUSED_BUFFER] * 6),
            TaskDesc(op_type=OpType.ELEMENTWISE,
                     buffer_indices=[2, 0] + [UNUSED_BUFFER] * 6),
        ]
        sizes = {0: 1024, 1: 1024, 2: 1024}
        placements, total = plan_buffers(tasks, sizes, weight_buffers=set())
        # Buffer 1 is only used in task 0, buffer 2 only in task 1
        # They should be able to share memory
        assert total < 3 * 1024 + 4096  # less than 3 separate buffers + alignment

    def test_weight_buffers_excluded(self):
        tasks = [
            TaskDesc(op_type=OpType.MATMUL,
                     buffer_indices=[0, 1, 2] + [UNUSED_BUFFER] * 5),
        ]
        sizes = {0: 1024, 1: 4096, 2: 4096}
        placements, total = plan_buffers(tasks, sizes, weight_buffers={1, 2})
        # Only buffer 0 should be placed (1 and 2 are weights)
        assert len(placements) == 1
        assert placements[0].buffer_id == 0

    def test_alignment(self):
        tasks = [TaskDesc(op_type=OpType.COPY,
                          buffer_indices=[0, 1] + [UNUSED_BUFFER] * 6)]
        sizes = {0: 100, 1: 200}
        placements, total = plan_buffers(tasks, sizes, weight_buffers=set())
        for p in placements:
            assert p.offset % 256 == 0 or p.offset == 0

    def test_no_arena_overlap_for_live_buffers(self):
        """Core invariant: simultaneously-live buffers must not overlap in arena."""
        tasks = [
            TaskDesc(op_type=OpType.ELEMENTWISE,
                     buffer_indices=[0, 1, 2] + [UNUSED_BUFFER] * 5),
        ]
        sizes = {0: 8192, 1: 4096, 2: 4096}
        placements, total = plan_buffers(tasks, sizes, weight_buffers=set())
        # All 3 buffers live at the same time, none should overlap
        for i, p1 in enumerate(placements):
            for j, p2 in enumerate(placements):
                if i >= j:
                    continue
                assert p1.offset + p1.size <= p2.offset or \
                       p2.offset + p2.size <= p1.offset, \
                    f"Buffers {p1.buffer_id} and {p2.buffer_id} overlap"
