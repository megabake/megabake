"""Arena buffer planner with liveness analysis and first-fit packing."""

from dataclasses import dataclass
from megabake.data_types import TaskDesc, UNUSED_BUFFER


@dataclass
class LiveInterval:
    buffer_id: int
    size: int
    first_use: int
    last_use: int


@dataclass
class Placement:
    buffer_id: int
    offset: int
    size: int


def plan_buffers(
    tasks: list[TaskDesc],
    buffer_sizes: dict[int, int],
    weight_buffers: set[int],
) -> tuple[list[Placement], int]:
    intervals: dict[int, LiveInterval] = {}

    for task_idx, task in enumerate(tasks):
        for slot in range(8):
            buf_id = task.buffer_indices[slot]
            if buf_id == UNUSED_BUFFER or buf_id in weight_buffers:
                continue
            if buf_id not in intervals:
                intervals[buf_id] = LiveInterval(
                    buffer_id=buf_id,
                    size=buffer_sizes[buf_id],
                    first_use=task_idx,
                    last_use=task_idx,
                )
            else:
                intervals[buf_id].last_use = task_idx

    sorted_intervals = sorted(intervals.values(), key=lambda iv: -iv.size)

    placements: list[Placement] = []

    def _overlaps_any(offset: int, size: int, iv: LiveInterval) -> bool:
        for p in placements:
            existing = intervals[p.buffer_id]
            arena_overlap = (offset < p.offset + p.size
                             and p.offset < offset + size)
            time_overlap = (iv.first_use <= existing.last_use
                            and existing.first_use <= iv.last_use)
            if arena_overlap and time_overlap:
                return True
        return False

    for iv in sorted_intervals:
        offset = 0
        while _overlaps_any(offset, iv.size, iv):
            best_jump = offset + 1
            for p in placements:
                existing = intervals[p.buffer_id]
                arena_overlap = (offset < p.offset + p.size
                                 and p.offset < offset + iv.size)
                time_overlap = (iv.first_use <= existing.last_use
                                and existing.first_use <= iv.last_use)
                if arena_overlap and time_overlap:
                    best_jump = max(best_jump, p.offset + p.size)
            offset = (best_jump + 255) & ~255

        placements.append(Placement(buffer_id=iv.buffer_id,
                                     offset=offset, size=iv.size))

    total = 0
    for p in placements:
        total = max(total, p.offset + p.size)
    total = (total + 4095) & ~4095

    return placements, total
