"""Serialize/deserialize .schedule binary files."""

from megabake.data_types import (
    TaskDesc, BufferDesc, WeightMapping, ScheduleHeader,
    SCHEDULE_MAGIC, SCHEDULE_VERSION,
)
from megabake.schedule_compiler.buffer_planner import Placement


def write_schedule(
    tasks: list[TaskDesc],
    placements: list[Placement],
    weight_names: dict[int, str],
    workspace_bytes: int,
    sm_version: int,
    compute_dtype: int,
    batch_range: tuple[int, int],
    seq_range: tuple[int, int],
) -> bytes:
    # Build BufferDesc list from placements
    buffer_descs = []
    for p in placements:
        buffer_descs.append(BufferDesc(offset=p.offset, size=p.size, dtype=compute_dtype, buffer_id=p.buffer_id))

    # Build weight mappings + string table
    string_table = b""
    weight_mappings = []
    for buf_id, key in sorted(weight_names.items()):
        key_bytes = key.encode("utf-8")
        weight_mappings.append(WeightMapping(
            buffer_index=buf_id,
            key_offset=len(string_table),
            key_length=len(key_bytes),
        ))
        string_table += key_bytes + b"\x00"

    header = ScheduleHeader(
        magic=SCHEDULE_MAGIC,
        version=SCHEDULE_VERSION,
        num_tasks=len(tasks),
        num_buffers=len(buffer_descs),
        workspace_bytes=workspace_bytes,
        num_weight_mappings=len(weight_mappings),
        batch_min=batch_range[0],
        batch_max=batch_range[1],
        seq_min=seq_range[0],
        seq_max=seq_range[1],
        sm_version=sm_version,
        compute_dtype=compute_dtype,
    )

    data = header.to_bytes()
    for t in tasks:
        data += t.to_bytes()
    for b in buffer_descs:
        data += b.to_bytes()
    for wm in weight_mappings:
        data += wm.to_bytes()
    data += string_table

    return data


def load_schedule(data: bytes) -> tuple[
    ScheduleHeader, list[TaskDesc], list[BufferDesc],
    list[WeightMapping], dict[int, str]
]:
    header = ScheduleHeader.from_bytes(data[:ScheduleHeader.STRUCT_SIZE])
    assert header.magic == SCHEDULE_MAGIC, "Not a megabake schedule file"
    assert header.version == SCHEDULE_VERSION

    off = ScheduleHeader.STRUCT_SIZE
    tasks = []
    for _ in range(header.num_tasks):
        tasks.append(TaskDesc.from_bytes(data[off:off + TaskDesc.STRUCT_SIZE]))
        off += TaskDesc.STRUCT_SIZE

    buffers = []
    for _ in range(header.num_buffers):
        buffers.append(BufferDesc.from_bytes(data[off:off + BufferDesc.STRUCT_SIZE]))
        off += BufferDesc.STRUCT_SIZE

    weight_maps = []
    for _ in range(header.num_weight_mappings):
        weight_maps.append(WeightMapping.from_bytes(data[off:off + WeightMapping.STRUCT_SIZE]))
        off += WeightMapping.STRUCT_SIZE

    string_table = data[off:]
    weights = {}
    for wm in weight_maps:
        name = string_table[wm.key_offset:wm.key_offset + wm.key_length]
        weights[wm.buffer_index] = name.decode("utf-8")

    return header, tasks, buffers, weight_maps, weights
