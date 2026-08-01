"""Serialize/deserialize .schedule binary files."""

import struct

from megabake.data_types import (
    TaskDesc, BufferDesc, WeightMapping, ScheduleHeader, SMQueueEntry,
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
    sm_queues: list[list] | None = None,
    dep_count: list[int] | None = None,
    successors: list[list[int]] | None = None,
    scheduler_type: int = 0,
) -> bytes:
    buffer_descs = []
    for p in placements:
        buffer_descs.append(BufferDesc(offset=p.offset, size=p.size, dtype=compute_dtype, buffer_id=p.buffer_id))

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

    num_sms = len(sm_queues) if sm_queues else 0
    max_queue_len = max((len(q) for q in sm_queues), default=0) if sm_queues else 0

    succ_flat = []
    if successors:
        for s in successors:
            succ_flat.extend(s)
    num_edges = len(succ_flat)

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
        num_sms=num_sms,
        max_queue_len=max_queue_len,
        num_edges=num_edges,
        scheduler_type=scheduler_type,
    )

    data = header.to_bytes()
    for t in tasks:
        data += t.to_bytes()
    for b in buffer_descs:
        data += b.to_bytes()
    for wm in weight_mappings:
        data += wm.to_bytes()
    data += string_table

    if num_sms > 0:
        assert sm_queues is not None and dep_count is not None and successors is not None
        # SM queue entries: each SM padded to max_queue_len (16 bytes each)
        for q in sm_queues:
            for entry in q:
                if isinstance(entry, SMQueueEntry):
                    data += entry.to_bytes()
                else:
                    tid, tile = entry
                    data += struct.pack("<IIII", tid, tile, 0xFFFFFFFF, 0)
            for _ in range(max_queue_len - len(q)):
                data += struct.pack("<IIII", 0, 0, 0xFFFFFFFF, 0)

        # SM queue lengths
        for q in sm_queues:
            data += struct.pack("<I", len(q))

        # dep_count per task
        for dc in dep_count:
            data += struct.pack("<I", dc)

        # tile_remaining per task
        for t in tasks:
            data += struct.pack("<I", max(t.num_tiles, 1))

        # successor_offset[num_tasks + 1]
        offset = 0
        for s in successors:
            data += struct.pack("<I", offset)
            offset += len(s)
        data += struct.pack("<I", offset)

        # successor_list (flat)
        for s in successors:
            for v in s:
                data += struct.pack("<I", v)

    return data


def load_schedule(data: bytes) -> tuple[
    ScheduleHeader, list[TaskDesc], list[BufferDesc],
    list[WeightMapping], dict[int, str],
    list[list[SMQueueEntry]] | None,
    list[int] | None,
    list[int] | None,
    list[int] | None,
    list[int] | None,
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

    str_table_start = off
    # String table ends where scheduler data begins (or EOF)
    if header.num_sms > 0:
        sched_start = str_table_start
        for wm in weight_maps:
            end = wm.key_offset + wm.key_length + 1  # +1 for null terminator
            if end > sched_start - str_table_start:
                sched_start = str_table_start + end
        off = sched_start
    else:
        off = len(data)

    string_table = data[str_table_start:off] if weight_maps else b""
    weights = {}
    for wm in weight_maps:
        name = string_table[wm.key_offset:wm.key_offset + wm.key_length]
        weights[wm.buffer_index] = name.decode("utf-8")

    sm_queues = None
    dep_count = None
    tile_remaining = None
    succ_offset = None
    succ_list = None

    if header.num_sms > 0:
        # SM queue entries
        sm_queues = []
        sq_entry_size = SMQueueEntry.STRUCT_SIZE
        for sm in range(header.num_sms):
            entries = []
            base = off + sm * header.max_queue_len * sq_entry_size
            for j in range(header.max_queue_len):
                e = SMQueueEntry.from_bytes(data[base + j * sq_entry_size:base + (j + 1) * sq_entry_size])
                entries.append(e)
            sm_queues.append(entries)
        off += header.num_sms * header.max_queue_len * sq_entry_size

        # SM queue lengths — trim each SM's entries
        sm_queue_lens = []
        for sm in range(header.num_sms):
            qlen = struct.unpack_from("<I", data, off)[0]
            sm_queue_lens.append(qlen)
            off += 4
        for sm in range(header.num_sms):
            sm_queues[sm] = sm_queues[sm][:sm_queue_lens[sm]]

        # dep_count
        dep_count = []
        for _ in range(header.num_tasks):
            dep_count.append(struct.unpack_from("<I", data, off)[0])
            off += 4

        # tile_remaining
        tile_remaining = []
        for _ in range(header.num_tasks):
            tile_remaining.append(struct.unpack_from("<I", data, off)[0])
            off += 4

        # successor_offset[num_tasks + 1]
        succ_offset = []
        for _ in range(header.num_tasks + 1):
            succ_offset.append(struct.unpack_from("<I", data, off)[0])
            off += 4

        # successor_list
        succ_list = []
        for _ in range(header.num_edges):
            succ_list.append(struct.unpack_from("<I", data, off)[0])
            off += 4

    return header, tasks, buffers, weight_maps, weights, sm_queues, dep_count, tile_remaining, succ_offset, succ_list
