from megabake.data_types import UNUSED_BUFFER


def build_dependency_dag(tasks):
    producers = {}
    dep_count = [0] * len(tasks)
    successors = [[] for _ in range(len(tasks))]

    for i, task in enumerate(tasks):
        out_buf = task.buffer_indices[0]
        if out_buf != UNUSED_BUFFER:
            producers[out_buf] = i

    for i, task in enumerate(tasks):
        seen = set()
        for buf in task.buffer_indices[1:]:
            if buf != UNUSED_BUFFER and buf in producers:
                src = producers[buf]
                if src != i and src not in seen:
                    seen.add(src)
                    dep_count[i] += 1
                    successors[src].append(i)

    return dep_count, successors


def add_arena_anti_dependences(tasks, buffer_descs, dep_count, successors):
    """Add WAR edges for arena-aliased buffers.

    The buffer planner reuses arena slots based on sequential liveness.
    With the counter-based scheduler, a later task can start writing to
    an aliased slot while an earlier task still reads from it. Fix: add
    edges from every reader of a buffer to every later writer whose
    output overlaps in the arena — but skip edges already implied by
    existing transitive dependencies.
    """
    n = len(tasks)
    arena = {}
    for bd in buffer_descs:
        arena[bd.buffer_id] = (bd.offset, bd.size)

    write_ranges = {}
    read_ranges = {}
    for i, t in enumerate(tasks):
        out = t.buffer_indices[0]
        if out != UNUSED_BUFFER and out in arena:
            write_ranges[i] = arena[out]
        reads = []
        for buf in t.buffer_indices[1:]:
            if buf != UNUSED_BUFFER and buf in arena:
                reads.append(arena[buf])
        if reads:
            read_ranges[i] = reads

    # Precompute reachability (transitive closure) via BFS from each node
    reachable = [set() for _ in range(n)]
    for start in range(n):
        visited = set()
        stack = list(successors[start])
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(successors[node])
        reachable[start] = visited

    existing_edges = set()
    for src, succs in enumerate(successors):
        for dst in succs:
            existing_edges.add((src, dst))

    for writer_idx in range(n):
        if writer_idx not in write_ranges:
            continue
        wo, ws = write_ranges[writer_idx]
        for reader_idx in range(writer_idx):
            if writer_idx in reachable[reader_idx]:
                continue
            if (reader_idx, writer_idx) in existing_edges:
                continue
            for ro, rs in read_ranges.get(reader_idx, []):
                if wo < ro + rs and ro < wo + ws:
                    existing_edges.add((reader_idx, writer_idx))
                    successors[reader_idx].append(writer_idx)
                    dep_count[writer_idx] += 1
                    reachable[reader_idx].add(writer_idx)
                    reachable[reader_idx].update(reachable[writer_idx])
                    break
