import heapq
from megabake.data_types import (
    OpType, SMQueueEntry, UNUSED_BUFFER, QFLAG_HANDOFF,
)


def estimate_cycles(task):
    op = task.op_type
    d = task.dimensions
    if op == OpType.MATMUL:
        M, N, K = max(d[0], 1), max(d[1], 1), max(d[2], 1)
        return M * N * K // max(task.num_tiles, 1)
    if op == OpType.ATTENTION:
        batch, heads, seq_q, seq_k = max(d[0], 1), max(d[1], 1), max(d[2], 1), max(d[3], 1)
        return batch * heads * seq_q * seq_k // max(task.num_tiles, 1)
    if op == OpType.REDUCE:
        rows, cols = max(d[0], 1), max(d[1], 1)
        return rows * cols // max(task.num_tiles, 1)
    total = max(d[0], 1)
    return total // max(task.num_tiles, 1)


def _critical_path_lengths(tasks, successors):
    n = len(tasks)
    cpl = [0] * n
    visited = [False] * n

    def dfs(u):
        stack = [(u, False)]
        while stack:
            node, done = stack.pop()
            if done:
                best = 0
                for s in successors[node]:
                    if cpl[s] > best:
                        best = cpl[s]
                cpl[node] = estimate_cycles(tasks[node]) + best
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for s in successors[node]:
                if not visited[s]:
                    stack.append((s, False))

    for i in range(n):
        if not visited[i]:
            dfs(i)
    return cpl


def _topo_sort_by_priority(dep_count, successors, priority):
    n = len(dep_count)
    dc = list(dep_count)
    heap = [(-priority[i], i) for i in range(n) if dc[i] == 0]
    heapq.heapify(heap)
    result = []
    while heap:
        _, u = heapq.heappop(heap)
        result.append(u)
        for v in successors[u]:
            dc[v] -= 1
            if dc[v] == 0:
                heapq.heappush(heap, (-priority[v], v))
    return result


def assign_tasks_to_sms(tasks, dep_count, successors, num_sms):
    cpl = _critical_path_lengths(tasks, successors)
    sorted_tasks = _topo_sort_by_priority(dep_count, successors, cpl)

    sm_queues = [[] for _ in range(num_sms)]
    sm_load = [0] * num_sms

    for task_id in sorted_tasks:
        task = tasks[task_id]
        tiles = max(task.num_tiles, 1)
        cost = estimate_cycles(task)
        if tiles == 1:
            sm = min(range(num_sms), key=lambda s: sm_load[s])
            sm_queues[sm].append((task_id, 0))
            sm_load[sm] += cost
        else:
            n = min(tiles, num_sms)
            for tile in range(tiles):
                sm = tile % n
                sm_queues[sm].append((task_id, tile))
                sm_load[sm] += cost // tiles

    return sm_queues


def smem_usage_estimate(task, sm90=True):
    op = task.op_type
    if op == OpType.MATMUL:
        M = max(task.dimensions[0], 1)
        K = max(task.dimensions[2], 1)
        if M <= 4:
            budget = 220 * 1024 if sm90 else 96 * 1024
            cols = min(256, max(task.dimensions[1], 1))
            bk = budget // (2 * (M + 2 * cols))
            bk = min(bk & ~7, K)
            if bk < 8:
                bk = 8
            return (M * bk + 2 * cols * bk) * 2
        return 98304 if sm90 else 32768
    if op == OpType.ATTENTION:
        head_dim = max(task.dimensions[4], 128) if len(task.dimensions) > 4 and task.dimensions[4] > 0 else 128
        return 256 + 64 * head_dim * 2 * 2 + 32
    if op == OpType.REDUCE:
        return 32
    return 0


def plan_prefetch(sm_queues, tasks, num_sms):
    sm90 = True
    smem_budget = 220 * 1024 if sm90 else 96 * 1024
    result = []

    for sm_id in range(len(sm_queues)):
        queue = sm_queues[sm_id]
        new_queue = []
        for i, (task_id, tile_id) in enumerate(queue):
            entry = SMQueueEntry(task_id=task_id, tile_id=tile_id)

            # ponytail: SMEM handoff disabled — skinny matmul uses nearly all SMEM
            # for double-buffered B, leaving no safe region for handoff data.
            # Enable when matmul gets explicit page-based SMEM management.

            new_queue.append(entry)
        result.append(new_queue)

    # Second pass: compute prefetch for next entries
    for sm_id in range(len(result)):
        queue = result[sm_id]
        for i in range(len(queue) - 1):
            cur_entry = queue[i]
            next_entry = queue[i + 1]
            next_tid = next_entry.task_id
            next_task = tasks[next_tid]

            if next_task.op_type != OpType.MATMUL:
                continue
            M = max(next_task.dimensions[0], 1)
            if M > 4:
                continue

            N = max(next_task.dimensions[1], 1)
            K = max(next_task.dimensions[2], 1)

            # Match skinny matmul's active_tiles = min(num_tiles, gridDim.x)
            active_tiles = min(max(next_task.num_tiles, 1), num_sms)
            cols_per_tile = (N + active_tiles - 1) // active_tiles
            next_tile = next_entry.tile_id & 0x7FFFFFFF
            n_start = next_tile * cols_per_tile
            n_end = min(n_start + cols_per_tile, N)
            # First N-chunk: blockDim.x = 256 columns max
            actual_cols = min(256, n_end - n_start)
            if actual_cols <= 0:
                continue

            # Only prefetch when bk >= K (single K-iteration, SMEM layout matches source)
            cols = 256  # blockDim.x
            bk = smem_budget // (2 * (M + 2 * cols))
            bk = (bk & ~7)
            if bk < K:
                continue

            prefetch_bytes = actual_cols * K * 2
            cur_smem = smem_usage_estimate(tasks[cur_entry.task_id], sm90)
            available = smem_budget - cur_smem
            if prefetch_bytes > available or prefetch_bytes <= 0:
                continue

            weight_buf = next_task.buffer_indices[2]
            if weight_buf == UNUSED_BUFFER:
                continue

            cur_entry.prefetch_buf_idx = weight_buf
            cur_entry.prefetch_bytes = prefetch_bytes

    return result
