import heapq
from megabake.data_types import OpType


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
