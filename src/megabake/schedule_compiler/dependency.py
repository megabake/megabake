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
