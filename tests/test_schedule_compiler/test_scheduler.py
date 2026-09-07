from megabake.data_types import TaskDesc, OpType
from megabake.schedule_compiler.dependency import build_dependency_dag
from megabake.schedule_compiler.scheduler import (
    assign_tasks_to_sms,
    _critical_path_lengths,
    _topo_sort_by_priority,
)


def _task(op_type, out, *inputs, num_tiles=1, dims=None):
    t = TaskDesc(op_type=op_type, num_tiles=num_tiles)
    t.buffer_indices[0] = out
    for i, buf in enumerate(inputs):
        t.buffer_indices[1 + i] = buf
    if dims:
        for i, d in enumerate(dims):
            t.dimensions[i] = d
    return t


def test_all_tasks_assigned():
    tasks = [
        _task(OpType.REDUCE, 0, dims=[1, 256]),
        _task(OpType.MATMUL, 1, 0, dims=[1, 512, 256]),
        _task(OpType.ELEMENTWISE, 2, 1, dims=[512]),
    ]
    dep, succ = build_dependency_dag(tasks)
    queues = assign_tasks_to_sms(tasks, dep, succ, num_sms=4)
    all_entries = [e for q in queues for e in q]
    task_ids = {e[0] for e in all_entries}
    assert task_ids == {0, 1, 2}


def test_topo_order_respected():
    # Chain: 0 -> 1 -> 2
    tasks = [
        _task(OpType.ELEMENTWISE, 10, dims=[100]),
        _task(OpType.ELEMENTWISE, 11, 10, dims=[100]),
        _task(OpType.ELEMENTWISE, 12, 11, dims=[100]),
    ]
    dep, succ = build_dependency_dag(tasks)
    queues = assign_tasks_to_sms(tasks, dep, succ, num_sms=2)
    for q in queues:
        task_ids = [e[0] for e in q]
        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                assert task_ids[i] < task_ids[j] or dep[task_ids[j]] == 0


def test_multi_tile_spread():
    tasks = [_task(OpType.MATMUL, 0, dims=[128, 128, 64], num_tiles=4)]
    dep, succ = build_dependency_dag(tasks)
    queues = assign_tasks_to_sms(tasks, dep, succ, num_sms=4)
    tiles = [(e[0], e[1]) for q in queues for e in q]
    assert sorted(tiles) == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_single_tile_load_balance():
    # 4 independent single-tile tasks on 2 SMs -> 2 each
    tasks = [_task(OpType.ELEMENTWISE, i, dims=[100]) for i in range(4)]
    dep, succ = build_dependency_dag(tasks)
    queues = assign_tasks_to_sms(tasks, dep, succ, num_sms=2)
    assert len(queues[0]) == 2
    assert len(queues[1]) == 2


def test_critical_path_priority():
    # 0 -> 1 (long chain), 2 standalone
    # Task 0 should have higher CPL than task 2
    tasks = [
        _task(OpType.MATMUL, 10, dims=[64, 512, 256]),
        _task(OpType.MATMUL, 11, 10, dims=[64, 512, 256]),
        _task(OpType.ELEMENTWISE, 12, dims=[100]),
    ]
    _, succ = build_dependency_dag(tasks)
    cpl = _critical_path_lengths(tasks, succ)
    assert cpl[0] > cpl[2]
    assert cpl[0] > cpl[1]


def test_empty_tasks():
    queues = assign_tasks_to_sms([], [], [], num_sms=4)
    assert all(q == [] for q in queues)


def test_topo_sort_all_independent():
    tasks = [_task(OpType.ELEMENTWISE, i, dims=[100]) for i in range(3)]
    dep, succ = build_dependency_dag(tasks)
    cpl = _critical_path_lengths(tasks, succ)
    order = _topo_sort_by_priority(dep, succ, cpl)
    assert sorted(order) == [0, 1, 2]
