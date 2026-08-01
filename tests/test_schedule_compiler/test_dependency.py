from megabake.data_types import TaskDesc, UNUSED_BUFFER
from megabake.schedule_compiler.dependency import build_dependency_dag


def _task(out, *inputs):
    t = TaskDesc(op_type=0x01, num_tiles=1)
    t.buffer_indices[0] = out
    for i, buf in enumerate(inputs):
        t.buffer_indices[1 + i] = buf
    return t


def test_linear_chain():
    tasks = [_task(0), _task(1, 0), _task(2, 1)]
    dep, succ = build_dependency_dag(tasks)
    assert dep == [0, 1, 1]
    assert succ == [[1], [2], []]


def test_diamond():
    #   0
    #  / \
    # 1   2
    #  \ /
    #   3
    tasks = [_task(10), _task(11, 10), _task(12, 10), _task(13, 11, 12)]
    dep, succ = build_dependency_dag(tasks)
    assert dep == [0, 1, 1, 2]
    assert succ == [[1, 2], [3], [3], []]


def test_no_deps():
    tasks = [_task(0), _task(1), _task(2)]
    dep, succ = build_dependency_dag(tasks)
    assert dep == [0, 0, 0]
    assert all(s == [] for s in succ)


def test_duplicate_input_buffers():
    t = _task(2, 0, 0, 0)
    tasks = [_task(0), t]
    dep, succ = build_dependency_dag(tasks)
    assert dep == [0, 1]
    assert succ == [[1], []]


def test_self_reference_ignored():
    t = _task(5, 5)
    tasks = [t]
    dep, succ = build_dependency_dag(tasks)
    assert dep == [0]
    assert succ == [[]]
