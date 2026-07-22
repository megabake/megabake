"""Level 2 tests: shape op stride resolution."""

import pytest
from megabake.schedule_compiler.shape_ops import (
    StridedView, contiguous_strides,
    resolve_reshape, resolve_transpose, resolve_permute,
    resolve_expand, resolve_squeeze, resolve_unsqueeze,
    resolve_slice, resolve_t,
)


class TestContiguousStrides:
    def test_1d(self):
        assert contiguous_strides([4096]) == [1]

    def test_2d(self):
        assert contiguous_strides([8, 4096]) == [4096, 1]

    def test_3d(self):
        assert contiguous_strides([2, 8, 64]) == [512, 64, 1]


class TestStridedView:
    def test_contiguous(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        assert v.is_contiguous()

    def test_transposed_not_contiguous(self):
        v = StridedView(0, [4096, 8], [1, 4096])
        assert not v.is_contiguous()

    def test_broadcast_contiguous(self):
        v = StridedView(0, [1, 4096], [0, 1])
        assert v.is_contiguous()


class TestReshape:
    def test_contiguous_reshape(self):
        v = StridedView(0, [2, 512], [512, 1])
        r = resolve_reshape(v, [1024])
        assert r is not None
        assert r.shape == [1024]
        assert r.is_contiguous()

    def test_non_contiguous_returns_none(self):
        v = StridedView(0, [4096, 8], [1, 4096])
        r = resolve_reshape(v, [32768])
        assert r is None


class TestTranspose:
    def test_basic(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        r = resolve_transpose(v, 0, 1)
        assert r.shape == [4096, 8]
        assert r.strides == [1, 4096]


class TestPermute:
    def test_3d(self):
        v = StridedView(0, [2, 8, 64], [512, 64, 1])
        r = resolve_permute(v, [0, 2, 1])
        assert r.shape == [2, 64, 8]
        assert r.strides == [512, 1, 64]


class TestSlice:
    def test_basic(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        r = resolve_slice(v, 0, 2, 6)
        assert r.shape == [4, 4096]
        assert r.offset == 2 * 4096


class TestSqueeze:
    def test_squeeze_dim(self):
        v = StridedView(0, [1, 8, 4096], [32768, 4096, 1])
        r = resolve_squeeze(v, 0)
        assert r.shape == [8, 4096]

    def test_squeeze_noop(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        r = resolve_squeeze(v, 0)
        assert r.shape == [8, 4096]


class TestUnsqueeze:
    def test_basic(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        r = resolve_unsqueeze(v, 0)
        assert r.shape == [1, 8, 4096]

    def test_middle(self):
        v = StridedView(0, [8, 4096], [4096, 1])
        r = resolve_unsqueeze(v, 1)
        assert r.shape == [8, 1, 4096]


class TestShapeOpsProduceNoTasks:
    """Key TDD invariant: shape ops must resolve to zero GPU work."""

    def test_reshape_chain_no_tasks(self):
        v = StridedView(0, [2, 8, 64], [512, 64, 1])
        v = resolve_reshape(v, [2, 512])
        assert v is not None
        v = resolve_reshape(v, [1024])
        assert v is not None
        assert v.buffer_id == 0
        assert v.is_contiguous()

    def test_transpose_then_reshape_requires_copy(self):
        v = StridedView(0, [8, 64], [64, 1])
        v = resolve_transpose(v, 0, 1)
        assert not v.is_contiguous()
        r = resolve_reshape(v, [512])
        assert r is None
