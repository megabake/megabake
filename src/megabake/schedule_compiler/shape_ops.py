"""Zero-cost shape op resolution via stride manipulation."""

from dataclasses import dataclass


@dataclass
class StridedView:
    buffer_id: int
    shape: list[int]
    strides: list[int]
    offset: int = 0

    def is_contiguous(self) -> bool:
        expected = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != expected and self.shape[i] != 1:
                return False
            expected *= self.shape[i]
        return True


def contiguous_strides(shape: list[int]) -> list[int]:
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides


def resolve_reshape(view: StridedView, new_shape: list[int]) -> StridedView | None:
    if not view.is_contiguous():
        return None
    if -1 in new_shape:
        total = 1
        for s in view.shape:
            total *= s
        known = 1
        for s in new_shape:
            if s != -1:
                known *= s
        new_shape = [total // known if s == -1 else s for s in new_shape]
    return StridedView(view.buffer_id, new_shape, contiguous_strides(new_shape), view.offset)


def resolve_transpose(view: StridedView, dim0: int, dim1: int) -> StridedView:
    new_shape = list(view.shape)
    new_strides = list(view.strides)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
    new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_permute(view: StridedView, dims: list[int]) -> StridedView:
    new_shape = [view.shape[d] for d in dims]
    new_strides = [view.strides[d] for d in dims]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_expand(view: StridedView, new_shape: list[int]) -> StridedView:
    resolved = list(new_shape)
    for i in range(len(resolved)):
        if resolved[i] == -1 and i < len(view.shape):
            resolved[i] = view.shape[i]
    new_strides = list(view.strides)
    for i in range(len(resolved)):
        if i >= len(view.shape) or view.shape[i] == 1:
            if resolved[i] != 1:
                new_strides[i] = 0
    return StridedView(view.buffer_id, resolved, new_strides, view.offset)


def resolve_squeeze(view: StridedView, dim: int | None) -> StridedView:
    if dim is not None:
        if view.shape[dim] != 1:
            return view
        new_shape = view.shape[:dim] + view.shape[dim + 1:]
        new_strides = view.strides[:dim] + view.strides[dim + 1:]
    else:
        new_shape = [s for s in view.shape if s != 1]
        new_strides = [view.strides[i] for i, s in enumerate(view.shape) if s != 1]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_unsqueeze(view: StridedView, dim: int) -> StridedView:
    new_shape = list(view.shape)
    new_strides = list(view.strides)
    insert_stride = view.strides[dim] * view.shape[dim] if dim < len(view.shape) else 1
    new_shape.insert(dim, 1)
    new_strides.insert(dim, insert_stride)
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_slice(view: StridedView, dim: int, start: int,
                  end: int, step: int = 1) -> StridedView:
    if step != 1:
        return None
    if end > view.shape[dim]:
        end = view.shape[dim]
    if start < 0:
        start = max(0, view.shape[dim] + start)
    new_shape = list(view.shape)
    new_shape[dim] = end - start
    new_offset = view.offset + start * view.strides[dim]
    return StridedView(view.buffer_id, new_shape, list(view.strides), new_offset)


def resolve_t(view: StridedView) -> StridedView:
    assert len(view.shape) == 2
    return resolve_transpose(view, 0, 1)
