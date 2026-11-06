"""Shared GEMM launch helpers used by body-aware code generation and runtime."""

import re

STREAM_TAG_SLOT = 2
TABLE_DESCRIPTORS_PER_GEMM = 4


def cute_gemm_tensors(a, b, c, splits=1):
    from cutlass.cute.runtime import from_dlpack

    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])
    if any(tensor.data_ptr() % 16 for tensor in (a, b, c)):
        raise ValueError("GEMM tensor base addresses must be 16-byte aligned")

    def view(tensor):
        leading_dim = next(index for index, stride in enumerate(tensor.stride()) if stride == 1)
        return from_dlpack(tensor.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    def split_view(tensor):
        rows, reduction = tensor.shape
        chunk = reduction // splits
        split = tensor.as_strided((rows, chunk, splits), (tensor.stride(0), 1, chunk))
        return from_dlpack(split, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    if splits > 1:
        return split_view(a), split_view(b), view(c)
    return view(a), view(b), view(c)


def has_bias(body):
    return body.name.endswith("_bias")


def has_fused_epilogue(body):
    return body.name.endswith("_epilogue")


def is_paired(body):
    return "_paired" in body.name


def table_gemm_shared_bytes(tiles, capacity):
    stages = a_bytes = b_bytes = c_bytes = 0
    for tile_m, tile_n in tiles:
        ab_bytes = (tile_m + tile_n) * 64 * 2
        epilogue_bytes = 4 * min(64, tile_m) * min(32, tile_n) * 2
        stage_count = (capacity - 1024 - epilogue_bytes) // ab_bytes
        stages = max(stages, stage_count)
        a_bytes = max(a_bytes, stage_count * tile_m * 64 * 2)
        b_bytes = max(b_bytes, stage_count * tile_n * 64 * 2)
        c_bytes = max(c_bytes, epilogue_bytes)
    barrier_bytes = stages * 2 * 8 + 4 * 4
    return -(-barrier_bytes // 1024) * 1024 + a_bytes + b_bytes + c_bytes


def paired_contraction(program):
    from ..ir import Accumulator

    def reads_second(expression):
        if isinstance(expression, Accumulator):
            return expression.index == 1
        return any(
            reads_second(getattr(expression, name))
            for name in ("operand", "lhs", "rhs", "condition", "true_value", "false_value")
            if getattr(expression, name, None) is not None
        )

    return program.post is not None and reads_second(program.post)


def epilogue_loads(program):
    return program.loads[3:] if paired_contraction(program) else program.loads[2:]


def epilogue_source(program, symbol):
    from .cuda.generic import _cast_expression, _emit_expression

    loads = epilogue_loads(program)
    value = _cast_expression(
        _emit_expression(
            program.post,
            {load.id: index for index, load in enumerate(loads)},
            {"o0": "o0", "o1": "o1"},
            tuple(loads),
        ),
        program.store.dtype,
    )
    return (
        "@cute.jit\n"
        f"def {symbol}(accumulator, accumulator_1, o0, o1, tensor_bindings):\n"
        f"    return {value.code}\n"
    )


def split_k(body):
    match = re.search(r"_k(\d+)(?:_bias|_epilogue)?$", body.name)
    return int(match.group(1)) if match else 1


def is_swap_ab(body):
    return body.provider == "sm100_tma" or (
        body.provider == "g0_range" and body.name.startswith("g0_range_swapab_")
    )


def kernel_tile(body):
    rows, columns = body.tiles.cta_shape[:2]
    return (columns, rows) if is_swap_ab(body) else (rows, columns)


__all__ = [
    "STREAM_TAG_SLOT",
    "TABLE_DESCRIPTORS_PER_GEMM",
    "cute_gemm_tensors",
    "epilogue_loads",
    "epilogue_source",
    "has_bias",
    "has_fused_epilogue",
    "is_paired",
    "is_swap_ab",
    "kernel_tile",
    "split_k",
    "table_gemm_shared_bytes",
]
