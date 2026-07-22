"""Compute tile counts for task types."""

from megabake.data_types import OpType


def compute_tiles(op_type: int, dims: list[int], sm_version: int) -> int:
    max_sms = {80: 108, 90: 132, 100: 144}.get(sm_version, 132)

    if op_type == OpType.MATMUL:
        M, N = dims[0], dims[1]
        tiles_m = (M + 127) // 128
        tiles_n = (N + 127) // 128
        return min(tiles_m * tiles_n, max_sms)
    elif op_type == OpType.ATTENTION:
        batch, num_heads = dims[0], dims[1]
        return min(batch * num_heads, max_sms)
    elif op_type == OpType.ELEMENTWISE:
        total = dims[0]
        return min((total + 4095) // 4096, max_sms)
    elif op_type == OpType.REDUCE:
        return min(max(dims[0], 1), max_sms)
    elif op_type == OpType.EMBEDDING:
        return min(max(dims[0], 1), max_sms)
    elif op_type == OpType.COPY:
        total = dims[0]
        return min((total + 4095) // 4096, max_sms)
    elif op_type == OpType.ROPE:
        batch, seq = dims[0], dims[1]
        return min(batch * seq, max_sms)
    return min(32, max_sms)
