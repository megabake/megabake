"""Compute tile counts for task types."""

import math
from megabake.data_types import OpType

# ponytail: only config 0 (128x128) implemented in CUDA. Add 64x128, 128x256 when profiling shows benefit on real prefill workloads.
MATMUL_CONFIGS = [
    (0, 128, 128),
    (1, 64, 128),
    (2, 128, 256),
]


def select_matmul_config(M, N, K, num_sms):
    if M <= 4:
        return -1
    best_score, best_id = 0.0, 0
    for config_id, bm, bn in MATMUL_CONFIGS:
        tiles = math.ceil(M / bm) * math.ceil(N / bn)
        sm_util = min(tiles, num_sms) / num_sms
        waves = math.ceil(tiles / num_sms)
        wave_eff = tiles / (waves * num_sms)
        score = sm_util * wave_eff
        if score > best_score:
            best_score, best_id = score, config_id
    return best_id


def compute_tiles(op_type: int, dims: list[int], sm_version: int, num_sms: int = 0) -> int:
    max_sms = num_sms if num_sms > 0 else {80: 108, 90: 132, 100: 144}.get(sm_version, 132)

    if op_type == OpType.MATMUL:
        M, N = dims[0], dims[1]
        if M <= 4:
            return max_sms
        tiles_m = (M + 127) // 128
        tiles_n = (N + 127) // 128
        return min(tiles_m * tiles_n, max_sms)
    elif op_type == OpType.ATTENTION:
        batch, num_heads, seq_q = dims[0], dims[1], dims[2]
        return min(batch * num_heads * seq_q, max_sms)
    elif op_type in (OpType.ELEMENTWISE, OpType.FUSED_ELEMENTWISE):
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
