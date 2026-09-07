"""Per-task cycle-level profiling for the megakernel."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from megabake.data_types import OpType
from megabake.schedule_compiler.serializer import load_schedule


def profile_model(compiled, model_or_sd, *inputs):
    """Run one profiled megakernel launch.

    Returns (tasks, cycles) where cycles[i] is a list of clock64 deltas
    per SM for task i.
    """
    import megabake

    megabake.run(compiled, model_or_sd, *inputs)
    torch.cuda.synchronize()

    _, tasks, *_ = load_schedule(compiled.schedule_bytes)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    buf = torch.zeros(len(tasks) * num_sms, dtype=torch.int64, device="cuda")
    megabake.run(compiled, model_or_sd, *inputs, task_timings_ptr=buf.data_ptr())
    torch.cuda.synchronize()

    flat = buf.cpu().tolist()
    cycles = [flat[i * num_sms:(i + 1) * num_sms] for i in range(len(tasks))]
    return tasks, cycles


def analyze(tasks, cycles):
    """Per-task stats from raw cycle data."""
    stats = []
    for i, task in enumerate(tasks):
        active = sorted(c for c in cycles[i] if c > 100)
        n_total = len(cycles[i])
        if not active:
            stats.append(_make_stat(i, task, 0, 0, 0, n_total, 0))
        else:
            med = active[len(active) // 2]
            mx = active[-1]
            stats.append(_make_stat(i, task, med, mx, len(active), n_total - len(active), mx - med))
    return stats


def _make_stat(task_id, task, median, maximum, active, idle, barrier):
    return {
        "task_id": task_id,
        "op_type": OpType(task.op_type).name,
        "op_code": task.op_code,
        "dims": [d for d in task.dimensions[:4] if d > 0],
        "num_tiles": task.num_tiles,
        "median_cycles": median,
        "max_cycles": maximum,
        "active_sms": active,
        "idle_sms": idle,
        "barrier_wait": barrier,
    }


def print_report(stats, num_sms):
    """Print per-task table and per-op-type summary."""
    hdr = (f"{'task':>4s} | {'op_type':<18s} | {'dims':<16s} | {'tiles':>5s} | "
           f"{'median':>10s} | {'max':>10s} | {'active':>6s} | {'idle':>4s} | {'barrier':>10s}")
    sep = "-" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")

    total_max = 0
    by_op: dict[str, dict] = {}

    for s in stats:
        dims_str = "x".join(str(d) for d in s["dims"]) or "-"
        print(f"{s['task_id']:>4d} | {s['op_type']:<18s} | {dims_str:<16s} | {s['num_tiles']:>5d} | "
              f"{s['median_cycles']:>10d} | {s['max_cycles']:>10d} | "
              f"{s['active_sms']:>6d} | {s['idle_sms']:>4d} | {s['barrier_wait']:>10d}")
        total_max += s["max_cycles"]
        entry = by_op.setdefault(s["op_type"], {"cycles": 0, "count": 0, "sm_sum": 0})
        entry["cycles"] += s["max_cycles"]
        entry["count"] += 1
        entry["sm_sum"] += s["active_sms"]

    print(sep)
    if total_max > 0:
        print(f"\nSummary by op_type (total = {total_max:,d} cycles):")
        for op, d in sorted(by_op.items(), key=lambda x: x[1]["cycles"], reverse=True):
            pct = d["cycles"] / total_max * 100
            avg_sm = d["sm_sum"] / d["count"]
            print(f"  {op:<18s} {pct:>5.1f}%  ({d['count']} tasks, avg {avg_sm:.0f}/{num_sms} SMs)")

    total_barrier = sum(s["barrier_wait"] for s in stats)
    total_idle = sum(s["max_cycles"] * s["idle_sms"] for s in stats)
    print(f"\nTotal barrier wait: {total_barrier:,d} cycles")
    print(f"Total SM-idle cycles: {total_idle:,d}\n")


def save_json(stats, path):
    """Save profile stats as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {path}")
