"""Compare eager, MegaBake and its CUDA graph over advancing KV caches."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from benchmarks.full_model import _load, decode_module
from megabake import graphed, make_backend


model_path, output_path = sys.argv[1:3]
capacity = 192
steps = 128
model = _load(model_path, None, "eager")
eager, state_shapes = decode_module(model, capacity)


def state():
    return [torch.zeros(shape, device="cuda", dtype=dtype)
            for layer in state_shapes for shape, dtype in layer]


eager_state, normal_state, graph_state = state(), state(), state()
backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
optimized = torch.compile(eager, backend=backend, fullgraph=True, dynamic=False)
token = torch.tensor([[1]], dtype=torch.long, device="cuda")
position = torch.tensor([[0]], dtype=torch.long, device="cuda")


def reset(values):
    for value in values:
        value.zero_()


with torch.no_grad():
    optimized(token, position, *normal_state)
    reset(normal_state)
    graph = graphed(optimized, token.clone(), position.clone(), *graph_state)
    if any(t.any().item() for t in graph_state):
        raise RuntimeError("graph capture changed the initial state")

    teacher = []
    eager_top_matches = 0
    eager_relative_l2 = []
    eager_mismatches = []
    graph_vs_normal_max_abs = 0.0
    graph_vs_normal_top_matches = 0
    teacher_all_finite = {"eager": True, "normal": True, "graph": True}
    for index in range(steps):
        current_token = torch.tensor([[([1, 504, 3458, 314][index % 4])]], device="cuda")
        current_position = torch.tensor([[index]], device="cuda")
        ref = eager(current_token, current_position, *eager_state).float()
        normal = optimized(current_token, current_position, *normal_state).float()
        actual = graph(current_token, current_position, *graph_state).float()
        for label, value in (("eager", ref), ("normal", normal), ("graph", actual)):
            teacher_all_finite[label] &= bool(torch.isfinite(value).all().item())
        difference = (actual - normal).abs().max().item()
        relative_l2 = ((actual - ref).norm() / ref.norm()).item()
        top_matches_eager = actual.argmax(-1).item() == ref.argmax(-1).item()
        eager_relative_l2.append(relative_l2)
        eager_top_matches += int(top_matches_eager)
        if not top_matches_eager:
            eager_mismatches.append(index)
        graph_vs_normal_max_abs = max(graph_vs_normal_max_abs, difference)
        graph_vs_normal_top_matches += int(actual.argmax(-1).item() == normal.argmax(-1).item())
        if index in (0, 1, 15, 31, 63, 127):
            teacher.append({
                "position": index,
                "relative_l2_vs_eager": relative_l2,
                "top_matches_eager": top_matches_eager,
                "graph_vs_normal_max_abs": difference,
                "finite": bool(torch.isfinite(actual).all().item()),
            })
    length_counts = {
        label: [int(v.item()) for v in values if v.dtype == torch.int32 and v.numel() == 1]
        for label, values in (("eager", eager_state), ("normal", normal_state), ("graph", graph_state))
    }
    graph_state_matches_normal = all(torch.equal(a, b) for a, b in zip(graph_state, normal_state, strict=True))

    # Repeat using each implementation's own argmax as the next input.
    for values in (eager_state, normal_state, graph_state):
        reset(values)
    next_tokens = [1, 1, 1]
    sequences = [[], [], []]
    first_greedy_difference = None
    for index in range(40):
        current_position = torch.tensor([[index]], device="cuda")
        logits = []
        for slot, (function, values) in enumerate(((eager, eager_state),
                                                   (optimized, normal_state),
                                                   (graph, graph_state))):
            current_token = torch.tensor([[next_tokens[slot]]], device="cuda")
            output = function(current_token, current_position, *values).float()
            logits.append(output)
            next_tokens[slot] = int(output.argmax(-1).item())
            sequences[slot].append(next_tokens[slot])
        if first_greedy_difference is None and next_tokens[0] != next_tokens[2]:
            reference = logits[0].flatten()
            actual = logits[2].flatten()
            first_greedy_difference = {
                "position": index + 1,
                "eager_token": next_tokens[0],
                "megabake_token": next_tokens[2],
                "eager_score_margin_for_those_tokens": (
                    reference[next_tokens[0]] - reference[next_tokens[2]]
                ).item(),
                "relative_l2_vs_eager": ((actual - reference).norm() / reference.norm()).item(),
            }

report = {
    "model": model_path,
    "capacity": capacity,
    "teacher_steps": steps,
    "teacher_top_matches_eager": eager_top_matches,
    "teacher_relative_l2_median_vs_eager": statistics.median(eager_relative_l2),
    "teacher_relative_l2_max_vs_eager": max(eager_relative_l2),
    "teacher_relative_l2_max_position": max(range(steps), key=eager_relative_l2.__getitem__),
    "teacher_top_mismatch_positions": eager_mismatches,
    "teacher_relative_l2_by_position": eager_relative_l2,
    "teacher_all_logits_finite": teacher_all_finite,
    "checkpoints": teacher,
    "graph_vs_normal_max_abs": graph_vs_normal_max_abs,
    "graph_vs_normal_top_matches": graph_vs_normal_top_matches,
    "graph_vs_normal_cache_equal": graph_state_matches_normal,
    "length_counts": length_counts,
    "greedy_steps": 40,
    "greedy_sequences": dict(zip(("eager", "normal", "graph"), sequences, strict=True)),
    "greedy_graph_matches_normal": sequences[2] == sequences[1],
    "greedy_graph_matches_eager": sequences[2] == sequences[0],
    "greedy_first_mismatch_eager": next(
        (index + 1 for index, (reference, actual) in enumerate(zip(sequences[0], sequences[2], strict=True))
         if reference != actual), None
    ),
    "greedy_first_mismatch_details": first_greedy_difference,
}
Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
