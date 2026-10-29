"""Compare teacher-forced decode with an FP32 copy of the BF16 weights."""
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from benchmarks.full_model import GraphedStep, _load, decode_module
from megabake import make_backend


model_path, output_path = sys.argv[1:3]
capacity, steps = 192, 128
torch.backends.cuda.matmul.allow_tf32 = False
model = _load(model_path, None, "eager")
fp32_model = copy.deepcopy(model).float().eval()
eager, shapes = decode_module(model, capacity)
fp32, fp32_shapes = decode_module(fp32_model, capacity)


def make_state(state_shapes, use_fp32=False):
    return [torch.zeros(shape, device="cuda", dtype=torch.float32 if use_fp32 and dtype == torch.bfloat16 else dtype)
            for layer in state_shapes for shape, dtype in layer]


fp32_state, eager_state, optimized_state, graph_state = (
    make_state(fp32_shapes, True), make_state(shapes), make_state(shapes), make_state(shapes)
)
backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
optimized = torch.compile(eager, backend=backend, fullgraph=True, dynamic=False)
results = {"eager": {"relative_l2": [], "mismatch_positions": []},
           "megabake": {"relative_l2": [], "mismatch_positions": []},
           "torch_graph": {"relative_l2": [], "mismatch_positions": []}}
with torch.no_grad():
    graph = GraphedStep(torch.compile(eager, fullgraph=True, dynamic=False),
                        torch.tensor([[1]], device="cuda"), graph_state)
    for index in range(steps):
        token = torch.tensor([[([1, 504, 3458, 314][index % 4])]], device="cuda")
        position = torch.tensor([[index]], device="cuda")
        reference = fp32(token, position, *fp32_state).float()
        for label, function, state in (("eager", eager, eager_state),
                                       ("megabake", optimized, optimized_state),
                                       ("torch_graph", graph, graph_state)):
            actual = function(token, position, *state).float()
            results[label]["relative_l2"].append(((actual - reference).norm() / reference.norm()).item())
            if actual.argmax(-1).item() != reference.argmax(-1).item():
                results[label]["mismatch_positions"].append(index)

for result in results.values():
    values = result["relative_l2"]
    result["median_relative_l2"] = statistics.median(values)
    result["max_relative_l2"] = max(values)
    result["max_relative_l2_position"] = max(range(steps), key=values.__getitem__)
    result["top_matches"] = steps - len(result["mismatch_positions"])
report = {"model": model_path, "steps": steps, "reference": "FP32 arithmetic on BF16-rounded weights",
          "tf32_enabled": False, "results": results}
Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
print(json.dumps({name: {k: v for k, v in result.items() if k != "relative_l2"}
                  for name, result in results.items()}, indent=2), flush=True)
