"""Locate first-token logit differences across native, eager and graph paths."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from benchmarks.full_model import GraphedStep, _graphable_kv_attention, _load, decode_module  # noqa: E402
from megabake import graphed, make_backend  # noqa: E402


model_path, output_path = sys.argv[1:3]
torch.backends.cuda.matmul.allow_tf32 = False
model = _load(model_path, None, "eager")
fp32_model = copy.deepcopy(model).float().eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
prompt_ids = tokenizer("Explain why the sky appears blue.", add_special_tokens=True)["input_ids"]
tokens = list(dict.fromkeys([1, 504, prompt_ids[0]]))
position = torch.zeros((1, 1), device="cuda", dtype=torch.long)


def cpu_logits(value):
    return value[0, -1].detach().float().cpu().clone()


with torch.no_grad():
    native = {i: cpu_logits(model(input_ids=torch.tensor([[i]], device="cuda"),
                                  use_cache=True).logits) for i in tokens}
    fp32_native = {i: cpu_logits(fp32_model(input_ids=torch.tensor([[i]], device="cuda"),
                                             use_cache=True).logits) for i in tokens}
    wrapper, shapes = decode_module(model, 64)
    fp32_wrapper, fp32_shapes = decode_module(fp32_model, 64)

    def make_state(state_shapes, fp32=False):
        return [torch.zeros(shape, dtype=(torch.float32 if fp32 and dtype == torch.bfloat16 else dtype),
                            device="cuda") for layer in state_shapes for shape, dtype in layer]

    def reset(state):
        for t in state:
            t.zero_()

    def execute(function, state):
        result = {}
        for i in tokens:
            reset(state)
            result[i] = cpu_logits(function(torch.tensor([[i]], device="cuda"), position, *state))
        return result

    outputs = {"native_bf16": native, "native_fp32": fp32_native}
    outputs["wrapper_bf16"] = execute(wrapper, make_state(shapes))
    outputs["wrapper_fp32"] = execute(fp32_wrapper, make_state(fp32_shapes, True))
    with _graphable_kv_attention():
        outputs["static_attention_eager"] = execute(wrapper, make_state(shapes))
    torch._dynamo.reset()
    outputs["compiled_dynamic_attention"] = execute(
        torch.compile(wrapper, fullgraph=True, dynamic=False), make_state(shapes))
    torch._dynamo.reset()
    with _graphable_kv_attention():
        outputs["compiled_static_attention"] = execute(
            torch.compile(wrapper, fullgraph=True, dynamic=False), make_state(shapes))
    torch._dynamo.reset()
    pt_state = make_state(shapes)
    pt_graph = GraphedStep(torch.compile(wrapper, fullgraph=True, dynamic=False),
                           torch.tensor([[1]], device="cuda"), pt_state)
    outputs["torch_graph"] = execute(pt_graph, pt_state)
    torch._dynamo.reset()
    mb_state = make_state(shapes)
    backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
    mb_graph = graphed(torch.compile(wrapper, backend=backend, fullgraph=True, dynamic=False),
                       torch.tensor([[1]], device="cuda"), position.clone(), *mb_state)
    outputs["megabake_graph"] = execute(mb_graph, mb_state)

report = {"model": model_path, "fp32_reference": "native FP32 arithmetic using BF16-rounded weights",
          "tf32_enabled": False, "tokens": {}}
for i in tokens:
    ref_eager, ref_fp32 = native[i], fp32_native[i]
    report["tokens"][str(i)] = {}
    for label, values in outputs.items():
        actual = values[i]
        report["tokens"][str(i)][label] = {
            "top_id": int(actual.argmax()),
            "finite": bool(torch.isfinite(actual).all()),
            "relative_l2_vs_native_bf16": float((actual - ref_eager).norm() / ref_eager.norm()),
            "relative_l2_vs_native_fp32": float((actual - ref_fp32).norm() / ref_fp32.norm()),
        }

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
