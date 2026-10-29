"""Independently retime decoder variants and check real prompts against native HF.

Run from the MegaBake checkout with its environment and a model's tuning file.
This uses a separately written timer and the same explicit-cache graph wrappers
as the original benchmark. Native eager uses Transformers' growing cache.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from benchmarks.full_model import GraphedStep, _load, decode_module  # noqa: E402
from megabake import graphed, make_backend  # noqa: E402


model_path, output_path = sys.argv[1:3]
prompts = [
    "The capital of France is",
    "Explain why the sky appears blue.",
    "Write a Python function to add two numbers.",
    "2 + 2 =",
    "Once upon a time in a distant village,",
]
warmups, samples, capacity, generated_tokens = 5, 24, 64, 16
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
prompt_ids = [tokenizer(p, add_special_tokens=True)["input_ids"] for p in prompts]
if any(not ids or len(ids) + generated_tokens >= capacity for ids in prompt_ids):
    raise ValueError("a prompt is empty or does not fit the audit cache")
model = _load(model_path, None, "eager")
token = torch.tensor([[504]], device="cuda", dtype=torch.long)
positions = [torch.tensor([[i]], device="cuda", dtype=torch.long)
             for i in range(warmups + samples)]


def timed(label, step):
    for index in range(warmups):
        step(index)
    torch.cuda.synchronize()
    wall, event = [], []
    for index in range(warmups, warmups + samples):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns()
        start.record()
        step(index)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns() - t0) / 1_000_000)
        event.append(start.elapsed_time(end))
    return {"name": label, "samples": samples, "warmups": warmups,
            "wall_median_ms": statistics.median(wall),
            "event_span_median_ms": statistics.median(event),
            "wall_samples_ms": wall, "event_samples_ms": event}


def native_reference(ids):
    cache = None
    prompt_logits = []
    for token_id in ids:
        out = model(input_ids=torch.tensor([[token_id]], device="cuda"),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        prompt_logits.append(out.logits[0, -1].detach().float().cpu())
    next_token = int(prompt_logits[-1].argmax().item())
    sequence = []
    for index in range(generated_tokens):
        sequence.append(next_token)
        if index + 1 < generated_tokens:
            out = model(input_ids=torch.tensor([[next_token]], device="cuda"),
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = int(out.logits[0, -1].argmax().item())
    return prompt_logits, sequence


report = {"model": model_path, "gpu": torch.cuda.get_device_name(),
          "capability": list(torch.cuda.get_device_capability()),
          "dtype": str(next(model.parameters()).dtype),
          "timing_notes": "Synchronized host wall time, with device positions prepared before timing. "
                          "Native eager uses HF's growing cache; other variants use capacity-64 "
                          "explicit state. CUDA event spans include device idle gaps from host work.",
          "timings": {}, "prompts": []}

with torch.no_grad():
    holder = {"cache": None}
    def native_step(_index):
        out = model(input_ids=token, past_key_values=holder["cache"], use_cache=True)
        holder["cache"] = out.past_key_values
        return out.logits
    report["timings"]["native_hf_eager"] = timed("native_hf_eager", native_step)
    del holder
    native = [native_reference(ids) for ids in prompt_ids]
    print("Native eager timing and prompt references complete", flush=True)

    eager, shapes = decode_module(model, capacity)
    def make_state():
        return [torch.zeros(shape, dtype=dtype, device="cuda")
                for layer in shapes for shape, dtype in layer]

    eager_state, torch_state, megabake_state = make_state(), make_state(), make_state()
    report["timings"]["wrapper_eager"] = timed(
        "wrapper_eager", lambda i: eager(token, positions[i], *eager_state))
    for value in eager_state:
        value.zero_()

    torch_graph = GraphedStep(torch.compile(eager, fullgraph=True, dynamic=False),
                              token.clone(), torch_state)
    report["timings"]["torch_graph"] = timed(
        "torch_graph", lambda i: torch_graph(token, positions[i], *torch_state))
    print("PyTorch graph timing complete", flush=True)

    backend = make_backend(mode="decode", compile_kernel=True, cache=False, scheduler="phase")
    compiled = torch.compile(eager, backend=backend, fullgraph=True, dynamic=False)
    megabake_graph = graphed(compiled, token.clone(), positions[0].clone(), *megabake_state)
    report["timings"]["megabake_graph"] = timed(
        "megabake_graph", lambda i: megabake_graph(token, positions[i], *megabake_state))
    print("MegaBake graph timing complete", flush=True)

    variants = (("wrapper_eager", eager, eager_state),
                ("torch_graph", torch_graph, torch_state),
                ("megabake_graph", megabake_graph, megabake_state))
    for prompt, ids, (native_logits, native_sequence) in zip(prompts, prompt_ids, native, strict=True):
        case = {"prompt": prompt, "tokens": len(ids), "native_greedy_ids": native_sequence,
                "variants": {}}
        for name, function, state in variants:
            for value in state:
                value.zero_()
            rels, matches, finite, out = [], 0, True, None
            for position, (token_id, ref) in enumerate(zip(ids, native_logits, strict=True)):
                out = function(torch.tensor([[token_id]], device="cuda"),
                               torch.tensor([[position]], device="cuda"), *state)
                actual = out[0, -1].detach().float().cpu()
                finite &= bool(torch.isfinite(actual).all())
                rels.append(float((actual - ref).norm() / ref.norm()))
                matches += int(actual.argmax().item() == ref.argmax().item())
            next_token = int(actual.argmax().item())
            sequence = []
            for index in range(generated_tokens):
                sequence.append(next_token)
                if index + 1 < generated_tokens:
                    output = function(torch.tensor([[next_token]], device="cuda"),
                                      torch.tensor([[len(ids) + index]], device="cuda"), *state)
                    finite &= bool(torch.isfinite(output).all().item())
                    next_token = int(output.argmax().item())
            mismatch = next((i + 1 for i, (a, b) in enumerate(
                zip(native_sequence, sequence, strict=True)) if a != b), None)
            case["variants"][name] = {
                "prompt_top_matches_native": matches, "prompt_positions": len(ids),
                "prompt_relative_l2_median_vs_native": statistics.median(rels),
                "prompt_relative_l2_max_vs_native": max(rels), "all_logits_finite": finite,
                "greedy_first_difference_vs_native": mismatch, "greedy_ids": sequence,
            }
        report["prompts"].append(case)
        print(f"Prompt check complete: {prompt}", flush=True)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"timing_ms": {k: round(v["wall_median_ms"], 3)
                                for k, v in report["timings"].items()},
                  "prompt_first_difference": {row["prompt"]: {
                      k: v["greedy_first_difference_vs_native"]
                      for k, v in row["variants"].items()} for row in report["prompts"]}}, indent=2),
      flush=True)
