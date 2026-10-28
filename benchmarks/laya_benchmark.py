"""Audit and benchmark the pinned Laya decision model on CUDA.

The capture variant deliberately sets ``compile_kernel=False``.  It exercises
MegaBake's complete Dynamo/AOT capture and lowering pipeline while executing the
captured PyTorch graph, so unsupported semantics can be inventoried before kernel
work starts.

    .venv/bin/python benchmarks/laya.py --variants eager capture
    .venv/bin/python benchmarks/laya.py --out artifacts/laya/baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from megabake import (  # noqa: E402
    canonicalize_semantics,
    dump_lowering_graph,
    import_fx,
    make_backend,
    semantic_coverage,
)


MODEL = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
LAYER_SOURCE_REVISION = "b9fe351ce81e2d974bca2bf219c1e4030c0a05ea"
STATE = "The customer says the service is excellent, but their payment is due tomorrow."
QUESTIONS = {
    "sentiment": {
        "type": "choice",
        "instructions": "What is the sentiment?",
        "criteria": {
            "positive": "The text is positive",
            "negative": "The text is negative",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does this require urgent action?",
    },
    "quality": {
        "type": "score",
        "instructions": "Rate the response quality",
        "criteria": ["poor", "acceptable", "excellent"],
    },
}
INPUT_NAMES = (
    "input_ids",
    "attention_mask",
    "marker_pos",
    "marker_mask",
    "qtype",
)


def _load_agent(model: str, revision: str):
    try:
        import laya
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned Laya source before running this benchmark: "
            f"uv pip install 'git+https://github.com/NandhaKishorM/laya.git@{LAYER_SOURCE_REVISION}'"
        ) from error
    return laya.load(model, revision=revision, device="cuda", backend="eager")


def _batch(agent) -> tuple[torch.Tensor, ...]:
    from laya.common import collate_items

    question_ids = list(QUESTIONS)
    for question_id in question_ids:
        agent._check_question(question_id, QUESTIONS[question_id])
    internal = {
        question_id: agent._to_internal(QUESTIONS[question_id])
        for question_id in question_ids
    }
    items = agent._encode_state(STATE, question_ids, internal)
    batch = collate_items([items], agent.tok.pad_token_id)
    assert batch is not None
    return tuple(batch[name].to("cuda") for name in INPUT_NAMES)


def _run(agent, function: Callable[..., Any], inputs: tuple[torch.Tensor, ...]):
    with torch.autocast(device_type="cuda", dtype=agent.dtype):
        return function(*inputs)


def _measure(call: Callable[[], Any], warmup: int, samples: int) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    host_samples = []
    device_samples = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        call()
        end.record()
        end.synchronize()
        host_samples.append((time.perf_counter() - host_start) * 1e3)
        device_samples.append(start.elapsed_time(end))
    return {
        "host_ms_median": statistics.median(host_samples),
        "device_ms_median": statistics.median(device_samples),
        "host_ms_samples": host_samples,
        "device_ms_samples": device_samples,
    }


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "values": value.tolist(),
    }


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().norm()
    relative_l2 = difference.norm() / denominator if denominator > 0 else difference.norm()
    return {
        "max_abs": difference.max().item(),
        "relative_l2": relative_l2.item(),
    }


def _capture(
    agent, inputs: tuple[torch.Tensor, ...], dump_dir: Path, compile_kernel: bool
):
    from megabake.fx_handler.hf_attention import register_transformers_attention

    register_transformers_attention()
    agent.model.encoder.set_attn_implementation("megabake")
    backend = make_backend(
        mode="prefill",
        fallback="error",
        compile_kernel=compile_kernel,
        cache=False,
        dump_dir=dump_dir,
        model_name="laya",
    )
    captured = torch.compile(
        agent.model,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    actual = _run(agent, captured, inputs)
    prepared = backend.megabake_last_prepared
    graph = canonicalize_semantics(import_fx(prepared.graph, prepared.context))
    lowering_path = dump_lowering_graph(graph, dump_dir / "lowering_graph.txt")
    return captured, actual, {
        "coverage": semantic_coverage(graph),
        "operations": len(graph.ops),
        "lowering_graph": str(lowering_path),
        "capture_phase": prepared.context.phase,
        "graph_hash": prepared.context.graph_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--variants", nargs="+", choices=("eager", "capture"), default=("eager", "capture"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--compile-kernel", action="store_true")
    parser.add_argument("--dump-dir", type=Path, default=Path("artifacts/laya/dumps"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Laya benchmark requires CUDA")
    torch.manual_seed(0)
    load_started = time.perf_counter()
    agent = _load_agent(args.model, args.revision)
    load_seconds = time.perf_counter() - load_started
    inputs = _batch(agent)

    with torch.inference_mode():
        expected = _run(agent, agent.model, inputs)
        result: dict[str, Any] = {
            "model": args.model,
            "revision": args.revision,
            "laya_source_revision": LAYER_SOURCE_REVISION,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "autocast_dtype": str(agent.dtype),
            "input_shapes": {
                name: list(value.shape) for name, value in zip(INPUT_NAMES, inputs, strict=True)
            },
            "load_seconds": load_seconds,
            "reference": [_tensor_summary(value) for value in expected],
            "variants": {},
        }
        if "eager" in args.variants:
            result["variants"]["eager"] = _measure(
                lambda: _run(agent, agent.model, inputs), args.warmup, args.samples
            )
        if "capture" in args.variants:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            from unittest.mock import patch

            from megabake.fx_handler.hf_attention import (
                megabake_scaled_dot_product_attention,
            )

            compile_started = time.perf_counter()
            with patch(
                "torch.nn.functional.scaled_dot_product_attention",
                megabake_scaled_dot_product_attention,
            ):
                captured, actual, audit = _capture(
                    agent, inputs, args.dump_dir, args.compile_kernel
                )
                torch.cuda.synchronize()
                audit["first_call_seconds"] = time.perf_counter() - compile_started
                audit["errors"] = [
                    _error_summary(actual_value, expected_value)
                    for actual_value, expected_value in zip(actual, expected, strict=True)
                ]
                audit.update(
                    _measure(
                        lambda: _run(agent, captured, inputs),
                        args.warmup,
                        args.samples,
                    )
                )
            result["variants"]["capture"] = audit

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
