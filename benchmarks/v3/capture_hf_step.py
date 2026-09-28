#!/usr/bin/env python3
"""Capture and validate the declared Hugging Face cached-step cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from hf_cached_step import (  # loaded by direct script execution
    CHECKPOINT,
    DEFAULT_CAPACITY,
    DEFAULT_LENGTHS,
    REVISION,
    HuggingFaceCachedStep,
    assert_step_outputs,
    build_manifest,
    cuda_driver_version,
    export_and_capture,
    graph_break_report,
    load_hf_model,
    make_step_inputs,
    native_reference_step,
    sha256_text,
    write_json,
    write_manifest,
)
from megabake.v3.frontend.capture import graph_hash
from shape_inventory import inventory_exported_graph


def target_name() -> str:
    prop = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    driver = cuda_driver_version()
    return f"{prop.name}; cc={major}.{minor}; visible_sms={prop.multi_processor_count}; total_memory={prop.total_memory}; driver={driver}"


def capture_cell(
    model: torch.nn.Module,
    *,
    length: int,
    capacity: int,
    batch_size: int,
    checkpoint: str,
    revision: str,
    manifests_dir: Path,
    artifacts_dir: Path,
    target: str,
) -> dict:
    config = model.config
    if length + 1 >= capacity:
        raise ValueError(f"capacity {capacity} must leave room for both advancing calls after L={length}")
    if length + 2 > config.max_position_embeddings:
        raise ValueError(f"L={length} exceeds checkpoint position limit {config.max_position_embeddings}")
    inputs = make_step_inputs(model, length, capacity, batch_size)
    cell_id = f"smollm2-135m-fp16-b{batch_size}-L{length}"
    cell_dir = artifacts_dir / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    state = inputs.old_cache
    stale_replay_state = state.clone()
    token = inputs.input_ids
    graph_records = []
    stale_cache_negative_rejected = False

    for position in (length, length + 1):
        step = HuggingFaceCachedStep(model, position, capacity).eval()
        args = (token, state)
        exported, program, step_abi = export_and_capture(step, args)
        before = state.clone()
        with torch.inference_mode():
            expected = native_reference_step(model, token, state, position)
            actual = program.run_reference(*args)
        errors = assert_step_outputs(expected, actual)
        if position == length + 1:
            stale_output = native_reference_step(model, token, stale_replay_state, position)
            try:
                assert_step_outputs(actual, stale_output)
            except AssertionError:
                stale_cache_negative_rejected = True
            if not stale_cache_negative_rejected:
                raise AssertionError("stale replayed KV state passed the consecutive-step oracle")
        if not torch.equal(before, state):
            raise AssertionError(f"old KV input mutated at L={position}")
        untouched = torch.ones(capacity, dtype=torch.bool, device=state.device)
        untouched[position] = False
        if not torch.equal(actual["cache"][:, :, :, :, untouched, :], state[:, :, :, :, untouched, :]):
            raise AssertionError(f"cache slots outside [L,L+1) changed at L={position}")

        graph_source = exported.graph_module.code
        graph_path = cell_dir / f"capture-L{position}.py"
        graph_path.write_text(graph_source)
        report = graph_break_report(exported)
        record = {
            "position": position,
            "graph_hash": graph_hash(program),
            "graph_source_sha256": sha256_text(graph_source),
            "fx_node_count": report["fx_node_count"],
            "user_inputs": [item.arg.name for item in exported.graph_signature.input_specs if str(item.kind).split(".")[-1].lower() == "user_input"],
            "lifted_binding_count": len(step_abi.lifted_bindings),
            "unsupported_custom_operations": report["unsupported_custom_operations"],
            "graph_breaks": report["graph_breaks"],
            "fx_hot_math": inventory_exported_graph(exported, config),
            "correctness_max_abs": errors,
            "old_state_immutable": True,
            "untouched_slots_preserved": True,
        }
        graph_records.append(record)
        state = actual["cache"]
        token = inputs.next_input_ids

        if position == length:
            manifest = build_manifest(
                program,
                step_abi=step_abi,
                checkpoint=checkpoint,
                revision=revision,
                valid_length=length,
                capacity=capacity,
                batch_size=batch_size,
                dtype=state.dtype,
                config=config,
                target=target,
                cell_id=cell_id,
            )
            manifest_path = manifests_dir / f"{cell_id}.json"
            write_manifest(manifest_path, manifest)

    capture_report = {
        "schema_version": 1,
        "task_id": "V3R-002H",
        "checkpoint": checkpoint,
        "revision": revision,
        "cell_id": cell_id,
        "position": length,
        "dtype": "float16",
        "target": target,
        "model_shape": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "layers": config.num_hidden_layers,
            "query_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "vocabulary": config.vocab_size,
        },
        "cache": {"capacity": capacity, "layout": "layers,kv,batch,heads,capacity,head_dim", "functional": True},
        "model_attention": {"implementation": model.config._attn_implementation, "cudnn_sdpa_enabled": False},
        "captures": graph_records,
        "unsupported_nodes": [],
        "manifest": str(manifest_path),
        "manifest_hash": manifest.contract_hash,
        "state_transition": "two consecutive advancing steps; only [L,L+1) writes each call",
        "negative_checks": {
            "stale_cache_replay_rejected": stale_cache_negative_rejected,
            "capacity_overflow_guard": "HuggingFaceCachedStep rejects L >= C",
            "use_cache_false_substitute": "not used; graph output requires logits, new KV and valid_length",
        },
    }
    report_path = cell_dir / "capture_report.json"
    write_json(report_path, capture_report)
    return {"cell_id": cell_id, "manifest": str(manifest_path), "manifest_hash": manifest.contract_hash, "report": str(report_path), "capture_hashes": [item["graph_hash"] for item in graph_records]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--manifests-dir", type=Path, default=Path("benchmarks/v3/manifests"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("ART/tasks/V3R-002H/capture"))
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths) or min(args.lengths) < 0:
        parser.error("lengths must be distinct non-negative integers")
    model = load_hf_model(args.checkpoint, args.revision)
    target = target_name()
    results = [
        capture_cell(
            model,
            length=length,
            capacity=args.capacity,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            revision=args.revision,
            manifests_dir=args.manifests_dir,
            artifacts_dir=args.artifacts_dir,
            target=target,
        )
        for length in args.lengths
    ]
    print({"target": target, "results": results})


if __name__ == "__main__":
    main()
