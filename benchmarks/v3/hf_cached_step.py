"""Pinned, fixed-capacity adapter for a complete Transformers decode step."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

import torch
from torch import nn

from megabake.v3.contracts import (
    BenchmarkCell,
    ExceptionalValuePolicy,
    NumericalPolicy,
    StepABI,
    StepManifest,
    ToleranceSpec,
    WorkloadSpec,
    canonical_json,
)
from megabake.v3.frontend.capture import capture_exported_program, graph_hash


CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DEFAULT_LENGTHS = (128, 2048)
DEFAULT_CAPACITY = 2050
OUTPUT_ATOL = 0.1
OUTPUT_RTOL = 0.005


class HuggingFaceCachedStep(nn.Module):
    """One length-specialized call with functional fixed-capacity KV state."""

    def __init__(self, model: nn.Module, valid_length: int, capacity: int):
        super().__init__()
        if not isinstance(valid_length, int) or valid_length < 0:
            raise ValueError("valid_length must be a non-negative integer")
        if not isinstance(capacity, int) or valid_length >= capacity:
            raise ValueError("valid_length must be below the positive cache capacity")
        self.model = model
        self.valid_length = valid_length
        self.capacity = capacity

    def forward(self, input_ids: torch.Tensor, old_cache: torch.Tensor) -> dict[str, torch.Tensor]:
        length = self.valid_length
        from transformers import DynamicCache

        past = DynamicCache(
            ddp_cache_data=tuple(
                (old_cache[i, 0, :, :, :length, :], old_cache[i, 1, :, :, :length, :])
                for i in range(self.model.config.num_hidden_layers)
            ),
            config=self.model.config,
        )
        position = torch.tensor([length], dtype=torch.long, device=input_ids.device)
        output = self.model(
            input_ids=input_ids,
            past_key_values=past,
            attention_mask=torch.ones(
                (input_ids.shape[0], length + 1), dtype=torch.long, device=input_ids.device
            ),
            cache_position=position,
            use_cache=True,
            return_dict=True,
        )

        updated = []
        for i in range(self.model.config.num_hidden_layers):
            layer = output.past_key_values.layers[i]
            key = old_cache[i, 0].index_copy(2, position, layer.keys[:, :, length:length + 1, :])
            value = old_cache[i, 1].index_copy(2, position, layer.values[:, :, length:length + 1, :])
            updated.append(torch.stack((key, value), dim=0))
        return {
            "logits": output.logits,
            "cache": torch.stack(updated),
            "valid_length": torch.tensor(length + 1, dtype=torch.long, device=input_ids.device),
        }


@dataclass(frozen=True)
class StepInputs:
    input_ids: torch.Tensor
    old_cache: torch.Tensor
    next_input_ids: torch.Tensor
    seed: int


def configure_sdpa() -> None:
    # This environment lacks cuDNN's runtime-compiled SDPA engine library.
    torch.backends.cuda.enable_cudnn_sdp(False)


def load_hf_model(
    checkpoint: str = CHECKPOINT,
    revision: str = REVISION,
    dtype: torch.dtype = torch.float16,
    *,
    local_files_only: bool = True,
) -> nn.Module:
    if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("checkpoint revision must be a full lowercase commit hash")
    if not torch.cuda.is_available():
        raise RuntimeError("V3R-002H requires the declared CUDA device")
    configure_sdpa()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=revision,
        dtype=dtype,
        local_files_only=local_files_only,
    ).cuda().eval()
    model.requires_grad_(False)
    return model


def make_step_inputs(
    model: nn.Module,
    valid_length: int,
    capacity: int,
    batch_size: int = 1,
    seed: int = 20260927,
) -> StepInputs:
    if valid_length < 0 or valid_length >= capacity:
        raise ValueError("valid_length must satisfy 0 <= valid_length < capacity")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    config = model.config
    generator = torch.Generator(device="cpu").manual_seed(seed + valid_length + batch_size)
    prefix_ids = torch.randint(
        config.vocab_size, (batch_size, valid_length), generator=generator, dtype=torch.long
    ).to(device)
    next_ids = torch.randint(
        config.vocab_size, (batch_size, 2), generator=generator, dtype=torch.long
    ).to(device)

    if valid_length:
        with torch.inference_mode():
            prefix = model(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=True,
                return_dict=True,
            ).past_key_values
    else:
        from transformers import DynamicCache

        prefix = DynamicCache(config=config)

    heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    state = torch.zeros(
        (config.num_hidden_layers, 2, batch_size, heads, capacity, head_dim),
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    for i, layer in enumerate(prefix.layers):
        if valid_length:
            state[i, 0, :, :, :valid_length, :].copy_(layer.keys)
            state[i, 1, :, :, :valid_length, :].copy_(layer.values)
    input_ids = torch.empty((batch_size, 1), dtype=next_ids.dtype, device=device)
    next_input_ids = torch.empty_like(input_ids)
    input_ids.copy_(next_ids[:, :1])
    next_input_ids.copy_(next_ids[:, 1:])
    return StepInputs(input_ids, state, next_input_ids, seed)


def native_reference_step(
    model: nn.Module, input_ids: torch.Tensor, old_cache: torch.Tensor, valid_length: int
) -> dict[str, torch.Tensor]:
    from transformers import DynamicCache

    config = model.config
    past = DynamicCache(
        ddp_cache_data=tuple(
            (
                old_cache[i, 0, :, :, :valid_length, :].clone(),
                old_cache[i, 1, :, :, :valid_length, :].clone(),
            )
            for i in range(config.num_hidden_layers)
        ),
        config=config,
    )
    position = torch.tensor([valid_length], dtype=torch.long, device=input_ids.device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            past_key_values=past,
            attention_mask=torch.ones(
                (input_ids.shape[0], valid_length + 1), dtype=torch.long, device=input_ids.device
            ),
            cache_position=position,
            use_cache=True,
            return_dict=True,
        )
    updated = []
    for i in range(config.num_hidden_layers):
        layer = output.past_key_values.layers[i]
        key = old_cache[i, 0].index_copy(2, position, layer.keys[:, :, valid_length:valid_length + 1, :])
        value = old_cache[i, 1].index_copy(2, position, layer.values[:, :, valid_length:valid_length + 1, :])
        updated.append(torch.stack((key, value), dim=0))
    return {
        "logits": output.logits,
        "cache": torch.stack(updated),
        "valid_length": position.new_tensor(valid_length + 1),
    }


def assert_step_outputs(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], *, atol: float = OUTPUT_ATOL,
    rtol: float = OUTPUT_RTOL,
) -> dict[str, float]:
    if not isinstance(actual, dict) or tuple(actual) != ("logits", "cache", "valid_length"):
        raise AssertionError("cached-step output tree changed")
    if tuple(expected) != tuple(actual):
        raise AssertionError("reference and candidate output trees differ")
    maxima: dict[str, float] = {}
    for name in ("logits", "cache"):
        left, right = expected[name], actual[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f"{name} shape/dtype differs: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}")
        maxima[name] = float((left.float() - right.float()).abs().max().item())
        torch.testing.assert_close(right, left, atol=atol, rtol=rtol, equal_nan=False)
    if not torch.equal(expected["valid_length"], actual["valid_length"]):
        raise AssertionError("valid_length differs")
    maxima["valid_length"] = 0.0
    return maxima


def make_step_abi(
    exported_program: Any,
    *,
    valid_length: int,
    capacity: int,
    batch_size: int,
    dtype: torch.dtype,
    config: Any,
    numerical_policy_hash: str,
) -> StepABI:
    user_specs = [
        item for item in exported_program.graph_signature.input_specs
        if str(item.kind).split(".")[-1].lower() == "user_input"
    ]
    user_names = [item.arg.name for item in user_specs]
    if user_names != ["input_ids", "old_cache"]:
        raise ValueError(f"unexpected exported user input signature: {user_names!r}")
    lifted = {}
    for item in exported_program.graph_signature.input_specs:
        kind = str(item.kind).split(".")[-1].lower()
        if kind not in {"parameter", "buffer", "constant_tensor"}:
            continue
        role = "weight" if kind == "parameter" else ("state" if kind == "buffer" else "constant")
        lifted[item.arg.name] = {"identity": item.target, "role": role, "lifetime": "session"}

    head_dim = config.hidden_size // config.num_attention_heads
    cache_shape = (config.num_hidden_layers, 2, batch_size, config.num_key_value_heads, capacity, head_dim)
    cache_strides = (
        2 * batch_size * config.num_key_value_heads * capacity * head_dim,
        batch_size * config.num_key_value_heads * capacity * head_dim,
        config.num_key_value_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    )
    return StepABI(
        ordered_user_inputs=(
            {"placeholder": "input_ids", "path": [0]},
            {"placeholder": "old_cache", "path": [1]},
        ),
        lifted_bindings=lifted,
        old_state_inputs=(
            {
                "placeholder": "old_cache",
                "state_id": "kv",
                "path": [1],
                "layout": "layers,kv,batch,heads,capacity,head_dim",
                "alias_set": "old-kv",
                "capacity": capacity,
            },
        ),
        state_effects=(
            {
                "effect_id": "functional-kv-append",
                "state_id": "kv",
                "order": 0,
                "reads": [{"state_id": "kv", "range": "[0,L) for every layer and key/value"}],
                "writes": [{"state_id": "kv", "range": "[L,L+1) for every layer and key/value"}],
            },
        ),
        new_state_outputs=({"state_id": "kv", "path": ["cache"], "source_id": "functional-index-copy"},),
        user_output_tree={
            "structure": {"logits": "tensor", "cache": "tensor", "valid_length": "tensor"},
            "leaves": [
                {"path": ["logits"], "ownership": "owned", "lifetime": "returned_to_caller"},
                {"path": ["cache"], "ownership": "owned", "lifetime": "returned_to_caller"},
                {"path": ["valid_length"], "ownership": "owned", "lifetime": "returned_to_caller"},
            ],
        },
        position_and_valid_length={
            "position_source": f"specialized_valid_length:{valid_length}",
            "old_valid_length_source": f"specialized_valid_length:{valid_length}",
            "append_position_expression": "L",
            "new_valid_length_expression": "L+1",
            "attend_range": "[0,L+1) after append",
        },
        batch_rule="uniform_valid_length",
        invocation_preparation={
            "setup_amortization": "stable_model_weights_per_session",
            "actions": [
                {"kind": "bind_token", "owner": "caller", "frequency": "per_invocation", "charged_to": "cached_step"},
                {"kind": "bind_old_state", "owner": "caller", "frequency": "per_invocation", "charged_to": "cached_step"},
            ],
        },
        guard_set={
            "shapes": {
                "input_ids": [batch_size, 1],
                "old_cache": list(cache_shape),
            },
            "strides": {"input_ids": [1, 1], "old_cache": list(cache_strides)},
            "dtypes": {"input_ids": "int64", "old_cache": str(dtype).removeprefix("torch.")},
            "capacity": {"old_cache": capacity},
            "position": {"specialized": valid_length},
            "numerical_policy_hash": numerical_policy_hash,
            "features": ["transformers.DynamicCache", "functional_append", "causal_attention",
                         "inference_only"],
        },
        state_mode="advancing",
        cache_update_mode="functional_append",
    )


def make_numerical_policy(config: Any, dtype: torch.dtype, valid_length: int) -> NumericalPolicy:
    return NumericalPolicy(
        reference_expansion=f"{config.model_type} causal-LM one-token cached decode at fixed valid length {valid_length}; HF SDPA with cuDNN SDPA disabled",
        intermediate_casts=("checkpoint/model operator cast boundaries",),
        accumulation_dtypes={"projection": "float32", "normalization": "float32", "attention": "float32"},
        output_casts={"logits": str(dtype).removeprefix("torch."), "kv": str(dtype).removeprefix("torch."), "valid_length": "int64"},
        # SDPA's online-softmax CUDA body reassociates the reference reduction;
        # admit that one operation only under the declared output tolerance.
        permitted_reassociation={"scaled_dot_product_attention": True},
        tolerances={"*": {"float16": ToleranceSpec(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, equal_nan=False)}},
        exceptional_value_policy=ExceptionalValuePolicy(False, False, False),
    )


def build_manifest(
    program: Any,
    *,
    step_abi: StepABI,
    checkpoint: str,
    revision: str,
    valid_length: int,
    capacity: int,
    batch_size: int,
    dtype: torch.dtype,
    config: Any,
    target: str,
    cell_id: str,
) -> StepManifest:
    numerical = make_numerical_policy(config, dtype, valid_length)
    workload = WorkloadSpec(
        batch_size=batch_size,
        context_length=valid_length,
        capacity=capacity,
        dtype=dtype,
        state_semantics="fixed_capacity_kv:read_old_write_new",
        mask_semantics="causal:[0,L+1) after append; no padding",
        cache_layout="layers,kv,batch,heads,capacity,head_dim",
        input_origin="gpu",
        output_ownership="owned",
        timed_unit="cached_step",
        checkpoint_id=checkpoint,
        config_id=f"{config.model_type}:hidden{config.hidden_size}:layers{config.num_hidden_layers}:heads{config.num_attention_heads}:kv{config.num_key_value_heads}",
        position=valid_length,
        shape_buckets=[f"B{batch_size}:L{valid_length}:C{capacity}:D{config.hidden_size // config.num_attention_heads}"],
        dropout=False,
        benchmark_cells=(BenchmarkCell(
            cell_id=cell_id,
            baseline="best_validated_equivalent_torch_compile",
            must_win=False,
            context_length=valid_length,
            notes="Workload-truth/baseline calibration cell; no MegaBake strict candidate is measured.",
        ),),
    )
    return StepManifest(
        workload=workload,
        numerical_policy=numerical,
        step_abi=step_abi,
        checkpoint_revision=revision,
        graph_hash=graph_hash(program),
        versions={
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "cuda_toolkit": cuda_toolkit_version(),
            "transformers": __import__("transformers").__version__,
            "target": target,
        },
        fixed_inputs={"seed": 20260927, "position": valid_length, "capacity": capacity, "input_length": 1},
        cell_status={cell_id: "not_measured"},
    )


def export_and_capture(
    step: HuggingFaceCachedStep,
    example_args: tuple[torch.Tensor, torch.Tensor],
) -> tuple[Any, Any, StepABI]:
    exported = torch.export.export(step, example_args, strict=False)
    policy = make_numerical_policy(
        step.model.config, example_args[1].dtype, step.valid_length
    )
    abi = make_step_abi(
        exported,
        valid_length=step.valid_length,
        capacity=step.capacity,
        batch_size=example_args[0].shape[0],
        dtype=example_args[1].dtype,
        config=step.model.config,
        numerical_policy_hash=policy.contract_hash,
    )
    program = capture_exported_program(
        exported,
        policy=policy,
        state_bindings={"old_cache": "kv"},
        step_abi=abi,
    )
    return exported, program, abi


def graph_break_report(exported_program: Any) -> dict[str, Any]:
    nodes = list(exported_program.graph_module.graph.nodes)
    custom = []
    histogram: dict[str, int] = {}
    for node in nodes:
        if node.op == "call_function":
            target = str(node.target)
            histogram[target] = histogram.get(target, 0) + 1
            if not target.startswith(("aten.", "torch.", "operator.", "_operator.", "<built-in function")):
                custom.append({"node_id": node.name, "target": target})
    return {
        "graph_breaks": [],
        "unsupported_custom_operations": custom,
        "fx_node_count": len(nodes),
        "call_function_count": sum(histogram.values()),
        "operator_histogram": dict(sorted(histogram.items())),
        "strict_cuda_lowering": "not_attempted",
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cuda_toolkit_version() -> str:
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return "not_installed"
    match = re.search(r"release ([0-9.]+), V([0-9.]+)", result.stdout)
    return f"{match.group(1)} (V{match.group(2)})" if match else "unknown"


def cuda_driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "not_available"
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else "not_available"


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_manifest(path: str | Path, manifest: StepManifest) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(manifest.to_json(indent=2) + "\n")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
