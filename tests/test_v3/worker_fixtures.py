"""Small FX programs used to exercise the generated CUDA worker entry."""

from __future__ import annotations

from typing import Any

import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import WorkerProgram, lower_worker_program
from megabake.v3.contracts import StepABI
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.frontend.semantic import index_program
from megabake.v3.logical import lower_logical_plan


def _block_graph(cast_dtype: torch.dtype = torch.float32) -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    bias = graph.placeholder("bias")
    gate = graph.placeholder("gate")
    cache = graph.placeholder("cache")
    position = graph.placeholder("position")
    transposed = graph.call_function(torch.ops.aten.transpose.int, (weight, 0, 1))
    projected = graph.call_function(
        torch.ops.aten.addmm.default, (bias, x, transposed),
        {"beta": -0.25, "alpha": 1.5},
    )
    cast = graph.call_function(torch.ops.aten._to_copy.default, (projected,),
                               {"dtype": cast_dtype})
    reduced = graph.call_function(torch.ops.aten.sum.dim_IntList, (cast, [1], True))
    activated = graph.call_function(torch.ops.aten.silu.default, (reduced,))
    gated = graph.call_function(torch.ops.aten.mul.Tensor, (activated, gate))
    update = graph.call_function(torch.ops.aten.view.default, (gated, [1, 1, 1, 1]))
    new_cache = graph.call_function(torch.ops.aten.index_copy.default,
                                    (cache, 2, position, update), name="index_copy")
    graph.output({"cache": new_cache, "current": gated})
    result = torch.fx.GraphModule({}, graph)
    result.graph.lint()
    result.recompile()
    return result


def block_inputs(*, device: str = "cpu", cast_dtype: torch.dtype = torch.float32,
                 seed: int = 2126) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = (
        torch.randn((1, 4), generator=generator, dtype=torch.float16),
        torch.randn((5, 4), generator=generator, dtype=torch.float16),
        torch.randn((5,), generator=generator, dtype=torch.float16),
        torch.randn((1, 1), generator=generator),
        torch.full((1, 1, 4, 1), -9.0),
        torch.tensor([1], dtype=torch.int64),
    )
    return tuple(value.to(device=device) for value in tensors)


def _step_abi(cast_dtype: torch.dtype) -> StepABI:
    dtype_name = str(cast_dtype).removeprefix("torch.")
    return StepABI.from_dict({
        "schema_version": 1,
        "ordered_user_inputs": [
            {"placeholder": name, "path": [index]}
            for index, name in enumerate(("x", "weight", "bias", "gate", "cache", "position"))
        ],
        "lifted_bindings": {},
        "old_state_inputs": [{
            "placeholder": "cache", "state_id": "kv", "path": [4],
            "layout": "b,h,capacity,d", "alias_set": "state:kv", "capacity": 4,
        }],
        "state_effects": [{
            "effect_id": "append-kv", "state_id": "kv", "order": 0,
            "reads": [{"state_id": "kv", "range": "[0,L)"}],
            "writes": [{"state_id": "kv", "range": "[L,L+1)"}],
        }],
        "new_state_outputs": [{"state_id": "kv", "path": ["cache"], "source_id": "index_copy"}],
        "user_output_tree": {
            "structure": {"cache": "tensor", "current": "tensor"},
            "leaves": [
                {"path": ["cache"], "ownership": "owned", "lifetime": "returned_to_caller"},
                {"path": ["current"], "ownership": "owned", "lifetime": "returned_to_caller"},
            ],
        },
        "position_and_valid_length": {
            "position_source": "position", "old_valid_length_source": "position",
            "append_position_expression": "L", "new_valid_length_expression": "L+1",
            "attend_range": "[0,L+1) after append",
        },
        "batch_rule": "uniform_valid_length",
        "invocation_preparation": {"setup_amortization": "none", "actions": []},
        "guard_set": {
            "shapes": {"x": [1, 4], "weight": [5, 4], "bias": [5], "gate": [1, 1],
                       "cache": [1, 1, 4, 1], "position": [1]},
            "strides": {"x": [4, 1], "weight": [4, 1], "bias": [1], "gate": [1, 1],
                        "cache": [4, 4, 1, 1], "position": [1]},
            "dtypes": {"x": "float16", "weight": "float16", "bias": "float16",
                       "gate": "float32", "cache": "float32", "position": "int64",
                       "cast": dtype_name},
            "capacity": {"cache": 4},
            "features": ["aten.transpose", "aten.addmm", "aten._to_copy", "aten.sum",
                         "aten.silu", "aten.mul", "aten.view", "aten.index_copy"],
            "numerical_policy_hash": f"g2-block-cast-{dtype_name}-v1",
        },
        "state_mode": "advancing",
        "cache_update_mode": "functional_append",
    })


def capture_unfamiliar_block(*, device: str = "cpu",
                             cast_dtype: torch.dtype = torch.float32):
    """Capture the predeclared view/addmm/cast/reduce/gate/state-write block."""
    examples = block_inputs(device=device)
    abi = _step_abi(cast_dtype)
    captured = capture_graph_module(
        _block_graph(cast_dtype), examples,
        input_spec={"args": len(examples)},
        output_spec=abi.user_output_tree["structure"],
        state_bindings={"cache": "kv"}, step_abi=abi, reexport=False,
    )
    indexed = index_program(captured)
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    plan = lower_logical_plan(indexed, choices, selected)
    return captured, indexed, choices, plan, examples


def capture_reduce_program(*, device: str = "cpu"):
    class ScaleThenReduce(torch.nn.Module):
        def forward(self, x):
            scaled = torch.ops.aten.mul.Scalar(x, 1.25)
            shifted = torch.ops.aten.add.Scalar(scaled, 0.125)
            return torch.ops.aten.sum.dim_IntList(shifted, [1], False)

    examples = (torch.randn((2, 65), device=device, dtype=torch.float32),)
    exported = torch.export.export(ScaleThenReduce(), examples)
    from megabake.v3.frontend.normalize import normalize_fx
    from megabake.v3.frontend.semantic import index_program

    captured = normalize_fx(exported, input_spec={})
    indexed = index_program(captured)
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    plan = lower_logical_plan(indexed, choices, selected)
    return captured, indexed, plan, examples


def make_worker(program: Any, logical_plan: Any, profile: CudaTargetProfile,
                *, block_threads: int = 32) -> WorkerProgram:
    return lower_worker_program(program, logical_plan, profile, block_threads=block_threads)


def reference_trace(captured: Any, examples: tuple[Any, ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node: Any) -> Any:
            value = super().run_node(node)
            values[node.name] = value
            return value

    Recorder(captured.graph_module).run(*examples)
    return values


__all__ = ["block_inputs", "capture_reduce_program", "capture_unfamiliar_block",
           "make_worker", "reference_trace"]
