"""Megabake: torch.export to megakernel pipeline."""

__version__ = "0.1.0"

import torch

from megabake.schedule_compiler.graph_walker import compile_model, CompiledModel
from megabake.runtime.loader import execute_model


def compile(
    model: torch.nn.Module,
    example_input,
    *,
    sm_version: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> CompiledModel:
    if sm_version is None:
        props = torch.cuda.get_device_properties(0)
        sm_version = props.major * 10 + props.minor
    return compile_model(model, example_input, sm_version, dtype=dtype)


def run(
    compiled: CompiledModel,
    model: torch.nn.Module,
    *inputs: torch.Tensor,
) -> torch.Tensor:
    """Execute a compiled schedule through the megakernel.

    Args:
        compiled: Result from megabake.compile().
        model: The original model (for weight access).
        *inputs: Input tensors.

    Returns:
        Output tensor matching the model's forward() result.
    """
    return execute_model(compiled, model.state_dict(), list(inputs))
