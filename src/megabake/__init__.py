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
    props = torch.cuda.get_device_properties(0)
    if sm_version is None:
        sm_version = props.major * 10 + props.minor
    num_sms = props.multi_processor_count
    return compile_model(model, example_input, sm_version, dtype=dtype, num_sms=num_sms)


_cached_state_dict: dict[int, dict[str, torch.Tensor]] = {}


def run(
    compiled: CompiledModel,
    model: torch.nn.Module,
    *inputs: torch.Tensor,
) -> torch.Tensor:
    model_id = id(model)
    if model_id not in _cached_state_dict:
        sd = model.state_dict()
        for name, buf in model.named_buffers():
            if name not in sd:
                sd[name] = buf
        _cached_state_dict[model_id] = sd
    return execute_model(compiled, _cached_state_dict[model_id], list(inputs))
