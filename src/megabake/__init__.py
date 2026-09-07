"""Megabake: torch.export to megakernel pipeline."""

__version__ = "0.1.0"

import torch

from megabake.schedule_compiler.graph_walker import (
    compile_model, compile_from_ep, CompiledModel,
)
from megabake.schedule_compiler.inductor_passes import optimize_graph
from megabake.runtime.loader import execute_model


def compile_fx(
    graph,
    example_args=None,
    *,
    sm_version: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> CompiledModel:
    """Compile an FX graph (ExportedProgram or GraphModule) to a megakernel.

    For ExportedProgram: example_args not needed (metadata already present).
    For GraphModule: example_args required for shape propagation.
    """
    from torch.export import ExportedProgram

    props = torch.cuda.get_device_properties(0)
    if sm_version is None:
        sm_version = props.major * 10 + props.minor
    num_sms = props.multi_processor_count

    if isinstance(graph, ExportedProgram):
        ep = optimize_graph(graph)
        return compile_from_ep(ep, sm_version, dtype=dtype, num_sms=num_sms)

    if isinstance(graph, torch.fx.GraphModule):
        if example_args is None:
            raise ValueError(
                "example_args required when passing a GraphModule "
                "(needed for shape propagation)"
            )
        if isinstance(example_args, torch.Tensor):
            example_args = (example_args,)
        else:
            example_args = tuple(example_args)

        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(graph).propagate(*example_args)

        ep = torch.export.export(graph, example_args, strict=False)
        ep = optimize_graph(ep)
        return compile_from_ep(ep, sm_version, dtype=dtype, num_sms=num_sms)

    raise TypeError(
        f"Expected ExportedProgram or GraphModule, got {type(graph).__name__}"
    )


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
    model_or_state_dict,
    *inputs: torch.Tensor,
    task_timings_ptr: int = 0,
) -> torch.Tensor:
    # ponytail: full eager fallback when unsupported ops exist.
    # graph splitting with segments if partial acceleration matters.
    if compiled.unsupported_ops:
        if not isinstance(model_or_state_dict, torch.nn.Module):
            raise RuntimeError(
                f"Model has unsupported ops ({', '.join(compiled.unsupported_ops)}) "
                f"and requires the original model (not a state_dict) for eager fallback."
            )
        with torch.no_grad():
            return model_or_state_dict(*inputs)

    if isinstance(model_or_state_dict, dict):
        sd = model_or_state_dict
    else:
        model = model_or_state_dict
        model_id = id(model)
        if model_id not in _cached_state_dict:
            sd = model.state_dict()
            for name, buf in model.named_buffers():
                if name not in sd:
                    sd[name] = buf
            _cached_state_dict[model_id] = sd
        sd = _cached_state_dict[model_id]
    return execute_model(compiled, sd, list(inputs), task_timings_ptr=task_timings_ptr)
