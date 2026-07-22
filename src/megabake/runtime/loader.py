"""Load a compiled schedule and execute it through the megakernel."""

import torch

from megabake.schedule_compiler.graph_walker import CompiledModel
from megabake.schedule_compiler.serializer import load_schedule
from megabake.runtime.launcher import _launch_cooperative


def execute_model(
    compiled: CompiledModel,
    state_dict: dict[str, torch.Tensor],
    inputs: list[torch.Tensor],
    num_sms: int | None = None,
) -> torch.Tensor:
    header, tasks, buffer_descs, weight_maps, weight_names = load_schedule(
        compiled.schedule_bytes
    )

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    weight_tensors = {}
    for wm in weight_maps:
        name = weight_names[wm.buffer_index]
        weight_tensors[wm.buffer_index] = state_dict[name].contiguous().cuda().half()

    workspace = {}
    for bd in buffer_descs:
        num_elements = max(bd.size // 2, 1)
        workspace[bd.buffer_id] = torch.zeros(
            num_elements, dtype=torch.float16, device="cuda"
        )

    ptrs = [0] * compiled.num_buffers
    for bd in buffer_descs:
        ptrs[bd.buffer_id] = workspace[bd.buffer_id].data_ptr()
    for buf_id, tensor in weight_tensors.items():
        ptrs[buf_id] = tensor.data_ptr()

    for i, input_buf_id in enumerate(compiled.input_buffer_ids):
        input_data = inputs[i].contiguous().cuda().half().flatten()
        workspace[input_buf_id][: input_data.numel()].copy_(input_data)

    task_bytes = b"".join(t.to_bytes() for t in tasks)
    d_tasks = torch.tensor(list(task_bytes), dtype=torch.uint8, device="cuda")
    ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
    d_dyn_dims = torch.zeros(8, dtype=torch.int32, device="cuda")

    _launch_cooperative(
        d_tasks.data_ptr(),
        len(tasks),
        ptr_tensor.data_ptr(),
        d_dyn_dims.data_ptr(),
        num_sms,
    )
    torch.cuda.synchronize()

    numel = 1
    for s in compiled.output_shape:
        numel *= s
    return workspace[compiled.output_buffer_id][:numel].reshape(
        compiled.output_shape
    ).clone()
