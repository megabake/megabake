"""Load a compiled schedule and execute it through the megakernel."""

import torch

from megabake.schedule_compiler.graph_walker import CompiledModel
from megabake.schedule_compiler.serializer import load_schedule
from megabake.runtime.launcher import _launch_cooperative


class _CachedRunner:
    """Cache GPU-side state so repeated execute_model() calls skip allocation."""

    __slots__ = (
        "_compiled_id", "_d_tasks", "_ptr_tensor", "_d_dyn_dims",
        "_num_tasks", "_num_sms", "_workspace", "_weight_tensors",
        "_input_buffer_ids", "_output_buffer_id", "_output_shape", "_num_buffers",
    )

    def __init__(self):
        self._compiled_id = None

    def _setup(self, compiled: CompiledModel, state_dict: dict[str, torch.Tensor]):
        _, tasks, buffer_descs, weight_maps, weight_names = load_schedule(
            compiled.schedule_bytes
        )
        self._num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        self._input_buffer_ids = compiled.input_buffer_ids
        self._output_buffer_id = compiled.output_buffer_id
        self._output_shape = compiled.output_shape
        self._num_buffers = compiled.num_buffers
        self._num_tasks = len(tasks)

        self._weight_tensors = {}
        for wm in weight_maps:
            name = weight_names[wm.buffer_index]
            self._weight_tensors[wm.buffer_index] = (
                state_dict[name].contiguous().cuda().half()
            )

        self._workspace = {}
        for bd in buffer_descs:
            num_elements = max(bd.size // 2, 1)
            self._workspace[bd.buffer_id] = torch.zeros(
                num_elements, dtype=torch.float16, device="cuda"
            )

        ptrs = [0] * compiled.num_buffers
        for bd in buffer_descs:
            ptrs[bd.buffer_id] = self._workspace[bd.buffer_id].data_ptr()
        for buf_id, tensor in self._weight_tensors.items():
            ptrs[buf_id] = tensor.data_ptr()

        task_bytes = b"".join(t.to_bytes() for t in tasks)
        self._d_tasks = torch.tensor(
            list(task_bytes), dtype=torch.uint8, device="cuda"
        )
        self._ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
        self._d_dyn_dims = torch.zeros(8, dtype=torch.int32, device="cuda")
        self._compiled_id = id(compiled)

    def run(
        self,
        compiled: CompiledModel,
        state_dict: dict[str, torch.Tensor],
        inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        if self._compiled_id != id(compiled):
            self._setup(compiled, state_dict)

        for i, input_buf_id in enumerate(self._input_buffer_ids):
            input_data = inputs[i].contiguous().cuda().half().flatten()
            self._workspace[input_buf_id][: input_data.numel()].copy_(input_data)

        _launch_cooperative(
            self._d_tasks.data_ptr(),
            self._num_tasks,
            self._ptr_tensor.data_ptr(),
            self._d_dyn_dims.data_ptr(),
            self._num_sms,
        )
        torch.cuda.synchronize()

        numel = 1
        for s in self._output_shape:
            numel *= s
        return self._workspace[self._output_buffer_id][:numel].reshape(
            self._output_shape
        ).clone()


_runner = _CachedRunner()


def execute_model(
    compiled: CompiledModel,
    state_dict: dict[str, torch.Tensor],
    inputs: list[torch.Tensor],
    num_sms: int | None = None,  # noqa: ARG001
) -> torch.Tensor:
    return _runner.run(compiled, state_dict, inputs)
