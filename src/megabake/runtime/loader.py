"""Load a compiled schedule and execute it through the megakernel."""

import torch

from megabake.data_types import OpType
from megabake.schedule_compiler.graph_walker import CompiledModel
from megabake.schedule_compiler.serializer import load_schedule
from megabake.runtime.launcher import _launch_cooperative


class _CachedRunner:
    """Cache GPU-side state so repeated execute_model() calls skip allocation."""

    __slots__ = (
        "_compiled_id", "_d_tasks", "_ptr_tensor", "_d_dyn_dims",
        "_num_tasks", "_num_sms", "_workspace", "_weight_tensors",
        "_input_buffer_ids", "_output_buffer_id", "_output_shape", "_num_buffers",
        "_arena", "_input_holders",
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
            if name.startswith("__folded__."):
                self._weight_tensors[wm.buffer_index] = (
                    compiled.folded_constants[wm.buffer_index]
                )
            else:
                self._weight_tensors[wm.buffer_index] = (
                    state_dict[name].contiguous().cuda().half()
                )

        # Pre-transpose non-transposed weights for skinny matmul (M<=4)
        # so they route through the fast skinny path instead of matmul_scalar
        for task in tasks:
            if (task.op_type == OpType.MATMUL
                    and task.dimensions[0] <= 4
                    and task.strides[0] == 0):
                wb = task.buffer_indices[2]
                if wb in self._weight_tensors:
                    self._weight_tensors[wb] = self._weight_tensors[wb].t().contiguous()
                    task.strides[0] = 1

        # Arena allocation: single cudaMalloc, carve sub-regions
        total_arena = 0
        for bd in buffer_descs:
            total_arena = max(total_arena, bd.offset + bd.size)
        total_arena = max(total_arena, 2)
        self._arena = torch.zeros(total_arena // 2, dtype=torch.float16, device="cuda")

        self._workspace = {}
        for bd in buffer_descs:
            offset_elems = bd.offset // 2
            size_elems = max(bd.size // 2, 1)
            self._workspace[bd.buffer_id] = self._arena[offset_elems:offset_elems + size_elems]

        ptrs = [0] * compiled.num_buffers
        for bd in buffer_descs:
            ptrs[bd.buffer_id] = self._workspace[bd.buffer_id].data_ptr()
        for buf_id, tensor in self._weight_tensors.items():
            ptrs[buf_id] = tensor.data_ptr()

        task_bytes = b"".join(t.to_bytes() for t in tasks)
        self._d_tasks = torch.frombuffer(bytearray(task_bytes), dtype=torch.uint8).cuda()

        self._ptr_tensor = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
        self._d_dyn_dims = torch.zeros(8, dtype=torch.int32, device="cuda")
        self._input_holders = [None] * len(compiled.input_buffer_ids)
        self._compiled_id = id(compiled)

    def run(
        self,
        compiled: CompiledModel,
        state_dict: dict[str, torch.Tensor],
        inputs: list[torch.Tensor],
        task_timings_ptr: int = 0,
    ) -> torch.Tensor:
        if self._compiled_id != id(compiled):
            self._setup(compiled, state_dict)

        for i, input_buf_id in enumerate(self._input_buffer_ids):
            inp = inputs[i]
            if inp.dtype == torch.float16 and inp.is_contiguous() and inp.is_cuda:
                self._ptr_tensor[input_buf_id] = inp.data_ptr()
                self._input_holders[i] = inp
            else:
                inp = inp.contiguous().cuda()
                workspace = self._workspace[input_buf_id]
                if inp.is_floating_point():
                    input_data = inp.half().flatten()
                    workspace[:input_data.numel()].copy_(input_data)
                else:
                    wb = workspace.view(torch.uint8)
                    ib = inp.flatten().view(torch.uint8)
                    wb[:ib.numel()].copy_(ib)
                self._ptr_tensor[input_buf_id] = workspace.data_ptr()
                self._input_holders[i] = None

        _launch_cooperative(
            self._d_tasks.data_ptr(),
            self._num_tasks,
            self._ptr_tensor.data_ptr(),
            self._d_dyn_dims.data_ptr(),
            self._num_sms,
            task_timings_ptr=task_timings_ptr,
        )

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
    task_timings_ptr: int = 0,
) -> torch.Tensor:
    return _runner.run(compiled, state_dict, inputs, task_timings_ptr=task_timings_ptr)
