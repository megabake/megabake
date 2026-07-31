"""Launch the megakernel via CUDA driver API cooperative launch."""

import ctypes
import ctypes.util

import torch

from megabake.data_types import TaskDesc
from megabake.runtime import get_sm_version
from megabake.runtime.cuda_compiler import get_megakernel_cubin


_cuda_driver = None
_module = None
_kernel = None


def _init_driver():
    global _cuda_driver
    if _cuda_driver is not None:
        return

    libcuda = ctypes.util.find_library("cuda")
    if libcuda is None:
        libcuda = "libcuda.so.1"
    _cuda_driver = ctypes.CDLL(libcuda)
    err = _cuda_driver.cuInit(0)
    if err != 0:
        raise RuntimeError(f"cuInit failed with error {err}")


def _load_module():
    global _module, _kernel
    if _module is not None:
        return

    _init_driver()
    cubin_path = get_megakernel_cubin()

    _module = ctypes.c_void_p()
    err = _cuda_driver.cuModuleLoad(ctypes.byref(_module), cubin_path.encode())
    if err != 0:
        raise RuntimeError(f"cuModuleLoad failed with error {err}")

    _kernel = ctypes.c_void_p()
    err = _cuda_driver.cuModuleGetFunction(
        ctypes.byref(_kernel), _module, b"megakernel"
    )
    if err != 0:
        raise RuntimeError(f"cuModuleGetFunction failed with error {err}")

    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
    smem_limit = 225280 if get_sm_version() >= 90 else 102400
    err = _cuda_driver.cuFuncSetAttribute(
        _kernel,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        smem_limit,
    )
    if err != 0:
        raise RuntimeError(f"cuFuncSetAttribute failed with error {err}")


def run_single_task(
    task: TaskDesc,
    input_tensors: list[torch.Tensor],
    output_shape: list[int],
    output_dtype: torch.dtype,
    num_sms: int | None = None,
) -> torch.Tensor:
    """
    Launch the megakernel with a single task for unit testing.
    """
    output = torch.empty(output_shape, dtype=output_dtype, device="cuda")

    all_tensors = [output] + input_tensors
    num_buffers = len(all_tensors)

    ptrs = [t.data_ptr() for t in all_tensors]
    while len(ptrs) < 8:
        ptrs.append(0)

    task.buffer_indices = list(range(num_buffers)) + [0xFFFFFFFF] * (8 - num_buffers)

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if task.num_tiles == 0:
        task.num_tiles = num_sms

    d_tasks = torch.tensor(list(task.to_bytes()), dtype=torch.uint8, device="cuda")
    ptr_array = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
    d_dyn_dims = torch.zeros(8, dtype=torch.int32, device="cuda")

    _launch_cooperative(
        d_tasks.data_ptr(),
        1,
        ptr_array.data_ptr(),
        d_dyn_dims.data_ptr(),
        num_sms,
    )

    torch.cuda.synchronize()
    return output


def run_tasks(
    tasks: list[TaskDesc],
    buffer_tensors: list[torch.Tensor],
    num_sms: int | None = None,
) -> None:
    """Launch the megakernel with multiple tasks."""
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    all_task_bytes = b"".join(t.to_bytes() for t in tasks)
    d_tasks = torch.tensor(list(all_task_bytes), dtype=torch.uint8, device="cuda")

    ptrs = [t.data_ptr() for t in buffer_tensors]
    ptr_array = torch.tensor(ptrs, dtype=torch.int64, device="cuda")

    d_dyn_dims = torch.zeros(8, dtype=torch.int32, device="cuda")

    _launch_cooperative(
        d_tasks.data_ptr(),
        len(tasks),
        ptr_array.data_ptr(),
        d_dyn_dims.data_ptr(),
        num_sms,
    )
    torch.cuda.synchronize()


def _launch_cooperative(
    d_tasks_ptr: int,
    num_tasks: int,
    d_buffers_ptr: int,
    d_dyn_dims_ptr: int,
    num_sms: int,
    task_timings_ptr: int = 0,
):
    """Launch megakernel via cuLaunchCooperativeKernel."""
    _load_module()

    smem_bytes = 225280 if get_sm_version() >= 90 else 102400

    arg_tasks = ctypes.c_void_p(d_tasks_ptr)
    arg_num_tasks = ctypes.c_int(num_tasks)
    arg_buffers = ctypes.c_void_p(d_buffers_ptr)
    arg_dyn_dims = ctypes.c_void_p(d_dyn_dims_ptr)
    arg_timings = ctypes.c_void_p(task_timings_ptr)

    args = (ctypes.c_void_p * 5)(
        ctypes.cast(ctypes.pointer(arg_tasks), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_num_tasks), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_buffers), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_dyn_dims), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(arg_timings), ctypes.c_void_p),
    )

    stream = torch.cuda.current_stream().cuda_stream

    err = _cuda_driver.cuLaunchCooperativeKernel(
        _kernel,
        num_sms, 1, 1,   # grid
        256, 1, 1,        # block
        smem_bytes,       # dynamic shared memory (48KB default)
        ctypes.c_void_p(stream),
        args,
    )
    if err != 0:
        raise RuntimeError(f"cuLaunchCooperativeKernel failed with error {err}")
