"""Tiny FX and nvcc helpers for testing generated V3 device bodies."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any

import torch

from megabake.v3.backends.cuda.bodies import CudaBody, emit_cuda_body
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program


def index_module(module: torch.nn.Module, *examples: torch.Tensor):
    exported = torch.export.export(module, examples)
    program = normalize_fx(exported, input_spec={})
    return program, index_program(program)


def trace_nodes(program: Any, *examples: torch.Tensor) -> dict[str, Any]:
    traced: dict[str, Any] = {}

    class Recorder(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            traced[node.name] = value
            return value

    Recorder(program.graph_module).run(*examples)
    return traced


def source_bundle(indexed: Any) -> tuple[str, dict[str, CudaBody]]:
    sources: list[str] = []
    bodies: dict[str, CudaBody] = {}
    for operation in indexed.operations:
        body = emit_cuda_body(indexed, operation)
        bodies[operation.local_reference.node_name] = body
        sources.append(body.source)
        if not body.entry_point or body.body_kind == "guard":
            continue
        suffix = body.entry_point
        if body.body_kind == "view_map":
            output = next(value for value in indexed.values if value.value_id == operation.outputs[0])
            shape = tuple(output.shape)
            coordinates = [f"int64_t i{axis} = 0;" for axis in range(len(shape))]
            decode = ["int64_t remaining = linear;"]
            for axis in range(len(shape) - 1, -1, -1):
                decode.extend((f"i{axis} = remaining % {shape[axis]};",
                               f"remaining /= {shape[axis]};"))
            call = ("linear" if operation.input_index_maps[0].mode == "reshape"
                    else ", ".join(f"i{axis}" for axis in range(len(shape))))
            if not call:
                call = ""
            source = f"""
__global__ void test_{suffix}(int64_t* out) {{
  for (int64_t linear = threadIdx.x; linear < {body.output_elements}; linear += blockDim.x) {{
    {''.join(coordinates)}
    {' '.join(decode) if operation.input_index_maps[0].mode != 'reshape' else ''}
    out[linear] = {body.entry_point}({call});
  }}
}}
extern "C" int launch_{suffix}(void* raw_out) {{
  test_{suffix}<<<1, 128>>>(static_cast<int64_t*>(raw_out));
  return static_cast<int>(cudaDeviceSynchronize());
}}
"""
            sources.append(source)
            continue
        typed_inputs = [f"const {dtype}* in{index}"
                        for index, dtype in enumerate(body.input_types)]
        typed_args = ", ".join(typed_inputs + [f"{body.output_type}* out"])
        call_args = ", ".join([f"in{index}" for index in range(len(body.input_types))] +
                               ["out", "0", str(body.output_elements)])
        raw_args = [f"const void* raw{index}" for index in range(len(body.input_types))]
        casts = [f"static_cast<const {dtype}*>(raw{index})"
                 for index, dtype in enumerate(body.input_types)]
        casts.append(f"static_cast<{body.output_type}*>(raw_out)")
        sources.append(f"""
__global__ void test_{suffix}({typed_args}) {{
  {body.entry_point}({call_args});
}}
extern "C" int launch_{suffix}({', '.join(raw_args + ['void* raw_out'])}) {{
  test_{suffix}<<<1, 128>>>({', '.join(casts)});
  return static_cast<int>(cudaDeviceSynchronize());
}}
""")
    return "\n".join(sources), bodies


def compile_library(nvcc: str, source: str, output: Path) -> tuple[ctypes.CDLL, list[str]]:
    cu = output.with_suffix(".cu")
    library = output.with_suffix(".so")
    cu.write_text(source)
    command = [nvcc, "-std=c++17", "-arch=sm_90", "-shared", "-Xcompiler", "-fPIC",
               str(cu), "-o", str(library)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise AssertionError(
            f"nvcc failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return ctypes.CDLL(str(library)), command


def compile_object(nvcc: str, source: str, output: Path) -> list[str]:
    cu = output.with_suffix(".cu")
    obj = output.with_suffix(".o")
    cu.write_text(source)
    command = [nvcc, "-std=c++17", "-arch=sm_90", "-c", str(cu), "-o", str(obj)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise AssertionError(
            f"nvcc failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return command


def run_indexed_gpu(indexed: Any, program: Any, examples: tuple[torch.Tensor, ...],
                    library: ctypes.CDLL, bodies: dict[str, CudaBody]) -> Any:
    traced = trace_nodes(program, *examples)
    nodes = {node.name: node for node in program.graph_module.graph.nodes}
    runtime = {value.value_id: traced[value.fx_node]
               for value in indexed.values
               if nodes[value.fx_node].op in {"placeholder", "get_attr"}}
    for operation in indexed.operations:
        name = operation.local_reference.node_name
        body = bodies[name]
        expected = traced[name]
        if body.body_kind == "view_map":
            output = traced[name]
            offsets = torch.empty((body.output_elements,), device=output.device, dtype=torch.int64)
            launch = getattr(library, f"launch_{body.entry_point}")
            launch.argtypes = [ctypes.c_void_p]
            launch.restype = ctypes.c_int
            code = launch(ctypes.c_void_p(offsets.data_ptr()))
            assert code == 0, f"{name} CUDA view-map launch returned {code}"
            expected_offsets = []
            source_reference = traced[next(value.fx_node for value in indexed.values
                                           if value.value_id == operation.inputs[0])]
            base_offset = output.storage_offset() - source_reference.storage_offset()
            for linear in range(body.output_elements):
                remaining = linear
                coords = [0] * output.ndim
                for axis in range(output.ndim - 1, -1, -1):
                    coords[axis] = remaining % output.shape[axis]
                    remaining //= output.shape[axis]
                expected_offsets.append(base_offset + sum(i * stride for i, stride in zip(coords, output.stride())))
            assert torch.equal(offsets.cpu(), torch.tensor(expected_offsets, dtype=torch.int64))
            runtime[operation.outputs[0]] = output
            continue
        if body.body_kind == "guard":
            if operation.outputs:
                runtime[operation.outputs[0]] = expected
            continue
        if not body.entry_point:
            continue
        arguments = [runtime[value_id] for value_id in operation.inputs]
        output = torch.empty_like(expected, memory_format=torch.preserve_format)
        launch = getattr(library, f"launch_{body.entry_point}")
        launch.argtypes = [ctypes.c_void_p] * (len(arguments) + 1)
        launch.restype = ctypes.c_int
        code = launch(*(ctypes.c_void_p(value.data_ptr()) for value in arguments),
                      ctypes.c_void_p(output.data_ptr()))
        assert code == 0, f"{name} CUDA launch returned {code}"
        torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
        runtime[operation.outputs[0]] = output
    leaves = [runtime[leaf.value_id] if leaf.value_id is not None else leaf.literal
              for leaf in indexed.outputs]
    if indexed.source_program.output_tree_spec is not None:
        import torch.utils._pytree as pytree
        return pytree.tree_unflatten(leaves, indexed.source_program.output_tree_spec)
    return leaves[0] if len(leaves) == 1 else tuple(leaves)
