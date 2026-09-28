import pytest
import torch

from megabake.v3.backends.cuda.bodies import CudaBodyError, emit_cuda_body
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.frontend.semantic import index_program
from tests.test_v3.body_codegen import compile_object, index_module, source_bundle
from tests.test_v3.cpu.test_generic_bodies import AddMM, UnfamiliarMapReduce
from tests.test_v3.fixtures import (
    capture_cache_append_program,
    make_cache_append_graph,
)


@pytest.mark.v3_toolchain
def test_v3r013_014_generic_map_reduce_view_and_addmm_sources_compile(v3_toolchain, tmp_path):
    x = torch.randn(2, 5)
    weight = torch.randn(5)
    _, indexed_map = index_module(UnfamiliarMapReduce(), x, weight)
    x_linear = torch.randn(1, 33)
    weight_nk = torch.randn(17, 33)
    bias = torch.randn(17)
    _, indexed_mm = index_module(AddMM(), x_linear, weight_nk, bias)
    map_source, _ = source_bundle(indexed_map)
    mm_source, _ = source_bundle(indexed_mm)

    command = compile_object(v3_toolchain, map_source + "\n" + mm_source, tmp_path / "v3_generic")
    assert command[command.index("-arch=sm_90")] == "-arch=sm_90"


@pytest.mark.v3_toolchain
def test_v3r015_bounded_functional_cache_bodies_compile_and_unknown_bounds_reject(v3_toolchain, tmp_path):
    cache = torch.full((1, 2, 17, 8), -17.0)
    index = torch.tensor([0], dtype=torch.int64)
    update = torch.ones((1, 2, 1, 8))
    program, _ = capture_cache_append_program(cache, index, update)
    indexed = index_program(program)
    source, _ = source_bundle(indexed)
    command = compile_object(v3_toolchain, source, tmp_path / "v3_cache_append")
    assert command[command.index("-arch=sm_90")] == "-arch=sm_90"

    unbounded = capture_graph_module(
        make_cache_append_graph(), (cache, index, update),
        input_spec={"structure": "(cache,index,update)"},
        output_spec={"cache": "tensor", "current": "tensor"}, state_bindings={"cache": "kv"},
    )
    unsupported = index_program(unbounded)
    writer = next(operation for operation in unsupported.operations if operation.kind == "Scatter/StateWrite")
    with pytest.raises(CudaBodyError, match="bounds contract"):
        emit_cuda_body(unsupported, writer)
