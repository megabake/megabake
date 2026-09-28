import pytest
import torch

from tests.test_v3.body_codegen import compile_object, index_module, source_bundle
from tests.test_v3.cpu.test_generic_bodies import AddMM, UnfamiliarMapReduce


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
