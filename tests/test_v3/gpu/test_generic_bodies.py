import torch
import pytest

from tests.test_v3.body_codegen import compile_library, index_module, run_indexed_gpu, source_bundle
from tests.test_v3.cpu.test_generic_bodies import AddMM, CatStack, Gate, MMKn, NormCastOrder, SliceReshapeExpand
from tests.test_v3.fixtures import (
    GATE_TINY,
    LINEAR_TINY,
    NORM_VARIANTS,
    STATE_POISON,
    capture_cache_append_program,
)


@pytest.mark.v3_gpu
def test_v3r013_generated_map_reduce_and_cast_bodies_match_norm_fixture(v3_device, v3_toolchain, tmp_path):
    case = NORM_VARIANTS()
    examples = (case.inputs["x"].to(v3_device), case.inputs["weight"].to(v3_device))
    program, indexed = index_module(NormCastOrder(), *examples)
    assert indexed.strict_supported
    source, bodies = source_bundle(indexed)
    library, _ = compile_library(v3_toolchain, source, tmp_path / "v3_norm")
    result = run_indexed_gpu(indexed, program, examples, library, bodies)
    torch.testing.assert_close(result.cpu(), case.expected["cast_before_weight"], rtol=2e-4, atol=2e-4)


@pytest.mark.v3_gpu
def test_v3r014_generated_addmm_and_transposed_mm_match_linear_tiny(v3_device, v3_toolchain, tmp_path):
    case = LINEAR_TINY()
    x = case.inputs["x"].to(v3_device)
    weight_nk = case.inputs["weight_nk"].to(v3_device)
    weight_kn = case.inputs["weight_kn"].to(v3_device)
    bias = case.inputs["bias"].to(v3_device)
    addmm_program, addmm = index_module(AddMM(), x, weight_nk, bias)
    mm_program, mm = index_module(MMKn(), x, weight_kn)
    addmm_source, addmm_bodies = source_bundle(addmm)
    mm_source, mm_bodies = source_bundle(mm)
    library, _ = compile_library(v3_toolchain, addmm_source + "\n" + mm_source, tmp_path / "v3_linear")

    addmm_result = run_indexed_gpu(addmm, addmm_program, (x, weight_nk, bias), library, addmm_bodies)
    mm_result = run_indexed_gpu(mm, mm_program, (x, weight_kn), library, mm_bodies)
    torch.testing.assert_close(addmm_result, torch.addmm(bias, x, weight_nk.t(), beta=-0.25, alpha=1.75),
                               rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(mm_result, x @ weight_kn, rtol=2e-4, atol=2e-4)


@pytest.mark.v3_gpu
def test_v3r015_functional_kv_append_and_read_match_poisoned_cache_positions(v3_device, v3_toolchain, tmp_path):
    case = STATE_POISON(position=0)
    cache = case.state_before["cache_k"].to(v3_device)
    index = torch.tensor([0], device=v3_device, dtype=torch.int64)
    update = case.inputs["k"].to(v3_device)
    program, _ = capture_cache_append_program(cache, index, update)
    from megabake.v3.frontend.semantic import index_program
    indexed = index_program(program)
    assert indexed.strict_supported
    source, bodies = source_bundle(indexed)
    library, _ = compile_library(v3_toolchain, source, tmp_path / "v3_cache_append")

    for position in (0, 1, 15, 16):
        case = STATE_POISON(position=position)
        old = case.state_before["cache_k"].to(v3_device)
        token = case.inputs["k"].to(v3_device)
        index = torch.tensor([position], device=v3_device, dtype=torch.int64)
        actual = run_indexed_gpu(indexed, program, (old, index, token), library, bodies)
        torch.testing.assert_close(actual["cache"].cpu(), case.expected["cache_k"], rtol=0, atol=0)
        torch.testing.assert_close(actual["current"].cpu(), token.cpu(), rtol=0, atol=0)
        assert torch.equal(actual["cache"][:, :, :position].cpu(), old[:, :, :position].cpu())
        assert torch.equal(actual["cache"][:, :, position + 1:].cpu(), old[:, :, position + 1:].cpu())


@pytest.mark.v3_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v3r013_half_and_bfloat_maps_match_gate_tiny(v3_device, v3_toolchain, tmp_path, dtype):
    case = GATE_TINY()
    gate = case.inputs["gate"].to(device=v3_device, dtype=dtype)
    up = case.inputs["up"].to(device=v3_device, dtype=dtype)
    program, indexed = index_module(Gate(), gate, up)
    source, bodies = source_bundle(indexed)
    library, _ = compile_library(v3_toolchain, source, tmp_path / f"v3_gate_{dtype}")
    result = run_indexed_gpu(indexed, program, (gate, up), library, bodies)
    torch.testing.assert_close(result, torch.nn.functional.silu(gate) * up, rtol=2e-3, atol=2e-3)


@pytest.mark.v3_gpu
def test_v3r013_generated_slice_reshape_expand_maps_match_gpu_views(v3_device, v3_toolchain, tmp_path):
    x = torch.arange(16, device=v3_device, dtype=torch.float32).reshape(4, 4)
    program, indexed = index_module(SliceReshapeExpand(), x)
    source, bodies = source_bundle(indexed)
    library, _ = compile_library(v3_toolchain, source, tmp_path / "v3_views")
    actual = run_indexed_gpu(indexed, program, (x,), library, bodies)
    expected = SliceReshapeExpand()(x)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.v3_gpu
def test_v3r028_cat_stack_bodies_copy_empty_identity_and_static_segments(v3_device, v3_toolchain, tmp_path):
    examples = (
        torch.empty((0,), device=v3_device, dtype=torch.float16),
        torch.arange(12, device=v3_device, dtype=torch.float16).reshape(1, 3, 2, 2),
        torch.arange(12, device=v3_device, dtype=torch.float16).reshape(1, 3, 2, 2) + 10,
        torch.arange(12, device=v3_device, dtype=torch.float16).reshape(1, 3, 1, 4) + 20,
    )
    program, indexed = index_module(CatStack(), *examples)
    assert indexed.strict_supported
    source, bodies = source_bundle(indexed)
    library, _ = compile_library(v3_toolchain, source, tmp_path / "v3_cat_stack")

    actual = run_indexed_gpu(indexed, program, examples, library, bodies)
    expected = CatStack()(*examples)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
