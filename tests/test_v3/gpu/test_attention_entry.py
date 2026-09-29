from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import (
    bind_worker_values, compile_worker_entry, inspect_worker_entry,
    launch_worker_entry, lower_worker_program, open_worker_session,
)
from tests.test_v3.attention_fixtures import (
    attention_inputs, capture_attention, oracle_inputs, rotary_attention_inputs,
)
from tests.test_v3.fixtures import ATTENTION_TINY, STATE_POISON


ROOT = Path(__file__).resolve().parents[3]


def _compiled_entry(tmp_path, *, capacity, dtype, heads_kv=2, mask_heads=1, scale=None,
                    rotary=False, depth=8):
    profile = CudaTargetProfile.from_json(
        (ROOT / "ART/tasks/V3R-021/target_profile.json").read_text())
    examples = (rotary_attention_inputs(0, dtype=dtype, device="cuda") if rotary else
                attention_inputs(0, capacity=capacity, dtype=dtype, device="cuda",
                                 heads_kv=heads_kv, mask_heads=mask_heads, depth=depth))
    reference, indexed, plan = capture_attention(examples, scale=scale, rotary=rotary)
    assert indexed.strict_supported
    worker = lower_worker_program(indexed, plan, profile, block_threads=32)
    compiled = compile_worker_entry(worker, profile, tmp_path / "attention.so",
                                    nvcc="/usr/local/cuda/bin/nvcc")
    assert compiled.return_code == 0, compiled.stderr
    admission = inspect_worker_entry(compiled.artifact_path, worker, profile, compiled)
    assert admission["launch_contract"]["admitted"], admission["launch_contract"]
    session = open_worker_session(compiled.artifact_path, worker, profile, admission)
    return profile, reference, indexed, worker, compiled, admission, session


def _run(entry, inputs):
    profile, reference, indexed, worker, compiled, admission, session = entry
    old_k, old_v = inputs[3].clone(), inputs[4].clone()
    values = bind_worker_values(indexed, inputs)
    launch = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                 admission=admission, session=session)
    assert launch["launched"], launch
    torch.cuda.synchronize()
    outputs = {item.fx_node: value for item, value in zip(indexed.values, values)}
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.MATH]):
        expected = reference.run_reference(*inputs)
    tolerance = (1e-5, 1e-5) if inputs[0].dtype == torch.float32 else (
        (0.07, 0.01) if inputs[0].dtype == torch.bfloat16 else (4e-3, 4e-3))
    torch.testing.assert_close(outputs["attention"], expected["output"],
                               atol=tolerance[0], rtol=tolerance[1])
    for name, before in (("k", old_k), ("v", old_v)):
        actual, wanted = outputs[f"write_{name}"], expected[f"cache_{name}"]
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        torch.testing.assert_close(inputs[3 if name == "k" else 4], before, rtol=0, atol=0)
        assert actual.data_ptr() != before.data_ptr()
    return outputs


@pytest.mark.v3_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_v3r027_one_grid_cached_attention_boundaries_masks_and_advancing(tmp_path, dtype):
    entry = _compiled_entry(tmp_path, capacity=17, dtype=dtype)
    profile, _, indexed, worker, compiled, admission, session = entry
    assert len(worker.stages) == 3
    assert [stage.body_kind for stage in worker.stages][-1] == "online_cached_attention_warp32"
    for position in (0, 1, 15, 16):
        for mask_kind in ("causal", "window", "empty"):
            inputs = attention_inputs(position, dtype=dtype, device="cuda", mask_kind=mask_kind)
            outputs = _run(entry, inputs)
            if mask_kind == "empty":
                assert torch.count_nonzero(outputs["attention"]) == 0

    first = attention_inputs(0, dtype=dtype, device="cuda")
    for position in (0, 1):
        inputs = list(attention_inputs(position, dtype=dtype, device="cuda"))
        inputs[3:5] = first[3:5]
        outputs = _run(entry, tuple(inputs))
        first = (*inputs[:3], outputs["write_k"], outputs["write_v"], *inputs[5:])

    bad = list(attention_inputs(0, dtype=dtype, device="cuda"))
    bad[5] = torch.tensor([17], device="cuda", dtype=torch.int64)
    values = bind_worker_values(indexed, bad)
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -5

    wrong_mask = list(attention_inputs(0, dtype=dtype, device="cuda"))
    wrong_mask[6] = wrong_mask[6].float()
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile,
                                   bind_worker_values(indexed, wrong_mask),
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -10

    future_mask = list(attention_inputs(0, dtype=dtype, device="cuda"))
    future_mask[6][..., 16] = True
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile,
                                   bind_worker_values(indexed, future_mask),
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -11
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile,
                                   bind_worker_values(indexed, future_mask),
                                   admission=admission)
    assert not rejected["launched"] and rejected["return_code"] == -11

    aliased_output = list(bind_worker_values(indexed, attention_inputs(0, dtype=dtype, device="cuda")))
    q_id = next(item.value_id for item in indexed.values if item.fx_node == "q")
    attention_value = next(item for item in indexed.values if item.fx_node == "attention")
    q = aliased_output[worker.value_ids.index(q_id)]
    aliased_output[worker.value_ids.index(attention_value.value_id)] = torch.as_strided(
        q, attention_value.shape, attention_value.strides)
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile, aliased_output,
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -10

    aliased = list(bind_worker_values(indexed, attention_inputs(0, dtype=dtype, device="cuda")))
    old_id = next(item.value_id for item in indexed.values if item.fx_node == "cache_k")
    new_id = next(item.value_id for item in indexed.values if item.fx_node == "write_k")
    aliased[worker.value_ids.index(new_id)] = aliased[worker.value_ids.index(old_id)]
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile, aliased,
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -10

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as trace:
        _run(entry, attention_inputs(15, dtype=dtype, device="cuda"))
    assert sum(event.name == "megabake_v3_entry" for event in trace.events()) == 1


@pytest.mark.v3_gpu
def test_v3r027_long_context_grid_stride_and_fp16(tmp_path):
    entry = _compiled_entry(tmp_path, capacity=2049, dtype=torch.float16)
    profile, _, _, worker, _, admission, _ = entry
    assert worker.grid_ctas <= profile.device_facts["visible_sms"].value
    assert admission["compiled_function"]["cooperative_resident_ctas"] >= worker.grid_ctas
    for position in (128, 2048):
        _run(entry, attention_inputs(position, capacity=2049,
                                     dtype=torch.float16, device="cuda"))
    for mask_kind in ("window", "empty"):
        _run(entry, attention_inputs(2048, capacity=2049, dtype=torch.float16,
                                     device="cuda", mask_kind=mask_kind))


@pytest.mark.v3_gpu
@pytest.mark.parametrize("scale", [0.125, 0.0, -0.5])
def test_v3r027_named_oracles_and_mqa_scale(tmp_path, scale):
    entry = _compiled_entry(tmp_path, capacity=17, dtype=torch.float32)
    for oracle in (ATTENTION_TINY, STATE_POISON):
        for position in (0, 1, 15, 16):
            case = oracle(position=position)
            result = _run(entry, oracle_inputs(case, device="cuda"))
            torch.testing.assert_close(result["attention"], case.expected["output"].cuda(),
                                       rtol=1e-5, atol=1e-5)
            for name in ("k", "v"):
                torch.testing.assert_close(result[f"write_{name}"],
                                           case.expected[f"cache_{name}"].cuda(), rtol=0, atol=0)

    mqa = _compiled_entry(tmp_path / "mqa", capacity=17, dtype=torch.float32,
                          heads_kv=1, mask_heads=4, scale=scale)
    inputs = list(attention_inputs(1, dtype=torch.float32, device="cuda",
                                   heads_kv=1, mask_heads=4))
    inputs[6][0, 2] = False
    result = _run(mqa, tuple(inputs))
    assert torch.count_nonzero(result["attention"][0, 2]) == 0


@pytest.mark.v3_gpu
def test_v3r027_rope_producer_is_in_the_owned_grid_and_bounds_are_guarded(tmp_path):
    entry = _compiled_entry(tmp_path, capacity=17, dtype=torch.float32, rotary=True)
    profile, _, indexed, worker, compiled, admission, session = entry
    assert [stage.operation_id for stage in worker.stages].index("iop:rotated_k") < [
        stage.operation_id for stage in worker.stages].index("iop:write_k")
    inputs = rotary_attention_inputs(1, dtype=torch.float32, device="cuda")
    _run(entry, inputs)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as trace:
        _run(entry, inputs)
    assert sum(event.name == "megabake_v3_entry" for event in trace.events()) == 1
    bad = list(inputs)
    bad[7] = bad[7].clone()
    bad[7][0] = 8
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile,
                                   bind_worker_values(indexed, bad),
                                   admission=admission, session=session)
    assert not rejected["launched"] and rejected["return_code"] == -5


@pytest.mark.v3_gpu
@pytest.mark.parametrize("depth", [16, 32])
def test_v3r027_warp_attention_other_head_depths(tmp_path, depth):
    entry = _compiled_entry(tmp_path, capacity=17, dtype=torch.float16, depth=depth)
    _run(entry, attention_inputs(16, dtype=torch.float16, device="cuda", depth=depth))


@pytest.mark.v3_gpu
def test_v3r027_default_launch_follows_active_cuda_stream(tmp_path):
    entry = _compiled_entry(tmp_path, capacity=17, dtype=torch.float32)
    profile, reference, indexed, worker, compiled, admission, session = entry
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = attention_inputs(1, dtype=torch.float32, device="cuda")
        values = bind_worker_values(indexed, inputs)
        launched = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                       admission=admission, session=session)
        assert launched["launched"]
    stream.synchronize()
    outputs = {value.fx_node: tensor for value, tensor in zip(indexed.values, values)}
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.MATH]):
        expected = reference.run_reference(*inputs)
    torch.testing.assert_close(outputs["attention"], expected["output"], atol=1e-5, rtol=1e-5)
    for name in ("k", "v"):
        torch.testing.assert_close(outputs[f"write_{name}"], expected[f"cache_{name}"], atol=0, rtol=0)
