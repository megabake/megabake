from pathlib import Path

import pytest
import torch

from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.session import CachedStepSessionError, open_cached_step_session
from megabake.v3.backends.cuda.worker import (
    bind_worker_values,
    compile_worker_entry,
    inspect_worker_entry,
    launch_worker_entry,
    lower_worker_program,
    open_worker_session,
)
from tests.test_v3.worker_fixtures import capture_reduce_program, capture_unfamiliar_block


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def worker_entries(tmp_path_factory):
    profile = CudaTargetProfile.from_json(
        (ROOT / "ART/tasks/V3R-021/target_profile.json").read_text()
    )
    root = tmp_path_factory.mktemp("v3-worker-entries")
    result = {}
    for name, capture in (
        ("reduce", lambda: capture_reduce_program(device="cuda")),
        ("block", lambda: capture_unfamiliar_block(device="cuda")),
    ):
        captured = capture()
        if name == "reduce":
            reference, indexed, plan, examples = captured
        else:
            reference, indexed, _, plan, examples = captured
        worker = lower_worker_program(indexed, plan, profile, block_threads=32)
        compiled = compile_worker_entry(
            worker, profile, root / f"{name}.so", nvcc="/usr/local/cuda/bin/nvcc"
        )
        assert compiled.return_code == 0, compiled.stderr
        admission = inspect_worker_entry(compiled.artifact_path, worker, profile, compiled)
        assert admission["launch_contract"]["admitted"], admission["launch_contract"]
        session = open_worker_session(compiled.artifact_path, worker, profile, admission)
        result[name] = (profile, reference, indexed, plan, worker, examples, compiled, admission, session)
    return result


def _output_map(indexed, values):
    by_id = {value.value_id: bound for value, bound in zip(indexed.values, values)}
    return {leaf.path: by_id[leaf.value_id] if leaf.value_id else leaf.literal
            for leaf in indexed.outputs}


@pytest.mark.v3_gpu
def test_v3r025_runs_repeated_multiphase_worker_and_traces_one_grid(worker_entries, tmp_path):
    profile, reference, indexed, plan, worker, examples, compiled, admission, session = worker_entries["reduce"]
    assert worker.grid_ctas == 5
    assert admission["compiled_function"]["cooperative_resident_ctas"] >= worker.grid_ctas
    assert admission["progress_proof"]["idle_workers_join"]

    values = bind_worker_values(indexed, examples)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as trace:
        launch = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                     admission=admission, session=session)
        assert launch["launched"], launch
        torch.cuda.synchronize()
    kernel_events = [event for event in trace.events() if event.name == "megabake_v3_entry"]
    assert len(kernel_events) == 1
    trace.export_chrome_trace(str(tmp_path / "worker_trace.json"))

    expected = reference.run_reference(*examples)
    for _ in range(4):
        values = bind_worker_values(indexed, examples)
        launch = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                     admission=admission, session=session)
        assert launch["launched"], launch
        torch.cuda.synchronize()
        output_id = indexed.outputs[0].value_id
        actual = {value.value_id: bound for value, bound in zip(indexed.values, values)}[output_id]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    resident = admission["compiled_function"]["cooperative_resident_ctas"]
    rejected = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                   admission=admission, grid_ctas=resident + 1)
    assert not rejected["launched"] and rejected["return_code"] == -2


@pytest.mark.v3_gpu
def test_v3r026_runs_unfamiliar_block_and_rejects_cast_or_state_guard_mismatch(worker_entries, tmp_path):
    profile, reference, indexed, plan, worker, examples, compiled, admission, session = worker_entries["block"]
    old_cache = examples[4].clone()
    values = bind_worker_values(indexed, examples)
    launched = launch_worker_entry(compiled.artifact_path, worker, profile, values,
                                   admission=admission, session=session)
    assert launched["launched"], launched
    torch.cuda.synchronize()

    expected = reference.run_reference(*examples)
    actual = _output_map(indexed, values)
    torch.testing.assert_close(actual[("current",)], expected["current"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual[("cache",)], expected["cache"], rtol=0, atol=0)
    assert torch.equal(examples[4], old_cache)
    assert actual[("cache",)].data_ptr() != examples[4].data_ptr()

    step_session = open_cached_step_session(
        compiled.artifact_path, reference, indexed, plan, worker, profile, admission
    )
    bad_session_args = list(examples)
    bad_session_args[0] = torch.empty((1, 2), dtype=torch.float16, device=examples[0].device)
    with pytest.raises(CachedStepSessionError, match="shape guard"):
        step_session.run(*bad_session_args)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as trace:
        first_outputs, first_state, first_record = step_session.run(*examples)
        torch.cuda.synchronize()
    entry_events = [event for event in trace.events() if event.name == "megabake_v3_entry"]
    assert len(entry_events) == 1
    trace.export_chrome_trace(str(tmp_path / "cached_step_session_trace.json"))
    expected_first = reference.run_reference(*examples)
    torch.testing.assert_close(first_outputs["current"], expected_first["current"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(first_outputs["cache"], expected_first["cache"], rtol=0, atol=0)
    assert first_state["kv"] is first_outputs["cache"]
    assert first_record.launched and first_record.return_code == 0
    assert first_record.grid_ctas == worker.grid_ctas
    assert first_outputs["cache"].data_ptr() != old_cache.data_ptr()
    assert torch.equal(examples[4], old_cache)

    next_examples = list(examples)
    next_examples[4] = first_state["kv"]
    next_examples[5] = torch.tensor([2], dtype=torch.int64, device=examples[5].device)
    second_outputs, second_state, second_record = step_session.run(*next_examples)
    torch.cuda.synchronize()
    expected_second = reference.run_reference(*next_examples)
    torch.testing.assert_close(second_outputs["current"], expected_second["current"], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(second_outputs["cache"], expected_second["cache"], rtol=0, atol=0)
    assert second_record.launched and second_record.return_code == 0
    assert second_outputs["cache"].data_ptr() != first_outputs["cache"].data_ptr()
    assert second_state["kv"] is second_outputs["cache"]

    _, cast_indexed, _, cast_plan, cast_examples = capture_unfamiliar_block(
        device="cuda", cast_dtype=torch.float16
    )
    cast_worker = lower_worker_program(cast_indexed, cast_plan, profile, block_threads=32)
    cast_values = bind_worker_values(cast_indexed, cast_examples)
    wrong_artifact = launch_worker_entry(compiled.artifact_path, cast_worker, profile,
                                         cast_values, admission=admission)
    assert not wrong_artifact["launched"] and wrong_artifact["return_code"] == -6

    bad_args = list(examples)
    bad_args[5] = torch.tensor([4], dtype=torch.int64, device="cuda")
    bad_values = bind_worker_values(indexed, bad_args)
    bad_index = launch_worker_entry(compiled.artifact_path, worker, profile, bad_values,
                                    admission=admission, session=session)
    assert not bad_index["launched"] and bad_index["return_code"] == -5
