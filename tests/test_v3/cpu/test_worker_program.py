from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import megabake.v3.backends.cuda.worker as worker_module
from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import (
    WorkerProgramError, _binding_error, _prelaunch_index_source, _runtime_guard_error,
    bind_worker_values, lower_worker_program,
)
from tests.test_v3.worker_fixtures import capture_reduce_program, capture_unfamiliar_block


ROOT = Path(__file__).resolve().parents[3]


def _profile():
    return CudaTargetProfile.from_json(
        (ROOT / "ART/tasks/V3R-021/target_profile.json").read_text()
    )


def test_v3r025_lowers_ordered_static_workers_with_idle_cta_joins():
    _, indexed, plan, _ = capture_reduce_program()
    worker = lower_worker_program(indexed, plan, _profile(), block_threads=32)

    assert indexed.strict_supported
    assert [stage.body_kind for stage in worker.stages] == ["map", "map", "reduction"]
    assert worker.grid_ctas == 5  # 130 map elements split over 32-thread workers.
    assert worker.to_dict()["progress_proof"]["idle_workers_join"] is True
    assert worker.source.count("grid.sync();") == len(worker.stages) - 1
    assert "blockIdx.x" in worker.source and "cudaLaunchDevice" not in worker.source
    assert worker.target == "sm_90"


def test_v3r028_computes_the_choice_inventory_once_per_worker_lowering(monkeypatch):
    _, indexed, plan, _ = capture_reduce_program()
    original = worker_module.enumerate_algorithm_choices
    calls = []

    def counted(program):
        calls.append(program)
        return original(program)

    monkeypatch.setattr(worker_module, "enumerate_algorithm_choices", counted)
    worker = lower_worker_program(indexed, plan, _profile(), block_threads=32)

    assert calls == [indexed]
    assert len(worker.stages) == 3


def test_v3r028_grouped_state_guard_requires_the_specialized_append_position():
    class RuntimeGuards:
        value_ids = ("position",)
        runtime_guards = ({"operation_id": "writer", "value_id": "position",
                           "lower": 0, "upper_exclusive": 8, "equals": 3},)

    assert _runtime_guard_error(RuntimeGuards(), (torch.tensor([3]),)) is None
    rejected = _runtime_guard_error(RuntimeGuards(), (torch.tensor([4]),))
    assert rejected["return_code"] == -12
    assert "differs from specialized position 3" in rejected["reason"]


def test_v3r028_binding_guard_ignores_empty_storage_pointer_collisions():
    program = SimpleNamespace(value_ids=("a", "b"), value_guards=(),
                              state_ownership=(), input_value_ids=(),
                              fresh_value_ids=("a", "b"))

    assert _binding_error(program, (torch.empty(0), torch.empty(0)), 0) is None
    shared = torch.ones(2)
    assert "overlaps an input or another fresh value" in _binding_error(
        program, (shared, shared), 0
    )


def test_v3r028_binds_alias_view_even_without_an_indexed_operation():
    _, indexed, _, _, examples = capture_unfamiliar_block()
    operation_outputs = {value for operation in indexed.operations for value in operation.outputs}
    view = next(value for value in indexed.values
                if value.alias_kind == "view" and value.value_id in operation_outputs)
    without_view_op = replace(
        indexed,
        operations=tuple(operation for operation in indexed.operations
                         if view.value_id not in operation.outputs),
    )

    values = dict(zip((item.value_id for item in indexed.values),
                      bind_worker_values(without_view_op, examples)))
    assert values[view.value_id].untyped_storage().data_ptr() == values[
        view.alias_sources[0]
    ].untyped_storage().data_ptr()

    broken_values = tuple(replace(item, alias_sources=()) if item.value_id == view.value_id else item
                          for item in indexed.values)
    with pytest.raises(WorkerProgramError, match="no exact alias source"):
        bind_worker_values(replace(without_view_op, values=broken_values), examples)


def test_v3r028_state_index_guard_traces_only_exact_bound_sources():
    values = (
        SimpleNamespace(value_id="constant", role="constant", alias_kind="tied_binding",
                        alias_sources=()),
        SimpleNamespace(value_id="copy", role="intermediate", alias_kind="fresh",
                        alias_sources=()),
        SimpleNamespace(value_id="view", role="intermediate", alias_kind="view",
                        alias_sources=("copy",)),
        SimpleNamespace(value_id="computed", role="intermediate", alias_kind="fresh",
                        alias_sources=()),
        SimpleNamespace(value_id="input", role="input", alias_kind="unknown",
                        alias_sources=()),
    )
    operations = (
        SimpleNamespace(kind="Copy", inputs=("constant",), outputs=("copy",)),
        SimpleNamespace(kind="Add", inputs=("constant", "constant"), outputs=("computed",)),
    )
    program = SimpleNamespace(values=values, operations=operations)

    assert _prelaunch_index_source(program, "view") == "constant"
    assert _prelaunch_index_source(program, "computed") is None
    assert _prelaunch_index_source(program, "input") == "input"


@pytest.mark.parametrize("mask_kind", ["causal", "window", "empty"])
def test_v3r027_fixed_capacity_attention_mask_guard_accepts_valid_rows(mask_kind):
    class RuntimeGuards:
        value_ids = ("position", "mask")
        runtime_guards = ({"operation_id": "attention", "value_id": "position",
                           "lower": 0, "upper_exclusive": 8, "equals": 3,
                           "mask_value_id": "mask"},)

    mask = torch.zeros((1, 1, 1, 8), dtype=torch.bool)
    if mask_kind == "causal":
        mask[..., :4] = True
    elif mask_kind == "window":
        mask[..., 1:4] = True

    assert _runtime_guard_error(RuntimeGuards(), (torch.tensor([3]), mask)) is None


@pytest.mark.parametrize("mask", [
    torch.tensor([[[[False, False, False, False, True, False, False, False]]]]),
    torch.zeros((1, 1, 1, 7), dtype=torch.bool),
    torch.zeros((1, 1, 1, 8), dtype=torch.uint8),
])
def test_v3r027_fixed_capacity_attention_mask_guard_rejects_future_or_bad_mask(mask):
    class RuntimeGuards:
        value_ids = ("position", "mask")
        runtime_guards = ({"operation_id": "attention", "value_id": "position",
                           "lower": 0, "upper_exclusive": 8, "equals": 3,
                           "mask_value_id": "mask"},)

    rejected = _runtime_guard_error(RuntimeGuards(), (torch.tensor([3]), mask))

    assert rejected is not None
    assert rejected["return_code"] in {-11, -13}


def test_v3r025_rejects_plan_target_and_worker_shape_mismatches():
    _, indexed, plan, _ = capture_reduce_program()
    profile = _profile()
    with pytest.raises(WorkerProgramError, match="different indexed semantics"):
        lower_worker_program(indexed, replace(plan, indexed_program_hash="tampered"), profile)
    with pytest.raises(WorkerProgramError, match="target limit"):
        lower_worker_program(indexed, plan, profile, block_threads=2048)

    facts = dict(profile.feature_attributes)
    facts["cooperative_launch"] = replace(facts["cooperative_launch"], value=False)
    with pytest.raises(WorkerProgramError, match="cooperative launch support"):
        lower_worker_program(indexed, plan, replace(profile, feature_attributes=facts))


def test_v3r026_emits_generic_origins_and_keys_cast_variants_separately():
    _, indexed, _, plan, _ = capture_unfamiliar_block()
    _, cast_indexed, _, cast_plan, _ = capture_unfamiliar_block(cast_dtype=torch.float16)
    profile = _profile()
    worker = lower_worker_program(indexed, plan, profile, block_threads=32)
    cast_variant = lower_worker_program(cast_indexed, cast_plan, profile, block_threads=32)

    names = [stage.origin_id for stage in worker.stages]
    assert indexed.strict_supported and cast_indexed.strict_supported
    assert len(worker.stages) == 6  # Two exact alias views remain in the FX coverage.
    assert len(indexed.operations) == 8
    assert worker.indexed_program_hash != cast_variant.indexed_program_hash
    assert worker.program_hash != cast_variant.program_hash
    assert any("addmm" in stage.operation_id for stage in worker.stages)
    assert any("sum" in stage.operation_id for stage in worker.stages)
    assert any("index_copy" in stage.operation_id for stage in worker.stages)
    assert len(names) == len(set(names))
    addmm_menu = next(item for item in worker.body_search
                      if "addmm" in item["operation_id"])
    assert addmm_menu["selected"] == "generic.indexed"
    assert any(item["provider"] == "simt" and item["status"] == "compatible"
               for item in addmm_menu["candidates"])
    assert any(item["provider"] == "tensor_core" and item["status"] == "rejected"
               for item in addmm_menu["candidates"])
