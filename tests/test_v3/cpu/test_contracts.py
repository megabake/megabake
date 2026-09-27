"""MB3-004: workload and numerical contracts are explicit and immutable."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from megabake.v3.contracts import (
    BenchmarkCell,
    ContractError,
    ExceptionalValuePolicy,
    NumericalPolicy,
    StepABI,
    StepManifest,
    ToleranceSpec,
    WorkloadSpec,
)


def _policy() -> NumericalPolicy:
    return NumericalPolicy(
        reference_expansion="rmsnorm:v1:float32_mean_then_cast",
        intermediate_casts=("input->float32", "rsqrt->float32"),
        accumulation_dtypes={"linear": "float32", "rmsnorm": "float32"},
        output_casts={"linear": "float16", "rmsnorm": "float16"},
        permitted_reassociation={"linear": False, "rmsnorm": False},
        tolerances={
            "linear": {"float16": ToleranceSpec(atol=0.02, rtol=0.01)},
            "rmsnorm": {"float16": {"atol": 0.03, "rtol": 0.02}},
        },
        exceptional_value_policy=ExceptionalValuePolicy(
            allow_nan=False, allow_pos_inf=False, allow_neg_inf=False
        ),
    )


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        batch_size=1,
        context_length=16,
        capacity=32,
        dtype="torch.float16",
        state_semantics="fixed_capacity_kv:read_old_write_new",
        mask_semantics="causal_valid_length",
        cache_layout="b,h,capacity,d",
        input_origin="gpu",
        output_ownership="owned",
        timed_unit="cached_decode_step",
        checkpoint_id=None,
        config_id="synthetic:block_tiny:v1",
        shape_buckets=("short:16", "long:32"),
        benchmark_cells=(
            BenchmarkCell("tiny-short", "torch.compile:reduce-overhead", True, 16),
        ),
    )


def test_contract_round_trip_and_hash_changes() -> None:
    workload = _workload()
    restored = WorkloadSpec.from_dict(workload.to_dict())
    assert restored == workload
    assert restored.contract_hash == workload.contract_hash

    changed_timing = WorkloadSpec.from_dict(
        {**workload.to_dict(), "timed_unit": "forward"}
    )
    assert changed_timing.contract_hash != workload.contract_hash

    policy = _policy()
    restored_policy = NumericalPolicy.from_dict(policy.to_dict())
    assert restored_policy.contract_hash == policy.contract_hash
    changed_policy = NumericalPolicy.from_dict(
        {**policy.to_dict(), "intermediate_casts": ["input->float16"]}
    )
    assert changed_policy.contract_hash != policy.contract_hash
    assert policy.tolerance_for("linear", "torch.float16").atol == 0.02


@pytest.mark.parametrize(
    "overrides",
    [
        {"capacity": 0},
        {"context_length": 33},
        {"position": 32},
        {"output_ownership": ""},
        {"timed_unit": "not-a-timed-unit"},
    ],
)
def test_invalid_workload_contracts_are_rejected(overrides: dict[str, object]) -> None:
    fields = _workload().to_dict()
    fields.update(overrides)
    with pytest.raises(ContractError):
        WorkloadSpec.from_dict(fields)


def test_missing_numerical_policy_fields_are_rejected() -> None:
    with pytest.raises(TypeError):
        NumericalPolicy(reference_expansion="missing-fields")  # type: ignore[call-arg]


def _tiny_manifest() -> StepManifest:
    path = Path(__file__).resolve().parents[3] / "benchmarks/v3/manifests/tiny_cached_step.json"
    return StepManifest.from_json(path.read_text())


def test_step_manifest_round_trip_and_complete_cache_abi_hash() -> None:
    manifest = _tiny_manifest()
    restored = StepManifest.from_json(manifest.to_json())
    assert restored.to_dict() == manifest.to_dict()
    assert restored.contract_hash == manifest.contract_hash
    assert set(manifest.not_measured_cells) == {"tiny-advance-L0", "tiny-advance-L1"}
    assert manifest.step_abi.state_mode == "advancing"
    assert manifest.step_abi.cache_update_mode == "functional_append"

    state_data = manifest.step_abi.to_dict()
    changed_layout_name = "b,capacity,h,d"
    state_data["old_state_inputs"][0]["layout"] = changed_layout_name
    workload_data = manifest.workload.to_dict()
    workload_data["cache_layout"] = changed_layout_name
    changed_layout = replace(
        manifest,
        workload=WorkloadSpec.from_dict(workload_data),
        step_abi=StepABI.from_dict(state_data),
    )
    assert changed_layout.contract_hash != manifest.contract_hash

    policy_data = manifest.numerical_policy.to_dict()
    policy_data["intermediate_casts"] = ["x->float16"]
    changed_policy = NumericalPolicy.from_dict(policy_data)
    abi_data = manifest.step_abi.to_dict()
    abi_data["guard_set"]["numerical_policy_hash"] = changed_policy.contract_hash
    changed_numeric_contract = replace(
        manifest,
        numerical_policy=changed_policy,
        step_abi=StepABI.from_dict(abi_data),
    )
    assert changed_numeric_contract.contract_hash != manifest.contract_hash

    replay_data = manifest.step_abi.to_dict()
    replay_data["state_mode"] = "fixed_replay"
    replay_data["invocation_preparation"]["fixed_state_reset"] = {
        "policy": "restore_initial_cache",
        "frequency": "per_invocation",
    }
    replay = replace(manifest, step_abi=StepABI.from_dict(replay_data))
    assert replay.contract_hash != manifest.contract_hash


@pytest.mark.parametrize("field,value", [("old_state_inputs", []), ("new_state_outputs", [])])
def test_cached_step_manifest_rejects_missing_state_edges(field: str, value: object) -> None:
    manifest = _tiny_manifest()
    payload = manifest.step_abi.to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="state"):
        StepABI.from_dict(payload)


def test_cached_step_manifest_rejects_uncached_calls_and_pointer_hashes() -> None:
    manifest = _tiny_manifest()
    payload = manifest.to_dict()
    payload["workload"]["timed_unit"] = "forward"
    with pytest.raises(ValueError, match="cached_step"):
        StepManifest.from_dict(payload)

    payload = manifest.to_dict()
    payload["fixed_inputs"]["cache_pointer"] = 123456
    with pytest.raises(ValueError, match="pointer"):
        StepManifest.from_dict(payload)

    abi = manifest.step_abi.to_dict()
    abi["cache_update_mode"] = "disabled"
    with pytest.raises(ValueError, match="functional_append"):
        StepABI.from_dict(abi)
