"""MB3-004: workload and numerical contracts are explicit and immutable."""

from __future__ import annotations

import pytest

from megabake.v3.contracts import (
    BenchmarkCell,
    ContractError,
    ExceptionalValuePolicy,
    NumericalPolicy,
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
