import pytest

from megabake.v3.contracts import ContractError
from megabake.v3.backends.cuda.profile import CudaTargetProfile, ProfileFact


def profile(*, architecture=("sm_90a",), family=None, compiler=("sm_90", "sm_90a")):
    known = lambda value, source="fixture": ProfileFact(value, source)
    unknown = lambda reason: ProfileFact(None, "fixture", reason)
    return CudaTargetProfile(
        device_index=0, device_name="synthetic CUDA target", device_uuid="fixture-uuid",
        device_facts={
            "compute_capability": known("9.0"),
            "visible_sms": known(60),
            "visible_memory_bytes": known(1_000_000),
        },
        resource_limits={"max_threads_per_block": known(1024)},
        feature_attributes={"cooperative_launch": known(True)},
        target_sets={
            "baseline": known(["sm_90"]),
            "architecture_specific": known(list(architecture)),
            "family_specific": unknown("no exact family target mapping") if family is None else known(list(family)),
            "compiler_targets": known(list(compiler)),
        },
        versions={"nvcc": known("12.8.93"), "cutlass": unknown("not installed")},
    )


def test_profile_round_trips_with_sources_unknowns_and_target_key():
    target = profile()
    restored = CudaTargetProfile.from_json(target.to_json())
    assert restored.to_dict() == target.to_dict()
    assert restored.profile_key == target.profile_key
    assert restored.supports_target("sm_90") is True
    assert restored.supports_target("sm_90a") is True
    assert restored.supports_target("sm_90f") is None
    assert "versions.cutlass" in restored.unknown_fields
    assert "target_sets.family_specific" in restored.unknown_fields


def test_architecture_and_family_targets_are_never_interchanged():
    target = profile(architecture=("sm_100a",), family=("sm_100f",), compiler=("sm_100a", "sm_100f"))
    assert target.supports_target("sm_100a") is True
    assert target.supports_target("sm_100f") is True
    assert target.supports_target("sm_90a") is False
    assert target.supports_target("sm_100") is False


def test_unknown_fact_needs_reason_and_tampered_profile_is_rejected():
    with pytest.raises(ContractError, match="unknown profile facts require a reason"):
        ProfileFact(None, "fixture")
    payload = profile().to_dict()
    payload["target_sets"]["architecture_specific"]["value"] = ["sm_100a"]
    with pytest.raises(ContractError, match="profile key"):
        CudaTargetProfile.from_dict(payload)

