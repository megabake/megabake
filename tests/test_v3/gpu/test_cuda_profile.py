import pytest
import torch

from megabake.v3.backends.cuda.profile import CudaTargetProfile, query_target_profile


@pytest.mark.v3_gpu
def test_selected_cuda_profile_queries_hardware_versions_and_provenance():
    if not torch.cuda.is_available():
        pytest.skip("no selected CUDA device")
    profile = query_target_profile()
    restored = CudaTargetProfile.from_json(profile.to_json())
    assert restored.profile_key == profile.profile_key
    assert profile.device_facts["visible_sms"].value == torch.cuda.get_device_properties(0).multi_processor_count
    assert profile.device_facts["visible_sms"].source.endswith("multi_processor_count")
    assert profile.resource_limits["max_threads_per_block"].value > 0
    assert profile.feature_attributes["cooperative_launch"].value in (True, False)
    assert profile.feature_attributes["cooperative_launch"].source.startswith(
        "cuda.bindings.runtime.cudaDeviceGetAttribute"
    )
    assert profile.versions["nvcc"].value
    assert profile.versions["cuda_runtime"].value
    assert profile.versions["cuda_driver"].value
    assert profile.target_sets["architecture_specific"].value is not None
    if profile.device_facts["compute_capability"].value == "9.0":
        assert profile.supports_target("sm_90a") is True
        assert profile.supports_target("sm_90f") is None
