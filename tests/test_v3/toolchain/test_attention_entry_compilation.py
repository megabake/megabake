from pathlib import Path

import pytest

from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import compile_worker_entry, lower_worker_program
from tests.test_v3.attention_fixtures import attention_inputs, capture_attention


@pytest.mark.v3_toolchain
def test_v3r027_compiles_the_exact_attention_and_cache_entry(tmp_path):
    root = Path(__file__).resolve().parents[3]
    profile = CudaTargetProfile.from_json(
        (root / "ART/tasks/V3R-021/target_profile.json").read_text())
    _, indexed, plan = capture_attention(attention_inputs(256, capacity=257))
    worker = lower_worker_program(indexed, plan, profile, block_threads=32)
    result = compile_worker_entry(worker, profile, tmp_path / "cached_attention.so",
                                  nvcc="/usr/local/cuda/bin/nvcc")
    assert result.return_code == 0, result.stderr
    assert result.target == "sm_90" and result.artifact_hash
    assert result.ptxas["spill_store_bytes"] == result.ptxas["spill_load_bytes"] == 0
    assert "cudaLaunchCooperativeKernel" in Path(result.artifact_path).with_suffix(".cu").read_text()
    assert worker.source.count("grid.sync();") == 2
