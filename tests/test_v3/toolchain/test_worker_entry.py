from pathlib import Path

import pytest

from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import compile_worker_entry, lower_worker_program
from tests.test_v3.worker_fixtures import capture_unfamiliar_block


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.v3_toolchain
def test_v3r025_026_compile_the_composed_worker_for_the_profile_target(tmp_path):
    profile = CudaTargetProfile.from_json(
        (ROOT / "ART/tasks/V3R-021/target_profile.json").read_text()
    )
    _, indexed, _, plan, _ = capture_unfamiliar_block()
    worker = lower_worker_program(indexed, plan, profile, block_threads=32)
    result = compile_worker_entry(
        worker, profile, tmp_path / "unfamiliar_worker.so", nvcc="/usr/local/cuda/bin/nvcc"
    )

    assert result.return_code == 0, result.stderr
    assert result.target == "sm_90"
    assert result.artifact_hash and result.artifact_bytes
    assert result.ptxas["registers_per_thread"] > 0
    assert result.ptxas["spill_store_bytes"] == result.ptxas["spill_load_bytes"] == 0
    source = Path(result.artifact_path).with_suffix(".cu").read_text()
    assert source.count("grid.sync();") == len(worker.stages) - 1
    assert "cudaLaunchCooperativeKernel" in source
