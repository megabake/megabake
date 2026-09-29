from dataclasses import replace
from pathlib import Path

import pytest
import torch

from megabake.v3.backends.cuda.profile import CudaTargetProfile
from megabake.v3.backends.cuda.worker import WorkerProgramError, lower_worker_program
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
