import pytest

from megabake.v3.backends.cuda.simt import ContractionShape, enumerate_simt_schedules
from megabake.v3.backends.cuda.tensor_core import (
    TensorCoreSchedule,
    enumerate_tensor_core_schedules,
    owner_grid_candidates,
    output_tiles,
    work_estimate,
)


def shape(**overrides):
    values = dict(
        m=1, n=17, k=33, x_m_stride=33, x_k_stride=1,
        w_n_stride=33, w_k_stride=1, input_dtype="float16",
        output_dtype="float16", accumulation_dtype="float32",
    )
    values.update(overrides)
    return ContractionShape(**values)


def test_output_major_family_accounts_for_batch_and_tail_padding():
    desc = shape()
    schedules = enumerate_tensor_core_schedules(desc)
    assert [(s.warps_per_cta, s.mainloop_depth) for s in schedules] == [
        (warps, depth) for warps in (1, 2, 4) for depth in (1, 2, 4)
    ]
    assert output_tiles(desc) == 2
    assert [work_estimate(desc, s)["worker_groups"] for s in schedules] == [
        2, 2, 2, 1, 1, 1, 1, 1, 1
    ]
    assert work_estimate(desc, schedules[0])["padded_work_ratio"] == pytest.approx(
        (2 * 16 * 16 * 48) / (17 * 33)
    )
    assert owner_grid_candidates(desc, schedules[0], 60, 960) == (2,)


def test_bfloat16_is_tensor_core_only_and_target_guarded():
    desc = shape(input_dtype="bfloat16", output_dtype="bfloat16")
    assert len(enumerate_tensor_core_schedules(desc)) == 9
    with pytest.raises(ValueError, match="fp16-only"):
        enumerate_simt_schedules(desc)
    assert enumerate_tensor_core_schedules(desc, target="sm_100a") == ()
    assert enumerate_tensor_core_schedules(shape(m=17)) == ()


def test_bad_schedule_and_precision_contract_reject():
    with pytest.raises(ValueError, match="warps_per_cta"):
        TensorCoreSchedule(3)
    with pytest.raises(ValueError, match="mainloop_depth"):
        TensorCoreSchedule(1, 3)
    with pytest.raises(ValueError, match="contraction probes"):
        enumerate_tensor_core_schedules(shape(output_dtype="bfloat16"))
