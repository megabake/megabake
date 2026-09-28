import pytest

from megabake.v3.backends.cuda.simt import (
    ContractionShape,
    SimtSchedule,
    enumerate_simt_schedules,
    owner_grid_candidates,
    owner_grid_size,
    output_tiles,
)


def shape(**overrides):
    values = dict(
        m=1, n=17, k=33, x_m_stride=33, x_k_stride=1,
        w_n_stride=33, w_k_stride=1, input_dtype="float16",
        output_dtype="float16", accumulation_dtype="float32",
    )
    values.update(overrides)
    return ContractionShape(**values)


def test_odd_contraction_generates_k_parallel_tail_schedules():
    desc = shape()
    schedules = enumerate_simt_schedules(desc)
    assert {(s.warps_per_cta, s.vector_width) for s in schedules} == {
        (1, 1), (1, 4), (2, 1), (2, 4), (4, 1), (4, 4),
    }
    assert [output_tiles(desc, s) for s in schedules] == [17, 17, 9, 9, 5, 5]
    assert owner_grid_size(desc, schedules[-1], 60) == 5
    assert owner_grid_candidates(desc, schedules[-1], 60, 960) == (5,)


def test_owner_grid_search_stays_inside_tiles_and_measured_residency():
    desc = shape(n=576, k=576, x_m_stride=576, w_n_stride=576)
    schedule = SimtSchedule(1, 1)
    assert owner_grid_candidates(desc, schedule, 60, 1920) == (60, 240, 576)
    assert owner_grid_candidates(desc, SimtSchedule(4, 1), 60, 960) == (60, 144)


def test_inventory_weight_orientation_maps_to_source_strides():
    record = {
        "gemm": {"M": 1, "N": 576, "K": 1536,
                 "effective_strides": [[1536, 1], [1, 1536]]},
        "dtypes": {"inputs": ["float16"], "outputs": ["float16"]},
        "accumulation_dtype": "float32", "call_count": 30,
    }
    desc = ContractionShape.from_inventory(record)
    assert (desc.m, desc.n, desc.k) == (1, 576, 1536)
    assert (desc.w_n_stride, desc.w_k_stride) == (1536, 1)

    transposed = shape(w_n_stride=1, w_k_stride=17)
    assert {s.vector_width for s in enumerate_simt_schedules(transposed)} == {1}


@pytest.mark.parametrize("bad", [
    {"n": 0}, {"k": -1}, {"w_k_stride": 0},
    {"input_dtype": "bfloat16"}, {"accumulation_dtype": "float16"},
])
def test_unsupported_descriptor_is_rejected(bad):
    with pytest.raises(ValueError):
        enumerate_simt_schedules(shape(**bad))


def test_unsupported_schedule_is_rejected():
    with pytest.raises(ValueError, match="warps_per_cta"):
        SimtSchedule(3, 1)
    with pytest.raises(ValueError, match="vector_width"):
        SimtSchedule(1, 2)
