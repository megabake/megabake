from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/v3"))
import body_harness


@pytest.fixture(scope="module")
def compiled_body(tmp_path_factory):
    output = tmp_path_factory.mktemp("v3-body")
    build = body_harness.compile_library(output)
    return body_harness.load_library(build["library"])


@pytest.mark.v3_gpu
def test_v3r005_standalone_and_admitted_owner_entry(compiled_body):
    report = body_harness.run_card(
        "005", compiled_body, samples=3, graph_launches=2, replays=1,
    )
    case = report["cases"][0]
    tactic = case["tactics"][0]
    owner = tactic["owner_entries"][0]
    assert tactic["standalone"]["correct"]
    assert owner["correct"]
    assert owner["resources"]["cooperative_launch"] == 1
    assert case["negative_case"]["rejected"]


@pytest.mark.v3_gpu
def test_v3r006_tails_alpha_beta_orientation_and_hot_shapes(compiled_body):
    report = body_harness.run_card(
        "006", compiled_body, samples=2, graph_launches=2, replays=1,
    )
    assert len(report["cases"]) == 6  # five inventoried shapes plus LINEAR_TINY
    tiny = report["cases"][-1]
    assert tiny["shape"]["N"] == 17 and tiny["shape"]["K"] == 33
    assert tiny["shape"]["weight_strides"] == [1, 17]
    assert tiny["shape"]["alpha"] == 1.75 and tiny["shape"]["beta"] == -0.25
    assert all(t["standalone"]["correct"] and
               all(e["correct"] for e in t["owner_entries"])
               for case in report["cases"] for t in case["tactics"])
    assert report["negative_case"]["rejected"]


@pytest.mark.v3_gpu
def test_v3r007_wmma_tails_bfloat16_and_owner_guards(compiled_body):
    report = body_harness.run_card(
        "007", compiled_body, samples=1, graph_launches=1, replays=1,
    )
    assert report["target_guard"] == "sm_90a"
    assert len(report["cases"]) == 9
    for case in report["cases"]:
        assert case["vendor_control"]["correct"]
        assert len(case["tactics"]) == 9
        for tactic in case["tactics"]:
            assert tactic["schedule"]["mma_shape"] == "m16n16k16"
            assert tactic["schedule"]["k_tile_per_mainloop"] == (
                16 * tactic["schedule"]["mainloop_depth"]
            )
            assert tactic["work_estimate"]["padded_work_ratio"] >= 1
            assert tactic["standalone"]["correct"]
            assert tactic["standalone"]["resources"]["cooperative_launch"] == 1
            assert tactic["owner_entries"]
            assert all(entry["correct"] for entry in tactic["owner_entries"])
    tiny = report["cases"][-4]
    assert tiny["shape"]["N"] == 17 and tiny["shape"]["K"] == 33
    assert tiny["shape"]["weight_strides"] == [1, 17]
    assert tiny["shape"]["alpha"] == 1.75 and tiny["shape"]["beta"] == -0.25
    assert report["cases"][-1]["shape"]["dtype"] == "bfloat16"
    assert report["cases"][-1]["shape"]["M"] == 3
    bf16_shapes = {
        (case["shape"]["M"], case["shape"]["N"], case["shape"]["K"])
        for case in report["cases"] if case["shape"]["dtype"] == "bfloat16"
    }
    assert {(1, 192, 576), (1, 49152, 576), (3, 17, 33)} <= bf16_shapes
    assert all(case["rejected"] for case in report["negative_cases"])
