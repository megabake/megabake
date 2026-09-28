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
