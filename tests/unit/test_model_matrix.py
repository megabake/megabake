from benchmarks.model_matrix import _cells


def _run(variant, value, code=0):
    key = "inductor_graphed" if variant == "inductor" else "megabake_graphed"
    return {
        "model": "test",
        "round": 0,
        "variant": variant,
        "exit_code": code,
        "report": {"decode_contexts": {key: {"16": {"pipelined_call_us": value}}}},
    }


def test_matrix_requires_successful_baseline_and_comparison_runs():
    runs = [_run("baseline", 120), _run("candidate", 85), _run("inductor", 100)]
    cell = next(
        item for item in _cells("test", runs) if item["context"] == 16 and item["round"] == 0
    )
    assert cell["status"] == "pass"
    runs[0]["exit_code"] = 1
    assert all(item["status"] == "gap" for item in _cells("test", runs))


def test_matrix_reports_missing_contexts_and_regressions():
    runs = [_run("baseline", 120), _run("candidate", 95), _run("inductor", 100)]
    cells = _cells("test", runs)
    assert len(cells) == 12
    assert sum(item["status"] == "fail" for item in cells) == 1
    assert sum(item["status"] == "gap" for item in cells) == 11


def test_model_round_requires_improvement_against_incumbent():
    from benchmarks.model_matrix import _model_rounds

    cells = [
        {
            "model": "test",
            "round": 0,
            "context": context,
            "candidate_incumbent_ratio": 0.99,
            "status": "pass",
        }
        for context in (16, 128, 1024, 4096)
    ]
    assert _model_rounds(cells)[0]["status"] == "fail"
    for cell in cells:
        cell["candidate_incumbent_ratio"] = 0.94
    assert _model_rounds(cells)[0]["status"] == "pass"


def test_per_context_benchmarks_compile_separate_cache_shapes(monkeypatch, tmp_path):
    from argparse import Namespace

    from benchmarks import full_model

    calls = []

    def measure(model, args):
        calls.append((tuple(args.decode_contexts), args.out))
        context = args.decode_contexts[0]
        return {
            "capacity": context + 128,
            "megabake_graphed": {str(context): {"pipelined_call_us": 100}},
        }

    monkeypatch.setattr(full_model, "_bench_context_group", measure)
    args = Namespace(
        context_capacity="per-context",
        decode_contexts=[16, 4096],
        out=tmp_path,
        variants=["megabake_graphed"],
    )
    report = full_model.bench_decode_contexts(None, args)
    assert [item[0] for item in calls] == [(16,), (4096,)]
    assert calls[0][1] != calls[1][1]
    assert report["capacity_by_context"] == {"16": 144, "4096": 4224}
    assert set(report["megabake_graphed"]) == {"16", "4096"}
