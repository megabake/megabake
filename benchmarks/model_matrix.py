"""Interleaved, per-cell decode acceptance matrix for two MegaBake checkouts.

Run with --baseline PATH --candidate PATH --out PATH. This orchestrates the
existing full-model benchmark; it does not introduce an execution backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

MODELS = (
    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-1.7B",
    "Qwen/Qwen2-0.5B",
    "jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0",
    "reflection/Beam-1.0-501B-A23B-DFlash2",
    "google/gemma-2b",
    "google/gemma-3-270m",
    "Qwen/Qwen3.5-0.8B",
)


def _cells(model, results):
    cells = []
    for context in (16, 128, 1024, 4096):
        for number in range(3):
            values = {}
            for item in results:
                if item["model"] != model or item["round"] != number or item["exit_code"] != 0:
                    continue
                key = "inductor_graphed" if item["variant"] == "inductor" else "megabake_graphed"
                row = item["report"].get("decode_contexts", {}).get(key, {}).get(str(context), {})
                if "pipelined_call_us" in row:
                    values[item["variant"]] = row["pipelined_call_us"]
            ratio = (
                values["candidate"] / values["inductor"]
                if len(values) == 3
                and all(math.isfinite(value) and value > 0 for value in values.values())
                else None
            )
            cells.append(
                {
                    "model": model,
                    "context": context,
                    "round": number,
                    "latency_us": values,
                    "candidate_inductor_ratio": ratio,
                    "candidate_incumbent_ratio": (
                        values["candidate"] / values["baseline"] if ratio is not None else None
                    ),
                    "status": "gap"
                    if ratio is None
                    else "pass"
                    if (
                        (model in MODELS[6:] or ratio <= 0.9)
                        and values["candidate"] <= values["baseline"] * 1.03
                    )
                    else "fail",
                }
            )
    return cells


def _model_rounds(cells):
    rows = []
    for model in dict.fromkeys(cell["model"] for cell in cells):
        for number in range(3):
            group = [cell for cell in cells if cell["model"] == model and cell["round"] == number]
            ratios = [cell["candidate_incumbent_ratio"] for cell in group]
            complete = len(group) == 4 and all(value is not None for value in ratios)
            geometric_mean = (
                math.exp(sum(math.log(value) for value in ratios) / 4) if complete else None
            )
            rows.append(
                {
                    "model": model,
                    "round": number,
                    "candidate_incumbent_geomean": geometric_mean,
                    "status": "gap"
                    if not complete
                    else "pass"
                    if (
                        geometric_mean <= (1.03 if model in MODELS[6:] else 0.95)
                        and all(cell["status"] == "pass" for cell in group)
                    )
                    else "fail",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for model in args.model:
        revision = None
        for round_id in range(3):
            variants = ["baseline", "candidate", "inductor"]
            variants = variants[round_id:] + variants[:round_id]
            for variant in variants:
                checkout = (args.baseline if variant == "baseline" else args.candidate).resolve()
                output = args.out / model.replace("/", "--") / str(round_id) / variant
                output.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "benchmarks/full_model.py",
                    "--model",
                    model,
                    "--modes",
                    "decode",
                    "--variants",
                    "inductor_graphed" if variant == "inductor" else "megabake_graphed",
                    "--samples",
                    "40",
                    "--warmup",
                    "5",
                    "--capacity",
                    "128",
                    "--decode-contexts",
                    "16",
                    "128",
                    "1024",
                    "4096",
                    "--out",
                    str(output),
                ]
                if revision:
                    command += ["--revision", revision]
                env = dict(
                    os.environ, PYTHONPATH=str(checkout / "src") + os.pathsep + str(checkout)
                )
                with (output / "run.log").open("w") as log:
                    process = subprocess.run(
                        command,
                        cwd=checkout,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                report_path = output / "report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                resolved = report.get("provenance", {}).get("checkpoint_revision")
                if resolved and revision and resolved != revision:
                    raise RuntimeError(f"Checkpoint changed for {model}: {revision} -> {resolved}")
                revision = revision or resolved
                results.append(
                    {
                        "model": model,
                        "round": round_id,
                        "variant": variant,
                        "exit_code": process.returncode,
                        "report": report,
                        "log": str(output / "run.log"),
                    }
                )
                cells = _cells(model, results)
                # Preserve completed model cells as subsequent models run.
                matrix_path = args.out / "matrix.json"
                previous = (
                    json.loads(matrix_path.read_text()) if matrix_path.exists() else {"cells": []}
                )
                all_cells = [cell for cell in previous["cells"] if cell["model"] != model] + cells
                matrix_path.write_text(
                    json.dumps(
                        {
                            "runs": results,
                            "cells": all_cells,
                            "model_rounds": _model_rounds(all_cells),
                            "decode_target_met": len(all_cells) == len(args.model) * 12
                            and all(row["status"] == "pass" for row in _model_rounds(all_cells)),
                            "prefill_validated": False,
                            "performance_objective_complete": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"{model} round={round_id} {variant}: exit={process.returncode}", flush=True)


if __name__ == "__main__":
    main()
