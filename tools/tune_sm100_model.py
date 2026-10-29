"""Tune SM100 GEMM shapes using phase timings from a complete decode model.

Example: python tools/tune_sm100_model.py --model /path/to/model --out tuning.json
The result is a starting point: confirm the combined choices with a full-model
benchmark and accuracy run, because phase timings include instrumentation and
independent phases can overlap.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from megabake.bodies.cuda.sm100.tuning import device_fingerprint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--profile-timeout", type=int, default=480,
                        help="maximum seconds for each model profile (default: 480)")
    parser.add_argument("--tiles", nargs="+", choices=("auto", "64", "128"), default=("64", "128"))
    parser.add_argument("--split-limits", nargs="+", choices=("1", "2", "4", "8"), default=("1", "2", "4", "8"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 1 or args.profile_timeout < 1 or args.capacity <= args.calls + 4:
        parser.error("calls and profile timeout must be positive and capacity must exceed calls + 4")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise SystemExit("An SM100 CUDA device is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out.parent / f"{args.out.stem}-profiles"
    runs_dir.mkdir(exist_ok=True)
    profiler = Path(__file__).with_name("phase_profile.py")
    measurements = defaultdict(list)
    failures = []
    for tile in args.tiles:
        for split_limit in args.split_limits:
            label = f"tile{tile}-split{split_limit}"
            result = runs_dir / f"{label}.json"
            log = runs_dir / f"{label}.log"
            env = os.environ.copy()
            env.pop("MEGABAKE_SM100_TUNING_FILE", None)
            env.update(MEGABAKE_SM100_TCGEN="1", MEGABAKE_SM100_TCGEN_TILE=tile,
                       MEGABAKE_SM100_SPLIT_K=split_limit)
            command = [sys.executable, str(profiler), "--model", args.model, "--mode", "decode",
                       "--capacity", str(args.capacity), "--calls", str(args.calls), "--cta", "--out", str(result)]
            print(f"Profiling {label}", flush=True)
            try:
                with log.open("w") as handle:
                    run = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                         check=False, timeout=args.profile_timeout)
            except subprocess.TimeoutExpired:
                failures.append({"candidate": label, "error": "timeout", "log": str(log)})
                print(f"{label} timed out; see {log}", flush=True)
                continue
            if run.returncode:
                failures.append({"candidate": label, "exit_code": run.returncode, "log": str(log)})
                print(f"{label} failed; see {log}", flush=True)
                continue
            profile = json.loads(result.read_text())
            grouped = defaultdict(list)
            for row in profile["cta"]["phases"]:
                if "gemm_shape" in row:
                    key = (tuple(row["gemm_shape"]), row["feature_tile"], row["split_k"])
                    grouped[key].append(row["us"])
            for (shape, feature_tile, splits), times in grouped.items():
                key = ":".join(map(str, shape))
                measurements[key].append({"tile": feature_tile, "splits": splits,
                                          "mean_critical_us": sum(times) / len(times),
                                          "occurrences": len(times), "profile": str(result)})
    choices = {}
    for shape, records in measurements.items():
        best = min(records, key=lambda record: record["mean_critical_us"])
        choices[shape] = {"tile": best["tile"], "splits": best["splits"]}
    report = {"schema": 1, "fingerprint": device_fingerprint(torch.cuda.current_device()),
              "model": args.model, "method": "instrumented full-model decode phases",
              "choices": choices, "measurements": dict(measurements), "failures": failures}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"choices": choices, "failures": failures}, indent=2), flush=True)
    if not choices:
        raise SystemExit("No SM100 GEMM phases were measured")


if __name__ == "__main__":
    main()
