"""Measure isolated SM100 GEMM choices and write an optional tuning file.

Example: python tools/tune_sm100.py --shapes 2048x2048 2048x8192 --out tuning.json
The first dimension of each shape is the number of output features (N), and
the second is the reduction size (K). Validate the resulting choices on the
complete model: an isolated GEMM does not reproduce all megakernel effects.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import megabake.compiler as compiler  # noqa: E402
from megabake import graphed, make_backend  # noqa: E402
from megabake.bodies.cuda.sm100.tuning import device_fingerprint  # noqa: E402
from megabake.bodies.cuda.sm90.gemm import split_k  # noqa: E402


class Gemm(torch.nn.Module):
    def forward(self, source, weight):
        return source @ weight.T


def kernel_samples(function, count: int, cache_scrub: torch.Tensor | None) -> list[float]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(count):
            if cache_scrub is not None:
                # A full model streams many different weight matrices. Reusing
                # a single small GEMM leaves its weights unrealistically hot.
                cache_scrub.sum()
            function()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "trace.json"
        profile.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
    samples = [float(event["dur"]) for event in trace["traceEvents"]
               if event.get("cat") == "kernel" and "megabake_kernel" in event.get("name", "")]
    if len(samples) != count:
        raise RuntimeError(f"expected {count} megakernel samples, got {len(samples)}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", nargs="+", required=True, help="N x K, e.g. 2048x8192")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--split-limits", nargs="+", choices=("1", "2", "4", "8"),
                        default=("1", "2", "4", "8"))
    parser.add_argument("--cold-cache-mib", type=int, default=256,
                        help="read this many MiB before each sample; 0 measures hot cache")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.rows > 128 or args.samples < 5 or args.cold_cache_mib < 0:
        parser.error("rows must be 1–128, samples at least 5, and cache size nonnegative")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise SystemExit("An SM100 CUDA device is required")
    shapes = []
    for shape in args.shapes:
        try:
            columns, reduction = map(int, shape.lower().split("x"))
            if columns < 128 or columns % 64 or reduction < 64 or reduction % 64:
                raise ValueError
            shapes.append((columns, reduction))
        except ValueError:
            parser.error(f"invalid N x K shape: {shape}")

    os.environ.pop("MEGABAKE_SM100_TUNING_FILE", None)
    os.environ["MEGABAKE_SM100_TCGEN"] = "1"
    os.environ["MEGABAKE_SM100_TCGEN_DECODE"] = "1"
    os.environ["MEGABAKE_SM100_TCGEN_MAX_ROWS"] = "128"
    os.environ["MEGABAKE_SM100_EPILOGUE"] = "0"
    compiler._TABLE_LAUNCH_OPS = 0
    torch.manual_seed(771)
    report = {"schema": 1, "fingerprint": device_fingerprint(torch.cuda.current_device()),
              "cache_scrub_bytes": args.cold_cache_mib * 1024 * 1024,
              "choices": {}, "measurements": {}}
    cache_scrub = (torch.zeros(args.cold_cache_mib * 1024 * 1024 // 2,
                               device="cuda", dtype=torch.float16)
                   if args.cold_cache_mib else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for columns, reduction in shapes:
        source = torch.randn(args.rows, reduction, device="cuda", dtype=torch.bfloat16) / 16
        weight = torch.randn(columns, reduction, device="cuda", dtype=torch.bfloat16) / 16
        with torch.no_grad():
            reference = Gemm()(source, weight).float()
        records, seen = [], set()
        for tile in (64, 128):
            if columns % tile:
                continue
            for cap in map(int, args.split_limits):
                os.environ["MEGABAKE_SM100_TCGEN_TILE"] = str(tile)
                os.environ["MEGABAKE_SM100_SPLIT_K"] = str(cap)
                torch._dynamo.reset()
                compiled = graph = backend = None
                try:
                    backend = make_backend(mode="decode" if args.rows == 1 else "prefill",
                                           compile_kernel=True, cache=False, scheduler="phase")
                    compiled = torch.compile(Gemm().eval(), backend=backend, fullgraph=True, dynamic=False)
                    with torch.no_grad():
                        output = compiled(source, weight).float()
                        phase = next(p for p in backend.megabake_last_compiled.plan.phases
                                     if p.body.provider == "sm100_tma")
                        actual_split = split_k(phase.body)
                        if (tile, actual_split) in seen:
                            continue
                        seen.add((tile, actual_split))
                        error = ((output - reference).norm() / reference.norm()).item()
                        if not torch.isfinite(output).all().item() or error > 0.02:
                            raise RuntimeError(f"relative L2 error {error:.6f} exceeds 0.02")
                        graph = graphed(compiled, source, weight)
                        for _ in range(5):
                            graph(source, weight)
                        torch.cuda.synchronize()
                        samples = kernel_samples(lambda: graph(source, weight), args.samples, cache_scrub)
                    record = {"tile": tile, "splits": actual_split,
                              "median_kernel_us": statistics.median(samples),
                              "relative_l2_vs_eager": error, "samples_us": samples}
                    records.append(record)
                    print(f"{args.rows}x{columns}x{reduction}: {record}", flush=True)
                except Exception as error:  # continue measuring other candidates
                    records.append({"tile": tile, "split_limit": cap,
                                    "error": f"{type(error).__name__}: {error}"[:1000]})
                    print(f"{args.rows}x{columns}x{reduction}: {records[-1]}", flush=True)
                finally:
                    del graph, compiled, backend
                    gc.collect()
                    torch.cuda.empty_cache()
        valid = [record for record in records if "median_kernel_us" in record]
        key = f"{args.rows}:{columns}:{reduction}"
        report["measurements"][key] = records
        if valid:
            best = min(valid, key=lambda record: record["median_kernel_us"])
            report["choices"][key] = {"tile": best["tile"], "splits": best["splits"]}
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        del source, weight, reference
        torch.cuda.empty_cache()
    if len(report["choices"]) != len(shapes):
        raise SystemExit("At least one shape had no valid measured choice")


if __name__ == "__main__":
    main()
