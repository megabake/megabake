"""Profile phase and barrier costs for the pinned Laya benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MEGABAKE_PHASE_TIMER"] = "cta" if "--cta" in sys.argv else "1"

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.laya_benchmark import MODEL, REVISION, _batch, _load_agent, _run  # noqa: E402
from megabake import make_backend  # noqa: E402
from megabake.fx_handler.hf_attention import register_transformers_attention  # noqa: E402
from tools.phase_profile import cta_profile, profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--cta", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    agent = _load_agent(args.model, args.revision)
    register_transformers_attention()
    agent.model.encoder.set_attn_implementation("megabake")
    inputs = _batch(agent)
    backend = make_backend(
        mode="prefill", fallback="error", compile_kernel=True, cache=False, scheduler="phase"
    )
    captured = torch.compile(agent.model, backend=backend, fullgraph=True, dynamic=False)
    run = lambda: _run(agent, captured, inputs)  # noqa: E731
    with torch.inference_mode():
        run()
        compiled = backend.megabake_last_compiled
        result = (
            cta_profile(captured, compiled, run, args.calls)
            if args.cta
            else profile(captured, compiled, run, args.calls)
        )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
