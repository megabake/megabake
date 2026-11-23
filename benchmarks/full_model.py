"""Time a Hugging Face causal LM under torch.compile: Inductor against MegaBake.

Both variants compile the same checkpoint with the same flags, and both replay
CUDA graphs:

- prefill (``--prompt-tokens``): the model itself. ``inductor`` is
  ``torch.compile`` with max-autotune, CUDA graphs and PDL; ``megabake`` is
  ``torch.compile(model, backend=megabake)``, replayed through ``megabake.graphed``.
- decode: one token per call, with ``--contexts`` tokens already in the cache.
  ``inductor`` is the model with its own attention and a Hugging Face
  ``StaticCache``, compiled the same way. ``megabake``
  compiles the same model through ``decode_module`` (attention declared as
  MegaBake ops over explicit KV tensors), because MegaBake does not compile
  Hugging Face's cache object. Cache contents stay zero: decode time does not
  depend on them.

Per variant: ``setup_s`` (load, compile, capture and first call), the median
synchronized call, the mean pipelined call (back to back, one sync), and the
kernel count and kernel time of one call. A variant that fails is recorded
with its error. Pass several ``--model``s for one report each under
``--out/<org>--<name>``.

    .venv/bin/python benchmarks/full_model.py
    .venv/bin/python benchmarks/full_model.py --model Qwen/Qwen2-0.5B --modes decode --contexts 16 4096
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # This checkout's sources, not whatever ``megabake`` the environment
    # installed (an editable install points at one tree only).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.measure import kernel_trace
from benchmarks.stateful_models import StatefulCausalLM, layer_types, text_config
from megabake import graphed, make_backend
from megabake.fx_handler.hf_attention import (
    ATTENTION_IMPLEMENTATION,
    DecodeStep,
    mixer_of,
    register_transformers_attention,
)
from tests.fixtures import SMOLLM_MODEL_NAME, SMOLLM_REVISION, StatefulSmolLM


def _flex_mask_without_compile_flag() -> None:
    """transformers 5.18 passes torch's deprecated ``_compile=True`` to
    ``create_block_mask``; its DeprecationWarning breaks ``fullgraph``. Without
    the flag Dynamo traces the mask inline, as the warning recommends.
    shortcut: patches transformers; drop once it stops passing ``_compile``."""
    from torch.nn.attention.flex_attention import create_block_mask
    from transformers import masking_utils

    masking_utils.create_block_mask = lambda *args, _compile=False, **kwargs: create_block_mask(*args, **kwargs)


PROMPT = [1, 504, 3458, 314]
VARIANTS = ("inductor", "megabake")


def _load(name: str, revision: str | None, attention: str = "eager", config=None) -> torch.nn.Module:
    """``attention="megabake"`` declares attention to MegaBake through the
    transformers attention interface instead of leaving it to pattern matching."""
    from transformers import AutoModelForCausalLM

    if attention == ATTENTION_IMPLEMENTATION:
        register_transformers_attention()
    # Checkpoints that ship their own modeling code run it only on request:
    # MEGABAKE_TRUST_REMOTE_CODE=1, with a pinned --revision of code you read.
    trust = os.environ.get("MEGABAKE_TRUST_REMOTE_CODE") == "1"
    if trust and revision is None:
        raise ValueError("MEGABAKE_TRUST_REMOTE_CODE=1 needs a pinned --revision")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        attn_implementation=attention,
        dtype=torch.bfloat16,
        trust_remote_code=trust,
        **({"config": config} if config is not None else {}),
    )
    model.config.use_cache = False
    text_config(model).use_cache = False
    return model.to("cuda").eval()


def decode_module(model, capacity: int, batch: int = 1) -> tuple[torch.nn.Module, list]:
    """The one-token decode wrapper and the shapes and dtypes of its state tensors.

    With ``batch`` > 1 every per-sequence state tensor (KV caches, conv and
    recurrent state) gets that leading dimension; the int32 valid lengths stay
    ``[1]``, shared by the batch, since every sequence decodes in lockstep.
    """
    stateful, state_shapes = _decode_module(model, capacity)
    if batch == 1:
        return stateful, state_shapes
    return stateful, [
        [((batch, *shape[1:]) if len(shape) > 1 else shape, dtype) for shape, dtype in layer]
        for layer in state_shapes
    ]


def _decode_module(model, capacity: int) -> tuple[torch.nn.Module, list]:
    if model.config.model_type == "llama":
        # Keep the hand-written Llama wrapper the tests and published numbers use.
        config = model.config
        layers = config.num_hidden_layers
        shape = (
            1,
            capacity,
            config.num_key_value_heads,
            config.hidden_size // config.num_attention_heads,
        )
        state_shapes = [[(shape, torch.bfloat16)]] * 2 * layers + [[((1,), torch.int32)]] * layers
        return StatefulSmolLM(model).eval(), state_shapes
    mixers = [module for module in model.modules() if mixer_of(module) is not None]
    if mixers and len(mixers) == model.config.get_text_config(decoder=True).num_hidden_layers:
        # Every layer's token mixer has a declaration (Mamba2, ComplexKDA and
        # its hybrid attention): ``megabake::linear_attention`` and
        # ``megabake::kv_attention`` over DecodeStep's explicit state.
        step = DecodeStep(model, capacity).eval()
        return step, step.state_shapes()
    if set(layer_types(model)) <= {"full_attention", "sliding_attention"}:
        # Any Hugging Face attention model: declared attention over explicit KV state.
        step = DecodeStep(model, capacity).eval()
        return step, step.state_shapes()
    # Hybrid token mixers (linear attention) are not declared ops yet.
    stateful = StatefulCausalLM(model).eval()
    return stateful, stateful.state_shapes(capacity)


def _megabake(mode: str, out: Path):
    return make_backend(
        mode=mode, compile_kernel=True, cache=False, dump_dir=out / "dumps" / mode, scheduler="phase"
    )


def _compile(module, backend: str, args) -> object:
    if backend == "inductor":
        # mode="max-autotune" plus PDL; compile takes mode or options, not both.
        options = {**torch._inductor.list_mode_options("max-autotune"), "triton.enable_pdl": True}
        return torch.compile(module, options=options, fullgraph=True, dynamic=False)
    return torch.compile(module, backend=_megabake(args.mode, args.out), fullgraph=True, dynamic=False)


def _calls_needed(args) -> int:
    return 1 + args.warmup + 2 * args.samples + 1  # first, warmup, synced, pipelined, trace


def _measure(name: str, build, call, args) -> dict:
    """``build()`` returns the function; ``call(function, index)`` makes call
    ``index``, each index once, so decode advances one position per call."""
    try:
        torch._dynamo.reset()
        index = iter(range(_calls_needed(args)))
        started = time.perf_counter()
        function = build()
        call(function, next(index))
        torch.cuda.synchronize()
        setup = time.perf_counter() - started
        for _ in range(args.warmup):
            call(function, next(index))
        synced = []
        for _ in range(args.samples):
            torch.cuda.synchronize()
            started = time.perf_counter()
            call(function, next(index))
            torch.cuda.synchronize()
            synced.append((time.perf_counter() - started) * 1e6)
        # A full collection over the compilers' objects inside the window would
        # add ~15 ms to one call.
        gc.collect()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.samples):
            call(function, next(index))
        torch.cuda.synchronize()
        pipelined = (time.perf_counter() - started) * 1e6 / args.samples
        trace = kernel_trace(lambda: call(function, next(index)), args.out / "traces" / f"{name}.json")
        entry = {
            "setup_s": setup,
            "complete_call_us": statistics.median(synced),
            "pipelined_call_us": pipelined,
            "launch_count": trace["launch_count"],
            "kernel_duration_us": trace["kernel_duration_us"],
        }
    except Exception as error:  # noqa: BLE001 - keep benchmarking the other variants
        traceback.print_exc()
        entry = {"error": f"{type(error).__name__}: {error}"}
    print(f"{name}: {json.dumps(entry)}", flush=True)
    torch.cuda.empty_cache()
    return entry


def bench_prefill(name: str, revision: str | None, args) -> dict:
    args.mode = "prefill"
    prompt = (PROMPT * -(-args.prompt_tokens // len(PROMPT)))[: args.prompt_tokens]
    ids = torch.tensor([prompt], device="cuda")
    mask = torch.ones_like(ids)
    model = _load(name, revision, args.attention)

    def build(backend):
        compiled = _compile(model, backend, args)
        call = lambda tokens, attention_mask: compiled(tokens, attention_mask=attention_mask)  # noqa: E731
        # Inductor replays its own CUDA graphs; MegaBake's are captured here.
        return call if backend == "inductor" else graphed(call, ids, mask)

    with torch.no_grad():
        results = {
            backend: _measure(f"prefill_{backend}", lambda: build(backend), lambda f, _: f(ids, mask), args)
            for backend in args.variants
        }
    return {"prompt_tokens": len(prompt), **results}


def _inductor_decode(name: str, revision: str | None, capacity: int, args):
    """The model with its own attention and a ``StaticCache``, compiled whole."""
    from transformers import AutoConfig, StaticCache

    # cuDNN's attention fails to load its sublibrary on the test machine;
    # SDPA then takes flash or memory-efficient attention.
    torch.backends.cuda.enable_cudnn_sdp(False)
    config = AutoConfig.from_pretrained(name, revision=revision)
    window = getattr(config, "sliding_window", None)
    if window and capacity <= window and getattr(config, "layer_types", None):
        # A window at least as long as the cache never drops a key, so sliding
        # layers compute full attention. Hugging Face's sliding static cache
        # keeps a Python length that recompiles the graph on every call.
        config.layer_types = ["full_attention"] * len(config.layer_types)
        # sliding_window stays set: Gemma 3 builds its sliding mask whatever
        # the layer types, and raises without a window.
        if hasattr(config, "use_sliding_window"):
            config.use_sliding_window = False
    if args.hf_attention == "flex_attention":
        _flex_mask_without_compile_flag()
    model = _load(name, revision, args.hf_attention, config=config)
    args.hf_attention_used = model.config._attn_implementation  # what Hugging Face resolved
    cache = StaticCache(config=model.config, max_cache_len=capacity)

    def step(forward, token, position):
        return forward(
            input_ids=token,
            position_ids=position,
            cache_position=position.view(-1),
            past_key_values=cache,
            use_cache=True,
        ).logits

    # StaticCache allocates its tensors on first use; one eager call does that
    # outside the CUDA graphs, so they stay the graphs' static inputs.
    step(model, torch.full((args.batch, 1), 504, device="cuda"), torch.zeros((1, 1), dtype=torch.long, device="cuda"))
    compiled = _compile(model, "inductor", args)
    return lambda token, position: step(compiled, token, position)


def _megabake_decode(name: str, revision: str | None, capacity: int, context: int, token, args):
    stateful, shapes = decode_module(_load(name, revision, args.attention), capacity, args.batch)
    state = [torch.zeros(shape, dtype=dtype, device="cuda") for layer in shapes for shape, dtype in layer]
    for tensor in state:
        if tensor.dtype == torch.int32 and tensor.numel() == 1:
            tensor.fill_(context)  # the per-layer valid length: start at the context
    function = graphed(_compile(stateful, "megabake", args), token, torch.tensor([[context]], device="cuda"), *state)
    return lambda token, position: function(token, position, *state)


def bench_decode(name: str, revision: str | None, context: int, args) -> dict:
    args.mode = "decode"
    # Each variant gets a cache just long enough for its calls: Hugging Face
    # attends over the whole static cache. Its last position trips an
    # out-of-bounds assert in a cache of exactly that length, hence the margin.
    capacity = context + _calls_needed(args) + 8
    token = torch.full((args.batch, 1), 504, device="cuda")
    # Made on the device ahead of time: a host copy would synchronize every call.
    positions = [torch.tensor([[context + i]], device="cuda") for i in range(_calls_needed(args))]
    builds = {
        "inductor": lambda: _inductor_decode(name, revision, capacity, args),
        "megabake": lambda: _megabake_decode(name, revision, capacity, context, token, args),
    }
    with torch.no_grad():
        results = {
            backend: _measure(
                f"decode_{context}_{backend}", builds[backend], lambda f, i: f(token, positions[i]), args
            )
            for backend in args.variants
        }
    attention = getattr(args, "hf_attention_used", None)
    return {"capacity": capacity, "batch": args.batch, "inductor_attention": attention, **results}


def _bench_model(name: str, revision: str | None, args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"== {name}@{revision or 'main'} -> {args.out}", flush=True)
    report: dict[str, object] = {
        "model": f"{name}@{revision or 'main'}",
        "attention": args.attention,
        "hf_attention": args.hf_attention,
        "gpu": torch.cuda.get_device_name(),
        "samples": args.samples,
        "warmup": args.warmup,
    }
    if "prefill" in args.modes:
        report["prefill"] = bench_prefill(name, revision, args)
    if "decode" in args.modes:
        report["decode"] = {str(c): bench_decode(name, revision, c, args) for c in args.contexts}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", default=[SMOLLM_MODEL_NAME], help="Hugging Face causal LM id(s)")
    parser.add_argument(
        "--revision", help=f"checkpoint revision (default: pinned for {SMOLLM_MODEL_NAME}, else main)"
    )
    parser.add_argument(
        "--attention",
        choices=("eager", ATTENTION_IMPLEMENTATION),
        default="eager",
        help="attention of the prefill model (both backends) and of MegaBake's decode load",
    )
    parser.add_argument(
        "--hf-attention",
        default=None,
        help="attention of the Inductor decode model (default: Hugging Face's choice; sdpa, eager, flex_attention, ...)",
    )
    parser.add_argument("--modes", nargs="+", choices=("prefill", "decode"), default=["prefill", "decode"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=len(PROMPT), help="prefill prompt length")
    parser.add_argument("--contexts", type=int, nargs="+", default=[0], help="decode: tokens already cached")
    parser.add_argument("--batch", type=int, default=1, help="decode batch size (sequences in lockstep)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/full_model"))
    args = parser.parse_args()
    root = args.out
    for name in args.model:
        revision = args.revision or (SMOLLM_REVISION if name == SMOLLM_MODEL_NAME else None)
        args.out = root / name.replace("/", "--") if len(args.model) > 1 else root
        _bench_model(name, revision, args)
        torch._dynamo.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
