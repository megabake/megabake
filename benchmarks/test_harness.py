#!/usr/bin/env python3
"""Megabake test harness: benchmark any model (HuggingFace or built-in).

Usage:
    python benchmarks/test_harness.py mlp_silu
    python benchmarks/test_harness.py meta-llama/Llama-3.2-1B --seq-len 128
    python benchmarks/test_harness.py --list
"""

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.profiler


# ---------------------------------------------------------------------------
# Model wrapper for HF causal LMs (returns logits tensor only)
# ---------------------------------------------------------------------------

class _CausalLMWrapper(nn.Module):
    """Wrap a CausalLM so forward() returns a plain logits tensor.

    Calls the base model + lm_head directly to avoid graph breaks
    in transformers v5's CausalLM.forward (slice_indices issue).
    """
    def __init__(self, hf_model):
        super().__init__()
        self.base_model = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids):
        hidden = self.base_model(input_ids, use_cache=False).last_hidden_state
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# Built-in models (self-contained, no imports from models.py)
# ---------------------------------------------------------------------------

def _cuda_half(shape):
    return torch.randn(shape, device="cuda", dtype=torch.float16)


class _MLPSiLU(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class _MLPGeLU(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _ThreeLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, 64, bias=False)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.silu(self.fc2(x))
        return self.fc3(x)


class _RMSNormMLP(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class _LNBlock(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _LlamaLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin):
        return self.layer(hidden_states, position_embeddings=(cos, sin))


def _make_llama_decoder():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, vocab_size=256, rms_norm_eps=1e-5,
        attn_implementation="sdpa",
    )
    layer = LlamaDecoderLayer(config, layer_idx=0).cuda().half().eval()
    return _LlamaLayerWrapper(layer)


BUILT_IN = {
    "linear_256x512": (
        lambda: nn.Linear(256, 512, bias=False).cuda().half().eval(),
        lambda: (_cuda_half((1, 256)),),
    ),
    "linear_512x1024": (
        lambda: nn.Linear(512, 1024, bias=False).cuda().half().eval(),
        lambda: (_cuda_half((4, 512)),),
    ),
    "mlp_silu": (
        lambda: _MLPSiLU().cuda().half().eval(),
        lambda: (_cuda_half((1, 256)),),
    ),
    "mlp_gelu": (
        lambda: _MLPGeLU().cuda().half().eval(),
        lambda: (_cuda_half((2, 128)),),
    ),
    "mlp_3layer": (
        lambda: _ThreeLayerMLP().cuda().half().eval(),
        lambda: (_cuda_half((1, 64)),),
    ),
    "rmsnorm_mlp": (
        lambda: _RMSNormMLP().cuda().half().eval(),
        lambda: (_cuda_half((1, 256)),),
    ),
    "layernorm_mlp": (
        lambda: _LNBlock().cuda().half().eval(),
        lambda: (_cuda_half((2, 128)),),
    ),
    "llama_decoder": (
        _make_llama_decoder,
        lambda: (
            _cuda_half((1, 8, 64)),
            torch.ones(1, 8, 32, device="cuda", dtype=torch.float16),
            torch.zeros(1, 8, 32, device="cuda", dtype=torch.float16),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name: str, batch_size: int, seq_len: int):
    """Load a model and create example inputs.

    Returns (model, inputs, display_name) where model is an nn.Module and
    inputs is a tuple of tensors.
    """
    if name in BUILT_IN:
        model_fn, input_fn = BUILT_IN[name]
        return model_fn(), input_fn(), name

    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16,
    ).cuda().eval()

    wrapped = _CausalLMWrapper(model)
    vocab_size = getattr(config, "vocab_size", 32000)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device="cuda",
    )

    short_name = name.split("/")[-1]
    display = f"{short_name} (bs={batch_size}, seq={seq_len})"
    return wrapped, (input_ids,), display


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

def _extract_tensor(out):
    """Pull a single tensor from model output (handles tuples, dataclasses)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return _extract_tensor(out[0])
    if hasattr(out, "logits"):
        return out.logits
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


def measure_latency(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    return statistics.median(timings)


def count_kernels(fn) -> int:
    fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        fn()
        torch.cuda.synchronize()

    count = 0
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = evt.name
        if name.startswith("Memcpy") or name.startswith("Memset"):
            continue
        if "FillFunctor" in name or "ZeroFunctor" in name:
            continue
        if "vectorized_elementwise_kernel" in name:
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Backend results
# ---------------------------------------------------------------------------

@dataclass
class Result:
    backend: str
    compile_ms: float
    latency_us: float
    kernels: int
    max_diff: float
    error: str | None = None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def run_eager(model, inputs, warmup, iters, **_) -> Result:
    def fn():
        with torch.no_grad():
            return model(*inputs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    latency = measure_latency(fn, warmup, iters)
    kernels = count_kernels(fn)

    return Result("eager", compile_ms, latency, kernels, 0.0)


def _compile_via_export(model, inputs):
    """Export the model then compile via inductor — avoids dynamo graph breaks
    on HF transformers v5 output classes."""
    from torch.export import export
    decomp_table = torch._decomp.core_aten_decompositions()
    for op in [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
    ]:
        decomp_table.pop(op, None)
    ep = export(model, inputs, strict=False)
    ep = ep.run_decompositions(decomp_table)
    return torch.compile(ep.module(), backend="inductor")


def run_torch_compile(model, inputs, warmup, iters, eager_ref, **_) -> Result:
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        with torch.no_grad():
            compiled(*inputs)
    except Exception:
        torch._dynamo.reset()
        try:
            compiled = torch.compile(model, backend="inductor", fullgraph=False)
            with torch.no_grad():
                compiled(*inputs)
        except Exception:
            compiled = _compile_via_export(model, inputs)
            with torch.no_grad():
                compiled(*inputs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        with torch.no_grad():
            compiled(*inputs)
    except Exception:
        torch._dynamo.reset()
        try:
            compiled = torch.compile(model, backend="inductor", fullgraph=False)
            with torch.no_grad():
                compiled(*inputs)
        except Exception:
            compiled = _compile_via_export(model, inputs)
            with torch.no_grad():
                compiled(*inputs)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    def fn():
        with torch.no_grad():
            return compiled(*inputs)

    latency = measure_latency(fn, warmup, iters)
    kernels = count_kernels(fn)

    out = _extract_tensor(fn())
    max_diff = (out.float() - eager_ref.float()).abs().max().item()

    return Result("torch.compile", compile_ms, latency, kernels, max_diff)


def run_megabake(model, inputs, warmup, iters, eager_ref, **_) -> Result:
    import megabake

    example = inputs[0] if len(inputs) == 1 else tuple(inputs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled = megabake.compile(model, example)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000

    def fn():
        return megabake.run(compiled, model, *inputs)

    latency = measure_latency(fn, warmup, iters)
    kernels = count_kernels(fn)

    out = fn()
    max_diff = (out.float() - eager_ref.float()).abs().max().item()

    return Result("megabake", compile_ms, latency, kernels, max_diff)


BACKENDS = {
    "eager": run_eager,
    "torch.compile": run_torch_compile,
    "megabake": run_megabake,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[Result], display_name: str) -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  SMs: {props.multi_processor_count}")
    print(f"Model: {display_name}\n")

    hdr = f"{'Backend':<16s} {'Compile(ms)':>11s} {'Latency(us)':>11s} {'Kernels':>8s} {'MaxDiff':>10s}"
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for r in results:
        if r.error:
            print(f"{r.backend:<16s} FAIL: {r.error}")
        else:
            diff_str = "—" if r.backend == "eager" else f"{r.max_diff:.6f}"
            compile_str = "—" if r.backend == "eager" else f"{r.compile_ms:>11.1f}"
            print(
                f"{r.backend:<16s} {compile_str:>11s} "
                f"{r.latency_us:>11.1f} "
                f"{r.kernels:>8d} "
                f"{diff_str:>10s}"
            )
    print(sep)

    eager = next((r for r in results if r.backend == "eager" and r.error is None), None)
    tc = next((r for r in results if r.backend == "torch.compile" and r.error is None), None)
    mb = next((r for r in results if r.backend == "megabake" and r.error is None), None)

    if mb and eager:
        print(f"megabake vs eager:          {eager.latency_us / mb.latency_us:.2f}x  "
              f"({mb.kernels} vs {eager.kernels} kernels)")
    if mb and tc:
        print(f"megabake vs torch.compile:  {tc.latency_us / mb.latency_us:.2f}x  "
              f"({mb.kernels} vs {tc.kernels} kernels)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Megabake test harness: benchmark any model",
    )
    parser.add_argument("model", nargs="?", help="HF model ID or built-in name")
    parser.add_argument("--list", action="store_true", help="List built-in models")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--backends", type=str, default="eager,torch.compile,megabake",
        help="Comma-separated backends (default: eager,torch.compile,megabake)",
    )
    args = parser.parse_args()

    if args.list:
        print("Built-in models:")
        for name in BUILT_IN:
            print(f"  {name}")
        return

    if not args.model:
        parser.error("model name required (or use --list)")

    backend_names = [b.strip() for b in args.backends.split(",")]
    for b in backend_names:
        if b not in BACKENDS:
            parser.error(f"unknown backend: {b!r}  (choices: {list(BACKENDS.keys())})")

    print(f"Loading model: {args.model} ...")
    model, inputs, display_name = load_model(args.model, args.batch_size, args.seq_len)

    eager_ref = None
    results: list[Result] = []

    for bname in backend_names:
        print(f"  [{bname}] ", end="", flush=True)
        try:
            r = BACKENDS[bname](
                model, inputs,
                warmup=args.warmup, iters=args.iters,
                eager_ref=eager_ref,
            )
            if bname == "eager" and r.error is None:
                with torch.no_grad():
                    eager_ref = _extract_tensor(model(*inputs)).clone()
            print(
                f"latency={r.latency_us:.1f}us  kernels={r.kernels}"
                if r.error is None else f"FAIL: {r.error}"
            )
        except Exception as e:
            r = Result(bname, -1, -1, -1, -1, error=str(e))
            print(f"FAIL: {e}")
        results.append(r)
        torch.cuda.empty_cache()

    print_results(results, display_name)


if __name__ == "__main__":
    main()
