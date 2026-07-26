#!/usr/bin/env python3
"""Megabake test harness: benchmark any model (HuggingFace or built-in).

Usage:
    python benchmarks/test_harness.py mlp_silu
    python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode decode
    python benchmarks/test_harness.py HuggingFaceTB/SmolLM2-135M --mode prefill --profile
    python benchmarks/test_harness.py --list
"""

import argparse
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

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


def _is_overhead_kernel(name: str) -> bool:
    """True for CUDA kernels that are infrastructure, not model compute."""
    return (name.startswith("Memcpy") or name.startswith("Memset")
            or "FillFunctor" in name or "ZeroFunctor" in name
            or "vectorized_elementwise_kernel" in name)


def measure_latency(fn, warmup: int, iters: int) -> float:
    """Wall-clock GPU latency via CUDA events. Returns median in microseconds."""
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


def profile_kernels(fn) -> list[dict]:
    """Profile one forward pass via torch.profiler.

    Returns list of {name, duration_us, count} for CUDA kernels,
    aggregated by kernel name, sorted by total duration descending.
    """
    fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        fn()
        torch.cuda.synchronize()

    by_name = defaultdict(lambda: {"duration_us": 0.0, "count": 0})
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if _is_overhead_kernel(evt.name):
            continue
        entry = by_name[evt.name]
        dur = getattr(evt, "device_time", None) or getattr(evt, "cuda_time", 0)
        entry["duration_us"] += dur
        entry["count"] += 1

    result = []
    for name, data in by_name.items():
        result.append({"name": name, "duration_us": data["duration_us"], "count": data["count"]})
    result.sort(key=lambda x: x["duration_us"], reverse=True)
    return result


def compute_gpu_kernel_time(kernel_profile: list[dict], backend: str) -> tuple[float, float]:
    """Compute GPU kernel time from profile data.

    Returns (compute_kernel_us, overhead_kernel_us).
    For megabake: separates the cooperative megakernel from clone/copy overhead.
    For others: all non-overhead kernels are compute.
    """
    total = sum(k["duration_us"] for k in kernel_profile)

    if backend == "megabake":
        mega_us = 0.0
        for k in kernel_profile:
            if "megakernel" in k["name"]:
                mega_us += k["duration_us"]
        return mega_us, total - mega_us

    return total, 0.0


# ---------------------------------------------------------------------------
# Backend results
# ---------------------------------------------------------------------------

@dataclass
class Result:
    backend: str
    compile_ms: float
    latency_us: float
    gpu_compute_us: float
    gpu_overhead_us: float
    kernels: int
    max_diff: float
    kernel_profile: list[dict] = field(default_factory=list)
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
    kp = profile_kernels(fn)
    gpu_compute, gpu_overhead = compute_gpu_kernel_time(kp, "eager")
    kernel_count = sum(k["count"] for k in kp)

    return Result("eager", compile_ms, latency, gpu_compute, gpu_overhead,
                  kernel_count, 0.0, kp)


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
    kp = profile_kernels(fn)
    gpu_compute, gpu_overhead = compute_gpu_kernel_time(kp, "torch.compile")
    kernel_count = sum(k["count"] for k in kp)

    out = _extract_tensor(fn())
    max_diff = (out.float() - eager_ref.float()).abs().max().item()

    return Result("torch.compile", compile_ms, latency, gpu_compute, gpu_overhead,
                  kernel_count, max_diff, kp)


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
    kp = profile_kernels(fn)
    gpu_compute, gpu_overhead = compute_gpu_kernel_time(kp, "megabake")
    kernel_count = sum(k["count"] for k in kp)

    out = fn()
    max_diff = (out.float() - eager_ref.float()).abs().max().item()

    return Result("megabake", compile_ms, latency, gpu_compute, gpu_overhead,
                  kernel_count, max_diff, kp)


BACKENDS = {
    "eager": run_eager,
    "torch.compile": run_torch_compile,
    "megabake": run_megabake,
}


# ---------------------------------------------------------------------------
# Bandwidth utilization
# ---------------------------------------------------------------------------

def _model_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total


def _peak_bandwidth_gb_s() -> float:
    """Estimate peak HBM bandwidth in GB/s from device name."""
    name = torch.cuda.get_device_name(0).lower()
    if "h200" in name:
        return 4800.0 if "mig" not in name else 500.0
    if "h100" in name:
        return 3350.0 if "mig" not in name else 400.0
    if "a100" in name:
        return 2039.0 if "mig" not in name else 300.0
    if "4090" in name:
        return 1008.0
    return 1000.0


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_results(results: list[Result], display_name: str, model: nn.Module,
                  show_profile: bool, seq_len: int, batch_size: int) -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  SMs: {props.multi_processor_count}")
    print(f"Model: {display_name}\n")

    hdr = (f"{'Backend':<16s} {'Compile(ms)':>11s} {'Latency(us)':>11s} "
           f"{'GPU Kern(us)':>12s} {'Host(us)':>9s} {'Kernels':>8s} {'MaxDiff':>10s}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for r in results:
        if r.error:
            print(f"{r.backend:<16s} FAIL: {r.error}")
            continue

        diff_str = "—" if r.backend == "eager" else f"{r.max_diff:.6f}"
        compile_str = "—" if r.backend == "eager" else f"{r.compile_ms:>11.1f}"
        host_us = r.latency_us - r.gpu_compute_us - r.gpu_overhead_us
        print(
            f"{r.backend:<16s} {compile_str:>11s} "
            f"{r.latency_us:>11.1f} "
            f"{r.gpu_compute_us:>12.1f} "
            f"{max(0, host_us):>9.1f} "
            f"{r.kernels:>8d} "
            f"{diff_str:>10s}"
        )
        if r.backend == "megabake" and r.gpu_overhead_us > 0:
            print(f"  {'megakernel':<14s} {'':>11s} {'':>11s} {r.gpu_compute_us:>12.1f}")
            print(f"  {'output clone':<14s} {'':>11s} {'':>11s} {r.gpu_overhead_us:>12.1f}")

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
        if mb.gpu_compute_us > 0 and tc.gpu_compute_us > 0:
            print(f"  GPU kernel only:          {tc.gpu_compute_us / mb.gpu_compute_us:.2f}x  "
                  f"(megakernel {mb.gpu_compute_us:.0f}us vs TC kernels {tc.gpu_compute_us:.0f}us)")
    print()

    # Bandwidth utilization (meaningful for decode: batch=1, small seq_len)
    mbytes = _model_bytes(model)
    if mbytes > 0 and batch_size <= 2 and seq_len <= 4:
        peak_bw = _peak_bandwidth_gb_s()
        print(f"Model weights: {mbytes / 1e6:.1f} MB")
        print(f"Peak HBM bandwidth: {peak_bw:.0f} GB/s")
        floor_us = (mbytes / (peak_bw * 1e9)) * 1e6
        print(f"Theoretical floor (weights/peak_bw): {floor_us:.0f} us")
        print()
        bw_parts = []
        for r in results:
            if r.error or r.latency_us <= 0:
                continue
            achieved = mbytes / (r.latency_us * 1e-6) / 1e9
            util = achieved / peak_bw * 100
            bw_parts.append(f"{r.backend} {util:.1f}%")
        if bw_parts:
            print(f"BW utilization: {', '.join(bw_parts)}")
            print()

    # Per-kernel profile
    if show_profile:
        for r in results:
            if r.error or not r.kernel_profile:
                continue
            total_us = sum(k["duration_us"] for k in r.kernel_profile)
            print(f"  [{r.backend}] Top kernels (total GPU: {total_us:.1f} us):")
            for k in r.kernel_profile[:15]:
                name = k["name"]
                if len(name) > 70:
                    name = name[:67] + "..."
                pct = k["duration_us"] / total_us * 100 if total_us > 0 else 0
                print(f"    {name:<72s} x{k['count']:<4d} {k['duration_us']:>9.1f} us ({pct:>5.1f}%)")
            if len(r.kernel_profile) > 15:
                rest = sum(k["duration_us"] for k in r.kernel_profile[15:])
                rest_count = sum(k["count"] for k in r.kernel_profile[15:])
                print(f"    {'... + ' + str(len(r.kernel_profile) - 15) + ' more':<72s} x{rest_count:<4d} {rest:>9.1f} us")
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
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--mode", choices=["decode", "prefill"],
        help="decode: seq=1,bs=1 (M=1 matmuls). prefill: seq=128,bs=1.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Show per-kernel GPU time breakdown",
    )
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

    # Mode presets (explicit --seq-len/--batch-size override)
    if args.mode == "decode":
        seq_len = args.seq_len if args.seq_len is not None else 1
        batch_size = args.batch_size if args.batch_size is not None else 1
    elif args.mode == "prefill":
        seq_len = args.seq_len if args.seq_len is not None else 128
        batch_size = args.batch_size if args.batch_size is not None else 1
    else:
        seq_len = args.seq_len if args.seq_len is not None else 128
        batch_size = args.batch_size if args.batch_size is not None else 1

    backend_names = [b.strip() for b in args.backends.split(",")]
    for b in backend_names:
        if b not in BACKENDS:
            parser.error(f"unknown backend: {b!r}  (choices: {list(BACKENDS.keys())})")

    print(f"Loading model: {args.model} ...")
    model, inputs, display_name = load_model(args.model, batch_size, seq_len)

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
                f"latency={r.latency_us:.1f}us  kernels={r.kernels}  "
                f"gpu_kern={r.gpu_compute_us:.1f}us"
                if r.error is None else f"FAIL: {r.error}"
            )
        except Exception as e:
            r = Result(bname, -1, -1, -1, -1, -1, -1, error=str(e))
            print(f"FAIL: {e}")
        results.append(r)
        torch.cuda.empty_cache()

    print_results(results, display_name, model, args.profile, seq_len, batch_size)


if __name__ == "__main__":
    main()
