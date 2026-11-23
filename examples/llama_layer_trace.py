"""One Llama decoder layer, traced through Inductor and through MegaBake.

The model follows hkproj/pytorch-llama ``model.py`` (``ModelArgs``,
``RMSNorm``, ``repeat_kv``, ``SelfAttention``, ``FeedForward``,
``EncoderBlock``), reduced to one block. Three changes make it compile with
MegaBake:

* Scores are scaled with ``* (1 / sqrt(head_dim))``, not ``/ sqrt(head_dim)``.
  The eager-attention matcher (``ops/attention.py:_score_chain``) only
  recognizes a ``mul`` scale, as Hugging Face writes it. With a ``div`` the
  attention falls to generic bodies, which currently fail at runtime binding
  on the decomposed softmax's tuple outputs (try ``--div-scale``).

* RoPE uses real cos/sin tables and ``rotate_half`` instead of
  ``torch.view_as_complex`` / ``torch.polar``. MegaBake has no complex dtype
  lowering, so the complex form stops at the lowering stage
  (try ``--complex-rope`` to see that diagnostic).
* No KV cache: this is a prefill forward over ``seq_len`` tokens with a causal
  mask. hkproj's model is decode-only (``seq_len == 1``) and writes its cache
  in place. MegaBake's decode path declares that state through
  ``megabake.kv_attention`` (see ``tests/fixtures.py:StatefulTransformerBlock``).

Run on an SM90 GPU:

    .venv/bin/python examples/llama_layer_trace.py
    .venv/bin/python examples/llama_layer_trace.py --out traces/llama_layer --seq-len 16

Everything it writes goes under ``--out``:

    inductor/dynamo_graph.txt      FX graph Dynamo hands to the backend
    inductor/post_grad_graph.txt   graph after AOT + Inductor's post-grad passes
    inductor/output_code_*.py      Triton/Python Inductor generated
    megabake/fx/*.txt              the same stages as MegaBake sees them
    megabake/compute_graph.txt     ComputeGraph (graph.py: import_fx)
    megabake/lowering_graph.txt    tensor program per op (lowering.py)
    megabake/region_graph.txt      regions + fusion candidates (regions.py)
    megabake/plan.txt              selected bodies, phases, barriers, workers
    megabake/fx/generated/*/module.py   the generated CuTe megakernel
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import DeviceType

import megabake
from megabake.graph import dump_compute_graph, import_fx
from megabake.lowering import canonicalize_semantics, dump_lowering_graph
from megabake.regions import dump_region_graph, form_regions

# --------------------------------------------------------------------------
# Model (hkproj/pytorch-llama structure, one layer)
# --------------------------------------------------------------------------


@dataclass
class ModelArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int | None = 4
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device, theta: float = 10000.0):
    """Real-valued version of hkproj's ``freqs_complex``: cos and sin of
    ``m * theta_i``, each repeated over both halves of the head dimension."""
    assert head_dim % 2 == 0
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta_i = 1.0 / (theta ** (theta_numerator / head_dim))
    m = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(m, theta_i)  # (seq_len, head_dim / 2)
    freqs = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_embeddings(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (B, seq_len, H, head_dim); cos/sin: (seq_len, head_dim)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (x * cos + rotate_half(x) * sin).type_as(x)


def apply_rotary_embeddings_complex(x: torch.Tensor, freqs_complex: torch.Tensor):
    """hkproj's original form, kept to show where MegaBake stops."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    return torch.view_as_real(x_rotated).reshape(*x.shape).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs, complex_rope: bool = False, div_scale: bool = False):
        super().__init__()
        self.div_scale = div_scale
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.complex_rope = complex_rope

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, rope, mask):
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # if self.complex_rope:
        #     xq = apply_rotary_embeddings_complex(xq, rope)
        #     xk = apply_rotary_embeddings_complex(xk, rope)
        # else:
        #     xq = apply_rotary_embeddings(xq, *rope)
        #     xk = apply_rotary_embeddings(xk, *rope)

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # (B, H_q, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3))
        if self.div_scale:
            scores = scores / math.sqrt(self.head_dim)  # hkproj; MegaBake's matcher misses it
        else:
            scores = scores * (1.0 / math.sqrt(self.head_dim))
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (B, H_q, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        return self.w2(swish * x_V)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs, complex_rope: bool = False, div_scale: bool = False):
        super().__init__()
        self.attention = SelfAttention(args, complex_rope, div_scale)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, rope, mask):
        h = x + self.attention.forward(self.attention_norm(x), rope, mask)
        return h 


# --------------------------------------------------------------------------
# Tracing helpers
# --------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _readable(gm: torch.fx.GraphModule) -> str:
    return gm.print_readable(print_output=False, include_stride=True, include_device=True)


def _cuda_kernel_count(fn) -> int:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for event in prof.events() if event.device_type == DeviceType.CUDA)


def _time_us(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def trace_inductor(model, inputs, out: Path):
    """``torch.compile`` default backend, with the Dynamo graph and the
    post-grad graph recorded on the way in."""
    from torch._inductor import config as inductor_config
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.utils import run_and_get_code

    def post_grad_dump(graph: torch.fx.Graph) -> None:
        _write(out / "post_grad_graph.txt", str(graph))

    def recording_inductor(gm: torch.fx.GraphModule, example_inputs):
        _write(out / "dynamo_graph.txt", _readable(gm))
        return compile_fx(gm, example_inputs)  # what backend="inductor" calls

    torch._dynamo.reset()
    with inductor_config.patch(post_grad_custom_post_pass=post_grad_dump, fx_graph_cache=False):
        compiled = torch.compile(model, backend=recording_inductor, fullgraph=True, dynamic=False)
        with torch.no_grad():
            output, sources = run_and_get_code(compiled, *inputs)
    for index, source in enumerate(sources):
        _write(out / f"output_code_{index}.py", source)
    return compiled, output


def trace_megabake(model, inputs, out: Path):
    """``torch.compile`` with the MegaBake backend. ``dump_dir`` makes the
    frontend write each FX stage; the ComputeGraph, lowering and region dumps
    rerun the stages ``compile_post_grad`` runs, on the same post-grad graph."""
    torch._dynamo.reset()
    backend = megabake.make_backend(
        mode="decode",
        compile_kernel=True,
        cache=False,
        dump_dir=out / "fx",
        model_name="llama_layer",
        scheduler="phase",
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    try:
        with torch.no_grad():
            output = compiled(*inputs)
    finally:
        # Dumped even when compilation fails, so the failing stage is visible.
        prepared = getattr(backend, "megabake_last_prepared", None)
        if prepared is not None:
            graph = import_fx(prepared.graph, prepared.context)
            dump_compute_graph(graph, out / "compute_graph.txt")
            lowered = canonicalize_semantics(graph)
            dump_lowering_graph(lowered, out / "lowering_graph.txt")
            dump_region_graph(form_regions(lowered), out / "region_graph.txt")

    kernel = backend.megabake_last_compiled
    plan = kernel.plan
    lines = [
        f"generated module: {kernel.generated.source_path}",
        f"grid: {plan.launch.grid_size} CTAs x {kernel.generated.launch.launch_threads} threads, "
        f"{kernel.generated.launch.shared_memory_bytes} B shared, "
        f"{kernel.resources.registers_per_thread} regs/thread",
        f"workspace: {plan.storage.workspace_bytes} B",
        f"phases: {len(plan.phases)}, grid barriers: {sum(p.barrier is not None for p in plan.phases)}",
        "",
        f"{'#':>3}  {'provider/body':<44} {'tiles':>5} {'barrier':>7} {'recompute':>9}  region",
    ]
    for phase in plan.phases:
        lines.append(
            f"{phase.index:>3}  {phase.body.provider + '/' + phase.body.name:<44} "
            f"{len(phase.work_items):>5} {'yes' if phase.barrier else '':>7} "
            f"{'yes' if phase.recompute else '':>9}  {phase.region_id}"
        )
    lines += ["", "region -> source ops:"]
    for selected in plan.selection.selections:
        ops = ", ".join(op.removeprefix("op:") for op in selected.source_op_ids)
        lines.append(f"  {selected.region_id}: {ops}")
    _write(out / "plan.txt", "\n".join(lines) + "\n")
    return compiled, output, backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("fx_traces/llama_layer"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--complex-rope", action="store_true", help="hkproj's complex RoPE")
    parser.add_argument("--div-scale", action="store_true", help="hkproj's scores / sqrt(d)")
    parser.add_argument("--skip-inductor", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    # Inductor's post-grad pattern matcher rewrites matmul-softmax-matmul into
    # scaled_dot_product_attention; its cuDNN backend needs runtime
    # sublibraries some installs lack. Flash/efficient SDPA need nothing extra.
    torch.backends.cuda.enable_cudnn_sdp(False)
    model_args = ModelArgs()
    model = EncoderBlock(model_args, args.complex_rope, args.div_scale).to(device, torch.bfloat16).eval()
    # for norm in (model.attention_norm, model.ffn_norm):
    #     nn.init.normal_(norm.weight, 1.0, 0.1)

    head_dim = model_args.dim // model_args.n_heads
    cos, sin = precompute_theta_pos_frequencies(head_dim, args.seq_len, device)
    if args.complex_rope:
        half = head_dim // 2
        rope = torch.polar(torch.ones_like(cos[:, :half]), torch.atan2(sin[:, :half], cos[:, :half]))
    else:
        rope = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    mask = torch.full((args.seq_len, args.seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1).to(torch.bfloat16)[None, None]  # (1, 1, S, S)
    x = torch.randn(args.batch, args.seq_len, model_args.dim, device=device, dtype=torch.bfloat16)
    inputs = (x, rope, mask)

    with torch.no_grad():
        expected = model(*inputs)

    print(f"writing traces to {args.out}/")
    if not args.skip_inductor:
        inductor_fn, inductor_out = trace_inductor(model, inputs, args.out / "inductor")
        print(f"inductor  rel-L2 vs eager: {_rel_l2(inductor_out, expected):.2e}")

    megabake_fn, megabake_out, _ = trace_megabake(model, inputs, args.out / "megabake")
    print(f"megabake  rel-L2 vs eager: {_rel_l2(megabake_out, expected):.2e}")
    print((args.out / "megabake" / "plan.txt").read_text())

    with torch.no_grad():
        rows = [("eager", lambda: model(*inputs)), ("megabake", lambda: megabake_fn(*inputs))]
        if not args.skip_inductor:
            rows.insert(1, ("inductor", lambda: inductor_fn(*inputs)))
        print(f"{'backend':<10} {'CUDA kernels':>12} {'us/call':>9}")
        for name, fn in rows:
            print(f"{name:<10} {_cuda_kernel_count(fn):>12} {_time_us(fn):>9.1f}")


if __name__ == "__main__":
    main()
