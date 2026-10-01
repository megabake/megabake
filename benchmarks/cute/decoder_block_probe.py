"""One-CTA CuTe DSL decoder-block composition probe with cached attention."""

from __future__ import annotations

import argparse
from datetime import date
import importlib.metadata
import json
import math

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch

from megakernel_probe import cuda_event_us, silu


@cute.kernel
def decoder_block_kernel(
    x: cute.Tensor,
    norm1_weight: cute.Tensor,
    norm2_weight: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    v_weight: cute.Tensor,
    o_weight: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    down_weight: cute.Tensor,
    cache_k: cute.Tensor,
    cache_v: cute.Tensor,
    scale: cute.Tensor,
    q: cute.Tensor,
    attention: cute.Tensor,
    residual: cute.Tensor,
    hidden: cute.Tensor,
    out: cute.Tensor,
    heads: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    h = x.shape[1]
    intermediate = gate_weight.shape[1]
    context = cache_k.shape[0] - 1
    head_dim = h // heads

    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[0, k])
            squares += value * value
        scale[0] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()

    for j in range(tid, h, 128):
        q_sum = cutlass.Float32(0.0)
        k_sum = cutlass.Float32(0.0)
        v_sum = cutlass.Float32(0.0)
        for k in range(h):
            value = cutlass.Float32(x[0, k]) * scale[0] * cutlass.Float32(norm1_weight[k])
            q_sum += value * cutlass.Float32(q_weight[k, j])
            k_sum += value * cutlass.Float32(k_weight[k, j])
            v_sum += value * cutlass.Float32(v_weight[k, j])
        q[j] = q_sum
        cache_k[context, j] = k_sum
        cache_v[context, j] = v_sum
    cute.arch.sync_threads()

    for j in range(tid, h, 128):
        head = j // head_dim
        denominator = cutlass.Float32(0.0)
        numerator = cutlass.Float32(0.0)
        for t in range(context + 1):
            dot = cutlass.Float32(0.0)
            for d in range(head_dim):
                index = head * head_dim + d
                dot += q[index] * cutlass.Float32(cache_k[t, index])
            probability = cute.exp(dot * (1.0 / math.sqrt(head_dim)))
            denominator += probability
            numerator += probability * cutlass.Float32(cache_v[t, j])
        attention[j] = numerator / denominator
    cute.arch.sync_threads()

    for j in range(tid, h, 128):
        projection = cutlass.Float32(0.0)
        for k in range(h):
            projection += attention[k] * cutlass.Float32(o_weight[k, j])
        residual[j] = cutlass.Float32(x[0, j]) + projection
    cute.arch.sync_threads()

    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            value = residual[k]
            squares += value * value
        scale[1] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()

    for j in range(tid, intermediate, 128):
        gate_sum = cutlass.Float32(0.0)
        up_sum = cutlass.Float32(0.0)
        for k in range(h):
            value = residual[k] * scale[1] * cutlass.Float32(norm2_weight[k])
            gate_sum += value * cutlass.Float32(gate_weight[k, j])
            up_sum += value * cutlass.Float32(up_weight[k, j])
        hidden[j] = silu(gate_sum) * up_sum
    cute.arch.sync_threads()

    for j in range(tid, h, 128):
        projection = cutlass.Float32(0.0)
        for k in range(intermediate):
            projection += hidden[k] * cutlass.Float32(down_weight[k, j])
        out[0, j] = residual[j] + projection


@cute.jit
def decoder_block(*args, heads: cutlass.Constexpr[int], stream: cuda.CUstream):
    decoder_block_kernel(*args, heads).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--context", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(0)
    h, intermediate, heads, context = args.hidden, args.intermediate, args.heads, args.context
    if h % heads:
        parser.error("hidden must be divisible by heads")
    device = "cuda"
    x = torch.randn(1, h, device=device) * 0.1
    norm1 = torch.randn(h, device=device) * 0.1 + 1.0
    norm2 = torch.randn(h, device=device) * 0.1 + 1.0
    q_w, k_w, v_w, o_w = (torch.randn(h, h, device=device) * 0.1 for _ in range(4))
    gate_w, up_w = (torch.randn(h, intermediate, device=device) * 0.1 for _ in range(2))
    down_w = torch.randn(intermediate, h, device=device) * 0.1
    cache_k = torch.randn(context + 1, h, device=device) * 0.1
    cache_v = torch.randn_like(cache_k) * 0.1
    reference_k, reference_v = cache_k.clone(), cache_v.clone()
    scale = torch.empty(2, device=device)
    q = torch.empty(h, device=device)
    attention = torch.empty(h, device=device)
    residual = torch.empty(h, device=device)
    hidden = torch.empty(intermediate, device=device)
    out = torch.empty_like(x)
    tensors = tuple(from_dlpack(t) for t in (
        x, norm1, norm2, q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        cache_k, cache_v, scale, q, attention, residual, hidden, out,
    ))
    compiled = cute.compile(decoder_block, *tensors, heads=heads, stream=cutlass_torch.current_stream())

    def reference():
        first = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5) * norm1
        query = first @ q_w
        reference_k[-1].copy_((first @ k_w)[0])
        reference_v[-1].copy_((first @ v_w)[0])
        query = query.reshape(heads, h // heads)
        keys = reference_k.reshape(context + 1, heads, h // heads).permute(1, 0, 2)
        values = reference_v.reshape(context + 1, heads, h // heads).permute(1, 0, 2)
        scores = (keys * query[:, None, :]).sum(-1) / math.sqrt(h // heads)
        probabilities = torch.softmax(scores, dim=-1)
        attended = (probabilities[:, :, None] * values).sum(1).reshape(1, h)
        resid = x + attended @ o_w
        second = resid * torch.rsqrt(resid.square().mean(dim=-1, keepdim=True) + 1.0e-5) * norm2
        return resid + (torch.nn.functional.silu(second @ gate_w) * (second @ up_w)) @ down_w

    def cute_call():
        compiled(*tensors, stream=cutlass_torch.current_stream())

    cute_call()
    expected = reference()
    torch.cuda.synchronize()
    max_abs = (out - expected).abs().max().item()
    if not torch.allclose(out, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError(f"decoder block output mismatch: {max_abs}")
    if not torch.allclose(cache_k[-1], reference_k[-1], atol=1.0e-4, rtol=1.0e-4):
        raise AssertionError("K cache update mismatch")
    if not torch.allclose(cache_v[-1], reference_v[-1], atol=1.0e-4, rtol=1.0e-4):
        raise AssertionError("V cache update mismatch")

    cute_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cute_graph):
        cute_call()
    out.fill_(float("nan"))
    cute_graph.replay()
    if not torch.allclose(out, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured decoder block did not execute")
    torch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_graph):
        graphed_reference = reference()
    torch_graph.replay()
    if not torch.allclose(graphed_reference, expected, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured PyTorch reference mismatch")
    print(json.dumps({
        "date_utc": date.today().isoformat(),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"hidden": h, "intermediate": intermediate, "heads": heads, "context": context},
        "dtype": "float32",
        "max_abs": max_abs,
        "cute_cudagraph_us": cuda_event_us(cute_graph.replay),
        "torch_cudagraph_us": cuda_event_us(torch_graph.replay),
    }, indent=2))


if __name__ == "__main__":
    main()
