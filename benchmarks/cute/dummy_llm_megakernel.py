"""One cooperative CuTe grid for an entire tiny two-layer cached decode step.

The grid owns Q/K/V, the KV append, stable attention, both residual/MLP paths,
final RMSNorm, and logits. Each replay consumes the next input embedding and
advances its own epoch. This is an execution feasibility probe, not a fast GEMV.
"""

from __future__ import annotations

import argparse
from datetime import date
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import median

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch

from megakernel_probe import cuda_event_single_us, silu, warp_sum


@cute.jit
def grid_barrier(arrivals: cute.Tensor, slot: cutlass.Int32, target: cutlass.Int32):
    """Release/acquire handoff among a resident cooperative grid."""
    cute.arch.sync_threads()
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        counter = arrivals.iterator + slot
        cute.arch.atomic_add(counter, 1, sem="release", scope="gpu")
        observed = cute.arch.atomic_add(counter, 0, sem="acquire", scope="gpu")
        while observed < target:
            observed = cute.arch.atomic_add(counter, 0, sem="acquire", scope="gpu")
    cute.arch.sync_threads()


@cute.kernel
def decode_step_kernel(
    inputs: cute.Tensor,
    norm1: cute.Tensor,
    norm2: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    v_weight: cute.Tensor,
    o_weight: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    down_weight: cute.Tensor,
    final_norm: cute.Tensor,
    lm_head: cute.Tensor,
    cache_k: cute.Tensor,
    cache_v: cute.Tensor,
    state: cute.Tensor,
    scale: cute.Tensor,
    query: cute.Tensor,
    attention: cute.Tensor,
    residual: cute.Tensor,
    hidden: cute.Tensor,
    logits: cute.Tensor,
    arrivals: cute.Tensor,
    epoch: cute.Tensor,
    partial_max: cute.Tensor,
    partial_den: cute.Tensor,
    partial_num: cute.Tensor,
    prefix: cutlass.Int32,
    warp_online_attention: cutlass.Constexpr[bool],
    warp_projections: cutlass.Constexpr[bool],
    attention_splits: cutlass.Constexpr[int],
):
    block, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    grid, _, _ = cute.arch.grid_dim()
    warp = tid // 32
    lane = tid % 32
    warps_per_cta = 4 if warp_projections else 1
    batch = inputs.shape[1]
    workers_per_seq = grid // batch
    seq = block // workers_per_seq
    worker = block % workers_per_seq
    if batch == 1:
        workers_per_seq = grid
        seq = 0
        worker = block
    layers, h = norm1.shape
    intermediate = gate_weight.shape[1] if warp_projections else gate_weight.shape[2]
    vocab = lm_head.shape[0] if warp_projections else lm_head.shape[1]
    head_dim = 32
    step = epoch[0]
    position = prefix + step
    target = (step + 1) * grid
    barrier_stride = 6 if attention_splits > 1 else 5
    barrier_extra = 1 if attention_splits > 1 else 0

    for layer in range(layers):
        # Norm and QKV. Every CTA computes its own scalar RMS scale; each
        # thread owns distinct output columns and one KV cache slice.
        if tid == 0:
            squares = cutlass.Float32(0.0)
            for k in range(h):
                value = cutlass.Float32(0.0)
                if layer == 0:
                    value = cutlass.Float32(inputs[step, seq, k])
                else:
                    value = state[seq, k]
                squares += value * value
            scale[seq, worker] = cute.rsqrt(squares / h + 1.0e-5)
        cute.arch.sync_threads()

        if warp_projections:
            for j in range(worker * 4 + warp, h, workers_per_seq * 4):
                q_sum = cutlass.Float32(0.0)
                k_sum = cutlass.Float32(0.0)
                v_sum = cutlass.Float32(0.0)
                for k in range(lane, h, 32):
                    value = cutlass.Float32(0.0)
                    if layer == 0:
                        value = cutlass.Float32(inputs[step, seq, k])
                    else:
                        value = state[seq, k]
                    value *= scale[seq, worker] * cutlass.Float32(norm1[layer, k])
                    q_sum += value * cutlass.Float32(q_weight[layer, j, k])
                    k_sum += value * cutlass.Float32(k_weight[layer, j, k])
                    v_sum += value * cutlass.Float32(v_weight[layer, j, k])
                q_sum = warp_sum(q_sum)
                k_sum = warp_sum(k_sum)
                v_sum = warp_sum(v_sum)
                if lane == 0:
                    query[seq, j] = q_sum
                    cache_k[layer, seq, position, j] = k_sum
                    cache_v[layer, seq, position, j] = v_sum
        else:
            for j in range(worker * 32 + tid, h, workers_per_seq * 32):
                q_sum = cutlass.Float32(0.0)
                k_sum = cutlass.Float32(0.0)
                v_sum = cutlass.Float32(0.0)
                for k in range(h):
                    value = cutlass.Float32(0.0)
                    if layer == 0:
                        value = cutlass.Float32(inputs[step, seq, k])
                    else:
                        value = state[seq, k]
                    value *= scale[seq, worker] * cutlass.Float32(norm1[layer, k])
                    q_sum += value * cutlass.Float32(q_weight[layer, k, j])
                    k_sum += value * cutlass.Float32(k_weight[layer, k, j])
                    v_sum += value * cutlass.Float32(v_weight[layer, k, j])
                query[seq, j] = q_sum
                cache_k[layer, seq, position, j] = k_sum
                cache_v[layer, seq, position, j] = v_sum
        grid_barrier(arrivals, layer * barrier_stride, target)

        if attention_splits > 1:
            # Parallelize a head over context partitions, then merge stable
            # softmax summaries before the output projection.
            for task in range(worker * warps_per_cta + warp, (h // head_dim) * attention_splits, workers_per_seq * warps_per_cta):
                head = task // attention_splits
                part = task % attention_splits
                index = head * head_dim + lane
                begin = (position + 1) * part // attention_splits
                end = (position + 1) * (part + 1) // attention_splits
                maximum = cutlass.Float32(-1.0e30)
                denominator = cutlass.Float32(0.0)
                numerator = cutlass.Float32(0.0)
                for t in range(begin, end):
                    product = query[seq, index] * cutlass.Float32(cache_k[layer, seq, t, index])
                    score = cute.arch.warp_reduction_sum(product) * (1.0 / math.sqrt(head_dim))
                    next_maximum = maximum
                    if score > maximum:
                        next_maximum = score
                    old_scale = cute.exp(maximum - next_maximum)
                    new_scale = cute.exp(score - next_maximum)
                    denominator = denominator * old_scale + new_scale
                    numerator = numerator * old_scale + new_scale * cutlass.Float32(cache_v[layer, seq, t, index])
                    maximum = next_maximum
                if lane == 0:
                    partial_max[seq, head, part] = maximum
                    partial_den[seq, head, part] = denominator
                partial_num[seq, head, part, lane] = numerator
            grid_barrier(arrivals, layer * barrier_stride + 1, target)
            for head in range(worker * warps_per_cta + warp, h // head_dim, workers_per_seq * warps_per_cta):
                maximum = cutlass.Float32(-1.0e30)
                for part in range(attention_splits):
                    part_max = partial_max[seq, head, part]
                    if part_max > maximum:
                        maximum = part_max
                denominator = cutlass.Float32(0.0)
                numerator = cutlass.Float32(0.0)
                for part in range(attention_splits):
                    factor = cute.exp(partial_max[seq, head, part] - maximum)
                    denominator += partial_den[seq, head, part] * factor
                    numerator += partial_num[seq, head, part, lane] * factor
                attention[seq, head * head_dim + lane] = numerator / denominator
        elif warp_online_attention:
            # One warp owns one head. Lanes form each Q·K score together;
            # online softmax consumes V in a single pass over the cache.
            for head in range(worker * warps_per_cta + warp, h // head_dim, workers_per_seq * warps_per_cta):
                maximum = cutlass.Float32(-1.0e30)
                denominator = cutlass.Float32(0.0)
                numerator = cutlass.Float32(0.0)
                index = head * head_dim + lane
                for t in range(position + 1):
                    product = query[seq, index] * cutlass.Float32(cache_k[layer, seq, t, index])
                    score = cute.arch.warp_reduction_sum(product) * (1.0 / math.sqrt(head_dim))
                    next_maximum = maximum
                    if score > maximum:
                        next_maximum = score
                    old_scale = cute.exp(maximum - next_maximum)
                    new_scale = cute.exp(score - next_maximum)
                    denominator = denominator * old_scale + new_scale
                    numerator = numerator * old_scale + new_scale * cutlass.Float32(cache_v[layer, seq, t, index])
                    maximum = next_maximum
                attention[seq, index] = numerator / denominator
        else:
            # Original scalar-output implementation for matched comparison.
            for j in range(worker * 32 + tid, h, workers_per_seq * 32):
                head = j // head_dim
                maximum = cutlass.Float32(-1.0e30)
                for t in range(position + 1):
                    dot = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        index = head * head_dim + d
                        dot += query[seq, index] * cutlass.Float32(cache_k[layer, seq, t, index])
                    score = dot * (1.0 / math.sqrt(head_dim))
                    if score > maximum:
                        maximum = score
                denominator = cutlass.Float32(0.0)
                numerator = cutlass.Float32(0.0)
                for t in range(position + 1):
                    dot = cutlass.Float32(0.0)
                    for d in range(head_dim):
                        index = head * head_dim + d
                        dot += query[seq, index] * cutlass.Float32(cache_k[layer, seq, t, index])
                    probability = cute.exp(dot * (1.0 / math.sqrt(head_dim)) - maximum)
                    denominator += probability
                    numerator += probability * cutlass.Float32(cache_v[layer, seq, t, j])
                attention[seq, j] = numerator / denominator
        grid_barrier(arrivals, layer * barrier_stride + 1 + barrier_extra, target)

        if warp_projections:
            for j in range(worker * 4 + warp, h, workers_per_seq * 4):
                projection = cutlass.Float32(0.0)
                for k in range(lane, h, 32):
                    projection += attention[seq, k] * cutlass.Float32(o_weight[layer, j, k])
                projection = warp_sum(projection)
                if lane == 0:
                    if layer == 0:
                        residual[seq, j] = cutlass.Float32(inputs[step, seq, j]) + projection
                    else:
                        residual[seq, j] = state[seq, j] + projection
        else:
            for j in range(worker * 32 + tid, h, workers_per_seq * 32):
                projection = cutlass.Float32(0.0)
                for k in range(h):
                    projection += attention[seq, k] * cutlass.Float32(o_weight[layer, k, j])
                if layer == 0:
                    residual[seq, j] = cutlass.Float32(inputs[step, seq, j]) + projection
                else:
                    residual[seq, j] = state[seq, j] + projection
        grid_barrier(arrivals, layer * barrier_stride + 2 + barrier_extra, target)

        if tid == 0:
            squares = cutlass.Float32(0.0)
            for k in range(h):
                squares += residual[seq, k] * residual[seq, k]
            scale[seq, worker] = cute.rsqrt(squares / h + 1.0e-5)
        cute.arch.sync_threads()
        if warp_projections:
            for j in range(worker * 4 + warp, intermediate, workers_per_seq * 4):
                gate_sum = cutlass.Float32(0.0)
                up_sum = cutlass.Float32(0.0)
                for k in range(lane, h, 32):
                    value = residual[seq, k] * scale[seq, worker] * cutlass.Float32(norm2[layer, k])
                    gate_sum += value * cutlass.Float32(gate_weight[layer, j, k])
                    up_sum += value * cutlass.Float32(up_weight[layer, j, k])
                gate_sum = warp_sum(gate_sum)
                up_sum = warp_sum(up_sum)
                if lane == 0:
                    hidden[seq, j] = silu(gate_sum) * up_sum
        else:
            for j in range(worker * 32 + tid, intermediate, workers_per_seq * 32):
                gate_sum = cutlass.Float32(0.0)
                up_sum = cutlass.Float32(0.0)
                for k in range(h):
                    value = residual[seq, k] * scale[seq, worker] * cutlass.Float32(norm2[layer, k])
                    gate_sum += value * cutlass.Float32(gate_weight[layer, k, j])
                    up_sum += value * cutlass.Float32(up_weight[layer, k, j])
                hidden[seq, j] = silu(gate_sum) * up_sum
        grid_barrier(arrivals, layer * barrier_stride + 3 + barrier_extra, target)

        if warp_projections:
            for j in range(worker * 4 + warp, h, workers_per_seq * 4):
                projection = cutlass.Float32(0.0)
                for k in range(lane, intermediate, 32):
                    projection += hidden[seq, k] * cutlass.Float32(down_weight[layer, j, k])
                projection = warp_sum(projection)
                if lane == 0:
                    state[seq, j] = residual[seq, j] + projection
        else:
            for j in range(worker * 32 + tid, h, workers_per_seq * 32):
                projection = cutlass.Float32(0.0)
                for k in range(intermediate):
                    projection += hidden[seq, k] * cutlass.Float32(down_weight[layer, k, j])
                state[seq, j] = residual[seq, j] + projection
        grid_barrier(arrivals, layer * barrier_stride + 4 + barrier_extra, target)

    # The final layer barrier makes state complete for every CTA. A final
    # device-side norm and vocabulary projection finish the full step.
    if tid == 0:
        squares = cutlass.Float32(0.0)
        for k in range(h):
            squares += state[seq, k] * state[seq, k]
        scale[seq, worker] = cute.rsqrt(squares / h + 1.0e-5)
    cute.arch.sync_threads()
    if warp_projections:
        for j in range(worker * 4 + warp, vocab, workers_per_seq * 4):
            projection = cutlass.Float32(0.0)
            for k in range(lane, h, 32):
                value = state[seq, k] * scale[seq, worker] * cutlass.Float32(final_norm[k])
                projection += value * cutlass.Float32(lm_head[j, k])
            projection = warp_sum(projection)
            if lane == 0:
                logits[seq, j] = projection
    else:
        for j in range(worker * 32 + tid, vocab, workers_per_seq * 32):
            projection = cutlass.Float32(0.0)
            for k in range(h):
                value = state[seq, k] * scale[seq, worker] * cutlass.Float32(final_norm[k])
                projection += value * cutlass.Float32(lm_head[k, j])
            logits[seq, j] = projection
    if block == 0 and tid == 0:
        epoch[0] = step + 1


@cute.jit
def decode_step(
    *args, prefix: cutlass.Int32, warp_online_attention: cutlass.Constexpr[bool],
    warp_projections: cutlass.Constexpr[bool], attention_splits: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    decode_step_kernel(*args, prefix, warp_online_attention, warp_projections, attention_splits).launch(
        grid=(args[15].shape[0] * args[15].shape[1], 1, 1),
        block=(128 if warp_projections else 32, 1, 1),
        cooperative=True, stream=stream
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--warp-online-attention", action="store_true")
    parser.add_argument("--warp-projections", action="store_true")
    parser.add_argument("--attention-splits", type=int, default=1)
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument(
        "--compile-mode", choices=("default", "max-autotune-no-cudagraphs"),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument("--output", type=Path, help="write the result as JSON")
    args = parser.parse_args()
    if args.hidden % 32 or min(args.hidden, args.intermediate, args.vocab, args.batch, args.grid, args.attention_splits) <= 0:
        parser.error("hidden must be divisible by 32; dimensions and grid must be positive")
    if args.grid % args.batch:
        parser.error("grid must be divisible by batch")
    torch.manual_seed(0)
    h, i, v, prefix, grid = args.hidden, args.intermediate, args.vocab, args.prefix, args.grid
    batch = args.batch
    layers = 2
    device = "cuda"
    inputs = torch.randn(2, batch, h, device=device) * 0.1
    norm1 = torch.randn(layers, h, device=device) * 0.1 + 1.0
    norm2 = torch.randn(layers, h, device=device) * 0.1 + 1.0
    q_w, k_w, v_w, o_w = (torch.randn(layers, h, h, device=device) * 0.1 for _ in range(4))
    gate_w, up_w = (torch.randn(layers, h, i, device=device) * 0.1 for _ in range(2))
    down_w = torch.randn(layers, i, h, device=device) * 0.1
    final_norm = torch.randn(h, device=device) * 0.1 + 1.0
    lm_head = torch.randn(h, v, device=device) * 0.1
    initial_k = torch.randn(layers, batch, prefix + 2, h, device=device) * 0.1
    initial_v = torch.randn_like(initial_k) * 0.1
    cache_k, cache_v = initial_k.clone(), initial_v.clone()
    reference_k, reference_v = initial_k.clone(), initial_v.clone()
    state = torch.empty(batch, h, device=device)
    scale = torch.empty(batch, grid // batch, device=device)
    query = torch.empty(batch, h, device=device)
    attention = torch.empty(batch, h, device=device)
    residual = torch.empty(batch, h, device=device)
    hidden = torch.empty(batch, i, device=device)
    logits = torch.empty(batch, v, device=device)
    arrivals = torch.zeros(layers * (6 if args.attention_splits > 1 else 5), device=device, dtype=torch.int32)
    epoch = torch.zeros(1, device=device, dtype=torch.int32)
    partial_max = torch.empty(batch, h // 32, args.attention_splits, device=device)
    partial_den = torch.empty_like(partial_max)
    partial_num = torch.empty(batch, h // 32, args.attention_splits, 32, device=device)
    if args.warp_projections:
        cute_weights = tuple(w.transpose(1, 2).contiguous() for w in (q_w, k_w, v_w, o_w, gate_w, up_w, down_w))
        cute_lm_head = lm_head.T.contiguous()
    else:
        cute_weights = (q_w, k_w, v_w, o_w, gate_w, up_w, down_w)
        cute_lm_head = lm_head
    raw = (
        inputs, norm1, norm2, *cute_weights,
        final_norm, cute_lm_head, cache_k, cache_v, state, scale, query, attention,
        residual, hidden, logits, arrivals, epoch, partial_max, partial_den,
        partial_num,
    )
    tensors = tuple(from_dlpack(t) for t in raw)
    compiled = cute.compile(
        decode_step, *tensors, prefix=prefix, warp_online_attention=args.warp_online_attention,
        warp_projections=args.warp_projections, attention_splits=args.attention_splits,
        stream=cutlass_torch.current_stream(),
    )

    def cute_call():
        compiled(*tensors, prefix, cutlass_torch.current_stream())

    def reference(step: int, kv_k: torch.Tensor, kv_v: torch.Tensor):
        x = inputs[step]
        position = prefix + step
        for layer in range(layers):
            normed = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5) * norm1[layer]
            q = normed @ q_w[layer]
            kv_k[layer, :, position].copy_(normed @ k_w[layer])
            kv_v[layer, :, position].copy_(normed @ v_w[layer])
            q = q.reshape(batch, h // 32, 32)
            keys = kv_k[layer, :, :position + 1].reshape(batch, position + 1, h // 32, 32).permute(0, 2, 1, 3)
            values = kv_v[layer, :, :position + 1].reshape(batch, position + 1, h // 32, 32).permute(0, 2, 1, 3)
            scores = (keys * q[:, :, None, :]).sum(-1) / math.sqrt(32)
            probabilities = torch.softmax(scores, dim=-1)
            attended = (probabilities[:, :, :, None] * values).sum(2).reshape(batch, h)
            residual_x = x + attended @ o_w[layer]
            normed = residual_x * torch.rsqrt(residual_x.square().mean(dim=-1, keepdim=True) + 1.0e-5) * norm2[layer]
            x = residual_x + (torch.nn.functional.silu(normed @ gate_w[layer]) * (normed @ up_w[layer])) @ down_w[layer]
        normed = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1.0e-5) * final_norm
        return x, normed @ lm_head

    errors = []
    for step in range(2):
        cute_call()
        expected_state, expected_logits = reference(step, reference_k, reference_v)
        torch.cuda.synchronize()
        errors.append({
            "step": step,
            "state_max_abs": (state - expected_state).abs().max().item(),
            "logits_max_abs": (logits - expected_logits).abs().max().item(),
            "k_max_abs": (cache_k - reference_k).abs().max().item(),
            "v_max_abs": (cache_v - reference_v).abs().max().item(),
        })
        for label, actual, expected in (
            ("state", state, expected_state), ("logits", logits, expected_logits),
            ("K cache", cache_k, reference_k), ("V cache", cache_v, reference_v),
        ):
            if not torch.allclose(actual, expected, atol=1.0e-3, rtol=1.0e-3):
                raise AssertionError(f"step {step} {label} mismatch: {errors[-1]}")
        if epoch.item() != step + 1 or not torch.all(arrivals == (step + 1) * grid):
            raise AssertionError(f"step {step} epoch/barrier counter mismatch")

    def reset_cute():
        cache_k.copy_(initial_k)
        cache_v.copy_(initial_v)
        arrivals.zero_()
        epoch.zero_()

    reset_cute()
    cute_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cute_graph):
        cute_call()
    reset_cute()
    logits.fill_(float("nan"))
    cute_graph.replay()
    graph_state, graph_logits = reference(0, reference_k, reference_v)
    expected_graph_k, expected_graph_v = initial_k.clone(), initial_v.clone()
    reference(0, expected_graph_k, expected_graph_v)
    if not torch.allclose(logits, graph_logits, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured CuTe kernel did not execute")
    if not torch.allclose(state, graph_state, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured CuTe state mismatch")
    if not torch.allclose(cache_k, expected_graph_k, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured CuTe K cache mismatch")
    if not torch.allclose(cache_v, expected_graph_v, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured CuTe V cache mismatch")

    # Keep a separate reference cache so both captured graphs own their state.
    baseline_k, baseline_v = initial_k.clone(), initial_v.clone()

    def baseline():
        return reference(0, baseline_k, baseline_v)

    torch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_graph):
        baseline_state, baseline_logits = baseline()
    torch_graph.replay()
    if not torch.allclose(baseline_logits, graph_logits, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured PyTorch reference mismatch")
    if not torch.allclose(baseline_state, graph_state, atol=1.0e-3, rtol=1.0e-3):
        raise AssertionError("captured PyTorch state mismatch")
    if not torch.equal(baseline_k, expected_graph_k) or not torch.equal(baseline_v, expected_graph_v):
        raise AssertionError("captured PyTorch KV cache mismatch")

    compiled_graph = None
    if args.compile_baseline:
        compiled_k, compiled_v = initial_k.clone(), initial_v.clone()

        def compiled_reference():
            return reference(0, compiled_k, compiled_v)

        compiled_reference = torch.compile(
            compiled_reference, mode=args.compile_mode, fullgraph=True
        )
        compiled_reference()
        compiled_k.copy_(initial_k)
        compiled_v.copy_(initial_v)
        torch.cuda.synchronize()
        compiled_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(compiled_graph):
            compiled_state, compiled_logits = compiled_reference()
        compiled_k.copy_(initial_k)
        compiled_v.copy_(initial_v)
        compiled_graph.replay()
        torch.cuda.synchronize()
        if not torch.allclose(compiled_logits, graph_logits, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured compiled PyTorch logits mismatch")
        if not torch.allclose(compiled_state, graph_state, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured compiled PyTorch state mismatch")
        if not torch.allclose(compiled_k, expected_graph_k, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured compiled PyTorch K cache mismatch")
        if not torch.allclose(compiled_v, expected_graph_v, atol=1.0e-3, rtol=1.0e-3):
            raise AssertionError("captured compiled PyTorch V cache mismatch")

    result = {
        "date_utc": date.today().isoformat(),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "shape": {"layers": layers, "batch": batch, "hidden": h, "intermediate": i, "heads": h // 32, "vocab": v, "prefix": prefix, "grid_ctas": grid},
        "dtype": "float32",
        "attention_math": (
            "split_online_softmax" if args.attention_splits > 1 else
            "warp_online_softmax" if args.warp_online_attention else
            "scalar_two_pass_softmax"
        ),
        "attention_splits": args.attention_splits,
        "projection_math": "prepacked_warp_reduction" if args.warp_projections else "scalar_output",
        "barriers_per_step": layers * (6 if args.attention_splits > 1 else 5),
        "cute_kernel_launches_per_step": 1,
        "two_advancing_steps_correct": True,
        "errors": errors,
        "cute_cudagraph_fixed_state_us": cuda_event_single_us(cute_graph.replay, reset_cute),
        "torch_cudagraph_fixed_state_us": cuda_event_single_us(torch_graph.replay, lambda: (baseline_k.copy_(initial_k), baseline_v.copy_(initial_v))),
    }
    if compiled_graph is not None:
        result["torch_compile_mode"] = f"{args.compile_mode}, externally captured"
        result["torch_compile_graph_fixed_state_us"] = cuda_event_single_us(
            compiled_graph.replay,
            lambda: (compiled_k.copy_(initial_k), compiled_v.copy_(initial_v)),
        )
        samples = {"cute": [], "torch_compile": []}
        workloads = {
            "cute": (cute_graph.replay, reset_cute),
            "torch_compile": (
                compiled_graph.replay,
                lambda: (compiled_k.copy_(initial_k), compiled_v.copy_(initial_v)),
            ),
        }
        for trial in range(30):
            order = ("cute", "torch_compile") if trial % 2 == 0 else ("torch_compile", "cute")
            for name in order:
                run, prepare = workloads[name]
                prepare()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0)
        result["paired_fixed_state_us"] = {name: median(values) for name, values in samples.items()}
        result["paired_samples_us"] = samples
    reset_cute()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        cute_graph.replay()
        torch.cuda.synchronize()
    result["profiled_cuda_kernels_per_replay"] = [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    if len(result["profiled_cuda_kernels_per_replay"]) != 1:
        raise AssertionError(f"expected one GPU kernel per step: {result['profiled_cuda_kernels_per_replay']}")
    baseline_k.copy_(initial_k)
    baseline_v.copy_(initial_v)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        torch_graph.replay()
        torch.cuda.synchronize()
    result["profiled_torch_cuda_kernel_count_per_replay"] = sum(
        event.device_type == torch.autograd.DeviceType.CUDA for event in prof.events()
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
