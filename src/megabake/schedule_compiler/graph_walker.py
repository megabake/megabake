"""Walk FX graph from torch.export, map ATen ops to megakernel tasks."""

from dataclasses import dataclass, field
import math
import operator
import struct as _struct
import torch
from torch.export import export

from megabake.data_types import TaskDesc, OpType, ElemCode, ReduceCode, UopCode, UNUSED_BUFFER, pack_uop
from megabake.schedule_compiler.op_table import ATEN_OP_MAP
from megabake.schedule_compiler.shape_ops import (
    StridedView, contiguous_strides,
    resolve_reshape, resolve_transpose, resolve_permute,
    resolve_expand, resolve_squeeze, resolve_unsqueeze,
    resolve_slice, resolve_t,
)
from megabake.schedule_compiler.buffer_planner import plan_buffers
from megabake.schedule_compiler.tiling import compute_tiles
from megabake.schedule_compiler.serializer import write_schedule
from megabake.schedule_compiler.inductor_passes import optimize_graph


@dataclass
class CompiledModel:
    schedule_bytes: bytes
    input_buffer_ids: list[int] = field(default_factory=list)
    output_buffer_id: int = -1
    output_shape: list[int] = field(default_factory=list)
    num_buffers: int = 0
    folded_constants: dict[int, torch.Tensor] = field(default_factory=dict)


_numel = math.prod

_IDENTITY_VALS = {ElemCode.MUL: 1.0, ElemCode.DIV: 1.0,
                  ElemCode.ADD: 0.0, ElemCode.SUB: 0.0}


def _dtype_bytes(dt: torch.dtype) -> int:
    return {
        torch.float16: 2, torch.bfloat16: 2, torch.float32: 4,
        torch.float8_e4m3fn: 1, torch.int8: 1, torch.int32: 4,
        torch.int64: 8, torch.bool: 1,
    }.get(dt, 2)


def _get_param_name(ep, node) -> str | None:
    sig = ep.graph_signature
    if hasattr(sig, "inputs_to_parameters"):
        name = sig.inputs_to_parameters.get(node.name)
        if name:
            return name
    if hasattr(sig, "inputs_to_buffers"):
        name = sig.inputs_to_buffers.get(node.name)
        if name:
            return name
    return None


def _resolve_shape_op(target, view: StridedView, args) -> StridedView | None:
    if target in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
        return resolve_reshape(view, list(args[1]))
    elif target == torch.ops.aten.transpose.int:
        return resolve_transpose(view, args[1], args[2])
    elif target == torch.ops.aten.permute.default:
        return resolve_permute(view, list(args[1]))
    elif target == torch.ops.aten.expand.default:
        return resolve_expand(view, list(args[1]))
    elif target == torch.ops.aten.t.default:
        return resolve_t(view)
    elif target == torch.ops.aten.unsqueeze.default:
        return resolve_unsqueeze(view, args[1])
    elif target in (torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim):
        dim = args[1] if len(args) > 1 else None
        return resolve_squeeze(view, dim)
    elif target == torch.ops.aten.slice.Tensor:
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else view.shape[dim]
        return resolve_slice(view, dim, start, end)
    elif target in (torch.ops.aten.split.Tensor, torch.ops.aten.unbind.int):
        return view
    return view


def _extract_dimensions(op_type: int, node, out_shape: list[int]) -> list[int]:
    dims = [0] * 8
    if op_type == OpType.MATMUL:
        args = node.args
        target = node.target
        # addmm has (bias, input, weight) args; mm has (input, weight)
        if target == torch.ops.aten.addmm.default:
            a_meta = args[1].meta.get("val") if hasattr(args[1], "meta") else None
            b_meta = args[2].meta.get("val") if hasattr(args[2], "meta") else None
        else:
            a_meta = args[0].meta.get("val") if hasattr(args[0], "meta") else None
            b_meta = args[1].meta.get("val") if hasattr(args[1], "meta") else None
        if a_meta is not None and b_meta is not None:
            a_shape = list(a_meta.shape)
            b_shape = list(b_meta.shape)
            if len(a_shape) >= 2:
                dims[0] = a_shape[-2]
            else:
                dims[0] = 1
            dims[1] = b_shape[-1]
            dims[2] = a_shape[-1]
            if len(a_shape) > 2:
                dims[3] = _numel(a_shape[:-2])
    elif op_type == OpType.ATTENTION:
        q_meta = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
        if q_meta is not None:
            q_shape = list(q_meta.shape)
            dims[0] = q_shape[0]
            dims[1] = q_shape[1]
            dims[2] = q_shape[2]
            dims[3] = q_shape[3]
            k_meta = node.args[1].meta.get("val")
            if k_meta is not None:
                k_shape = list(k_meta.shape)
                dims[4] = k_shape[2]
                dims[5] = k_shape[1]
    elif op_type == OpType.ELEMENTWISE:
        dims[0] = _numel(out_shape)
    elif op_type == OpType.REDUCE:
        in_meta = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
        if in_meta is not None:
            in_shape = list(in_meta.shape)
            dims[1] = in_shape[-1] if len(in_shape) > 1 else _numel(in_shape)
            dims[0] = _numel(in_shape) // dims[1] if dims[1] > 0 else 1
    elif op_type == OpType.EMBEDDING:
        dims[0] = _numel(out_shape) // out_shape[-1] if len(out_shape) > 0 else 1
        dims[1] = out_shape[-1] if len(out_shape) > 0 else 0
        table_meta = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
        if table_meta is not None:
            dims[2] = list(table_meta.shape)[0]
    elif op_type == OpType.COPY:
        dims[0] = _numel(out_shape)
    elif op_type == OpType.ROPE:
        if len(out_shape) >= 4:
            dims[0] = out_shape[0]
            dims[1] = out_shape[1]
            dims[2] = out_shape[2]
            dims[3] = out_shape[3]
    return dims


_EPILOGUE_FUSE = {
    ElemCode.SILU: OpType.MATMUL_SILU,
    ElemCode.GELU: OpType.MATMUL_GELU,
    ElemCode.GELU_TANH: OpType.MATMUL_GELU_TANH,
}


def _read_counts(tasks: list[TaskDesc]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for t in tasks:
        for buf in t.buffer_indices[1:]:
            if buf != UNUSED_BUFFER:
                counts[buf] = counts.get(buf, 0) + 1
    return counts


def _fuse_tasks(
    tasks: list[TaskDesc], buffer_sizes: dict[int, int]
) -> list[TaskDesc]:
    """Fuse adjacent MATMUL + unary ELEMENTWISE into a single fused op."""
    read_counts = _read_counts(tasks)

    fused: list[TaskDesc] = []
    skip = False
    for i, task in enumerate(tasks):
        if skip:
            skip = False
            continue

        if (
            task.op_type == OpType.MATMUL
            and i + 1 < len(tasks)
            and tasks[i + 1].op_type == OpType.ELEMENTWISE
            and tasks[i + 1].op_code in _EPILOGUE_FUSE
        ):
            nxt = tasks[i + 1]
            mid_buf = task.buffer_indices[0]
            if nxt.buffer_indices[1] == mid_buf and read_counts.get(mid_buf, 0) == 1:
                task.op_type = _EPILOGUE_FUSE[nxt.op_code]
                task.buffer_indices[0] = nxt.buffer_indices[0]
                buffer_sizes.pop(mid_buf, None)
                skip = True

        fused.append(task)

    return fused


_FUSABLE_UNARY = frozenset({
    ElemCode.SILU, ElemCode.GELU, ElemCode.GELU_TANH, ElemCode.RELU,
    ElemCode.SIGMOID, ElemCode.TANH, ElemCode.EXP, ElemCode.LOG,
    ElemCode.RSQRT, ElemCode.NEG, ElemCode.ABS,
})

_FUSABLE_BINARY = frozenset({
    ElemCode.ADD, ElemCode.MUL, ElemCode.SUB, ElemCode.DIV,
})

_ELEM_TO_UOP = {
    ElemCode.ADD: UopCode.ADD, ElemCode.MUL: UopCode.MUL,
    ElemCode.SUB: UopCode.SUB, ElemCode.DIV: UopCode.DIV,
    ElemCode.SILU: UopCode.SILU, ElemCode.GELU: UopCode.GELU,
    ElemCode.RELU: UopCode.RELU, ElemCode.SIGMOID: UopCode.SIGMOID,
    ElemCode.TANH: UopCode.TANH, ElemCode.EXP: UopCode.EXP,
    ElemCode.LOG: UopCode.LOG, ElemCode.RSQRT: UopCode.RSQRT,
    ElemCode.NEG: UopCode.NEG, ElemCode.ABS: UopCode.ABS,
    ElemCode.GELU_TANH: UopCode.GELU_TANH,
}


def _is_fusable_elem(task: TaskDesc) -> bool:
    if task.op_type != OpType.ELEMENTWISE:
        return False
    if task.op_code in _FUSABLE_UNARY:
        return True
    if task.op_code in _FUSABLE_BINARY:
        if task.buffer_indices[2] == UNUSED_BUFFER:
            return False
        if task.dimensions[2] != 0 or task.dimensions[3] != 0:
            return False
        return True
    return False


def _fuse_elementwise_chains(
    tasks: list[TaskDesc], buffer_sizes: dict[int, int], sm_version: int,
    num_sms: int = 0,
) -> list[TaskDesc]:
    """Fuse consecutive same-numel ELEMENTWISE tasks into a single FUSED_ELEMENTWISE
    with a micro-op program interpreted on-GPU."""
    read_counts = _read_counts(tasks)

    chains: list[list[int]] = []
    i = 0
    while i < len(tasks):
        if _is_fusable_elem(tasks[i]):
            chain = [i]
            j = i + 1
            while j < len(tasks) and _is_fusable_elem(tasks[j]):
                prev_out = tasks[chain[-1]].buffer_indices[0]
                curr = tasks[j]
                if (
                    prev_out in curr.buffer_indices[1:8]
                    and read_counts.get(prev_out, 0) == 1
                    and curr.dimensions[0] == tasks[chain[-1]].dimensions[0]
                ):
                    chain.append(j)
                    j += 1
                else:
                    break
            if len(chain) >= 2:
                chains.append(chain)
            i = j
        else:
            i += 1

    if not chains:
        return tasks

    skip_indices: set[int] = set()
    fused_at: dict[int, TaskDesc] = {}

    for chain in chains:
        chain_tasks = [tasks[idx] for idx in chain]
        numel = chain_tasks[0].dimensions[0]

        intermediate_bufs = {tasks[idx].buffer_indices[0] for idx in chain[:-1]}
        final_output = chain_tasks[-1].buffer_indices[0]

        buffer_slots = [final_output]
        buf_to_slot: dict[int, int] = {}
        buf_to_reg: dict[int, int] = {}
        uops: list[int] = []
        next_reg = 0

        def _ensure_loaded(buf: int) -> int:
            nonlocal next_reg
            if buf in buf_to_reg:
                return buf_to_reg[buf]
            if buf not in buf_to_slot:
                buf_to_slot[buf] = len(buffer_slots)
                buffer_slots.append(buf)
            reg = next_reg; next_reg += 1
            uops.append(pack_uop(UopCode.LOAD, dst=reg, src1=buf_to_slot[buf]))
            buf_to_reg[buf] = reg
            return reg

        for ct in chain_tasks:
            s1_reg = _ensure_loaded(ct.buffer_indices[1])

            if ct.op_code in _FUSABLE_BINARY:
                s2_reg = _ensure_loaded(ct.buffer_indices[2])
                dst_reg = next_reg; next_reg += 1
                uops.append(pack_uop(_ELEM_TO_UOP[ct.op_code],
                                     dst=dst_reg, src1=s1_reg, src2=s2_reg))
            else:
                dst_reg = next_reg; next_reg += 1
                uops.append(pack_uop(_ELEM_TO_UOP[ct.op_code],
                                     dst=dst_reg, src1=s1_reg))

            buf_to_reg[ct.buffer_indices[0]] = dst_reg

        uops.append(pack_uop(UopCode.STORE, dst=0,
                              src1=buf_to_reg[final_output]))

        if len(uops) > 8 or len(buffer_slots) > 8 or next_reg > 8:
            continue

        buf_indices = buffer_slots + [UNUSED_BUFFER] * (8 - len(buffer_slots))
        dims = [numel, len(uops)] + [0] * 6
        strides = uops + [0] * (8 - len(uops))

        fused_task = TaskDesc(
            op_type=OpType.FUSED_ELEMENTWISE,
            op_code=0,
            num_tiles=compute_tiles(OpType.FUSED_ELEMENTWISE, dims, sm_version, num_sms),
            buffer_indices=buf_indices,
            dimensions=dims,
            strides=strides,
        )

        fused_at[chain[0]] = fused_task
        for idx in chain[1:]:
            skip_indices.add(idx)
        for buf in intermediate_bufs:
            buffer_sizes.pop(buf, None)

    result = []
    for i, task in enumerate(tasks):
        if i in skip_indices:
            continue
        if i in fused_at:
            result.append(fused_at[i])
        else:
            result.append(task)
    return result


def _eliminate_redundant_copies(
    tasks: list[TaskDesc], buffer_sizes: dict[int, int],
) -> list[TaskDesc]:
    """Remove flat COPY tasks whose source is only consumed once.

    When a flat COPY (not CAT, not strided) has a source buffer that is
    only read by this single COPY, redirect all downstream consumers of
    the destination to use the source directly, and drop the COPY.
    """
    read_counts = _read_counts(tasks)

    remap: dict[int, int] = {}
    drop: set[int] = set()

    for i, task in enumerate(tasks):
        if (task.op_type == OpType.COPY
                and task.op_code == 0
                and task.strides[0] <= 0):
            src = task.buffer_indices[1]
            dst = task.buffer_indices[0]
            if src != UNUSED_BUFFER and read_counts.get(src, 0) == 1:
                remap[dst] = src
                drop.add(i)
                buffer_sizes.pop(dst, None)

    if not drop:
        return tasks

    def _resolve(buf: int) -> int:
        seen = set()
        while buf in remap and buf not in seen:
            seen.add(buf)
            buf = remap[buf]
        return buf

    result = []
    for i, task in enumerate(tasks):
        if i in drop:
            continue
        for j in range(8):
            b = task.buffer_indices[j]
            if b != UNUSED_BUFFER:
                task.buffer_indices[j] = _resolve(b)
        result.append(task)
    return result


def _constant_fold(ep, model=None):
    """Evaluate graph nodes that don't depend on user inputs.

    Returns a dict mapping node name → Tensor for every constant node
    that uses an op not in ATEN_OP_MAP (i.e. ops we can't run on the
    megakernel).  Supported-op constants are left for the normal walker
    since they already have CUDA task implementations.
    """
    graph = ep.graph_module.graph
    sig = ep.graph_signature

    if model is not None:
        sd = model.state_dict()
        named_bufs = dict(model.named_buffers())
    else:
        sd = dict(ep.state_dict)
        named_bufs = dict(ep.named_buffers())

    user_inputs: set[str] = set()
    node_values: dict[str, object] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            pname = _get_param_name(ep, node)
            if pname is None:
                user_inputs.add(node.name)
            else:
                val = sd.get(pname)
                if val is None:
                    val = named_bufs.get(pname)
                if val is not None:
                    node_values[node.name] = val

    dynamic: set[str] = set(user_inputs)
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        for arg in node.args:
            if hasattr(arg, "name") and arg.name in dynamic:
                dynamic.add(node.name)
                break
            if isinstance(arg, (list, tuple)):
                for a in arg:
                    if hasattr(a, "name") and a.name in dynamic:
                        dynamic.add(node.name)
                        break
                if node.name in dynamic:
                    break

    def _resolve(arg):
        if hasattr(arg, "name"):
            return node_values.get(arg.name)
        if isinstance(arg, (list, tuple)):
            resolved = [_resolve(a) for a in arg]
            if any(r is None and hasattr(a, "name") for r, a in zip(resolved, arg)):
                return None
            return type(arg)(resolved)
        return arg

    folded: dict[str, torch.Tensor] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.name in dynamic:
            continue
        if node.target is operator.getitem:
            src = node.args[0]
            idx = node.args[1]
            if hasattr(src, "name") and src.name in node_values:
                val = node_values[src.name]
                if isinstance(val, (tuple, list)):
                    node_values[node.name] = val[idx]
            continue

        resolved_args = [_resolve(a) for a in node.args]
        if any(
            r is None and hasattr(a, "name")
            for r, a in zip(resolved_args, node.args)
        ):
            continue

        resolved_kwargs = {}
        skip = False
        for k, v in node.kwargs.items():
            rv = _resolve(v)
            if rv is None and hasattr(v, "name"):
                skip = True
                break
            resolved_kwargs[k] = rv
        if skip:
            continue

        try:
            with torch.no_grad():
                result = node.target(*resolved_args, **resolved_kwargs)
            node_values[node.name] = result
            mapping = ATEN_OP_MAP.get(node.target)
            if mapping is None:
                if isinstance(result, torch.Tensor):
                    folded[node.name] = result
        except Exception:
            pass

    return folded


def _build_users_map(graph) -> dict[str, list]:
    users: dict[str, list] = {}
    for n in graph.nodes:
        for a in n.args:
            if hasattr(a, "name"):
                users.setdefault(a.name, []).append(n)
            if isinstance(a, (list, tuple)):
                for aa in a:
                    if hasattr(aa, "name"):
                        users.setdefault(aa.name, []).append(n)
    return users


def _find_rmsnorm_patterns(graph, users=None):
    """Detect decomposed RMSNorm patterns.

    Variant A (SmolLM2): _to_copy→pow(2)→mean→add(eps)→rsqrt→mul→_to_copy(fp16)→mul(weight)
    Variant B (Gemma):   _to_copy→pow(2)→mean→add(eps)→rsqrt→mul→mul(weight+1)→_to_copy(fp16)

    Returns dict mapping the final node name → (input_node, weight_node,
    skip_nodes_set, eps_float, weight_plus_one_bool).
    """
    if users is None:
        users = _build_users_map(graph)
    patterns = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target != torch.ops.aten._to_copy.default:
            continue
        out_meta = node.meta.get("val")
        if out_meta is None or not isinstance(out_meta, torch.Tensor):
            continue
        if out_meta.dtype != torch.float32:
            continue
        in_meta = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
        if in_meta is None or not isinstance(in_meta, torch.Tensor):
            continue
        if in_meta.dtype != torch.float16:
            continue

        cast_node = node
        cast_users = users.get(cast_node.name, [])

        pow_node = None
        for u in cast_users:
            if u.target == torch.ops.aten.pow.Tensor_Scalar and len(u.args) >= 2 and u.args[1] == 2:
                pow_node = u
                break
        if pow_node is None:
            continue

        mean_node = None
        for u in users.get(pow_node.name, []):
            if u.target == torch.ops.aten.mean.dim:
                mean_node = u
                break
        if mean_node is None:
            continue

        add_node = None
        eps_value = 1e-5
        for u in users.get(mean_node.name, []):
            if u.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
                add_node = u
                for a in u.args:
                    if isinstance(a, (int, float)) and a < 0.01:
                        eps_value = float(a)
                break
        if add_node is None:
            continue

        rsqrt_node = None
        for u in users.get(add_node.name, []):
            if u.target == torch.ops.aten.rsqrt.default:
                rsqrt_node = u
                break
        if rsqrt_node is None:
            continue

        mul_fp32_node = None
        for u in users.get(rsqrt_node.name, []):
            if u.target == torch.ops.aten.mul.Tensor:
                other_arg = u.args[0] if (hasattr(u.args[1], "name") and u.args[1].name == rsqrt_node.name) else u.args[1]
                if hasattr(other_arg, "name") and other_arg.name == cast_node.name:
                    mul_fp32_node = u
                    break
        if mul_fp32_node is None:
            continue

        base_skip = {cast_node.name, pow_node.name, mean_node.name,
                     add_node.name, rsqrt_node.name, mul_fp32_node.name}
        orig_input = cast_node.args[0]

        # Variant A: mul_fp32 → _to_copy(fp16) → mul(weight)
        cast_back_node = None
        for u in users.get(mul_fp32_node.name, []):
            if u.target == torch.ops.aten._to_copy.default:
                u_meta = u.meta.get("val")
                if u_meta is not None and isinstance(u_meta, torch.Tensor) and u_meta.dtype == torch.float16:
                    cast_back_node = u
                    break

        if cast_back_node is not None:
            mul_weight_node = None
            weight_node = None
            for u in users.get(cast_back_node.name, []):
                if u.target == torch.ops.aten.mul.Tensor:
                    for a in u.args:
                        if hasattr(a, "name") and a.name != cast_back_node.name:
                            weight_node = a
                            break
                    if weight_node is not None:
                        mul_weight_node = u
                        break
            if mul_weight_node is not None:
                skip = base_skip | {cast_back_node.name, mul_weight_node.name}
                patterns[mul_weight_node.name] = (orig_input, weight_node, skip, eps_value, False)
                continue

        # Variant B: mul_fp32 → mul(weight_processed) → _to_copy(fp16)
        for u in users.get(mul_fp32_node.name, []):
            if u.target != torch.ops.aten.mul.Tensor:
                continue
            other = u.args[0] if (hasattr(u.args[1], "name") and u.args[1].name == mul_fp32_node.name) else u.args[1]
            if not hasattr(other, "name"):
                continue

            # Check for _to_copy(fp16) after this mul
            cb = None
            for u2 in users.get(u.name, []):
                if u2.target == torch.ops.aten._to_copy.default:
                    u2_meta = u2.meta.get("val")
                    if u2_meta is not None and isinstance(u2_meta, torch.Tensor) and u2_meta.dtype == torch.float16:
                        cb = u2
                        break
            if cb is None:
                continue

            # Trace the weight: might be add(cast(weight), 1.0)
            weight_node = other
            extra_skip = set()
            weight_plus_one = False
            if hasattr(other, "target") and other.target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar):
                has_one = any(isinstance(a, (int, float)) and a == 1.0 for a in other.args)
                if has_one:
                    weight_plus_one = True
                    extra_skip.add(other.name)
                    for a in other.args:
                        if hasattr(a, "name") and a.name != other.name:
                            # Might be a _to_copy(weight)
                            inner = a
                            if hasattr(inner, "target") and inner.target == torch.ops.aten._to_copy.default:
                                extra_skip.add(inner.name)
                                weight_node = inner.args[0]
                            else:
                                weight_node = inner
                            break

            skip = base_skip | {u.name, cb.name} | extra_skip
            patterns[cb.name] = (orig_input, weight_node, skip, eps_value, weight_plus_one)
            break

    return patterns


def _find_rope_patterns(graph, users=None):
    """Detect decomposed RoPE patterns: x*cos + rotate_half(x)*sin.

    Returns dict mapping the output add node name → (input_node, cos_node,
    sin_node, skip_nodes_set).
    """
    if users is None:
        users = _build_users_map(graph)

    patterns: dict = {}
    for node in graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.cat.default:
            continue
        cat_args = node.args[0]
        if not isinstance(cat_args, (list, tuple)) or len(cat_args) != 2:
            continue

        neg_idx = -1
        for j, a in enumerate(cat_args):
            if hasattr(a, "target") and a.target == torch.ops.aten.neg.default:
                neg_idx = j
                break
        if neg_idx < 0:
            continue

        neg_node = cat_args[neg_idx]
        other_slice = cat_args[1 - neg_idx]
        neg_src = neg_node.args[0]

        if not (hasattr(neg_src, "target") and neg_src.target == torch.ops.aten.slice.Tensor):
            continue
        if not (hasattr(other_slice, "target") and other_slice.target == torch.ops.aten.slice.Tensor):
            continue

        src = neg_src.args[0]
        if not (hasattr(src, "name") and hasattr(other_slice.args[0], "name")
                and src.name == other_slice.args[0].name):
            continue

        mul_sin = None
        for u in users.get(node.name, []):
            if hasattr(u, "target") and u.target == torch.ops.aten.mul.Tensor:
                mul_sin = u
                break
        if mul_sin is None:
            continue

        mul_cos = None
        for u in users.get(src.name, []):
            if hasattr(u, "target") and u.target == torch.ops.aten.mul.Tensor:
                if u.name != mul_sin.name:
                    mul_cos = u
                    break
        if mul_cos is None:
            continue

        add_node = None
        for u in users.get(mul_cos.name, []):
            if hasattr(u, "target") and u.target == torch.ops.aten.add.Tensor:
                if any(hasattr(a, "name") and a.name == mul_sin.name for a in u.args):
                    add_node = u
                    break
        if add_node is None:
            for u in users.get(mul_sin.name, []):
                if hasattr(u, "target") and u.target == torch.ops.aten.add.Tensor:
                    if any(hasattr(a, "name") and a.name == mul_cos.name for a in u.args):
                        add_node = u
                        break
        if add_node is None:
            continue

        cos_node = None
        for a in mul_cos.args:
            if hasattr(a, "name") and a.name != src.name:
                cos_node = a
                break
        sin_node = None
        for a in mul_sin.args:
            if hasattr(a, "name") and a.name != node.name:
                sin_node = a
                break
        if cos_node is None or sin_node is None:
            continue

        skip = {neg_src.name, other_slice.name, neg_node.name, node.name,
                mul_cos.name, mul_sin.name, add_node.name}
        patterns[add_node.name] = (src, cos_node, sin_node, skip)

    return patterns



def compile_model(
    model: torch.nn.Module,
    example_input,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
    num_sms: int = 0,
) -> CompiledModel:
    if isinstance(example_input, torch.Tensor):
        example_args = (example_input,)
    else:
        example_args = tuple(example_input)

    ep = export(model, example_args, strict=False)
    ep = optimize_graph(ep)

    return compile_from_ep(
        ep, sm_version,
        dtype=dtype, batch_range=batch_range, seq_range=seq_range,
        num_sms=num_sms, model=model,
    )


def compile_from_ep(
    ep,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
    num_sms: int = 0,
    model: torch.nn.Module | None = None,
) -> CompiledModel:
    folded_constants = _constant_fold(ep, model)

    graph = ep.graph_module.graph
    users = _build_users_map(graph)
    rmsnorm_patterns = _find_rmsnorm_patterns(graph, users)
    rmsnorm_skip: set[str] = set()
    for _, (_, _, skip_set, _, _) in rmsnorm_patterns.items():
        rmsnorm_skip.update(skip_set)

    rope_patterns = _find_rope_patterns(graph, users)
    rope_skip: set[str] = set()
    for _, (_, _, _, skip_set) in rope_patterns.items():
        rope_skip.update(skip_set)

    tasks: list[TaskDesc] = []
    buffer_map: dict[str, int] = {}
    buffer_sizes: dict[int, int] = {}
    weight_buffers: set[int] = set()
    weight_names: dict[int, str] = {}
    folded_weight_tensors: dict[int, torch.Tensor] = {}
    view_map: dict[str, StridedView] = {}
    next_buffer_id = 0

    input_buffer_ids: list[int] = []
    output_buffer_id = -1
    output_shape: list[int] = []

    def alloc_buffer(name: str, shape: list[int], dt: torch.dtype) -> int:
        nonlocal next_buffer_id
        buf_id = next_buffer_id
        next_buffer_id += 1
        buffer_map[name] = buf_id
        buffer_sizes[buf_id] = _numel(shape) * _dtype_bytes(dt)
        view_map[name] = StridedView(buf_id, shape, contiguous_strides(shape))
        return buf_id

    def materialize_view(name: str) -> int:
        """Emit a strided COPY task to materialize a non-contiguous view."""
        view = view_map[name]
        shape = view.shape
        numel = _numel(shape)
        ndim = len(shape)
        mat_name = name + "_mat"
        dt = torch.float16
        mat_buf = alloc_buffer(mat_name, shape, dt)
        task = TaskDesc(
            op_type=OpType.COPY, op_code=0,
            num_tiles=compute_tiles(OpType.COPY, [numel] + [0] * 7, sm_version, num_sms),
            buffer_indices=[mat_buf, view.buffer_id] + [UNUSED_BUFFER] * 6,
            dimensions=[numel] + list(shape[:7]) + [0] * (7 - min(ndim, 7)),
        )
        task.strides[0] = ndim
        for d in range(min(ndim, 7)):
            task.strides[1 + d] = view.strides[d]
        task.strides[7] = view.offset
        tasks.append(task)
        buffer_map[name] = mat_buf
        view_map[name] = StridedView(mat_buf, shape, contiguous_strides(shape))
        return mat_buf

    unsupported_ops: list[str] = []

    for node in graph.nodes:
        if node.op == "placeholder":
            meta = node.meta.get("val")
            if meta is None:
                continue
            if isinstance(meta, torch.Tensor):
                shape = [int(s) for s in meta.shape]
                dt = meta.dtype
            else:
                continue
            buf_id = alloc_buffer(node.name, shape, dt)
            param_name = _get_param_name(ep, node)
            if param_name is not None:
                weight_buffers.add(buf_id)
                weight_names[buf_id] = param_name
            else:
                input_buffer_ids.append(buf_id)

        elif node.op == "call_function":
            target = node.target

            if target is operator.getitem:
                src = node.args[0]
                idx = node.args[1]
                if hasattr(src, "name") and src.name in buffer_map and idx == 0:
                    buffer_map[node.name] = buffer_map[src.name]
                    if src.name in view_map:
                        view_map[node.name] = view_map[src.name]
                continue

            mapping = ATEN_OP_MAP.get(target)

            if node.name in rmsnorm_skip and node.name not in rmsnorm_patterns:
                continue

            if node.name in rope_skip and node.name not in rope_patterns:
                continue

            if node.name in rope_patterns:
                src_node, cos_node, sin_node, _ = rope_patterns[node.name]
                out_meta = node.meta.get("val")
                if out_meta is not None and isinstance(out_meta, torch.Tensor):
                    out_shape = [int(s) for s in out_meta.shape]
                    out_buf = alloc_buffer(node.name, out_shape, torch.float16)
                    if src_node.name in view_map and not view_map[src_node.name].is_contiguous():
                        materialize_view(src_node.name)
                    if cos_node.name in view_map and not view_map[cos_node.name].is_contiguous():
                        materialize_view(cos_node.name)
                    if sin_node.name in view_map and not view_map[sin_node.name].is_contiguous():
                        materialize_view(sin_node.name)
                    in_buf = buffer_map.get(src_node.name, UNUSED_BUFFER)
                    cos_buf = buffer_map.get(cos_node.name, UNUSED_BUFFER)
                    sin_buf = buffer_map.get(sin_node.name, UNUSED_BUFFER)
                    if len(out_shape) >= 4:
                        batch = out_shape[0]
                        num_heads = out_shape[1]
                        seq_len = out_shape[2]
                        head_dim = out_shape[3]
                    elif len(out_shape) == 3:
                        batch = out_shape[0]
                        num_heads = 1
                        seq_len = out_shape[1]
                        head_dim = out_shape[2]
                    else:
                        batch = 1
                        num_heads = 1
                        seq_len = out_shape[0] if len(out_shape) > 0 else 1
                        head_dim = out_shape[1] if len(out_shape) > 1 else 1
                    cos_meta = cos_node.meta.get("val") if hasattr(cos_node, "meta") else None
                    cos_stride = head_dim
                    if cos_meta is not None and isinstance(cos_meta, torch.Tensor):
                        cos_stride = int(cos_meta.shape[-1])
                    is_bhsd = len(out_shape) >= 4 and out_shape[1] != seq_len
                    dims = [batch, seq_len, num_heads, head_dim, cos_stride, 0, 0, 0]
                    task = TaskDesc(
                        op_type=OpType.ROPE,
                        op_code=0,
                        num_tiles=compute_tiles(OpType.ROPE, dims, sm_version, num_sms),
                        buffer_indices=[out_buf, in_buf, cos_buf, sin_buf] + [UNUSED_BUFFER] * 4,
                        dimensions=dims,
                    )
                    if is_bhsd:
                        task.strides[0] = 1
                    tasks.append(task)
                continue

            if node.name in rmsnorm_patterns:
                orig_input, weight_node, _, eps_val, w_plus_one = rmsnorm_patterns[node.name]
                out_meta = node.meta.get("val")
                if out_meta is not None and isinstance(out_meta, torch.Tensor):
                    out_shape = [int(s) for s in out_meta.shape]
                    out_buf = alloc_buffer(node.name, out_shape, torch.float16)
                    if orig_input.name in view_map and not view_map[orig_input.name].is_contiguous():
                        materialize_view(orig_input.name)
                    in_buf = buffer_map.get(orig_input.name, UNUSED_BUFFER)
                    w_buf = buffer_map.get(weight_node.name, UNUSED_BUFFER)
                    num_rows = _numel(out_shape) // out_shape[-1] if len(out_shape) > 1 else 1
                    row_size = out_shape[-1]
                    eps_bits = _struct.unpack("<I", _struct.pack("<f", eps_val))[0]
                    dims = [num_rows, row_size, eps_bits] + [0] * 5
                    task = TaskDesc(
                        op_type=OpType.REDUCE,
                        op_code=ReduceCode.RMSNORM,
                        num_tiles=compute_tiles(OpType.REDUCE, dims, sm_version, num_sms),
                        buffer_indices=[out_buf, in_buf, w_buf] + [UNUSED_BUFFER] * 5,
                        dimensions=dims,
                    )
                    if w_plus_one:
                        task.strides[0] = 1
                    tasks.append(task)
                continue

            if node.name in folded_constants:
                tensor = folded_constants[node.name]
                fp16 = tensor.contiguous().cuda().half()
                shape = [int(s) for s in fp16.shape]
                buf_id = alloc_buffer(node.name, shape, torch.float16)
                weight_buffers.add(buf_id)
                weight_names[buf_id] = f"__folded__.{node.name}"
                folded_weight_tensors[buf_id] = fp16
                continue

            if mapping == "STRIDE_CHANGE":
                if node.args and hasattr(node.args[0], "name") and node.args[0].name in view_map:
                    input_view = view_map[node.args[0].name]
                    new_view = _resolve_shape_op(target, input_view, node.args)
                    if new_view is None:
                        src_name = node.args[0].name
                        if not view_map[src_name].is_contiguous():
                            materialize_view(src_name)
                        contig_view = view_map[src_name]
                        new_view = _resolve_shape_op(target, contig_view, node.args)
                        if new_view is not None:
                            view_map[node.name] = new_view
                            buffer_map[node.name] = new_view.buffer_id
                        else:
                            out_meta = node.meta.get("val")
                            if out_meta is not None and isinstance(out_meta, torch.Tensor):
                                out_shape = [int(s) for s in out_meta.shape]
                                copy_buf = alloc_buffer(node.name, out_shape, out_meta.dtype)
                                tasks.append(TaskDesc(
                                    op_type=OpType.COPY, op_code=0,
                                    num_tiles=compute_tiles(OpType.COPY, [_numel(out_shape)] + [0]*7, sm_version, num_sms),
                                    buffer_indices=[copy_buf, contig_view.buffer_id] + [UNUSED_BUFFER]*6,
                                    dimensions=[_numel(out_shape)] + [0]*7,
                                ))
                    else:
                        view_map[node.name] = new_view
                        buffer_map[node.name] = new_view.buffer_id

            elif mapping is None:
                op_name = getattr(target, "__name__", str(target))
                unsupported_ops.append(op_name)

            elif isinstance(mapping, tuple):
                op_type, op_code = mapping
                out_meta = node.meta.get("val")
                if out_meta is None:
                    continue

                if isinstance(out_meta, (tuple, list)):
                    out_meta = out_meta[0]
                if not isinstance(out_meta, torch.Tensor):
                    continue

                # Eliminate no-op CAST: when _to_copy has same dtype in/out,
                # just alias the buffer instead of emitting a grid.sync() task.
                if (op_type == OpType.ELEMENTWISE
                        and op_code == ElemCode.CAST
                        and target == torch.ops.aten._to_copy.default):
                    in_arg = node.args[0] if node.args else None
                    if (in_arg is not None and hasattr(in_arg, "meta")):
                        in_meta = in_arg.meta.get("val")
                        if (isinstance(in_meta, torch.Tensor)
                                and in_meta.dtype == out_meta.dtype
                                and hasattr(in_arg, "name")
                                and in_arg.name in buffer_map):
                            buffer_map[node.name] = buffer_map[in_arg.name]
                            if in_arg.name in view_map:
                                view_map[node.name] = view_map[in_arg.name]
                            continue

                # Eliminate identity elementwise: MUL(1.0), ADD(0.0), SUB(0.0), DIV(1.0)
                if op_type == OpType.ELEMENTWISE and op_code in _IDENTITY_VALS:
                    identity_val = _IDENTITY_VALS[op_code]
                    scalar_arg = None
                    tensor_arg = None
                    for arg in node.args:
                        if isinstance(arg, (int, float)):
                            scalar_arg = float(arg)
                        elif hasattr(arg, "name") and arg.name in buffer_map:
                            tensor_arg = arg
                    if (scalar_arg is not None
                            and scalar_arg == identity_val
                            and tensor_arg is not None):
                        buffer_map[node.name] = buffer_map[tensor_arg.name]
                        if tensor_arg.name in view_map:
                            view_map[node.name] = view_map[tensor_arg.name]
                        continue

                out_shape = [int(s) for s in out_meta.shape]
                out_buf = alloc_buffer(node.name, out_shape, out_meta.dtype)

                task = TaskDesc(op_type=op_type, op_code=op_code)
                task.buffer_indices[0] = out_buf

                # Identify the B (weight) argument for matmul ops —
                # skip materializing it so the transpose flag is preserved.
                _matmul_b_name = None
                if op_type == OpType.MATMUL:
                    if target == torch.ops.aten.addmm.default:
                        _b_arg = node.args[2] if len(node.args) > 2 else None
                    else:
                        _b_arg = node.args[1] if len(node.args) > 1 else None
                    if _b_arg is not None and hasattr(_b_arg, "name"):
                        _matmul_b_name = _b_arg.name

                # Materialize non-contiguous inputs (except
                # transposed-B matmul, handled separately via strides[0])
                for arg in node.args:
                    if hasattr(arg, "name") and arg.name in view_map:
                        if arg.name == _matmul_b_name:
                            continue
                        v = view_map[arg.name]
                        if not v.is_contiguous():
                            materialize_view(arg.name)

                # For CAT: args[0] is a list of tensor nodes
                if target == torch.ops.aten.cat.default:
                    cat_inputs = node.args[0]
                    cat_dim = node.args[1] if len(node.args) > 1 else 0
                    if cat_dim < 0:
                        cat_dim = len(out_shape) + cat_dim
                    for ci in cat_inputs:
                        if hasattr(ci, "name") and ci.name in view_map:
                            v = view_map[ci.name]
                            if not v.is_contiguous():
                                materialize_view(ci.name)
                    arg_slot = 1
                    for ci in cat_inputs:
                        if hasattr(ci, "name") and ci.name in buffer_map:
                            if arg_slot < 8:
                                task.buffer_indices[arg_slot] = buffer_map[ci.name]
                                arg_slot += 1
                    in0_meta = cat_inputs[0].meta.get("val")
                    in0_last = list(in0_meta.shape)[cat_dim] if in0_meta is not None else 0
                    task.dimensions[0] = _numel(out_shape)
                    task.dimensions[1] = int(in0_last)
                    task.dimensions[2] = int(out_shape[cat_dim])
                elif op_type == OpType.ATTENTION:
                    for i, arg in enumerate(node.args[:3]):
                        if hasattr(arg, "name") and arg.name in buffer_map:
                            task.buffer_indices[1 + i] = buffer_map[arg.name]
                else:
                    arg_slot = 1
                    for arg in node.args:
                        if hasattr(arg, "name") and arg.name in buffer_map:
                            if arg_slot < 8:
                                task.buffer_indices[arg_slot] = buffer_map[arg.name]
                                arg_slot += 1

                task.dimensions = _extract_dimensions(op_type, node, out_shape) if target != torch.ops.aten.cat.default else task.dimensions
                task.num_tiles = compute_tiles(op_type, task.dimensions, sm_version, num_sms)

                # For all MATMUL ops: detect transposed B operand
                if op_type == OpType.MATMUL:
                    if target == torch.ops.aten.addmm.default:
                        b_arg = node.args[2] if len(node.args) > 2 else None
                        # Fix buffer order: A=input(args[1]), B=weight(args[2])
                        a_arg = node.args[1] if len(node.args) > 1 else None
                        if a_arg is not None and hasattr(a_arg, "name") and a_arg.name in buffer_map:
                            if b_arg is not None and hasattr(b_arg, "name") and b_arg.name in buffer_map:
                                task.buffer_indices[1] = buffer_map[a_arg.name]
                                task.buffer_indices[2] = buffer_map[b_arg.name]
                                bias_arg = node.args[0]
                                if hasattr(bias_arg, "name") and bias_arg.name in buffer_map:
                                    task.buffer_indices[3] = buffer_map[bias_arg.name]
                    else:
                        b_arg = node.args[1] if len(node.args) > 1 else None

                    if b_arg is not None and hasattr(b_arg, "name") and b_arg.name in view_map:
                        b_view = view_map[b_arg.name]
                        if not b_view.is_contiguous():
                            task.strides[0] = 1

                # For ATTENTION: detect is_causal or bool mask
                if op_type == OpType.ATTENTION:
                    if len(node.args) >= 6 and node.args[5] is True:
                        task.strides[0] = 1
                    elif len(node.args) >= 4 and hasattr(node.args[3], "meta"):
                        mask_meta = node.args[3].meta.get("val")
                        if isinstance(mask_meta, torch.Tensor) and mask_meta.dtype == torch.bool:
                            task.strides[0] = 1

                # For GELU: detect approximate='tanh' kwarg
                if op_type == OpType.ELEMENTWISE and op_code == ElemCode.GELU:
                    approx = node.kwargs.get("approximate", "none")
                    if approx == "tanh":
                        task.op_code = ElemCode.GELU_TANH

                # For ELEMENTWISE: handle scalar args and broadcast
                if op_type == OpType.ELEMENTWISE:
                    for arg in node.args:
                        if isinstance(arg, (int, float)):
                            scalar_f = float(arg)
                            task.dimensions[1] = _struct.unpack(
                                '<I', _struct.pack('<f', scalar_f)
                            )[0]
                            break

                    tensor_args = [
                        a for a in node.args
                        if hasattr(a, "name") and a.name in buffer_map
                    ]
                    if len(tensor_args) >= 2:
                        in0_meta = tensor_args[0].meta.get("val")
                        in1_meta = tensor_args[1].meta.get("val")
                        if (in0_meta is not None and in1_meta is not None
                                and isinstance(in0_meta, torch.Tensor)
                                and isinstance(in1_meta, torch.Tensor)):
                            n0, n1 = int(in0_meta.numel()), int(in1_meta.numel())
                            if n0 < n1:
                                task.buffer_indices[1], task.buffer_indices[2] = (
                                    task.buffer_indices[2], task.buffer_indices[1]
                                )
                                in0_meta, in1_meta = in1_meta, in0_meta
                                n0, n1 = n1, n0
                            if n1 < n0:
                                task.dimensions[2] = n1
                                s0 = list(in0_meta.shape)
                                s1 = list(in1_meta.shape)
                                while len(s1) < len(s0):
                                    s1 = [1] + s1
                                repeat = 1
                                for d in range(len(s0) - 1, -1, -1):
                                    if s1[d] == 1 and s0[d] > 1:
                                        repeat *= s0[d]
                                    else:
                                        break
                                task.dimensions[3] = repeat

                tasks.append(task)

        elif node.op == "output":
            out_args = node.args[0]
            if isinstance(out_args, (tuple, list)):
                out_node = out_args[0]
            else:
                out_node = out_args

            if hasattr(out_node, "name"):
                if out_node.name in view_map:
                    output_buffer_id = view_map[out_node.name].buffer_id
                elif out_node.name in buffer_map:
                    output_buffer_id = buffer_map[out_node.name]

                out_meta = out_node.meta.get("val")
                if isinstance(out_meta, (tuple, list)):
                    out_meta = out_meta[0]
                if isinstance(out_meta, torch.Tensor):
                    output_shape = [int(s) for s in out_meta.shape]

    if unsupported_ops:
        unique = sorted(set(unsupported_ops))
        raise RuntimeError(
            f"Unsupported ATen ops ({len(unique)}): {', '.join(unique)}"
        )

    tasks = _fuse_tasks(tasks, buffer_sizes)
    tasks = _fuse_elementwise_chains(tasks, buffer_sizes, sm_version, num_sms)
    tasks = _eliminate_redundant_copies(tasks, buffer_sizes)

    placements, total_workspace = plan_buffers(tasks, buffer_sizes, weight_buffers)

    dtype_code = {torch.float16: 0x00, torch.bfloat16: 0x01, torch.float32: 0x02}.get(dtype, 0x00)

    schedule_bytes = write_schedule(
        tasks=tasks,
        placements=placements,
        weight_names=weight_names,
        workspace_bytes=total_workspace,
        sm_version=sm_version,
        compute_dtype=dtype_code,
        batch_range=batch_range,
        seq_range=seq_range,
    )

    return CompiledModel(
        schedule_bytes=schedule_bytes,
        input_buffer_ids=input_buffer_ids,
        output_buffer_id=output_buffer_id,
        output_shape=output_shape,
        num_buffers=next_buffer_id,
        folded_constants=folded_weight_tensors,
    )


def compile_schedule(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
    num_sms: int = 0,
) -> bytes:
    return compile_model(
        model, example_input, sm_version,
        dtype=dtype, batch_range=batch_range, seq_range=seq_range,
        num_sms=num_sms,
    ).schedule_bytes
