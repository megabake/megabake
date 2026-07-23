"""Walk FX graph from torch.export, map ATen ops to megakernel tasks."""

from dataclasses import dataclass, field
import operator
import struct as _struct
import torch
from torch.export import export

from megabake.data_types import TaskDesc, OpType, ElemCode, UopCode, UNUSED_BUFFER, pack_uop
from megabake.schedule_compiler.op_table import ATEN_OP_MAP, SHAPE_OPS
from megabake.schedule_compiler.shape_ops import (
    StridedView, contiguous_strides,
    resolve_reshape, resolve_transpose, resolve_permute,
    resolve_expand, resolve_squeeze, resolve_unsqueeze,
    resolve_slice, resolve_t,
)
from megabake.schedule_compiler.buffer_planner import plan_buffers
from megabake.schedule_compiler.tiling import compute_tiles
from megabake.schedule_compiler.serializer import write_schedule


@dataclass
class CompiledModel:
    schedule_bytes: bytes
    input_buffer_ids: list[int] = field(default_factory=list)
    output_buffer_id: int = -1
    output_shape: list[int] = field(default_factory=list)
    num_buffers: int = 0


def _numel(shape: list[int]) -> int:
    r = 1
    for s in shape:
        r *= s
    return r


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
                dims[4] = list(k_meta.shape)[2]
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
}


def _fuse_tasks(
    tasks: list[TaskDesc], buffer_sizes: dict[int, int]
) -> list[TaskDesc]:
    """Fuse adjacent MATMUL + unary ELEMENTWISE into a single fused op."""
    read_counts: dict[int, int] = {}
    for t in tasks:
        for buf in t.buffer_indices[1:]:
            if buf != UNUSED_BUFFER:
                read_counts[buf] = read_counts.get(buf, 0) + 1

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
    ElemCode.SILU, ElemCode.GELU, ElemCode.RELU, ElemCode.SIGMOID,
    ElemCode.TANH, ElemCode.EXP, ElemCode.LOG, ElemCode.RSQRT,
    ElemCode.NEG, ElemCode.ABS,
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
) -> list[TaskDesc]:
    """Fuse consecutive same-numel ELEMENTWISE tasks into a single FUSED_ELEMENTWISE
    with a micro-op program interpreted on-GPU."""
    read_counts: dict[int, int] = {}
    for t in tasks:
        for buf in t.buffer_indices[1:]:
            if buf != UNUSED_BUFFER:
                read_counts[buf] = read_counts.get(buf, 0) + 1

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
            num_tiles=compute_tiles(OpType.FUSED_ELEMENTWISE, dims, sm_version),
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


def compile_model(
    model: torch.nn.Module,
    example_input,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
) -> CompiledModel:
    if isinstance(example_input, torch.Tensor):
        example_args = (example_input,)
    else:
        example_args = tuple(example_input)

    decomp_table = torch._decomp.core_aten_decompositions()
    preserve_ops = [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
    ]
    for op in preserve_ops:
        decomp_table.pop(op, None)

    ep = export(model, example_args, strict=False)
    ep = ep.run_decompositions(decomp_table)

    graph = ep.graph_module.graph

    tasks: list[TaskDesc] = []
    buffer_map: dict[str, int] = {}
    buffer_sizes: dict[int, int] = {}
    weight_buffers: set[int] = set()
    weight_names: dict[int, str] = {}
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
            num_tiles=compute_tiles(OpType.COPY, [numel] + [0] * 7, sm_version),
            buffer_indices=[mat_buf, view.buffer_id] + [UNUSED_BUFFER] * 6,
            dimensions=[numel] + list(shape[:4]) + [0] * (7 - min(ndim, 4)),
        )
        task.strides[0] = ndim
        for d in range(min(ndim, 4)):
            task.strides[1 + d] = view.strides[d]
        task.strides[5] = view.offset
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
                                    num_tiles=compute_tiles(OpType.COPY, [_numel(out_shape)] + [0]*7, sm_version),
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

                out_shape = [int(s) for s in out_meta.shape]
                out_buf = alloc_buffer(node.name, out_shape, out_meta.dtype)

                task = TaskDesc(op_type=op_type, op_code=op_code)
                task.buffer_indices[0] = out_buf

                # Materialize non-contiguous inputs (except
                # transposed-B matmul, handled separately via strides[0])
                if target != torch.ops.aten.mm.default:
                    for arg in node.args:
                        if hasattr(arg, "name") and arg.name in view_map:
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
                else:
                    arg_slot = 1
                    for arg in node.args:
                        if hasattr(arg, "name") and arg.name in buffer_map:
                            if arg_slot < 8:
                                task.buffer_indices[arg_slot] = buffer_map[arg.name]
                                arg_slot += 1

                task.dimensions = _extract_dimensions(op_type, node, out_shape) if target != torch.ops.aten.cat.default else task.dimensions
                task.num_tiles = compute_tiles(op_type, task.dimensions, sm_version)

                # For MATMUL with mm: detect transposed B operand
                if op_type == OpType.MATMUL and target == torch.ops.aten.mm.default:
                    b_arg = node.args[1]
                    if hasattr(b_arg, "name") and b_arg.name in view_map:
                        b_view = view_map[b_arg.name]
                        if not b_view.is_contiguous():
                            task.strides[0] = 1

                # For ATTENTION: detect is_causal flag
                if op_type == OpType.ATTENTION:
                    if len(node.args) >= 6 and node.args[5] is True:
                        task.strides[0] = 1

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

    tasks = _fuse_tasks(tasks, buffer_sizes)
    tasks = _fuse_elementwise_chains(tasks, buffer_sizes, sm_version)

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
    )


def compile_schedule(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
) -> bytes:
    return compile_model(
        model, example_input, sm_version,
        dtype=dtype, batch_range=batch_range, seq_range=seq_range,
    ).schedule_bytes
