"""Conservative SIMT source generation for indexed maps, reductions, and GEMM."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping

from ....diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ....semantics.indexed import IndexedOp, IndexedValue


_CUDA_TYPES = {
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "float32": "float",
    "bool": "bool",
}
_MATH_MAPS = {
    "add", "sub", "mul", "div", "neg", "exp", "rsqrt", "sqrt", "square",
    "sigmoid", "tanh", "relu", "silu", "gelu", "where", "eq", "ne", "gt",
    "ge", "lt", "le", "to", "_to_copy", "convert_element_type", "maximum",
    "minimum", "pow",
}
_VIEWS = {"view", "reshape", "transpose", "permute", "t", "slice", "select",
          "squeeze", "unsqueeze", "expand", "detach", "alias", "flatten"}
_CONTRACTIONS = {"mm", "bmm", "addmm", "matmul", "linear"}


@dataclass(frozen=True)
class CudaBody:
    """A device-callable tile body; the owner supplies its logical output range."""

    source: str
    entry_point: str | None
    input_types: tuple[str, ...]
    output_type: str | None
    output_elements: int
    body_kind: str
    node_id: str
    host_guard: Mapping[str, Any] | None = None


class CudaBodyError(ValueError):
    def __init__(self, node_id: str, message: str):
        self.node_id = node_id
        self.message = message
        super().__init__(f"{node_id}: {message}")

    def diagnostic(self) -> DiagnosticRecord:
        return DiagnosticRecord(
            DiagnosticCode.UNSUPPORTED_SEMANTICS,
            self.message,
            DiagnosticSeverity.ERROR,
            node_id=self.node_id,
            details={"backend": "cuda", "negative_case": self.message},
        )


def _static_shape(value: IndexedValue, node_id: str) -> tuple[int, ...]:
    if value.layout not in (None, "strided"):
        raise CudaBodyError(node_id, f"layout {value.layout!r} is unsupported by the dense CUDA body")
    shape = value.shape
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise CudaBodyError(node_id, f"{value.value_id} has a dynamic or invalid shape")
    if len(value.strides) != len(shape):
        raise CudaBodyError(node_id, f"{value.value_id} has no exact strided layout")
    if any(stride < 0 for stride in value.strides):
        raise CudaBodyError(node_id, f"{value.value_id} has a negative stride")
    return tuple(shape)


def _numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def _output_offset(value: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> str:
    if len(value.strides) != len(output_shape):
        raise CudaBodyError(node_id, "output stride rank does not match its shape")
    return " + ".join(f"idx[{axis}]*{stride}" for axis, stride in enumerate(value.strides)) or "0"


def _broadcast_map(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> tuple[str, ...]:
    if len(input_shape) > len(output_shape):
        return ("UNKNOWN",)
    pad = len(output_shape) - len(input_shape)
    padded = (1,) * pad + input_shape
    return tuple("0" if dim == 1 else f"i{axis}"
                 for axis, dim in enumerate(padded))


def _dtype(value: IndexedValue, node_id: str) -> str:
    if value.layout not in (None, "strided"):
        raise CudaBodyError(node_id, f"layout {value.layout!r} is unsupported by the dense CUDA body")
    try:
        return _CUDA_TYPES[value.dtype or ""]
    except KeyError as exc:
        raise CudaBodyError(node_id, f"dtype {value.dtype!r} is unsupported by the CUDA body") from exc


def _operator(operation: IndexedOp) -> str:
    return str(operation.attributes.get("operator_name", operation.target.rsplit("::", 1)[-1].split(".", 1)[0]))


def _name(operation: IndexedOp) -> str:
    origin = operation.origin_ids[0] if operation.origin_ids else operation.target
    suffix = hashlib.sha256(str(origin).encode()).hexdigest()[:8]
    return "v3_" + re.sub(r"[^a-zA-Z0-9_]", "_", operation.op_id) + "_" + suffix


def _literal(value: Any, node_id: str) -> str:
    if isinstance(value, bool):
        return "1.0f" if value else "0.0f"
    if isinstance(value, int):
        return f"{value}.0f"
    if isinstance(value, float):
        if math.isnan(value):
            return "NAN"
        if math.isinf(value):
            return "INFINITY" if value > 0 else "(-INFINITY)"
        rendered = repr(value)
        if "." not in rendered and "e" not in rendered.lower():
            rendered += ".0"
        return rendered + "f"
    raise CudaBodyError(node_id, f"non-scalar map constant {value!r} is unsupported")


def _index_expr(expression: str, node_id: str) -> str:
    if not re.fullmatch(r"[0-9ir k+*().-]+", expression):
        raise CudaBodyError(node_id, f"unproved index expression {expression!r}")
    expression = re.sub(r"\bi(\d+)\b", r"idx[\1]", expression)
    expression = re.sub(r"\br(\d+)\b", r"red[\1]", expression)
    expression = re.sub(r"\bk\b", "red[0]", expression)
    return expression


def _input_offset(index_map: Any, value: IndexedValue, output_shape: tuple[int, ...],
                  node_id: str) -> str:
    shape = tuple(value.shape)
    expressions = index_map.expressions
    if index_map.mode == "strided_view" and len(expressions) == 1:
        return _index_expr(expressions[0], node_id)
    if index_map.mode == "reshape":
        if _numel(shape) != _numel(output_shape):
            raise CudaBodyError(node_id, "reshape map changes the element count")
        coordinates = [f"reshape_index[{axis}]" for axis in range(len(shape))]
        stride = [f"{item}*{value.strides[axis]}" for axis, item in enumerate(coordinates)]
        return " + ".join(stride) or "0"
    if index_map.mode in {"broadcast", "zero_stride_broadcast"}:
        if len(shape) > len(output_shape) or len(expressions) != len(output_shape):
            raise CudaBodyError(node_id, "broadcast index map rank does not match its output")
        strides = (0,) * (len(output_shape) - len(shape)) + tuple(value.strides)
        return " + ".join(
            f"({_index_expr(expr, node_id)})*{stride}"
            for expr, stride in zip(expressions, strides) if stride
        ) or "0"
    if len(expressions) != len(shape):
        raise CudaBodyError(node_id, "input index map rank does not match its tensor")
    return " + ".join(
        f"({_index_expr(expr, node_id)})*{stride}"
        for expr, stride in zip(expressions, value.strides)
    ) or "0"


def _helpers(prefix: str) -> str:
    return f"""#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
__device__ __forceinline__ float {prefix}_load(const float* p, int64_t i) {{ return p[i]; }}
__device__ __forceinline__ float {prefix}_load(const __half* p, int64_t i) {{ return __half2float(p[i]); }}
__device__ __forceinline__ float {prefix}_load(const __nv_bfloat16* p, int64_t i) {{ return __bfloat162float(p[i]); }}
__device__ __forceinline__ float {prefix}_load(const bool* p, int64_t i) {{ return p[i] ? 1.0f : 0.0f; }}
__device__ __forceinline__ float {prefix}_store(float x, float*) {{ return x; }}
__device__ __forceinline__ __half {prefix}_store(float x, __half*) {{ return __float2half_rn(x); }}
__device__ __forceinline__ __nv_bfloat16 {prefix}_store(float x, __nv_bfloat16*) {{ return __float2bfloat16_rn(x); }}
__device__ __forceinline__ bool {prefix}_store(float x, bool*) {{ return x != 0.0f; }}
"""


def _value_expression(value: Any, loaded: Mapping[str, str], node_id: str) -> str:
    if isinstance(value, dict):
        if "value" in value:
            try:
                return loaded[str(value["value"])]
            except KeyError as exc:
                raise CudaBodyError(node_id, f"map operand {value['value']!r} has no proved index map") from exc
        raise CudaBodyError(node_id, "typed or dynamic scalar operand is unsupported")
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_value_expression(item, loaded, node_id) for item in value) + ")"
    return _literal(value, node_id)


def _map_expression(operation: IndexedOp, loaded: Mapping[str, str], node_id: str) -> str:
    name = _operator(operation)
    args = operation.attributes.get("arguments", ())
    kwargs = operation.attributes.get("keywords", {})
    if name in {"to", "_to_copy", "convert_element_type"}:
        return _value_expression(args[0], loaded, node_id)
    if name == "contiguous":
        return _value_expression(args[0], loaded, node_id)
    if name == "gelu":
        values = [_value_expression(args[0], loaded, node_id)]
        approximate = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
        if isinstance(approximate, dict):
            approximate = approximate.get("value")
        if approximate != "none":
            raise CudaBodyError(node_id, f"GELU approximation {approximate!r} is unsupported")
        return f"(0.5f * {values[0]} * (1.0f + erff({values[0]} * 0.7071067811865475f)))"
    if name == "div":
        values = [_value_expression(item, loaded, node_id) for item in args[:2]]
        rounding = kwargs.get("rounding_mode", args[2] if len(args) > 2 else None)
        if rounding is not None:
            raise CudaBodyError(node_id, f"division rounding mode {rounding!r} is unsupported")
        if len(values) < 2:
            raise CudaBodyError(node_id, "div requires two operands")
        return f"({values[0]} / {values[1]})"
    if name in {"add", "sub"}:
        values = [_value_expression(item, loaded, node_id) for item in args[:2]]
        if len(values) < 2:
            raise CudaBodyError(node_id, f"{name} requires two operands")
        alpha_raw = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        alpha = _literal(alpha_raw, node_id)
        return f"({values[0]} {'+' if name == 'add' else '-'} ({alpha} * {values[1]}))"
    if name in {"mul", "pow", "maximum", "minimum"}:
        values = [_value_expression(item, loaded, node_id) for item in args[:2]]
        if len(values) < 2:
            raise CudaBodyError(node_id, f"{name} requires two operands")
        left, right = values
        if name == "mul": return f"({left} * {right})"
        if name == "pow": return f"powf({left}, {right})"
        fn = "fmaxf" if name == "maximum" else "fminf"
        return f"(isnan({left}) || isnan({right}) ? NAN : {fn}({left}, {right}))"
    if name == "where":
        values = [_value_expression(item, loaded, node_id) for item in args[:3]]
        if len(values) != 3:
            raise CudaBodyError(node_id, "where requires condition and two values")
        return f"(({values[0]} != 0.0f) ? {values[1]} : {values[2]})"
    if name in {"eq", "ne", "gt", "ge", "lt", "le"}:
        values = [_value_expression(item, loaded, node_id) for item in args[:2]]
        if len(values) < 2:
            raise CudaBodyError(node_id, f"{name} requires two operands")
        symbol = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[name]
        return f"({values[0]} {symbol} {values[1]} ? 1.0f : 0.0f)"
    values = [_value_expression(item, loaded, node_id) for item in args[:1]]
    if name == "neg": return f"(-{values[0]})"
    if name == "exp": return f"expf({values[0]})"
    if name == "rsqrt": return f"rsqrtf({values[0]})"
    if name == "sqrt": return f"sqrtf({values[0]})"
    if name == "square": return f"({values[0]} * {values[0]})"
    if name == "sigmoid": return f"(1.0f / (1.0f + expf(-{values[0]})))"
    if name == "tanh": return f"tanhf({values[0]})"
    if name == "relu": return f"(isnan({values[0]}) ? NAN : fmaxf({values[0]}, 0.0f))"
    if name == "silu": return f"({values[0]} / (1.0f + expf(-{values[0]})))"
    raise CudaBodyError(node_id, f"map operator {name!r} is unsupported")


def _pointer_arguments(input_types: tuple[str, ...], output_type: str) -> str:
    inputs = [f"const {dtype}* in{index}" for index, dtype in enumerate(input_types)]
    inputs.append(f"{output_type}* out")
    inputs.extend(("int64_t tile_begin", "int64_t tile_end"))
    return ", ".join(inputs)


def _coordinates(shape: tuple[int, ...], *, source: str = "idx", flat: str = "linear") -> list[str]:
    lines = [f"int64_t {source}[{max(len(shape), 1)}] = {{0}};"]
    if shape:
        lines.append(f"int64_t {source}_remaining = {flat};")
        for axis in range(len(shape) - 1, -1, -1):
            lines.append(f"{source}[{axis}] = {source}_remaining % {shape[axis]};")
            lines.append(f"{source}_remaining /= {shape[axis]};")
    return lines


def _body_loop(entry: str, args: str, output_elements: int, lines: list[str]) -> str:
    body = "\n  ".join(lines)
    return f"""__device__ __forceinline__ void {entry}({args}) {{
  const int64_t begin = tile_begin < 0 ? 0 : tile_begin;
  const int64_t end = tile_end < {output_elements} ? tile_end : {output_elements};
    for (int64_t linear = begin + threadIdx.x; linear < end; linear += blockDim.x) {{
    {body}
  }}
}}
"""


def _view_body(operation: IndexedOp, values: Mapping[str, IndexedValue], node_id: str,
               output_shape: tuple[int, ...]) -> CudaBody:
    if len(operation.inputs) != 1 or len(operation.input_index_maps) != 1:
        raise CudaBodyError(node_id, "view requires one statically mapped input")
    source = values[operation.inputs[0]]
    _static_shape(source, node_id)
    _dtype(source, node_id)
    output = values[operation.outputs[0]]
    if output.alias_kind not in {"view", "copy"}:
        raise CudaBodyError(node_id, "view alias or materialization relation is unproved")
    index_map = operation.input_index_maps[0]
    prefix = _name(operation)
    if index_map.mode == "reshape":
        if _numel(tuple(source.shape)) != _numel(output_shape):
            raise CudaBodyError(node_id, "reshape map changes the element count")
        if output.alias_kind == "copy":
            return _emit_view_copy(operation, values, output, source, output_shape, node_id)
        args = "int64_t linear"
        body = [f"int64_t src[{max(len(source.shape), 1)}] = {{0}};", "int64_t remaining = linear;"]
        for axis in range(len(source.shape) - 1, -1, -1):
            body.append(f"src[{axis}] = remaining % {source.shape[axis]};")
            body.append(f"remaining /= {source.shape[axis]};")
        expression = " + ".join(f"src[{axis}]*{stride}" for axis, stride in enumerate(source.strides)) or "0"
        code = ("#include <stdint.h>\n" +
                f"__device__ __forceinline__ int64_t {prefix}_offset({args}) {{\n  " +
                "\n  ".join(body) + f"\n  return {expression};\n}}\n")
    else:
        args = ", ".join(f"int64_t i{axis}" for axis in range(len(output_shape))) or "void"
        if index_map.mode not in {"strided_view", "broadcast", "zero_stride_broadcast"} and len(index_map.expressions) != len(source.shape):
            raise CudaBodyError(node_id, "view index map rank does not match its source")
        coordinates = ", ".join(f"i{axis}" for axis in range(len(output_shape)))
        expression = _input_offset(index_map, source, output_shape, node_id)
        code = f"""#include <stdint.h>
__device__ __forceinline__ int64_t {prefix}_offset({args}) {{
  int64_t idx[{max(len(output_shape), 1)}] = {{{coordinates or '0'}}};
  return {expression};
}}
"""
    return CudaBody(code, f"{prefix}_offset", (), None, _numel(output_shape), "view_map", node_id)


def _emit_view_copy(operation: IndexedOp, values: Mapping[str, IndexedValue],
                    output: IndexedValue, source_value: IndexedValue,
                    output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    input_type, output_type = _dtype(source_value, node_id), _dtype(output, node_id)
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    index_map = operation.input_index_maps[0]
    if index_map.mode == "reshape":
        lines.extend(_coordinates(tuple(source_value.shape), source="reshape_index", flat="linear"))
    offset = _input_offset(index_map, source_value, output_shape, node_id)
    lines.append(f"const float value = {prefix}_load(in0, {offset});")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(entry, _pointer_arguments((input_type,), output_type),
                                           _numel(output_shape), lines)
    return CudaBody(source, entry, (input_type,), output_type, _numel(output_shape), "view_copy", node_id)


def _emit_map_or_view(operation: IndexedOp, values: Mapping[str, IndexedValue],
                      output: IndexedValue, output_shape: tuple[int, ...],
                      node_id: str) -> CudaBody:
    name = _operator(operation)
    if operation.kind == "Broadcast/View" or name in _VIEWS:
        return _view_body(operation, values, node_id, output_shape)
    if name not in _MATH_MAPS and name != "contiguous":
        raise CudaBodyError(node_id, f"map operator {name!r} is unsupported")
    if len(operation.inputs) != len(operation.input_index_maps):
        raise CudaBodyError(node_id, "map operands and input index maps differ")
    input_values = [values[value_id] for value_id in operation.inputs]
    input_types = tuple(_dtype(value, node_id) for value in input_values)
    output_type = _dtype(output, node_id)
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    loaded: dict[str, str] = {}
    for index, (value_id, value, index_map) in enumerate(zip(operation.inputs, input_values, operation.input_index_maps)):
        offset = _input_offset(index_map, value, output_shape, node_id)
        name_i = f"v{index}"
        lines.append(f"const float {name_i} = {prefix}_load(in{index}, {offset});")
        loaded[value_id] = name_i
    expression = _map_expression(operation, loaded, node_id)
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store({expression}, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(entry, _pointer_arguments(input_types, output_type),
                                           _numel(output_shape), lines)
    return CudaBody(source, entry, input_types, output_type, _numel(output_shape), "map", node_id)


def _emit_reduce(operation: IndexedOp, values: Mapping[str, IndexedValue],
                 output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    if len(operation.inputs) != 1 or len(operation.input_index_maps) != 1:
        raise CudaBodyError(node_id, "reduction requires one statically mapped input")
    value = values[operation.inputs[0]]
    input_shape = _static_shape(value, node_id)
    input_type = _dtype(value, node_id)
    output_type = _dtype(output, node_id)
    reduction = operation.attributes.get("reduction", {})
    axes = reduction.get("axes")
    keepdim = reduction.get("keepdim")
    if not isinstance(axes, list) or not isinstance(keepdim, bool):
        raise CudaBodyError(node_id, "reduction axes or keepdim are unknown")
    expected = tuple(1 if keepdim and axis in axes else dim
                     for axis, dim in enumerate(input_shape) if keepdim or axis not in axes)
    if expected != output_shape:
        raise CudaBodyError(node_id, "reduction output shape does not match its declared axes")
    red_shape = tuple(input_shape[axis] for axis in axes)
    red_count = _numel(red_shape) if axes else 1
    name = _operator(operation)
    if name not in {"sum", "mean", "amax", "max"}:
        raise CudaBodyError(node_id, f"reduction {name!r} is unsupported")
    if name in {"mean", "amax", "max"} and red_count == 0:
        raise CudaBodyError(node_id, "empty mean/max reductions are unsupported")
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    lines.append(f"float acc = {_literal(reduction.get('initial', 0.0), node_id)};")
    lines.append(f"for (int64_t red_linear = 0; red_linear < {red_count}; ++red_linear) {{")
    lines.append(f"  int64_t red[{max(len(axes), 1)}] = {{0}};")
    lines.append("  int64_t red_remaining = red_linear;")
    for index in range(len(axes) - 1, -1, -1):
        extent = red_shape[index]
        lines.append(f"  red[{index}] = red_remaining % {extent};")
        lines.append(f"  red_remaining /= {extent};")
    offset = _input_offset(operation.input_index_maps[0], value, output_shape, node_id)
    lines.append(f"  const float x = {prefix}_load(in0, {offset});")
    if name in {"amax", "max"}:
        lines.append("  if (isnan(x) || x > acc) acc = x;")
    else:
        lines.append("  acc += x;")
    lines.append("}")
    if name == "mean":
        lines.append(f"acc /= {red_count}.0f;")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(acc, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(entry, _pointer_arguments((input_type,), output_type),
                                           _numel(output_shape), lines)
    return CudaBody(source, entry, (input_type,), output_type, _numel(output_shape), "reduction", node_id)


def _emit_contraction(operation: IndexedOp, values: Mapping[str, IndexedValue],
                      output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    name = _operator(operation)
    if name not in _CONTRACTIONS:
        raise CudaBodyError(node_id, f"contraction {name!r} is unsupported")
    inputs = [values[value_id] for value_id in operation.inputs]
    shapes = [_static_shape(value, node_id) for value in inputs]
    input_types = tuple(_dtype(value, node_id) for value in inputs)
    output_type = _dtype(output, node_id)
    if len(operation.input_index_maps) != len(inputs):
        raise CudaBodyError(node_id, "contraction operands and index maps differ")
    def require_map(index: int, expected: tuple[str, ...], mode: str = "contraction") -> None:
        current = operation.input_index_maps[index]
        if current.mode != mode or current.expressions != expected:
            raise CudaBodyError(node_id, f"operand {index} does not match its exact {mode} map")

    if name == "addmm":
        if len(shapes) != 3 or len(shapes[1]) != 2 or len(shapes[2]) != 2 or len(output_shape) != 2:
            raise CudaBodyError(node_id, "addmm requires dense rank-two matrix operands and rank-two output")
        left, right = shapes[1], shapes[2]
        bias_map, left_map, right_map = operation.input_index_maps
        if left[1] != right[0] or len(left_map.expressions) != 2 or len(right_map.expressions) != 2:
            raise CudaBodyError(node_id, "addmm K dimensions or index maps are incompatible")
        if output_shape != (left[0], right[1]):
            raise CudaBodyError(node_id, "addmm output shape does not match its matrix maps")
        require_map(1, ("i0", "k"))
        require_map(2, ("k", "i1"))
        expected_bias = _broadcast_map(shapes[0], output_shape)
        if bias_map.mode != "broadcast" or bias_map.expressions != expected_bias:
            raise CudaBodyError(node_id, "addmm bias map is not the exact declared broadcast")
        beta = operation.attributes.get("contraction", {}).get("beta", 1.0)
        alpha = operation.attributes.get("contraction", {}).get("alpha", 1.0)
    elif name in {"mm", "matmul"}:
        if len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2 or len(output_shape) != 2:
            raise CudaBodyError(node_id, f"{name} supports dense rank-two operands in this body")
        left, right = shapes
        if left[1] != right[0] or output_shape != (left[0], right[1]):
            raise CudaBodyError(node_id, f"{name} dimensions or output map are incompatible")
        require_map(0, ("i0", "k"))
        require_map(1, ("k", "i1"))
        beta, alpha = 0.0, 1.0
    elif name == "bmm":
        if len(shapes) != 2 or len(shapes[0]) != 3 or len(shapes[1]) != 3 or len(output_shape) != 3:
            raise CudaBodyError(node_id, "bmm requires dense rank-three operands")
        left, right = shapes
        if left[0] != right[0] or left[2] != right[1] or output_shape != (left[0], left[1], right[2]):
            raise CudaBodyError(node_id, "bmm batch/K dimensions or output map are incompatible")
        require_map(0, ("i0", "i1", "k"))
        require_map(1, ("i0", "k", "i2"))
        beta, alpha = 0.0, 1.0
    else:  # linear
        if len(shapes) not in {2, 3} or len(shapes[0]) < 1 or len(shapes[1]) != 2:
            raise CudaBodyError(node_id, "linear requires a dense input and a rank-two [N,K] weight")
        if shapes[0][-1] != shapes[1][1] or output_shape != shapes[0][:-1] + (shapes[1][0],):
            raise CudaBodyError(node_id, "linear K dimensions or output shape are incompatible")
        if len(operation.input_index_maps) != len(shapes):
            raise CudaBodyError(node_id, "linear operand maps are incomplete")
        require_map(0, tuple(f"i{axis}" for axis in range(len(shapes[0]) - 1)) + ("k",))
        require_map(1, (f"i{len(shapes[0]) - 1}", "k"))
        if len(shapes) == 3:
            expected_bias = _broadcast_map(shapes[2], output_shape)
            require_map(2, expected_bias, "broadcast")
        beta, alpha = 0.0, 1.0
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise CudaBodyError(node_id, "contraction alpha must be a finite scalar")
    if not isinstance(beta, (int, float)) or not math.isfinite(beta):
        raise CudaBodyError(node_id, "contraction beta must be a finite scalar")
    k_extent = operation.reduction_domain[0].extent if operation.reduction_domain else None
    if not isinstance(k_extent, int) or k_extent < 0:
        raise CudaBodyError(node_id, "contraction K extent is unknown")
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    lines.append("float acc = 0.0f;")
    lines.append(f"for (int64_t red_linear = 0; red_linear < {k_extent}; ++red_linear) {{")
    lines.append("  int64_t red[1] = {red_linear};")
    for index, (value, index_map) in enumerate(zip(inputs, operation.input_index_maps)):
        if name == "addmm" and index == 0 or name == "linear" and index == 2:
            continue
        offset = _input_offset(index_map, value, output_shape, node_id)
        lines.append(f"  const float x{index} = {prefix}_load(in{index}, {offset});")
    left_index, right_index = (1, 2) if name == "addmm" else (0, 1)
    lines.append(f"  acc += x{left_index} * x{right_index};")
    lines.append("}")
    if name == "addmm":
        bias_offset = _input_offset(operation.input_index_maps[0], inputs[0], output_shape, node_id)
        lines.append(f"const float result = ({_literal(beta, node_id)} * {prefix}_load(in0, {bias_offset})) + ({_literal(alpha, node_id)} * acc);")
    elif name == "linear" and len(inputs) == 3:
        bias_offset = _input_offset(operation.input_index_maps[2], inputs[2], output_shape, node_id)
        lines.append(f"const float result = acc + {prefix}_load(in2, {bias_offset});")
    else:
        lines.append(f"const float result = {_literal(alpha, node_id)} * acc;")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(result, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(entry, _pointer_arguments(input_types, output_type),
                                           _numel(output_shape), lines)
    return CudaBody(source, entry, input_types, output_type, _numel(output_shape), "contraction", node_id)


def emit_cuda_body(program: Any, operation: IndexedOp | str) -> CudaBody:
    """Emit one conservative in-grid body or exact view address map.

    The input pointers address each indexed value's logical element zero. The
    owner passes a disjoint flattened output interval; no body launches a grid,
    assumes a block index, or uses cross-thread synchronization.
    """
    if isinstance(operation, str):
        operation = next(item for item in program.operations if item.op_id == operation)
    values = {item.value_id: item for item in program.values}
    node_id = operation.local_reference.node_name
    if operation.kind == "Guard":
        return CudaBody("", None, (), None, 0, "guard", node_id,
                        {"kind": operation.attributes.get("guard_kind"),
                         "arguments": operation.attributes.get("arguments"),
                         "keywords": operation.attributes.get("keywords")})
    if not operation.outputs or operation.outputs[0] not in values:
        raise CudaBodyError(node_id, "operation has no typed output value")
    output = values[operation.outputs[0]]
    output_shape = _static_shape(output, node_id)
    if output.layout not in (None, "strided"):
        raise CudaBodyError(node_id, f"output layout {output.layout!r} is unsupported")
    if operation.kind == "Map" or operation.kind == "Broadcast/View":
        return _emit_map_or_view(operation, values, output, output_shape, node_id)
    if operation.kind == "Reduce":
        return _emit_reduce(operation, values, output, output_shape, node_id)
    if operation.kind == "Contraction":
        return _emit_contraction(operation, values, output, output_shape, node_id)
    raise CudaBodyError(node_id, f"indexed operation kind {operation.kind!r} has no generic CUDA body")


__all__ = ["CudaBody", "CudaBodyError", "emit_cuda_body"]
