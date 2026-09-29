"""Conservative SIMT source generation for indexed maps, reductions, and GEMM."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping

from ....contracts import ContractError
from ....diagnostics import DiagnosticCode, DiagnosticRecord, DiagnosticSeverity
from ....semantics.indexed import IndexedOp, IndexedValue, InputIndexMap


_CUDA_TYPES = {
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "float32": "float",
    "bool": "bool",
    "int32": "int32_t",
    "int64": "int64_t",
}
_MATH_MAPS = {
    "add", "sub", "mul", "div", "neg", "exp", "rsqrt", "sqrt", "square",
    "sigmoid", "tanh", "relu", "silu", "gelu", "cos", "sin", "where", "eq", "ne", "gt",
    "ge", "lt", "le", "__and__", "to", "_to_copy", "convert_element_type", "maximum",
    "minimum", "pow",
}
_VIEWS = {"view", "reshape", "transpose", "permute", "t", "slice", "select",
          "squeeze", "unsqueeze", "expand", "detach", "detach_", "alias", "flatten"}
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
__device__ __forceinline__ float {prefix}_load(const int32_t* p, int64_t i) {{ return static_cast<float>(p[i]); }}
__device__ __forceinline__ float {prefix}_load(const int64_t* p, int64_t i) {{ return static_cast<float>(p[i]); }}
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


def _integer_expression(value: Any, loaded: Mapping[str, str], node_id: str) -> str:
    if isinstance(value, dict):
        if "value" in value:
            try:
                return loaded[str(value["value"])]
            except KeyError as exc:
                raise CudaBodyError(node_id, f"integer operand {value['value']!r} has no proved index map") from exc
        raise CudaBodyError(node_id, "typed or dynamic integer scalar operand is unsupported")
    if isinstance(value, bool):
        return "1LL" if value else "0LL"
    if isinstance(value, int):
        return f"{value}LL"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_integer_expression(item, loaded, node_id) for item in value) + ")"
    raise CudaBodyError(node_id, f"non-integral operand {value!r} is unsupported for integer maps")


def _integer_map_expression(operation: IndexedOp, loaded: Mapping[str, str], node_id: str) -> str:
    name = _operator(operation)
    args = operation.attributes.get("arguments", ())
    kwargs = operation.attributes.get("keywords", {})
    if name in {"add", "sub"}:
        if len(args) < 2:
            raise CudaBodyError(node_id, f"integer {name} requires two operands")
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        alpha = alpha.get("value") if isinstance(alpha, dict) else alpha
        if not isinstance(alpha, int) or isinstance(alpha, bool):
            raise CudaBodyError(node_id, "integer add/sub requires an integer alpha")
        left, right = (_integer_expression(item, loaded, node_id) for item in args[:2])
        symbol = "+" if name == "add" else "-"
        return (f"static_cast<int64_t>(static_cast<uint64_t>({left}) {symbol} "
                f"(static_cast<uint64_t>({alpha}LL) * static_cast<uint64_t>({right})))")
    if name in {"eq", "ne", "gt", "ge", "lt", "le"}:
        if len(args) < 2:
            raise CudaBodyError(node_id, f"integer {name} requires two operands")
        left, right = (_integer_expression(item, loaded, node_id) for item in args[:2])
        symbol = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}[name]
        return f"({left} {symbol} {right} ? 1.0f : 0.0f)"
    raise CudaBodyError(node_id, f"integer map operator {name!r} is unsupported")


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
    if name == "__and__":
        values = [_value_expression(item, loaded, node_id) for item in args[:2]]
        if len(values) != 2:
            raise CudaBodyError(node_id, "boolean and requires two operands")
        return f"(({values[0]} != 0.0f) && ({values[1]} != 0.0f) ? 1.0f : 0.0f)"
    values = [_value_expression(item, loaded, node_id) for item in args[:1]]
    if name == "neg": return f"(-{values[0]})"
    if name == "exp": return f"expf({values[0]})"
    if name == "rsqrt": return f"rsqrtf({values[0]})"
    if name == "sqrt": return f"sqrtf({values[0]})"
    if name == "square": return f"({values[0]} * {values[0]})"
    if name == "sigmoid": return f"(1.0f / (1.0f + expf(-{values[0]})))"
    if name == "tanh": return f"tanhf({values[0]})"
    if name == "cos": return f"cosf({values[0]})"
    if name == "sin": return f"sinf({values[0]})"
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
    if name == "__and__" and (input_types != ("bool", "bool") or output_type != "bool"):
        raise CudaBodyError(node_id, "logical and body requires boolean inputs and output")
    integer_types = {"int32_t", "int64_t"}
    integer_arithmetic = (name in {"add", "sub"} and output_type in integer_types and
                          all(item in integer_types for item in input_types))
    integer_comparison = (name in {"eq", "ne", "gt", "ge", "lt", "le"} and
                          bool(input_types) and all(item in integer_types for item in input_types))
    integer_cast = (name in {"to", "_to_copy", "convert_element_type"} and
                    output_type in integer_types and len(input_types) == 1 and
                    input_types[0] in integer_types)
    if output_type in integer_types and not (integer_arithmetic or integer_cast):
        raise CudaBodyError(node_id, "integer output map needs an exact integer body")
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    loaded: dict[str, str] = {}
    for index, (value_id, value, index_map) in enumerate(zip(operation.inputs, input_values, operation.input_index_maps)):
        offset = _input_offset(index_map, value, output_shape, node_id)
        name_i = f"v{index}"
        if integer_arithmetic or integer_comparison or integer_cast:
            lines.append(f"const int64_t {name_i} = static_cast<int64_t>(in{index}[{offset}]);")
        else:
            lines.append(f"const float {name_i} = {prefix}_load(in{index}, {offset});")
        loaded[value_id] = name_i
    if integer_arithmetic or integer_comparison:
        expression = _integer_map_expression(operation, loaded, node_id)
    elif integer_cast:
        expression = f"static_cast<{output_type}>(v0)"
    else:
        expression = _map_expression(operation, loaded, node_id)
    if integer_arithmetic or integer_cast:
        lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {expression};")
    else:
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
    elif name == "mm":
        if len(shapes) != 2 or len(shapes[0]) != 2 or len(shapes[1]) != 2 or len(output_shape) != 2:
            raise CudaBodyError(node_id, "mm requires dense rank-two operands")
        left, right = shapes
        if left[1] != right[0] or output_shape != (left[0], right[1]):
            raise CudaBodyError(node_id, "mm dimensions or output map are incompatible")
        require_map(0, ("i0", "k"))
        require_map(1, ("k", "i1"))
        beta, alpha = 0.0, 1.0
    elif name == "matmul":
        if len(shapes) != 2 or not shapes[0] or not shapes[1]:
            raise CudaBodyError(node_id, "matmul requires non-scalar dense operands")
        left, right = shapes
        left_vector, right_vector = len(left) == 1, len(right) == 1
        left_k = left[-1]
        right_k = right[-1] if right_vector else right[-2]
        batch_rank = max(len(left) - 2, len(right) - 2, 0)
        left_batch, right_batch = left[:-2] if not left_vector else (), \
            right[:-2] if not right_vector else ()
        padded_left = (1,) * (batch_rank - len(left_batch)) + left_batch
        padded_right = (1,) * (batch_rank - len(right_batch)) + right_batch
        batch_shape = tuple(max(a, b) for a, b in zip(padded_left, padded_right))
        if (left_k != right_k or any(a not in (1, extent) or b not in (1, extent)
                                     for a, b, extent in zip(padded_left, padded_right, batch_shape))):
            raise CudaBodyError(node_id, "matmul K or batch dimensions are incompatible")
        expected_shape = batch_shape + (() if left_vector and right_vector else
            ((right[-1],) if left_vector else ()))
        if not left_vector and not right_vector:
            expected_shape = batch_shape + (left[-2], right[-1])
        elif right_vector and not left_vector:
            expected_shape = batch_shape + (left[-2],)
        left_map = (("k",) if left_vector else
                    _broadcast_map(left_batch, batch_shape) + (f"i{batch_rank}", "k"))
        right_map = (_broadcast_map(right_batch, batch_shape) + ("k",) if right_vector else
                     _broadcast_map(right_batch, batch_shape) + ("k", f"i{batch_rank + 1}"))
        if output_shape != expected_shape:
            raise CudaBodyError(node_id, "matmul output shape differs from broadcast matrix dimensions")
        require_map(0, left_map)
        require_map(1, right_map)
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


def _emit_cat_stack(operation: IndexedOp, values: Mapping[str, IndexedValue],
                    output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    name = _operator(operation)
    contract = operation.attributes.get("tensor_construct")
    if (name not in {"cat", "stack"} or not isinstance(contract, Mapping) or
            contract.get("operator") != name or not operation.inputs or
            len(operation.input_index_maps) != len(operation.inputs)):
        raise CudaBodyError(node_id, "cat/stack has no exact tensor construction contract")
    sources = [values[value_id] for value_id in operation.inputs]
    shapes = [_static_shape(value, node_id) for value in sources]
    input_types = tuple(_dtype(value, node_id) for value in sources)
    output_type = _dtype(output, node_id)
    if (any(item != output_type for item in input_types) or output.non_overlapping is not True or
            not isinstance(contract.get("dimension"), int) or
            not isinstance(contract.get("segments"), (tuple, list)) or
            len(contract["segments"]) != len(sources)):
        raise CudaBodyError(node_id, "cat/stack requires equal dtypes, disjoint output, and a valid dimension")

    dimension = contract["dimension"]
    segments = tuple(contract["segments"])
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in segments):
        raise CudaBodyError(node_id, "cat/stack segment sizes must be non-negative constants")
    expected_maps = []
    if name == "cat":
        if not output_shape or not 0 <= dimension < len(output_shape):
            raise CudaBodyError(node_id, "cat dimension is outside the output rank")
        normalized_shapes = []
        for shape in shapes:
            if shape == (0,) and len(shape) != len(output_shape):
                normalized_shapes.append(None)
            elif len(shape) == len(output_shape):
                normalized_shapes.append(shape)
            else:
                raise CudaBodyError(node_id, "cat input rank differs from the output rank")
        nonempty = [shape for shape in normalized_shapes if shape is not None]
        if not nonempty and output_shape != (0,):
            raise CudaBodyError(node_id, "all-empty cat must produce an empty vector")
        if nonempty and any(any(shape[axis] != nonempty[0][axis]
                                for axis in range(len(output_shape)) if axis != dimension)
                            for shape in nonempty[1:]):
            raise CudaBodyError(node_id, "cat non-concatenated dimensions differ")
        exact_segments = tuple(0 if shape is None else shape[dimension] for shape in normalized_shapes)
        expected_shape = list(nonempty[0] if nonempty else (0,))
        expected_shape[dimension] = sum(exact_segments)
        if tuple(expected_shape) != output_shape or segments != exact_segments:
            raise CudaBodyError(node_id, "cat output shape or segment proof is inconsistent")
        offset = 0
        for value_id, shape, size in zip(operation.inputs, normalized_shapes, exact_segments):
            if shape is None:
                expected_maps.append(InputIndexMap(value_id, ("0",), "empty_cat"))
            else:
                expressions = [f"i{axis}" for axis in range(len(output_shape))]
                expressions[dimension] = f"i{dimension}-{offset}" if offset else f"i{dimension}"
                expected_maps.append(InputIndexMap(value_id, tuple(expressions), "cat"))
            offset += size
    else:
        if not output_shape or not 0 <= dimension < len(output_shape):
            raise CudaBodyError(node_id, "stack dimension is outside the output rank")
        if any(len(shape) != len(output_shape) - 1 or shape != shapes[0] for shape in shapes):
            raise CudaBodyError(node_id, "stack inputs must have identical shapes one rank below the output")
        expected_shape = list(shapes[0])
        expected_shape.insert(dimension, len(sources))
        if tuple(expected_shape) != output_shape or segments != (1,) * len(sources):
            raise CudaBodyError(node_id, "stack output shape or segment proof is inconsistent")
        expected_maps = [InputIndexMap(
            value_id, tuple(f"i{axis if axis < dimension else axis + 1}"
                            for axis in range(len(output_shape) - 1)), "stack"
        ) for value_id in operation.inputs]
    if tuple(expected_maps) != operation.input_index_maps:
        raise CudaBodyError(node_id, "cat/stack input maps do not match their exact source regions")

    entry = f"{_name(operation)}_tile"
    lines = _coordinates(output_shape) if _numel(output_shape) else []
    if _numel(output_shape) and name == "cat":
        start = 0
        branches = 0
        for index, size in enumerate(segments):
            end = start + size
            if size:
                condition = f"idx[{dimension}] >= {start} && idx[{dimension}] < {end}"
                lines.append(("if" if branches == 0 else "else if") + f" ({condition}) {{")
                coordinates = [f"idx[{axis}]" for axis in range(len(output_shape))]
                coordinates[dimension] = f"idx[{dimension}] - {start}"
                offset = " + ".join(
                    f"({coordinate})*{stride}"
                    for coordinate, stride in zip(coordinates, sources[index].strides) if stride
                ) or "0"
                lines.append(f"  out[{_output_offset(output, output_shape, node_id)}] = in{index}[{offset}];")
                lines.append("}")
                branches += 1
            start = end
        if branches:
            lines.append("else { continue; }")
    elif _numel(output_shape):
        branches = 0
        for index, source in enumerate(sources):
            lines.append(("if" if branches == 0 else "else if") + f" (idx[{dimension}] == {index}) {{")
            source_axes = [axis for axis in range(len(output_shape)) if axis != dimension]
            offset = " + ".join(
                f"idx[{output_axis}]*{stride}"
                for output_axis, stride in zip(source_axes, source.strides) if stride
            ) or "0"
            lines.append(f"  out[{_output_offset(output, output_shape, node_id)}] = in{index}[{offset}];")
            lines.append("}")
            branches += 1
        if branches:
            lines.append("else { continue; }")
    source = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n" + _body_loop(
        entry, _pointer_arguments(input_types, output_type), _numel(output_shape), lines
    )
    return CudaBody(source, entry, input_types, output_type, _numel(output_shape),
                    "tensor_construct", node_id)


def _emit_index_select(operation: IndexedOp, values: Mapping[str, IndexedValue],
                       output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    if len(operation.inputs) != 2 or len(operation.input_index_maps) != 2:
        raise CudaBodyError(node_id, "index_select requires one tensor and one index vector")
    source, indices = (values[value_id] for value_id in operation.inputs)
    source_shape, index_shape = _static_shape(source, node_id), _static_shape(indices, node_id)
    source_type, index_type, output_type = _dtype(source, node_id), _dtype(indices, node_id), _dtype(output, node_id)
    args = operation.attributes.get("arguments", ())
    dim = args[1] if len(args) > 1 else 0
    dim = dim + len(source_shape) if isinstance(dim, int) and dim < 0 else dim
    bounds = operation.attributes.get("index_bounds")
    if (not isinstance(dim, int) or not 0 <= dim < len(source_shape) or len(index_shape) != 1 or
            index_shape != (output_shape[dim],) or
            output_shape != source_shape[:dim] + index_shape + source_shape[dim + 1:] or
            index_type not in {"int32_t", "int64_t"} or not isinstance(bounds, Mapping) or
            bounds.get("upper_exclusive") != source_shape[dim]):
        raise CudaBodyError(node_id, "index_select lacks an exact output map or bounded index guard")
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    lines.append(f"const int64_t selected = static_cast<int64_t>(in1[idx[{dim}]*{indices.strides[0]}]);")
    lines.append(f"if (selected < 0 || selected >= {source_shape[dim]}) return;")
    offset = " + ".join(
        f"({f'selected' if axis == dim else f'idx[{axis}]'})*{stride}"
        for axis, stride in enumerate(source.strides)
    ) or "0"
    lines.append(f"const float value = {prefix}_load(in0, {offset});")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    source_code = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments((source_type, index_type), output_type), _numel(output_shape), lines
    )
    return CudaBody(source_code, entry, (source_type, index_type), output_type,
                    _numel(output_shape), "gather", node_id, dict(bounds))


def _emit_constructor(operation: IndexedOp, values: Mapping[str, IndexedValue],
                      output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    contract = operation.attributes.get("constructor")
    name = _operator(operation)
    if not isinstance(contract, Mapping) or contract.get("operator") != name:
        raise CudaBodyError(node_id, "constructor has no exact static output contract")
    output_type = _dtype(output, node_id)
    if output.non_overlapping is not True or tuple(contract.get("shape", output_shape)) != output_shape:
        raise CudaBodyError(node_id, "constructor output ownership or shape differs from its proof")
    inputs = [values[value_id] for value_id in operation.inputs]
    input_types = tuple(_dtype(value, node_id) for value in inputs)
    if (name == "new_ones" and len(inputs) != 1) or (name != "new_ones" and inputs):
        raise CudaBodyError(node_id, "constructor input metadata differs from its exact operator form")
    if name == "arange":
        start, stop, step = contract.get("start"), contract.get("stop"), contract.get("step")
        if (len(output_shape) != 1 or output_type not in {"int32_t", "int64_t"} or
                not all(isinstance(value, int) and not isinstance(value, bool)
                        for value in (start, stop, step)) or step == 0 or
                len(range(start, stop, step)) != output_shape[0]):
            raise CudaBodyError(node_id, "arange bounds do not match its dense integer output")
        expression = f"({start} + linear * {step})"
        lines = [f"out[linear] = static_cast<{output_type}>({expression});"]
    elif name in {"ones", "new_ones"}:
        if contract.get("fill") != 1 or output_type not in {"bool", "int32_t", "int64_t", "float", "__half", "__nv_bfloat16"}:
            raise CudaBodyError(node_id, "ones constructor has an unsupported value or dtype")
        lines = _coordinates(output_shape)
        lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {_helpers_store_one(output_type)};")
    else:
        raise CudaBodyError(node_id, f"constructor {name!r} has no CUDA body")
    prefix, entry = _name(operation), f"{_name(operation)}_tile"
    # Constructor bodies intentionally consume metadata-only new_ones inputs so
    # the worker ABI still preserves the FX dependency and device provenance.
    source = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments(input_types, output_type), _numel(output_shape), lines
    )
    return CudaBody(source, entry, input_types, output_type, _numel(output_shape),
                    "constructor", node_id, dict(contract))


def _helpers_store_one(dtype: str) -> str:
    return {"bool": "true", "int32_t": "1", "int64_t": "1", "float": "1.0f",
            "__half": "__float2half_rn(1.0f)",
            "__nv_bfloat16": "__float2bfloat16_rn(1.0f)"}.get(dtype, "1")


def _emit_copy(operation: IndexedOp, values: Mapping[str, IndexedValue],
               output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    if len(operation.inputs) != 1 or len(operation.input_index_maps) != 1:
        raise CudaBodyError(node_id, "copy requires one exact source map")
    source = values[operation.inputs[0]]
    source_shape = _static_shape(source, node_id)
    source_type, output_type = _dtype(source, node_id), _dtype(output, node_id)
    expected = InputIndexMap(operation.inputs[0], tuple(f"i{axis}" for axis in range(len(output_shape))), "copy")
    if (source_shape != output_shape or source_type != output_type or
            operation.input_index_maps[0] != expected or output.alias_kind != "fresh" or
            output.non_overlapping is not True):
        raise CudaBodyError(node_id, "copy shape, dtype, map, or output ownership differs from its proof")
    prefix, entry = _name(operation), f"{_name(operation)}_tile"
    lines = _coordinates(output_shape)
    source_offset = _input_offset(expected, source, output_shape, node_id)
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = in0[{source_offset}];")
    source_code = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments((source_type,), output_type), _numel(output_shape), lines
    )
    return CudaBody(source_code, entry, (source_type,), output_type, _numel(output_shape),
                    "copy", node_id)


def _emit_advanced_index(operation: IndexedOp, values: Mapping[str, IndexedValue],
                         output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    contract = operation.attributes.get("advanced_index")
    if (not isinstance(contract, Mapping) or len(operation.inputs) != 3 or
            len(operation.input_index_maps) != 3):
        raise CudaBodyError(node_id, "advanced index requires its exact bounded two-axis proof")
    base, row_index, column_index = (values[value_id] for value_id in operation.inputs)
    base_shape = _static_shape(base, node_id)
    row_shape, column_shape = _static_shape(row_index, node_id), _static_shape(column_index, node_id)
    input_types = tuple(_dtype(value, node_id) for value in (base, row_index, column_index))
    output_type = _dtype(output, node_id)
    if (tuple(contract.get("base_shape", ())) != base_shape or
            tuple(contract.get("output_shape", ())) != output_shape or len(base_shape) != 2 or
            input_types[1:] not in {("int32_t", "int32_t"), ("int64_t", "int64_t"),
                                    ("int32_t", "int64_t"), ("int64_t", "int32_t")} or
            output_type != input_types[0] or output.alias_kind != "fresh" or
            output.non_overlapping is not True):
        raise CudaBodyError(node_id, "advanced index base, output, indices, or ownership differ from its proof")
    expected_maps = [InputIndexMap(
        operation.inputs[0], tuple(contract["base_axis_expressions"]), "advanced_index")
    ]
    for value_id, shape in zip(operation.inputs[1:], (row_shape, column_shape)):
        padded = (1,) * (len(output_shape) - len(shape)) + shape
        expected_maps.append(InputIndexMap(
            value_id, tuple("0" if size == 1 else f"i{axis}"
                            for axis, size in enumerate(padded)), "advanced_index_value"
        ))
    if tuple(expected_maps) != operation.input_index_maps:
        raise CudaBodyError(node_id, "advanced index tensors do not match the exact output-axis maps")
    prefix, entry = _name(operation), f"{_name(operation)}_tile"
    lines = _coordinates(output_shape)
    row_offset = _input_offset(operation.input_index_maps[1], row_index, output_shape, node_id)
    column_offset = _input_offset(operation.input_index_maps[2], column_index, output_shape, node_id)
    lines.append(f"const int64_t row = static_cast<int64_t>(in1[{row_offset}]);")
    lines.append(f"const int64_t column = static_cast<int64_t>(in2[{column_offset}]);")
    lines.append(f"if (row < 0 || row >= {base_shape[0]} || column < 0 || column >= {base_shape[1]}) return;")
    base_offset = f"row * {base.strides[0]} + column * {base.strides[1]}"
    lines.append(f"const float value = {prefix}_load(in0, {base_offset});")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments(input_types, output_type), _numel(output_shape), lines
    )
    return CudaBody(source, entry, input_types, output_type, _numel(output_shape),
                    "advanced_index", node_id, dict(contract))


def _emit_embedding(operation: IndexedOp, values: Mapping[str, IndexedValue],
                    output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    contract = operation.attributes.get("embedding")
    if (not isinstance(contract, Mapping) or len(operation.inputs) != 2 or
            len(operation.input_index_maps) != 2):
        raise CudaBodyError(node_id, "embedding requires an exact weight, index, and output contract")
    weight, indices = (values[value_id] for value_id in operation.inputs)
    weight_shape, index_shape = _static_shape(weight, node_id), _static_shape(indices, node_id)
    weight_type, index_type, output_type = (
        _dtype(weight, node_id), _dtype(indices, node_id), _dtype(output, node_id)
    )
    if (len(weight_shape) != 2 or output_shape != index_shape + (weight_shape[1],) or
            contract.get("vocabulary") != weight_shape[0] or
            contract.get("embedding_dim") != weight_shape[1]):
        raise CudaBodyError(node_id, "embedding dimensions differ from its proof")
    bounds = operation.attributes.get("index_bounds")
    expected_maps = (
        InputIndexMap(operation.inputs[0], ("index[" + ",".join(f"i{axis}" for axis in range(len(index_shape))) + "]",
                                             f"i{len(index_shape)}"), "embedding"),
        InputIndexMap(operation.inputs[1], tuple(f"i{axis}" for axis in range(len(index_shape))), "embedding_index"),
    )
    if (weight_type != output_type or index_type not in {"int32_t", "int64_t"} or
            tuple(expected_maps) != operation.input_index_maps or not isinstance(bounds, Mapping) or
            bounds.get("upper_exclusive") != weight_shape[0] or
            output.alias_kind != "fresh" or output.non_overlapping is not True):
        raise CudaBodyError(node_id, "embedding index map, dtype, bounds, or output ownership differ from its proof")
    prefix, entry = _name(operation), f"{_name(operation)}_tile"
    lines = _coordinates(output_shape)
    index_offset = " + ".join(f"idx[{axis}]*{stride}"
                               for axis, stride in enumerate(indices.strides)) or "0"
    lines.append(f"const int64_t token = static_cast<int64_t>(in1[{index_offset}]);")
    lines.append(f"if (token < 0 || token >= {weight_shape[0]}) return;")
    weight_offset = f"token * {weight.strides[0]} + idx[{len(index_shape)}] * {weight.strides[1]}"
    lines.append(f"const float value = {prefix}_load(in0, {weight_offset});")
    lines.append(f"out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    source = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments((weight_type, index_type), output_type), _numel(output_shape), lines
    )
    return CudaBody(source, entry, (weight_type, index_type), output_type,
                    _numel(output_shape), "embedding", node_id, dict(bounds))


def _emit_index_copy(operation: IndexedOp, values: Mapping[str, IndexedValue],
                     output: IndexedValue, output_shape: tuple[int, ...], node_id: str) -> CudaBody:
    if len(operation.inputs) != 3 or len(operation.input_index_maps) != 3:
        raise CudaBodyError(node_id, "index_copy requires old state, one index, and an update")
    old, indices, update = (values[value_id] for value_id in operation.inputs)
    old_shape, index_shape, update_shape = (
        _static_shape(value, node_id) for value in (old, indices, update)
    )
    old_type, index_type, update_type, output_type = (
        _dtype(value, node_id) for value in (old, indices, update, output)
    )
    args = operation.attributes.get("arguments", ())
    dim = args[1] if len(args) > 1 else 0
    dim = dim + len(old_shape) if isinstance(dim, int) and dim < 0 else dim
    bounds = operation.attributes.get("index_bounds")
    transition = operation.attributes.get("state_transition")
    alias_rule = transition.get("alias_rule") if isinstance(transition, Mapping) else None
    transition_writers = transition.get("writer_values", ()) if isinstance(transition, Mapping) else ()
    transition_owns_output = (alias_rule == "functional_new_value" or
                              alias_rule == "functional_grouped_new_value" and
                              output.value_id in transition_writers)
    if (not isinstance(dim, int) or not 0 <= dim < len(old_shape) or output_shape != old_shape or
            index_shape != (1,) or update_shape != old_shape[:dim] + (1,) + old_shape[dim + 1:] or
            old_type != output_type or update_type != output_type or index_type not in {"int32_t", "int64_t"} or
            not isinstance(bounds, Mapping) or bounds.get("upper_exclusive") != old_shape[dim] or
            not isinstance(transition, Mapping) or not transition_owns_output):
        raise CudaBodyError(node_id, "index_copy lacks an exact functional append and bounds contract")
    prefix = _name(operation)
    entry = f"{prefix}_tile"
    lines = _coordinates(output_shape)
    lines.append(f"const int64_t selected = static_cast<int64_t>(in1[0]);")
    lines.append(f"if (selected < 0 || selected >= {old_shape[dim]}) return;")
    lines.append(f"int64_t source_offset = 0;")
    lines.append(f"if (idx[{dim}] == selected) {{")
    for axis, stride in enumerate(update.strides):
        coordinate = "0" if axis == dim else f"idx[{axis}]"
        lines.append(f"  source_offset += {coordinate} * {stride};")
    lines.append(f"  const float value = {prefix}_load(in2, source_offset);")
    lines.append(f"  out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    lines.append("} else {")
    old_offset = " + ".join(f"idx[{axis}]*{stride}" for axis, stride in enumerate(old.strides)) or "0"
    lines.append(f"  const float value = {prefix}_load(in0, {old_offset});")
    lines.append(f"  out[{_output_offset(output, output_shape, node_id)}] = {prefix}_store(value, ({output_type}*)0);")
    lines.append("}")
    source_code = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments((old_type, index_type, update_type), output_type),
        _numel(output_shape), lines,
    )
    return CudaBody(source_code, entry, (old_type, index_type, update_type), output_type,
                    _numel(output_shape), "state_write", node_id, dict(bounds))


def _emit_attention(operation: IndexedOp, values: Mapping[str, IndexedValue],
                    output: IndexedValue, output_shape: tuple[int, ...], node_id: str,
                    policy: Any) -> CudaBody:
    contract = operation.attributes.get("attention")
    if not isinstance(contract, Mapping) or len(operation.inputs) != 4:
        raise CudaBodyError(node_id, "cached attention has no proved semantic contract")
    if policy is None or not policy.reassociation_allowed("scaled_dot_product_attention"):
        raise CudaBodyError(node_id, "online softmax needs an explicit reassociation policy")
    try:
        policy.tolerance_for("scaled_dot_product_attention", output.dtype)
    except ContractError as exc:
        raise CudaBodyError(node_id, "online softmax needs an output-dtype tolerance") from exc
    q, k, v, mask = (values[item] for item in operation.inputs)
    for item in (q, k, v, mask):
        _static_shape(item, node_id)
    dtype = _dtype(q, node_id)
    if (dtype not in {"__half", "__nv_bfloat16", "float"} or
            any(_dtype(item, node_id) != dtype for item in (k, v, output)) or
            _dtype(mask, node_id) != "bool" or
            output.non_overlapping is not True or output.strides[-1] != 1 or
            contract["head_dim"] > 128):
        raise CudaBodyError(node_id, "cached attention requires dense output, supported dtype and D<=128")
    depth = contract["head_dim"]
    capacity = contract["capacity"]
    prefix, entry = _name(operation), f"{_name(operation)}_tile"
    if depth <= 32:
        work_items = contract["batch"] * contract["heads_q"] * 32
        lines = [
            "const int64_t row = linear / 32;",
            "const int lane = static_cast<int>(linear % 32);",
            f"const int64_t batch = row / {contract['heads_q']};",
            f"const int64_t head = row % {contract['heads_q']};",
            f"const int64_t kv_head = head / {contract['heads_q'] // contract['heads_kv']};",
            f"float accumulator[{depth}] = {{0.0f}};",
            "float maximum = -INFINITY;",
            "float normalizer = 0.0f;",
            "unsigned has_nan = 0;",
            f"for (int64_t token = lane; token < {capacity}; token += 32) {{",
            f"  if (!in3[batch*{mask.strides[0]} + (head % {contract['mask_heads']})*{mask.strides[1]} + token*{mask.strides[3]}]) continue;",
            "  float score = 0.0f;",
            f"  for (int d = 0; d < {depth}; ++d) {{",
            f"    const float query = {prefix}_load(in0, batch*{q.strides[0]} + head*{q.strides[1]} + d*{q.strides[3]});",
            f"    const float key = {prefix}_load(in1, batch*{k.strides[0]} + kv_head*{k.strides[1]} + token*{k.strides[2]} + d*{k.strides[3]});",
            "    score += query * key;",
            "  }",
            f"  score *= {repr(contract['scale'])}f;",
            "  if (isnan(score)) { has_nan = 1; continue; }",
            "  const float next_maximum = fmaxf(maximum, score);",
            "  const float old_scale = normalizer == 0.0f ? 0.0f : expf(maximum - next_maximum);",
            "  const float weight = expf(score - next_maximum);",
            "  normalizer = normalizer * old_scale + weight;",
            f"  for (int d = 0; d < {depth}; ++d) {{",
            f"    const float value = {prefix}_load(in2, batch*{v.strides[0]} + kv_head*{v.strides[1]} + token*{v.strides[2]} + d*{v.strides[3]});",
            "    accumulator[d] = accumulator[d] * old_scale + weight * value;",
            "  }",
            "  maximum = next_maximum;",
            "}",
            "float group_maximum = maximum;",
            "for (int offset = 16; offset > 0; offset >>= 1) {",
            "  group_maximum = fmaxf(group_maximum, __shfl_xor_sync(0xffffffff, group_maximum, offset));",
            "  has_nan |= __shfl_xor_sync(0xffffffff, has_nan, offset);",
            "}",
            "const float rescale = normalizer == 0.0f ? 0.0f : expf(maximum - group_maximum);",
            "normalizer *= rescale;",
            f"for (int d = 0; d < {depth}; ++d) accumulator[d] *= rescale;",
            "for (int offset = 16; offset > 0; offset >>= 1) {",
            "  normalizer += __shfl_xor_sync(0xffffffff, normalizer, offset);",
            f"  for (int d = 0; d < {depth}; ++d) accumulator[d] += __shfl_xor_sync(0xffffffff, accumulator[d], offset);",
            "}",
            "if (lane == 0) {",
            f"  for (int d = 0; d < {depth}; ++d) {{",
            "    const float result = has_nan ? NAN : (normalizer == 0.0f ? 0.0f : accumulator[d] / normalizer);",
            f"    out[batch*{output.strides[0]} + head*{output.strides[1]} + d] = {prefix}_store(result, ({dtype}*)0);",
            "  }",
            "}",
        ]
        source = _helpers(prefix) + _body_loop(
            entry, _pointer_arguments((dtype, dtype, dtype, "bool"), dtype),
            work_items, lines)
        return CudaBody(source, entry, (dtype, dtype, dtype, "bool"), dtype,
                        work_items, "online_cached_attention_warp32", node_id,
                        {"numerical_policy_hash": policy.contract_hash, "warp_size": 32})
    lines = [
        f"if (linear % {depth}) continue;",
        f"const int64_t row = linear / {depth};",
        f"const int64_t batch = row / {contract['heads_q']};",
        f"const int64_t head = row % {contract['heads_q']};",
        f"const int64_t kv_head = head / {contract['heads_q'] // contract['heads_kv']};",
        f"float accumulator[{depth}] = {{0.0f}};",
        "float maximum = -INFINITY;",
        "float normalizer = 0.0f;",
        "bool has_nan = false;",
        f"for (int64_t token = 0; token < {capacity}; ++token) {{",
        f"  if (!in3[batch*{mask.strides[0]} + (head % {contract['mask_heads']})*{mask.strides[1]} + token*{mask.strides[3]}]) continue;",
        "  float score = 0.0f;",
        f"  for (int d = 0; d < {depth}; ++d) {{",
        f"    const float query = {prefix}_load(in0, batch*{q.strides[0]} + head*{q.strides[1]} + d*{q.strides[3]});",
        f"    const float key = {prefix}_load(in1, batch*{k.strides[0]} + kv_head*{k.strides[1]} + token*{k.strides[2]} + d*{k.strides[3]});",
        "    score += query * key;",
        "  }",
        f"  score *= {repr(contract['scale'])}f;",
        "  if (isnan(score)) { has_nan = true; continue; }",
        "  const float next_maximum = fmaxf(maximum, score);",
        "  const float old_scale = normalizer == 0.0f ? 0.0f : expf(maximum - next_maximum);",
        "  const float weight = expf(score - next_maximum);",
        "  normalizer = normalizer * old_scale + weight;",
        f"  for (int d = 0; d < {depth}; ++d) {{",
        f"    const float value = {prefix}_load(in2, batch*{v.strides[0]} + kv_head*{v.strides[1]} + token*{v.strides[2]} + d*{v.strides[3]});",
        "    accumulator[d] = accumulator[d] * old_scale + weight * value;",
        "  }",
        "  maximum = next_maximum;",
        "}",
        f"for (int d = 0; d < {depth}; ++d) {{",
        "  const float result = has_nan ? NAN : (normalizer == 0.0f ? 0.0f : accumulator[d] / normalizer);",
        f"  out[batch*{output.strides[0]} + head*{output.strides[1]} + d] = {prefix}_store(result, ({dtype}*)0);",
        "}",
    ]
    source = _helpers(prefix) + _body_loop(
        entry, _pointer_arguments((dtype, dtype, dtype, "bool"), dtype),
        _numel(output_shape), lines)
    return CudaBody(source, entry, (dtype, dtype, dtype, "bool"), dtype,
                    _numel(output_shape), "online_cached_attention", node_id,
                    {"numerical_policy_hash": policy.contract_hash})


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
    if operation.kind == "Cat/Stack":
        return _emit_cat_stack(operation, values, output, output_shape, node_id)
    if operation.kind == "Constructor":
        return _emit_constructor(operation, values, output, output_shape, node_id)
    if operation.kind == "Copy":
        return _emit_copy(operation, values, output, output_shape, node_id)
    if operation.kind == "Gather" and _operator(operation) == "index":
        return _emit_advanced_index(operation, values, output, output_shape, node_id)
    if operation.kind == "Gather" and _operator(operation) == "embedding":
        return _emit_embedding(operation, values, output, output_shape, node_id)
    if operation.kind == "Gather" and _operator(operation) == "index_select":
        return _emit_index_select(operation, values, output, output_shape, node_id)
    if operation.kind == "Scatter/StateWrite" and _operator(operation) == "index_copy":
        return _emit_index_copy(operation, values, output, output_shape, node_id)
    if operation.kind == "Attention" and _operator(operation) == "scaled_dot_product_attention":
        return _emit_attention(operation, values, output, output_shape, node_id,
                               program.source_program.policy)
    raise CudaBodyError(node_id, f"indexed operation kind {operation.kind!r} has no generic CUDA body")


__all__ = ["CudaBody", "CudaBodyError", "emit_cuda_body"]
