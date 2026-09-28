"""Exact logical read/write maps over parametric task tiles."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import product
import re
from typing import Any, Mapping

from .domains import TileInstance


class AccessMapError(ValueError):
    pass


@dataclass(frozen=True)
class RegionMap:
    value_id: str
    expressions: tuple[str, ...]
    mode: str
    axes: tuple[str, ...]
    source_shape: tuple[Any, ...]
    iteration_shape: tuple[Any, ...]
    predicate: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"value_id": self.value_id, "expressions": list(self.expressions),
                "mode": self.mode, "axes": list(self.axes),
                "source_shape": [_stable(item) for item in self.source_shape],
                "iteration_shape": [_stable(item) for item in self.iteration_shape],
                "predicate": list(self.predicate)}


def _stable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _eval(expr: str, variables: Mapping[str, int]) -> int:
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        raise AccessMapError(f"unsupported index expression {expr!r}") from exc

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in variables:
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult,
                                                                  ast.FloorDiv, ast.Mod)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise AccessMapError("index expression divides by zero")
            return left // right if isinstance(node.op, ast.FloorDiv) else left % right
        raise AccessMapError(f"unsupported index expression {expr!r}")

    return visit(tree)


def _unravel(flat: int, shape: tuple[Any, ...]) -> tuple[int, ...]:
    if any(not isinstance(dim, int) for dim in shape):
        raise AccessMapError("reshape map has symbolic source extent")
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        extent = shape[axis]
        if extent <= 0:
            raise AccessMapError("reshape map has an empty axis")
        result[axis] = flat % extent
        flat //= extent
    if flat:
        raise AccessMapError("reshape map index is outside the source shape")
    return tuple(result)


def map_indices(region: RegionMap, variables: Mapping[str, int], *,
                dynamic_inputs: Mapping[str, Any] = ()) -> tuple[int, ...]:
    """Evaluate a supported map without executing arbitrary expression code."""
    dynamic_inputs = {} if dynamic_inputs == () else dynamic_inputs
    if region.mode == "reshape":
        if any(not isinstance(dim, int) for dim in region.iteration_shape):
            raise AccessMapError("reshape map has symbolic iteration extent")
        output_index = tuple(variables[axis] for axis in region.axes
                             if axis.startswith("i"))
        if len(output_index) != len(region.iteration_shape):
            raise AccessMapError("reshape map iteration rank mismatch")
        flat = 0
        for index, extent in zip(output_index, region.iteration_shape):
            flat = flat * extent + index
        return _unravel(flat, region.source_shape)
    if region.mode in {"indirect_select", "index"}:
        if region.mode == "indirect_select":
            values = dynamic_inputs.get("index")
            if values is None:
                raise AccessMapError("indirect index requires a runtime index binding")
            import torch
            result = []
            for expression in region.expressions:
                indirect = re.fullmatch(r"index\[(i\d+)\]", expression)
                if indirect is None:
                    result.append(_eval(expression, variables))
                    continue
                index = variables[indirect.group(1)]
                result.append(int(values[index].item()) if isinstance(values, torch.Tensor)
                              else int(values[index]))
            return tuple(result)
        return tuple(_eval(expr, variables) for expr in region.expressions)
    if region.mode in {"indirect", "strided_view"} or any(expr == "UNKNOWN" for expr in region.expressions):
        raise AccessMapError(f"{region.mode} does not have a statically evaluable access map")
    return tuple(_eval(expr, variables) for expr in region.expressions)


def region_for_input(operation: Any, input_map: Any, values: Mapping[str, Any]) -> RegionMap:
    source = values[input_map.value_id]
    return RegionMap(input_map.value_id, tuple(input_map.expressions), input_map.mode,
                     tuple(axis.name for axis in operation.iteration_domain + operation.reduction_domain),
                     tuple(source.shape), tuple(axis.extent for axis in operation.iteration_domain))


def region_for_output(operation: Any, value_id: str, values: Mapping[str, Any], *,
                      predicate: tuple[str, ...] = ()) -> RegionMap:
    output = values[value_id]
    return RegionMap(value_id, tuple(operation.output_index_map), "affine",
                     tuple(axis.name for axis in operation.iteration_domain),
                     tuple(output.shape), tuple(axis.extent for axis in operation.iteration_domain), predicate)


def tile_variables(tile: TileInstance):
    bounds = tile.axis_bounds()
    names = [axis.name for axis in tile.domain.output_axes + tile.domain.reduction_axes]
    ranges = [range(*bounds[name]) for name in names]
    for indices in (product(*ranges) if ranges else [()]):
        yield dict(zip(names, indices))


def enumerate_region(region: RegionMap, tile: TileInstance, *,
                     dynamic_inputs: Mapping[str, Any] = (), limit: int = 1_000_000) -> tuple[tuple[int, ...], ...]:
    """Enumerate concrete addresses for tiny fixtures; production plans stay symbolic."""
    count = 1
    for axis in tile.domain.output_axes + tile.domain.reduction_axes:
        start, end = tile.axis_bounds()[axis.name]
        count *= end - start
    if count > limit:
        raise AccessMapError(f"access enumeration exceeds limit {limit}")
    return tuple(map_indices(region, variables, dynamic_inputs=dynamic_inputs)
                 for variables in tile_variables(tile))


def index_expressions_supported(operation: Any) -> bool:
    try:
        for access in operation.input_index_maps:
            # Indirect maps remain exact symbolic guards; enumeration needs values.
            if access.mode in {"indirect_select", "index"}:
                continue
            if access.mode == "reshape":
                continue
            for expression in access.expressions:
                if expression != "UNKNOWN":
                    _eval(expression, {"i0": 1, "i1": 1, "i2": 1, "i3": 1,
                                       "k": 1, "r0": 1, "r1": 1, "r2": 1})
        for expression in operation.output_index_map:
            if expression == "UNKNOWN":
                return False
            _eval(expression, {"i0": 1, "i1": 1, "i2": 1, "i3": 1,
                               "k": 1, "r0": 1, "r1": 1, "r2": 1})
        return not any(expression == "UNKNOWN" for access in operation.input_index_maps
                       for expression in access.expressions)
    except AccessMapError:
        return False


__all__ = ["AccessMapError", "RegionMap", "enumerate_region", "index_expressions_supported",
           "map_indices", "region_for_input", "region_for_output", "tile_variables"]
