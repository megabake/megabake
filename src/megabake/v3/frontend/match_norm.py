"""Guarded RMSNorm recognition; near misses intentionally remain reference FX."""

from __future__ import annotations

from typing import Any

from .capture import NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


def _target(node: Any) -> str:
    return str(getattr(node, "target", ""))


def _is(node: Any, *suffixes: str) -> bool:
    return any(_target(node).endswith(suffix) for suffix in suffixes)


def _axes(node: Any) -> Any:
    if "dim" in getattr(node, "kwargs", {}):
        return node.kwargs["dim"]
    if len(getattr(node, "args", ())) > 1:
        return node.args[1]
    return None


def _weight_transform(node: Any) -> tuple[Any, str]:
    if _is(node, "add.Tensor", "add.Scalar") and len(node.args) == 2:
        left, right = node.args
        if left == 1 or left == 1.0:
            return right, "one_plus_weight"
        if right == 1 or right == 1.0:
            return left, "one_plus_weight"
    return node, "weight"


def _explicit_chain(final: Any) -> tuple[Any, Any, Any, Any, str, tuple[Any, ...]] | None:
    """Prove ``x * rsqrt(mean(square(x)) + eps) * weight`` exactly."""
    if not _is(final, "mul.Tensor", "mul.default") or len(final.args) != 2:
        return None
    for normalized, scale in ((final.args[0], final.args[1]), (final.args[1], final.args[0])):
        if not _is(normalized, "mul.Tensor", "mul.default") or len(normalized.args) != 2:
            continue
        for x, reciprocal in ((normalized.args[0], normalized.args[1]), (normalized.args[1], normalized.args[0])):
            if not _is(reciprocal, "rsqrt.default") or not reciprocal.args:
                continue
            add = reciprocal.args[0]
            if not _is(add, "add.Tensor", "add.Scalar") or len(add.args) != 2:
                continue
            mean, eps = add.args
            if not _is(mean, "mean.dim", "mean.default") or not mean.args:
                continue
            square = mean.args[0]
            if not (_is(square, "square.default") or _is(square, "pow.Tensor_Scalar")) or not square.args:
                continue
            if square.args[0] is not x:
                continue
            weight, transform = _weight_transform(scale)
            return x, _axes(mean), eps, weight, transform, (square, mean, add, reciprocal, normalized, final)
    return None


def _cast_user(node: Any) -> Any | None:
    casts = [user for user in node.users if _is(user, "to.dtype")]
    return casts[0] if len(casts) == 1 else None


def match_rmsnorm(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = _target(node)
        if target.endswith("rms_norm.default"):
            x = node.args[0] if node.args else None
            dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("normalized_shape")
            weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
            eps = node.args[3] if len(node.args) > 3 else node.kwargs.get("eps")
            x_fact = facts.for_node(x)
            if not x_fact or not isinstance(dims, (tuple, list)) or tuple(dims) != tuple(x_fact.shape[-len(dims):]):
                continue
            cast = _cast_user(node)
            origins = (node, cast) if cast is not None else (node,)
            matches.append(semantic_node(node if cast is None else cast, facts, "RMSNorm", attributes={
                "axes": tuple(range(-len(dims), 0)), "normalized_shape": tuple(dims), "eps": eps,
                "eps_placement": "inside_rsqrt", "weight_transform": "weight",
                "output_cast": _target(cast) if cast is not None else None,
                "input": getattr(x, "name", None), "weight": getattr(weight, "name", None),
            }, origins=origins))
            continue
        chain = _explicit_chain(node)
        if chain is None:
            continue
        x, axes, eps, weight, transform, origins = chain
        x_fact = facts.for_node(x)
        if not x_fact or axes is None:
            continue
        axes_tuple = (axes,) if isinstance(axes, int) else tuple(axes) if isinstance(axes, (tuple, list)) else None
        if not axes_tuple or tuple(axes_tuple) != tuple(range(-len(axes_tuple), 0)):
            continue
        cast = _cast_user(node)
        target_node = cast if cast is not None else node
        matches.append(semantic_node(target_node, facts, "RMSNorm", attributes={
            "axes": axes_tuple, "eps": eps, "eps_placement": "inside_rsqrt",
            "weight_transform": transform, "output_cast": _target(cast) if cast is not None else None,
            "input": getattr(x, "name", None), "weight": getattr(weight, "name", None),
            "explicit_chain": True,
        }, origins=origins + ((cast,) if cast is not None else ())))
    return tuple(matches)
