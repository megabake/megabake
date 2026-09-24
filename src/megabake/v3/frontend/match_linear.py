"""Schema-aware recognition of the narrow supported linear forms."""

from __future__ import annotations

from typing import Any

from .capture import NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


def _number(value: Any, default: float) -> float | None:
    return float(value) if isinstance(value, (int, float)) else default if value is None else None


def _schema_value(node: Any, name: str, position: int, default: Any) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    return node.args[position] if len(node.args) > position else default


def match_linear(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        simple = target.endswith("linear.default")
        mm = target.endswith("mm.default") or target.endswith("matmul.default")
        addmm = target.endswith("addmm.default")
        if not (simple or mm or addmm):
            continue
        args = node.args
        if simple and len(args) >= 2:
            activation, weight = args[0], args[1]
            bias = _schema_value(node, "bias", 2, None)
            alpha, beta, layout = 1.0, 1.0, "NK"
        elif addmm and len(args) >= 3:
            bias, activation, weight = args[:3]
            beta = _number(_schema_value(node, "beta", 3, 1), 1.0)
            alpha = _number(_schema_value(node, "alpha", 4, 1), 1.0)
            layout = "KN"
        elif mm and len(args) >= 2:
            activation, weight, bias = args[0], args[1], None
            alpha, beta, layout = 1.0, 0.0, "KN"
        else:
            continue
        if alpha is None or beta is None:
            continue
        activation_fact, weight_fact, output_fact = facts.for_node(activation), facts.for_node(weight), facts.for_node(node)
        if not activation_fact or not weight_fact or not output_fact or len(activation_fact.shape) not in (1, 2) or len(weight_fact.shape) != 2:
            continue
        m = 1 if len(activation_fact.shape) == 1 else activation_fact.shape[-2]
        k = activation_fact.shape[-1]
        n = output_fact.shape[-1] if output_fact.shape else None
        if layout == "NK" and tuple(weight_fact.shape) != (n, k):
            continue
        if layout == "KN" and tuple(weight_fact.shape) != (k, n):
            continue
        bias_fact = facts.for_node(bias) if hasattr(bias, "name") else None
        matches.append(semantic_node(node, facts, "Linear", attributes={
            "M": m, "N": n, "K": k, "weight_layout": layout,
            "weight_strides": weight_fact.strides, "bias": bias is not None,
            "bias_shape": bias_fact.shape if bias_fact else None,
            "alpha": alpha, "beta": beta, "output_dtype": output_fact.dtype,
            "input_dtype": activation_fact.dtype, "weight_dtype": weight_fact.dtype,
            "input_offset": activation_fact.storage_offset, "weight_offset": weight_fact.storage_offset,
            "weight_alias": weight_fact.alias_set, "weight_role": weight_fact.role,
        }))
    return tuple(matches)
