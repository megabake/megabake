"""Exact, inspectable pointwise DAG recognition; no activation-name guessing."""

from __future__ import annotations

from .capture import NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


def _target(node: object) -> str:
    return str(getattr(node, "target", ""))


def _fact_summary(node: object, facts: FactTable) -> dict[str, object]:
    fact = facts.for_node(node)
    return {} if fact is None else {"shape": fact.shape, "dtype": fact.dtype, "strides": fact.strides}


def _constant_or_node(value: object) -> object:
    return getattr(value, "name", value)


def match_pointwise(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    swiglu_nodes = set()
    # Prefer the larger exact region, while leaving non-overlapping pointwise
    # expressions visible.  Recognition must not consume SiLU before seeing its
    # multiply consumer.
    for node in program.graph_module.graph.nodes:
        target = _target(node)
        if node.op != "call_function" or not (target.endswith("mul.Tensor") or target.endswith("mul.default")) or len(node.args) < 2:
            continue
        left, right = node.args[:2]
        silu = left if _target(left).endswith("silu.default") else right if _target(right).endswith("silu.default") else None
        if silu is None:
            continue
        other = right if silu is left else left
        gate_fact, up_fact, output_fact = facts.for_node(silu.args[0]), facts.for_node(other), facts.for_node(node)
        matches.append(semantic_node(node, facts, "SwiGLU", attributes={
            "activation": "silu", "gate": getattr(silu.args[0], "name", None), "up": getattr(other, "name", None),
            "operand_order": (getattr(left, "name", str(left)), getattr(right, "name", str(right))),
            "gate_shape": gate_fact.shape if gate_fact else None,
            "up_shape": up_fact.shape if up_fact else None,
            "output_dtype": output_fact.dtype if output_fact else None,
            "broadcast": bool(gate_fact and up_fact and gate_fact.shape != up_fact.shape),
        }, origins=(silu, node)))
        swiglu_nodes.update((silu.name, node.name))
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if node.name in swiglu_nodes:
            continue
        target = _target(node)
        if target.endswith("mul.Tensor") or target.endswith("mul.default"):
            left, right = node.args[:2] if len(node.args) >= 2 else (None, None)
            matches.append(semantic_node(node, facts, "Pointwise", attributes={
                "expression": "mul", "operand_order": (_constant_or_node(left), _constant_or_node(right)),
                "left": _fact_summary(left, facts), "right": _fact_summary(right, facts),
                "output": _fact_summary(node, facts),
            }))
        elif any(target.endswith(suffix) for suffix in ("silu.default", "gelu.default", "add.Tensor", "sub.Tensor", "div.Tensor", "to.dtype")):
            expression = "gelu" if target.endswith("gelu.default") else "silu" if target.endswith("silu.default") else target.rsplit(".", 2)[-2]
            matches.append(semantic_node(node, facts, "Pointwise", attributes={
                "expression": expression, "target": target, "kwargs": dict(node.kwargs),
                "cast_boundary": target.endswith("to.dtype"),
                "gelu_approximate": node.kwargs.get("approximate", node.args[1] if expression == "gelu" and len(node.args) > 1 else None),
                "inputs": tuple(_constant_or_node(arg) for arg in node.args),
                "input_facts": tuple(_fact_summary(arg, facts) for arg in node.args),
                "output": _fact_summary(node, facts),
            }))
    return tuple(matches)
