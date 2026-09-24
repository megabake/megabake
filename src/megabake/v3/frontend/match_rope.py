"""RoPE records only graph-proven pairing and supplied frequency inputs."""

from __future__ import annotations

from typing import Any

from .capture import NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


def _target(node: Any) -> str:
    return str(getattr(node, "target", ""))


def _is(node: Any, *suffixes: str) -> bool:
    return any(_target(node).endswith(suffix) for suffix in suffixes)


def _mul_operands(node: Any) -> tuple[Any, Any] | None:
    return tuple(node.args[:2]) if _is(node, "mul.Tensor", "mul.default") and len(node.args) >= 2 else None


def _rotation_terms(node: Any) -> tuple[tuple[Any, Any], tuple[Any, Any], str] | None:
    """Return the two products of a signed rotation term."""
    if not _is(node, "add.Tensor", "add.Scalar", "sub.Tensor", "sub.Scalar") or len(node.args) != 2:
        return None
    first, second = _mul_operands(node.args[0]), _mul_operands(node.args[1])
    if first is None or second is None:
        return None
    return first, second, "sub" if "sub." in _target(node) else "add"


def _half_split(node: Any, facts: FactTable) -> dict[str, Any] | None:
    if not _is(node, "cat.default") or not node.args:
        return None
    values = node.args[0]
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        return None
    left, right = _rotation_terms(values[0]), _rotation_terms(values[1])
    if left is None or right is None or left[2] != "sub" or right[2] != "add":
        return None
    # Terms must contain the same two source halves and the same sin/cos
    # values, crossed in the second equation.  Object identity is deliberate:
    # similar arithmetic with unrelated operands is not RoPE.
    (lx, lc), (ly, ls), _ = left
    (rx, rc), (ry, rs), _ = right
    if not (lx is ry and ly is rx and lc is rc and ls is rs):
        return None
    out_fact = facts.for_node(node)
    left_fact = facts.for_node(values[0])
    if not out_fact or not left_fact or not isinstance(left_fact.shape[-1], int):
        return None
    return {
        "pairing": "half_split", "rotary_dim": left_fact.shape[-1] * 2,
        "positions": getattr(lc, "name", None), "frequency": getattr(ls, "name", None),
        "scaling": "none", "target": _target(node), "preserves_nonrotary": out_fact.shape[-1] != left_fact.shape[-1] * 2,
    }


def _interleaved(node: Any, facts: FactTable) -> dict[str, Any] | None:
    # Common explicit form: stack((-odd, even), dim=-1).flatten(...).  We do
    # not need to trust its name; the stack of two signed, same-source lanes is
    # the proof of the pairing convention.
    if not _is(node, "stack.default") or not node.args:
        return None
    values = node.args[0]
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        return None
    first, second = values
    if not (_is(first, "neg.default") and getattr(first, "args", ())):
        return None
    source_a, source_b = first.args[0], second
    a_fact, b_fact = facts.for_node(source_a), facts.for_node(source_b)
    if not a_fact or not b_fact or a_fact.shape != b_fact.shape or not isinstance(a_fact.shape[-1], int):
        return None
    return {"pairing": "interleaved", "rotary_dim": a_fact.shape[-1] * 2,
            "positions": None, "frequency": None, "scaling": "none", "target": _target(node),
            "preserves_nonrotary": False}


def match_rope(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = _target(node).lower()
        attrs = _half_split(node, facts) or _interleaved(node, facts)
        # A preserved rotary op is acceptable only when its schema itself names
        # the convention and dimensions.  Metadata alone is never proof.
        if attrs is None and ("rope" in target or "rotary" in target):
            pairing = node.kwargs.get("pairing")
            rotary_dim = node.kwargs.get("rotary_dim")
            if pairing in {"half_split", "interleaved"} and isinstance(rotary_dim, int) and rotary_dim > 0:
                attrs = {"pairing": pairing, "rotary_dim": rotary_dim,
                         "positions": node.kwargs.get("positions"), "frequency": node.kwargs.get("frequency"),
                         "scaling": node.kwargs.get("scaling", "none"), "target": _target(node),
                         "preserves_nonrotary": False}
        if attrs is not None:
            matches.append(semantic_node(node, facts, "RoPE", attributes=attrs))
    return tuple(matches)
