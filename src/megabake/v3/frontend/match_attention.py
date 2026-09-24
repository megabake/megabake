"""Narrow SDPA matcher that retains all mask/head/output semantics."""

from __future__ import annotations

from .capture import NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


def _arg(node: object, index: int, name: str, default: object = None) -> object:
    if name in getattr(node, "kwargs", {}):
        return node.kwargs[name]
    args = getattr(node, "args", ())
    return args[index] if len(args) > index else getattr(node, "kwargs", {}).get(name, default)


def _validate_shapes(q: object, k: object, v: object, facts: FactTable, enable_gqa: bool) -> tuple[object, object, object] | None:
    q_fact, k_fact, v_fact = facts.for_node(q), facts.for_node(k), facts.for_node(v)
    if not q_fact or not k_fact or not v_fact:
        return None
    if any(len(fact.shape) != 4 for fact in (q_fact, k_fact, v_fact)):
        return None
    if q_fact.shape[0] != k_fact.shape[0] or k_fact.shape[0] != v_fact.shape[0]:
        return None
    if k_fact.shape[1] != v_fact.shape[1] or k_fact.shape[-2] != v_fact.shape[-2]:
        return None
    if q_fact.shape[-1] != k_fact.shape[-1] or k_fact.shape[-1] != v_fact.shape[-1]:
        return None
    if q_fact.shape[1] != k_fact.shape[1] and (not enable_gqa or not isinstance(q_fact.shape[1], int)
                                                or not isinstance(k_fact.shape[1], int)
                                                or q_fact.shape[1] % k_fact.shape[1] != 0):
        return None
    return q_fact, k_fact, v_fact


def _mask_kind(mask: object, facts: FactTable) -> str | None:
    if mask is None:
        return "none"
    fact = facts.for_node(mask)
    if not fact or len(fact.shape) > 4:
        return None
    return "boolean" if fact.dtype == "bool" else "additive"


def _target(node: object) -> str:
    return str(getattr(node, "target", ""))


def _is(node: object, *suffixes: str) -> bool:
    return any(_target(node).endswith(suffix) for suffix in suffixes)


def _decomposed_sdpa(node: object, facts: FactTable) -> SemanticNode | None:
    """Recognize the exact fixture-shaped matmul/scale/mask/softmax/matmul form."""
    if not _is(node, "matmul.default", "mm.default") or len(getattr(node, "args", ())) != 2:
        return None
    softmax, v = node.args
    if not _is(softmax, "softmax.int", "_softmax.default") or not softmax.args:
        return None
    score = softmax.args[0]
    mask = None
    if _is(score, "add.Tensor", "add.Scalar") and len(score.args) == 2:
        left, right = score.args
        score, mask = (left, right) if hasattr(left, "op") else (right, left)
    scale = None
    if _is(score, "mul.Tensor", "mul.Scalar") and len(score.args) == 2:
        left, right = score.args
        score, scale = (left, right) if hasattr(left, "op") else (right, left)
    if not _is(score, "matmul.default", "mm.default") or len(score.args) != 2:
        return None
    q, kt = score.args
    k = kt.args[0] if _is(kt, "transpose.int", "permute.default") and kt.args else None
    if k is None:
        return None
    shape_facts = _validate_shapes(q, k, v, facts, False)
    mask_kind = _mask_kind(mask, facts)
    if not shape_facts or mask_kind is None:
        return None
    q_fact, k_fact, v_fact = shape_facts
    origins = tuple(item for item in (score, kt, softmax, node) if item is not None)
    if mask is not None and hasattr(mask, "op"):
        origins += (mask,)
    if scale is not None and hasattr(scale, "op"):
        origins += (scale,)
    return semantic_node(node, facts, "SDPA", attributes={
        "batch": q_fact.shape[0], "heads_q": q_fact.shape[1], "heads_kv": k_fact.shape[1],
        "head_dim": q_fact.shape[-1], "query_length": q_fact.shape[-2], "key_length": k_fact.shape[-2],
        "gqa": False, "enable_gqa": False, "scale": scale, "mask": mask_kind,
        "is_causal": False, "dropout_p": 0.0, "valid_lengths": "key_length" if mask is None else "mask_constrained",
        "output_dtype": facts.for_node(node).dtype if facts.for_node(node) else None, "decomposed": True,
    }, origins=origins)


def match_attention(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function" or "scaled_dot_product_attention" not in str(node.target):
            decomposed = _decomposed_sdpa(node, facts) if node.op == "call_function" else None
            if decomposed is not None:
                matches.append(decomposed)
            continue
        q, k, v = _arg(node, 0, "query"), _arg(node, 1, "key"), _arg(node, 2, "value")
        dropout = _arg(node, 4, "dropout_p", 0.0)
        if dropout not in (0, 0.0, None):
            continue
        mask = _arg(node, 3, "attn_mask")
        enable_gqa = bool(_arg(node, 7, "enable_gqa", False))
        shape_facts = _validate_shapes(q, k, v, facts, enable_gqa)
        mask_kind = _mask_kind(mask, facts)
        if not shape_facts or mask_kind is None:
            continue
        q_fact, k_fact, v_fact = shape_facts
        hq, hkv = q_fact.shape[1], k_fact.shape[1]
        matches.append(semantic_node(node, facts, "SDPA", attributes={
            "batch": q_fact.shape[0], "heads_q": hq, "heads_kv": hkv, "head_dim": q_fact.shape[-1],
            "query_length": q_fact.shape[-2], "key_length": k_fact.shape[-2], "gqa": hq != hkv,
            "scale": _arg(node, 6, "scale"),
            "mask": mask_kind,
            "is_causal": bool(_arg(node, 5, "is_causal", False)), "dropout_p": 0.0,
            "valid_lengths": "key_length" if mask is None else "mask_constrained",
            "output_dtype": facts.for_node(node).dtype if facts.for_node(node) else None,
            "q_strides": q_fact.strides, "k_strides": k_fact.strides, "v_strides": v_fact.strides,
            "enable_gqa": enable_gqa, "decomposed": False,
        }))
    return tuple(matches)
