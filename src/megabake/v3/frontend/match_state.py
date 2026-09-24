"""Functional fixed-capacity KV updates and conservative FX recognition."""

from __future__ import annotations

from typing import Any

from .capture import FrontendError, NormalizedProgram
from .facts import FactTable
from .semantic import SemanticNode, semantic_node


class CacheSemanticsError(FrontendError):
    pass


def _contiguous(shape: Any, strides: Any) -> bool:
    expected = 1
    for dim, stride in zip(reversed(tuple(shape)), reversed(tuple(strides))):
        if not isinstance(dim, int) or stride != expected:
            return False
        expected *= dim
    return True


def validate_kv_cache(cache_k: Any, cache_v: Any, *, heads_kv: int, head_dim: int) -> None:
    """Validate the fixed B/Hkv/capacity/D cache contract before matching it."""
    for name, cache in (("K", cache_k), ("V", cache_v)):
        if len(getattr(cache, "shape", ())) != 4 or cache.shape[1] != heads_kv or cache.shape[-1] != head_dim:
            raise CacheSemanticsError(f"{name} cache does not match GQA head/cache dimensions")
        if not bool(getattr(cache, "is_contiguous", lambda: False)()):
            raise CacheSemanticsError(f"{name} cache must be contiguous for the initial fixed-capacity contract")
    if cache_k.shape != cache_v.shape:
        raise CacheSemanticsError("K and V cache shapes must match")


def functional_cache_update(cache: Any, update: Any, position: Any, *, valid_length: int | None = None) -> tuple[Any, int]:
    """Return a new contiguous cache with exactly one append-only slot replaced."""
    capacity = cache.shape[-2]
    if not bool(getattr(cache, "is_contiguous", lambda: False)()):
        raise CacheSemanticsError("cache must be contiguous")
    if getattr(update, "shape", ())[:-2] != cache.shape[:-2] or update.shape[-2] != 1 or update.shape[-1] != cache.shape[-1]:
        raise CacheSemanticsError("cache/update shapes must agree except for one sequence slot")
    import torch
    if isinstance(position, int):
        if (position < 0 or position >= capacity or (valid_length is not None and
                (valid_length < 0 or valid_length >= capacity or position != valid_length))):
            raise CacheSemanticsError(f"append-only cache position {position} is outside valid range")
        index, next_valid = torch.tensor([position], device=cache.device), position + 1
    elif hasattr(position, "reshape") and getattr(position, "numel", lambda: 0)() == 1:
        # Keep this tensor path capture-safe: index_copy supplies bounds checks
        # and the next valid length remains a graph value, not ``.item()``.
        dtype = str(getattr(position, "dtype", "")).removeprefix("torch.")
        if dtype not in {"int32", "int64"}:
            raise CacheSemanticsError("tensor position must have an integer dtype")
        index, next_valid = position.reshape(1).to(device=cache.device, dtype=torch.long), position + 1
    else:
        raise CacheSemanticsError("position must be an integer or one-element integer tensor")
    result = cache.clone()
    result.index_copy_(-2, index, update)
    return result, next_valid


def match_cache_updates(program: NormalizedProgram, facts: FactTable) -> tuple[SemanticNode, ...]:
    matches: list[SemanticNode] = []
    for node in program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if not any(name in target for name in ("index_copy", "scatter", "slice_scatter")):
            continue
        cache = node.args[0] if node.args else None
        cache_fact = facts.for_node(cache)
        output_fact = facts.for_node(node)
        if (not cache_fact or not output_fact or len(cache_fact.shape) != 4 or cache_fact.shape != output_fact.shape
                or not _contiguous(cache_fact.shape, cache_fact.strides)):
            continue
        if "slice_scatter" in target:
            index, update = None, node.args[1] if len(node.args) > 1 else node.kwargs.get("src")
        else:
            # index_copy/scatter schemas are self, dim, index, source.
            index = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
            update = node.args[3] if len(node.args) > 3 else node.kwargs.get("source", node.kwargs.get("src"))
        index_fact, update_fact = facts.for_node(index), facts.for_node(update)
        if (not update_fact or update_fact.shape[:-2] != cache_fact.shape[:-2]
                or update_fact.shape[-2:] != (1, cache_fact.shape[-1])):
            continue
        if index is not None and not index_fact:
            continue
        matches.append(semantic_node(node, facts, "CacheUpdate", attributes={
            "capacity": cache_fact.shape[-2], "layout": cache_fact.strides,
            "old_state": getattr(cache, "name", None), "append_only": True,
            "position": getattr(index, "name", None), "valid_length_after": "position_plus_one",
            "new_state": node.name, "contiguous": True,
        }, effects=(f"state:{cache_fact.value_id}",)))
    return tuple(matches)
