"""Materialized reference for the guarded one-token cached SDPA algorithm."""


def materialized_cached_attention(q, cache_k, cache_v, mask, *, scale=None):
    import torch

    group = q.shape[1] // cache_k.shape[1]
    key = cache_k.repeat_interleave(group, dim=1).float()
    value = cache_v.repeat_interleave(group, dim=1).float()
    scores = torch.matmul(q.float(), key.transpose(-2, -1))
    scores = scores * (q.shape[-1] ** -0.5 if scale is None else scale)
    scores = scores.masked_fill(~mask, -torch.inf)
    has_key = mask.any(dim=-1, keepdim=True)
    safe_scores = torch.where(has_key, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(has_key, probabilities, torch.zeros_like(probabilities))
    return torch.matmul(probabilities, value).to(q.dtype)
