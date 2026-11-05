"""SM12x stateful decode attention provider."""

from ..sm90.attention import AttentionBodyProvider


class SM12xAttentionBodyProvider(AttentionBodyProvider):
    """Select the CUDA-core attention body on SM120 and SM121."""

    provider_name = "sm12x_attention"
    architectures = frozenset({"sm120", "sm121"})
    requires_wgmma = False
