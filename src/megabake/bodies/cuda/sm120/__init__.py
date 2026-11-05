"""Blackwell Ultra (SM12x): decode-oriented GEMM and attention bodies."""

from ...interface import Generation
from .attention import SM12xAttentionBodyProvider
from .gemm import SM12xGemmProvider

SM12X_ARCHITECTURES = frozenset({"sm120", "sm121"})

GENERATION = Generation(
    architectures=SM12X_ARCHITECTURES,
    providers=(SM12xGemmProvider(), SM12xAttentionBodyProvider()),
)
