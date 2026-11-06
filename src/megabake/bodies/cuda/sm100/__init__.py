"""Blackwell bodies: SM100 CUTLASS GEMM and warp-MMA fallback."""

from ...interface import Generation
from . import tuning
from .attention import AttentionBodyProvider
from .gemm import BLACKWELL_ARCHITECTURES, BlackwellGemmProvider
from .table import TABLE_GEMM

GENERATION = Generation(
    architectures=BLACKWELL_ARCHITECTURES,
    providers=(BlackwellGemmProvider(), AttentionBodyProvider()),
    table_gemms=(TABLE_GEMM,),
    tuner=tuning,
)
