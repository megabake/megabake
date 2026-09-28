"""CPU-only FX frontend for the opt-in V3 compiler path."""

from .capture import (
    BindingError,
    FrontendError,
    NormalizedProgram,
    UnsupportedGraphError,
    capture_exported_program,
    capture_graph_module,
    graph_hash,
)
from .normalize import normalize_fx
from .facts import FactTable, TensorFacts, collect_facts
from .semantic import SemanticGraph, SemanticRegistry, index_program, recognize

__all__ = [
    "BindingError", "FactTable", "FrontendError", "NormalizedProgram",
    "SemanticGraph", "SemanticRegistry", "TensorFacts", "UnsupportedGraphError",
    "capture_exported_program", "capture_graph_module", "collect_facts", "graph_hash", "index_program", "normalize_fx",
    "recognize",
]
