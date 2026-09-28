"""Backend-neutral indexed tensor semantics and local FX references."""

from .indexed import IndexedTensorProgram, lower_indexed_program
from .verify import verify_algorithm_choice_cover, verify_indexed_program

__all__ = ["IndexedTensorProgram", "lower_indexed_program", "verify_algorithm_choice_cover",
           "verify_indexed_program"]
