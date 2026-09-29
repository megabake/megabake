"""Higher-level, backend-neutral V3 execution algorithms."""

from .choices import AlgorithmChoice, choice_guard_failures, enumerate_algorithm_choices
from .attention import materialized_cached_attention
from .repeat import CarriedValue, RepeatRegion, RepeatRegionError, recover_repeat_region

__all__ = [
    "AlgorithmChoice", "CarriedValue", "RepeatRegion", "RepeatRegionError",
    "choice_guard_failures", "enumerate_algorithm_choices", "recover_repeat_region",
    "materialized_cached_attention",
]
