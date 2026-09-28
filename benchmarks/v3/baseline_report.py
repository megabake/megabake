"""Validation and selection rules for matched V3R-003 baseline records."""

from __future__ import annotations

import statistics
from typing import Any, Iterable


def rejection_reasons(candidate: dict[str, Any]) -> list[str]:
    checks = {
        "compiled": "compile failed",
        "equivalent_outputs": "output tree or logits differ",
        "equivalent_state": "KV state differs",
        "old_state_unchanged": "old KV input was mutated",
        "output_ownership": "returned outputs are reused across calls",
        "unfiltered_trace": "complete operation trace is missing",
    }
    reasons = [reason for key, reason in checks.items() if not candidate.get(key, False)]
    if candidate.get("graph_break_count") != 0:
        reasons.append("torch.compile graph breaks are present or unknown")
    if candidate.get("cudagraph_requested") and not candidate.get("cudagraph_observed"):
        reasons.append("requested CUDA Graph capture was not verified")
    if not candidate.get("complete_call_wall_us"):
        reasons.append("complete-call sample series is empty")
    return reasons


def select_best_baseline(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in candidates if not rejection_reasons(item)]
    return min(eligible, key=lambda item: statistics.median(item["complete_call_wall_us"]), default=None)
