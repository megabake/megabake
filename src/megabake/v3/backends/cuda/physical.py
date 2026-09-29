"""Bounded target-specific whole-entry candidates for the first CUDA slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

from ...contracts import canonical_json
from ...logical import LogicalExecutionPlan
from .bodies.registry import BodyTacticRegistry, BodyTacticSpec
from .profile import CudaTargetProfile


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class PhysicalCandidate:
    candidate_id: str
    indexed_program_hash: str
    logical_plan_hash: str
    target_profile_key: str
    operation_id: str
    tactic_id: str
    algorithm_id: str
    provider: str
    provider_version: str
    body_source_hash: str
    target: str
    block_threads: int
    outputs_per_cta: int
    grid_ctas: int
    output_elements: int
    reduction_participants: int
    control_candidate_id: str
    ownership: Mapping[str, Any]
    placement: Mapping[str, Any]
    event_protocol: Mapping[str, Any]
    resource_estimates: Mapping[str, Any]
    progress_obligations: tuple[str, ...]
    selection_score: tuple[int, int, int]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported physical candidate schema {self.schema_version}")
        if self.grid_ctas != math.ceil(self.output_elements / self.outputs_per_cta):
            raise ValueError("grid does not cover the declared output tiles")
        if self.block_threads <= 0 or self.outputs_per_cta <= 0:
            raise ValueError("candidate block and output tile sizes must be positive")
        canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "indexed_program_hash": self.indexed_program_hash,
            "logical_plan_hash": self.logical_plan_hash,
            "target_profile_key": self.target_profile_key,
            "operation_id": self.operation_id,
            "tactic_id": self.tactic_id,
            "algorithm_id": self.algorithm_id,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "body_source_hash": self.body_source_hash,
            "target": self.target,
            "block_threads": self.block_threads,
            "outputs_per_cta": self.outputs_per_cta,
            "grid_ctas": self.grid_ctas,
            "output_elements": self.output_elements,
            "reduction_participants": self.reduction_participants,
            "control_candidate_id": self.control_candidate_id,
            "ownership": dict(self.ownership),
            "placement": dict(self.placement),
            "event_protocol": dict(self.event_protocol),
            "resource_estimates": dict(self.resource_estimates),
            "progress_obligations": list(self.progress_obligations),
            "selection_score": list(self.selection_score),
            "latency_us": None,
            "claim": "not_measured",
        }


@dataclass(frozen=True)
class PhysicalSearchReport:
    indexed_program_hash: str
    logical_plan_hash: str
    registry_key: str
    target_profile_key: str
    control_candidate_id: str
    selected_candidate_id: str
    selection_policy: str
    candidates: tuple[PhysicalCandidate, ...]
    rejections: tuple[Mapping[str, Any], ...]
    search_budget: int = 24

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "indexed_program_hash": self.indexed_program_hash,
            "logical_plan_hash": self.logical_plan_hash,
            "registry_key": self.registry_key,
            "target_profile_key": self.target_profile_key,
            "control_candidate_id": self.control_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_policy": self.selection_policy,
            "search_budget": self.search_budget,
            "candidates": [item.to_dict() for item in self.candidates],
            "rejections": [dict(item) for item in self.rejections],
            "claim": "not_measured",
        }


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _candidate(plan: LogicalExecutionPlan, tactic: BodyTacticSpec,
               profile: CudaTargetProfile, *, block_threads: int, outputs_per_cta: int,
               output_elements: int, reduction_participants: int,
               control_id: str = "") -> PhysicalCandidate:
    candidate_id = f"{tactic.tactic_id}.b{block_threads}.t{outputs_per_cta}"
    visible_sms = profile.device_facts["visible_sms"].value
    shared = tactic.scratch_contract.get("shared_bytes")
    return PhysicalCandidate(
        candidate_id=candidate_id,
        indexed_program_hash=plan.indexed_program_hash,
        logical_plan_hash=plan.structural_hash,
        target_profile_key=profile.profile_key,
        operation_id=tactic.operation_id,
        tactic_id=tactic.tactic_id,
        algorithm_id=tactic.algorithm_id,
        provider=tactic.provider,
        provider_version=tactic.provider_version,
        body_source_hash=tactic.source_hash or "",
        target=tactic.target or "",
        block_threads=block_threads,
        outputs_per_cta=outputs_per_cta,
        grid_ctas=math.ceil(output_elements / outputs_per_cta),
        output_elements=output_elements,
        reduction_participants=reduction_participants,
        control_candidate_id=control_id,
        ownership={
            "kind": "disjoint_flat_output_intervals",
            "begin": "blockIdx.x * outputs_per_cta",
            "end": "min(begin + outputs_per_cta, output_elements)",
            "exactly_one_writer_per_output": True,
        },
        placement={"memory": "global", "schedule": "barrier_control", "fusion": "none"},
        event_protocol={
            "kind": "none",
            "reason": "each output tile is independent; no inter-CTA producer or consumer exists",
            "async_operations": [],
        },
        resource_estimates={
            "registers_per_thread": "UNKNOWN until whole-entry compile",
            "static_shared_bytes": shared if isinstance(shared, int) else "UNKNOWN",
            "local_bytes": "UNKNOWN until whole-entry compile",
            "spill_bytes": "UNKNOWN until whole-entry compile",
            "code_bytes": "UNKNOWN until whole-entry compile",
            "visible_sms_bound": visible_sms,
        },
        progress_obligations=(
            "finite grid with one output interval per CTA",
            "all required CTA instructions are unconditional for this atomic body",
            "no CTA waits on another CTA or consumes another CTA's storage",
        ),
        selection_score=(
            min(math.ceil(output_elements / outputs_per_cta),
                visible_sms if isinstance(visible_sms, int) else 0),
            reduction_participants,
            -block_threads,
        ),
    )


def search_physical_candidates(plan: LogicalExecutionPlan,
                               registry: BodyTacticRegistry,
                               profile: CudaTargetProfile) -> PhysicalSearchReport:
    """Enumerate a small complete-entry menu for one stateless dense contraction."""
    if plan.indexed_program_hash != registry.indexed_program_hash:
        raise ValueError("logical plan and body registry use different indexed semantics")
    if registry.profile_key != profile.profile_key:
        raise ValueError("body registry and CUDA profile keys differ")
    cooperative = profile.feature_attributes.get("cooperative_launch")
    if cooperative is None or cooperative.value is not True:
        raise ValueError("cooperative launch support must be known true for this entry")
    max_threads = profile.resource_limits.get("max_threads_per_block")
    max_threads = max_threads.value if max_threads else None
    if not isinstance(max_threads, int) or max_threads <= 0:
        raise ValueError("maximum threads per block must be known for candidate search")

    operation_ids = {family.operation_id for family in plan.families}
    by_operation: dict[str, list[BodyTacticSpec]] = {}
    for tactic in registry.compatible_tactics:
        if tactic.operation_id in operation_ids:
            by_operation.setdefault(tactic.operation_id, []).append(tactic)
    if len(by_operation) != 1:
        raise ValueError("the initial physical search requires exactly one covered operation")
    operation_id, tactics = next(iter(by_operation.items()))
    output_counts = {
        item.output_tile_map.get("output_elements") for item in tactics
    }
    if len(output_counts) != 1:
        raise ValueError("body tactics disagree on output cardinality")
    output_elements = next(iter(output_counts))
    if not isinstance(output_elements, int) or output_elements <= 0:
        raise ValueError("the first physical search requires a positive static output extent")

    target = profile.target_sets["baseline"].value
    target = target[0] if isinstance(target, (list, tuple)) and target else None
    if target is None or profile.supports_target(target) is not True:
        raise ValueError("the exact baseline target is not known legal and compiler-supported")

    rejections: list[Mapping[str, Any]] = []
    rejections.extend({
        "tactic_id": tactic.tactic_id,
        "provider": tactic.provider,
        "reasons": list(tactic.rejection_reasons),
    } for tactic in registry.rejected_tactics)
    rejections.extend({
        "provider": item.get("provider", "unknown"),
        "reasons": [item.get("reason", "provider did not produce a tactic")],
    } for item in registry.provider_rejections)
    candidates: list[PhysicalCandidate] = []
    control_id = ""
    for tactic in tactics:
        if (not tactic.compatible or not tactic.device_callable or not tactic.source_hash
                or tactic.source_text is None or tactic.target != target):
            rejections.append({"tactic_id": tactic.tactic_id,
                               "reason": "candidate lacks a compatible exact-target callable source"})
            continue
        out_count = _positive_int(tactic.output_tile_map.get("output_elements"))
        if out_count != output_elements:
            rejections.append({"tactic_id": tactic.tactic_id,
                               "reason": "candidate output extent differs from indexed semantics"})
            continue
        if tactic.provider == "simt":
            dims = tactic.block_contract.get("dimensions")
            roles = tactic.block_contract.get("roles", ())
            if (not isinstance(dims, list) or len(dims) != 3 or dims[1:] != [1, 1]
                    or roles != ["one_logical_output_per_warp", "all_32_lanes_reduce_k"]):
                rejections.append({"tactic_id": tactic.tactic_id,
                                   "reason": "SIMT block/warp ownership contract is unsupported"})
                continue
            block = dims[0]
            if block > max_threads or block % 32:
                rejections.append({"tactic_id": tactic.tactic_id,
                                   "reason": f"SIMT block {block} is incompatible with target limit/warp"})
                continue
            warps = block // 32
            candidates.append(_candidate(
                plan, tactic, profile, block_threads=block, outputs_per_cta=warps,
                output_elements=output_elements, reduction_participants=32,
            ))
        elif tactic.provider == "generic":
            dims = tactic.block_contract.get("dimensions")
            if dims != ["x"]:
                rejections.append({"tactic_id": tactic.tactic_id,
                                   "reason": "generic body requires a one-dimensional CTA"})
                continue
            block = min(32, max_threads)
            for tile in dict.fromkeys((1, min(4, output_elements), output_elements)):
                candidates.append(_candidate(
                    plan, tactic, profile, block_threads=block, outputs_per_cta=tile,
                    output_elements=output_elements, reduction_participants=1,
                ))
                if tile == output_elements:
                    control_id = f"{tactic.tactic_id}.b{block}.t{tile}"
        else:
            rejections.append({"tactic_id": tactic.tactic_id,
                               "reason": f"no whole-entry adapter for provider {tactic.provider!r}"})

    if not candidates or not control_id:
        raise ValueError("candidate frontier lacks a compatible tactic or barrier control")
    candidates = [
        PhysicalCandidate(**{**item.__dict__, "control_candidate_id": control_id})
        for item in candidates
    ]
    if len(candidates) > 24:
        raise ValueError("physical candidate menu exceeded its fixed search budget of 24")
    # This ranks only utilization/participant proxies. Unknown latency and actual
    # resources never become zero-cost estimates or a measured performance win.
    def specialization(item: PhysicalCandidate) -> int:
        if item.provider != "simt":
            return 1
        return int(item.tactic_id.rsplit(".v", 1)[1])

    selected = max(candidates, key=lambda item: (
        item.selection_score, -specialization(item), item.candidate_id
    ))
    return PhysicalSearchReport(
        indexed_program_hash=plan.indexed_program_hash,
        logical_plan_hash=plan.structural_hash,
        registry_key=registry.registry_key,
        target_profile_key=profile.profile_key,
        control_candidate_id=control_id,
        selected_candidate_id=selected.candidate_id,
        selection_policy=("maximize independent CTA count up to visible SMs, then reduction "
                          "participants, then minimize block size and body specialization; "
                          "estimates only, no latency ranking"),
        candidates=tuple(sorted(candidates, key=lambda item: item.candidate_id)),
        rejections=tuple(rejections),
    )


__all__ = ["PhysicalCandidate", "PhysicalSearchReport", "search_physical_candidates"]
