"""Seeded CPU reference fixtures and strict comparison helpers for V3.

The fixtures intentionally use ordinary PyTorch operations.  They are executable
oracles for later normalized/FatOp/device paths, not a second V3 executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from megabake.v3.contracts import ExceptionalValuePolicy


DEFAULT_FIXTURE_SEED = 1729


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@dataclass
class ReferenceCase:
    name: str
    seed: int
    inputs: Mapping[str, Any]
    expected: Any
    metadata: Mapping[str, Any]
    state_before: Any = None
    state_after: Any = None



def make_linear_tiny(seed: int = DEFAULT_FIXTURE_SEED, *, dtype: torch.dtype = torch.float32) -> ReferenceCase:
    """LINEAR_TINY: M=1, N=17, K=33 and both weight orientations."""

    generator = _generator(seed)
    x = torch.randn((1, 33), generator=generator, dtype=dtype)
    weight_nk = torch.randn((17, 33), generator=generator, dtype=dtype).contiguous()
    weight_kn = weight_nk.t()  # explicit KxN non-contiguous view
    bias = torch.randn((17,), generator=generator, dtype=dtype)
    output_nk = x @ weight_nk.t()
    output_kn = x @ weight_kn
    output_with_bias = output_nk + bias
    return ReferenceCase(
        name="LINEAR_TINY",
        seed=seed,
        inputs={"x": x, "weight_nk": weight_nk, "weight_kn": weight_kn, "bias": bias},
        expected={
            "nk": output_nk,
            "kn": output_kn,
            "multiple_live": (output_nk, output_with_bias),
        },
        metadata={
            "M": 1,
            "N": 17,
            "K": 33,
            "weight_nk_stride": tuple(weight_nk.stride()),
            "weight_kn_stride": tuple(weight_kn.stride()),
        },
    )


def make_gate_tiny(seed: int = DEFAULT_FIXTURE_SEED, *, dtype: torch.dtype = torch.float32) -> ReferenceCase:
    """GATE_TINY: H=32, I=65, exact SiLU/GELU and tail chunks."""

    generator = _generator(seed)
    gate = torch.randn((1, 65), generator=generator, dtype=dtype)
    up = torch.randn((1, 65), generator=generator, dtype=dtype)
    chunks = ((0, 16), (16, 32), (32, 48), (48, 64), (64, 65))
    silu = F.silu(gate) * up
    gelu = F.gelu(gate, approximate="none") * up
    return ReferenceCase(
        name="GATE_TINY",
        seed=seed,
        inputs={"gate": gate, "up": up},
        expected={
            "silu": silu,
            "gelu": gelu,
            "silu_chunks": tuple(silu[:, start:end] for start, end in chunks),
            "gelu_chunks": tuple(gelu[:, start:end] for start, end in chunks),
        },
        metadata={"H": 32, "I": 65, "chunks": chunks},
    )


def make_norm_variants(seed: int = DEFAULT_FIXTURE_SEED) -> ReferenceCase:
    """NORM_VARIANTS: preserve epsilon, weight, and cast-order distinctions."""

    generator = _generator(seed)
    random_row = torch.randn((1, 32), generator=generator, dtype=torch.float32)
    cancellation = torch.tensor(
        [[1.0, -1.0] * 16], dtype=torch.float32
    )
    extreme = torch.tensor(
        [[65504.0, -65504.0] + [0.25, -0.25] * 15], dtype=torch.float32
    )
    zero = torch.zeros((1, 32), dtype=torch.float32)
    x = torch.cat((random_row, cancellation, extreme, zero), dim=0)
    weight = torch.randn((32,), generator=generator, dtype=torch.float32)
    eps = 1e-5
    mean_square = x.square().mean(dim=-1, keepdim=True)
    normalized_inside = x * torch.rsqrt(mean_square + eps)
    # This is deliberately a different reference expression, not a permitted
    # rewrite of the first one.
    normalized_outside = x * (torch.rsqrt(mean_square) + eps)
    output_cast_before_weight = normalized_inside.to(torch.float16) * weight.to(torch.float16)
    output_cast_after_weight = (normalized_inside * weight).to(torch.float16)
    one_plus_weight = normalized_inside * (1.0 + weight)
    return ReferenceCase(
        name="NORM_VARIANTS",
        seed=seed,
        inputs={"x": x, "weight": weight, "eps": eps},
        expected={
            "eps_inside": normalized_inside,
            "eps_outside": normalized_outside,
            "weight": normalized_inside * weight,
            "one_plus_weight": one_plus_weight,
            "cast_before_weight": output_cast_before_weight,
            "cast_after_weight": output_cast_after_weight,
        },
        metadata={
            "rows": 4,
            "width": 32,
            "contains_zero": True,
            "contains_cancellation": True,
            "contains_extreme_finite": True,
        },
    )


LINEAR_TINY = make_linear_tiny
GATE_TINY = make_gate_tiny
NORM_VARIANTS = make_norm_variants


def make_attention_tiny(seed: int = DEFAULT_FIXTURE_SEED, *, position: int = 0) -> ReferenceCase:
    """ATTENTION_TINY with GQA heads and an append-only capacity-17 KV cache."""

    if position not in (0, 1, 15, 16):
        raise ValueError("ATTENTION_TINY position must be one of 0, 1, 15, 16")
    generator = _generator(seed)
    q = torch.randn((1, 4, 1, 8), generator=generator)
    k = torch.randn((1, 2, 1, 8), generator=generator)
    v = torch.randn((1, 2, 1, 8), generator=generator)
    cache_k = torch.full((1, 2, 17, 8), -17.0)
    cache_v = torch.full((1, 2, 17, 8), -19.0)
    next_k, next_v = cache_k.clone(), cache_v.clone()
    next_k[:, :, position:position + 1] = k
    next_v[:, :, position:position + 1] = v
    valid = position + 1
    output = F.scaled_dot_product_attention(q, next_k[:, :, :valid], next_v[:, :, :valid], enable_gqa=True)
    return ReferenceCase(
        name="ATTENTION_TINY", seed=seed,
        inputs={"q": q, "k": k, "v": v, "cache_k": cache_k, "cache_v": cache_v, "position": position},
        expected={"output": output, "cache_k": next_k, "cache_v": next_v, "valid_length": valid},
        metadata={"B": 1, "Hq": 4, "Hkv": 2, "D": 8, "capacity": 17, "position": position},
        state_before={"cache_k": cache_k, "cache_v": cache_v}, state_after={"cache_k": next_k, "cache_v": next_v},
    )


ATTENTION_TINY = make_attention_tiny


def make_state_poison(seed: int = DEFAULT_FIXTURE_SEED, *, position: int = 0) -> ReferenceCase:
    """A cache oracle whose untouched slots are deliberately distinguishable."""
    case = make_attention_tiny(seed, position=position)
    return ReferenceCase(
        name="STATE_POISON", seed=case.seed, inputs=case.inputs, expected=case.expected,
        metadata={**case.metadata, "poison_k": -17.0, "poison_v": -19.0},
        state_before=case.state_before, state_after=case.state_after,
    )


STATE_POISON = make_state_poison


@dataclass(frozen=True)
class StateRegion:
    """A path and slice tuple used for independent state validation."""

    path: tuple[Any, ...]
    slices: tuple[slice, ...]


@dataclass
class ComparisonReport:
    structure_ok: bool
    exact_values_ok: bool
    numerical_ok: bool
    state_regions_ok: bool
    exceptional_values_ok: bool
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues


def _resolve_path(value: Any, path: Sequence[Any]) -> Any:
    for component in path:
        value = value[component]
    return value


def _exceptional_issue(
    expected: torch.Tensor,
    actual: torch.Tensor,
    policy: ExceptionalValuePolicy,
    path: str,
) -> list[str]:
    issues = []
    for label, mask, allowed in (
        ("NaN", torch.isnan(actual), policy.allow_nan),
        ("+Inf", torch.isposinf(actual), policy.allow_pos_inf),
        ("-Inf", torch.isneginf(actual), policy.allow_neg_inf),
    ):
        if bool(mask.any()) and not allowed:
            issues.append(f"{path}: unexpected {label} value")
    # Even when an exceptional value is allowed, it must occur in the same
    # region of the oracle; an allowed NaN cannot hide a finite wrong result.
    for label, expected_mask, actual_mask in (
        ("NaN", torch.isnan(expected), torch.isnan(actual)),
        ("+Inf", torch.isposinf(expected), torch.isposinf(actual)),
        ("-Inf", torch.isneginf(expected), torch.isneginf(actual)),
    ):
        if not torch.equal(expected_mask, actual_mask):
            issues.append(f"{path}: {label} locations differ")
    return issues


def _compare_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    policy: ExceptionalValuePolicy,
    issues: list[str],
    exact_box: list[bool],
    numerical_box: list[bool],
    exceptional_box: list[bool],
) -> None:
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            issues.append(f"{path}: tensor/non-tensor structure mismatch")
            return
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            issues.append(
                f"{path}: tensor structure mismatch (expected shape/dtype "
                f"{tuple(expected.shape)}/{expected.dtype}, got "
                f"{tuple(actual.shape)}/{actual.dtype})"
            )
            return
        if expected.is_floating_point() or expected.is_complex():
            exceptional = _exceptional_issue(expected, actual, policy, path)
            if exceptional:
                exceptional_box[0] = False
                issues.extend(exceptional)
            try:
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=policy.allow_nan,
                )
            except AssertionError as exc:
                numerical_box[0] = False
                issues.append(f"{path}: numerical mismatch ({exc})")
        else:
            try:
                if not torch.equal(actual, expected):
                    raise AssertionError("integer/index values differ")
            except AssertionError as exc:
                exact_box[0] = False
                issues.append(f"{path}: exact value mismatch ({exc})")
        return
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            issues.append(f"{path}: mapping/non-mapping structure mismatch")
            return
        if set(expected) != set(actual):
            issues.append(f"{path}: mapping keys differ")
            return
        for key in expected:
            _compare_value(
                expected[key], actual[key], path=f"{path}.{key}", atol=atol,
                rtol=rtol, policy=policy, issues=issues, exact_box=exact_box,
                numerical_box=numerical_box, exceptional_box=exceptional_box,
            )
        return
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if not isinstance(expected, type(actual)) or len(expected) != len(actual):
            issues.append(f"{path}: sequence structure mismatch")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_value(
                left, right, path=f"{path}[{index}]", atol=atol, rtol=rtol,
                policy=policy, issues=issues, exact_box=exact_box,
                numerical_box=numerical_box, exceptional_box=exceptional_box,
            )
        return
    if type(expected) is not type(actual) or expected != actual:
        exact_box[0] = False
        issues.append(f"{path}: scalar value/type mismatch")


def compare_outputs(
    expected: Any,
    actual: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    exceptional_value_policy: ExceptionalValuePolicy | None = None,
    state_regions: Sequence[StateRegion] = (),
) -> ComparisonReport:
    """Compare structure, exact values, numerical values, and state regions separately."""

    if atol < 0 or rtol < 0:
        raise ValueError("comparison tolerances must be non-negative")
    policy = exceptional_value_policy or ExceptionalValuePolicy(
        allow_nan=False, allow_pos_inf=False, allow_neg_inf=False
    )
    issues: list[str] = []
    exact_box = [True]
    numerical_box = [True]
    exceptional_box = [True]
    _compare_value(
        expected, actual, path="output", atol=atol, rtol=rtol, policy=policy,
        issues=issues, exact_box=exact_box, numerical_box=numerical_box,
        exceptional_box=exceptional_box,
    )
    state_start = len(issues)
    for region in state_regions:
        expected_region = _resolve_path(expected, region.path)[region.slices]
        actual_region = _resolve_path(actual, region.path)[region.slices]
        _compare_value(
            expected_region, actual_region, path=f"state{region.path}", atol=atol,
            rtol=rtol, policy=policy, issues=issues, exact_box=exact_box,
            numerical_box=numerical_box, exceptional_box=exceptional_box,
        )
    return ComparisonReport(
        structure_ok=not any("structure mismatch" in issue or "structure" in issue for issue in issues),
        exact_values_ok=exact_box[0],
        numerical_ok=numerical_box[0],
        state_regions_ok=len(issues) == state_start,
        exceptional_values_ok=exceptional_box[0],
        issues=issues,
    )


def assert_reference_match(expected: Any, actual: Any, **kwargs: Any) -> ComparisonReport:
    report = compare_outputs(expected, actual, **kwargs)
    if not report.ok:
        raise AssertionError("reference comparison failed:\n" + "\n".join(report.issues))
    return report


__all__ = [
    "ComparisonReport",
    "DEFAULT_FIXTURE_SEED",
    "GATE_TINY",
    "LINEAR_TINY",
    "NORM_VARIANTS",
    "ReferenceCase",
    "StateRegion",
    "assert_reference_match",
    "compare_outputs",
    "make_gate_tiny",
    "make_linear_tiny",
    "make_norm_variants",
]
