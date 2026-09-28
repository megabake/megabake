from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from megabake.v3.algorithms import choice_guard_failures, enumerate_algorithm_choices, recover_repeat_region
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from megabake.v3.semantics.verify import verify_algorithm_choice_cover
from tests.test_v3.cpu.test_repeat import _two_layer_step


def _policy(allow_reassociation):
    return SimpleNamespace(
        contract_hash=f"reassociation-{allow_reassociation}",
        reassociation_allowed=lambda _operation: allow_reassociation,
    )


def _indexed(module, args):
    exported = torch.export.export(module, args)
    program = normalize_fx(exported, input_spec={})
    return program, index_program(program)


def test_projection_choices_keep_indexed_reference_and_require_numerical_permission():
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(33, 17)

        def forward(self, x):
            return self.projection(x)

    module = Projection()
    x = torch.randn(1, 33)
    program, indexed = _indexed(module, (x,))
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]

    choices = enumerate_algorithm_choices(indexed, numerical_policy=_policy(True))
    by_algorithm = {choice.algorithm: choice for choice in choices}
    assert {"indexed", "projection_k_parallel", "projection_output_major",
            "projection_transposed_tensorcore", "stable_weight_pack"} <= set(by_algorithm)
    assert by_algorithm["projection_k_parallel"].reference_expansion == by_algorithm["indexed"].reference_expansion
    assert by_algorithm["projection_output_major"].origin_ids == by_algorithm["indexed"].origin_ids
    assert all(not choice_guard_failures(indexed, choice, numerical_policy=_policy(True))
               for choice in choices)

    bound_inputs = {name: program.binding_values[binding.target]
                    for name, binding in program.lifted_bindings.items()}
    actual = indexed.evaluate({"x": x, **bound_inputs})
    torch.testing.assert_close(actual, program.run_reference(x), rtol=0, atol=0)

    denied = enumerate_algorithm_choices(indexed, numerical_policy=_policy(False))
    k_parallel = next(choice for choice in denied if choice.algorithm == "projection_k_parallel")
    assert any("does not permit reassociation" in item
               for item in choice_guard_failures(indexed, k_parallel, numerical_policy=_policy(False)))
    transposed = next(choice for choice in choices if choice.algorithm == "projection_transposed_tensorcore")
    assert any("does not permit reassociation" in item
               for item in choice_guard_failures(indexed, transposed, numerical_policy=_policy(False)))
    operation_key = str(k_parallel.numerical_requirements["operation"])
    assert operation_key != "contraction"
    specific_denial = SimpleNamespace(
        contract_hash="specific-denial",
        reassociation_allowed=lambda operation: operation != operation_key,
    )
    specific_choices = enumerate_algorithm_choices(indexed, numerical_policy=specific_denial)
    specifically_denied = next(choice for choice in specific_choices
                                if choice.algorithm == "projection_k_parallel")
    assert any("does not permit reassociation" in item for item in choice_guard_failures(
        indexed, specifically_denied, numerical_policy=specific_denial))
    assert any("reference expansion differs" in item for item in choice_guard_failures(
        indexed, replace(by_algorithm["projection_output_major"], reference_expansion=()),
        numerical_policy=_policy(True)))
    tampered_requirements = replace(
        by_algorithm["projection_k_parallel"],
        numerical_requirements={"reassociation": "preserved"},
    )
    assert any("numerical requirements changed" in item
               for item in choice_guard_failures(indexed, tampered_requirements,
                                                 numerical_policy=_policy(True)))
    packed_weight = by_algorithm["stable_weight_pack"]
    tampered_preparation = replace(
        packed_weight,
        preparation_actions=({**packed_weight.preparation_actions[0], "cost": 0},),
    )
    assert any("preparation action or cost" in item
               for item in choice_guard_failures(indexed, tampered_preparation))


def test_qkv_pack_is_an_overlapping_choice_with_exact_full_cover():
    class QKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(8, 8, bias=False)
            self.k = torch.nn.Linear(8, 4, bias=False)
            self.v = torch.nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.q(x), self.k(x), self.v(x)

    program, indexed = _indexed(QKV(), (torch.randn(1, 8),))
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    choices = enumerate_algorithm_choices(indexed)
    packed = next(choice for choice in choices if choice.algorithm == "qkv_packed")
    separate = next(choice for choice in choices if choice.algorithm == "qkv_separate")
    assert len(packed.operation_ids) == 3
    assert packed.operation_ids == separate.operation_ids
    assert packed.preparation_actions[0]["cost"] == "UNKNOWN"
    reordered = replace(packed, operation_ids=tuple(reversed(packed.operation_ids)),
                        reference_expansion=tuple(reversed(packed.reference_expansion)))
    assert any("indexed program order" in item for item in choice_guard_failures(indexed, reordered))
    assert len({choice.operation_ids for choice in choices if choice.algorithm == "indexed"}) == len(indexed.operations)

    selected = [choice.choice_id for choice in choices if choice.algorithm == "indexed"
                and not set(choice.operation_ids).intersection(packed.operation_ids)] + [packed.choice_id]
    report, diagnostics = verify_algorithm_choice_cover(indexed, choices, tuple(selected))
    assert not diagnostics, [item.message for item in diagnostics]
    assert report["valid"]


def test_gated_mlp_choices_preserve_silu_and_gelu_is_not_swiglu():
    class GatedMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(8, 12, bias=False)
            self.up = torch.nn.Linear(8, 12, bias=False)
            self.down = torch.nn.Linear(12, 8, bias=False)

        def forward(self, x):
            return self.down(F.silu(self.gate(x)) * self.up(x))

    _, indexed = _indexed(GatedMLP(), (torch.randn(1, 8),))
    choices = enumerate_algorithm_choices(indexed, numerical_policy=_policy(True))
    algorithms = {choice.algorithm for choice in choices}
    assert {"gated_mlp_full", "gated_mlp_streamed"} <= algorithms
    streamed = next(choice for choice in choices if choice.algorithm == "gated_mlp_streamed")
    assert streamed.numerical_requirements["reassociation"] == "required"

    class GeluGate(torch.nn.Module):
        def forward(self, gate, up):
            return F.gelu(gate, approximate="tanh") * up

    _, gelu_indexed = _indexed(GeluGate(), (torch.randn(1, 12), torch.randn(1, 12)))
    gelu_choices = enumerate_algorithm_choices(gelu_indexed)
    assert not any(choice.algorithm.startswith("gated_mlp") or choice.algorithm == "swiglu_exact"
                   for choice in gelu_choices)


def test_repeat_unroll_choice_uses_verified_flat_expansion():
    indexed, spans, _ = _two_layer_step()
    region = recover_repeat_region(indexed, spans)
    choices = enumerate_algorithm_choices(indexed, repeat_regions=(region,))
    unroll = next(choice for choice in choices if choice.algorithm == "repeat_unroll")
    assert unroll.reference_expansion == region.expand_to_flat()
    assert unroll.operation_ids == region.expand_to_flat()


def test_exact_rmsnorm_choice_requires_the_named_matcher():
    class RMS(torch.nn.Module):
        def forward(self, x, weight):
            return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight

    _, indexed = _indexed(RMS(), (torch.randn(2, 8), torch.randn(8)))
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    choices = enumerate_algorithm_choices(indexed)
    rms = next(choice for choice in choices if choice.algorithm == "rmsnorm_exact")
    assert rms.reference_expansion
    assert not choice_guard_failures(indexed, rms)

    class WrongAxis(torch.nn.Module):
        def forward(self, x, weight):
            return x * torch.rsqrt(x.square().mean(dim=0, keepdim=True) + 1e-5) * weight

    _, wrong = _indexed(WrongAxis(), (torch.randn(2, 8), torch.randn(8)))
    assert not any(choice.algorithm == "rmsnorm_exact" for choice in enumerate_algorithm_choices(wrong))


def test_online_attention_is_an_inventory_candidate_but_not_strict_coverage():
    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    _, indexed = _indexed(Attention(), (torch.randn(1, 2, 1, 8),
                                        torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)))
    choices = enumerate_algorithm_choices(indexed, numerical_policy=_policy(True))
    online = next(choice for choice in choices if choice.algorithm == "attention_online_softmax")
    assert not choice_guard_failures(indexed, online, numerical_policy=_policy(True))
    report, diagnostics = verify_algorithm_choice_cover(indexed, choices, (online.choice_id,),
                                                        numerical_policy=_policy(True))
    assert not report["valid"]
    assert diagnostics  # The current indexed IR leaves SDPA reference-only.

    class DropoutAttention(torch.nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

    _, dropout = _indexed(DropoutAttention(), (torch.randn(1, 2, 1, 8),
                                              torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8)))
    assert not any(choice.algorithm == "attention_online_softmax"
                   for choice in enumerate_algorithm_choices(dropout, numerical_policy=_policy(True)))
