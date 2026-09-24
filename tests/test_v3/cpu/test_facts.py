import torch

from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.facts import FactError, collect_facts, require_safe_write


def test_facts_keep_transpose_strides_and_guards():
    class Transpose(torch.nn.Module):
        def forward(self, x):
            return x.t()
    exported = torch.export.export(Transpose(), (torch.ones(3, 4),))
    facts = collect_facts(normalize_fx(exported, input_spec={}))
    transpose = next(fact for fact in facts.facts.values() if fact.shape == (4, 3))
    assert transpose.strides == (1, 4)
    assert facts.guards


def test_facts_guard_shape_stride_and_reject_expanded_writes():
    class Expand(torch.nn.Module):
        def forward(self, x):
            return x.expand(2, 3)
    program = normalize_fx(torch.export.export(Expand(), (torch.ones(1, 3),)), input_spec={})
    facts = collect_facts(program)
    guard = facts.guards[-1]
    with __import__("pytest").raises(FactError):
        facts.validate({guard.value_id: torch.ones(2, 3)})
    expanded = next(fact for fact in facts.facts.values() if fact.shape == (2, 3))
    with __import__("pytest").raises(FactError):
        require_safe_write(expanded)
