import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize
from megabake.v3.frontend.match_rope import match_rope


def test_rope_matcher_never_guesses_from_plain_fx():
    class _Program: graph_module = type("G", (), {"graph": type("Graph", (), {"nodes": ()})()})()
    assert match_rope(_Program(), type("Facts", (), {})()) == ()


def test_half_split_rope_requires_the_signed_crossed_rotation_structure():
    class HalfSplit(torch.nn.Module):
        def forward(self, x, cos, sin):
            left, right = x[:, :2], x[:, 2:]
            return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
    program = normalize_fx(torch.export.export(HalfSplit(), (
        torch.ones(1, 4), torch.ones(1, 2), torch.ones(1, 2),
    )), input_spec={})
    rope = next(op for op in recognize(program, facts=collect_facts(program)).operations if op.name == "RoPE")
    assert rope.attributes["pairing"] == "half_split" and rope.attributes["rotary_dim"] == 4
