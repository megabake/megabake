import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize


class _Attention(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, enable_gqa=True)


def test_sdpa_keeps_gqa_head_counts():
    program = normalize_fx(torch.export.export(_Attention(), (torch.ones(1, 4, 1, 8), torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8))), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    attention = next(op for op in graph.operations if op.name == "SDPA")
    assert attention.attributes["gqa"] and attention.attributes["heads_q"] == 4


def test_sdpa_rejects_nonzero_dropout():
    class BadDropout(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.1, enable_gqa=True)
    program = normalize_fx(torch.export.export(BadDropout(), (
        torch.ones(1, 2, 1, 8), torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8),
    )), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert not any(op.name == "SDPA" for op in graph.operations)


def test_sdpa_keeps_nonsquare_boolean_mask_and_custom_scale():
    class Masked(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.25)
    program = normalize_fx(torch.export.export(Masked(), (
        torch.ones(1, 2, 1, 8), torch.ones(1, 2, 3, 8), torch.ones(1, 2, 3, 8),
        torch.tensor([[[[True, False, True]]]]),
    )), input_spec={})
    attention = next(op for op in recognize(program, facts=collect_facts(program)).operations if op.name == "SDPA")
    assert attention.attributes["mask"] == "boolean" and attention.attributes["scale"] == 0.25
    assert attention.attributes["query_length"] == 1 and attention.attributes["key_length"] == 3
