import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize


def test_layer_norm_is_not_misidentified_as_rmsnorm():
    program = normalize_fx(torch.export.export(torch.nn.LayerNorm(4), (torch.ones(2, 4),)), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert not any(op.name == "RMSNorm" for op in graph.operations)


def test_explicit_rmsnorm_keeps_axes_epsilon_and_weight_transform():
    class RMS(torch.nn.Module):
        def forward(self, x, weight):
            value = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
            return value * (1.0 + weight)
    program = normalize_fx(torch.export.export(RMS(), (torch.ones(2, 4), torch.ones(4))), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    rms = next(op for op in graph.operations if op.name == "RMSNorm")
    assert rms.attributes["axes"] == (-1,) and rms.attributes["eps_placement"] == "inside_rsqrt"
    assert rms.attributes["weight_transform"] == "one_plus_weight"


def test_nontrailing_reduction_is_not_rmsnorm():
    class WrongAxis(torch.nn.Module):
        def forward(self, x):
            return x * torch.rsqrt(x.square().mean(dim=0, keepdim=True) + 1e-5)
    program = normalize_fx(torch.export.export(WrongAxis(), (torch.ones(2, 4),)), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert not any(op.name == "RMSNorm" for op in graph.operations)
