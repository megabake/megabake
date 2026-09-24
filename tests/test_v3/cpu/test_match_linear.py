import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize


def test_linear_inventory_keeps_non_square_mnk_and_bias():
    module = torch.nn.Linear(33, 17, bias=True)
    program = normalize_fx(torch.export.export(module, (torch.ones(1, 33),)), input_spec={})
    operation = recognize(program, facts=collect_facts(program)).operations[0]
    assert operation.attributes["M"] == 1 and operation.attributes["N"] == 17 and operation.attributes["K"] == 33
    assert operation.attributes["bias"]


def test_addmm_keeps_nonunit_alpha_and_beta():
    class AddMM(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.addmm(bias, x, weight, beta=2.0, alpha=0.5)
    program = normalize_fx(torch.export.export(AddMM(), (
        torch.ones(1, 3), torch.ones(3, 2), torch.ones(2),
    )), input_spec={})
    linear = next(op for op in recognize(program, facts=collect_facts(program)).operations if op.name == "Linear")
    assert linear.attributes["alpha"] == 0.5 and linear.attributes["beta"] == 2.0


def test_mm_kn_view_is_recognized_without_claiming_batched_matmul():
    class MM(torch.nn.Module):
        def forward(self, x, weight):
            return x @ weight
    program = normalize_fx(torch.export.export(MM(), (torch.ones(1, 3), torch.ones(3, 2))), input_spec={})
    linear = next(op for op in recognize(program, facts=collect_facts(program)).operations if op.name == "Linear")
    assert linear.attributes["weight_layout"] == "KN" and (linear.attributes["M"], linear.attributes["N"], linear.attributes["K"]) == (1, 2, 3)
