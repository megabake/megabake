import torch

from megabake.v3.frontend.normalize import normalize_fx


def test_normalize_uses_replacement_pass_without_mutating_capture():
    exported = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    seen = []
    def replacement(gm):
        seen.append(gm)
        return gm
    program = normalize_fx(exported, input_spec={"x": "tensor"}, passes=(replacement,))
    assert seen and program.normalization_path == "caller_allowlisted" and torch.equal(program.run_reference(torch.ones(2)), torch.ones(2))


def test_normalize_does_not_consume_a_one_shot_pass_iterable():
    exported = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    seen = []
    def replacement(gm):
        seen.append(gm)
        return gm
    program = normalize_fx(exported, input_spec={}, passes=(item for item in (replacement,)))
    assert seen and program.normalization_path == "caller_allowlisted"


def test_normalize_accepts_in_place_pass_and_keeps_two_requests_isolated():
    exported = torch.export.export(torch.nn.ReLU(), (torch.ones(2),))
    seen = []
    def in_place(gm):
        seen.append(id(gm))
        return None
    first = normalize_fx(exported, input_spec={}, passes=(in_place,))
    second = normalize_fx(exported, input_spec={})
    assert seen and first.graph_module is not second.graph_module
    assert torch.equal(second.run_reference(torch.ones(2)), torch.ones(2))
