import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize


class _Gate(torch.nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up


def test_swiglu_is_recognized_only_from_silu_gate():
    program = normalize_fx(torch.export.export(_Gate(), (torch.ones(1, 65), torch.ones(1, 65))), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert any(op.name == "SwiGLU" and op.attributes["activation"] == "silu" for op in graph.operations)


def test_gelu_gate_stays_pointwise_and_preserves_approximation():
    class GeluGate(torch.nn.Module):
        def forward(self, gate, up):
            return torch.nn.functional.gelu(gate, approximate="tanh") * up
    program = normalize_fx(torch.export.export(GeluGate(), (torch.ones(1, 4), torch.ones(1, 4))), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert not any(op.name == "SwiGLU" for op in graph.operations)
    gelu = next(op for op in graph.operations if op.name == "Pointwise" and op.attributes["expression"] == "gelu")
    assert gelu.attributes["gelu_approximate"] == "tanh"
