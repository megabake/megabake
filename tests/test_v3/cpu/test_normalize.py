import torch
import pytest

from megabake.v3.frontend.normalize import NormalizationCompatibilityError, normalize_fx


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


def test_normalize_direct_graphmodule_keeps_signature_without_reexport(monkeypatch):
    graph = torch.fx.symbolic_trace(lambda x, y: x + y)
    monkeypatch.setattr(torch.export, "export", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hidden export")))
    program = normalize_fx(graph, (torch.ones(2), torch.ones(2)), input_spec=("x", "y"))
    assert program.source_graph_module is graph
    assert program.graph_module is not graph
    assert [node.name for node in program.graph_module.graph.nodes if node.op == "placeholder"] == ["x", "y"]
    assert torch.equal(program.run_reference(torch.ones(2), torch.ones(2)), torch.full((2,), 2.0))


def test_normalize_tracks_inserted_nodes_and_rejects_removed_state_effect():
    graph = torch.fx.symbolic_trace(lambda x: x + 1)

    def insert_unused(gm):
        x = next(node for node in gm.graph.nodes if node.op == "placeholder")
        output = next(node for node in gm.graph.nodes if node.op == "output")
        with gm.graph.inserting_before(output):
            gm.graph.call_function(torch.ops.aten.neg.default, (x,))
        return None

    normalized = normalize_fx(graph, (torch.ones(2),), input_spec={}, passes=(insert_unused,))
    inserted = next(node for node in normalized.graph_module.graph.nodes if node.target is torch.ops.aten.neg.default)
    origin = normalized.origin_map[inserted.name][0]
    assert origin.startswith("fxgen:") and normalized.origin_lineage[origin]
    assert origin in normalized.origin_history and normalized.origin_history[origin] == (inserted.name,)

    fx = torch.fx.Graph()
    x = fx.placeholder("x")
    update = fx.call_function(torch.ops.aten.add_.Tensor, (x, 1))
    fx.output(update)
    stateful = torch.fx.GraphModule({}, fx)
    positive = normalize_fx(stateful, (torch.ones(2),), input_spec={})
    assert any(effect.kind == "fx_mutation" for effect in positive.effects)
    state = torch.ones(2)
    assert torch.equal(positive.run_reference(state), torch.full((2,), 2.0))
    assert torch.equal(state, torch.full((2,), 2.0))

    def erase_update(gm):
        update_node = next(node for node in gm.graph.nodes if node.op == "call_function")
        output = next(node for node in gm.graph.nodes if node.op == "output")
        output.args = (update_node.args[0],)
        gm.graph.erase_node(update_node)

    with pytest.raises(NormalizationCompatibilityError, match="removed required fx_mutation origin"):
        normalize_fx(stateful, (torch.ones(2),), input_spec={}, passes=(erase_update,))


def test_normalize_runs_only_selected_copied_default_decompositions():
    class Silu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.silu(x)

    example = torch.tensor([-2.0, 0.0, 1.0])
    exported = torch.export.export(Silu(), (example,))
    original_nodes = tuple((node.name, str(node.target)) for node in exported.graph_module.graph.nodes)
    program = normalize_fx(
        exported, input_spec={}, decomposition_ops=(torch.ops.aten.silu.default,)
    )
    assert "default-decompositions" in program.normalization_path
    assert tuple((node.name, str(node.target)) for node in exported.graph_module.graph.nodes) == original_nodes
    torch.testing.assert_close(program.run_reference(example), exported.module()(example), rtol=0, atol=0)
