import torch
import pytest
import torch.nn as nn

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


def test_detach_inplace_fact_preserves_the_source_storage_alias():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    detached = graph.call_function(torch.ops.aten.detach_.default, (x,))
    graph.output(detached)
    module = torch.fx.GraphModule({}, graph)
    facts = collect_facts(normalize_fx(module, (torch.ones(2, 3),), input_spec={}))
    fact = facts.for_node(detached)
    assert fact.alias_kind == "view"
    assert fact.alias_sources == (facts.node_to_value["x"],)


def _reshape_graph(shape):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    transposed = graph.call_function(torch.ops.aten.transpose.int, (x, 0, 1))
    reshaped = graph.call_function(torch.ops.aten.reshape.default, (transposed, list(shape)))
    graph.output(reshaped)
    result = torch.fx.GraphModule({}, graph)
    result.graph.lint()
    result.recompile()
    return result, reshaped.name


def test_reshape_transfer_distinguishes_noncontiguous_view_from_copy():
    example = torch.arange(12.0).reshape(3, 4)
    view_graph, view_name = _reshape_graph((4, 3))
    view = collect_facts(normalize_fx(view_graph, (example,), input_spec={}))
    view_fact = view.for_node(view_name)
    assert view_fact.alias_kind == "view" and view_fact.alias_sources
    assert view_fact.strides == (1, 4)

    copy_graph, copy_name = _reshape_graph((12,))
    copied = collect_facts(normalize_fx(copy_graph, (example,), input_spec={}))
    copy_fact = copied.for_node(copy_name)
    assert copy_fact.alias_kind == "copy" and copy_fact.alias_set is not None
    assert copy_fact.alias_set != copied.for_node("x").alias_set

    class ExpandedReshape(torch.nn.Module):
        def forward(self, x):
            return x.expand(2, 3).reshape(6)

    expanded = torch.export.export(ExpandedReshape(), (torch.ones(1, 3),))
    expanded_facts = collect_facts(normalize_fx(expanded, input_spec={}))
    expanded_copy = next(fact for fact in expanded_facts.facts.values() if fact.shape == (6,))
    assert expanded_copy.alias_kind == "copy"
    assert expanded_copy.alias_set != expanded_facts.for_node("x").alias_set


def test_tied_parameters_cache_effects_cast_points_and_unknown_alignment():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3, 3))
            self.alias = self.weight

        def forward(self, x):
            return x @ self.weight + x @ self.alias

    tied_graph = torch.fx.Graph()
    x = tied_graph.placeholder("x")
    weight = tied_graph.get_attr("weight")
    alias = tied_graph.get_attr("alias")
    left = tied_graph.call_function(torch.ops.aten.mm.default, (x, weight))
    right = tied_graph.call_function(torch.ops.aten.mm.default, (x, alias))
    tied_graph.output(tied_graph.call_function(torch.ops.aten.add.Tensor, (left, right)))
    tied = normalize_fx(torch.fx.GraphModule(Tied(), tied_graph), (torch.ones(1, 3),), input_spec={})
    tied_facts = collect_facts(tied)
    weights = [tied_facts.for_node(node.name) for node in tied.graph_module.graph.nodes if node.op == "get_attr"]
    assert len(weights) == 2 and weights[0].alias_set == weights[1].alias_set

    graph = torch.fx.Graph()
    cache = graph.placeholder("cache")
    position = graph.placeholder("position")
    update = graph.placeholder("update")
    next_cache = graph.call_function(torch.ops.aten.index_copy.default, (cache, 2, position, update))
    graph.output(next_cache)
    stateful_graph = torch.fx.GraphModule({}, graph)
    stateful = normalize_fx(
        stateful_graph,
        (torch.zeros(1, 2, 3, 4), torch.tensor([0]), torch.ones(1, 2, 1, 4)),
        input_spec={}, state_bindings={"cache": "kv"},
    )
    facts = collect_facts(stateful)
    update = next(node for node in stateful.graph_module.graph.nodes if "index_copy" in str(node.target))
    update_fact = facts.for_node(update.name)
    assert update_fact.write_effects and facts.effect_nodes[update.name]
    assert facts.for_node("cache").alias_set == "state:kv"
    assert update_fact.alias_kind == "fresh"
    assert all(fact.alignment_bytes is None for fact in facts.facts.values())

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    converted = graph.call_function(torch.ops.aten.to.dtype, (x, torch.float16))
    graph.output(converted)
    cast = normalize_fx(torch.fx.GraphModule({}, graph), (torch.ones(2),), input_spec={})
    cast_facts = collect_facts(cast)
    assert cast_facts.cast_points[cast_facts.node_to_value[converted.name]] == (converted.name,)


def test_mutating_write_with_unknown_input_alias_is_rejected():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    updated = graph.call_function(torch.ops.aten.add_.Tensor, (x, 1))
    graph.output(updated)
    module = torch.fx.GraphModule({}, graph)
    facts = collect_facts(normalize_fx(module, (torch.ones(2),), input_spec={}))
    fact = facts.for_node(updated.name)
    assert fact.mutability == "mutated" and fact.alias_set is None
    with pytest.raises(FactError, match="unknown alias"):
        require_safe_write(fact)


def test_as_strided_overlap_is_proved_and_in_place_write_rejected():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    overlapping = graph.call_function(torch.ops.aten.as_strided.default, (x, [2, 2], [1, 1]))
    graph.output(overlapping)
    module = torch.fx.GraphModule({}, graph)
    program = normalize_fx(module, (torch.ones(3),), input_spec={}, state_bindings={"x": "kv"})
    facts = collect_facts(program)
    fact = facts.for_node(overlapping.name)
    assert fact.alias_kind == "view" and fact.alias_set == "state:kv"
    assert fact.non_overlapping is False
    with pytest.raises(FactError, match="overlapping or unknown storage layout"):
        require_safe_write(fact)
