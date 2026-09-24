import operator

import torch

from megabake.v3.frontend.capture import EffectFact, NormalizedProgram
from megabake.v3.frontend.effects import effect_aware_dce
from megabake.v3.frontend.normalize import normalize_fx


def test_effect_dce_removes_unreachable_pure_arithmetic():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(operator.mul, (x, 0))
    live = graph.call_function(operator.add, (x, 1))
    graph.output(live)
    program = NormalizedProgram(torch.fx.GraphModule({}, graph), lambda x: x + 1, None, {}, {})
    reduced = effect_aware_dce(program)
    assert len(list(reduced.graph_module.graph.nodes)) < len(list(program.graph_module.graph.nodes))


def test_effect_dce_keeps_required_functional_cache_transition():
    graph = torch.fx.Graph()
    cache, index, update = graph.placeholder("cache"), graph.placeholder("index"), graph.placeholder("update")
    state = graph.call_function(torch.ops.aten.index_copy.default, (cache, 2, index, update))
    graph.output(cache)
    program = NormalizedProgram(
        torch.fx.GraphModule({}, graph), lambda cache, index, update: cache, None, {}, {},
        effects=(EffectFact(state.name, "buffer_mutation", "cache"),),
    )
    reduced = effect_aware_dce(program)
    assert any(node.name == state.name for node in reduced.graph_module.graph.nodes)
