import torch

from megabake.v3.frontend import collect_facts, normalize_fx, recognize
from megabake.v3.frontend.semantic import SemanticDefinition, SemanticError, SemanticRegistry


def test_registry_is_versioned_and_linear_stays_fx_backed():
    registry = SemanticRegistry()
    definition = SemanticDefinition("Example", "v1", lambda attrs: attrs)
    registry.register(definition)
    with __import__("pytest").raises(SemanticError):
        registry.register(definition)
    module = torch.nn.Linear(3, 2)
    program = normalize_fx(torch.export.export(module, (torch.ones(1, 3),)), input_spec={})
    graph = recognize(program, facts=collect_facts(program))
    assert [op.name for op in graph.operations] == ["Linear"]
    assert callable(graph.operations[0].reference)
    assert graph.operations[0].reference(torch.ones(1, 3)).shape == (1, 2)
    assert all(callable(region.reference) for region in graph.reference_regions)
