from megabake.v3.frontend.composites import CompositeCandidate, enumerate_composites, validate_cover
from megabake.v3.frontend.semantic import SemanticNode


def _op(op_id, name, inputs=(), outputs=(), effects=()):
    return SemanticNode(op_id, name, "v1", inputs, outputs, (op_id,), effects=effects)


def test_composites_preserve_overlaps_and_reject_duplicate_effect_cover():
    graph = type("Graph", (), {"operations": (
        _op("a", "RMSNorm", outputs=("norm",)),
        _op("b", "Linear", inputs=("norm",), outputs=("linear",)),
        _op("c", "Pointwise", inputs=("linear",), outputs=("out",)),
    )})()
    assert {candidate.kind for candidate in enumerate_composites(graph)} == {"norm_linear", "linear_epilogue"}
    effect = CompositeCandidate("cache", ("state",), (), ("state",))
    with __import__("pytest").raises(ValueError):
        validate_cover((effect, effect))
