from megabake.v3.frontend.layers import fingerprint_operations
from megabake.v3.frontend.semantic import SemanticNode


def test_layer_fingerprint_excludes_binding_names_but_keeps_norm_difference():
    first = SemanticNode("one", "RMSNorm", "v1", (), (), ("one",), {"weight": "layer0.weight", "eps": 1e-5})
    renamed = SemanticNode("two", "RMSNorm", "v1", (), (), ("two",), {"weight": "layer1.weight", "eps": 1e-5})
    changed = SemanticNode("three", "RMSNorm", "v1", (), (), ("three",), {"weight": "layer1.weight", "eps": 1e-6})
    assert fingerprint_operations((first,)) == fingerprint_operations((renamed,))
    assert fingerprint_operations((first,)) != fingerprint_operations((changed,))
