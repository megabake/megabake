import pytest
import torch

from megabake.schedule_compiler.graph_walker import compile_from_ep, compile_model


def _guarded_identity():
    class GuardedIdentity(torch.nn.Module):
        def forward(self, value):
            torch.ops.aten._assert_tensor_metadata.default(
                value,
                size=[2, 3],
                stride=[3, 1],
                dtype=torch.float32,
                device=torch.device("cpu"),
                layout=torch.strided,
            )
            return value

    return GuardedIdentity()


def test_matching_static_tensor_metadata_assert_does_not_force_eager_fallback():
    model = _guarded_identity()

    compiled = compile_model(
        model, torch.empty(2, 3), sm_version=90, dtype=torch.float32
    )

    assert compiled.unsupported_ops == []


def test_unproven_tensor_metadata_assert_still_forces_eager_fallback():
    exported = torch.export.export(
        _guarded_identity(), (torch.empty(2, 3),), strict=False
    )
    guard = next(
        node for node in exported.graph_module.graph.nodes
        if node.target == torch.ops.aten._assert_tensor_metadata.default
    )
    guard.args = (*guard.args[:3], torch.float64)

    with pytest.warns(UserWarning, match="Unsupported ATen ops"):
        compiled = compile_from_ep(exported, sm_version=90, dtype=torch.float32)

    assert compiled.unsupported_ops == ["_assert_tensor_metadata.default"]
