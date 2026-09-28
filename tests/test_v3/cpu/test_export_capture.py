from pathlib import Path

import pytest
import torch

from megabake.v3.contracts import ContractError, StepABI, StepManifest
from megabake.v3.frontend.capture import BindingError, capture_exported_program


class _Captured(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.register_buffer("bias", torch.ones(2))

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        return {"logits": y, "nested": (x, y + 1)}


def test_export_capture_preserves_lifted_bindings_and_output_tree():
    exported = torch.export.export(_Captured(), (torch.ones(1, 3),))
    program = capture_exported_program(exported)
    actual = program.run_reference(torch.ones(1, 3))
    assert set(program.lifted_bindings) == {"p_weight", "b_bias"}
    assert actual["nested"][0].shape == (1, 3)
    with __import__("pytest").raises(BindingError):
        program.binding_for("missing")


def test_export_capture_keeps_tied_parameter_identity_and_constant_buffers():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.alias = self.weight
            self.register_buffer("persistent", torch.ones(2))
            self.register_buffer("nonpersistent", torch.ones(2), persistent=False)

        def forward(self, x):
            return (x @ self.weight + x @ self.alias) + self.persistent + self.nonpersistent

    program = capture_exported_program(torch.export.export(Tied(), (torch.ones(1, 2),)))
    roles = {binding.role for binding in program.lifted_bindings.values()}
    assert {"weight", "state"}.issubset(roles)
    assert torch.equal(program.run_reference(torch.ones(1, 2)), torch.full((1, 2), 6.0))


def test_step_abi_accepts_lifted_module_buffer_role_and_rejects_unknown_role():
    manifest_path = Path(__file__).resolve().parents[3] / "benchmarks/v3/manifests/tiny_cached_step.json"
    abi = StepManifest.from_json(manifest_path.read_text()).step_abi.to_dict()
    abi["lifted_bindings"]["b_buffer"] = {
        "identity": "rotary.inv_freq", "role": "state", "lifetime": "session"
    }
    assert StepABI.from_dict(abi).lifted_bindings["b_buffer"]["role"] == "state"
    abi["lifted_bindings"]["b_buffer"]["role"] = "mystery"
    with pytest.raises(ContractError, match="role is invalid"):
        StepABI.from_dict(abi)


def test_export_capture_keeps_inplace_mutation_as_an_effect_root():
    class Mutates(torch.nn.Module):
        def forward(self, x):
            x.add_(1)
            return x * 2
    program = capture_exported_program(torch.export.export(Mutates(), (torch.ones(2),)))
    assert any(effect.kind == "fx_mutation" and effect.node_id.startswith("add_") for effect in program.effects)
