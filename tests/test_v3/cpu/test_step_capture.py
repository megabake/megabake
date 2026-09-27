from __future__ import annotations

from pathlib import Path

import pytest
import torch

from megabake.v3.contracts import StepABI, StepManifest
from megabake.v3.frontend.capture import (
    BindingError,
    FrontendError,
    UnsupportedGraphError,
    capture_exported_program,
    capture_graph_module,
    graph_hash,
)
from tests.test_v3.fixtures import assert_reference_match, make_tiny_stateful_graph


def _manifest() -> StepManifest:
    path = Path(__file__).resolve().parents[3] / "benchmarks/v3/manifests/tiny_cached_step.json"
    return StepManifest.from_json(path.read_text())


def _examples(position: int = 0):
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    cache = torch.full((1, 1, 5, 4), -17.0)
    return x, cache, torch.tensor([position], dtype=torch.int64)


def test_tiny_step_capture_preserves_signature_bindings_outputs_and_advancing_state() -> None:
    manifest = _manifest()
    abi = manifest.step_abi
    graph_module = make_tiny_stateful_graph()
    program = capture_graph_module(
        graph_module,
        _examples(),
        input_spec={"structure": "(x, cache, position)"},
        output_spec=abi.user_output_tree["structure"],
        state_bindings={"cache": "kv"},
        step_abi=abi,
    )

    assert graph_hash(program) == manifest.graph_hash
    assert tuple(program.lifted_bindings) == ("p_weight",)
    assert any(effect.kind == "functional_state_update" and effect.target == "kv" for effect in program.effects)
    assert program.output_tree_spec is not None

    old_reference = _examples()[1]
    old_captured = old_reference.clone()
    for position in (0, 1):
        x = torch.tensor([[position + 1.0, -2.0, 3.0, -4.0]])
        index = torch.tensor([position], dtype=torch.int64)
        original_input = old_reference.clone()
        captured_input = old_captured.clone()
        expected = graph_module(x, original_input, index)
        actual = program.run_reference(x, captured_input, index)
        assert_reference_match(expected, actual, atol=0, rtol=0)
        assert torch.equal(original_input, old_reference)
        assert torch.equal(captured_input, old_captured)
        old_reference = expected["cache"]
        old_captured = actual["cache"]


def test_exported_program_capture_checks_step_signature_bindings_and_effects() -> None:
    manifest = _manifest()
    graph_module = make_tiny_stateful_graph()
    exported = torch.export.export(graph_module, _examples(), strict=False)
    program = capture_exported_program(
        exported,
        state_bindings={"cache": "kv"},
        step_abi=manifest.step_abi,
    )
    assert tuple(program.lifted_bindings) == ("p_weight",)
    assert any(effect.target == "kv" for effect in program.effects)


def test_graphmodule_step_capture_rejects_missing_bindings_and_fragmented_signature() -> None:
    manifest = _manifest()
    graph_module = make_tiny_stateful_graph()
    with pytest.raises(FrontendError, match="output_spec and state_bindings"):
        capture_graph_module(
            graph_module, _examples(), input_spec={}, output_spec=manifest.step_abi.user_output_tree["structure"],
            step_abi=manifest.step_abi,
        )

    abi_data = manifest.step_abi.to_dict()
    abi_data["ordered_user_inputs"] = abi_data["ordered_user_inputs"][:2]
    incomplete = StepABI.from_dict(abi_data)
    with pytest.raises(UnsupportedGraphError, match="signature is incomplete or reordered"):
        capture_graph_module(
            graph_module,
            _examples(),
            input_spec={},
            output_spec=incomplete.user_output_tree["structure"],
            state_bindings={"cache": "kv"},
            step_abi=incomplete,
        )


def test_graphmodule_step_capture_rejects_unbound_lifted_weight() -> None:
    manifest = _manifest()
    abi_data = manifest.step_abi.to_dict()
    abi_data["lifted_bindings"]["p_weight"]["identity"] = "different_weight"
    mismatched = StepABI.from_dict(abi_data)
    with pytest.raises(BindingError, match="lifted binding"):
        capture_graph_module(
            make_tiny_stateful_graph(),
            _examples(),
            input_spec={},
            output_spec=mismatched.user_output_tree["structure"],
            state_bindings={"cache": "kv"},
            step_abi=mismatched,
        )
