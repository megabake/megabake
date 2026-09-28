import json
from dataclasses import replace

import pytest
import torch

from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.semantics.verify import verify_indexed_program
from tests.test_v3.fixtures import (
    ATTENTION_TINY,
    LINEAR_TINY,
    STATE_POISON,
    cache_append_step_abi,
    make_cache_append_graph,
)


class _MapReduceView(torch.nn.Module):
    def forward(self, x, weight):
        projected = torch.mm(x, weight.transpose(0, 1))
        shifted = projected + 0.25
        return shifted.sum(dim=1, keepdim=True)


def test_indexed_linear_map_reduce_view_has_local_references_and_full_origin_coverage():
    case = LINEAR_TINY()
    exported = torch.export.export(_MapReduceView(), (case.inputs["x"], case.inputs["weight_nk"]))
    program = normalize_fx(exported, input_spec={})
    indexed = index_program(program)

    assert indexed.strict_supported
    assert {operation.kind for operation in indexed.operations} == {
        "Broadcast/View", "Contraction", "Map", "Reduce"
    }
    assert {entry["status"] for entry in indexed.coverage} >= {
        "input_value", "indexed_op", "output_tree", "output_leaf"
    }
    actual = indexed.evaluate({"x": case.inputs["x"], "weight": case.inputs["weight_nk"]})
    expected = exported.module()(case.inputs["x"], case.inputs["weight_nk"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    traced = {}

    class Record(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            traced[node.name] = value
            return value

    Record(program.graph_module).run(case.inputs["x"], case.inputs["weight_nk"])
    for operation in indexed.operations:
        node = next(node for node in program.graph_module.graph.nodes
                    if node.name == operation.local_reference.node_name)
        env = {indexed_value: traced[input_node.name]
               for indexed_value, input_node in zip(operation.inputs, node.all_input_nodes)}
        local = indexed.evaluate_local(operation.op_id, env)
        torch.testing.assert_close(local, traced[node.name], rtol=0, atol=0)

    encoded = json.dumps(indexed.to_dict(), sort_keys=True)
    assert json.loads(encoded)["indexed_program_hash"] == indexed.structural_hash


def test_indexed_addmm_uses_matrix_k_extent_and_exact_bias_broadcast():
    class AddMM(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.addmm(bias, x, weight.transpose(0, 1), beta=0.25, alpha=0.75)

    case = LINEAR_TINY()
    exported = torch.export.export(
        AddMM(), (case.inputs["x"], case.inputs["weight_nk"], case.inputs["bias"])
    )
    indexed = index_program(normalize_fx(exported, input_spec={}))
    operation = next(operation for operation in indexed.operations if operation.kind == "Contraction")

    assert indexed.strict_supported
    assert [(axis.name, axis.extent) for axis in operation.reduction_domain] == [("k", 33)]
    assert operation.input_index_maps[0].expressions == ("0", "i1")
    torch.testing.assert_close(indexed.evaluate({
        "x": case.inputs["x"], "weight": case.inputs["weight_nk"], "bias": case.inputs["bias"]
    }), exported.module()(case.inputs["x"], case.inputs["weight_nk"], case.inputs["bias"]), rtol=0, atol=0)


def test_indexed_functional_state_write_owns_effect_and_preserves_old_state():
    graph = torch.fx.Graph()
    cache = graph.placeholder("cache")
    index = graph.placeholder("index")
    update = graph.placeholder("update")
    next_cache = graph.call_function(torch.ops.aten.index_copy.default, (cache, 2, index, update))
    graph.output({"next_cache": next_cache, "same_position": index})
    module = torch.fx.GraphModule({}, graph)
    old = torch.full((1, 1, 4, 2), -9.0)
    index_value = torch.tensor([2])
    update_value = torch.ones(1, 1, 1, 2)
    program = normalize_fx(module, (old, index_value, update_value), input_spec={},
                           state_bindings={"cache": "kv"})
    indexed = index_program(program)
    write = next(operation for operation in indexed.operations if operation.kind == "Scatter/StateWrite")
    assert write.effect_edges
    assert not indexed.strict_supported
    assert any(item.code == "missing_facts" and "bounds guard" in item.message for item in indexed.diagnostics)
    actual = indexed.evaluate({"cache": old.clone(), "index": index_value, "update": update_value})
    assert torch.equal(actual["next_cache"][:, :, 2], update_value[:, :, 0])
    assert torch.equal(old, torch.full_like(old, -9.0))


def _cache_append_program(*, with_abi=True):
    old = torch.full((1, 2, 17, 8), -17.0)
    index = torch.tensor([0], dtype=torch.int64)
    update = torch.ones((1, 2, 1, 8))
    abi = cache_append_step_abi() if with_abi else None
    program = capture_graph_module(
        make_cache_append_graph(), (old, index, update),
        input_spec={"structure": "(cache,index,update)"},
        output_spec=abi.user_output_tree["structure"] if abi else {"cache": "tensor", "current": "tensor"},
        state_bindings={"cache": "kv"}, step_abi=abi,
    )
    return program, index_program(program), old, update


def test_indexed_functional_kv_append_is_typed_bounded_and_published_before_gather():
    program, indexed, old, _ = _cache_append_program()
    assert indexed.strict_supported
    assert len(indexed.state_transitions) == 1
    transition = indexed.state_transitions[0]
    assert transition.to_dict() == {
        "effect_id": "append-kv", "state_id": "kv",
        "old_value": next(value.value_id for value in indexed.values if value.fx_node == "cache"),
        "new_value": next(value.value_id for value in indexed.values if value.fx_node == "index_copy"),
        "index_value": next(value.value_id for value in indexed.values if value.fx_node == "index"),
        "valid_length_before": next(value.value_id for value in indexed.values if value.fx_node == "index"),
        "valid_length_after": "L+1", "capacity": 17, "axis": 2, "order": 0,
        "write_footprint": {"axis": 2, "range": "[L,L+1)", "other_axes": "full"},
        "alias_rule": "functional_new_value",
    }
    read = next(operation for operation in indexed.operations if operation.kind == "Gather")
    dependency = next(edge for edge in read.effect_edges if edge.kind == "state_read_after_publish")
    assert dependency.depends_on == (transition.effect_id,)
    assert dependency.reads == (transition.new_value,)
    transition.validate_index(16)
    with pytest.raises(ValueError, match="outside"):
        transition.validate_index(17)
    with pytest.raises(ValueError, match="outside"):
        transition.validate_index(-1)

    for position in (0, 1, 15, 16):
        case = STATE_POISON(position=position)
        cache = case.state_before["cache_k"].clone()
        update = case.inputs["k"]
        index = torch.tensor([position], dtype=torch.int64)
        actual = indexed.evaluate({"cache": cache, "index": index, "update": update})
        torch.testing.assert_close(actual["cache"], case.expected["cache_k"], rtol=0, atol=0)
        torch.testing.assert_close(actual["current"], update, rtol=0, atol=0)
        torch.testing.assert_close(cache, case.state_before["cache_k"], rtol=0, atol=0)
        assert torch.equal(actual["cache"][:, :, :position], cache[:, :, :position])
        assert torch.equal(actual["cache"][:, :, position + 1:], cache[:, :, position + 1:])

    first_index = torch.tensor([0], dtype=torch.int64)
    first = indexed.evaluate({"cache": old, "index": first_index, "update": torch.ones_like(old[:, :, :1])})
    second_index = torch.tensor([1], dtype=torch.int64)
    second = indexed.evaluate({"cache": first["cache"], "index": second_index,
                               "update": torch.full_like(old[:, :, :1], 2.0)})
    assert torch.equal(second["cache"][:, :, 0], first["cache"][:, :, 0])
    assert torch.equal(second["cache"][:, :, 1], torch.full_like(second["cache"][:, :, 1], 2.0))
    assert torch.equal(old, torch.full_like(old, -17.0))

    without_read_edge = tuple(
        replace(operation, effect_edges=tuple(edge for edge in operation.effect_edges
                                               if edge.kind != "state_read_after_publish"))
        if operation is read else operation
        for operation in indexed.operations
    )
    _, verification = verify_indexed_program(
        program, without_read_edge, program.value_ids, indexed.outputs, indexed.state_transitions
    )
    assert any("missing its writer publication dependency" in item.message for item in verification)


def test_indexed_state_write_rejects_unknown_bounds_and_duplicate_effect_owner():
    program, indexed, _, _ = _cache_append_program(with_abi=False)
    assert not indexed.strict_supported
    assert any(item.code == "missing_facts" and "bounds" in item.message for item in indexed.diagnostics)

    program, indexed, _, _ = _cache_append_program()
    program.effects = program.effects + (program.effects[0],)
    duplicated = index_program(program)
    assert not duplicated.strict_supported
    assert any("duplicated" in item.message for item in duplicated.diagnostics)


def test_indexed_unknown_live_operator_is_a_precise_non_strict_diagnostic():
    class Inverse(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.inv(x)

    example = torch.eye(3)
    program = normalize_fx(torch.export.export(Inverse(), (example,)), input_spec={})
    indexed = index_program(program)
    assert not indexed.strict_supported
    diagnostic = next(item for item in indexed.diagnostics if item.node_id and item.node_id.startswith("linalg_inv"))
    assert diagnostic.code == "unsupported_semantics"
    assert "linalg_inv" in diagnostic.message
    assert any(item["status"] == "diagnostic" for item in indexed.coverage)


def test_indexed_softmax_is_not_misrepresented_as_a_plain_reduction():
    example = torch.randn(2, 5)
    exported = torch.export.export(torch.nn.Softmax(dim=-1), (example,))
    indexed = index_program(normalize_fx(exported, input_spec={}))

    assert not indexed.strict_supported
    assert any(item.code == "unsupported_semantics" and "softmax" in item.message
               for item in indexed.diagnostics)
