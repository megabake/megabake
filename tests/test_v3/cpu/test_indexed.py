import json

import pytest
import torch

from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from tests.test_v3.fixtures import LINEAR_TINY


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
