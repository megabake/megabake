from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from megabake.v3.contracts import StepABI
from megabake.v3.diagnostics import DiagnosticRecord
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.frontend.semantic import index_program
from megabake.v3.algorithms.repeat import RepeatRegionError, recover_repeat_region


def _two_layer_step(*, offsets=(0.125, 0.125), cast_second=False, capacities=(4, 4)):
    class Weights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight0 = torch.nn.Parameter(torch.eye(4) * 0.75)
            self.weight1 = torch.nn.Parameter(torch.eye(4) * 1.25)

    root = Weights()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    cache0 = graph.placeholder("cache0")
    cache1 = graph.placeholder("cache1")
    position = graph.placeholder("position")
    layer_nodes = []
    writes = []
    hidden = x
    for layer, (cache, offset) in enumerate(((cache0, offsets[0]), (cache1, offsets[1]))):
        start = len(list(graph.nodes))
        weight = graph.get_attr(f"weight{layer}")
        projected = graph.call_function(torch.ops.aten.mm.default, (hidden, weight))
        if cast_second and layer == 1:
            hidden = graph.call_function(torch.ops.aten.to.dtype, (projected, torch.float16))
        else:
            hidden = graph.call_function(torch.ops.aten.add.Scalar, (projected, offset))
        update = graph.call_function(torch.ops.aten.view.default, (hidden, [1, 1, 1, 4]))
        next_cache = graph.call_function(torch.ops.aten.index_copy.default,
                                         (cache, 2, position, update))
        layer_nodes.append(tuple(node.name for node in list(graph.nodes)[start:]
                                 if node.op == "call_function"))
        writes.append(next_cache)
    graph.output({"hidden": hidden, "cache0": writes[0], "cache1": writes[1]})
    module = torch.fx.GraphModule(root, graph)
    module.graph.lint()
    module.recompile()

    cache_dtypes = (torch.float32, torch.float16 if cast_second else torch.float32)
    cache_shapes = tuple((1, 1, capacity, 4) for capacity in capacities)
    example_args = (
        torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float32),
        torch.full(cache_shapes[0], -17.0, dtype=cache_dtypes[0]),
        torch.full(cache_shapes[1], -19.0, dtype=cache_dtypes[1]),
        torch.tensor([1], dtype=torch.int64),
    )
    abi = StepABI.from_dict({
        "schema_version": 1,
        "ordered_user_inputs": [
            {"placeholder": "x", "path": [0]},
            {"placeholder": "cache0", "path": [1]},
            {"placeholder": "cache1", "path": [2]},
            {"placeholder": "position", "path": [3]},
        ],
        "lifted_bindings": {
            f"weight{layer}": {"identity": f"weight{layer}", "role": "weight", "lifetime": "checkpoint"}
            for layer in range(2)
        },
        "old_state_inputs": [
            {"placeholder": f"cache{layer}", "state_id": f"kv{layer}", "path": [f"cache{layer}"],
             "layout": "b,h,capacity,d", "alias_set": f"state:kv{layer}", "capacity": capacities[layer]}
            for layer in range(2)
        ],
        "state_effects": [
            {"effect_id": f"append-kv{layer}", "state_id": f"kv{layer}", "order": layer,
             "reads": [{"state_id": f"kv{layer}", "range": "[0,L)"}],
             "writes": [{"state_id": f"kv{layer}", "range": "[L,L+1)"}]}
            for layer in range(2)
        ],
        "new_state_outputs": [
            {"state_id": f"kv{layer}", "path": [f"cache{layer}"], "source_id": writes[layer].name}
            for layer in range(2)
        ],
        "user_output_tree": {
            "structure": {"hidden": "tensor", "cache0": "tensor", "cache1": "tensor"},
            "leaves": [
                {"path": [name], "ownership": "owned", "lifetime": "returned_to_caller"}
                for name in ("hidden", "cache0", "cache1")
            ],
        },
        "position_and_valid_length": {
            "position_source": "position", "old_valid_length_source": "position",
            "append_position_expression": "L", "new_valid_length_expression": "L+1",
            "attend_range": "[0,L+1) after append",
        },
        "batch_rule": "uniform_valid_length",
        "invocation_preparation": {"setup_amortization": "none", "actions": []},
        "guard_set": {
            "shapes": {"x": [1, 4], "cache0": list(cache_shapes[0]), "cache1": list(cache_shapes[1]), "position": [1]},
            "strides": {"x": [4, 1], "cache0": [capacities[0] * 4, capacities[0] * 4, 4, 1],
                        "cache1": [capacities[1] * 4, capacities[1] * 4, 4, 1], "position": [1]},
            "dtypes": {"x": "float32", "cache0": str(cache_dtypes[0]).removeprefix("torch."),
                       "cache1": str(cache_dtypes[1]).removeprefix("torch."), "position": "int64"},
            "capacity": {"cache0": capacities[0], "cache1": capacities[1]},
            "features": ["aten.mm", "aten.index_copy"], "numerical_policy_hash": "repeat-region-test-v1",
        },
        "state_mode": "advancing", "cache_update_mode": "functional_append",
    })
    program = capture_graph_module(
        module, example_args,
        input_spec={"structure": "(x,cache0,cache1,position)"},
        output_spec=abi.user_output_tree["structure"],
        state_bindings={"cache0": "kv0", "cache1": "kv1"},
        step_abi=abi, reexport=False,
    )
    indexed = index_program(program)
    spans = tuple(tuple(f"iop:{name}" for name in names) for names in layer_nodes)
    return indexed, spans, example_args


def test_repeat_region_executes_two_layers_with_distinct_weights_and_state_outputs():
    indexed, spans, example_args = _two_layer_step()
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    region = recover_repeat_region(indexed, spans)

    assert region.expand_to_flat() == tuple(op.op_id for op in indexed.operations)
    assert region.iteration_count == 2
    assert json.loads(json.dumps(region.to_dict()))["iteration_count"] == 2
    assert set(region.per_iteration_parameter_bindings[0].values()) == {"weight0"}
    assert set(region.per_iteration_parameter_bindings[1].values()) == {"weight1"}
    assert set(region.per_iteration_state_bindings[0].values()) == {"kv0"}
    assert set(region.per_iteration_state_bindings[1].values()) == {"kv1"}
    assert any(item.producer_iteration == 0 and item.consumer_iteration == 1
               and item.value_id in region.entry_values[1] for item in region.carried_values)

    inputs = dict(zip(("x", "cache0", "cache1", "position"), example_args))
    original_states = (example_args[1].clone(), example_args[2].clone())
    actual = region.execute(indexed, inputs)
    expected = indexed.evaluate(inputs)
    assert set(actual) == {("hidden",), ("cache0",), ("cache1",)}
    for name in ("hidden", "cache0", "cache1"):
        torch.testing.assert_close(actual[(name,)], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(example_args[1], original_states[0], rtol=0, atol=0)
    torch.testing.assert_close(example_args[2], original_states[1], rtol=0, atol=0)
    assert len(region.exit_values[0]) and len(region.exit_values[1])


@pytest.mark.parametrize(
    ("kwargs", "mismatch_index"),
    [
        ({"offsets": (0.125, 0.25)}, 1),
        ({"cast_second": True}, 1),
        ({"capacities": (4, 5)}, 3),
    ],
)
def test_repeat_region_rejects_layer_semantic_differences_with_location(kwargs, mismatch_index):
    indexed, spans, _ = _two_layer_step(**kwargs)
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    with pytest.raises(RepeatRegionError) as caught:
        recover_repeat_region(indexed, spans)
    error = caught.value
    assert error.iteration == 1
    assert error.operation_id == spans[1][mismatch_index]
    assert error.field == "semantic_signature"


def test_repeat_region_rejects_reused_operation_ownership():
    indexed, spans, _ = _two_layer_step()
    with pytest.raises(RepeatRegionError, match="more than one repeat iteration") as caught:
        recover_repeat_region(indexed, (spans[0], spans[0]))
    assert caught.value.iteration == 1


def test_repeat_region_rejects_gaps_between_layer_spans():
    indexed, spans, _ = _two_layer_step()
    with pytest.raises(RepeatRegionError, match="gap between its layer spans") as caught:
        recover_repeat_region(indexed, (spans[0][:-1], spans[1]))
    assert caught.value.field == "flat_order"


def test_repeat_region_rejects_non_strict_indexed_programs():
    indexed, spans, _ = _two_layer_step()
    unsupported = replace(indexed, diagnostics=(
        DiagnosticRecord("unsupported_semantics", "fixture diagnostic"),
    ))
    with pytest.raises(RepeatRegionError, match="strict-supported") as caught:
        recover_repeat_region(unsupported, spans)
    assert caught.value.field == "strict_supported"
