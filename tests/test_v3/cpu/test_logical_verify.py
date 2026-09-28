from dataclasses import replace

import pytest
import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.contracts import StepABI
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from megabake.v3.logical import (Readiness, analyze_logical_plan, can_overlay, execute_schedule,
                                 lower_logical_plan, topological_orders, verify_logical_plan)
from tests.test_v3.fixtures import LINEAR_TINY, cache_append_step_abi


class _LinearThenReduce(torch.nn.Module):
    def forward(self, x, weight):
        return torch.mm(x, weight.transpose(0, 1)).sum(dim=1, keepdim=True)


def _linear_plan():
    case = LINEAR_TINY()
    args = case.inputs["x"], case.inputs["weight_nk"]
    exported = torch.export.export(_LinearThenReduce(), args)
    indexed = index_program(normalize_fx(exported, input_spec={}))
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    base = lower_logical_plan(indexed, choices, selected, tile_sizes={"i1": 8, "k": 16, "r0": 8})
    return indexed, analyze_logical_plan(base, indexed)


def _stateful_independent_program():
    graph = torch.fx.Graph()
    cache, index, update, side = (graph.placeholder(name) for name in ("cache", "index", "update", "side"))
    next_cache = graph.call_function(torch.ops.aten.index_copy.default, (cache, 2, index, update),
                                     name="index_copy")
    current = graph.call_function(torch.ops.aten.index_select.default, (next_cache, 2, index),
                                  name="index_select")
    side_output = graph.call_function(torch.ops.aten.mul.Tensor, (side, side))
    graph.output({"cache": next_cache, "current": current, "side": side_output})
    module = torch.fx.GraphModule({}, graph)

    cache_value = torch.full((1, 2, 17, 8), -5.0)
    index_value = torch.tensor([4], dtype=torch.int64)
    update_value = torch.ones((1, 2, 1, 8))
    side_value = torch.tensor([2.0, 3.0])
    abi_payload = cache_append_step_abi().to_dict()
    abi_payload["ordered_user_inputs"].append({"placeholder": "side", "path": [3]})
    abi_payload["user_output_tree"]["structure"]["side"] = "tensor"
    abi_payload["user_output_tree"]["leaves"].append(
        {"path": ["side"], "ownership": "owned", "lifetime": "returned_to_caller"}
    )
    abi_payload["guard_set"]["shapes"]["side"] = [2]
    abi_payload["guard_set"]["strides"]["side"] = [1]
    abi_payload["guard_set"]["dtypes"]["side"] = "float32"
    abi_payload["guard_set"]["features"].append("aten.mul")
    abi = StepABI.from_dict(abi_payload)
    args = cache_value, index_value, update_value, side_value
    program = capture_graph_module(
        module, args, input_spec={"structure": "(cache,index,update,side)"},
        output_spec=abi.user_output_tree["structure"], state_bindings={"cache": "kv"},
        step_abi=abi, reexport=False,
    )
    return program, index_program(program), dict(zip(("cache", "index", "update", "side"), args))


def _analyze(indexed):
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    base = lower_logical_plan(indexed, choices, selected)
    return analyze_logical_plan(base, indexed)


def _diagnostic(report, case):
    return next(item for item in report.diagnostics if item.details.get("negative_case") == case)


def test_region_dependencies_reduction_order_and_safe_overlay():
    indexed, (plan, report) = _linear_plan()
    assert report.valid, [item.message for item in report.diagnostics]
    assert Readiness.OUTPUT_PUBLISHED.value in plan.readiness_states
    assert Readiness.SOURCE_RETIRED.value in plan.readiness_states
    assert Readiness.STORAGE_REUSABLE.value in plan.readiness_states
    assert any(item.kind == "region_overlap" for item in plan.dependencies)
    reduction_edge = next(item for item in plan.dependencies if item.kind == "reduction_finalize")
    assert reduction_edge.producer_readiness == Readiness.COMPUTE_COMPLETE
    assert reduction_edge.consumer_requires == Readiness.REDUCTION_FINAL
    assert any(item.scope == "all_reduction_contributors" for item in report.task_dependencies)

    contraction = next(item for item in indexed.operations if item.kind == "Contraction")
    reduction = next(item for item in indexed.operations if item.kind == "Reduce")
    old_value, new_value = contraction.outputs[0], reduction.outputs[0]
    lifetimes = {item.value_id: item for item in plan.storage_lifetimes}
    assert (old_value, new_value) in report.safe_overlay_pairs
    assert not can_overlay(lifetimes[old_value], lifetimes[f"partial:{new_value}"],
                           {(edge.producer_family_id, edge.consumer_family_id)
                            for edge in plan.dependencies})


def test_cache_read_waits_for_state_publication_and_legal_schedules_preserve_results():
    _, indexed, inputs = _stateful_independent_program()
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    plan, report = _analyze(indexed)
    assert report.valid, [item.message for item in report.diagnostics]
    state_edges = [item for item in plan.dependencies if item.kind == "state_publish"]
    assert len(state_edges) == 1
    assert state_edges[0].producer_readiness == Readiness.OUTPUT_PUBLISHED
    assert any(item.scope == "publication_barrier" for item in report.task_dependencies)

    schedules = topological_orders(plan, limit=8)
    assert len(schedules) >= 2  # independent side work can move around the state chain
    expected = indexed.evaluate(inputs)
    for schedule in schedules:
        actual = execute_schedule(indexed, plan, schedule, inputs)
        for path, value in actual.items():
            reference = expected
            for component in path:
                reference = reference[component]
            torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_verifier_rejects_missing_producer_reduction_publication_and_cycles():
    indexed, (plan, _) = _linear_plan()
    producer_relation = next(item for item in plan.dependencies if item.kind == "region_overlap")
    missing_family = replace(plan, families=tuple(
        item for item in plan.families if item.family_id != producer_relation.producer_family_id
    ))
    missing_family_report = verify_logical_plan(missing_family, indexed)
    assert _diagnostic(missing_family_report, "missing producer")

    missing_region = replace(plan, dependencies=tuple(
        item for item in plan.dependencies if item.kind != "region_overlap"
    ))
    assert not verify_logical_plan(missing_region, indexed).valid
    assert _diagnostic(verify_logical_plan(missing_region, indexed), "missing producer")

    missing_reduction = replace(plan, dependencies=tuple(
        item for item in plan.dependencies if item.kind != "reduction_finalize"
    ))
    assert _diagnostic(verify_logical_plan(missing_reduction, indexed), "premature reduction finalizer")

    _, state_indexed, _ = _stateful_independent_program()
    state_plan, _ = _analyze(state_indexed)
    missing_publish = replace(state_plan, dependencies=tuple(
        item for item in state_plan.dependencies if item.kind != "state_publish"
    ))
    assert _diagnostic(verify_logical_plan(missing_publish, state_indexed),
                       "cache read before publication")

    relation = next(item for item in plan.dependencies if item.kind == "region_overlap")
    cycle = replace(relation, relation_id="negative-cycle", producer_family_id=relation.consumer_family_id,
                    consumer_family_id=relation.producer_family_id)
    cyclic = replace(plan, dependencies=plan.dependencies + (cycle,))
    assert _diagnostic(verify_logical_plan(cyclic, indexed), "dependency cycle")


def test_recomputation_is_only_kept_when_a_pure_choice_declares_it():
    indexed, (base, _) = _linear_plan()
    choices = list(enumerate_algorithm_choices(indexed))
    pure_value = next(item.outputs[0] for item in indexed.operations if item.kind == "Contraction")
    pure_choice_index = next(index for index, item in enumerate(choices)
                             if item.algorithm == "indexed" and pure_value in item.output_values)
    pure_choice = choices[pure_choice_index]
    action = {"kind": "recompute", "value_id": pure_value}
    guards = dict(pure_choice.guards)
    guards["preparation_actions"] = [action]
    choices[pure_choice_index] = replace(pure_choice, preparation_actions=(action,), guards=guards)
    recompute_plan = lower_logical_plan(
        indexed, tuple(choices), base.selected_choice_ids,
        tile_sizes={"i1": 8, "k": 16, "r0": 8},
    )
    assert pure_value in recompute_plan.recompute_values

    _, state_indexed, _ = _stateful_independent_program()
    state_choices = list(enumerate_algorithm_choices(state_indexed))
    transition = state_indexed.state_transitions[0]
    state_choice_index = next(index for index, item in enumerate(state_choices)
                              if item.algorithm == "indexed" and transition.new_value in item.output_values)
    state_choice = state_choices[state_choice_index]
    action = {"kind": "recompute", "value_id": transition.new_value}
    guards = dict(state_choice.guards)
    guards["preparation_actions"] = [action]
    state_choices[state_choice_index] = replace(state_choice, preparation_actions=(action,), guards=guards)
    state_selected = tuple(item.choice_id for item in state_choices if item.algorithm == "indexed")
    with pytest.raises(ValueError, match="pure, effect-free"):
        lower_logical_plan(state_indexed, tuple(state_choices), state_selected)
