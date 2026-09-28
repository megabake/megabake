from dataclasses import replace

import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from megabake.v3.semantics.verify import verify_algorithm_choice_cover, verify_indexed_program
from tests.test_v3.cpu.test_indexed import _cache_append_program


class _ProjectionAndResidual(torch.nn.Module):
    def forward(self, x, weight, residual):
        projected = torch.mm(x, weight)
        cast = projected.to(torch.float64)
        return {"main": cast + residual, "activation": x}


def _indexed(module, args):
    exported = torch.export.export(module, args)
    program = normalize_fx(exported, input_spec={})
    return program, index_program(program)


def _base_cover(indexed):
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    return choices, selected


def test_unfused_cover_has_complete_outputs_origins_casts_and_residuals():
    args = (torch.randn(2, 3), torch.randn(3, 4), torch.randn(2, 4, dtype=torch.float64))
    program, indexed = _indexed(_ProjectionAndResidual(), args)
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    choices, selected = _base_cover(indexed)
    report, diagnostics = verify_algorithm_choice_cover(indexed, choices, selected)
    assert report["valid"], [item.message for item in diagnostics]
    assert len(report["output_paths"]) == 2

    wrong_origin = replace(indexed, outputs=(replace(indexed.outputs[0], origin_id="wrong-origin"),
                                             *indexed.outputs[1:]))
    wrong_origin_report, wrong_origin_diagnostics = verify_algorithm_choice_cover(
        wrong_origin, choices, selected
    )
    assert not wrong_origin_report["valid"]
    assert any("wrong original FX origin" in item.message for item in wrong_origin_diagnostics)

    wrong_value = replace(indexed, outputs=(replace(indexed.outputs[0],
                                                     value_id=indexed.outputs[1].value_id),
                                            *indexed.outputs[1:]))
    wrong_value_report, wrong_value_diagnostics = verify_algorithm_choice_cover(
        wrong_value, choices, selected
    )
    assert not wrong_value_report["valid"]
    assert any("wrong normalized source value" in item.message for item in wrong_value_diagnostics)

    outputless_coverage, outputless = verify_indexed_program(
        program, indexed.operations, program.value_ids, outputs=(), state_transitions=indexed.state_transitions
    )
    assert any(item.details.get("negative_case") == "omitted output leaf" for item in outputless)
    missing_output = next(item for item in outputless if item.details.get("negative_case") == "omitted output leaf")
    assert missing_output.details["origin_ids"]
    assert not any(item["status"] == "output_leaf" for item in outputless_coverage)

    cast = next(operation for operation in indexed.operations if operation.target.endswith("to.dtype"))
    without_cast = tuple(operation for operation in indexed.operations if operation is not cast)
    _, cast_diagnostics = verify_indexed_program(
        program, without_cast, program.value_ids, indexed.outputs, indexed.state_transitions
    )
    assert any(item.node_id == cast.local_reference.node_name and "no indexed operation" in item.message
               for item in cast_diagnostics)
    assert any(item.details.get("origin_ids") for item in cast_diagnostics
               if item.node_id == cast.local_reference.node_name)

    residual = next(operation for operation in indexed.operations if operation.target.endswith("add.Tensor"))
    without_residual = tuple(operation for operation in indexed.operations if operation is not residual)
    _, residual_diagnostics = verify_indexed_program(
        program, without_residual, program.value_ids, indexed.outputs, indexed.state_transitions
    )
    assert any(item.node_id == residual.local_reference.node_name for item in residual_diagnostics)
    assert any(item.details.get("origin_ids") for item in residual_diagnostics
               if item.node_id == residual.local_reference.node_name)


def test_selected_cover_rejects_omitted_or_duplicated_state_effects():
    program, indexed, _, _ = _cache_append_program()
    choices, selected = _base_cover(indexed)
    report, diagnostics = verify_algorithm_choice_cover(indexed, choices, selected)
    assert report["valid"], [item.message for item in diagnostics]

    write = next(operation for operation in indexed.operations if operation.kind == "Scatter/StateWrite")
    without_write = tuple(choice_id for choice_id in selected
                          if write.op_id not in next(choice for choice in choices
                                                     if choice.choice_id == choice_id).operation_ids)
    missing, missing_diagnostics = verify_algorithm_choice_cover(indexed, choices, without_write)
    assert not missing["valid"]
    assert any("not covered" in item.message or "invalid selected ownership" in item.message
               for item in missing_diagnostics)
    assert any(item.details.get("origin_id") or item.details.get("origin_ids")
               for item in missing_diagnostics)

    duplicated = selected + (next(choice.choice_id for choice in choices
                                   if write.op_id in choice.operation_ids),)
    duplicate_report, duplicate_diagnostics = verify_algorithm_choice_cover(indexed, choices, duplicated)
    assert not duplicate_report["valid"]
    assert any("covered 2 times" in item.message for item in duplicate_diagnostics)


def test_cover_rejects_reference_only_fx_work_and_altered_candidate_metadata():
    class Inverse(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.inv(x)

    _, unsupported = _indexed(Inverse(), (torch.eye(3),))
    choices, selected = _base_cover(unsupported)
    report, diagnostics = verify_algorithm_choice_cover(unsupported, choices, selected)
    assert not report["valid"]
    assert any(item.code == "unsupported_semantics" and item.node_id for item in diagnostics)

    program, indexed = _indexed(torch.nn.Linear(3, 2), (torch.randn(1, 3),))
    choices, selected = _base_cover(indexed)
    choice = next(item for item in choices if item.choice_id == selected[0])
    mutated = tuple(replace(item, output_values=()) if item.choice_id == choice.choice_id else item
                    for item in choices)
    bad, bad_diagnostics = verify_algorithm_choice_cover(indexed, mutated, selected)
    assert not bad["valid"]
    assert any("candidate outputs differ" in item.message for item in bad_diagnostics)
