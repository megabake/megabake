from dataclasses import replace
import subprocess
import sys

import pytest
import torch

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from megabake.v3.logical import (LogicalPlanError, TileAxis, TileDomain, enumerate_region,
                                 lower_logical_plan)
from megabake.v3.logical.maps import tile_variables
from tests.test_v3.fixtures import LINEAR_TINY


class _LinearThenReduce(torch.nn.Module):
    def forward(self, x, weight):
        projected = torch.mm(x, weight.transpose(0, 1))
        return projected.sum(dim=1, keepdim=True)


def _plan(*, tile_sizes=None):
    case = LINEAR_TINY()
    args = (case.inputs["x"], case.inputs["weight_nk"])
    exported = torch.export.export(_LinearThenReduce(), args)
    captured = normalize_fx(exported, input_spec={})
    indexed = index_program(captured)
    assert indexed.strict_supported, [item.message for item in indexed.diagnostics]
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    return indexed, lower_logical_plan(indexed, choices, selected, tile_sizes=tile_sizes or {})


def test_linear_domains_cover_all_outputs_reduction_elements_and_tails():
    indexed, plan = _plan(tile_sizes={"i1": 8, "k": 16, "r0": 8})
    contraction = next(operation for operation in indexed.operations if operation.kind == "Contraction")
    contributors = next(family for family in plan.families
                        if family.operation_id == contraction.op_id and family.phase == "reduction_contributor")
    finalizer = next(family for family in plan.families
                     if family.operation_id == contraction.op_id and family.phase == "reduction_finalizer")
    assert contributors.domain.cardinality == 9  # 3 output-channel tiles x 3 K tiles
    assert finalizer.finalizes_family == contributors.family_id
    assert contributors.contributes_to_family == finalizer.family_id
    assert contributors.domain.output_axes[1].tile_count == 3
    assert contributors.domain.reduction_axes[0].tile_count == 3
    assert contributors.enumerate()[-1].axis_bounds()["i1"] == (16, 17)
    assert contributors.enumerate()[-1].axis_bounds()["k"] == (32, 33)
    assert "tile_i1*8 <= i1 < min((tile_i1+1)*8,17)" in plan.guards

    values = {value.value_id: value for value in indexed.values}
    weight_map = contributors.reads[1]
    seen_weight_addresses = []
    for tile in contributors.enumerate():
        seen_weight_addresses.extend(enumerate_region(weight_map, tile))
    expected_weight_addresses = {(k, n) for k in range(33) for n in range(17)}
    assert set(seen_weight_addresses) == expected_weight_addresses
    assert len(seen_weight_addresses) == len(expected_weight_addresses)

    output_addresses = []
    for tile in finalizer.enumerate():
        output_addresses.extend(enumerate_region(finalizer.writes[0], tile))
    assert output_addresses == [(0, n) for n in range(17)]

    reduction = next(operation for operation in indexed.operations if operation.kind == "Reduce")
    reduction_contributors = next(family for family in plan.families
                                 if family.operation_id == reduction.op_id
                                 and family.phase == "reduction_contributor")
    assert reduction_contributors.domain.cardinality == 3
    assert plan.to_dict()["logical_plan_hash"] == plan.structural_hash


def test_parametric_domains_keep_symbolic_cardinality_without_guessing():
    domain = TileDomain((TileAxis("i0", "batch", 8),))
    assert domain.cardinality == "ceil_div(batch,8)"
    with pytest.raises(ValueError, match="symbolic extents"):
        domain.enumerate("symbolic")


def test_every_fixture_access_map_matches_its_direct_fx_index_rule():
    indexed, plan = _plan(tile_sizes={"i1": 8, "k": 16, "r0": 8})
    families = {item.operation_id: item for item in plan.families
                if item.phase in {"compute", "reduction_contributor"}}
    checked = 0
    for operation in indexed.operations:
        operator = operation.attributes.get("operator_name")
        if operator == "transpose":
            oracle = [lambda indices: (indices["i1"], indices["i0"])]
        elif operation.kind == "Contraction":
            oracle = [lambda indices: (indices["i0"], indices["k"]),
                      lambda indices: (indices["k"], indices["i1"])]
        elif operation.kind == "Reduce":
            oracle = [lambda indices: (indices["i0"], indices["r0"])]
        else:
            continue
        family = families[operation.op_id]
        assert len(family.reads) == len(oracle)
        for access, direct_fx_index in zip(family.reads, oracle):
            for tile in family.enumerate():
                points = enumerate_region(access, tile)
                expected = [direct_fx_index(indices) for indices in tile_variables(tile)]
                assert list(points) == expected
            checked += 1
    assert checked == 4


def test_domains_reject_unknown_maps_and_nonpositive_tile_sizes():
    indexed, _ = _plan()
    contraction = next(operation for operation in indexed.operations if operation.kind == "Contraction")
    bad_contraction = replace(
        contraction,
        input_index_maps=(replace(contraction.input_index_maps[0], expressions=("UNKNOWN",)),
                          *contraction.input_index_maps[1:]),
    )
    mutated = replace(indexed, operations=tuple(
        bad_contraction if operation.op_id == contraction.op_id else operation
        for operation in indexed.operations
    ))
    choices = enumerate_algorithm_choices(mutated)
    selected = tuple(choice.choice_id for choice in choices if choice.algorithm == "indexed")
    with pytest.raises(LogicalPlanError, match="access map that cannot be proved"):
        lower_logical_plan(mutated, choices, selected)

    with pytest.raises(ValueError, match="positive integer"):
        TileAxis("i0", 17, 0)

    bad_output = replace(contraction, output_index_map=("UNKNOWN", "i1"))
    bad_output_program = replace(indexed, operations=tuple(
        bad_output if operation.op_id == contraction.op_id else operation
        for operation in indexed.operations
    ))
    bad_output_choices = enumerate_algorithm_choices(bad_output_program)
    bad_output_selected = tuple(choice.choice_id for choice in bad_output_choices
                                if choice.algorithm == "indexed")
    with pytest.raises(LogicalPlanError, match="access map that cannot be proved"):
        lower_logical_plan(bad_output_program, bad_output_choices, bad_output_selected)


def test_common_logical_import_does_not_load_cuda_backend():
    subprocess.run(
        [sys.executable, "-c", "import sys; import megabake.v3.logical; "
         "assert 'megabake.v3.backends.cuda' not in sys.modules"],
        check=True, capture_output=True, text=True,
    )
