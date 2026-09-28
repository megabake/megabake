from dataclasses import replace

import pytest
import torch

from megabake.v3.backends.cuda.bodies import CudaBodyError, emit_cuda_body
from megabake.v3.backends.cuda.simt import ContractionShape, enumerate_simt_schedules
from megabake.v3.backends.cuda.tensor_core import enumerate_tensor_core_schedules
from megabake.v3.frontend.facts import FactError, collect_facts, require_safe_write
from megabake.v3.frontend.normalize import normalize_fx
from megabake.v3.frontend.semantic import index_program
from tests.test_v3.body_codegen import index_module
from tests.test_v3.fixtures import GATE_TINY, LINEAR_TINY, NORM_VARIANTS


class UnfamiliarMapReduce(torch.nn.Module):
    def forward(self, x, weight):
        transposed = x.transpose(0, 1)
        x = transposed.transpose(0, 1)
        score = torch.where(
            x > 0,
            torch.exp(x * 0.125) + weight,
            torch.rsqrt(x.square() + 0.25),
        )
        return score.sum(dim=1, keepdim=True)


class NormCastOrder(torch.nn.Module):
    def forward(self, x, weight):
        mean_square = x.square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + 1e-5)
        return normalized.to(torch.float16) * weight.to(torch.float16)


class AddMM(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.addmm(bias, x, weight.transpose(0, 1), beta=-0.25, alpha=1.75)


class MMKn(torch.nn.Module):
    def forward(self, x, weight_kn):
        return torch.mm(x, weight_kn)


class Gate(torch.nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up


class MultiAxisReduce(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=(0, 2))


class Expand(torch.nn.Module):
    def forward(self, x):
        return x.expand(2, 3)


class SliceReshapeExpand(torch.nn.Module):
    def forward(self, x):
        sliced = x[-3:4:2, 0:4:2]
        flattened = sliced.reshape(4)
        expanded = x[:, :1].expand(4, 3)
        return flattened, expanded


def test_v3r013_unfamiliar_map_view_reduce_emits_device_bodies_and_keeps_origins():
    x = torch.randn(2, 5)
    weight = torch.randn(5)
    program, indexed = index_module(UnfamiliarMapReduce(), x, weight)

    assert indexed.strict_supported
    assert {operation.kind for operation in indexed.operations} >= {
        "Map", "Reduce", "Broadcast/View"
    }
    assert all(operation.origin_ids for operation in indexed.operations)
    assert all(emit_cuda_body(indexed, operation).source for operation in indexed.operations)
    reduction = next(operation for operation in indexed.operations if operation.kind == "Reduce")
    assert reduction.input_index_maps[0].expressions == ("i0", "r0")
    assert "SwiGLU" not in "".join(emit_cuda_body(indexed, item).source for item in indexed.operations)
    torch.testing.assert_close(indexed.evaluate({"x": x, "weight": weight}),
                               UnfamiliarMapReduce()(x, weight), rtol=0, atol=0)


def test_v3r013_norm_fixture_keeps_epsilon_and_half_cast_boundaries():
    case = NORM_VARIANTS()
    program, indexed = index_module(NormCastOrder(), case.inputs["x"], case.inputs["weight"])
    assert indexed.strict_supported
    operations = indexed.operations
    assert any(emit_cuda_body(indexed, operation).body_kind == "reduction" for operation in operations)
    metadata_guards = [emit_cuda_body(indexed, operation) for operation in operations
                       if operation.kind == "Guard"]
    assert all(body.entry_point is None and body.host_guard["kind"] == "tensor_metadata"
               for body in metadata_guards)
    cast_ops = [operation for operation in operations
                if operation.attributes.get("operator_name") in {"to", "_to_copy"}]
    assert len(cast_ops) == 2
    assert all(emit_cuda_body(indexed, operation).output_type == "__half" for operation in cast_ops)
    reduced = next(operation for operation in operations if operation.kind == "Reduce")
    assert reduced.attributes["reduction"]["initial"] == 0.0
    result = indexed.evaluate({"x": case.inputs["x"], "weight": case.inputs["weight"]})
    torch.testing.assert_close(result, case.expected["cast_before_weight"], rtol=0, atol=0)
    # The distinct outside-epsilon variant remains a different reference expression.
    assert not torch.equal(case.expected["eps_inside"], case.expected["eps_outside"])


def test_v3r013_gate_tiny_matches_local_silu_reference():
    case = GATE_TINY()
    _, indexed = index_module(Gate(), case.inputs["gate"], case.inputs["up"])
    assert indexed.strict_supported
    actual = indexed.evaluate({"gate": case.inputs["gate"], "up": case.inputs["up"]})
    torch.testing.assert_close(actual, case.expected["silu"], rtol=0, atol=0)


def test_v3r013_reduction_axes_are_numbered_by_reduction_domain():
    x = torch.randn(2, 3, 4)
    _, indexed = index_module(MultiAxisReduce(), x)
    operation = next(operation for operation in indexed.operations if operation.kind == "Reduce")
    body = emit_cuda_body(indexed, operation)
    assert operation.attributes["reduction"]["axes"] == [0, 2]
    assert [(axis.name, axis.extent) for axis in operation.reduction_domain] == [("r0", 2), ("r1", 4)]
    assert operation.input_index_maps[0].expressions == ("r0", "i0", "r1")
    assert body.body_kind == "reduction" and body.output_elements == 3


def test_v3r013_slice_reshape_and_expand_keep_exact_view_maps():
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    _, indexed = index_module(SliceReshapeExpand(), x)
    assert indexed.strict_supported
    view_bodies = [emit_cuda_body(indexed, operation) for operation in indexed.operations
                   if operation.kind == "Broadcast/View"]
    assert any(body.body_kind == "view_map" for body in view_bodies)
    assert any(body.body_kind == "view_copy" for body in view_bodies)
    assert any(operation.input_index_maps[0].mode == "zero_stride_broadcast"
               for operation in indexed.operations if operation.kind == "Broadcast/View")
    torch.testing.assert_close(indexed.evaluate({"x": x}), SliceReshapeExpand()(x), rtol=0, atol=0)


def test_v3r013_rejects_as_strided_without_storage_bounds():
    class AsStrided(torch.nn.Module):
        def forward(self, x):
            return x.as_strided((2, 3), (3, 1), 1)

    _, indexed = index_module(AsStrided(), torch.randn(3, 3))
    operation = next(operation for operation in indexed.operations if operation.kind == "Broadcast/View")
    assert not indexed.strict_supported
    assert any(item.code == "missing_facts" and "unproved index map" in item.message
               for item in indexed.diagnostics)
    with pytest.raises(CudaBodyError, match="unproved index expression"):
        emit_cuda_body(indexed, operation)


def test_v3r013_rejects_unsupported_math_and_unproved_broadcast_map():
    class Unsupported(torch.nn.Module):
        def forward(self, x):
            return torch.special.erf(x)

    program, indexed = index_module(Unsupported(), torch.randn(4))
    operation = indexed.operations[0]
    assert not indexed.strict_supported
    with pytest.raises(CudaBodyError, match="no generic CUDA body"):
        emit_cuda_body(indexed, operation)

    case = GATE_TINY()
    _, mapped = index_module(torch.nn.SiLU(), case.inputs["gate"])
    operation = next(item for item in mapped.operations if item.kind == "Map")
    bad_map = replace(operation.input_index_maps[0], expressions=("UNKNOWN", "i1"))
    with pytest.raises(CudaBodyError, match="unproved index expression"):
        emit_cuda_body(mapped, replace(operation, input_index_maps=(bad_map,)))


def test_v3r013_rejects_writes_through_expanded_storage():
    program, _ = index_module(Expand(), torch.randn(1, 3))
    facts = collect_facts(program)
    expanded = next(fact for fact in facts.facts.values() if fact.shape == (2, 3))
    assert expanded.has_zero_stride
    with pytest.raises(FactError, match="zero-stride expanded view cannot be written"):
        require_safe_write(expanded)


def test_v3r014_addmm_and_both_weight_orientations_share_strided_core():
    case = LINEAR_TINY()
    program, addmm = index_module(
        AddMM(), case.inputs["x"], case.inputs["weight_nk"], case.inputs["bias"]
    )
    contraction = next(operation for operation in addmm.operations if operation.kind == "Contraction")
    body = emit_cuda_body(addmm, contraction)
    assert addmm.strict_supported and body.body_kind == "contraction"
    assert contraction.reduction_domain[0].extent == 33
    assert contraction.attributes["contraction"] == {"operator": "addmm", "alpha": 1.75, "beta": -0.25}
    assert contraction.input_index_maps[1].expressions == ("i0", "k")
    assert contraction.input_index_maps[2].expressions == ("k", "i1")
    assert "red_linear < 33" in body.source and body.output_elements == 17
    torch.testing.assert_close(addmm.evaluate({"x": case.inputs["x"], "weight": case.inputs["weight_nk"],
                                                "bias": case.inputs["bias"]}),
                               AddMM()(case.inputs["x"], case.inputs["weight_nk"],
                                       case.inputs["bias"]), rtol=0, atol=0)

    _, mm = index_module(MMKn(), case.inputs["x"], case.inputs["weight_kn"])
    operation = next(operation for operation in mm.operations if operation.kind == "Contraction")
    body_kn = emit_cuda_body(mm, operation)
    weight_value = next(value for value in mm.values if value.fx_node == "weight_kn")
    assert weight_value.strides == (1, 33)
    assert body_kn.body_kind == "contraction" and body_kn.input_types == ("float", "float")


def test_v3r014_rejects_wrong_matrix_map_complex_and_sparse_layout():
    case = LINEAR_TINY()
    _, indexed = index_module(MMKn(), case.inputs["x"], case.inputs["weight_kn"])
    operation = next(operation for operation in indexed.operations if operation.kind == "Contraction")
    wrong = replace(operation.input_index_maps[1], expressions=("i1", "k"))
    with pytest.raises(CudaBodyError, match="exact contraction map"):
        emit_cuda_body(indexed, replace(operation, input_index_maps=(operation.input_index_maps[0], wrong)))

    class ComplexMM(torch.nn.Module):
        def forward(self, x, weight):
            return torch.mm(x, weight)

    _, complex_ir = index_module(ComplexMM(), torch.randn(2, 3, dtype=torch.complex64),
                                 torch.randn(3, 4, dtype=torch.complex64))
    complex_op = next(operation for operation in complex_ir.operations if operation.kind == "Contraction")
    with pytest.raises(CudaBodyError, match="complex64"):
        emit_cuda_body(complex_ir, complex_op)

    values = tuple(replace(value, layout="sparse_coo", strides=()) if value.value_id == operation.inputs[1] else value
                   for value in indexed.values)
    sparse_ir = replace(indexed, values=values)
    with pytest.raises(CudaBodyError, match="sparse_coo"):
        emit_cuda_body(sparse_ir, operation)

    graph = torch.fx.Graph()
    x_node, w_node = graph.placeholder("x"), graph.placeholder("w")
    product = graph.call_function(torch.ops.aten.mm.default, (x_node, w_node))
    graph.output(product)
    sparse_value = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0]]), torch.tensor([1.0, 2.0]), (2, 2),
        is_coalesced=True, check_invariants=True,
    )
    sparse_program = normalize_fx(torch.fx.GraphModule({}, graph), (sparse_value, torch.randn(2, 2)),
                                   input_spec={})
    sparse_indexed = index_program(sparse_program)
    sparse_operation = next(operation for operation in sparse_indexed.operations
                            if operation.kind == "Contraction")
    with pytest.raises(CudaBodyError, match="exact strided layout"):
        emit_cuda_body(sparse_indexed, sparse_operation)


def test_v3r014_existing_simt_and_tensor_core_schedules_consume_indexed_contractions():
    case = LINEAR_TINY()
    x = case.inputs["x"].half()
    weight = case.inputs["weight_nk"].half()
    bias = case.inputs["bias"].half()
    _, indexed = index_module(AddMM(), x, weight, bias)
    operation = next(operation for operation in indexed.operations if operation.kind == "Contraction")
    shape = ContractionShape.from_indexed(indexed, operation)

    assert (shape.m, shape.n, shape.k) == (1, 17, 33)
    assert (shape.w_n_stride, shape.w_k_stride) == (33, 1)
    assert len(enumerate_simt_schedules(shape)) == 6
    assert len(enumerate_tensor_core_schedules(shape, target="sm_90a")) == 9
    # The descriptor feeds schedule generation only; the indexed op retains its epilogue semantics.
    assert operation.attributes["contraction"] == {"operator": "addmm", "alpha": 1.75, "beta": -0.25}
    weight_kn = case.inputs["weight_kn"].half()
    _, indexed_kn = index_module(MMKn(), x, weight_kn)
    kn = next(operation for operation in indexed_kn.operations if operation.kind == "Contraction")
    shape_kn = ContractionShape.from_indexed(indexed_kn, kn)
    assert (shape_kn.m, shape_kn.n, shape_kn.k) == (shape.m, shape.n, shape.k)
    assert (shape_kn.w_n_stride, shape_kn.w_k_stride) == (shape.w_n_stride, shape.w_k_stride)
