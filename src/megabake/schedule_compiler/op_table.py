"""ATen op -> (OpType, op_code) mapping table."""

import torch
from megabake.data_types import OpType, ElemCode, ReduceCode, IndexCode

ATEN_OP_MAP: dict[object, tuple[int, int] | str] = {
    # --- OP_MATMUL ---
    torch.ops.aten.mm.default:             (OpType.MATMUL, 0),
    torch.ops.aten.addmm.default:          (OpType.MATMUL, 0),
    torch.ops.aten.bmm.default:            (OpType.MATMUL, 0),
    torch.ops.aten.linear.default:         (OpType.MATMUL, 0),

    # --- OP_ATTENTION ---
    torch.ops.aten.scaled_dot_product_attention.default: (OpType.ATTENTION, 0),

    # --- OP_ELEMENTWISE ---
    torch.ops.aten.add.Tensor:             (OpType.ELEMENTWISE, ElemCode.ADD),
    torch.ops.aten.add.Scalar:             (OpType.ELEMENTWISE, ElemCode.ADD),
    torch.ops.aten.mul.Tensor:             (OpType.ELEMENTWISE, ElemCode.MUL),
    torch.ops.aten.mul.Scalar:             (OpType.ELEMENTWISE, ElemCode.MUL),
    torch.ops.aten.sub.Tensor:             (OpType.ELEMENTWISE, ElemCode.SUB),
    torch.ops.aten.div.Tensor:             (OpType.ELEMENTWISE, ElemCode.DIV),
    torch.ops.aten.silu.default:           (OpType.ELEMENTWISE, ElemCode.SILU),
    torch.ops.aten.gelu.default:           (OpType.ELEMENTWISE, ElemCode.GELU),
    torch.ops.aten.relu.default:           (OpType.ELEMENTWISE, ElemCode.RELU),
    torch.ops.aten.sigmoid.default:        (OpType.ELEMENTWISE, ElemCode.SIGMOID),
    torch.ops.aten.tanh.default:           (OpType.ELEMENTWISE, ElemCode.TANH),
    torch.ops.aten.exp.default:            (OpType.ELEMENTWISE, ElemCode.EXP),
    torch.ops.aten.log.default:            (OpType.ELEMENTWISE, ElemCode.LOG),
    torch.ops.aten.rsqrt.default:          (OpType.ELEMENTWISE, ElemCode.RSQRT),
    torch.ops.aten.neg.default:            (OpType.ELEMENTWISE, ElemCode.NEG),
    torch.ops.aten.abs.default:            (OpType.ELEMENTWISE, ElemCode.ABS),
    torch.ops.aten.clamp.default:          (OpType.ELEMENTWISE, ElemCode.CLAMP),
    torch.ops.aten.where.self:             (OpType.ELEMENTWISE, ElemCode.WHERE),
    torch.ops.aten._to_copy.default:       (OpType.ELEMENTWISE, ElemCode.CAST),
    torch.ops.aten.masked_fill.Scalar:     (OpType.ELEMENTWISE, ElemCode.MASKED_FILL),
    torch.ops.aten.pow.Tensor_Scalar:      (OpType.ELEMENTWISE, ElemCode.POW),

    # --- OP_REDUCE ---
    torch.ops.aten.sum.dim_IntList:        (OpType.REDUCE, ReduceCode.SUM),
    torch.ops.aten.mean.dim:               (OpType.REDUCE, ReduceCode.MEAN),
    torch.ops.aten.amax.default:           (OpType.REDUCE, ReduceCode.MAX),
    torch.ops.aten._softmax.default:       (OpType.REDUCE, ReduceCode.SOFTMAX),
    torch.ops.aten.native_layer_norm.default: (OpType.REDUCE, ReduceCode.LAYERNORM),
    torch.ops.aten.argmax.default:         (OpType.REDUCE, ReduceCode.ARGMAX),

    # --- OP_EMBEDDING ---
    torch.ops.aten.embedding.default:      (OpType.EMBEDDING, 0),

    # --- OP_INDEX ---
    torch.ops.aten.gather.default:         (OpType.INDEX, IndexCode.GATHER),
    torch.ops.aten.index_select.default:   (OpType.INDEX, IndexCode.INDEX_SELECT),

    # --- OP_COPY ---
    torch.ops.aten.cat.default:            (OpType.COPY, 1),
    torch.ops.aten.clone.default:          (OpType.COPY, 0),

    # --- ZERO-COST STRIDE CHANGES ---
    torch.ops.aten.view.default:           "STRIDE_CHANGE",
    torch.ops.aten.reshape.default:        "STRIDE_CHANGE",
    torch.ops.aten.transpose.int:          "STRIDE_CHANGE",
    torch.ops.aten.permute.default:        "STRIDE_CHANGE",
    torch.ops.aten.expand.default:         "STRIDE_CHANGE",
    torch.ops.aten.t.default:              "STRIDE_CHANGE",
    torch.ops.aten.unsqueeze.default:      "STRIDE_CHANGE",
    torch.ops.aten.squeeze.default:        "STRIDE_CHANGE",
    torch.ops.aten.squeeze.dim:            "STRIDE_CHANGE",
    torch.ops.aten.slice.Tensor:           "STRIDE_CHANGE",
    torch.ops.aten.split.Tensor:           "STRIDE_CHANGE",
    torch.ops.aten.unbind.int:             "STRIDE_CHANGE",
}

SHAPE_OPS = {k for k, v in ATEN_OP_MAP.items() if v == "STRIDE_CHANGE"}
