"""Effect-aware DCE that treats state transitions as graph roots."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from .capture import EffectFact, NormalizedProgram


_PURE_ATEN_SUFFIXES = (
    "add.Tensor", "add.Scalar", "sub.Tensor", "sub.Scalar", "mul.Tensor", "mul.Scalar",
    "div.Tensor", "div.Scalar", "pow.Tensor_Scalar", "neg.default", "silu.default",
    "gelu.default", "relu.default", "tanh.default", "sigmoid.default", "rsqrt.default",
    "mean.dim", "sum.dim_IntList", "sqrt.default", "square.default", "view.default",
    "reshape.default", "transpose.int", "permute.default", "slice.Tensor", "select.int",
    "squeeze.dim", "unsqueeze.default", "expand.default", "detach.default", "alias.default",
    "to.dtype", "mm.default", "addmm.default", "linear.default", "matmul.default",
    "scaled_dot_product_attention.default", "softmax.int", "_softmax.default", "cat.default",
    "stack.default", "flatten.using_ints",
)


def is_pure(node: Any) -> bool:
    if node.op != "call_function":
        return False
    target = str(node.target)
    # This is deliberately an allowlist, not "all aten is pure": RNG, mutation,
    # alias-changing and custom operators must remain DCE roots until modeled.
    if target.startswith("aten."):
        return target.endswith(_PURE_ATEN_SUFFIXES)
    return target.startswith(("<built-in function add", "<built-in function sub", "<built-in function mul",
                              "<built-in function truediv", "_operator.add", "_operator.sub",
                              "_operator.mul", "_operator.truediv"))


def effect_aware_dce(program: NormalizedProgram, *, session_state_nodes: Iterable[str] = ()) -> NormalizedProgram:
    graph_module = deepcopy(program.graph_module)
    graph = graph_module.graph
    names = {effect.node_id for effect in program.effects if effect.required} | set(session_state_nodes)
    live: set[Any] = set()
    stack: list[Any] = []
    for node in graph.nodes:
        if node.op == "output" or node.name in names or not is_pure(node):
            stack.append(node)
    while stack:
        node = stack.pop()
        if node in live:
            continue
        live.add(node)
        stack.extend(node.all_input_nodes)
    for node in reversed(list(graph.nodes)):
        if node not in live and is_pure(node) and not node.users:
            graph.erase_node(node)
    graph.lint()
    graph_module.recompile()
    return program.with_graph(graph_module)


def required_effects(program: NormalizedProgram) -> tuple[EffectFact, ...]:
    return tuple(effect for effect in program.effects if effect.required)
