"""Inductor pre-grad passes integration for megabake.

Adds Inductor's pre_grad_passes (pattern matching, CSE, DCE, constant
folding) on top of core_aten decompositions. select_decomp_table() is
not used because it corrupts global Inductor state, breaking subsequent
torch.compile calls in the same process.
"""

import torch


def optimize_graph(ep):
    decomp_table = torch._decomp.core_aten_decompositions()

    for op in [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten.silu.default,
        torch.ops.aten.gelu.default,
    ]:
        decomp_table.pop(op, None)

    ep = ep.run_decompositions(decomp_table)

    gm = ep.graph_module
    from torch._inductor.fx_passes.pre_grad import pre_grad_passes
    pre_grad_passes(gm)

    return ep
