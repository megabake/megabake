"""Exact one-token functional KV append followed by masked GQA SDPA."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from megabake.v3.algorithms import enumerate_algorithm_choices
from megabake.v3.contracts import ExceptionalValuePolicy, NumericalPolicy, StepABI
from megabake.v3.frontend.capture import capture_graph_module
from megabake.v3.frontend.semantic import index_program
from megabake.v3.logical import lower_logical_plan


def attention_graph(*, dropout=0.0, causal=False, enable_gqa=True, scale=None,
                    publish_current=True, rotary=False):
    graph = torch.fx.Graph()
    q = graph.placeholder("q")
    update_k = graph.placeholder("update_k")
    update_v = graph.placeholder("update_v")
    cache_k = graph.placeholder("cache_k")
    cache_v = graph.placeholder("cache_v")
    position = graph.placeholder("position")
    mask = graph.placeholder("mask")
    if rotary:
        rope_index = graph.placeholder("rope_index")
        rope_cos = graph.placeholder("rope_cos")
        rope_signed_sin = graph.placeholder("rope_signed_sin")
        q_partner = graph.call_function(torch.ops.aten.index_select.default,
                                        (q, 3, rope_index), name="q_partner")
        k_partner = graph.call_function(torch.ops.aten.index_select.default,
                                        (update_k, 3, rope_index), name="k_partner")
        q = graph.call_function(torch.ops.aten.add.Tensor, (
            graph.call_function(torch.ops.aten.mul.Tensor, (q, rope_cos), name="q_cos"),
            graph.call_function(torch.ops.aten.mul.Tensor, (q_partner, rope_signed_sin), name="q_sin")),
            name="rotated_q")
        update_k = graph.call_function(torch.ops.aten.add.Tensor, (
            graph.call_function(torch.ops.aten.mul.Tensor, (update_k, rope_cos), name="k_cos"),
            graph.call_function(torch.ops.aten.mul.Tensor, (k_partner, rope_signed_sin), name="k_sin")),
            name="rotated_k")
    new_k = graph.call_function(torch.ops.aten.index_copy.default,
                                (cache_k, 2, position, update_k), name="write_k")
    new_v = graph.call_function(torch.ops.aten.index_copy.default,
                                (cache_v, 2, position, update_v), name="write_v")
    output = graph.call_function(torch.ops.aten.scaled_dot_product_attention.default,
                                 (q, new_k if publish_current else cache_k, new_v, mask, dropout, causal),
                                 {"enable_gqa": enable_gqa, "scale": scale}, name="attention")
    graph.output({"output": output, "cache_k": new_k, "cache_v": new_v})
    module = torch.fx.GraphModule({}, graph)
    module.graph.lint()
    module.recompile()
    return module


def attention_inputs(position=0, *, capacity=17, dtype=torch.float32, device="cpu",
                     mask_kind="causal", batch=1, heads_kv=2, mask_heads=1, depth=8):
    generator = torch.Generator(device="cpu").manual_seed(1729 + position)
    q = torch.randn(batch, 4, 1, depth, generator=generator, dtype=dtype)
    update_k = torch.randn(batch, heads_kv, 1, depth, generator=generator, dtype=dtype)
    update_v = torch.randn(batch, heads_kv, 1, depth, generator=generator, dtype=dtype)
    cache_k = torch.full((batch, heads_kv, capacity, depth), -17.0, dtype=dtype)
    cache_v = torch.full((batch, heads_kv, capacity, depth), -19.0, dtype=dtype)
    mask = torch.zeros((batch, mask_heads, 1, capacity), dtype=torch.bool)
    if mask_kind == "causal":
        mask[..., :position + 1] = True
    elif mask_kind == "window":
        mask[..., max(0, position - 3):position + 1] = True
    elif mask_kind != "empty":
        raise ValueError(mask_kind)
    return tuple(value.to(device) for value in
                 (q, update_k, update_v, cache_k, cache_v,
                  torch.tensor([position], dtype=torch.int64), mask))


def oracle_inputs(case, *, device="cpu"):
    inputs = case.inputs
    position = inputs["position"]
    mask = torch.zeros((1, 1, 1, inputs["cache_k"].shape[2]), dtype=torch.bool)
    mask[..., :position + 1] = True
    return tuple(value.to(device) if isinstance(value, torch.Tensor) else value
                 for value in (inputs["q"], inputs["k"], inputs["v"],
                               inputs["cache_k"], inputs["cache_v"],
                               torch.tensor([position], dtype=torch.int64), mask))


def attention_abi(inputs, *, declare_rotary_bounds=True):
    names = ("q", "update_k", "update_v", "cache_k", "cache_v", "position", "mask")
    if len(inputs) == 10:
        names += ("rope_index", "rope_cos", "rope_signed_sin")
    capacity = inputs[3].shape[2]
    states = (("cache_k", "k", "write_k", "append-k", 3, 0),
              ("cache_v", "v", "write_v", "append-v", 4, 1))
    return StepABI.from_dict({
        "schema_version": 1,
        "ordered_user_inputs": [{"placeholder": name, "path": [i]} for i, name in enumerate(names)],
        "lifted_bindings": {},
        "old_state_inputs": [{"placeholder": name, "state_id": state, "path": [index],
                              "layout": "b,h,capacity,d", "alias_set": f"state:{state}",
                              "capacity": capacity}
                             for name, state, _, _, index, _ in states],
        "state_effects": [{"effect_id": effect, "state_id": state, "order": order,
                           "reads": [{"state_id": state, "range": "[0,L)"}],
                           "writes": [{"state_id": state, "range": "[L,L+1)"}]}
                          for _, state, _, effect, _, order in states],
        "new_state_outputs": [{"state_id": state, "path": [name], "source_id": source}
                              for name, state, source, _, _, _ in states],
        "user_output_tree": {"structure": {"output": "tensor", "cache_k": "tensor", "cache_v": "tensor"},
                             "leaves": [{"path": [name], "ownership": "owned",
                                         "lifetime": "returned_to_caller"}
                                        for name in ("output", "cache_k", "cache_v")]},
        "position_and_valid_length": {
            "position_source": "position", "old_valid_length_source": "position",
            "append_position_expression": "L", "new_valid_length_expression": "L+1",
            "attend_range": "[0,L+1) after append",
        },
        "batch_rule": "uniform_valid_length",
        "invocation_preparation": {"setup_amortization": "none", "actions": []},
        "guard_set": {
            "shapes": {name: list(value.shape) for name, value in zip(names, inputs)},
            "strides": {name: list(value.stride()) for name, value in zip(names, inputs)},
            "dtypes": {name: str(value.dtype).removeprefix("torch.") for name, value in zip(names, inputs)},
            "capacity": {"cache_k": capacity, "cache_v": capacity},
            "position": {"source": "position", "specialized": int(inputs[5].item())},
            "features": ["aten.index_copy", "aten.scaled_dot_product_attention"],
            "numerical_policy_hash": attention_policy(inputs[0].dtype).contract_hash,
            **({"index_bounds": {"rope_index": [0, 8]}}
               if len(inputs) == 10 and declare_rotary_bounds else {}),
        },
        "state_mode": "advancing", "cache_update_mode": "functional_append",
    })


def capture_attention(inputs, **graph_options):
    abi = attention_abi(inputs, declare_rotary_bounds=graph_options.pop("declare_rotary_bounds", True))
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.MATH]):
        captured = capture_graph_module(
            attention_graph(**graph_options), inputs, input_spec={"args": len(inputs)},
            output_spec=abi.user_output_tree["structure"],
            state_bindings={"cache_k": "k", "cache_v": "v"}, step_abi=abi,
            policy=attention_policy(inputs[0].dtype), reexport=False,
        )
    indexed = index_program(captured)
    choices = enumerate_algorithm_choices(indexed)
    selected = tuple(choice.choice_id for choice in choices
                     if choice.algorithm in {"indexed", "online_softmax"})
    plan = lower_logical_plan(indexed, choices, selected) if indexed.strict_supported else None
    return captured, indexed, plan


def rotary_attention_inputs(position=1, *, dtype=torch.float32, device="cpu"):
    inputs = attention_inputs(position, dtype=dtype, device=device)
    angle = torch.arange(4, dtype=torch.float32) * 0.13 + 0.31
    cosine = torch.cat((angle.cos(), angle.cos())).to(dtype).reshape(1, 1, 1, 8)
    signed_sine = torch.cat((-angle.sin(), angle.sin())).to(dtype).reshape(1, 1, 1, 8)
    index = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], dtype=torch.int64)
    return (*inputs, index.to(device), cosine.to(device), signed_sine.to(device))


def attention_policy(dtype):
    name = str(dtype).removeprefix("torch.")
    return NumericalPolicy(
        reference_expansion="aten.scaled_dot_product_attention with published fixed KV and bool mask",
        intermediate_casts=(),
        accumulation_dtypes={"scaled_dot_product_attention": "float32"},
        output_casts={"scaled_dot_product_attention": name},
        permitted_reassociation={"scaled_dot_product_attention": True},
        tolerances={"scaled_dot_product_attention": {
            "float32": {"atol": 1e-5, "rtol": 1e-5},
            "float16": {"atol": 4e-3, "rtol": 4e-3},
            "bfloat16": {"atol": 0.07, "rtol": 0.01},
        }},
        exceptional_value_policy=ExceptionalValuePolicy(
            allow_nan=False, allow_pos_inf=False, allow_neg_inf=False),
    )
