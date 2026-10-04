import argparse
from functools import partial
from pathlib import Path

import torch
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_func
from torch._inductor import config as inductor_config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import (
    _recursive_post_grad_passes,
    _recursive_record_original_output_strides,
    compile_fx_forward,
    create_compiler_config_extra,
    fake_tensor_prop,
    get_cuda_device_context,
    get_input_idxs_to_check,
    get_num_model_outputs,
    graph_returns_tuple,
    make_graph_return_tuple,
    partition_fn,
    run_pre_grad_passes,
)
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.post_grad import view_to_reshape
from torch._inductor.output_code import CompiledFxGraphConstantsWithGm
from torch._inductor.virtualized import V
from torch.export.graph_signature import InputKind
from transformers import AutoModelForCausalLM


OUTPUTS = {
    "export": "phase_0_torch_export.txt",
    "normalized": "phase_1_normalized.txt",
    "pre_grad": "phase_2_pre_grad.txt",
    "aot": "phase_3_aot_inference.txt",
    "prepared": "phase_4_prepared.txt",
    "cache": "phase_5_cache_check.txt",
    "post_grad": "phase_6_post_grad.txt",
}

parser = argparse.ArgumentParser(
    description="Dump FX traces for a causal language model."
)
parser.add_argument(
    "--model-name",
    default="HuggingFaceTB/SmolLM-135M",
    help="Hugging Face model ID or local model path",
)
args = parser.parse_args()
model_name = args.model_name
model_output_dir = (
    Path(__file__).resolve().parents[2]
    / "fx_traces"
    / model_name.replace("/", "__")
)
model_output_dir.mkdir(parents=True, exist_ok=True)


def dump_graph(phase, gm, details=""):
    path = model_output_dir / OUTPUTS[phase]
    graph = gm.print_readable(
        print_output=False, include_stride=True, include_device=True
    )
    path.write_text(f"# {phase}\n{details}\n\n{graph}\n", encoding="utf-8")
    print(f"Wrote {phase} FX graph to {path}")


model = AutoModelForCausalLM.from_pretrained(
    model_name, attn_implementation="eager"
).eval()
input_ids = torch.tensor([[1, 2, 3, 4]])
attention_mask = torch.ones_like(input_ids)
exported = torch.export.export(
    model,
    (input_ids,),
    kwargs={"attention_mask": attention_mask, "use_cache": False, "return_dict": False},
)
dump_graph("export", exported.graph_module, "Raw torch.export FX graph.")
user_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "use_cache": False,
    "return_dict": False,
}
example_inputs = []
for spec in exported.graph_signature.input_specs:
    if spec.kind == InputKind.PARAMETER:
        example_inputs.append(exported.state_dict[spec.target])
    elif spec.kind == InputKind.BUFFER:
        example_inputs.append(
            exported.state_dict[spec.target]
            if spec.target in exported.state_dict
            else model.get_buffer(spec.target)
        )
    elif spec.kind == InputKind.CONSTANT_TENSOR:
        example_inputs.append(exported.constants[spec.target])
    elif spec.kind == InputKind.USER_INPUT:
        example_inputs.append(user_inputs[spec.arg.name])
    else:
        raise RuntimeError(f"Unsupported exported input kind: {spec.kind}")

# torch.export already flattens this model's input signature. Match Inductor's
# tuple-output convention if this model/exporter version returns a single value.
gm = exported.graph_module
if not graph_returns_tuple(gm):
    make_graph_return_tuple(gm, example_inputs, lambda graph, _inputs: graph)
dump_graph(
    "normalized",
    gm,
    "Normalized torch.export FX graph; inputs follow graph_signature order.",
)
compiler_config_extra = create_compiler_config_extra(gm)


def pre_grad_callback(graph, inputs):
    graph = run_pre_grad_passes(graph, inputs)
    dump_graph("pre_grad", graph)
    return graph


def cache_and_run_fx_passes(graph, inputs, **graph_kwargs):
    output_node = next(node for node in graph.graph.nodes if node.op == "output")
    metadata = {
        key: output_node.meta[key]
        for key in ("original_output_strides", "user_visible_output_idxs")
        if key in output_node.meta
    }
    dump_graph("prepared", graph, f"output_node_metadata={metadata!r}")

    graph_kwargs.setdefault("is_backward", False)
    graph_kwargs.setdefault("is_inference", True)
    graph_kwargs.setdefault("cpp_wrapper", False)
    graph_kwargs.setdefault("fx_wrapper", False)
    graph_kwargs.setdefault("layout_opt", None)
    graph_kwargs.setdefault("get_decomp_fn", select_decomp_table)

    cache_key = None
    cache_hit = False
    cache_state = "disabled"
    cache_details = ""
    if inductor_config.fx_graph_cache and not inductor_config.force_disable_caches:
        inputs_to_check = get_input_idxs_to_check(
            inputs, graph_kwargs.get("static_input_idxs", ())
        )
        key_info, cache_info = FxGraphCache.prepare_key(
            graph, inputs, graph_kwargs, inputs_to_check, remote=False
        )
        if key_info is None:
            cache_state = cache_info.get("cache_state", "bypass")
            cache_details = cache_info.get("cache_bypass_reason", "")
        else:
            cache_key, debug_lines = key_info
            # ponytail: inspect the local cache only; add remote lookup if needed.
            cached_graph, cache_info = FxGraphCache.load_with_key(
                cache_key,
                debug_lines,
                inputs,
                local=True,
                remote_cache=None,
                is_backward=False,
                constants=CompiledFxGraphConstantsWithGm(graph),
            )
            cache_hit = cached_graph is not None
            cache_state = cache_info.get("cache_state", "hit" if cache_hit else "miss")
            cache_details = f"key={cache_key}"

    dump_graph(
        "cache",
        graph,
        f"cache_state={cache_state}\n{cache_details}".rstrip(),
    )
    if cache_hit:
        dump_graph(
            "post_grad",
            graph,
            "Skipped post-grad passes because the FX graph cache hit.",
        )
        return make_boxed_func(graph.forward)

    view_to_reshape(graph)
    fake_mode = fake_tensor_prop(graph, inputs)
    _recursive_record_original_output_strides(graph)
    with V.set_fake_mode(fake_mode), get_cuda_device_context(graph):
        _recursive_post_grad_passes(graph, is_inference=True)
    graph.recompile()
    dump_graph("post_grad", graph)
    return make_boxed_func(graph.forward)


def compile_inference_graph(graph, inputs):
    dump_graph(
        "aot",
        graph,
        "AOTAutograd inference FX graph after decomposition; inference skips joint backward partitioning.",
    )
    return compile_fx_forward(
        graph,
        inputs,
        num_orig_model_outputs=get_num_model_outputs(gm),
        num_example_inputs=len(example_inputs),
        compiler_config_extra=compiler_config_extra,
        inner_compile=partial(
            cache_and_run_fx_passes, get_decomp_fn=select_decomp_table
        ),
        is_inference=True,
    )


with inductor_config.patch({"fx_graph_cache": True}), torch.no_grad():
    # AOTAutograd owns the pre-grad callback and hands its decomposed inference
    # GraphModule to Inductor's forward-graph preparation callback above. The
    # callback returns the FX forward itself, so this script never lowers IR or
    # compiles kernels.
    aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=compile_inference_graph,
        bw_compiler=lambda graph, _inputs: make_boxed_func(graph.forward),
        inference_compiler=compile_inference_graph,
        partition_fn=partition_fn,
        decompositions=select_decomp_table(),
        keep_inference_input_mutations=True,
        pre_grad_passes=pre_grad_callback,
    )
