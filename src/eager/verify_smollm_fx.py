import argparse
import hashlib
import json
import re
import shutil
import subprocess
from functools import partial
from pathlib import Path

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
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


def _graph_body(text):
    start = text.find("\n\nclass ")
    return text[start + 2 :].rstrip() if start >= 0 else text.rstrip()


def _logits(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected logits tensor, got {type(output).__name__}")
    return output


def _model_revision(model_name, requested_revision, model):
    revision = (
        getattr(model.config, "_commit_hash", None)
        or getattr(model, "_commit_hash", None)
    )
    if revision or Path(model_name).exists():
        return revision or "local"
    config_path = try_to_load_from_cache(
        model_name, "config.json", revision=requested_revision or "main"
    )
    if isinstance(config_path, str):
        return Path(config_path).parent.name
    return requested_revision or "unknown"


def _nvcc_version():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    try:
        result = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release ([0-9.]+)", result.stdout)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Compare eager SmolLM logits with the live Inductor post-grad FX graph."
    )
    parser.add_argument(
        "--model-name",
        default="HuggingFaceTB/SmolLM-135M",
        help="Hugging Face model ID or local model path",
    )
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face revision; the resolved model commit is recorded",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this verification")

    root = Path(__file__).resolve().parents[2]
    trace_dir = root / "fx_traces" / args.model_name.replace("/", "__") / "cuda"
    output_dir = trace_dir / "eager_verify"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        attn_implementation="eager",
    ).to(device).eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    attention_mask = torch.ones_like(input_ids)
    run_metadata = {
        "model": args.model_name,
        "requested_revision": args.revision,
        "resolved_revision": _model_revision(args.model_name, args.revision, model),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_toolkit_nvcc": _nvcc_version(),
        "transformers_version": transformers.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "input_ids": input_ids.cpu().tolist(),
        "attention_mask": attention_mask.cpu().tolist(),
        "use_cache": False,
        "return_dict": False,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
    }

    def dump_graph(phase, graph_module, details=""):
        path = output_dir / OUTPUTS[phase]
        graph = graph_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        if phase == "post_grad":
            run_metadata["post_grad_graph_sha256"] = hashlib.sha256(
                graph.rstrip().encode("utf-8")
            ).hexdigest()
            previous_path = trace_dir / OUTPUTS["post_grad"]
            if previous_path.exists():
                previous = _graph_body(previous_path.read_text(encoding="utf-8"))
                previous_hash = hashlib.sha256(previous.encode("utf-8")).hexdigest()
                run_metadata["previous_post_grad_graph_sha256"] = previous_hash
                run_metadata["matches_previous_post_grad_graph"] = (
                    previous_hash == run_metadata["post_grad_graph_sha256"]
                )

        metadata = json.dumps(run_metadata, sort_keys=True)
        path.write_text(
            f"# {phase}\ndevice={device}\nmetadata={metadata}\n{details}\n\n{graph}\n",
            encoding="utf-8",
        )
        print(f"Wrote {phase} FX graph to {path}")

    exported = torch.export.export(
        model,
        (input_ids,),
        kwargs={
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": False,
        },
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

    gm = exported.graph_module
    if not graph_returns_tuple(gm):
        make_graph_return_tuple(gm, example_inputs, lambda graph, _inputs: graph)
    dump_graph(
        "normalized",
        gm,
        "Normalized torch.export FX graph; inputs follow graph_signature order.",
    )
    compiler_config_extra = create_compiler_config_extra(gm)
    post_grad_captures = []

    def pre_grad_callback(graph, inputs):
        graph = run_pre_grad_passes(graph, inputs)
        dump_graph("pre_grad", graph)
        return graph

    def cache_and_run_fx_passes(graph, inputs, **graph_kwargs):
        output_node = next(node for node in graph.graph.nodes if node.op == "output")
        output_metadata = {
            key: output_node.meta[key]
            for key in ("original_output_strides", "user_visible_output_idxs")
            if key in output_node.meta
        }
        dump_graph("prepared", graph, f"output_node_metadata={output_metadata!r}")

        graph_kwargs.setdefault("is_backward", False)
        graph_kwargs.setdefault("is_inference", True)
        graph_kwargs.setdefault("cpp_wrapper", False)
        graph_kwargs.setdefault("fx_wrapper", False)
        graph_kwargs.setdefault("layout_opt", None)
        graph_kwargs.setdefault("get_decomp_fn", select_decomp_table)

        cache_hit = False
        cache_state = "disabled"
        cache_details = ""
        if inductor_config.fx_graph_cache and not inductor_config.force_disable_caches:
            checked_inputs = get_input_idxs_to_check(
                inputs, graph_kwargs.get("static_input_idxs", ())
            )
            key_info, cache_info = FxGraphCache.prepare_key(
                graph, inputs, graph_kwargs, checked_inputs, remote=False
            )
            if key_info is None:
                cache_state = cache_info.get("cache_state", "bypass")
                cache_details = cache_info.get("cache_bypass_reason", "")
            else:
                cache_key, debug_lines = key_info
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
                cache_state = cache_info.get(
                    "cache_state", "hit" if cache_hit else "miss"
                )
                cache_details = f"key={cache_key}"

        dump_graph("cache", graph, f"cache_state={cache_state}\n{cache_details}".rstrip())
        if cache_hit:
            dump_graph(
                "post_grad",
                graph,
                "Skipped post-grad passes because the FX graph cache hit.",
            )
            post_grad_captures.append(graph)
            return make_boxed_func(graph.forward)

        view_to_reshape(graph)
        fake_mode = fake_tensor_prop(graph, inputs)
        _recursive_record_original_output_strides(graph)
        with V.set_fake_mode(fake_mode), get_cuda_device_context(graph):
            _recursive_post_grad_passes(graph, is_inference=True)
        graph.recompile()
        dump_graph("post_grad", graph)
        post_grad_captures.append(graph)
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

    if not post_grad_captures:
        raise RuntimeError("Inductor did not produce a post-grad FX graph")
    post_grad_graph = post_grad_captures[-1]
    placeholder_count = sum(node.op == "placeholder" for node in post_grad_graph.graph.nodes)
    if placeholder_count != len(example_inputs):
        raise RuntimeError(
            f"post-grad graph expects {placeholder_count} inputs; "
            f"export supplied {len(example_inputs)}"
        )
    with torch.no_grad():
        eager_output = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
    eager_logits = _logits(eager_output)
    run_metadata["eager_logits"] = {
        "shape": list(eager_logits.shape),
        "dtype": str(eager_logits.dtype),
    }
    with torch.no_grad():
        post_grad_output = post_grad_graph(*example_inputs)
    post_grad_logits = _logits(post_grad_output)

    for field, actual, expected in (
        ("shape", post_grad_logits.shape, eager_logits.shape),
        ("dtype", post_grad_logits.dtype, eager_logits.dtype),
        ("device", post_grad_logits.device, eager_logits.device),
    ):
        if actual != expected:
            raise AssertionError(f"logit {field} mismatch: eager={expected}, post-grad={actual}")
    torch.testing.assert_close(post_grad_logits, eager_logits)
    max_abs_error = (post_grad_logits.float() - eager_logits.float()).abs().max().item()
    run_metadata["post_grad_logits"] = {
        "shape": list(post_grad_logits.shape),
        "dtype": str(post_grad_logits.dtype),
        "max_abs_error": max_abs_error,
        "matches_eager": True,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        "eager vs post-grad logits: PASS "
        f"shape={tuple(eager_logits.shape)} dtype={eager_logits.dtype} "
        f"max_abs_error={max_abs_error}"
    )
    previous_match = run_metadata.get("matches_previous_post_grad_graph")
    if previous_match is None:
        print("saved CUDA phase-6 structural match: no previous trace found")
    else:
        print(f"saved CUDA phase-6 structural match: {previous_match}")
    print(f"run metadata: {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
