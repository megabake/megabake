"""Version-bounded FX normalization that retains executable capture provenance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from .capture import (
    FrontendError, NormalizedProgram, UnsupportedGraphError, _ORIGIN_META,
    _assert_same_value, _clone_tree, _stable_target, capture_exported_program,
    capture_graph_module,
)


class NormalizationCompatibilityError(FrontendError):
    pass


def _minimal_compatibility_name() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - surfaced by capture first on normal installs
        raise NormalizationCompatibilityError("minimal normalization requires PyTorch") from exc
    version = str(torch.__version__).split("+", 1)[0]
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise NormalizationCompatibilityError(f"unparseable PyTorch version for minimal normalization: {version!r}")
    return f"minimal-torch-{parts[0]}.{parts[1]}"


def _result_graph(result: Any, previous: Any) -> Any:
    return previous if result is None else result


def _node_ids(node: Any) -> tuple[str, ...]:
    return tuple(getattr(node, "meta", {}).get(_ORIGIN_META, ()))


def _node_refs(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "op") and hasattr(value, "name"):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(ref for item in value.values() for ref in _node_refs(item))
    if isinstance(value, (tuple, list)):
        return tuple(ref for item in value for ref in _node_refs(item))
    return ()


def _transform_key(transform: Callable[..., Any], index: int) -> str:
    return f"{getattr(transform, '__module__', type(transform).__module__)}.{getattr(transform, '__qualname__', type(transform).__qualname__)}:{index}"


def _attach_origins(graph: Any, previous: Any, program: NormalizedProgram,
                    transform_key: str) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    old_nodes = {node.name: node for node in previous.graph.nodes}
    lineage = dict(program.origin_lineage)
    origins: dict[str, tuple[str, ...]] = {}
    for index, node in enumerate(graph.graph.nodes):
        ids = _node_ids(node)
        if not ids and node.name in old_nodes:
            ids = _node_ids(old_nodes[node.name]) or program.origin_map.get(node.name, ())
        if not ids and node.op == "output":
            old_output = next((item for item in old_nodes.values() if item.op == "output"), None)
            if old_output is not None:
                ids = _node_ids(old_output) or program.origin_map.get(old_output.name, ())
        if not ids:
            parents = tuple(dict.fromkeys(origin for ref in _node_refs((node.args, node.kwargs)) for origin in _node_ids(ref)))
            key = json.dumps([transform_key, index, _stable_target(node.target), parents], separators=(",", ":"))
            generated = "fxgen:" + hashlib.sha256(key.encode()).hexdigest()
            ids = (generated,)
            lineage[generated] = parents
            node.meta[_ORIGIN_META] = ids
        origins[node.name] = ids
    graph.graph.lint()
    graph.recompile()
    return origins, lineage


def _origin_history(origins: Mapping[str, tuple[str, ...]], lineage: Mapping[str, tuple[str, ...]],
                    previous: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    all_ids = set(previous) | set(lineage) | {origin for values in origins.values() for origin in values}

    def ancestors(origin: str, visiting: frozenset[str] = frozenset()) -> set[str]:
        if origin in visiting:
            return set()
        result = {origin}
        for parent in lineage.get(origin, ()):
            result.update(ancestors(parent, visiting | {origin}))
        return result

    return {
        origin: tuple(name for name, node_origins in origins.items()
                      if origin in set().union(*(ancestors(item) for item in node_origins)))
        for origin in all_ids
    }


def _apply_passes(graph_module: Any, passes: Iterable[Callable[[Any], Any]], program: NormalizedProgram
                  ) -> tuple[Any, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    graph = graph_module
    origins = dict(program.origin_map)
    lineage = dict(program.origin_lineage)
    history = dict(program.origin_history)
    for index, transform in enumerate(passes):
        for node in graph.graph.nodes:
            if not _node_ids(node) and node.name in origins:
                node.meta[_ORIGIN_META] = origins[node.name]
        previous = graph
        graph = _result_graph(transform(graph), graph)
        if not hasattr(graph, "graph"):
            raise NormalizationCompatibilityError("normalization pass did not return an FX GraphModule")
        origins, lineage = _attach_origins(graph, previous, program.with_graph(previous, origin_map=origins,
                                                                                origin_lineage=lineage),
                                           _transform_key(transform, index))
        history = _origin_history(origins, lineage, history)
    graph.graph.lint()
    graph.recompile()
    return graph, origins, lineage, history


def apply_passes(graph_module: Any, passes: Iterable[Callable[[Any], Any]]) -> Any:
    """Apply mutating or replacement-returning FX passes to a graph copy."""
    graph = graph_module
    for transform in passes:
        graph = _result_graph(transform(graph), graph)
    if not hasattr(graph, "graph"):
        raise NormalizationCompatibilityError("normalization pass did not return an FX GraphModule")
    graph.graph.lint()
    graph.recompile()
    return graph


def _example_inputs(program: NormalizedProgram, example_args: Any) -> tuple[Any, ...] | None:
    if example_args is not None:
        return example_args if isinstance(example_args, tuple) else tuple(example_args) if isinstance(example_args, list) else (example_args,)
    exported = program.source_exported_program
    pair = getattr(exported, "example_inputs", None)
    if isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], Mapping) and not pair[1]:
        return tuple(pair[0])
    return None


def _contains_non_cpu_tensor(value: Any) -> bool:
    if hasattr(value, "device"):
        return str(value.device) != "cpu"
    if isinstance(value, Mapping):
        return any(_contains_non_cpu_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_non_cpu_tensor(item) for item in value)
    return False


def _has_random_operator(graph_module: Any) -> bool:
    return any(
        node.op == "call_function" and any(name in str(node.target).lower()
                                            for name in ("dropout", "rand", "random", "bernoulli", "multinomial"))
        for node in graph_module.graph.nodes
    )


def _run_graph(program: NormalizedProgram, graph_module: Any, user_args: tuple[Any, ...]) -> tuple[Any, Any]:
    copied_inputs = _clone_tree(user_args)
    copied_inputs = tuple(copied_inputs) if isinstance(copied_inputs, (tuple, list)) else (copied_inputs,)
    if program.source_kind != "exported_program":
        return graph_module(*copied_inputs), copied_inputs
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    user_names = [node.name for node in placeholders if node.name not in program.lifted_bindings]
    if len(user_names) != len(copied_inputs):
        raise NormalizationCompatibilityError(
            f"cannot bind normalization examples: graph expects {len(user_names)} user inputs, got {len(copied_inputs)}"
        )
    users = dict(zip(user_names, copied_inputs))
    args = []
    for node in placeholders:
        binding = program.lifted_bindings.get(node.name)
        if binding is None:
            args.append(users[node.name])
        else:
            args.append(program.binding_values[binding.target])
    return graph_module(*args), copied_inputs


def _verify_effect_origins(program: NormalizedProgram, history: Mapping[str, tuple[str, ...]]) -> None:
    for effect in program.effects:
        source_origins = program.origin_map.get(effect.node_id, ())
        if not source_origins or not any(history.get(origin) for origin in source_origins):
            raise NormalizationCompatibilityError(
                f"normalization removed required {effect.kind} origin {effect.node_id!r}"
            )


def _selected_decompositions(exported_program: Any, selected: Iterable[Any]) -> Any:
    selected = tuple(selected)
    if not selected:
        return exported_program
    try:
        import torch
        table = dict(torch.export.default_decompositions())
    except (ImportError, AttributeError) as exc:
        raise NormalizationCompatibilityError("installed PyTorch lacks ExportedProgram decompositions") from exc
    missing = [str(target) for target in selected if target not in table]
    if missing:
        raise NormalizationCompatibilityError(f"no default decomposition for selected operator(s): {', '.join(missing)}")
    selected_table = {target: table[target] for target in selected}
    try:
        return exported_program.run_decompositions(selected_table)
    except Exception as exc:
        raise NormalizationCompatibilityError(f"selected ExportedProgram decomposition failed: {type(exc).__name__}: {exc}") from exc


def normalize_fx(graph: Any, example_args: Any = None, *, input_spec: Any, policy: Any = None,
                 passes: Iterable[Callable[[Any], Any]] = (), decomposition_ops: Iterable[Any] = (),
                 state_bindings: Mapping[str, Any] | None = None, output_spec: Any = None,
                 allow_minimal: bool = True) -> NormalizedProgram:
    """Normalize only explicit FX rewrites and selected, copied Export decompositions."""
    selected_passes = tuple(passes)
    selected_decompositions = tuple(decomposition_ops)
    is_export = hasattr(graph, "graph_signature") and hasattr(graph, "graph_module")
    if is_export:
        program = capture_exported_program(graph, input_spec=input_spec, policy=policy,
                                           state_bindings=state_bindings)
    else:
        if selected_decompositions:
            raise NormalizationCompatibilityError("ExportedProgram decompositions require an ExportedProgram input")
        program = capture_graph_module(graph, example_args, input_spec=input_spec, policy=policy,
                                       state_bindings=state_bindings, output_spec=output_spec,
                                       reexport=False)
    if not allow_minimal and not selected_passes and not selected_decompositions:
        raise NormalizationCompatibilityError("no compatible pinned normalization sequence was supplied")

    normalized_export = _selected_decompositions(graph, selected_decompositions) if is_export and selected_decompositions else None
    starting_graph = normalized_export.graph_module if normalized_export is not None else program.graph_module
    normalized_graph = deepcopy(starting_graph)
    if normalized_export is not None:
        origins, lineage = _attach_origins(
            normalized_graph, program.graph_module, program,
            "torch.export.run_decompositions:" + ",".join(_stable_target(item) for item in selected_decompositions),
        )
        history = _origin_history(origins, lineage, program.origin_history)
    else:
        origins, lineage, history = dict(program.origin_map), dict(program.origin_lineage), dict(program.origin_history)
    # Pass provenance accumulates after decomposition; each pass sees a private graph.
    if selected_passes:
        pass_program = program.with_graph(normalized_graph, origin_map=origins,
                                          origin_lineage=lineage, origin_history=history)
        normalized_graph, origins, lineage, history = _apply_passes(normalized_graph, selected_passes, pass_program)
    normalized = program.with_graph(normalized_graph, origin_map=origins,
                                    origin_lineage=lineage, origin_history=history)
    _verify_effect_origins(program, history)

    examples = _example_inputs(program, example_args)
    if examples is not None:
        try:
            if _contains_non_cpu_tensor(examples) and _has_random_operator(normalized_graph):
                raise NormalizationCompatibilityError(
                    "cannot compare a stochastic non-CPU FX graph without a target RNG-state adapter"
                )
            import torch
            rng_state = torch.random.get_rng_state()
            original_graph = program.source_graph_module or program.graph_module
            try:
                actual, original_inputs_after = _run_graph(program, original_graph, examples)
                original_rng_after = torch.random.get_rng_state()
                torch.random.set_rng_state(rng_state)
                candidate, normalized_inputs_after = _run_graph(program, normalized_graph, examples)
                candidate_rng_after = torch.random.get_rng_state()
            finally:
                torch.random.set_rng_state(rng_state)
            _assert_same_value(actual, candidate)
            _assert_same_value(original_inputs_after, normalized_inputs_after, "mutated_inputs")
            _assert_same_value(original_rng_after, candidate_rng_after, "cpu_rng_state")
        except UnsupportedGraphError:
            raise
        except Exception as exc:
            raise NormalizationCompatibilityError(
                f"could not compare original and normalized behavior: {type(exc).__name__}: {exc}"
            ) from exc

    if normalized_export is not None:
        path = f"torch-{_minimal_compatibility_name().removeprefix('minimal-torch-')}-default-decompositions"
    elif selected_passes:
        path = "caller_allowlisted"
    else:
        path = _minimal_compatibility_name()
    return replace(normalized, normalization_path=path)
