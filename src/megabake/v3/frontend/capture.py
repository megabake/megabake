"""Preserve FX/export ABI information without touching CUDA.

The records here intentionally keep the original PyTorch objects alive.  They are
not a second graph format: later frontend passes operate on ``graph_module`` and
carry this ABI envelope forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Callable, Mapping

from ..contracts import StepABI


class FrontendError(ValueError):
    """A graph is outside V3's explicit CPU frontend contract."""


class BindingError(FrontendError):
    """A lifted parameter, buffer, or constant cannot be bound by name."""


class UnsupportedGraphError(FrontendError):
    """The graph has an effect/control-flow form that V3 does not model yet."""


@dataclass(frozen=True)
class LiftedBinding:
    placeholder: str
    target: str
    role: str
    persistent: bool | None = None


@dataclass(frozen=True)
class EffectFact:
    node_id: str
    kind: str
    target: str | None = None
    required: bool = True


@dataclass
class NormalizedProgram:
    """FX plus the ABI facts that cannot be reconstructed from node order."""

    graph_module: Any
    reference: Callable[..., Any]
    export_signature: Any | None
    lifted_bindings: Mapping[str, LiftedBinding]
    binding_values: Mapping[str, Any]
    constraints: tuple[Any, ...] = ()
    effects: tuple[EffectFact, ...] = ()
    input_spec: Any = None
    input_tree_spec: Any = None
    output_tree_spec: Any = None
    value_ids: Mapping[str, str] = field(default_factory=dict)
    source_kind: str = "graph_module"
    policy: Any = None
    normalization_path: str = "capture"
    state_bindings: Mapping[str, str] = field(default_factory=dict)

    def with_graph(self, graph_module: Any, *, effects: tuple[EffectFact, ...] | None = None) -> "NormalizedProgram":
        return replace(self, graph_module=graph_module, effects=self.effects if effects is None else effects)

    def binding_for(self, placeholder: str) -> Any:
        try:
            binding = self.lifted_bindings[placeholder]
            return self.binding_values[binding.target]
        except KeyError as exc:
            raise BindingError(f"missing lifted binding for {placeholder!r}") from exc

    def run_reference(self, *args: Any, **kwargs: Any) -> Any:
        return self.reference(*args, **kwargs)


def _value_ids(graph_module: Any) -> dict[str, str]:
    return {node.name: f"v{index}" for index, node in enumerate(graph_module.graph.nodes)}


def _fx_value(value: Any) -> Any:
    if hasattr(value, "op") and hasattr(value, "name"):
        return {"node": value.name}
    if isinstance(value, Mapping):
        return {str(key): _fx_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_fx_value(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [_fx_value(value.start), _fx_value(value.stop), _fx_value(value.step)]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ in {"dtype", "device", "layout", "SymInt", "SymBool"}:
        return {"type": value_type.__name__, "value": str(value)}
    raise UnsupportedGraphError(f"cannot hash FX argument of type {value_type.__name__} without unstable object identity")


def graph_hash(program: "NormalizedProgram") -> str:
    """Hash graph meaning without tensor data pointers or Python object IDs."""
    nodes = [
        [node.op, str(node.target), _fx_value(node.args), _fx_value(node.kwargs)]
        for node in program.graph_module.graph.nodes
    ]
    payload = {
        "nodes": nodes,
        "user_inputs": list(_user_placeholders(program)),
        "lifted_bindings": {
            name: {"target": item.target, "role": item.role, "persistent": item.persistent}
            for name, item in sorted(program.lifted_bindings.items())
        },
        "state_bindings": dict(sorted(program.state_bindings.items())),
        "effects": [
            [effect.node_id, effect.kind, effect.target, effect.required]
            for effect in program.effects
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_state_bindings(graph_module: Any, state_bindings: Mapping[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    placeholders = {node.name for node in graph_module.graph.nodes if node.op == "placeholder"}
    for placeholder, value in (state_bindings or {}).items():
        if placeholder not in placeholders:
            raise BindingError(f"state binding refers to unknown placeholder {placeholder!r}")
        identity = value.get("state_id") if isinstance(value, Mapping) else value
        if not isinstance(identity, str) or not identity.strip():
            raise BindingError(f"state binding for {placeholder!r} needs a non-empty state_id")
        result[placeholder] = identity.strip()
    return result


def _state_update_effects(graph_module: Any, state_bindings: Mapping[str, str]) -> tuple[EffectFact, ...]:
    """Mark functional state writes whose source descends from a declared state input."""
    state_nodes = {name: identity for name, identity in state_bindings.items()}
    effects: list[EffectFact] = []
    state_updates = {"index_copy", "slice_scatter", "scatter", "index_put", "copy"}
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target).removeprefix("aten.")
        operator = target.split(".", 1)[0]
        sources = [state_nodes[item.name] for item in node.all_input_nodes if item.name in state_nodes]
        if sources and operator.removesuffix("_") in state_updates:
            effects.append(EffectFact(node.name, "functional_state_update", sources[0]))
        if sources:
            state_nodes[node.name] = sources[0]
    return tuple(effects)


def _user_placeholders(program: NormalizedProgram) -> tuple[str, ...]:
    signature = program.export_signature
    if signature is None:
        return tuple(node.name for node in program.graph_module.graph.nodes if node.op == "placeholder")
    result = []
    for spec in getattr(signature, "input_specs", ()):
        kind = str(getattr(spec, "kind", "")).split(".")[-1].lower()
        name = getattr(getattr(spec, "arg", None), "name", None)
        if kind == "user_input" and name is not None:
            result.append(name)
    return tuple(result)


def _validate_step_abi_capture(program: NormalizedProgram, step_abi: StepABI, output: Any = None) -> None:
    expected_inputs = tuple(item["placeholder"] for item in step_abi.ordered_user_inputs)
    actual_inputs = _user_placeholders(program)
    if actual_inputs != expected_inputs:
        raise UnsupportedGraphError(
            f"capture signature is incomplete or reordered: expected user inputs {expected_inputs!r}, got {actual_inputs!r}"
        )
    expected_states = {item["placeholder"]: item["state_id"] for item in step_abi.old_state_inputs}
    if dict(program.state_bindings) != expected_states:
        raise BindingError(
            f"state bindings differ from StepABI: expected {expected_states!r}, got {dict(program.state_bindings)!r}"
        )
    expected_bindings = step_abi.lifted_bindings
    if set(program.lifted_bindings) != set(expected_bindings):
        raise BindingError("lifted bindings differ from StepABI")
    for placeholder, expected in expected_bindings.items():
        actual = program.lifted_bindings[placeholder]
        if actual.target != expected["identity"] or actual.role != expected["role"]:
            raise BindingError(f"lifted binding {placeholder!r} differs from StepABI")
    actual_states = {effect.target for effect in program.effects if effect.required}
    missing_effects = {item["state_id"] for item in step_abi.state_effects}.difference(actual_states)
    if missing_effects:
        raise UnsupportedGraphError(f"capture omitted required state effects: {sorted(missing_effects)!r}")
    if output is None:
        if program.output_tree_spec is None:
            raise UnsupportedGraphError("capture is missing the declared user output tree")
        return
    _validate_tree(output, step_abi.user_output_tree["structure"])
    for state_output in step_abi.new_state_outputs:
        _value_at_path(output, state_output["path"])


def _value_at_path(value: Any, path: Any) -> Any:
    for component in path:
        value = value[component]
    return value


def _validate_tree(value: Any, spec: Any, path: str = "output") -> None:
    """Check a small JSON tree descriptor (or a PyTorch TreeSpec) against a value."""
    if hasattr(spec, "num_leaves") and hasattr(spec, "children_specs"):
        try:
            import torch.utils._pytree as pytree
            actual = pytree.tree_structure(value)
        except (ImportError, AttributeError):
            raise UnsupportedGraphError(f"{path}: cannot validate supplied output TreeSpec")
        if actual != spec:
            raise UnsupportedGraphError(f"{path}: output tree differs from supplied TreeSpec")
        return
    if isinstance(spec, Mapping):
        if not isinstance(value, Mapping) or set(value.keys()) != set(spec.keys()):
            raise UnsupportedGraphError(f"{path}: output mapping differs from supplied tree")
        for key, child in spec.items():
            _validate_tree(value[key], child, f"{path}.{key}")
        return
    if isinstance(spec, (tuple, list)):
        if not isinstance(value, type(spec)) or len(value) != len(spec):
            raise UnsupportedGraphError(f"{path}: output sequence differs from supplied tree")
        for index, (item, child) in enumerate(zip(value, spec)):
            _validate_tree(item, child, f"{path}[{index}]")
        return
    if spec in {"tensor", "torch.Tensor"}:
        try:
            import torch
            is_tensor = isinstance(value, torch.Tensor)
        except ImportError:
            is_tensor = False
        if not is_tensor:
            raise UnsupportedGraphError(f"{path}: expected a tensor output leaf")


def _clone_tree(value: Any) -> Any:
    try:
        import torch
        import torch.utils._pytree as pytree
    except ImportError:  # pragma: no cover - capture reports missing torch before use
        return value
    return pytree.tree_map(lambda item: item.clone() if isinstance(item, torch.Tensor) else item, value)


def _assert_same_value(expected: Any, actual: Any, path: str = "output") -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - capture reports missing torch before use
        return
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise UnsupportedGraphError(f"re-export changed tensor structure at {path}")
        if expected.shape != actual.shape or expected.dtype != actual.dtype or expected.device != actual.device:
            raise UnsupportedGraphError(f"re-export changed tensor metadata at {path}")
        if expected.is_floating_point() or expected.is_complex():
            try:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
            except AssertionError as exc:
                raise UnsupportedGraphError(f"re-export changed values at {path}") from exc
        elif not torch.equal(expected, actual):
            raise UnsupportedGraphError(f"re-export changed values at {path}")
        return
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping) or set(expected.keys()) != set(actual.keys()):
            raise UnsupportedGraphError(f"re-export changed output mapping at {path}")
        for key in expected:
            _assert_same_value(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (tuple, list)) or isinstance(actual, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise UnsupportedGraphError(f"re-export changed output sequence at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_same_value(left, right, f"{path}[{index}]")
        return
    if type(expected) is not type(actual) or expected != actual:
        raise UnsupportedGraphError(f"re-export changed output leaf at {path}")


def _binding_data(exported_program: Any) -> tuple[dict[str, LiftedBinding], dict[str, Any], tuple[EffectFact, ...]]:
    signature = exported_program.graph_signature
    state = dict(exported_program.state_dict)
    constants = dict(getattr(exported_program, "constants", {}) or {})
    bindings: dict[str, LiftedBinding] = {}
    # Keep the two namespaces explicit while accepting the Export API's stable
    # target names.  A collision would make a captured program ambiguous.
    overlap = set(state).intersection(constants)
    if overlap:
        raise BindingError(f"binding target is both state and constant: {sorted(overlap)!r}")
    values = {**state, **constants}
    for spec in getattr(signature, "input_specs", ()):
        kind = str(getattr(spec, "kind", "")).split(".")[-1].lower()
        argument = getattr(spec, "arg", None)
        placeholder = getattr(argument, "name", None)
        target = getattr(spec, "target", None)
        if placeholder is None or target is None:
            continue
        role = {
            "parameter": "weight",
            "buffer": "state",
            "constant_tensor": "constant",
            "custom_obj": "constant",
        }.get(kind, kind)
        if target not in values:
            raise BindingError(f"missing {kind} binding {target!r} for placeholder {placeholder!r}")
        bindings[placeholder] = LiftedBinding(placeholder, target, role, getattr(spec, "persistent", None))
    effects: list[EffectFact] = []
    for spec in getattr(signature, "output_specs", ()):
        kind = str(getattr(spec, "kind", "")).split(".")[-1].lower()
        if "mutation" in kind:
            argument = getattr(spec, "arg", None)
            node_id = getattr(argument, "name", None)
            if not node_id:
                raise BindingError(f"mutation output lacks an FX value name for target {getattr(spec, 'target', None)!r}")
            effects.append(EffectFact(node_id, kind, getattr(spec, "target", None)))
    # Some supported torch versions preserve in-place FX nodes without a
    # corresponding output-spec mutation entry.  Keep those transitions
    # explicit instead of assuming visible user outputs are the only roots.
    known_effects = {effect.node_id for effect in effects}
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        operator = target.removeprefix("aten.").split(".", 1)[0]
        if operator.endswith("_") or operator in {"copy", "index_put", "scatter", "set"}:
            if node.name not in known_effects:
                state = node.args[0] if node.args else None
                effects.append(EffectFact(node.name, "fx_mutation", getattr(state, "name", None)))
                known_effects.add(node.name)
    return bindings, values, tuple(effects)


def capture_exported_program(
    exported_program: Any,
    *,
    input_spec: Any = None,
    policy: Any = None,
    state_bindings: Mapping[str, Any] | None = None,
    step_abi: StepABI | None = None,
) -> NormalizedProgram:
    """Wrap an ``ExportedProgram`` without recapturing or changing its bindings."""

    if not hasattr(exported_program, "graph_module") or not hasattr(exported_program, "graph_signature"):
        raise TypeError("capture_exported_program requires torch.export.ExportedProgram")
    bindings, values, effects = _binding_data(exported_program)
    normalized_states = _normalize_state_bindings(exported_program.graph_module, state_bindings)
    effects += _state_update_effects(exported_program.graph_module, normalized_states)
    call_spec = getattr(exported_program, "call_spec", None)
    program = NormalizedProgram(
        graph_module=exported_program.graph_module,
        reference=exported_program.module(),
        export_signature=exported_program.graph_signature,
        lifted_bindings=bindings,
        binding_values=values,
        constraints=tuple(getattr(exported_program, "range_constraints", {}).items()),
        effects=effects,
        input_spec=input_spec,
        input_tree_spec=getattr(call_spec, "in_spec", None),
        output_tree_spec=getattr(call_spec, "out_spec", None),
        value_ids=_value_ids(exported_program.graph_module),
        source_kind="exported_program",
        policy=policy,
        state_bindings=normalized_states,
    )
    if step_abi is not None:
        if not isinstance(step_abi, StepABI):
            raise TypeError("step_abi must be a StepABI")
        _validate_step_abi_capture(program, step_abi)
    return program


def _as_args(example_args: Any) -> tuple[Any, ...]:
    if isinstance(example_args, tuple):
        return example_args
    if isinstance(example_args, list):
        return tuple(example_args)
    return (example_args,)


def _validate_graphmodule_capture_surface(graph_module: Any) -> None:
    """Reject an FX graph that would require executing an unknown callback.

    ``torch.export`` necessarily evaluates supported tensor operations to obtain
    an export.  The frontend must not use it as a generic metadata probe for
    arbitrary Python code, however.  The small allowlist below accepts ordinary
    torch/operator call targets and rejects higher-order/control-flow and custom
    functions before export is attempted.
    """
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op == "call_method":
            raise UnsupportedGraphError(f"unsupported FX method/effect node {node.name}: {node.target}")
        if node.op == "call_module":
            try:
                module = graph_module.get_submodule(str(node.target))
            except (AttributeError, KeyError) as exc:
                raise UnsupportedGraphError(f"missing FX module node {node.name}: {node.target}") from exc
            if not module.__class__.__module__.startswith("torch.nn"):
                raise UnsupportedGraphError(f"unsupported FX module/control-flow node {node.name}: {node.target}")
            continue
        if node.op != "call_function":
            raise UnsupportedGraphError(f"unsupported FX node {node.name}: {node.op}")
        target = str(node.target)
        if " at 0x" in target:
            target = ".".join(filter(None, (
                getattr(node.target, "__module__", None),
                getattr(node.target, "__qualname__", getattr(node.target, "__name__", None)),
            ))) or type(node.target).__name__
        permitted = target.startswith(("aten.", "torch.", "<built-in function", "_operator."))
        forbidden = ("cond", "while_loop", "map_impl", "higher_order", "python")
        if not permitted or any(marker in target.lower() for marker in forbidden):
            raise UnsupportedGraphError(f"unsupported custom effect/control-flow node {node.name}: {target}")


def capture_graph_module(
    graph_module: Any,
    example_args: Any = None,
    *,
    input_spec: Any = None,
    output_spec: Any = None,
    state_bindings: Mapping[str, Any] | None = None,
    step_abi: StepABI | None = None,
    policy: Any = None,
) -> NormalizedProgram:
    """Capture a GraphModule through export using caller-provided examples only."""

    if example_args is None or input_spec is None:
        raise FrontendError("GraphModule capture requires example_args and input_spec")
    if step_abi is not None and (output_spec is None or state_bindings is None):
        raise FrontendError("StepABI GraphModule capture requires explicit output_spec and state_bindings")
    try:
        import torch
        from torch.fx import GraphModule
    except ImportError as exc:  # pragma: no cover - import diagnostic on minimal installs
        raise FrontendError("GraphModule capture requires PyTorch") from exc
    if not isinstance(graph_module, GraphModule):
        raise TypeError("capture_graph_module requires torch.fx.GraphModule")
    _validate_graphmodule_capture_surface(graph_module)
    args = _as_args(example_args)
    placeholders = tuple(node.name for node in graph_module.graph.nodes if node.op == "placeholder")
    if len(placeholders) != len(args):
        raise BindingError(f"GraphModule expects {len(placeholders)} positional inputs, got {len(args)} examples")
    normalized_states = _normalize_state_bindings(graph_module, state_bindings)
    # Export performs functionalization/metadata capture; it does not inspect a device.
    try:
        exported = torch.export.export(graph_module, args, strict=False)
    except Exception as exc:
        raise UnsupportedGraphError(f"GraphModule export failed: {type(exc).__name__}: {exc}") from exc
    program = capture_exported_program(
        exported, input_spec=input_spec, policy=policy, state_bindings=normalized_states,
    )
    if _user_placeholders(program) != placeholders:
        raise UnsupportedGraphError(
            f"re-export changed the user input signature: expected {placeholders!r}, got {_user_placeholders(program)!r}"
        )
    original_args = _clone_tree(args)
    exported_args = _clone_tree(args)
    try:
        original_output = graph_module(*original_args)
        captured_output = program.run_reference(*exported_args)
    except Exception as exc:
        raise UnsupportedGraphError(f"could not compare original and exported graph behavior: {type(exc).__name__}: {exc}") from exc
    if output_spec is not None:
        _validate_tree(original_output, output_spec)
        _validate_tree(captured_output, output_spec)
    _assert_same_value(original_output, captured_output)
    _assert_same_value(original_args, exported_args, "mutated_inputs")
    if step_abi is not None:
        if not isinstance(step_abi, StepABI):
            raise TypeError("step_abi must be a StepABI")
        _validate_step_abi_capture(program, step_abi, captured_output)
    return program
