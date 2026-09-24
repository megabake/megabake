"""Preserve FX/export ABI information without touching CUDA.

The records here intentionally keep the original PyTorch objects alive.  They are
not a second graph format: later frontend passes operate on ``graph_module`` and
carry this ABI envelope forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping


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


def capture_exported_program(exported_program: Any, *, input_spec: Any = None, policy: Any = None) -> NormalizedProgram:
    """Wrap an ``ExportedProgram`` without recapturing or changing its bindings."""

    if not hasattr(exported_program, "graph_module") or not hasattr(exported_program, "graph_signature"):
        raise TypeError("capture_exported_program requires torch.export.ExportedProgram")
    bindings, values, effects = _binding_data(exported_program)
    call_spec = getattr(exported_program, "call_spec", None)
    return NormalizedProgram(
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
    )


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
        permitted = target.startswith(("aten.", "torch.", "<built-in function", "_operator."))
        forbidden = ("cond", "while_loop", "map_impl", "higher_order", "python")
        if not permitted or any(marker in target.lower() for marker in forbidden):
            raise UnsupportedGraphError(f"unsupported custom effect/control-flow node {node.name}: {target}")


def capture_graph_module(graph_module: Any, example_args: Any = None, *, input_spec: Any = None, policy: Any = None) -> NormalizedProgram:
    """Capture a GraphModule through export using caller-provided examples only."""

    if example_args is None or input_spec is None:
        raise FrontendError("GraphModule capture requires example_args and input_spec")
    try:
        import torch
        from torch.fx import GraphModule
    except ImportError as exc:  # pragma: no cover - import diagnostic on minimal installs
        raise FrontendError("GraphModule capture requires PyTorch") from exc
    if not isinstance(graph_module, GraphModule):
        raise TypeError("capture_graph_module requires torch.fx.GraphModule")
    _validate_graphmodule_capture_surface(graph_module)
    args = _as_args(example_args)
    # Export performs functionalization/metadata capture; it does not inspect a device.
    try:
        exported = torch.export.export(graph_module, args, strict=False)
    except Exception as exc:
        raise UnsupportedGraphError(f"GraphModule export failed: {type(exc).__name__}: {exc}") from exc
    return capture_exported_program(exported, input_spec=input_spec, policy=policy)
