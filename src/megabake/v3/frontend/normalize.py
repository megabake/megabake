"""Version-bounded, side-effect-free normalization adapter."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Iterable

from .capture import FrontendError, NormalizedProgram, capture_exported_program, capture_graph_module


class NormalizationCompatibilityError(FrontendError):
    pass


def _minimal_compatibility_name() -> str:
    """Record the installed torch major/minor without importing private APIs."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - surfaced by capture first on normal installs
        raise NormalizationCompatibilityError("minimal normalization requires PyTorch") from exc
    version = str(torch.__version__).split("+", 1)[0]
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise NormalizationCompatibilityError(f"unparseable PyTorch version for minimal normalization: {version!r}")
    # The minimal route intentionally invokes no unstable Inductor private API.
    # Its version label makes the compatibility boundary explicit in artifacts.
    return f"minimal-torch-{parts[0]}.{parts[1]}"


def _result_graph(result: Any, previous: Any) -> Any:
    """Accept both mutating passes (None) and replacement-returning passes."""
    return previous if result is None else result


def apply_passes(graph_module: Any, passes: Iterable[Callable[[Any], Any]]) -> Any:
    graph = graph_module
    for transform in passes:
        graph = _result_graph(transform(graph), graph)
    if not hasattr(graph, "graph"):
        raise NormalizationCompatibilityError("normalization pass did not return an FX GraphModule")
    graph.graph.lint()
    graph.recompile()
    return graph


def normalize_fx(graph: Any, example_args: Any = None, *, input_spec: Any, policy: Any = None,
                 passes: Iterable[Callable[[Any], Any]] = (), allow_minimal: bool = True) -> NormalizedProgram:
    """Create a ``NormalizedProgram`` with an explicit, local transform sequence.

    The installed PyTorch private Inductor pass APIs vary substantially.  V3's
    safe initial path is deliberately the named minimal path: no private global
    pass is called unless a caller supplies one in ``passes``.
    """

    # Materialise once: callers may deliberately provide a one-shot iterable.
    # Checking compatibility must not silently skip its transforms.
    selected_passes = tuple(passes)
    if hasattr(graph, "graph_signature") and hasattr(graph, "graph_module"):
        program = capture_exported_program(graph, input_spec=input_spec, policy=policy)
    else:
        program = capture_graph_module(graph, example_args, input_spec=input_spec, policy=policy)
    if not allow_minimal and not selected_passes:
        raise NormalizationCompatibilityError("no compatible pinned normalization sequence was supplied")
    # Never mutate the capture graph or global decomposition tables: subsequent
    # requests and torch.compile calls must see their original state.
    normalized_graph = deepcopy(program.graph_module)
    normalized_graph = apply_passes(normalized_graph, selected_passes)
    path = _minimal_compatibility_name() if not selected_passes else "caller_allowlisted"
    return replace(program.with_graph(normalized_graph), normalization_path=path)
