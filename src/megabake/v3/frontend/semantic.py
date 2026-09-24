"""Small versioned semantic dialect kept on top of normalized FX."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping

from .capture import NormalizedProgram
from .facts import FactTable, collect_facts


class SemanticError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticDefinition:
    name: str
    version: str
    expand: Callable[[Mapping[str, Any]], Any]
    effects: tuple[str, ...] = ()

    def key(self) -> tuple[str, str]:
        return self.name, self.version


@dataclass(frozen=True)
class SemanticNode:
    op_id: str
    name: str
    version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    origin_nodes: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    effects: tuple[str, ...] = ()
    live_boundaries: tuple[str, ...] = ()
    reference: Callable[[], Any] | None = None

    @property
    def is_effectful(self) -> bool:
        return bool(self.effects)


@dataclass(frozen=True)
class ReferenceRegion:
    node_id: str
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    reference: Callable[..., Any] | None = None


@dataclass
class SemanticGraph:
    program: NormalizedProgram
    facts: FactTable
    operations: tuple[SemanticNode, ...]
    reference_regions: tuple[ReferenceRegion, ...]
    composite_alternatives: tuple[Any, ...] = ()

    def by_origin(self) -> dict[str, SemanticNode]:
        return {origin: operation for operation in self.operations for origin in operation.origin_nodes}

    @property
    def structural_hash(self) -> str:
        payload = [(op.name, op.version, op.inputs, op.outputs, _stable(op.attributes), op.effects) for op in self.operations]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class SemanticRegistry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], SemanticDefinition] = {}

    def register(self, definition: SemanticDefinition) -> None:
        if definition.key() in self._definitions:
            raise SemanticError(f"duplicate semantic definition {definition.name}:{definition.version}")
        self._definitions[definition.key()] = definition

    def definition(self, name: str, version: str = "v1") -> SemanticDefinition:
        try:
            return self._definitions[(name, version)]
        except KeyError as exc:
            raise SemanticError(f"unknown semantic definition {name}:{version}") from exc

    def expand(self, node: SemanticNode) -> Any:
        definition = self.definition(node.name, node.version)
        result = definition.expand(node.attributes)
        if result is None:
            raise SemanticError(f"{node.op_id}: reference expansion returned None")
        return result


def _linear_reference(attributes: Mapping[str, Any]) -> Mapping[str, Any]:
    # The actual executable original FX region remains the canonical reference;
    # this structured description validates that a Linear has all its semantics.
    required = ("alpha", "beta", "weight_layout")
    missing = [name for name in required if name not in attributes]
    if missing:
        raise SemanticError(f"Linear reference missing attributes: {', '.join(missing)}")
    return {"op": "Linear", **dict(attributes)}


DEFAULT_REGISTRY = SemanticRegistry()
DEFAULT_REGISTRY.register(SemanticDefinition("Linear", "v1", _linear_reference))
DEFAULT_REGISTRY.register(SemanticDefinition("Pointwise", "v1", lambda attrs: {"op": "Pointwise", **dict(attrs)}))
DEFAULT_REGISTRY.register(SemanticDefinition("RMSNorm", "v1", lambda attrs: {"op": "RMSNorm", **dict(attrs)}))
DEFAULT_REGISTRY.register(SemanticDefinition("RoPE", "v1", lambda attrs: {"op": "RoPE", **dict(attrs)}))
DEFAULT_REGISTRY.register(SemanticDefinition("SDPA", "v1", lambda attrs: {"op": "SDPA", **dict(attrs)}))
DEFAULT_REGISTRY.register(SemanticDefinition("CacheUpdate", "v1", lambda attrs: {"op": "CacheUpdate", **dict(attrs)}, ("state",)))
DEFAULT_REGISTRY.register(SemanticDefinition("SwiGLU", "v1", lambda attrs: {"op": "SwiGLU", **dict(attrs)}))


def _node_inputs(node: Any, facts: FactTable) -> tuple[str, ...]:
    values: list[str] = []
    for input_node in node.all_input_nodes:
        value_id = facts.node_to_value.get(input_node.name)
        if value_id:
            values.append(value_id)
    return tuple(values)


def semantic_node(node: Any, facts: FactTable, name: str, *, attributes: Mapping[str, Any],
                  origins: tuple[Any, ...] | None = None, effects: tuple[str, ...] = (),
                  reference: Callable[[], Any] | None = None) -> SemanticNode:
    output = facts.node_to_value.get(node.name, node.name)
    origin_nodes = tuple(item.name if hasattr(item, "name") else str(item) for item in (origins or (node,)))
    live_boundaries = tuple(
        facts.node_to_value[user.name] for user in node.users
        if user.name in facts.node_to_value and user.name not in origin_nodes
    )
    return SemanticNode(f"op:{node.name}", name, "v1", _node_inputs(node, facts), (output,), origin_nodes,
                        dict(attributes), effects, live_boundaries, reference)


def _validate_operation(operation: SemanticNode, facts: FactTable, definition: SemanticDefinition) -> None:
    if not operation.inputs or not operation.outputs:
        raise SemanticError(f"{operation.op_id}: semantic operation needs explicit live inputs and outputs")
    unknown = [value for value in operation.inputs + operation.outputs if value not in facts.facts]
    if unknown:
        raise SemanticError(f"{operation.op_id}: semantic operation refers to unknown values {unknown!r}")
    if len(set(operation.origin_nodes)) != len(operation.origin_nodes):
        raise SemanticError(f"{operation.op_id}: duplicate origin node")
    if definition.effects and not operation.effects:
        raise SemanticError(f"{operation.op_id}: {operation.name} omitted required effects")


def recognize(program: NormalizedProgram, *, facts: FactTable | None = None,
              registry: SemanticRegistry = DEFAULT_REGISTRY) -> SemanticGraph:
    """Recognize the intentionally small V3 subset; retain every other node."""
    facts = facts or collect_facts(program)
    from .match_attention import match_attention
    from .match_linear import match_linear
    from .match_norm import match_rmsnorm
    from .match_pointwise import match_pointwise
    from .match_rope import match_rope
    from .match_state import match_cache_updates
    # Larger semantic regions claim their FX origins before their constituent
    # pointwise operations.  The latter stay represented as live boundaries.
    matchers = (match_linear, match_rmsnorm, match_rope, match_attention, match_cache_updates, match_pointwise)
    operations: list[SemanticNode] = []
    consumed: set[str] = set()
    for matcher in matchers:
        for operation in matcher(program, facts):
            if any(origin in consumed for origin in operation.origin_nodes):
                continue
            definition = registry.definition(operation.name, operation.version)
            _validate_operation(operation, facts, definition)
            registry.expand(operation)
            # A semantic region always retains an executable oracle.  It is the
            # original exported/FX reference callable, never a lowerer rewrite.
            operations.append(SemanticNode(
                operation.op_id, operation.name, operation.version, operation.inputs,
                operation.outputs, operation.origin_nodes, operation.attributes,
                operation.effects, operation.live_boundaries, program.run_reference,
            ))
            consumed.update(operation.origin_nodes)
    references: list[ReferenceRegion] = []
    for node in program.graph_module.graph.nodes:
        if node.op in {"placeholder", "output"} or node.name in consumed:
            continue
        references.append(ReferenceRegion(node.name, str(node.target), _node_inputs(node, facts),
                                          (facts.node_to_value[node.name],) if node.name in facts.node_to_value else (),
                                          program.run_reference))
    return SemanticGraph(program, facts, tuple(operations), tuple(references))
