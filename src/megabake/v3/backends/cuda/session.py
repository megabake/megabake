"""Complete-call binding for an admitted V3 CUDA worker program."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Mapping

from ...frontend.capture import graph_hash
from .profile import CudaTargetProfile
from .worker import (
    WorkerEntrySession,
    WorkerProgram,
    WorkerProgramError,
    bind_worker_values,
    open_worker_session,
)


class CachedStepSessionError(ValueError):
    """A cached-step ABI or invocation is not covered by its compiled worker."""


@dataclass(frozen=True)
class ExecutionRecord:
    """Host dispatch record; it does not claim device completion latency."""

    worker_program_hash: str
    target_profile_key: str
    runtime_target: str
    grid_ctas: int
    binding_host_ns: int
    launch_host_ns: int
    launched: bool
    return_code: int
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_program_hash": self.worker_program_hash,
            "target_profile_key": self.target_profile_key,
            "runtime_target": self.runtime_target,
            "grid_ctas": self.grid_ctas,
            "binding_host_ns": self.binding_host_ns,
            "launch_host_ns": self.launch_host_ns,
            "launched": self.launched,
            "return_code": self.return_code,
            "failure_reason": self.failure_reason,
            "timing_scope": "host_dispatch_only; CUDA completion must be synchronized by the caller",
        }


def _at_path(value: Any, path: Any) -> Any:
    for component in path:
        value = value[component]
    return value


def _has_owned_storage_root(value_id: str, values: Mapping[str, Any], seen: frozenset[str] = frozenset()) -> bool:
    """Prove an output view ultimately refers to a fresh graph allocation."""
    if value_id in seen:
        return False
    value = values.get(value_id)
    if value is None:
        return False
    if value.alias_kind in {"fresh", "copy"}:
        return value.alias_set == f"storage:{value_id}"
    if value.alias_kind != "view" or len(value.alias_sources) != 1:
        return False
    return _has_owned_storage_root(value.alias_sources[0], values, seen | {value_id})


def _storage_key(tensor: Any) -> tuple[str, int, int] | None:
    if not tensor.numel():
        return None
    try:
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        return (str(tensor.device), pointer, int(storage.nbytes())) if pointer else None
    except (AttributeError, RuntimeError):
        return (str(tensor.device), int(tensor.data_ptr()), -1)


def _unflatten_outputs(program: Any, indexed: Any, values: tuple[Any, ...]) -> Any:
    by_id = {item.value_id: bound for item, bound in zip(indexed.values, values)}
    leaves = [by_id[item.value_id] if item.value_id is not None else item.literal
              for item in indexed.outputs]
    spec = program.output_tree_spec
    if spec is not None and hasattr(spec, "num_leaves"):
        try:
            import torch.utils._pytree as pytree
            return pytree.tree_unflatten(leaves, spec)
        except (ImportError, TypeError, ValueError) as exc:
            raise CachedStepSessionError("captured output TreeSpec cannot be reconstructed") from exc

    def container(next_path: Any) -> Any:
        return [] if isinstance(next_path, int) else {}

    result: Any = None
    for output, leaf in zip(indexed.outputs, leaves):
        if not output.path:
            result = leaf
            continue
        if result is None:
            result = container(output.path[0])
        cursor = result
        for index, component in enumerate(output.path[:-1]):
            next_component = output.path[index + 1]
            if isinstance(cursor, list):
                while len(cursor) <= component:
                    cursor.append(None)
                if cursor[component] is None:
                    cursor[component] = container(next_component)
                cursor = cursor[component]
            else:
                if component not in cursor:
                    cursor[component] = container(next_component)
                cursor = cursor[component]
        final = output.path[-1]
        if isinstance(cursor, list):
            while len(cursor) <= final:
                cursor.append(None)
            cursor[final] = leaf
        else:
            cursor[final] = leaf
    return result


def validate_cached_step_inputs(step_abi: Any, inputs: Mapping[str, Any]) -> None:
    """Check caller-visible tensor guards before allocating outputs or launching."""
    import torch

    ordered = tuple(item["placeholder"] for item in step_abi.ordered_user_inputs)
    if set(inputs) != set(ordered):
        raise CachedStepSessionError(
            f"cached-step inputs must be exactly {ordered!r}, got {tuple(inputs)!r}"
        )
    guards = step_abi.guard_set
    for field, actual in (("shapes", "shape"), ("strides", "stride"),
                          ("dtypes", "dtype")):
        for name, expected in guards.get(field, {}).items():
            if name not in inputs:
                continue  # Some StepABI guards describe internal values.
            value = inputs.get(name)
            if not isinstance(value, torch.Tensor):
                raise CachedStepSessionError(f"{name} must be a tensor")
            if actual == "shape":
                observed = tuple(value.shape)
                wanted = tuple(expected)
            elif actual == "stride":
                observed = tuple(value.stride())
                wanted = tuple(expected)
            else:
                observed = str(value.dtype).removeprefix("torch.")
                wanted = str(expected).removeprefix("torch.")
            if observed != wanted:
                raise CachedStepSessionError(
                    f"{name} {actual} guard failed: expected {wanted!r}, got {observed!r}"
                )
    for state in step_abi.old_state_inputs:
        axes = tuple(part.strip() for part in state["layout"].split(","))
        if "capacity" not in axes:
            raise CachedStepSessionError(f"{state['placeholder']} layout has no capacity axis")
        capacity_axis = axes.index("capacity")
        value = inputs[state["placeholder"]]
        if capacity_axis >= value.ndim:
            raise CachedStepSessionError(f"{state['placeholder']} layout rank exceeds its tensor rank")
        expected = state["capacity"]
        if value.shape[capacity_axis] != expected:
            raise CachedStepSessionError(
                f"{state['placeholder']} capacity guard failed: expected {expected}, "
                f"got {value.shape[capacity_axis]}"
            )


class CachedStepSession:
    """Bind stable lifted values and invoke one exact, admitted cooperative entry.

    ``run`` accepts the ordered user tensors (or their exact placeholder names),
    allocates fresh graph values for each call, and returns ``(outputs, state,
    record)``. It never runs a reference graph or changes to another backend.
    """

    def __init__(self, library_path: str | Path, program: Any, indexed: Any,
                 logical_plan: Any, worker: WorkerProgram, profile: CudaTargetProfile,
                 admission: Mapping[str, Any]):
        abi = program.step_abi
        if abi is None:
            raise CachedStepSessionError("cached-step session requires a captured StepABI")
        if not indexed.strict_supported:
            raise CachedStepSessionError("cached-step session requires strict-supported indexed semantics")
        indexed_abi = indexed.source_program.step_abi
        if (indexed_abi is None or indexed_abi.contract_hash != abi.contract_hash):
            raise CachedStepSessionError("session StepABI differs from the ABI used to build indexed semantics")
        if graph_hash(indexed.source_program) != graph_hash(program):
            raise CachedStepSessionError("indexed program belongs to a different captured program")
        if worker.indexed_program_hash != indexed.structural_hash:
            raise CachedStepSessionError("worker program belongs to different indexed semantics")
        if worker.logical_plan_hash != logical_plan.structural_hash:
            raise CachedStepSessionError("worker program belongs to a different logical plan")

        bindings = program.lifted_bindings
        if set(bindings) != set(abi.lifted_bindings):
            raise CachedStepSessionError("captured lifted bindings differ from the StepABI")
        for placeholder, expected in abi.lifted_bindings.items():
            binding = bindings[placeholder]
            if binding.target != expected["identity"]:
                raise CachedStepSessionError(f"lifted binding {placeholder!r} differs from the StepABI")

        declared_outputs = {tuple(item["path"]): item for item in abi.user_output_tree["leaves"]}
        actual_outputs = {tuple(item.path): item for item in indexed.outputs}
        if set(declared_outputs) != set(actual_outputs):
            raise CachedStepSessionError("captured output leaves differ from the StepABI")
        values = {item.value_id: item for item in indexed.values}
        for path, leaf in actual_outputs.items():
            contract = declared_outputs[path]
            if contract["ownership"] != "owned" or leaf.value_id is None:
                raise CachedStepSessionError(f"output at {path!r} is not a tensor owned by the caller")
            if not _has_owned_storage_root(leaf.value_id, values):
                raise CachedStepSessionError(f"output at {path!r} has no proved fresh storage root")
        for state in abi.new_state_outputs:
            if tuple(state["path"]) not in actual_outputs:
                raise CachedStepSessionError(f"new state {state['state_id']!r} has no captured output")

        self.program = program
        self.indexed = indexed
        self.logical_plan = logical_plan
        self.worker = worker
        self.profile = profile
        self.entry: WorkerEntrySession = open_worker_session(
            library_path, worker, profile, admission
        )

    def _bind_user_inputs(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        records = self.program.step_abi.ordered_user_inputs
        names = tuple(item["placeholder"] for item in records)
        if args and kwargs:
            raise CachedStepSessionError("pass cached-step inputs positionally or by name, not both")
        if kwargs:
            if set(kwargs) != set(names):
                raise CachedStepSessionError(f"cached-step keyword inputs must be exactly {names!r}")
            return dict(kwargs)
        if len(args) != len(names):
            raise CachedStepSessionError(f"cached-step expects {len(names)} user inputs, got {len(args)}")
        return dict(zip(names, args))

    def run(self, *args: Any, **kwargs: Any) -> tuple[Any, Mapping[str, Any] | None, ExecutionRecord]:
        inputs = self._bind_user_inputs(args, kwargs)
        validate_cached_step_inputs(self.program.step_abi, inputs)

        placeholders = [node.name for node in self.program.graph_module.graph.nodes
                        if node.op == "placeholder"]
        by_name = dict(inputs)
        for name in self.program.lifted_bindings:
            by_name[name] = self.program.binding_for(name)
        if set(by_name) != set(placeholders):
            raise CachedStepSessionError("captured placeholder set differs from the user and lifted bindings")
        ordered_args = tuple(by_name[name] for name in placeholders)

        binding_start = time.perf_counter_ns()
        values = bind_worker_values(self.indexed, ordered_args)
        binding_host_ns = time.perf_counter_ns() - binding_start
        launch_start = time.perf_counter_ns()
        launch = self.entry.launch(values)
        launch_host_ns = time.perf_counter_ns() - launch_start
        record = ExecutionRecord(
            self.worker.program_hash, self.profile.profile_key,
            self.entry.runtime_target, self.worker.grid_ctas,
            binding_host_ns, launch_host_ns, bool(launch["launched"]),
            int(launch["return_code"]), launch.get("reason"),
        )
        if not launch["launched"]:
            return None, None, record

        outputs = _unflatten_outputs(self.program, self.indexed, values)
        state = {
            item["state_id"]: _at_path(outputs, item["path"])
            for item in self.program.step_abi.new_state_outputs
        }
        input_storages = {_storage_key(value) for value in by_name.values()
                          if hasattr(value, "data_ptr")}
        input_storages.discard(None)
        for path, leaf in {tuple(item.path): item for item in self.indexed.outputs}.items():
            tensor = _at_path(outputs, path)
            storage = _storage_key(tensor)
            if storage is not None and storage in input_storages:
                raise WorkerProgramError(f"output at {path!r} aliases an invocation input")
        return outputs, state, record


def open_cached_step_session(library_path: str | Path, program: Any, indexed: Any,
                             logical_plan: Any, worker: WorkerProgram,
                             profile: CudaTargetProfile,
                             admission: Mapping[str, Any]) -> CachedStepSession:
    """Open a strict session; mismatched target/program bindings are rejected."""
    return CachedStepSession(library_path, program, indexed, logical_plan,
                            worker, profile, admission)


__all__ = [
    "CachedStepSession",
    "CachedStepSessionError",
    "ExecutionRecord",
    "open_cached_step_session",
    "validate_cached_step_inputs",
]
