"""Target-qualified CUDA facts; importing this module never queries a device."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ...contracts import ContractError, canonical_json


PROFILE_SCHEMA_VERSION = 1
_ARCH_TARGETS = {(9, 0): "sm_90a", (10, 0): "sm_100a"}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ProfileFact:
    value: Any
    source: str
    unknown_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ContractError("profile fact requires a source")
        encoded = canonical_json(self.value)
        object.__setattr__(self, "value", _freeze(json.loads(encoded)))
        if self.value is None and (not isinstance(self.unknown_reason, str)
                                   or not self.unknown_reason.strip()):
            raise ContractError("unknown profile facts require a reason")
        if self.value is not None and self.unknown_reason is not None:
            raise ContractError("known profile facts cannot have an unknown reason")

    def to_dict(self) -> dict[str, Any]:
        return {"value": json.loads(canonical_json(self.value)), "source": self.source,
                "unknown_reason": self.unknown_reason}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileFact":
        return cls(value["value"], value["source"], value.get("unknown_reason"))


def _facts(value: Mapping[str, ProfileFact | Mapping[str, Any]], name: str) -> Mapping[str, ProfileFact]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ContractError(f"{name} keys must be non-empty strings")
        fact = item if isinstance(item, ProfileFact) else ProfileFact.from_dict(item)
        result[key] = fact
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class CudaTargetProfile:
    device_index: int
    device_name: str
    device_uuid: str | None
    device_facts: Mapping[str, ProfileFact | Mapping[str, Any]]
    resource_limits: Mapping[str, ProfileFact | Mapping[str, Any]]
    feature_attributes: Mapping[str, ProfileFact | Mapping[str, Any]]
    target_sets: Mapping[str, ProfileFact | Mapping[str, Any]]
    versions: Mapping[str, ProfileFact | Mapping[str, Any]]
    schema_version: int = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ContractError(f"unsupported CUDA profile schema {self.schema_version}")
        if not isinstance(self.device_index, int) or self.device_index < 0:
            raise ContractError("device_index must be a non-negative integer")
        if not isinstance(self.device_name, str) or not self.device_name.strip():
            raise ContractError("device_name must be non-empty")
        if self.device_uuid is not None and not isinstance(self.device_uuid, str):
            raise ContractError("device_uuid must be a string or null")
        for name in ("device_facts", "resource_limits", "feature_attributes",
                     "target_sets", "versions"):
            object.__setattr__(self, name, _facts(getattr(self, name), name))
        for key in ("compute_capability", "visible_sms", "visible_memory_bytes"):
            if key not in self.device_facts:
                raise ContractError(f"CUDA profile is missing device fact {key}")
        for key in ("baseline", "architecture_specific", "family_specific", "compiler_targets"):
            if key not in self.target_sets:
                raise ContractError(f"CUDA profile is missing target set {key}")

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        groups = ("device_facts", "resource_limits", "feature_attributes", "target_sets", "versions")
        return tuple(f"{group}.{key}" for group in groups
                     for key, fact in getattr(self, group).items() if fact.value is None)

    def supports_target(self, target: str) -> bool | None:
        if target.endswith("a"):
            group = "architecture_specific"
        elif target.endswith("f"):
            group = "family_specific"
        else:
            group = "baseline"
        legal, compiled = self.target_sets[group].value, self.target_sets["compiler_targets"].value
        if legal is None or compiled is None:
            return None
        return target in legal and target in compiled

    def fact(self, group: str, name: str) -> ProfileFact:
        return getattr(self, group)[name]

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": "cuda",
            "device_index": self.device_index,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "device_facts": {k: v.to_dict() for k, v in self.device_facts.items()},
            "resource_limits": {k: v.to_dict() for k, v in self.resource_limits.items()},
            "feature_attributes": {k: v.to_dict() for k, v in self.feature_attributes.items()},
            "target_sets": {k: v.to_dict() for k, v in self.target_sets.items()},
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "unknown_fields": list(self.unknown_fields),
        }

    @property
    def profile_key(self) -> str:
        return hashlib.sha256(canonical_json(self._payload()).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "profile_key": self.profile_key}

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CudaTargetProfile":
        if value.get("schema_version") != PROFILE_SCHEMA_VERSION or value.get("backend") != "cuda":
            raise ContractError("invalid CUDA target profile schema or backend")
        profile = cls(
            device_index=value["device_index"], device_name=value["device_name"],
            device_uuid=value.get("device_uuid"), device_facts=value["device_facts"],
            resource_limits=value["resource_limits"],
            feature_attributes=value["feature_attributes"], target_sets=value["target_sets"],
            versions=value["versions"], schema_version=value["schema_version"],
        )
        if tuple(value.get("unknown_fields", ())) != profile.unknown_fields:
            raise ContractError("CUDA profile unknown_fields do not match its facts")
        if value.get("profile_key") != profile.profile_key:
            raise ContractError("CUDA profile key does not match its contents")
        return profile

    @classmethod
    def from_json(cls, value: str) -> "CudaTargetProfile":
        try:
            return cls.from_dict(json.loads(value))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid CUDA target profile: {exc}") from exc


def _known(value: Any, source: str) -> ProfileFact:
    return ProfileFact(value, source)


def _unknown(source: str, reason: str) -> ProfileFact:
    return ProfileFact(None, source, reason)


def _version_from_cuda(value: int) -> str:
    return f"{value // 1000}.{(value % 1000) // 10}"


def _runtime_result(runtime: Any, name: str) -> tuple[Any | None, str | None]:
    function = getattr(runtime, name, None)
    if function is None:
        return None, f"{name} is not exposed by cuda.bindings.runtime"
    try:
        result = function()
        status, value = result
        if int(status) != 0:
            return None, f"{name} returned {status}"
        return value, None
    except Exception as exc:
        return None, f"{name} failed: {type(exc).__name__}: {exc}"


def _device_attribute(runtime: Any, name: str, device_index: int) -> tuple[int | None, str]:
    source = f"cuda.bindings.runtime.cudaDeviceGetAttribute(cudaDeviceAttr.{name}, {device_index})"
    attr = getattr(getattr(runtime, "cudaDeviceAttr", object()), name, None)
    if attr is None:
        return None, source + " (attribute unavailable in selected runtime bindings)"
    try:
        status, value = runtime.cudaDeviceGetAttribute(attr, device_index)
        if int(status) != 0:
            return None, source + f" (query returned {status})"
        return int(value), source
    except Exception as exc:
        return None, source + f" (query failed: {type(exc).__name__}: {exc})"


def _probe_arch_target(executable: str, target: str) -> tuple[bool | None, str]:
    source = ("__global__ void megabake_v3_profile_probe() {}\n")
    try:
        with tempfile.TemporaryDirectory(prefix="megabake-v3-profile-") as temp:
            source_path = Path(temp) / "probe.cu"
            output_path = Path(temp) / "probe.o"
            source_path.write_text(source)
            result = subprocess.run(
                [executable, "-std=c++17", f"-arch={target}", "-c", str(source_path),
                 "-o", str(output_path)],
                text=True, capture_output=True, check=False, timeout=30,
            )
        command = f"{executable} -std=c++17 -arch={target} -c <temporary probe.cu> (exit={result.returncode})"
        return result.returncode == 0, command
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{executable} -std=c++17 -arch={target} compile probe ({type(exc).__name__}: {exc})"


def _nvcc_facts(major: int, minor: int) -> tuple[str | None, str, list[str] | None, str, bool | None, str]:
    executable = shutil.which("nvcc")
    if executable is None:
        missing = "nvcc was not found on PATH"
        return None, "shutil.which('nvcc')", None, missing, None, missing
    version = subprocess.run([executable, "--version"], text=True, capture_output=True,
                             check=False, timeout=10)
    version_source = f"{executable} --version"
    banner = version.stdout + version.stderr
    match = re.search(r",\s*V([0-9.]+)", banner) or re.search(r"release\s+([0-9.]+)", banner)
    version_text = match.group(1) if version.returncode == 0 and match else None
    target_run = subprocess.run([executable, "--list-gpu-code"], text=True,
                                capture_output=True, check=False, timeout=10)
    target_source = f"{executable} --list-gpu-code"
    targets = sorted(set(re.findall(r"\bsm_[0-9]+[af]?\b", target_run.stdout)))
    if target_run.returncode != 0:
        return version_text, version_source, None, target_source, None, target_source + " failed"
    compiler_source = target_source
    arch_target = _ARCH_TARGETS.get((major, minor))
    arch_source = target_source
    arch_supported: bool | None = None
    if arch_target:
        if arch_target in targets:
            arch_supported, arch_source = True, target_source
        else:
            arch_supported, arch_source = _probe_arch_target(executable, arch_target)
            if arch_supported:
                targets.append(arch_target)
                targets.sort()
                compiler_source += "; " + arch_source
    return version_text, version_source, targets, compiler_source, arch_supported, arch_source


def _metadata_version(name: str) -> ProfileFact:
    try:
        return _known(importlib.metadata.version(name), f"importlib.metadata.version({name!r})")
    except importlib.metadata.PackageNotFoundError:
        return _unknown(f"importlib.metadata.version({name!r})", "distribution is not installed")


def _target_sets(major: int, minor: int, compiler_targets: list[str] | None,
                 compiler_source: str, architecture_supported: bool | None,
                 architecture_source: str) -> dict[str, ProfileFact]:
    baseline = f"sm_{major}{minor}"
    if compiler_targets is None:
        baseline_fact = _unknown("CUDA baseline target mapping plus nvcc --list-gpu-code",
                                 "compiler target list is unknown")
        architecture_fact = _unknown(architecture_source, "compiler target support is unknown")
    else:
        baseline_fact = _known([baseline] if baseline in compiler_targets else [],
                               f"compute capability {major}.{minor}; {compiler_source}")
        arch_target = _ARCH_TARGETS.get((major, minor))
        if arch_target is None:
            architecture_fact = _unknown("V3 exact architecture-specific target table",
                                          f"no reviewed _a target mapping for {(major, minor)}")
        elif architecture_supported is None:
            architecture_fact = _unknown(architecture_source,
                                         f"compiler support for exact target {arch_target} is unknown")
        else:
            architecture_fact = _known([arch_target] if architecture_supported else [],
                                       f"exact capability lookup {(major, minor)} in V3 _a target table; {architecture_source}")
    return {
        "baseline": baseline_fact,
        "architecture_specific": architecture_fact,
        "family_specific": _unknown("V3 CUDA _f compatibility table",
                                     "no reviewed family-specific mapping for this profile"),
        "compiler_targets": _known(compiler_targets, compiler_source) if compiler_targets is not None
        else _unknown(compiler_source, "nvcc did not return its supported code list"),
    }


def query_target_profile(device_index: int | None = None) -> CudaTargetProfile:
    """Query the selected CUDA device and installed toolchain, recording unknowns."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("query_target_profile requires the installed PyTorch CUDA runtime") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("query_target_profile requires an available CUDA device")
    index = torch.cuda.current_device() if device_index is None else int(device_index)
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not visible")
    props = torch.cuda.get_device_properties(index)
    version, nvcc_source, compiler_targets, compiler_target_source, arch_supported, arch_source = _nvcc_facts(
        int(props.major), int(props.minor))

    runtime = None
    try:
        from cuda.bindings import runtime as runtime
    except ImportError:
        pass

    device_facts = {
        "device_index": _known(index, "query_target_profile(device_index) or torch.cuda.current_device()"),
        "compute_capability": _known(f"{props.major}.{props.minor}",
                                      f"torch.cuda.get_device_properties({index}).major/minor"),
        "visible_sms": _known(int(props.multi_processor_count),
                               f"torch.cuda.get_device_properties({index}).multi_processor_count"),
        "visible_memory_bytes": _known(int(props.total_memory),
                                        f"torch.cuda.get_device_properties({index}).total_memory"),
        "device_name": _known(str(props.name), f"torch.cuda.get_device_properties({index}).name"),
        "device_uuid": _known(str(props.uuid), f"torch.cuda.get_device_properties({index}).uuid"),
        "pci_bus_id": _known(int(props.pci_bus_id), f"torch.cuda.get_device_properties({index}).pci_bus_id"),
    }
    try:
        device_uuid = str(props.uuid)
        if "0x" in device_uuid or "object at" in device_uuid:
            device_uuid = None
    except Exception:
        device_uuid = None

    property_limits = {
        "max_threads_per_block": "max_threads_per_block",
        "max_threads_per_multiprocessor": "max_threads_per_multi_processor",
        "registers_per_multiprocessor": "regs_per_multiprocessor",
        "shared_memory_per_block": "shared_memory_per_block",
        "shared_memory_per_block_optin": "shared_memory_per_block_optin",
        "shared_memory_per_multiprocessor": "shared_memory_per_multiprocessor",
        "warp_size": "warp_size",
    }
    resource_limits = {
        name: _known(int(getattr(props, prop)), f"torch.cuda.get_device_properties({index}).{prop}")
        for name, prop in property_limits.items() if hasattr(props, prop)
    }
    for name, attr in (
        ("max_threads_per_block", "cudaDevAttrMaxThreadsPerBlock"),
        ("max_threads_per_multiprocessor", "cudaDevAttrMaxThreadsPerMultiProcessor"),
        ("max_registers_per_block", "cudaDevAttrMaxRegistersPerBlock"),
        ("max_registers_per_multiprocessor", "cudaDevAttrMaxRegistersPerMultiprocessor"),
        ("shared_memory_per_block", "cudaDevAttrMaxSharedMemoryPerBlock"),
        ("shared_memory_per_block_optin", "cudaDevAttrMaxSharedMemoryPerBlockOptin"),
        ("shared_memory_per_multiprocessor", "cudaDevAttrMaxSharedMemoryPerMultiprocessor"),
        ("max_blocks_per_multiprocessor", "cudaDevAttrMaxBlocksPerMultiprocessor"),
    ):
        if runtime is None:
            if name not in resource_limits:
                resource_limits[name] = _unknown(attr, "cuda.bindings.runtime is not installed")
            continue
        value, source = _device_attribute(runtime, attr, index)
        if value is not None:
            resource_limits[name] = _known(value, source)
        elif name not in resource_limits:
            resource_limits[name] = _unknown(source, "selected runtime did not expose this resource")

    feature_attributes: dict[str, ProfileFact] = {}
    for name, attr in (("cooperative_launch", "cudaDevAttrCooperativeLaunch"),
                       ("cluster_launch", "cudaDevAttrClusterLaunch")):
        if runtime is None:
            feature_attributes[name] = _unknown(attr, "cuda.bindings.runtime is not installed")
            continue
        value, source = _device_attribute(runtime, attr, index)
        feature_attributes[name] = (_known(bool(value), source) if value is not None else
                                    _unknown(source, "selected runtime did not expose this feature"))

    runtime_value = driver_value = None
    runtime_reason = driver_reason = "cuda.bindings.runtime is not installed"
    if runtime is not None:
        runtime_value, runtime_reason = _runtime_result(runtime, "cudaRuntimeGetVersion")
        driver_value, driver_reason = _runtime_result(runtime, "cudaDriverGetVersion")
    versions = {
        "nvcc": _known(version, nvcc_source) if version else _unknown(nvcc_source, "nvcc version query failed"),
        "cuda_runtime": (_known(_version_from_cuda(runtime_value),
                                 "cuda.bindings.runtime.cudaRuntimeGetVersion")
                         if runtime_value is not None else
                         _unknown("cuda.bindings.runtime.cudaRuntimeGetVersion", runtime_reason or "unknown")),
        "cuda_driver": (_known(_version_from_cuda(driver_value),
                                "cuda.bindings.runtime.cudaDriverGetVersion")
                        if driver_value is not None else
                        _unknown("cuda.bindings.runtime.cudaDriverGetVersion", driver_reason or "unknown")),
        "torch": _known(str(torch.__version__), "torch.__version__"),
        "torch_cuda_build": (_known(str(torch.version.cuda), "torch.version.cuda")
                             if torch.version.cuda else
                             _unknown("torch.version.cuda", "PyTorch was not built with CUDA")),
        "cutlass": _metadata_version("nvidia-cutlass"),
        "cuda_bindings": _metadata_version("cuda-bindings"),
        "cublasdx": _unknown("optional V3R-008 provider discovery",
                              "cuBLASDx is not a registered installed provider"),
        "body_abi": _known("1", "MegaBake V3 CUDA body registry schema v1"),
    }
    return CudaTargetProfile(
        device_index=index, device_name=str(props.name), device_uuid=device_uuid,
        device_facts=device_facts, resource_limits=resource_limits,
        feature_attributes=feature_attributes,
        target_sets=_target_sets(int(props.major), int(props.minor), compiler_targets,
                                 compiler_target_source, arch_supported, arch_source),
        versions=versions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = query_target_profile(args.device).to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()


__all__ = ["CudaTargetProfile", "PROFILE_SCHEMA_VERSION", "ProfileFact",
           "query_target_profile"]
