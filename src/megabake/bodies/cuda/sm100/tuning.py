"""Read optional, device-specific measurements for SM100 GEMM shapes."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


@lru_cache(maxsize=8)
def _device_details(device_index: int) -> dict:
    import torch
    from cuda.bindings import driver

    properties = torch.cuda.get_device_properties(device_index)
    root = Path(__file__).resolve().parents[3]
    repository = root.parents[1]
    files = (Path(__file__).with_name("tcgen_gemm.py"),
             Path(__file__).with_name("gemm.py"),
             root / "bodies/cuda/sm90/gemm.py", root / "bodies/cuda/sm90/attention.py",
             *(root / name for name in ("codegen.py", "compiler.py", "plan.py", "planner.py",
                                        "regions.py", "schedule.py")),
             repository / "benchmarks/full_model.py", repository / "benchmarks/stateful_models.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    try:
        cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        cutlass_version = None
    status, driver_version = driver.cuDriverGetVersion()
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": cutlass_version,
        "driver": int(driver_version) if int(status) == 0 else None,
        "implementation_sha256": digest.hexdigest(),
    }


def device_fingerprint(device_index: int) -> dict:
    from ..sm90.attention import DECODE_SECOND_WAVE_KEYS, _prefill_value_block

    token_setting = os.environ.get("MEGABAKE_SM100_TOKEN_TILE", "8")
    return {**_device_details(device_index),
            "fused_epilogues": os.environ.get("MEGABAKE_SM100_EPILOGUE") == "1",
            "decode_second_wave_keys": DECODE_SECOND_WAVE_KEYS,
            "token_tile": int(token_setting) if token_setting in {"8", "16", "32"} else 8,
            "prefill_value_block": _prefill_value_block()}


@lru_cache(maxsize=16)
def _read(path: str, mtime_ns: int, size: int) -> dict:
    del mtime_ns, size
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("SM100 tuning file must use schema 1")
    if not isinstance(data.get("choices"), dict):
        raise ValueError("SM100 tuning file has no choices mapping")
    return data


def choice_for_shape(target, rows: int, columns: int, reduction: int):
    path = os.environ.get("MEGABAKE_SM100_TUNING_FILE")
    if not path:
        return None
    import torch

    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    data = _read(str(resolved), stat.st_mtime_ns, stat.st_size)
    fingerprint = device_fingerprint(torch.cuda.current_device())
    if data.get("fingerprint") != fingerprint:
        raise ValueError(f"SM100 tuning file {resolved} does not match this device or implementation")
    if (target.architecture != "sm100"
            or target.multiprocessor_count != fingerprint["multiprocessor_count"]):
        raise ValueError("SM100 tuning file does not match the compilation target")
    choice = data["choices"].get(f"{rows}:{columns}:{reduction}")
    if choice is None:
        return None
    if (not isinstance(choice, dict) or choice.get("tile") not in (64, 128)
            or choice.get("splits") not in (1, 2, 4, 8)):
        raise ValueError(f"invalid SM100 tuning choice for {(rows, columns, reduction)}")
    tile, splits = choice["tile"], choice["splits"]
    if columns % tile or reduction % (64 * splits):
        raise ValueError(f"incompatible SM100 tuning choice for {(rows, columns, reduction)}")
    return tile, splits
