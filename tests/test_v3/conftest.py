"""Independent V3 test lanes with lazy optional-environment checks."""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import pytest


DEFAULT_V3_SEED = 1729


def lane_skip_reason(lane: str) -> str | None:
    """Return a reason for an unavailable optional lane, without raising."""

    lane = lane.strip().lower()
    if lane == "cpu":
        return None
    if lane == "toolchain":
        if os.environ.get("V3_FORCE_TOOLCHAIN_RAISE"):
            return "toolchain discovery forced to fail by V3_FORCE_TOOLCHAIN_RAISE"
        selected = os.environ.get("V3_NVCC")
        if selected:
            path = Path(selected)
            if path.is_file() and os.access(path, os.X_OK):
                return None
            return f"selected V3_NVCC is not an executable file: {selected}"
        if shutil.which("nvcc") is None:
            return "nvcc was not found on PATH"
        return None
    if lane == "gpu":
        if os.environ.get("V3_FORCE_DEVICE_RAISE"):
            return "device discovery forced to fail by V3_FORCE_DEVICE_RAISE"
        try:
            import torch

            if not torch.cuda.is_available():
                return "CUDA is unavailable"
        except Exception as exc:  # discovery is a skip, not a collection error
            return f"CUDA device discovery failed: {type(exc).__name__}: {exc}"
        return None
    raise ValueError(f"unknown V3 lane {lane!r}; expected cpu, toolchain, or gpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Setup happens after collection. Optional checks therefore cannot make
    # ordinary CPU collection initialize CUDA or load model weights.
    if item.get_closest_marker("v3_gpu") is not None:
        reason = lane_skip_reason("gpu")
        if reason:
            pytest.skip(reason)
    if item.get_closest_marker("v3_toolchain") is not None:
        reason = lane_skip_reason("toolchain")
        if reason:
            pytest.skip(reason)


@pytest.fixture(scope="session")
def v3_seed() -> int:
    seed = int(os.environ.get("V3_TEST_SEED", DEFAULT_V3_SEED))
    _seed_everything(seed)
    return seed


@pytest.fixture(scope="session")
def v3_device() -> Any:
    reason = lane_skip_reason("gpu")
    if reason:
        pytest.skip(reason)
    import torch

    try:
        torch.cuda.get_device_properties(0)
    except Exception as exc:
        pytest.skip(f"selected CUDA device is unusable: {exc}")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def v3_toolchain() -> str:
    reason = lane_skip_reason("toolchain")
    if reason:
        pytest.skip(reason)
    path = os.environ.get("V3_NVCC") or shutil.which("nvcc")
    assert path is not None
    return path


selected_device = v3_device
selected_toolchain = v3_toolchain
