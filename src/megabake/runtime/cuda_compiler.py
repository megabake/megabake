"""Compile CUDA sources into a cubin and launch the megakernel."""

import os
import subprocess
import tempfile
from pathlib import Path

import torch

_CUDA_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "cuda"
_COMPILED_CACHE: dict[str, str] = {}


def _get_sm_version() -> int:
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor


def _compile_megakernel(force: bool = False) -> str:
    """Compile all CUDA sources into a cubin. Returns path to .cubin."""
    cache_key = "megakernel"
    if not force and cache_key in _COMPILED_CACHE:
        so_path = _COMPILED_CACHE[cache_key]
        if os.path.exists(so_path):
            return so_path

    src_dir = _CUDA_SRC_DIR
    task_files = sorted((src_dir / "tasks").glob("*.cu"))

    build_dir = Path(tempfile.mkdtemp(prefix="megabake_build_"))

    # Concatenate all sources into one file to avoid rdc/linking complexity.
    # The megakernel.cu #includes nothing from tasks/ -- all task functions
    # are forward-declared and defined in their own .cu files. By concatenating,
    # the compiler sees everything in one translation unit.
    combined_src = build_dir / "combined.cu"
    parts = []

    # First: data_types.cuh content (via include)
    # Then: all task implementations
    # Finally: megakernel.cu (scheduler + dispatch)

    for task_file in task_files:
        content = task_file.read_text()
        # Remove the #include "../data_types.cuh" since we'll include it once
        content = content.replace('#include "../data_types.cuh"', '')
        parts.append(f"// === {task_file.name} ===\n{content}\n")

    megakernel_content = (src_dir / "megakernel.cu").read_text()
    # Remove includes of data_types since we include it at the top
    megakernel_content = megakernel_content.replace('#include "data_types.cuh"', '')
    # Remove forward declarations since the functions are already defined above
    lines = megakernel_content.split('\n')
    filtered = []
    skip_fwd_decl = False
    for line in lines:
        # Skip forward declarations of task functions
        if line.strip().startswith('__device__ void task_') and line.strip().endswith(';'):
            continue
        filtered.append(line)
    megakernel_content = '\n'.join(filtered)

    combined = f'#include <cooperative_groups.h>\n#include <cuda_fp16.h>\n'
    # Inline data_types.cuh
    combined += (src_dir / "data_types.cuh").read_text() + "\n"
    combined += '\n'.join(parts)
    combined += f"\n// === megakernel.cu ===\n{megakernel_content}\n"

    combined_src.write_text(combined)

    sm = _get_sm_version()
    cubin_path = str(build_dir / f"megakernel_sm{sm}.cubin")

    cmd = [
        "nvcc",
        str(combined_src),
        "-cubin",
        f"-arch=sm_{sm}",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-o", cubin_path,
        "-diag-suppress=177",  # suppress unused variable warnings
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"CUDA compilation failed:\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    _COMPILED_CACHE[cache_key] = cubin_path
    return cubin_path


def get_megakernel_cubin() -> str:
    """Get path to compiled megakernel cubin, compiling if needed."""
    return _compile_megakernel()
