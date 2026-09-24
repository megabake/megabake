"""Compile CUDA sources into a cubin and launch the megakernel."""

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

from megabake.runtime import get_sm_version

_CUDA_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "cuda"
_COMPILED_CACHE: dict[str, str] = {}


def _resolve_cutlass_include() -> Path:
    """Find a complete CUTLASS/CuTe header tree for nvcc.

    PyTorch binary wheels deliberately do not ship PyTorch's source-tree
    dependencies. Prefer an explicit override, then the pinned
    ``nvidia-cutlass`` package installed in the active virtual environment,
    before retaining the legacy PyTorch-source location for developers who
    build PyTorch from source.
    """
    candidates: list[Path] = []

    if cutlass_path := os.environ.get("CUTLASS_PATH"):
        candidates.append(Path(cutlass_path) / "include")

    cutlass_spec = importlib.util.find_spec("cutlass_library")
    if cutlass_spec and cutlass_spec.submodule_search_locations:
        package_dir = Path(next(iter(cutlass_spec.submodule_search_locations)))
        candidates.append(package_dir / "source" / "include")

    candidates.extend((
        Path("/home/devuser/pytorch/third_party/cutlass/include"),
    ))

    for include_dir in candidates:
        if (include_dir / "cute" / "tensor.hpp").is_file():
            return include_dir

    checked = "\n  ".join(str(path) for path in candidates)
    raise RuntimeError(
        "CUTLASS/CuTe headers were not found. Install the project's "
        "requirements, or set CUTLASS_PATH to a CUTLASS checkout containing "
        "include/cute/tensor.hpp. Checked:\n  " + checked
    )



def _compile_megakernel(force: bool = False, portable: bool = False) -> str:
    """Compile all CUDA sources into a cubin/fatbin.

    Args:
        force: Recompile even if cached.
        portable: If True, produce a fatbin with SM80 + SM90a cubins and a
            compute_80 PTX fallback.  If False (default), produce a single
            cubin for the current GPU — faster compilation for development.
    """
    cache_key = "megakernel" + ("_portable" if portable else "")
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

    # Resolve the optional headers only for an actual compile request.
    cutlass_include = _resolve_cutlass_include()

    sm = get_sm_version()

    if portable:
        out_path = str(build_dir / "megakernel.fatbin")
        cmd = [
            "nvcc",
            str(combined_src),
            "-fatbin",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-gencode=arch=compute_80,code=compute_80",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            f"-I{cutlass_include}",
            "-o", out_path,
            "-diag-suppress=177",
        ]
    else:
        out_path = str(build_dir / f"megakernel_sm{sm}.cubin")
        arch = f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"
        cmd = [
            "nvcc",
            str(combined_src),
            "-cubin",
            f"-arch={arch}",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            f"-I{cutlass_include}",
            "-o", out_path,
            "-diag-suppress=177",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300 if portable else 120)
    if result.returncode != 0:
        raise RuntimeError(
            f"CUDA compilation failed:\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    _COMPILED_CACHE[cache_key] = out_path
    return out_path


def get_megakernel_cubin(portable: bool = False) -> str:
    """Get path to compiled megakernel cubin/fatbin, compiling if needed."""
    return _compile_megakernel(portable=portable)
