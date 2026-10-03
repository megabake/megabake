<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# MegaBake

MegaBake's north star is to turn a captured PyTorch inference graph into a planned, composable CUDA megakernel. PyTorch and Hugging Face stay the model, parameter, and runtime interface; MegaBake owns FX-region fusion, device-body selection, scheduling, and execution planning.

## Architecture

1. TorchDynamo captures a model invocation as an FX graph and provides its specialization guards.
2. TorchInductor runs its post-grad FX transformations, including normalization and functionalization.
3. MegaBake takes the still-FX graph and its shape, layout, and guard metadata at a version-pinned handoff before Inductor lowering and scheduling. This integration uses Inductor internals, which are not a stable public API.
4. MegaBake pattern-matches and fuses FX regions, recording inputs, outputs, dependencies, state effects, layouts, and numerical requirements.
5. Supported regions map to device-callable CuTe DSL bodies. MegaBake plans tiling, CTA work assignment, storage lifetimes, synchronization, and stage order, then composes compatible bodies into one persistent CUDA kernel where legal.

An FX region is a description of tensor work, not a GPU kernel. It can join a megakernel only if its implementation exposes device-side work that can run inside the enclosing kernel and satisfies that kernel's ownership, synchronization, and resource constraints. A separately launched wrapper is not composable just because it was generated with CuTe DSL.

The captured graph remains the semantic reference. Correctness includes graph guards, outputs, mutations, state updates, aliasing, and numerical behavior. Performance is measured end to end against an equivalent `torch.compile` baseline using the same model, inputs, precision, and GPU. Claims apply only to measured workloads and GPU targets.

See [north-star.md](north-star.md) for the fuller brief.

## Environment setup

The checked-in requirements target Python 3.13, PyTorch with CUDA 13.0, and CuTe DSL's CUDA 13 package. Use an NVIDIA GPU and driver supported by that PyTorch wheel. Kernel compilation needs the CUDA 13.0 development toolkit and its `nvcc`; installing Python requirements does not install `nvcc`. The CUDA version reported by `nvidia-smi` describes driver support, not the installed compiler toolkit.

From the repository root, create and activate a virtual environment, then install the project and its pinned direct dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Point `CUDA_HOME` at the CUDA 13.0 toolkit before compiling kernels, especially if `/usr/local/cuda` selects another version:

```bash
export CUDA_HOME=/path/to/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
```

Check the selected Python, PyTorch CUDA build, GPU visibility, and CUDA compiler:

```bash
python --version
python -c 'import torch; print("torch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("GPU available:", torch.cuda.is_available())'
nvcc --version  # should report release 13.0
```

Activate the environment again in new shells with `source .venv/bin/activate`.
