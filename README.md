<p align="center">
  <img src="assets/megabake.png" width="200" />
</p>

# MegaBake

MegaBake's north star is to turn a captured PyTorch inference graph into planned, composable CUDA execution. PyTorch and Hugging Face stay the model, parameter, and runtime interface; MegaBake owns semantic recovery, fusion decisions, device-body selection, and execution planning.

## Architecture

1. TorchDynamo captures a model invocation as an FX graph and provides its specialization guards.
2. TorchInductor runs its post-grad FX transformations, including normalization and functionalization.
3. MegaBake takes the still-FX graph and its shape, layout, and guard metadata at a version-pinned handoff before Inductor lowering and scheduling. This integration uses Inductor internals, which are not a stable public API.
4. MegaBake imports generic tensor operations, canonicalizes equivalent forms, and recognizes parameterized computations such as RMSNorm, RoPE, attention, and MLP branches while retaining their source subgraphs.
5. MegaBake proposes fusions from explicit index maps and device-body capabilities, then checks legality, implementation feasibility, and profitability. Its first scheduler handles CTA-local fusion. The downstream execution plan specifies CuTe bodies or fallback, launches, tiling, storage lifetimes, and synchronization; persistence is a later option where it improves measured workloads.

A recognized composite describes computation; a fusion candidate proposes grouping; a kernel plan specifies one GPU launch. The initial planner combines work only when one CTA can own each producer-consumer tile and a composable CuTe device body fits the resource budget. A separately launched wrapper is not composable just because it was generated with CuTe DSL.

The captured graph remains the semantic reference. Correctness includes graph guards, outputs, mutations, state updates, aliasing, and numerical behavior. Performance is measured end to end against an equivalent `torch.compile` baseline using the same model, inputs, precision, and GPU. Claims apply only to measured workloads and GPU targets.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current architecture and build order. [north-star.md](north-star.md) records the earlier brief.

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
