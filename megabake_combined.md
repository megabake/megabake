
# Megabake: torch.export to Megakernel Pipeline

## 1. Problem Statement

Today, deploying an open model from HuggingFace for inference requires two separate steps: downloading the model weights, and compiling the compute kernels. Weights are solved -- safetensors provides a safe, fast, portable format that every framework supports. Compute is not solved -- every user independently runs torch.compile, waits minutes to hours, and ends up with the same kernels everyone else computes. This is redundant, wasteful, and a barrier to adoption.

Meanwhile, megakernel research (Hazy Research, Mirage MPK, Luminal, Together AI) has demonstrated 1.5-3.5x inference speedups by fusing an entire model forward pass into a single persistent GPU kernel. But these systems are standalone -- they have their own APIs, their own runtimes, and no distribution mechanism. Users cannot download a pre-compiled megakernel the way they download pre-trained weights.

Megabake bridges this gap: a torch.export-native megakernel pipeline that produces distributable artifacts, hosted on HuggingFace alongside safetensors weights. Users download both and run at megakernel speed immediately, with zero compilation.

## 2. Design Goals

1. **torch.export as input.** Any model that traces through torch.export is a valid input. No model porting, no new API to learn.
2. **AOTI-compatible output.** The artifact loads without Python, integrates with PyTorch's existing ahead-of-time infrastructure, and can be upstreamed into PyTorch core.
3. **Compile once, distribute everywhere.** A build service compiles megakernels for popular models and hardware targets. Users download, not compile.
4. **Safe format.** No pickle, no arbitrary code execution on the host. The artifact contains GPU machine code (cubin) and a data-only task schedule.
5. **Graceful fallback.** If no Megabake artifact matches the user's hardware, torch.compile runs as usual. Megabake is an acceleration layer, not a hard dependency.

## 3. Core Design Principles

**1. The goal is any torch model -> one megakernel.**
The primary objective is full fusion -- every op in the forward pass runs inside a single persistent kernel. The fallback gradient (graph splitting, separate launches) exists for correctness, but the design optimizes for the full-fusion path.

**2. BSP execution model (from cross-domain research).**
Cross-domain research (Graphcore IPU, CGRA compilers, dataflow architectures) showed that transformer inference is layer-sequential. A Bulk Synchronous Parallel model (all SMs execute same task -> barrier -> next task) is simpler than MPK-style event-driven scheduling and nearly as fast for our use case. This simplifies the scheduler from ~500 to ~50 lines of CUDA.

**3. Graceful degradation when full fusion isn't possible.**

```
Best:     Entire forward pass in one megakernel (all ops fused)
Good:     Most ops fused, 1-2 unsupported ops as separate launches
Okay:     Multiple megakernel segments with standard kernels between them
Baseline: Standard torch.compile (no megakernel at all)
```

The user always gets correct results. The question is how much speedup.

**4. Virtualized task interface (from microkernel/WASM research).**
Tasks receive virtual resource descriptors (input/output pointers, dimensions, tile index). They don't know they're inside a megakernel. This enables standalone testing of each task and clean composition.

**5. CUDA Device LTO for register optimization.**
Task implementations live in separate `.cu` files, compiled with `-dlto`. The compiler globally optimizes register allocation across all tasks. Combined with PERKS-style low occupancy (1 block/SM, max registers + SMEM), this addresses register pressure without manual tuning.

## 4. Architecture Overview

### Pipeline

A pipeline with four components:

```
[1] Schedule Compiler    [2] Kernel Runtime    [3] Runtime Loader    [4] Distribution
(Python)                 (CUDA)                (Python/C++)          (HF Kernels Hub)

torch.export(model)      Persistent kernel     Load cubin +          megabake-runtime
  -> FX graph             scheduler +           schedule +            on Kernels Hub
  -> pattern match        task dispatch +       weights               + per-model
  -> task schedule         task implementations  -> launch              schedules
```

### BSP Execution Model

A single persistent CUDA kernel, launched once via cooperative launch, occupies all SMs for the entire forward pass:

```
One CUDA __global__ function, launched once via cooperative launch, occupies all SMs:

  for each task in schedule:
    +-------------------------------------------+
    | ALL SMs execute the same task              |
    | (each SM works on a different tile)        |
    |                                           |
    |   SM 0: task(tile=0)                      |
    |   SM 1: task(tile=1)                      |
    |   SM 2: task(tile=2)                      |
    |   ...                                     |
    |   SM N: task(tile=N) or idle if N > tiles |
    +-------------------------------------------+
    |
    cooperative_groups::this_grid().sync()  // barrier
    |
    next task
```

No scheduler SMs needed. No work queues. No atomic task popping. No event granularity derivation. All SMs do useful compute. One cooperative_groups barrier between tasks.

**Upgrade path:** If benchmarks show the ~10-15% gap from losing inter-task pipelining matters, upgrade to MPK-style event-driven scheduling. Same task implementations (`__device__` functions), only the scheduler wrapper changes.

### Two-Part Artifact Design

The Megabake artifact contains two distinct components:

**1. Kernel binary (cubin) -- model-independent, compiled once per (hardware, dtype):**

The compiled megakernel containing the scheduler loop and all task implementations. A CUTLASS matmul handles arbitrary M/N/K via runtime parameters. A FlashAttention task handles arbitrary head counts and sequence lengths. The kernel binary does not change when the model changes -- only the task schedule does.

Expected kernel binaries (total, not per model):
- megabake-sm_90-fp16 (H100, FP16)
- megabake-sm_90-bf16 (H100, BF16)
- megabake-sm_90-fp8 (H100, FP8)
- megabake-sm_80-fp16 (A100, FP16)
- megabake-sm_80-bf16 (A100, BF16)
- megabake-sm_100-fp16 (Blackwell, FP16)
- megabake-sm_100-fp8 (Blackwell, FP8)

**2. Task schedule (binary data) -- model-specific, generated per model:**

A data structure describing what tasks to run, in what order, with what parameters and dependencies. Generated by walking the torch.export FX graph. No CUDA compilation required -- pure Python graph analysis.

Contents:
- Task list: operation type, dimensions, dependency barrier, trigger barrier
- Barrier configuration: producer counts per barrier
- Buffer layout: sizes, offsets, aliasing relationships
- Weight map: buffer indices to state_dict keys (for linking with safetensors)
- Shape bucket specification: which input shape ranges this schedule covers

### End-to-End Flow

```
Model author / build service:

  torch.export(model)
      |
      v
  ExportedProgram (FX graph + state_dict)
      |
      v
  Schedule Compiler (Python)
      |-- Graph analysis: pattern-match FX nodes to task types
      |-- Schedule generation: tile tasks, derive barriers, plan buffers
      |
      v
  .schedule artifact --> upload to HF model repo

End user:

  pip install megabake
  Download megabake-runtime (kernel binary) + schedule + safetensors (weights)
      |
      v
  Runtime loader
      |-- Load kernel binary (cubin) via cuModuleLoad
      |-- Load task schedule (binary data)
      |-- Map weight pointers from safetensors to buffer layout
      |
      v
  Run inference at megakernel speed, zero compilation
```

No pickle. No executable host code. Kernel binary runs in GPU's sandboxed execution environment.

## 5. Component Details

### 5.1 Schedule Compiler

**What:** Python tool that takes a torch.export ExportedProgram and produces a task schedule.

**Input:** ExportedProgram (FX graph + state_dict metadata)

**Output:** Binary task schedule file (.schedule)

**Pipeline phases:**

```
ExportedProgram
    |
    v
Phase 1: Graph Analysis
    Walk FX graph nodes
    Pattern match ATen ops to task types:
      aten.mm / aten.addmm / aten.linear  ->  OP_MATMUL
      aten.scaled_dot_product_attention    ->  OP_ATTENTION
      aten._native_batch_norm_legit        ->  OP_RMSNORM (after decomposition)
      aten.silu + aten.mul                 ->  OP_SILUMUL
      aten.embedding                       ->  OP_EMBEDDING
      ...
    Extract dimensions, strides, dtypes from node metadata
    Identify unsupported ops -> graph split points
    |
    v
Phase 2: Task Tiling
    For each task, determine:
      Grid dimensions (how many SM groups work on this task)
      Block dimensions
      Input/output tensor partitioning across SMs
    Strategy per task type:
      Matmul: tile M and N dimensions, reduce along K
      Attention: tile along batch and head dimensions
      Norms: tile along batch dimension, full reduction per row
    |
    v
Phase 3: Dependency Analysis & Barrier Derivation
    Build dependency graph from FX graph edges
    Compute GCD-based event granularity (adapted from MPK's annotated_graph.cc)
    Assign barrier IDs: each task gets a trigger_barrier and dep_barrier
    Determine num_triggers per barrier
    |
    v
Phase 4: Buffer Planning
    Static liveness analysis on task graph
    First-fit packing of non-overlapping buffers into arena (adapted from Luminal)
    Compute buffer aliasing (output of task A = input of task B -> same pointer)
    Map state_dict keys to buffer indices (for linking with safetensors at runtime)
    Compute total workspace bytes
    |
    v
Phase 5: Serialize
    Write task schedule:
      Task array: [op_type, dimensions, dep_barrier, trigger_barrier, buffer_indices]
      Barrier array: [num_triggers, first_consumer_task, last_consumer_task]
      Buffer layout: [offset, size, dtype] per buffer in the arena
      Weight map: [buffer_index -> state_dict_key]
      Launch config: grid dims, block dims, shared memory bytes
      Metadata: model_arch_hash, shape_bucket_range, compute_dtype, weight_dtype
```

**What we need to build:**
- FX graph walker + pattern matcher (~500 lines Python)
- Placement: assign task types to SM groups by resource needs (~200 lines Python)
- Scheduling: topological sort with placement constraints (~200 lines Python)
- Arena buffer planner, adapted from Luminal's approach (~300 lines Python)
- Schedule serializer (~200 lines Python)

**Total: ~1400 lines of Python.**

**Key design decisions:**
- **BSP scheduling (from cross-domain research).** No GCD-based barrier derivation needed. Tasks execute sequentially with grid-wide barriers between them. This eliminates ~400 lines of the most complex schedule compiler code (event granularity derivation).
- **Space-time decoupling (from CGRA compiler research).** Solve placement (which SMs run what) separately from scheduling (when). Simpler algorithm, faster compilation.
- Preserve SDPA as a single node (don't decompose, following Luminal's approach in pt2.py)
- Re-internalize lifted parameters (following Luminal's approach for torch.compile compatibility)
- Generate separate schedules for decode vs prefill (same kernel binary, different task ordering and tiling)
- Generate separate schedules per shape bucket (e.g., seq_len 1-128, 128-512, 512-2048)

### 5.2 Kernel Runtime

**What:** A single persistent CUDA kernel that executes any model's forward pass given a task schedule.

**Input at launch:** Task schedule (in GPU memory), buffer pointers, dynamic dimensions

**Output:** Model outputs written to designated buffers

**Task implementations (what's inside the switch):**

| Task Type | Implementation Source | Lines of CUDA | Difficulty |
|---|---|---|---|
| OP_MATMUL | CuTe DSL (Python -> CUDA) or CUTLASS CollectiveMainloop, using MPK's WGMMA patterns as reference | 200-500 | **Hard** |
| OP_ATTENTION | Hand-written using MPK's paged attention as reference, or FlashInfer primitives | 400-800 | **Hard** |
| OP_RMSNORM | Custom CUDA reduction | 50 | Easy |
| OP_LAYERNORM | Custom CUDA reduction | 50 | Easy |
| OP_SILUMUL | Custom CUDA elementwise | 20 | Trivial |
| OP_GELU | Custom CUDA elementwise | 20 | Trivial |
| OP_ROPE | Custom CUDA | 30 | Easy |
| OP_EMBEDDING | Custom CUDA lookup | 10 | Trivial |
| OP_RESIDUAL_ADD | Custom CUDA elementwise | 10 | Trivial |
| OP_SOFTMAX | Custom CUDA reduction | 40 | Easy |
| OP_ARGMAX | Custom CUDA reduction | 30 | Easy |
| OP_CAST | Custom CUDA | 10 | Trivial |

**What we need to build:**
- BSP scheduler loop with cooperative_groups barrier (~50 lines CUDA)
- Task dispatch switch (~50 lines CUDA)
- Matmul task via CuTe DSL or CUTLASS mid-level API (~200-500 lines)
- Attention task using MPK as reference (~400-800 lines)
- Simple tasks (norm, silu, rope, embedding, etc.) (~300 lines total)

**Total: ~1000-1700 lines of CUDA.**

**Register pressure mitigation:** Compile task implementations as separate `.cu` files, link with CUDA Device LTO (`-dlto`). The compiler globally optimizes register allocation across all tasks, avoiding worst-case-per-task allocation. Combined with PERKS-style low occupancy (1 block per SM, maximum registers + shared memory), this addresses the register pressure risk.

**The kernel binary is model-independent.** A matmul handles arbitrary M/N/K via runtime parameters in the task descriptor. The kernel binary changes only when:
- A new task type is added
- A new hardware target is supported
- A new dtype is supported

**How it's compiled:**
- Each task implementation in a separate `.cu` file
- Compiled with CUDA Device LTO (`-dlto`) for cross-task register optimization
- NVRTC or nvcc -> cubin (GPU machine code)
- Launched via `cudaLaunchCooperativeKernel` (required for grid-wide sync)
- OR: published as HF Kernels Hub repo, HF's kernel-builder Nix flake compiles for all platforms automatically

### 5.3 Runtime Loader

**What:** Python/C++ library that loads a kernel binary + task schedule + weights, and runs inference.

**API:**

```python
from megabake import load, bake

# === End user path (download pre-baked) ===
model = load(
    model_id="meta-llama/Llama-3-8B",
    hardware="auto",           # auto-detect SM arch
    dtype="fp16",
)
output = model(input_ids)

# === Model author path (bake and upload) ===
import torch
from transformers import AutoModelForCausalLM

pt_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
bake(
    pt_model,
    example_input=torch.randint(0, 32000, (1, 128)),
    hardware="sm_90",
    dtype="fp16",
    push_to_hub=True,
)
```

**What load() does under the hood:**

```
1. Download kernel binary
     From HF Kernels Hub: get_kernel("megabake/megabake-runtime")
     HF handles hardware matching (SM arch, CUDA version, PyTorch version)
     Cached after first download

2. Download task schedule
     From model repo: hf_hub_download("meta-llama/Llama-3-8B", "megabake/decode-sm_90-fp16.schedule")
     Picks the right schedule for user's hardware + dtype

3. Download weights
     safetensors: load_file("meta-llama/Llama-3-8B/model.safetensors")
     Standard, already works

4. Initialize
     Load cubin into GPU driver (cuModuleLoad)
     Load task schedule into GPU memory
     Allocate workspace arena (size from schedule metadata)
     Map safetensors weight pointers into the buffer layout
     using the weight map from the schedule

5. Run
     Launch the persistent megakernel (one launch)
     Kernel reads task schedule, executes all tasks, writes output
     Return output tensor to Python
```

**What bake() does under the hood:**

```
1. torch.export(model, example_input)
     Produces ExportedProgram with FX graph

2. Schedule compiler (Component 1)
     FX graph -> task schedule

3. Validate
     Run both original model and megakernel on test inputs
     Assert torch.allclose(original_output, megabake_output, atol=1e-5)

4. Upload
     Push schedule file(s) to model repo on HF
     If kernel binary not yet published, publish to Kernels Hub
```

**What we need to build:**
- load() function with HF download + hardware matching (~200 lines Python)
- cubin loader via CUDA driver API (~100 lines C++/Python)
- Task schedule parser (~100 lines Python)
- Weight pointer mapping from safetensors to buffer layout (~100 lines Python)
- Workspace arena allocator (~50 lines C++)
- Kernel launch wrapper (~50 lines C++/Python)
- bake() function wrapping Components 1 + validation (~200 lines Python)

**Total: ~800 lines Python/C++.**

### 5.4 Distribution

**What:** Infrastructure for publishing and discovering Megabake artifacts on HuggingFace.

**Architecture (leverages existing HF infrastructure):**

```
HF Kernels Hub:
  megabake/megabake-runtime              <- one repo, all models share this
    Built by HF kernel-builder for every platform
    Contains: persistent kernel scheduler + all task implementations
    User loads via: get_kernel("megabake/megabake-runtime")

HF Model Repos:
  meta-llama/Llama-3-8B/
    model.safetensors                    <- weights (already exists)
    config.json                          <- model config (already exists)
    megabake/
      index.json                         <- maps (hardware, dtype) -> schedule file
      decode-sm_90-fp16.schedule         <- task schedule for H100 FP16 decode
      decode-sm_90-fp8.schedule          <- task schedule for H100 FP8 decode
      decode-sm_80-fp16.schedule         <- task schedule for A100 FP16 decode
      prefill-sm_90-fp16.schedule        <- task schedule for H100 FP16 prefill
```

**index.json example:**

```json
{
  "megabake_version": "0.1.0",
  "runtime_kernel": "megabake/megabake-runtime@0.1.0",
  "schedules": {
    "sm_90": {
      "fp16": {
        "decode": "megabake/decode-sm_90-fp16.schedule",
        "prefill": "megabake/prefill-sm_90-fp16.schedule"
      },
      "fp8": {
        "decode": "megabake/decode-sm_90-fp8.schedule"
      }
    },
    "sm_80": {
      "fp16": {
        "decode": "megabake/decode-sm_80-fp16.schedule"
      }
    }
  },
  "workspace_bytes": {
    "sm_90-fp16-decode": 134217728,
    "sm_90-fp8-decode": 67108864
  }
}
```

**Build service (automated):**

```
For each new/updated model on HF:
  1. Read config.json -> determine architecture family
  2. If supported architecture:
     a. torch.export the model
     b. Run schedule compiler for each (hardware, dtype) target
     c. Validate numerics against reference
     d. Upload schedule files to model repo
     e. Update index.json
```

**What we need to build:**
- index.json schema definition (~50 lines)
- Schedule upload/download helpers wrapping huggingface_hub (~200 lines Python)
- Build service script for automated compilation (~500 lines Python)
- CI pipeline for the megabake-runtime Kernels Hub repo (~100 lines config)

**Total: ~850 lines Python + config.**

## 6. Task Type Coverage

### Generic Parameterized Task Types

Megabake uses a small number of generic, parameterized task types rather than one-per-op. This follows the "uber-shader" pattern from game engine rendering -- parameterized types avoid combinatorial task explosion.

| Task Type | Covers | Parameters |
|---|---|---|
| OP_MATMUL | mm, addmm, linear, grouped GEMM | M, N, K, dtype, batch_count |
| OP_ATTENTION | scaled_dot_product_attention, paged attention | num_heads, head_dim, seq_len, page_table_ptr |
| OP_ELEMENTWISE | silu*mul, gelu, relu, residual add, cast | op_code (enum), in/out dtypes |
| OP_REDUCE | rmsnorm, layernorm, softmax, argmax | reduce_type (enum), axis, epsilon |
| OP_EMBEDDING | embedding lookup | vocab_size, embed_dim |
| OP_INDEX | gather, scatter, index_select | index_mode (enum) |
| OP_COPY | contiguous, clone, slice | src/dst strides, offsets |
| OP_ROPE | rotary position embedding | max_seq_len, base_freq, head_dim |

### Shape Ops: Resolved at Compile Time

Many ATen ops are pure shape/stride manipulations: view, reshape, transpose, permute, expand, contiguous. These have zero runtime cost -- they are resolved during schedule compilation by adjusting the buffer layout's stride metadata. The task schedule references tensors by buffer index + stride descriptor; a "transpose" just swaps stride entries. No GPU work required.

### Task Implementation Sources

| Source | Task Types | Notes |
|---|---|---|
| CuTe DSL | OP_MATMUL | Write matmul in Python using CUTLASS abstractions. Generates WGMMA/TMA automatically. Supports cooperative kernel launch. Graduating summer 2026. |
| FlashInfer | OP_ATTENTION | More composable than FlashAttention. JIT compilation. Lower-level primitives adaptable for persistent kernel context. |
| Raw CUDA | OP_ELEMENTWISE, OP_REDUCE, OP_EMBEDDING, OP_INDEX, OP_COPY, OP_ROPE | Trivial tasks: ~10-50 lines each. |

Total task implementation code: ~1000-1700 lines of CUDA. The entire megakernel (scheduler + dispatch + all tasks) is under 2000 lines.

## 7. Solving Hard Problems

### Problem 1: Custom Ops

**The challenge:** Real HF models contain custom CUDA extensions, custom attention implementations, and ops that aren't in the standard ATen op set. torch.export may trace them as opaque `call_function` nodes or `higher_order_op` nodes.

**Design:**

```
FX graph analysis encounters an unknown op:
  |
  +--> Is it decomposable into known ATen ops?
  |      Yes --> decompose, pattern match the decomposition
  |      No  --> mark as UNSUPPORTED
  |
  v
UNSUPPORTED op triggers a graph split:

  [=== megakernel segment 1 ===]
  [custom_op]                     <-- launched as standard kernel
  [=== megakernel segment 2 ===]
```

**Graph splitting strategy:**

1. Walk the FX graph. Mark each node as SUPPORTED (maps to a task type) or UNSUPPORTED.
2. Find connected components of SUPPORTED nodes -- these become megakernel segments.
3. Each segment gets its own task schedule.
4. The runtime executes: megakernel segment 1 -> sync -> custom op -> sync -> megakernel segment 2.
5. Data between segments flows through HBM (standard PyTorch tensors).

**Optimization: minimize split cost.**
- If an UNSUPPORTED op is surrounded by lightweight ops (elementwise), absorb the surrounding ops into the adjacent megakernel segments rather than creating tiny segments.
- Merge adjacent segments separated by a single cheap UNSUPPORTED op -- the sync cost of splitting may exceed the cost of just running that op outside the megakernel.

**Custom op registration (Phase 2):**

```python
from megabake import register_task

@register_task("my_custom_attention")
def my_attention_task(cuda_source: str, shared_mem: int):
    return cuda_source  # __device__ function

# When the schedule compiler sees this op, it maps to the registered task
```

This is the Phase 2 composable task architecture -- tasks as plugins.

**Phase:** 1 (graph splitting), 2 (custom task registration)

### Problem 2: Exotic Model Architectures

#### MoE (Mixtral, DeepSeek)

**The challenge:** Mixture of Experts has dynamic routing -- a gating network selects which experts process each token. The number of tokens per expert varies at runtime. This is data-dependent control flow.

**Design:**

```
MoE forward pass:
  1. Gate: compute expert scores, top-k selection    [OP_GATE_TOPK]
  2. Permute: route tokens to selected experts       [OP_MOE_PERMUTE]
  3. Expert compute: N parallel FFNs                 [OP_MOE_GEMM]
  4. Unpermute: route results back                   [OP_MOE_UNPERMUTE]
```

- **Gate + TopK:** New task type. Simple softmax + topk selection. ~50 lines CUDA.
- **Permute/Unpermute:** Token routing based on gate output. Data-dependent but deterministic once the gate fires. New task type. ~80 lines CUDA.
- **Expert GEMMs:** Variable number of tokens per expert. Two approaches:
  - **Grouped GEMM:** CUTLASS has grouped GEMM support (ptr-array batched GEMM). Each expert is a separate GEMM in the group. MPK already has `fp8_group_gemm_sm100.cuh` for this. One task launch, all experts computed.
  - **Padded fixed-size:** Pad each expert's token count to the maximum, waste some compute but keep the schedule static. Simpler but slower.

**Recommended:** Grouped GEMM task type. CUTLASS 4.x has `CollectiveMma` specializations for ptr-array grouped GEMM (added for MoE). This maps naturally to a single task.

**Phase:** 2

#### Mamba / State Space Models

**The challenge:** SSMs have a different compute pattern -- selective scan instead of attention. Not a transformer, so the standard task type set doesn't apply.

**Design:** Defer. Mamba requires a new `OP_SELECTIVE_SCAN` task type with its own CUDA implementation (~200-300 lines). Add when there's demand. Until then, Mamba layers trigger graph splits and run as standard kernels.

**Phase:** 3+

#### Multimodal (LLaVA, etc.)

**The challenge:** Vision encoder (ViT) + language model (LLaMA) + projection layer. Different architectures concatenated.

**Design:** Each sub-model gets its own megakernel segment:

```
[=== vision encoder megakernel ===]    (ViT: attention + MLP + norms)
[projection layer]                      (standard linear, could be fused into either segment)
[=== language model megakernel ===]     (LLaMA: attention + MLP + norms)
```

The schedule compiler generates separate task schedules for the vision and language segments. The runtime chains them. No special handling needed -- graph splitting handles this naturally.

**Phase:** 1 (graph splitting handles naturally)

### Problem 3: Paged Attention & KV Cache

**The challenge:** Production inference uses paged KV cache for memory efficiency. Pages are allocated/freed dynamically as requests arrive and complete. The megakernel needs to access paged memory.

**Phased design:**

**Phase 1 MVP: KV cache as graph inputs.**
KV cache tensors are inputs to the megakernel, managed by PyTorch outside the kernel. Each decode step:
1. Python allocates/grows KV cache tensors
2. Passes cache pointers to the megakernel via the buffer layout
3. Megakernel reads/writes cache in the attention task
4. Python manages cache lifecycle

This is what Luminal does. It works but has overhead: KV cache pointers change every step, requiring buffer re-mapping.

**Phase 2: Paged attention task + on-GPU page management.**

Add two things:
1. **OP_PAGED_ATTENTION task type:** Attention kernel that reads K/V from a page table instead of contiguous tensors. MPK's `multitoken_paged_attention_hopper.cuh` is the reference. ~800 lines CUDA.

2. **On-GPU page allocator:** Circular page queue (from MPK's design). Allocated in the megakernel's workspace. Pages are allocated when KV grows, freed when requests complete.

**Page table structure:**

```
page_table: [max_requests, max_pages_per_request] -> page_id
page_pool:  [max_pages, page_size, num_kv_heads, head_dim] -> KV data
page_queue: circular queue of free page_ids
```

**Phase 3: Continuous batching (on-GPU batch scheduler).**
For serving integration (vLLM/SGLang), the megakernel handles request arrival/completion without returning to CPU:
- Pinned ring buffer for CPU -> GPU request submission (MPK's MODE_ONLINE_PINNED)
- On-GPU batch scheduler processes the ring buffer between iterations
- Kernel stays resident across all decode steps for all requests

### Problem 4: LoRA and QLoRA

**The challenge:** LoRA adds low-rank adaptation matrices to linear layers: `output = Wx + BAx` where `A` (down-project) and `B` (up-project) are the adapter weights. Different users apply different LoRA adapters to the same base model.

#### Option A: Separate schedule per adapter (recommended for MVP)

Each LoRA adapter changes the computation graph. The schedule compiler generates a new task schedule for each (base_model, adapter) pair:

```
Base model schedule:
  OP_MATMUL(x, W) -> output

LoRA schedule:
  OP_MATMUL(x, W) -> base_output      # same as base
  OP_MATMUL(x, A) -> down             # LoRA down-project (small matmul)
  OP_MATMUL(down, B) -> lora_output   # LoRA up-project (small matmul)
  OP_RESIDUAL_ADD(base_output, lora_output) -> output
```

The kernel binary is the same. Only the task schedule changes. LoRA adapters are small, so the extra schedule files are kilobytes.

**HF distribution:**

```
meta-llama/Llama-3-8B/
  megabake/
    decode-sm_90-fp16.schedule                    # base model
    decode-sm_90-fp16-lora-adapter-X.schedule     # with adapter X
    decode-sm_90-fp16-lora-adapter-Y.schedule     # with adapter Y
```

#### Option B: Dynamic adapter weight swapping (Phase 3)

For serving scenarios with many LoRA adapters (like S-LoRA), compile one schedule with LoRA branches, and swap adapter weights at runtime without recompiling the schedule:

```python
model = megabake.load("meta-llama/Llama-3-8B")
model.load_adapter("adapter-X")  # swaps A, B weight pointers in buffer layout
output = model(input_ids)
model.load_adapter("adapter-Y")  # swaps again, same schedule
output = model(input_ids)
```

This works because the task schedule references buffer indices, not weight values. Swapping adapter weights just changes which GPU pointers the buffer indices map to.

#### QLoRA

QLoRA = LoRA + quantized base weights (4-bit NF4 or FP4). The base model matmul becomes a dequantize + matmul:

```
QLoRA schedule:
  OP_DEQUANT(W_quant, scale, zero) -> W_fp16    # dequantize base weights
  OP_MATMUL(x, W_fp16) -> base_output
  OP_MATMUL(x, A) -> down                        # LoRA (always in fp16/bf16)
  OP_MATMUL(down, B) -> lora_output
  OP_RESIDUAL_ADD(base_output, lora_output) -> output
```

Requires an OP_DEQUANT task type (or fused DEQUANT_MATMUL). ~50-100 lines CUDA.

**Phase:** 1 (separate schedule), 3 (dynamic swapping, QLoRA)

### Problem 5: Quantization

**The challenge:** Different quant schemes (FP8, INT8, INT4, GPTQ, AWQ, NF4) require different compute kernels with different tensor core instructions.

**Level 1: Different kernel binaries per dtype family.**

```
megabake-sm_90-fp16.cubin     contains: matmul_fp16, attention_fp16
megabake-sm_90-fp8.cubin      contains: matmul_fp8, attention_fp8
megabake-sm_90-int8.cubin     contains: matmul_int8_w8a8, attention_fp16
megabake-sm_90-int4.cubin     contains: matmul_int4_w4a16, attention_fp16
```

Each uses different tensor core instructions (HMMA for FP16, QMMA for INT8, etc.). The task type enum encodes the dtype: `OP_MATMUL_FP16`, `OP_MATMUL_FP8`, `OP_MATMUL_INT4_W4A16`.

**Level 2: Quant-specific task types.**
- **FP8 dynamic quantization:** `OP_QUANTIZE_FP8` (compute per-tensor scale, quantize activations)
- **GPTQ/AWQ:** `OP_DEQUANT_MATMUL` (fused dequantize + matmul)
- **NF4 (QLoRA):** `OP_DEQUANT_NF4` (NormalFloat4 dequantization lookup table)

The schedule compiler reads the model's quantization config (from HF config.json or quantization_config.json) and selects the appropriate kernel binary and task types.

**Supported quant matrix (phased):**

| Scheme | Phase | Task types needed |
|---|---|---|
| FP16/BF16 | 1 (MVP) | OP_MATMUL_FP16 |
| FP8 (E4M3) | 2 | OP_MATMUL_FP8, OP_QUANTIZE_FP8 |
| INT8 (W8A8) | 2 | OP_MATMUL_INT8 |
| INT4 (W4A16, GPTQ/AWQ) | 3 | OP_DEQUANT_MATMUL_INT4 |
| NF4 (QLoRA) | 3 | OP_DEQUANT_NF4, OP_MATMUL_FP16 |

### Problem 6: Speculative Decoding

**The challenge:** Speculative decoding uses a small draft model to predict multiple tokens, then the target model verifies them in one pass.

**Design:**

```
Speculative decode iteration:
  1. Draft model generates K candidate tokens    [draft megakernel]
  2. Target model verifies all K in one pass     [target megakernel]
  3. Accept/reject tokens                        [OP_SPEC_VERIFY]
  4. Commit accepted tokens to KV cache          [OP_SPEC_COMMIT]
```

Two separate megakernels (draft and target), plus verify/commit task types. MPK's Eagle3 implementation is the reference -- they have `eagle3_commit_kernel` and `target_verify.cuh`.

**Phase:** 3

### Problem 7: Dynamic Shapes

**The challenge:** Input shapes (batch size, sequence length) change at runtime. The task schedule is compiled for specific shapes.

**Design: Shape buckets with fallback.**

```
Schedule compiler generates multiple schedules per model:

  decode-sm_90-fp16-seq1-128.schedule       batch=1, seq_len 1-128
  decode-sm_90-fp16-seq128-512.schedule     batch=1, seq_len 128-512
  decode-sm_90-fp16-seq512-2048.schedule    batch=1, seq_len 512-2048
  decode-sm_90-fp16-batch4-seq1-128.schedule  batch=1-4, seq_len 1-128
```

**Runtime selects the best matching bucket:**

```python
def select_schedule(batch_size, seq_len, available_schedules):
    # Find the tightest bucket that contains the input shape
    # If no bucket matches, fall back to torch.compile
    for schedule in available_schedules:
        if schedule.batch_range.contains(batch_size) and \
           schedule.seq_range.contains(seq_len):
            return schedule
    return None  # fallback to torch.compile
```

Within a bucket, dimensions are runtime parameters. The task descriptors carry concrete dimension values (M, N, K for matmul). CUTLASS handles variable dimensions via runtime dispatch. The schedule is valid for any shape within the bucket's range.

**Bucket boundary selection:** Autotune by profiling at multiple shapes within each range and selecting boundaries where the optimal tiling strategy changes.

**Phase:** 1

### Problem 8: Multi-GPU / Tensor Parallelism

**The challenge:** Models >8B parameters require tensor parallelism across multiple GPUs. The megakernel needs cross-GPU communication.

**Design (Phase 2):** Add OP_ALLREDUCE task type. Two implementation options:

**Option A: NVSHMEM (MPK's approach)**
- Allreduce via `nvshmem_putmem` + barrier + local reduce
- Requires NVSHMEM installation (non-trivial dependency)
- Works on NVLink-connected GPUs

**Option B: mKernel (Berkeley's approach)**
- GPU-initiated RDMA via libibverbs
- No NCCL or NVSHMEM dependency
- Works on NVLink and InfiniBand/EFA
- Hopper only currently

**Schedule impact:** For TP, the schedule compiler:
1. Reads the TP degree from model config
2. Shards weight buffer indices across ranks
3. Inserts OP_ALLREDUCE tasks after attention and MLP projections
4. Generates per-rank schedules (each rank has different weight mappings)

```
Per-rank schedule files:
  decode-sm_90-fp16-tp4-rank0.schedule
  decode-sm_90-fp16-tp4-rank1.schedule
  decode-sm_90-fp16-tp4-rank2.schedule
  decode-sm_90-fp16-tp4-rank3.schedule
```

**Phase:** 2

### Problem 9: Numerical Correctness & Debugging

**The challenge:** Megakernels are notoriously hard to debug. Wrong results are silent -- no intermediate tensors to inspect.

**Validation at bake time:**

```python
def bake(model, example_input, ...):
    # 1. Get reference output
    ref_output = model(example_input)

    # 2. Compile schedule
    schedule = compile_schedule(torch.export(model, example_input))

    # 3. Run megakernel
    mk_output = run_megakernel(schedule, model.state_dict(), example_input)

    # 4. Compare
    if not torch.allclose(ref_output, mk_output, atol=1e-5):
        # Bisect: run with each task type individually as separate launch
        # to isolate which task produces wrong results
        failing_task = bisect_tasks(schedule, model, example_input)
        raise MegabakeValidationError(f"Task {failing_task} produces incorrect output")
```

**Debug mode at runtime:**

```python
model = megabake.load("...", debug=True)
# In debug mode:
#   - Each task writes its output to a debug buffer
#   - After execution, compare each task's output against
#     the equivalent PyTorch op's output
#   - Report first divergence with task type, dimensions, max error
```

**Per-task unit tests:**

```python
def test_matmul_task():
    A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    ref = A @ B
    mk_output = run_single_task(OP_MATMUL, A, B)
    assert torch.allclose(ref, mk_output, atol=1e-3)
```

**NaN/Inf detection:** The runtime checks for NaN/Inf in the output after megakernel completion (borrowed from Luminal's `first_nonfinite_f32_buffer`). Catches silent corruption before it propagates.

**Phase:** 1

## 8. Cross-Domain Research

The megakernel problem is not unique to ML inference. Restated abstractly:

> Given a DAG of heterogeneous tasks and a fixed-topology parallel processor with limited per-unit resources, compile the DAG into a persistent execution plan that minimizes synchronization overhead while respecting resource constraints.

We surveyed seven fields that solve variants of this problem independently. Each arrived at structurally similar solutions despite different terminology. Below is what we found, what applies, and what we rejected.

### 8.1 Real-Time Operating Systems (AUTOSAR, ARINC 653, ROS 2)

**How they formulate it:** DAG of "runnables" (AUTOSAR) or "callbacks" (ROS 2) with data dependencies, mapped to multi-core processors with timing guarantees.

**Their solution:** Static ILP-based scheduling assigns tasks to cores and clusters them. AUTOSAR's Logical Execution Time (LET) decouples producers from consumers for deterministic data flow. ARINC 653 uses hierarchical two-level scheduling: one level partitions time, another schedules within partitions. The 2024 RTAS paper introduces "execution groups" -- sets of sub-jobs that must co-execute on the same processor.

**What applies to Megabake:**
- **ACCEPT: Two-level scheduling (dedicated scheduler cores + worker cores).** ARINC 653's pattern of dedicating some cores to scheduling while others execute matches MPK's scheduler SM + worker SM split. Validates the design.
- **ACCEPT: Execution groups.** The concept of co-scheduling related tasks onto the same SM group maps to Megabake's tiling -- e.g., all tiles of a matmul should run on the same SM group.
- **REJECT: Static offline scheduling.** RTOS computes schedules offline with hard real-time guarantees. Megabake needs high throughput, not worst-case timing. The overhead of computing optimal schedules is unnecessary.

Sources: [AUTOSAR DAG scheduling](https://www.sciencedirect.com/science/article/abs/pii/S1383762126002110), [RTAS 2024 Execution Groups](https://daes.cs.tu-dortmund.de/storages/daes-cs/r/Bilder/Beschaeftigte/M._Sc._Mario_Guenzel/publications/shi24rtas_group.pdf), [ReDAGRT](https://arxiv.org/pdf/2603.18238)

### 8.2 Dataflow Architectures (MIT Tagged-Token, Manchester Dataflow)

**How they formulate it:** Execute computation graphs directly in hardware using the "firing rule" -- an instruction executes when matched tokens arrive on all its input arcs.

**Their solution:** Token matching via content-addressable memory. Fine-grained dataflow: each instruction fires independently when inputs are ready. Failed commercially because fine-grained token matching was too expensive.

**What applies to Megabake:**
- **ACCEPT: The firing rule itself.** This is the universal primitive underlying ALL solutions in this survey. Whether it's an AUTOSAR runnable waiting for predecessor events, a dataflow instruction waiting for matched tokens, or an MPK task waiting for an atomic counter, the mechanism is identical. Megabake's barrier system is a coarsened firing rule.
- **ACCEPT: Coarsening as the universal solution.** Classical dataflow failed because fine-grained matching was expensive. Modern systems (MPK, ETC, Megabake) coarsen: event fusion reduces synchronization points, atomic counters replace associative matching. This validates our design of ~8 task types with coarse barriers, not fine-grained per-element synchronization.
- **ACCEPT: Kitsune (2025)** bridges dataflow and GPUs directly, constructing spatial pipelines where different operators map to different CTAs communicating via L2-pinned ring buffers -- 2.8x speedup with 99% off-chip traffic reduction.

Sources: [MIT TTDA](https://ieeexplore.ieee.org/document/48862/), [Kitsune](https://dl.acm.org/doi/10.1145/3777466)

### 8.3 FPGA High-Level Synthesis

**How they formulate it:** Compile a computation graph into persistent spatial hardware with FIFO-connected pipeline stages.

**Their solution:** Vivado/Vitis HLS `#pragma HLS dataflow` creates task-level pipelines connected by FIFO channels with backpressure. The result is a spatially-mapped DAG with zero launch overhead and deterministic latency. TAPA framework (UCLA/Cornell) compiles hierarchical task-parallel programs with typed inter-task channels.

**What applies to Megabake:**
- **ACCEPT: The GPU megakernel IS a software emulation of an FPGA dataflow pipeline.** This framing clarifies the design: each SM is a "processing element," shared memory is the "FIFO channel," barriers are "backpressure." The FPGA community has decades of experience optimizing these pipelines.
- **ACCEPT: Dynamic Loop Fusion (FPGA 2025)** uses hardware disambiguation units for runtime hazard detection when fusing tasks with unpredictable shared memory dependencies -- 14x speedup over static HLS. Relevant for MoE and other data-dependent patterns.
- **REJECT: Spatial mapping (one op = one hardware unit).** FPGAs can dedicate separate hardware to each operation permanently. GPUs must time-share SMs across operations. The spatial mapping doesn't translate, but the scheduling insights do.

Sources: [TAPA](https://dl.acm.org/doi/10.1145/3609335), [Dynamic Loop Fusion](https://arxiv.org/html/2501.09231v1)

### 8.4 Game Engine Rendering (Megakernels Considered Harmful)

**How they formulate it:** Execute a ray tracing pipeline (material shading, intersection, lighting) as either one megakernel or split specialized kernels.

**Their solution:** NVIDIA's "Megakernels Considered Harmful" (HPG 2013) found that rendering megakernels suffer severe SIMT divergence (threads hit different materials) and register pressure (196 registers). Their wavefront solution splits into ~10 specialized kernels connected by global memory queues.

**What applies to Megabake:**
- **CRITICAL INSIGHT: Rendering reached the OPPOSITE conclusion from ML inference.** They SPLIT megakernels; we FUSE into megakernels. The difference is workload character: rendering has irreducible divergence (ray tracing is fundamentally chaotic), while NN inference has uniform operations (every token does the same thing). This validates our approach -- the megakernel anti-pattern for rendering is the right pattern for inference.
- **ACCEPT: Uber-shaders as precedent.** Game engines use uber-shaders: one source file with compile-time specialization into hundreds of shader variants, avoiding runtime divergence. Megabake's parameterized task types (OP_ELEMENTWISE with op_code, OP_REDUCE with reduce_type) are the same pattern.
- **ACCEPT: GPU Coroutines (SIGGRAPH Asia 2024)** offer automated splitting/scheduling of monolithic GPU code. Could be relevant for future megakernel designs where the compiler automatically determines split points.
- **REJECT: The wavefront splitting approach.** Only applies when there's irreducible divergence, which NN inference doesn't have.

Sources: [Megakernels Considered Harmful](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf), [GPU Coroutines](https://dl.acm.org/doi/10.1145/3687766)

### 8.5 GPU Database Query Execution

**How they formulate it:** Compile a SQL query plan (DAG of relational operators) into GPU kernels with maximal fusion.

**Their solution:** Crystal (SIGMOD 2020) uses tile-based execution: load data into shared memory once, run all operators in-place. Kernel Weaver (MICRO 2012) automatically fuses relational operator kernels -- 2.89x speedup. HeavyDB does JIT fusion via LLVM, spending 90%+ of compute in a single top kernel.

**What applies to Megabake:**
- **ACCEPT: Tile-based in-place execution.** Crystal's pattern -- load tile into SMEM, run all operators, write back -- is directly applicable to fusing elementwise chains in Megabake. Load activations into SMEM -> norm -> activation -> residual add -> write back. No HBM round-trips between ops.
- **ACCEPT: JIT fusion via LLVM.** HeavyDB's approach of JIT-compiling fused operator chains maps to Megabake's model: the schedule compiler generates fused task implementations for chains of compatible ops.
- **REJECT: The relational model.** SQL operators are simpler and more uniform than NN operators. Database fusion strategies don't handle the heterogeneity of matmul + attention + norm.

Sources: [Crystal](https://github.com/anilshanbhag/crystal), [Kernel Weaver](https://ieeexplore.ieee.org/document/6493612/), [GPU DB characterization (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p441-cao.pdf)

### 8.6 CGRA Compilers (ADRES, DRESC)

**How they formulate it:** Map a data flow graph (DFG) onto a 2D array of processing elements (PEs) with limited local storage, connected via fixed interconnect. This is architecturally the MOST similar to GPU megakernels.

**Their solution:** The Modulo Routing Resource Graph (MRRG) unfolds the hardware topology across time, turning the mapping problem into subgraph homeomorphism. DRESC uses simulated annealing over placement and routing.

**What applies to Megabake:**
- **ACCEPT: Space-time decoupling.** Recent CGRA work achieved 10,000x compilation speedup by solving placement (which PE/SM runs what) separately from scheduling (when). Megabake's schedule compiler should decouple: Phase 1 assigns task types to SM groups based on resource needs, Phase 2 determines execution order given the placement.
- **ACCEPT: PEs as dual-role compute/routing resources.** In CGRAs, a PE not doing useful work can relay data. In megakernels, an SM not executing a task can prefetch data for the next task or relay results between producer and consumer SMs via shared memory.
- **ACCEPT: SAT/ILP formulations.** CGRA literature provides formal methods for encoding mapping constraints. Could be used for provably-valid schedule generation, though heuristics are likely sufficient for MVP.
- **REJECT: Simulated annealing for placement.** Too slow for ahead-of-time compilation where schedule generation needs to take seconds, not hours. Use greedy heuristics instead.

Sources: [CGRA Survey (NUS)](https://www.comp.nus.edu.sg/~tulika/CGRA-Survey.pdf), [Space-time decoupling](https://arxiv.org/pdf/2512.02859), [DRESC](https://www.semanticscholar.org/paper/DRESC/2026190e9ddbc75016881cbdd0edef2409fd856e)

### 8.7 Spatial Computing (Graphcore IPU, SambaNova RDU, Cerebras)

**How they formulate it:** Compile computation graphs onto spatial architectures where each processing element has local memory and communicates via an on-chip network.

**Their solution:** Graphcore's IPU uses Bulk Synchronous Parallel (BSP): local compute -> barrier -> data exchange -> barrier -> repeat. SambaNova's SambaFlow treats compilation as explicit place-and-route. Cerebras maps entire models onto a wafer-scale array.

**What applies to Megabake:**
- **ACCEPT: BSP model from Graphcore.** This is potentially the single most important finding. Transformer inference is layer-sequential -- layer N's output is layer N+1's input. A BSP-style megakernel (all SMs compute the same task -> barrier -> next task) is simpler and nearly as fast as MPK's event-driven model.
- **ACCEPT: Joint tensor/operation placement.** Graphcore's Poplar compiler decides tensor placement and operation placement jointly -- which tile stores each tensor and which tile computes each op. Megabake should do the same: the buffer planning phase should consider which SM group will consume each buffer to minimize data movement.
- **REJECT: Full spatial mapping.** Cerebras/SambaNova have enough PEs to spatially map entire models (one PE per op). GPUs have ~132 SMs -- not enough for spatial mapping. Time-sharing is required.

Sources: [Graphcore IPU BSP](https://arxiv.org/pdf/2311.04417), [SambaNova RDU](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf), [TileLoom](https://arxiv.org/html/2512.22168v1)

### 8.8 The Standalone-to-Embedded Problem

Across all fields, embedding a standalone function into a shared execution context is solved by one of three strategies:

**Strategy 1: Virtualize the resource interface.**
Give each sub-task a virtual view of resources (virtual blockIdx, SMEM partition). The sub-task doesn't know it's embedded. Used by: microkernels (capability-based resource delegation), WASM (linear memory per module), MPK (tasks receive task_desc pointers, don't know they're in a megakernel).

**Strategy 2: Re-optimize after merging.**
Merge compilation units and let the compiler globally re-allocate resources. Used by: JIT compilers (cross-boundary inlining), CUDA Device LTO (`-dlto` inlines `__device__` functions across translation units and globally optimizes register allocation). Works when resources are fungible (CPU registers) but struggles with physical constraints (GPU: max 255 registers/thread).

**Strategy 3: Time-share with explicit handoff.**
Don't merge resource usage -- run tasks sequentially on the same processor with barriers, saving/restoring state through shared memory. Used by: persistent thread models, BSP (Graphcore), cooperative groups, Event Tensor Compiler, Mirage MPK.

**Megabake uses all three in composition:** Strategy 1 (virtualize) for the task interface + Strategy 2 (LTO) for register optimization + Strategy 3 (time-share) for the execution model.

### 8.9 CUDA Device LTO for Register Pressure

CUDA Device LTO (`-dlto`, CUDA 11.2+) performs cross-file inlining, dead code elimination, and register optimization at link time. JIT LTO (CUDA 12.0+, via nvJitLink) does this at runtime.

**Direct application to Megabake:** Write each task implementation as a `__device__` function in a separate `.cu` file. Compile with `-dlto`. The compiler sees all tasks as one compilation unit and optimally allocates registers across the entire megakernel, avoiding worst-case-per-task register allocation. This could solve the register pressure problem without manual tuning.

Sources: [NVIDIA Device LTO](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/), [nvJitLink JIT LTO](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/)

### 8.10 PERKS -- Intentionally Low Occupancy

The PERKS paper (2022) showed that persistent kernels benefit from intentionally launching fewer thread blocks per SM. Each block gets more registers and shared memory. For megakernels: launch one block per SM, give it all 255 registers and all 228KB shared memory. The block runs forever (persistent), so occupancy doesn't matter the way it does for short-lived kernels.

**Direct application:** Megabake should launch exactly one thread block per SM with maximum register and shared memory allocation. This maximizes per-task resource availability. MPK does this implicitly; now we understand the theoretical justification.

Sources: [PERKS](https://arxiv.org/pdf/2204.02064)

### 8.11 BSP vs Event-Driven Scheduling

The cross-domain research surfaced a key architecture question: should Megabake use MPK-style event-driven scheduling or Graphcore-style BSP scheduling?

**MPK-style (event-driven):**
```
Scheduler SMs process events, push tasks to worker SM queues.
Workers pop tasks atomically, execute, fire events.
Fine-grained: tasks can overlap across layers.
~500 lines CUDA. Complex. Proven at OSDI 2026.
```

**BSP-style (barrier-based, from Graphcore):**
```cuda
__global__ void megakernel(Task* tasks, int num_tasks, void** buffers) {
    for (int i = 0; i < num_tasks; i++) {
        int my_tile = blockIdx.x;
        if (my_tile < tasks[i].num_tiles) {
            switch (tasks[i].op_type) { ... }
        }
        cooperative_groups::this_grid().sync();  // barrier
    }
}
```
```
All SMs execute the same task (different tiles) then barrier.
No scheduler SMs needed. No work queues. No event derivation.
~50 lines CUDA. Simple. cooperative_groups::grid.sync() is one instruction.
```

**Analysis:**

| Aspect | Event-driven (MPK) | BSP (Graphcore-inspired) |
|---|---|---|
| Scheduler complexity | ~500 lines CUDA | ~50 lines CUDA |
| Cross-task pipelining | Yes (weight prefetch during compute) | No (barrier between tasks) |
| Wave quantization within a task | Eliminated (dynamic SM assignment) | Present (SMs with fewer tiles idle) |
| Scheduler SM overhead | 4-8 SMs dedicated to scheduling | Zero (all SMs compute) |
| Schedule compiler complexity | GCD-based event derivation needed | Flat task list, trivial |
| Performance gap | Baseline (best) | ~10-15% slower (no pipelining, some wave quant) |
| Implementation risk | Medium (complex, many moving parts) | Low (simple loop + barrier) |

**Decision: Start with BSP for MVP, upgrade to event-driven if benchmarks demand it.**

Transformer inference is layer-sequential -- there's almost no cross-layer parallelism for event-driven scheduling to exploit. The 10-15% performance gap from losing inter-task pipelining is acceptable for the massive simplicity gain. The upgrade path is clean: BSP and event-driven use the same task implementations. Only the scheduler wrapper changes.

### 8.12 Compiler Theory: NP-Hardness

The mapping problem (DAG -> fixed parallel machine) decomposes into subproblems, each NP-hard: task partitioning (graph partitioning), ordering (DAG scheduling with precedence), minimizing peak memory (proven NP-complete, IEEE 2025), and choosing fusion boundaries.

**Practical implication:** Optimal schedules are intractable. Use heuristic scheduling (greedy topological order, resource-aware placement). This is what every production system does -- MPK, Luminal, TVM, Halide all use heuristics, not optimal solvers.

**The WELDER system (OSDI 2023)** provides a tile-graph abstraction that discovered 89 fusion patterns missed by rule-based systems. Relevant for finding non-obvious fusion opportunities in the FX graph.

Sources: [Memory-constrained scheduling NP-completeness](https://ieeexplore.ieee.org/document/11078551/), [WELDER](https://www.usenix.org/conference/osdi23/presentation/shi), [Halide autoscheduler](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)

### 8.13 Summary: What We Take From Cross-Domain Research

| Technique | Source Domain | How It Applies to Megabake | Phase |
|---|---|---|---|
| BSP execution model | Graphcore IPU | Simplifies scheduler from ~500 to ~50 lines CUDA. All SMs execute same task, barrier, next task. | 1 (MVP) |
| Space-time decoupling | CGRA compilers | Solve placement (which SMs) separately from scheduling (when). 10,000x compilation speedup. | 1 |
| CUDA Device LTO | Compiler technology | Compile tasks as separate `.cu` files, LTO merges and optimally allocates registers globally. | 1 |
| PERKS (low occupancy) | Persistent thread research | Launch 1 block/SM with max registers + SMEM. Persistent kernel doesn't need high occupancy. | 1 |
| Virtualized task interface | Microkernels, WASM | Tasks receive virtual resource descriptors, don't know they're embedded. Strategy 1. | 1 |
| Tile-based in-place execution | GPU databases (Crystal) | Fuse elementwise chains: load tile into SMEM, run all ops, write back. Zero HBM between ops. | 1 |
| The firing rule (coarsened) | Dataflow architectures | Atomic counter barriers are coarsened firing rules. Validates our synchronization design. | 1 |
| Event-driven scheduling upgrade | MPK / RTOS | If BSP benchmarks show >15% gap, upgrade to event-driven scheduler. Same task code, different wrapper. | 2 |
| Joint tensor/op placement | Graphcore Poplar | Buffer planning considers which SM group consumes each buffer. Minimize data movement. | 2 |
| Uber-shader parameterization | Game engines | Parameterized task types (OP_ELEMENTWISE with op_code) avoid combinatorial task explosion. | 1 |

## 9. Competitive Landscape

| | Compilation | Distribution | PyTorch Native | Automated |
|---|---|---|---|---|
| Hazy Research | Hand-built | Internal only | No | No |
| Together AI | Hand-built | Production only | No | No |
| Mirage MPK | Semi-auto | No | No (own API) | Partially |
| Luminal | Automated | No | Thin bridge | Yes |
| AutoMegaKernel | LLM-agent | No | No | Yes |
| HF Kernels Hub | N/A (individual ops) | Yes (HF native) | No | N/A |
| AOTI | Inductor (no megakernels) | .so files | Yes | Yes |
| **Megabake** | **Automated** | **HF native** | **Yes (torch.export)** | **Yes** |

### What to Take from Luminal

Luminal's `pt2.py` is the closest existing code to what Megabake needs. Here's what's directly reusable vs what needs to change:

**Directly applicable (steal the approach):**

1. **Parameter re-internalization** (pt2.py:344-393) -- When `torch.compile` lifts model weights as extra inputs, Luminal puts them back as graph attributes. Megabake needs this exact same logic to ensure weights end up in the exported state_dict, not as runtime inputs.

2. **SymInt stripping** (pt2.py:208-303) -- Converts Dynamo's symbolic shape placeholders into `tensor.size(dim)` calls. Essential for clean graph export with dynamic shapes.

3. **SDPA preservation** (pt2.py:113-136) -- The decomposition table explicitly keeps `scaled_dot_product_attention` as a single node rather than decomposing it into 20 ops. Megabake needs this -- SDPA maps to a single attention task.

4. **Zero-copy weight sharing** -- Passing device pointers from the original PyTorch model to the compiled runtime. Megabake does this at load time (map safetensors pointers into the megakernel's buffer layout).

5. **Lazy compilation for dynamic shapes** (pt2.py:671-735) -- Deferring `torch.export` to the first call to avoid corrupting Dynamo's ShapeEnv. If Megabake ever runs inside a `torch.compile` backend frame, this pattern is needed.

**Needs to diverge:**

1. **The Rust translator** -- Luminal translates ATen ops to its own 15 primitive ops. Megabake translates ATen ops to megakernel task types (~10-15 ops, but different granularity -- coarser, fused). Write this in Python, not Rust.

2. **E-graph optimization** -- Luminal's core value is search-based optimization via egglog. Megabake doesn't need this for the task schedule -- the schedule is deterministic from the graph structure. Search-based optimization is for kernel code generation, which Megabake does not do (it uses CUTLASS/FlashAttention).

3. **The runtime** -- Luminal's `CudaRuntime` is a full execution engine. Megabake's runtime is much thinner: load cubin, load task schedule, set up buffers, launch one kernel.

4. **No save/load** -- Luminal has zero serialization. This is Megabake's entire product.

### What to Take from MPK

MPK is the only project that has shipped all ops inside a single persistent kernel, on real models, across three GPU architectures, validated at OSDI 2026. MPK's task implementations are BSD-licensed -- the WGMMA patterns, TMA usage, shared memory layouts, and pipeline strategies are all there to learn from.

Key insight: The two hardest problems (matmul and attention as persistent kernel tasks) are solved by MPK but NOT by Luminal. MPK's hand-written WGMMA kernels per architecture (~500 lines each) and paged attention implementations (~800 lines each) are the primary reference for Megabake's hard kernel implementations.

## 10. Gap Analysis

### How MPK and Luminal Handle Each Gap

| Gap | **Mirage MPK** | **Luminal** | **Inspiration for Megabake** |
|-----|----------------|-------------|-------------------------------|
| **KV Cache** | Paged KV cache managed **entirely on-GPU**. Circular page queue, allocation/free in `prepare_next_batch` without returning to CPU. Compile-time max pages/page size baked via `-D` flags. | No built-in KV cache. Relies on HuggingFace `DynamicCache` via pytree serialization. Each new cache size triggers Dynamo recompile. | MPK's on-GPU page allocator is the gold standard. For Megabake MVP, start with Luminal's approach (let torch.export handle KV cache as a graph input), graduate to on-GPU paging later. |
| **Continuous Batching** | Three modes: `MODE_OFFLINE`, `MODE_ONLINE`, `MODE_ONLINE_PINNED` (true continuous batching via pinned ring buffers -- CPU writes requests, GPU drains them each iteration). | Not supported. | MPK's `MODE_ONLINE_PINNED` is the target for serving integration. But for MVP, offline batch-1 decode is sufficient. Continuous batching is a Phase 2 concern. |
| **Multi-GPU / TP** | NVSHMEM inside the persistent kernel. Two strategies: allgather+reduce (general) and tile-based NVSHMEM allreduce (SM>=90 with multicast). | Not supported. | Defer to Phase 2. The task schedule format should reserve space for multi-GPU metadata, but don't implement it until single-GPU is solid. |
| **Prefill vs Decode** | Same megakernel, different behavior. `prepare_next_batch` checks if `prompt_length - step > 0`. Chunked prefill supported. | Implicit via the `DimBucket` system -- separate compiled buckets for different dim ranges. | **Two separate task schedules** -- one for decode, one for prefill. Same kernel binary, different task ordering and tiling. Cleaner than MPK's runtime branching. |
| **Numerical Validation** | Extensive per-op tests comparing against PyTorch reference. | Every test follows: `torch.compile` -> run both -> `torch.allclose`. Rust-side has random graph fuzzing. Runtime has post-execution NaN/Inf detection. | **Both approaches.** Per-task unit tests (MPK-style) plus end-to-end `torch.allclose` (Luminal-style). Add NaN detection. |
| **Memory Management** | All on-GPU. Up to 207KB shared memory per SM on Hopper. | Single-arena allocator per bucket. Liveness analysis + first-fit packing. | Luminal's arena approach fits Megabake well. Static buffer layout with aliased slots -- the runtime allocates one contiguous arena. Report total workspace bytes in metadata. |
| **LoRA** | **No.** | **No.** | Separate schedule per (model, adapter) pair, or defer LoRA support entirely. |
| **Speculative Decoding** | **Full Eagle3 integration.** Draft model as separate task graph, on-GPU commit/verify kernels. | **No.** | Spec decoding is a task-schedule-level concern. Add draft model tasks and verify/commit task types. Defer to post-MVP. |
| **Version Compatibility** | `_save_kernel_metadata` saves JSON with mode, shapes, cuda_cc, tensor names. `_validate_kernel_compatibility` checks all fields on load. | **None.** No artifacts are persisted. | Header includes `format_version`, `export_ir_version`, `cuda_driver_min_version`, `cubin_sm_version`, `megabake_runtime_version`, and task type registry hash. |
| **Serialization** | `.so` + `task_graph.json` + `kernel_metadata.json`. Can save and reload. | **Nothing.** Compilation is entirely ephemeral. | This is the gap Megabake fills. Megabake replaces MPK's fragile `.so` (tied to Python version, CPU arch) with a clean cubin + data format. |

### Priority Matrix

| Priority | Gap | Action |
|----------|-----|--------|
| **Must have (MVP)** | Prefill vs decode | Two task schedules per model, same kernel binary |
| **Must have (MVP)** | Numerical validation | Per-task unit tests + end-to-end torch.allclose |
| **Must have (MVP)** | Version compatibility | Header fields for cuda driver, SM version, runtime version, task registry hash |
| **Must have (MVP)** | Memory overhead | Report workspace bytes in header, arena allocator, document VRAM cost |
| **Should have (Phase 2)** | KV cache / paged attention | Start with KV as graph input, add on-GPU paging task later |
| **Should have (Phase 2)** | Continuous batching | Online pinned mode (MPK-style ring buffers) |
| **Should have (Phase 2)** | Multi-GPU / TP | NVSHMEM allreduce tasks, multi-rank task schedules |
| **Nice to have (Phase 3)** | Speculative decoding | Draft model tasks + verify/commit task types |
| **Nice to have (Phase 3)** | LoRA serving | Separate schedules per adapter, or dynamic weight remapping |
| **Trivial** | Licensing | CUTLASS BSD-3, FlashAttention BSD-3, all deps permissive |
| **Trivial** | Warm-up latency | Benchmark and document cuModuleLoad time |

## 11. Performance Targets

### Published Megakernel Benchmarks (Existing Systems)

| System | Model | Hardware | Speedup | Source |
|--------|-------|----------|---------|--------|
| Hazy Research | Llama-1B | H100 | 2.5x vs vLLM, 1.5x vs SGLang | Stanford, May 2025 |
| Hazy Research | Llama-70B TP | H100 | 1.22x throughput vs SGLang | Stanford, Sep 2025 |
| Together AI | Llama-3.2-1B | H100 | 3.6x vs baseline | Production, 2025 |
| Together AI | Qwen-2.5-1.5B | B200 | 2.3x vs baseline | Production, 2025 |
| MPK | Multiple | H100 | 1.2-6.7x latency reduction | OSDI 2026 |
| Luminal | Llama-3-8B Q8 | H100 | 3.2x vs vLLM throughput | Luminal, 2026 |

### Expected Megabake Performance

Megabake uses the same megakernel execution model but wraps library kernels (CUTLASS, FlashAttention) rather than hand-tuned per-architecture implementations. Expected ~90-95% of hand-tuned per-op quality.

**Batch-1 / small-batch decode (the sweet spot):**

| Metric | torch.compile | AOTI + CUDA Graphs | Megabake | Hand-tuned (MPK/Hazy) |
|---|---|---|---|---|
| Relative latency | 1.0x | ~0.75x | ~0.40-0.50x | ~0.30-0.40x |
| HBM bandwidth util | ~45% | ~55% | ~65-75% | ~75-80% |
| Kernel launches/fwd | ~100 | 1 (graph) | 1 (persistent) | 1 (persistent) |

**Speedup sources:**

| Source | Gain | Applies to Megabake |
|--------|------|------------------------|
| Kernel launch elimination | 10-15% | Yes |
| Wave quantization removal | 10-20% | Yes |
| Memory bubble elimination | 15-30% | Yes |
| Register reuse across ops | 5-15% | Partially |
| Reduced HBM round-trips | 10-20% | Yes |

**Combined realistic estimate:**
- 2.0-2.8x over torch.compile (batch-1 decode)
- 1.3-1.7x over AOTI + CUDA Graphs (batch-1 decode)
- 1.1-1.5x improvement for larger batch / prefill (compute-bound, diminishing returns)

**Versus hand-tuned megakernels (Hazy/MPK/Together):**
- ~10-15% slower due to library kernels vs hand-tuned per-architecture implementations
- Gap closes over time as CUTLASS and FlashAttention improve
- Selective hand-tuning of the matmul task (60-70% of runtime) would close most of the gap

### Performance Ceiling

The theoretical minimum for decode is: `model_size_bytes / HBM_bandwidth`.

On H100 (3.35 TB/s), Llama-8B FP16 (~16GB): **~4.8ms per token**.

Megakernels achieve 70-80% of this limit. Standard frameworks achieve 40-50%. Megabake targets 65-75%.

## 12. Feasibility Analysis

### What Will Just Work

**1. torch.export -> FX graph -> task list (Python).** Feasibility: 100%.
Pattern matching ATen ops to task types is a lookup table. Extracting dimensions from node metadata is trivial. Schedule generation (topological sort, dependency analysis, buffer planning) is standard compiler work in Python. This is the easy part.

**2. The file format.** Feasibility: 100%.
A container with a JSON header + binary task schedule + cubin blob. Reader/writer in a few hundred lines of Python/C++. Format design is never the hard part.

**3. The scheduler/dispatch loop (CUDA).** Feasibility: 95%.
MPK has proven this works at OSDI 2026. The BSP model simplifies this further to ~50 lines of CUDA. The only question is tuning (cooperative launch configuration), not feasibility.

**4. HuggingFace integration.** Feasibility: 95%.
HF already has a Kernels Hub. Adding a new file type to their repos is their existing infrastructure. kernels.json index, hardware matching, conditional download -- all solved problems.

### Hard But Solvable

**5. Event granularity derivation from FX graph.** Feasibility: 85%.
MPK's GCD-based algorithm in `annotated_graph.cc` works on their custom graph IR. Porting it to work on FX graph edges requires understanding the tiling relationship between producer and consumer ops. Note: with BSP scheduling, this becomes less critical since tasks execute sequentially with grid-wide barriers.

**6. Buffer/memory planning.** Feasibility: 90%.
Static liveness analysis on the task graph, first-fit packing of non-overlapping buffers into an arena. Luminal does this well (`plan_intermediate_buffers` in their runtime). Standard compiler technique. The challenge is getting aliasing right -- if task A's output buffer is task B's input, they must share the same pointer. Bugs here cause silent corruption.

**7. Shape bucketing.** Feasibility: 90%.
Compiling separate task schedules for different shape ranges (seq_len 1-128, 128-512, etc.). The kernel binary stays the same, only the task schedule changes. The runtime picks the right bucket. Luminal already does this with `DimBucket`. The question is how many buckets and where the boundaries are.

### Genuinely Hard

**8. Matmul as a task inside a persistent kernel.** Feasibility: 60%.

This is the single biggest technical risk in the entire project. CUTLASS is designed to own the entire kernel launch -- it manages its own grid, its own threadblocks, its own shared memory, its own warp scheduling, its own pipeline stages. You can't just call `cutlass::gemm()` from inside another kernel as a `__device__` function.

What MPK does instead: they don't use CUTLASS's high-level API at all. Their matmul tasks are hand-written WGMMA kernels using CUTLASS's low-level building blocks (`cute::copy`, `cute::gemm`, WGMMA instructions, TMA descriptors). Their `linear_hopper.cuh` is ~500 lines of hand-tuned CUDA, not a CUTLASS wrapper.

Options:
- **Option A:** Write hand-tuned matmul tasks per architecture (like MPK). ~500 lines per architecture x 3 architectures = ~1500 lines of the hardest CUDA you'll ever write.
- **Option B:** Use CUTLASS 3.x's `CollectiveMainloop` and `CollectiveEpilogue` components as mid-level building blocks -- tile-level control without writing raw WGMMA. ~200-300 lines per matmul variant.
- **Option C (recommended):** Use NVIDIA's [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) -- CUTLASS's new Python DSL (graduating summer 2026). Write the matmul task in Python, get WGMMA/TMA for free. Supports cooperative kernel launch natively. Orders of magnitude faster compile times than nvcc. This could solve the hardest problem without hand-writing WGMMA.

Budget 2-3 months for matmul tasks regardless of approach.

**9. Attention as a task inside a persistent kernel.** Feasibility: 55%.

Same fundamental problem. FlashAttention is a standalone `__global__` kernel with its own tiling, its own shared memory management, its own online softmax state. You can't call it as a device function.

MPK's solution: hand-written paged attention kernels per architecture (`multitoken_paged_attention_hopper.cuh` -- ~800 lines of CUDA).

Options:
- **Option A:** Hand-write attention tasks using MPK's implementations as reference.
- **Option B:** Use [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s lower-level primitives. FlashInfer is more composable than FlashAttention and has JIT compilation. Still requires adaptation for the persistent kernel context.
- **Option C (pragmatic MVP):** Don't fuse attention into the megakernel. Run attention as a separate kernel launch, fuse everything else. You lose some benefit but attention is already heavily optimized. This might be the right MVP path.

Budget 2-3 months of hard CUDA work for full fusion. Consider Option C for MVP.

**10. Register pressure from the dispatch switch.** Feasibility: 80%.

The megakernel's `_execute_task()` function has a switch statement over all task types. Every task's `__device__` function is compiled into the same kernel, which means the compiler must allocate registers for the worst-case task. If the matmul task uses 128 registers and RMSNorm uses 32, every SM pays the 128-register cost even when running RMSNorm.

Mitigation: CUDA Device LTO (`-dlto`) + PERKS-style low occupancy. Keep the number of task types small (~10-15), keep each task's register usage reasonable, profile and tune aggressively.

**11. torch.export coverage for real models.** Feasibility for LLaMA MVP: 90%. Feasibility for "any model": 60%.

Not all HuggingFace models trace cleanly through torch.export. Common issues: custom attention implementations with Python control flow, data-dependent shapes (e.g., MoE routing), custom CUDA extensions that aren't traceable. For LLaMA-family models (the MVP target), torch.export works well. Expanding beyond LLaMA will hit export limitations.

### Which Hard Problems Are Already Solved

Every hard problem has been solved by at least one existing project:

| Hard Problem | MPK | Luminal |
|---|---|---|
| Matmul inside persistent kernel | **Solved.** Hand-written WGMMA kernels per arch. ~500 lines each. | Solved differently. Generates matmul CUDA via e-graph, but uses CUDA graphs not persistent kernel. |
| Attention inside persistent kernel | **Solved.** Hand-written paged attention per arch. ~800 lines each. | Not solved for megakernels. |
| Register pressure | **Solved.** Ships across Ampere/Hopper/Blackwell. Tasks are coarse-grained, absorbing dispatch overhead. | Solved differently. Dynamic scheduling with lightweight symbolic queue entries. |
| Event/barrier granularity | **Solved.** GCD-based event computation. Published at OSDI 2026. | Solved differently. Increment-decrement barriers. |
| Buffer/memory planning | **Solved.** Static buffer planning, raw pointers in task descriptors. | **Solved.** Arena allocator with liveness analysis and first-fit packing. |
| torch.export coverage | N/A (own API). | **Solved for ~80 ATen ops.** Translator handles binary, unary, matmul, SDPA, conv, reductions, gather/scatter, embedding. |

## 13. Technology Stack & Tooling

```
Layer 4: Distribution
  huggingface_hub                pip install, use directly
  HF Kernels Hub                 host megabake-runtime, HF builds all variants
  HF kernel-builder              Nix flake, compiles for all platforms

Layer 3: Loader
  safetensors                    pip install, weight loading
  CUDA driver API                cuModuleLoad for cubin loading

Layer 2: Task Implementations
  CuTe DSL                       matmul task in Python, generates WGMMA/TMA
  FlashInfer                     attention primitives (or hand-write using MPK ref)
  Raw CUDA                       trivial tasks (norm, silu, rope, embedding)

Layer 1: Scheduler
  BSP scheduler                  cooperative_groups barrier, ~50 lines CUDA
  NVRTC                          compile to cubin

Layer 0: Input
  torch.export                   model tracing, FX graph, already in PyTorch
```

**Dependencies (all permissively licensed):**

| Dependency | License | Purpose |
|---|---|---|
| PyTorch | BSD | torch.export, FX graph |
| CUTLASS | BSD-3 | CuTe DSL, GEMM building blocks |
| FlashInfer | Apache-2.0 | Attention primitives |
| safetensors | Apache-2.0 | Weight loading |
| huggingface_hub | Apache-2.0 | Distribution |
| CUDA Toolkit | NVIDIA EULA | NVRTC, driver API |

**Available tooling from related projects:**

| Tool | What it does for Megabake |
|------|--------------------------|
| `torch.export` | Traces PyTorch model -> ExportedProgram (FX graph). The input format. |
| `safetensors` | Weight loading at runtime. Map tensor names -> GPU pointers into the megakernel's buffer layout. |
| `huggingface_hub` | Upload/download artifacts, hardware matching, file filtering. |
| NVRTC | Compile CUDA source -> cubin at build time. Ships with CUDA toolkit. |
| [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) | Write CUDA kernels in Python using CUTLASS abstractions. Supports cooperative kernel launch. Generates optimized WGMMA/TMA automatically. Graduating summer 2026. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | More composable attention kernels than FlashAttention. JIT compilation. Lower-level primitives adaptable for persistent kernel context. |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | Hazy Research's tile-level kernel DSL. The original megakernel was built using this. |
| [mKernel](https://uccl-project.github.io/posts/mkernel/) | Berkeley's multi-GPU fused kernel library. GPU-initiated RDMA inside a persistent kernel without NCCL/NVSHMEM. Hopper only. Phase 2. |

## 14. Scope & Timeline

### Phase 1: Proof of Concept (Months 1-3)

**Goal:** End-to-end megakernel execution for LLaMA-8B decode on H100 with trivial ops only (no matmul, no attention -- use separate kernel launches as fallback).

**Deliverables:**
- Schedule compiler: FX graph -> task schedule (Python)
- Persistent kernel scheduler with dispatch loop (CUDA)
- Simple task implementations: RMSNorm, SiLU*Mul, RoPE, embedding, residual add, cast
- Matmul and attention as fallback separate kernel launches
- .schedule file format
- Runtime loader: load schedule + cubin + weights -> run
- Proof that the scheduler/dispatch architecture works end-to-end

**Success metric:** LLaMA-8B decode runs correctly (torch.allclose) through the megakernel, even if matmul/attention are launched separately.

### Phase 2: Full Megakernel (Months 3-6)

**Goal:** Matmul and attention fused into the persistent kernel. First real performance numbers.

**Deliverables:**
- Matmul task via CuTe DSL or CUTLASS mid-level API
- Attention task (hand-written using MPK reference or FlashInfer adaptation)
- Register pressure profiling and tuning
- First full-model megakernel (all ops fused, zero separate launches)
- Benchmark: Megabake vs torch.compile vs AOTI+CUDAGraphs on LLaMA-8B decode
- MoE support (OP_GATE_TOPK, OP_MOE_GEMM via CUTLASS grouped GEMM)
- KV cache as paged attention + on-GPU page management
- Multi-GPU / TP via OP_ALLREDUCE (NVSHMEM or mKernel)
- FP8, INT8 dtype support
- Continuous batching (on-GPU batch scheduler)

**Success metric:** 1.5x+ speedup over torch.compile on batch-1 LLaMA-8B decode.

### Phase 3: Polish & Coverage (Months 6-9)

**Goal:** Production-quality megakernel with multiple hardware targets and dtypes.

**Deliverables:**
- A100 (sm_80) support
- Blackwell (sm_100) support
- Prefill megakernel (separate task schedule, same kernel binary)
- Shape bucketing with autotuned bucket boundaries
- Numerical validation suite: per-task unit tests + end-to-end model tests
- Performance tuning: close the gap to MPK numbers
- INT4 (GPTQ/AWQ), NF4 (QLoRA) quantization support
- Speculative decoding (draft + target megakernels, verify/commit tasks)
- Dynamic LoRA adapter weight swapping
- Mamba / SSM task types

**Success metric:** 2.0x+ over torch.compile, <15% gap vs MPK, correct on all LLaMA variants.

### Phase 4: Ecosystem (Months 9-12)

**Goal:** Megabake artifacts available on HuggingFace for top models.

**Deliverables:**
- megabake-runtime published on HF Kernels Hub
- Schedule files published for top-20 HF models
- Build service for automated schedule generation
- `pip install megabake` with `load()` and `bake()` API
- PyTorch RFC for upstream loader integration
- Integration guide for vLLM/SGLang
- Documentation, benchmarks, blog post

**Success metric:** A user can `pip install megabake` and run LLaMA-8B at megakernel speed in under 60 seconds.

## 15. Risk Matrix

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Matmul can't be embedded in persistent kernel | **Fatal** | Medium | CuTe DSL, CUTLASS mid-level API, or hand-write like MPK. **Prototype in first 2 weeks.** |
| Attention can't be embedded | **High** | Medium | FlashInfer primitives, hand-write using MPK reference, or accept partial megakernel (attention as separate launch) |
| Register pressure kills occupancy | **High** | Medium | CUDA Device LTO (`-dlto`), PERKS-style 1 block/SM, limit task types (~12), profile with Nsight Compute |
| torch.export doesn't trace target model | **Medium** | Low for LLaMA | Start with LLaMA family, expand incrementally |
| HF doesn't want to host schedule files | **Low** | Very Low | Model repos already accept arbitrary files. No HF changes needed. |
| MPK or Luminal adds distribution | **Medium** | Low-Medium | Ship first. PyTorch-native position is the moat. |

**Critical first step:** Prototype a CuTe DSL matmul as a `__device__` function inside a minimal persistent kernel in weeks 1-2. If this works, the project is feasible. If not, decide between hand-writing (like MPK, adds 2 months) or partial megakernel (matmul as separate launch, reduces speedup to ~1.3-1.5x).

## 16. Strategic Position

**Why this wins:**

1. **torch.export in, AOTI out.** Any model that works with torch.export works with Megabake. The loader can be upstreamed into PyTorch. No separate framework to install.

2. **Distribution is the product.** MPK and Luminal answer "how do I compile a megakernel?" Megabake answers "how do I get one without compiling?" The compiler is swappable. The ecosystem integration compounds.

3. **Upstreamable.** Built by a PyTorch Dynamo maintainer, on PyTorch's own IR. The path to `torch.megabake.load()` is a PyTorch RFC, not a third-party integration. Once upstreamed, every inference framework gets it for free.

4. **HF-native distribution.** Kernel binary on Kernels Hub (HF builds it), schedules in model repos (data files). No new infrastructure. Users download and run.

**What this is NOT:**
- Not a new ML framework
- Not a new kernel language
- Not a research project -- it's engineering on proven designs (MPK scheduler, CUTLASS matmul, FlashInfer attention)
- Not competing with MPK/Luminal on compiler quality -- competing on distribution and ecosystem integration

## 17. What We Don't Build

| Thing | Why not |
|---|---|
| A new file format | Kernel binary distributed via HF Kernels Hub (existing infra). Schedules are plain binary data files in model repos. |
| A new graph IR | torch.export's FX graph is the IR. No second representation. |
| A new runtime/framework | The runtime is one CUDA kernel + a thin Python loader. Not a framework. |
| Multi-platform support | NVIDIA CUDA only. The persistent kernel primitives don't exist elsewhere. |
| A new kernel language | CUDA C++ for task implementations. CuTe DSL for matmul. No new language. |
| Training support | Inference only. Decode-first, prefill second. |

## 18. Summary Decision Matrix

| Problem | Phase | Strategy |
|---|---|---|
| **Scheduler architecture** | 1 | **BSP model** (from Graphcore IPU research). All SMs execute same task, cooperative_groups barrier, next task. ~50 lines CUDA. Upgrade to event-driven (MPK-style) if benchmarks show >15% gap. |
| **Register pressure** | 1 | **CUDA Device LTO** (`-dlto`). Tasks in separate `.cu` files, compiler globally optimizes registers. **PERKS-style** 1 block/SM with max registers + SMEM. |
| **Schedule compilation** | 1 | **Space-time decoupling** (from CGRA research). Solve placement (which SMs) separately from scheduling (when). |
| Custom ops | 1 | Graph splitting. Custom ops run as separate launches between megakernel segments. |
| MoE | 2 | New task types: OP_GATE_TOPK, OP_MOE_PERMUTE, OP_MOE_GEMM (grouped GEMM via CUTLASS). |
| Mamba/SSM | 3+ | New OP_SELECTIVE_SCAN task type. Until then, graph split fallback. |
| Multimodal | 1 | Graph splitting handles naturally. Each sub-model is a megakernel segment. |
| KV cache | 1: graph input, 2: paged | Phase 1: PyTorch manages cache externally. Phase 2: on-GPU paged attention + page allocator. |
| Continuous batching | 2 | On-GPU batch scheduler with pinned ring buffers (MPK's MODE_ONLINE_PINNED). |
| LoRA | 1 | Separate schedule per (base_model, adapter) pair. Same kernel binary. |
| QLoRA | 3 | OP_DEQUANT_NF4 task type + LoRA schedule. |
| LoRA serving (many adapters) | 3 | Dynamic weight swapping -- same schedule, swap buffer pointers at runtime. |
| FP16/BF16 | 1 | Default dtype. Single kernel binary. |
| FP8 | 2 | Separate kernel binary with OP_MATMUL_FP8. OP_QUANTIZE_FP8 for dynamic quant. |
| INT8/INT4 (GPTQ/AWQ) | 3 | Separate kernel binary. OP_DEQUANT_MATMUL fused task. |
| Speculative decoding | 3 | Two megakernels (draft + target) + OP_SPEC_VERIFY/COMMIT tasks. |
| Dynamic shapes | 1 | Shape buckets. Multiple schedules per model, runtime selects best match. |
| Multi-GPU TP | 2 | OP_ALLREDUCE task type. NVSHMEM or mKernel. Per-rank schedule files. |
| Correctness | 1 | Validation at bake time (torch.allclose). Debug mode. Per-task unit tests. NaN detection. |
| Unsupported models | 1 | Graceful fallback to torch.compile when no schedule matches or export fails. |

## 19. Implementation Guide

This section contains everything an implementation agent running on a CUDA-equipped machine needs to build Megabake from scratch. Every data structure, binary layout, algorithm, and build step is specified precisely enough to write code without ambiguity.

### 20.1 Repository Structure

```
megabake/
  src/
    megabake/                     # Python package
      __init__.py                 # load(), bake(), version
      schedule_compiler/          # Component 1: torch.export -> .schedule
        __init__.py
        graph_walker.py           # FX graph traversal + ATen op mapping
        shape_ops.py              # Zero-cost stride resolution
        buffer_planner.py         # Arena allocation with liveness analysis
        tiling.py                 # Per-task tile count computation
        serializer.py             # Write binary .schedule files
        op_table.py               # ATen op -> (task_type, op_code) mapping
      runtime/                    # Component 3: load + launch
        __init__.py
        loader.py                 # Load cubin + schedule + weights
        launcher.py               # cudaLaunchCooperativeKernel wrapper
        weight_mapper.py          # safetensors key -> buffer index mapping
        shape_selector.py         # Pick schedule for input shape bucket
      distribution/               # Component 4: HF integration
        __init__.py
        hub.py                    # Upload/download schedules via huggingface_hub
        index.py                  # Read/write index.json
        build_service.py          # Automated schedule generation for models
      cli.py                      # megabake bake / megabake load entry point
    cuda/                         # Component 2: kernel runtime (CUDA sources)
      scheduler.cu                # BSP scheduler loop + dispatch switch
      tasks/
        matmul.cu                 # OP_MATMUL: CUTLASS mid-level or CuTe DSL
        attention.cu              # OP_ATTENTION: hand-written or FlashInfer
        elementwise.cu            # OP_ELEMENTWISE: add, mul, silu, gelu, etc.
        reduce.cu                 # OP_REDUCE: sum, mean, softmax, rmsnorm, layernorm
        embedding.cu              # OP_EMBEDDING: table lookup
        index.cu                  # OP_INDEX: gather, scatter, index_select
        copy.cu                   # OP_COPY: cat, stack, contiguous
        rope.cu                   # OP_ROPE: rotary position embedding
      megakernel.cu               # Includes all tasks + scheduler, compilation entry
      data_types.cuh              # Shared struct definitions (TaskDesc, etc.)
    tests/
      test_tasks/                 # Level 1: per-task unit tests
        test_matmul.py
        test_elementwise.py
        test_reduce.py
        test_attention.py
        test_embedding.py
        test_index.py
        test_copy.py
        test_rope.py
      test_schedule_compiler/     # Level 2: schedule compiler tests
        test_graph_walker.py
        test_buffer_planner.py
        test_shape_ops.py
        test_serializer.py
      test_e2e/                   # Level 3: end-to-end model tests
        test_llama_1b.py
        test_llama_8b.py
        test_mistral_7b.py
        test_qwen_7b.py
      test_perf/                  # Level 4: performance benchmarks
        bench_decode_latency.py
        bench_bandwidth.py
  pyproject.toml
  CMakeLists.txt
```

### 20.2 Data Structures

#### C Structs (GPU-side, defined in `data_types.cuh`)

```cuda
#pragma once
#include <cstdint>

// Op type codes -- which task function to call
#define OP_MATMUL       0x01
#define OP_ATTENTION    0x02
#define OP_ELEMENTWISE  0x03
#define OP_REDUCE       0x04
#define OP_EMBEDDING    0x05
#define OP_INDEX        0x06
#define OP_COPY         0x07
#define OP_ROPE         0x08

// Elementwise op codes (task.op_code when task.op_type == OP_ELEMENTWISE)
#define ELEM_ADD          0x00
#define ELEM_MUL          0x01
#define ELEM_SUB          0x02
#define ELEM_DIV          0x03
#define ELEM_SILU         0x04
#define ELEM_GELU         0x05
#define ELEM_RELU         0x06
#define ELEM_SIGMOID      0x07
#define ELEM_TANH         0x08
#define ELEM_EXP          0x09
#define ELEM_LOG          0x0A
#define ELEM_RSQRT        0x0B
#define ELEM_NEG          0x0C
#define ELEM_ABS          0x0D
#define ELEM_CLAMP        0x0E
#define ELEM_WHERE        0x0F
#define ELEM_CAST         0x10
#define ELEM_MASKED_FILL  0x11
#define ELEM_POW          0x12

// Reduce op codes (task.op_code when task.op_type == OP_REDUCE)
#define REDUCE_SUM        0x00
#define REDUCE_MEAN       0x01
#define REDUCE_MAX        0x02
#define REDUCE_SOFTMAX    0x03
#define REDUCE_RMSNORM    0x04
#define REDUCE_LAYERNORM  0x05
#define REDUCE_ARGMAX     0x06

// Index op codes (task.op_code when task.op_type == OP_INDEX)
#define INDEX_GATHER        0x00
#define INDEX_SCATTER       0x01
#define INDEX_INDEX_SELECT  0x02
#define INDEX_INDEX_PUT     0x03

// Dtype codes
#define DTYPE_FP16    0x00
#define DTYPE_BF16    0x01
#define DTYPE_FP32    0x02
#define DTYPE_FP8E4M3 0x03
#define DTYPE_INT8    0x04
#define DTYPE_INT32   0x05
#define DTYPE_INT64   0x06
#define DTYPE_BOOL    0x07

// 80 bytes per task descriptor, tightly packed
struct __align__(16) TaskDesc {
    uint16_t op_type;             // Which task function to dispatch to
    uint16_t op_code;             // Sub-operation (e.g., ADD/MUL for OP_ELEMENTWISE)
    uint32_t num_tiles;           // How many SMs work on this task
    uint32_t buffer_indices[8];   // Input/output buffer IDs in the arena
                                  //   Convention: [0]=output, [1..]=inputs
                                  //   0xFFFFFFFF = unused slot
    uint32_t dimensions[8];       // M, N, K, head_dim, num_heads, reduce_dim, etc.
                                  //   Interpretation depends on op_type
    int32_t  strides[8];          // For non-contiguous inputs
                                  //   Stride changes from reshape/transpose
                                  //   stored here instead of emitting a task
};

// 24 bytes per buffer descriptor (18 bytes logical, padded to 24 for alignment)
struct __align__(8) BufferDesc {
    uint64_t offset;              // Byte offset in workspace arena
    uint64_t size;                // Size in bytes
    uint16_t dtype;               // DTYPE_* code
    uint16_t _pad0;
    uint32_t _pad1;
};

// 12 bytes per weight mapping (10 bytes logical, padded to 12)
struct __align__(4) WeightMapping {
    uint32_t buffer_index;        // Which buffer this weight occupies
    uint32_t key_offset;          // Byte offset into the string table
    uint16_t key_length;          // Length of the safetensors key (bytes, no null)
    uint16_t _pad;
};

// File header: 52 bytes
struct ScheduleHeader {
    uint32_t magic;               // 0x4D454741 ("MEGA")
    uint32_t version;             // Format version (currently 1)
    uint32_t num_tasks;           // Number of TaskDesc entries
    uint32_t num_buffers;         // Number of BufferDesc entries
    uint64_t workspace_bytes;     // Total arena allocation required
    uint32_t num_weight_mappings; // Number of WeightMapping entries
    uint32_t batch_min;           // Shape bucket: minimum batch size
    uint32_t batch_max;           // Shape bucket: maximum batch size
    uint32_t seq_min;             // Shape bucket: minimum sequence length
    uint32_t seq_max;             // Shape bucket: maximum sequence length
    uint32_t sm_version;          // Target SM architecture (e.g., 90 for H100)
    uint16_t compute_dtype;       // DTYPE_* code for compute precision
    uint16_t _padding;            // Align to 4-byte boundary
};
```

#### Python Dataclasses (schedule compiler side)

```python
from dataclasses import dataclass, field
import struct

@dataclass
class TaskDesc:
    op_type: int            # OP_* constant
    op_code: int = 0        # Sub-operation code
    num_tiles: int = 0      # SM count for this task
    buffer_indices: list[int] = field(
        default_factory=lambda: [0xFFFFFFFF] * 8
    )
    dimensions: list[int] = field(default_factory=lambda: [0] * 8)
    strides: list[int] = field(default_factory=lambda: [0] * 8)

    STRUCT_FORMAT = "<HHI 8I 8I 8i"  # 80 bytes
    STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # == 80

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.op_type, self.op_code, self.num_tiles,
            *self.buffer_indices,
            *self.dimensions,
            *self.strides,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TaskDesc":
        vals = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            op_type=vals[0], op_code=vals[1], num_tiles=vals[2],
            buffer_indices=list(vals[3:11]),
            dimensions=list(vals[11:19]),
            strides=list(vals[19:27]),
        )


@dataclass
class BufferDesc:
    offset: int     # Byte offset in workspace arena
    size: int       # Size in bytes
    dtype: int      # DTYPE_* code

    STRUCT_FORMAT = "<QQH2xI"  # 24 bytes (18 + 6 padding)
    STRUCT_SIZE = 24

    def to_bytes(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT,
                           self.offset, self.size, self.dtype, 0)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BufferDesc":
        offset, size, dtype, _ = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(offset=offset, size=size, dtype=dtype)


@dataclass
class WeightMapping:
    buffer_index: int   # Which buffer this weight occupies
    key_offset: int     # Byte offset into string table
    key_length: int     # Length of safetensors key

    STRUCT_FORMAT = "<IIH2x"  # 12 bytes (10 + 2 padding)
    STRUCT_SIZE = 12

    def to_bytes(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT,
                           self.buffer_index, self.key_offset, self.key_length)

    @classmethod
    def from_bytes(cls, data: bytes) -> "WeightMapping":
        buf_idx, key_off, key_len = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(buffer_index=buf_idx, key_offset=key_off, key_length=key_len)


@dataclass
class ScheduleHeader:
    magic: int = 0x4D454741
    version: int = 1
    num_tasks: int = 0
    num_buffers: int = 0
    workspace_bytes: int = 0
    num_weight_mappings: int = 0
    batch_min: int = 0
    batch_max: int = 0
    seq_min: int = 0
    seq_max: int = 0
    sm_version: int = 0
    compute_dtype: int = 0

    STRUCT_FORMAT = "<IIIIQ IIIII IH2x"  # 52 bytes
    STRUCT_SIZE = 52

    def to_bytes(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.magic, self.version, self.num_tasks, self.num_buffers,
            self.workspace_bytes, self.num_weight_mappings,
            self.batch_min, self.batch_max, self.seq_min, self.seq_max,
            self.sm_version, self.compute_dtype,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ScheduleHeader":
        vals = struct.unpack(cls.STRUCT_FORMAT, data)
        return cls(
            magic=vals[0], version=vals[1], num_tasks=vals[2],
            num_buffers=vals[3], workspace_bytes=vals[4],
            num_weight_mappings=vals[5],
            batch_min=vals[6], batch_max=vals[7],
            seq_min=vals[8], seq_max=vals[9],
            sm_version=vals[10], compute_dtype=vals[11],
        )
```

### 20.3 Schedule File Binary Format

The `.schedule` file is a flat binary blob. No headers-within-headers, no indirection tables. The reader walks forward sequentially.

```
Byte offset    Field                                Size (bytes)
-----------    -----                                ------------
[0..3]         magic: 0x4D454741 ("MEGA")           4
[4..7]         version: 1                           4
[8..11]        num_tasks (T)                        4
[12..15]       num_buffers (B)                      4
[16..23]       workspace_bytes                      8
[24..27]       num_weight_mappings (W)              4
[28..31]       batch_min                            4
[32..35]       batch_max                            4
[36..39]       seq_min                              4
[40..43]       seq_max                              4
[44..47]       sm_version                           4
[48..49]       compute_dtype                        2
[50..51]       padding (zero)                       2
[52..]         TaskDesc[T]       (each 80 bytes)    T * 80
[..]           BufferDesc[B]     (each 24 bytes)    B * 24
[..]           WeightMapping[W]  (each 12 bytes)    W * 12
[..]           String table (UTF-8, null-terminated weight names)
```

All integer values are **little-endian**. No pickle, no executable code, no pointers. The file is safe to mmap directly into GPU memory (after endian check on big-endian hosts, which are not a target).

Reading the file in Python:

```python
def load_schedule(path: str) -> tuple[ScheduleHeader, list[TaskDesc],
                                       list[BufferDesc], list[WeightMapping],
                                       dict[int, str]]:
    with open(path, "rb") as f:
        data = f.read()

    header = ScheduleHeader.from_bytes(data[:52])
    assert header.magic == 0x4D454741, "Not a megabake schedule file"
    assert header.version == 1, f"Unsupported version {header.version}"

    off = 52
    tasks = []
    for _ in range(header.num_tasks):
        tasks.append(TaskDesc.from_bytes(data[off : off + 80]))
        off += 80

    buffers = []
    for _ in range(header.num_buffers):
        buffers.append(BufferDesc.from_bytes(data[off : off + 24]))
        off += 24

    weight_maps = []
    for _ in range(header.num_weight_mappings):
        weight_maps.append(WeightMapping.from_bytes(data[off : off + 12]))
        off += 12

    # String table: remaining bytes, null-terminated strings
    string_table = data[off:]

    # Resolve weight names
    weights = {}
    for wm in weight_maps:
        name = string_table[wm.key_offset : wm.key_offset + wm.key_length]
        weights[wm.buffer_index] = name.decode("utf-8")

    return header, tasks, buffers, weight_maps, weights
```

### 20.4 Complete ATen Op Mapping Table

Every ATen op the schedule compiler must handle, organized by task type. This table lives in `op_table.py` and drives the FX graph walker.

```python
from enum import IntEnum

class OpType(IntEnum):
    MATMUL      = 0x01
    ATTENTION   = 0x02
    ELEMENTWISE = 0x03
    REDUCE      = 0x04
    EMBEDDING   = 0x05
    INDEX       = 0x06
    COPY        = 0x07
    ROPE        = 0x08

class ElemCode(IntEnum):
    ADD         = 0x00
    MUL         = 0x01
    SUB         = 0x02
    DIV         = 0x03
    SILU        = 0x04
    GELU        = 0x05
    RELU        = 0x06
    SIGMOID     = 0x07
    TANH        = 0x08
    EXP         = 0x09
    LOG         = 0x0A
    RSQRT       = 0x0B
    NEG         = 0x0C
    ABS         = 0x0D
    CLAMP       = 0x0E
    WHERE       = 0x0F
    CAST        = 0x10
    MASKED_FILL = 0x11
    POW         = 0x12

class ReduceCode(IntEnum):
    SUM       = 0x00
    MEAN      = 0x01
    MAX       = 0x02
    SOFTMAX   = 0x03
    RMSNORM   = 0x04
    LAYERNORM = 0x05
    ARGMAX    = 0x06

class IndexCode(IntEnum):
    GATHER       = 0x00
    SCATTER      = 0x01
    INDEX_SELECT = 0x02
    INDEX_PUT    = 0x03

# Maps ATen op target -> (OpType, op_code)
# "STRIDE_CHANGE" means zero-cost, no task emitted
# "FORCE_COPY" means OP_COPY is always emitted

import torch

ATEN_OP_MAP: dict[object, tuple[int, int] | str] = {
    # --- OP_MATMUL (0x01) ---
    torch.ops.aten.mm.default:             (OpType.MATMUL, 0),
    torch.ops.aten.addmm.default:          (OpType.MATMUL, 0),
    torch.ops.aten.bmm.default:            (OpType.MATMUL, 0),
    torch.ops.aten.linear.default:         (OpType.MATMUL, 0),
    torch.ops.aten._scaled_mm.default:     (OpType.MATMUL, 0),

    # --- OP_ATTENTION (0x02) ---
    torch.ops.aten.scaled_dot_product_attention.default: (OpType.ATTENTION, 0),

    # --- OP_ELEMENTWISE (0x03) ---
    torch.ops.aten.add.Tensor:             (OpType.ELEMENTWISE, ElemCode.ADD),
    torch.ops.aten.add.Scalar:             (OpType.ELEMENTWISE, ElemCode.ADD),
    torch.ops.aten.mul.Tensor:             (OpType.ELEMENTWISE, ElemCode.MUL),
    torch.ops.aten.mul.Scalar:             (OpType.ELEMENTWISE, ElemCode.MUL),
    torch.ops.aten.sub.Tensor:             (OpType.ELEMENTWISE, ElemCode.SUB),
    torch.ops.aten.div.Tensor:             (OpType.ELEMENTWISE, ElemCode.DIV),
    torch.ops.aten.silu.default:           (OpType.ELEMENTWISE, ElemCode.SILU),
    torch.ops.aten.gelu.default:           (OpType.ELEMENTWISE, ElemCode.GELU),
    torch.ops.aten.relu.default:           (OpType.ELEMENTWISE, ElemCode.RELU),
    torch.ops.aten.sigmoid.default:        (OpType.ELEMENTWISE, ElemCode.SIGMOID),
    torch.ops.aten.tanh.default:           (OpType.ELEMENTWISE, ElemCode.TANH),
    torch.ops.aten.exp.default:            (OpType.ELEMENTWISE, ElemCode.EXP),
    torch.ops.aten.log.default:            (OpType.ELEMENTWISE, ElemCode.LOG),
    torch.ops.aten.rsqrt.default:          (OpType.ELEMENTWISE, ElemCode.RSQRT),
    torch.ops.aten.neg.default:            (OpType.ELEMENTWISE, ElemCode.NEG),
    torch.ops.aten.abs.default:            (OpType.ELEMENTWISE, ElemCode.ABS),
    torch.ops.aten.clamp.default:          (OpType.ELEMENTWISE, ElemCode.CLAMP),
    torch.ops.aten.where.self:             (OpType.ELEMENTWISE, ElemCode.WHERE),
    torch.ops.aten.to.dtype:               (OpType.ELEMENTWISE, ElemCode.CAST),
    torch.ops.aten.masked_fill.Scalar:     (OpType.ELEMENTWISE, ElemCode.MASKED_FILL),
    torch.ops.aten.pow.Tensor_Scalar:      (OpType.ELEMENTWISE, ElemCode.POW),

    # --- OP_REDUCE (0x04) ---
    torch.ops.aten.sum.dim_IntList:        (OpType.REDUCE, ReduceCode.SUM),
    torch.ops.aten.mean.dim:               (OpType.REDUCE, ReduceCode.MEAN),
    torch.ops.aten.max.dim:                (OpType.REDUCE, ReduceCode.MAX),
    torch.ops.aten.softmax.int:            (OpType.REDUCE, ReduceCode.SOFTMAX),
    torch.ops.aten._softmax.default:       (OpType.REDUCE, ReduceCode.SOFTMAX),
    # RMSNORM: detected via pattern matching, not a single ATen op
    torch.ops.aten.layer_norm.default:     (OpType.REDUCE, ReduceCode.LAYERNORM),
    torch.ops.aten.native_layer_norm.default: (OpType.REDUCE, ReduceCode.LAYERNORM),
    torch.ops.aten.argmax.default:         (OpType.REDUCE, ReduceCode.ARGMAX),

    # --- OP_EMBEDDING (0x05) ---
    torch.ops.aten.embedding.default:      (OpType.EMBEDDING, 0),

    # --- OP_INDEX (0x06) ---
    torch.ops.aten.gather.default:         (OpType.INDEX, IndexCode.GATHER),
    torch.ops.aten.scatter.value:          (OpType.INDEX, IndexCode.SCATTER),
    torch.ops.aten.scatter.src:            (OpType.INDEX, IndexCode.SCATTER),
    torch.ops.aten.index_select.default:   (OpType.INDEX, IndexCode.INDEX_SELECT),
    torch.ops.aten.index_put_.default:     (OpType.INDEX, IndexCode.INDEX_PUT),

    # --- OP_COPY (0x07) - always emitted ---
    torch.ops.aten.cat.default:            (OpType.COPY, 0),
    torch.ops.aten.stack.default:          (OpType.COPY, 0),

    # --- OP_ROPE (0x08) ---
    # Detected via pattern matching: cos*x + sin*rotate_half(x)

    # --- ZERO-COST STRIDE CHANGES (no task emitted) ---
    torch.ops.aten.reshape.default:        "STRIDE_CHANGE",
    torch.ops.aten.view.default:           "STRIDE_CHANGE",
    torch.ops.aten.transpose.int:          "STRIDE_CHANGE",
    torch.ops.aten.permute.default:        "STRIDE_CHANGE",
    torch.ops.aten.expand.default:         "STRIDE_CHANGE",
    torch.ops.aten.t.default:              "STRIDE_CHANGE",
    torch.ops.aten.unsqueeze.default:      "STRIDE_CHANGE",
    torch.ops.aten.squeeze.default:        "STRIDE_CHANGE",
    torch.ops.aten.squeeze.dim:            "STRIDE_CHANGE",
    torch.ops.aten.slice.Tensor:           "STRIDE_CHANGE",  # may become COPY
    torch.ops.aten.split.Tensor:           "STRIDE_CHANGE",
    torch.ops.aten.unbind.int:             "STRIDE_CHANGE",

    # --- OPS THAT FORCE A COPY ---
    # aten.contiguous emits OP_COPY only if input strides are non-contiguous
    # aten.cat / aten.stack always emit OP_COPY (listed above)
}
```

**RMSNorm pattern matching:** RMSNorm decomposes into `rsqrt(mean(x^2) + eps) * x * weight`. The schedule compiler matches this pattern across consecutive FX nodes:

```python
def match_rmsnorm(node) -> bool:
    """Detect the RMSNorm pattern: mul(mul(rsqrt(add(mean(pow(x,2)), eps)), x), weight)"""
    if not _is_op(node, torch.ops.aten.mul.Tensor):
        return False
    lhs, rhs = node.args[0], node.args[1]
    # One branch should be rsqrt(add(mean(pow(x,2)), eps))
    # The other should be x (the original input, possibly through a view)
    # Walk the chain and verify the structure
    rsqrt_node = _find_in_args(lhs, rhs, torch.ops.aten.rsqrt.default)
    if rsqrt_node is None:
        return False
    add_node = rsqrt_node.args[0]
    if not _is_op(add_node, torch.ops.aten.add.Tensor):
        return False
    mean_node = add_node.args[0]
    if not _is_op(mean_node, torch.ops.aten.mean.dim):
        return False
    pow_node = mean_node.args[0]
    if not _is_op(pow_node, torch.ops.aten.pow.Tensor_Scalar):
        return False
    if pow_node.args[1] != 2:
        return False
    return True
```

When the pattern is matched, the entire subgraph is replaced by a single `OP_REDUCE` task with `op_code = REDUCE_RMSNORM`. The matched nodes are consumed and not emitted individually.

### 20.5 BSP Scheduler CUDA Code

The complete scheduler, defined in `scheduler.cu`. This is the core of the megakernel.

```cuda
// scheduler.cu
#include <cooperative_groups.h>
#include "data_types.cuh"

// Forward declarations for all task __device__ functions.
// Each is implemented in its own .cu file under tasks/.
__device__ void task_matmul(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id);
__device__ void task_attention(const TaskDesc& task, void** buffers,
                               const int* dyn_dims, int tile_id);
__device__ void task_elementwise(const TaskDesc& task, void** buffers,
                                 const int* dyn_dims, int tile_id);
__device__ void task_reduce(const TaskDesc& task, void** buffers,
                             const int* dyn_dims, int tile_id);
__device__ void task_embedding(const TaskDesc& task, void** buffers,
                                const int* dyn_dims, int tile_id);
__device__ void task_index(const TaskDesc& task, void** buffers,
                            const int* dyn_dims, int tile_id);
__device__ void task_copy(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id);
__device__ void task_rope(const TaskDesc& task, void** buffers,
                           const int* dyn_dims, int tile_id);

// Dispatch a single task to the correct handler.
__device__ __forceinline__ void dispatch_task(
    const TaskDesc& task, void** buffers, const int* dyn_dims, int tile_id
) {
    switch (task.op_type) {
        case OP_MATMUL:      task_matmul(task, buffers, dyn_dims, tile_id);      break;
        case OP_ATTENTION:   task_attention(task, buffers, dyn_dims, tile_id);   break;
        case OP_ELEMENTWISE: task_elementwise(task, buffers, dyn_dims, tile_id); break;
        case OP_REDUCE:      task_reduce(task, buffers, dyn_dims, tile_id);      break;
        case OP_EMBEDDING:   task_embedding(task, buffers, dyn_dims, tile_id);   break;
        case OP_INDEX:       task_index(task, buffers, dyn_dims, tile_id);       break;
        case OP_COPY:        task_copy(task, buffers, dyn_dims, tile_id);        break;
        case OP_ROPE:        task_rope(task, buffers, dyn_dims, tile_id);        break;
    }
}

// The megakernel entry point.
//
// Launched via cudaLaunchCooperativeKernel with one block per SM.
// All SMs execute the same task (different tiles), then barrier, then next task.
//
// __launch_bounds__(1024, 1): 1024 threads per block, 1 block per SM.
// This gives each block maximum registers and shared memory (PERKS).
__global__ void __launch_bounds__(1024, 1) megakernel(
    const TaskDesc* __restrict__ tasks,   // Task schedule array
    int num_tasks,                         // Length of tasks array
    void** __restrict__ buffers,           // Buffer pointer table
    const int* __restrict__ dyn_dims       // Dynamic dimensions (batch, seq_len, etc.)
) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    const int sm_id = blockIdx.x;  // One block per SM, so blockIdx.x == SM index

    for (int i = 0; i < num_tasks; i++) {
        const TaskDesc& task = tasks[i];

        // Only SMs within this task's tile range do work.
        // Idle SMs skip to the barrier.
        if (sm_id < static_cast<int>(task.num_tiles)) {
            dispatch_task(task, buffers, dyn_dims, sm_id);
        }

        // Grid-wide barrier: all SMs wait here.
        // This is the BSP synchronization point -- task i's outputs
        // are visible to all SMs before task i+1 begins.
        grid.sync();
    }
}
```

**Launch configuration (host side):**

```cpp
#include <cuda_runtime.h>
#include <cstdio>

void launch_megakernel(
    const TaskDesc* d_tasks, int num_tasks,
    void** d_buffers, const int* d_dyn_dims,
    int num_sms, cudaStream_t stream
) {
    // Set maximum dynamic shared memory for this kernel
    cudaFuncSetAttribute(
        megakernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        228 * 1024  // 228 KB on Hopper (cudaDevAttrMaxSharedMemoryPerBlockOptin)
    );

    // Cooperative launch configuration
    void* args[] = {
        (void*)&d_tasks,
        (void*)&num_tasks,
        (void*)&d_buffers,
        (void*)&d_dyn_dims,
    };

    dim3 grid(num_sms);      // One block per SM
    dim3 block(1024);        // 1024 threads per block (32 warps)

    cudaLaunchCooperativeKernel(
        (void*)megakernel,
        grid, block,
        args,
        228 * 1024,          // Dynamic shared memory per block
        stream
    );
}
```

**Key details:**

- `cudaLaunchCooperativeKernel` is required for `cooperative_groups::this_grid().sync()`. Standard `<<<>>>` launch cannot do grid-wide sync.
- One block per SM: query `cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device)` to get the SM count. H100 SXM has 132 SMs; A100 SXM has 108.
- `__launch_bounds__(1024, 1)` tells the compiler: max 1024 threads, max 1 block per SM. This allows the compiler to use up to 255 registers per thread (1024 threads * 255 regs = 261120, within the 65536 per-SM register file... note: the compiler will find the actual optimal allocation).
- Dynamic shared memory is set to the device maximum via `cudaFuncSetAttribute`. On Hopper this is 228 KB; on Ampere it is 164 KB (query `cudaDevAttrMaxSharedMemoryPerBlockOptin`).

### 20.6 Buffer Planning Algorithm

The buffer planner assigns byte offsets within a single contiguous workspace arena. Weight buffers are excluded from the arena -- they point directly into safetensors mmap memory.

```python
from dataclasses import dataclass

@dataclass
class LiveInterval:
    buffer_id: int
    size: int       # bytes
    first_use: int  # task index of first read or write
    last_use: int   # task index of last read

@dataclass
class Placement:
    buffer_id: int
    offset: int     # byte offset in arena
    size: int


def plan_buffers(
    tasks: list[TaskDesc],
    buffer_sizes: dict[int, int],   # buffer_id -> size in bytes
    weight_buffers: set[int],       # buffer IDs that are weights (excluded)
) -> tuple[list[Placement], int]:
    """
    Assign byte offsets to activation buffers within the workspace arena.
    Returns (placements, total_workspace_bytes).
    """

    # Step 1: Compute live intervals
    intervals: dict[int, LiveInterval] = {}

    for task_idx, task in enumerate(tasks):
        for slot in range(8):
            buf_id = task.buffer_indices[slot]
            if buf_id == 0xFFFFFFFF or buf_id in weight_buffers:
                continue
            if buf_id not in intervals:
                intervals[buf_id] = LiveInterval(
                    buffer_id=buf_id,
                    size=buffer_sizes[buf_id],
                    first_use=task_idx,
                    last_use=task_idx,
                )
            else:
                intervals[buf_id].last_use = task_idx

    # Step 2: Sort by size descending (large buffers first = better packing)
    sorted_intervals = sorted(intervals.values(), key=lambda iv: -iv.size)

    # Step 3: Greedy first-fit placement
    placements: list[Placement] = []

    def overlaps_any(candidate_offset: int, candidate_size: int,
                     candidate_interval: LiveInterval) -> bool:
        """Check if placing this buffer here conflicts with any already-placed buffer."""
        for p in placements:
            existing_iv = intervals[p.buffer_id]
            # Two buffers conflict if their arena ranges overlap AND
            # their live intervals overlap (they are alive at the same time)
            arena_overlap = (
                candidate_offset < p.offset + p.size
                and p.offset < candidate_offset + candidate_size
            )
            time_overlap = (
                candidate_interval.first_use <= existing_iv.last_use
                and existing_iv.first_use <= candidate_interval.last_use
            )
            if arena_overlap and time_overlap:
                return True
        return False

    for iv in sorted_intervals:
        # Try offset 0 first, then bump past each conflicting placement
        offset = 0
        while overlaps_any(offset, iv.size, iv):
            # Find the next candidate offset: jump past whichever placed
            # buffer ends latest at the current position
            best_jump = offset + 1
            for p in placements:
                existing_iv = intervals[p.buffer_id]
                arena_overlap = (offset < p.offset + p.size
                                 and p.offset < offset + iv.size)
                time_overlap = (iv.first_use <= existing_iv.last_use
                                and existing_iv.first_use <= iv.last_use)
                if arena_overlap and time_overlap:
                    best_jump = max(best_jump, p.offset + p.size)
            # Align to 256-byte boundary for GPU memory coalescing
            offset = (best_jump + 255) & ~255

        placements.append(Placement(buffer_id=iv.buffer_id,
                                     offset=offset, size=iv.size))

    # Step 4: Compute total workspace
    total = 0
    for p in placements:
        total = max(total, p.offset + p.size)

    # Align total to 4KB page boundary
    total = (total + 4095) & ~4095

    return placements, total
```

**Important invariants:**
- Weight buffers (parameters, embedding tables) are never placed in the arena. At runtime, the weight mapper sets their buffer pointers to the safetensors mmap addresses directly.
- The input buffer (token IDs) is placed at offset 0 by convention. The output buffer (logits) is placed at a known offset recorded in the schedule header.
- All offsets are 256-byte aligned for GPU memory coalescing.
- The `total_workspace_bytes` value is written into `ScheduleHeader.workspace_bytes`. The runtime allocates exactly this many bytes via `cudaMalloc`.

### 20.7 Shape Op Resolution Rules

Shape/stride operations produce no GPU work. Instead, the schedule compiler tracks a virtual stride table and adjusts it as shape ops are encountered in the FX graph. The computed strides are baked into the downstream task's `TaskDesc.strides` field.

```python
from dataclasses import dataclass

@dataclass
class StridedView:
    """Virtual view of a buffer: shape + strides + byte offset into the buffer."""
    buffer_id: int
    shape: list[int]
    strides: list[int]   # in elements, not bytes
    offset: int = 0      # element offset from buffer start

    def is_contiguous(self) -> bool:
        """Check if strides match the C-contiguous layout for the given shape."""
        expected_stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != expected_stride and self.shape[i] != 1:
                return False
            expected_stride *= self.shape[i]
        return True


def resolve_reshape(view: StridedView, new_shape: list[int]) -> StridedView | None:
    """
    Compute strides for reshape/view. Returns a new StridedView if the reshape
    can be done as a zero-cost stride change, or None if a copy is needed.
    """
    if not view.is_contiguous():
        return None  # Non-contiguous input requires OP_COPY first
    # Contiguous reshape: recompute strides from new shape
    new_strides = [0] * len(new_shape)
    stride = 1
    for i in range(len(new_shape) - 1, -1, -1):
        new_strides[i] = stride
        stride *= new_shape[i]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_transpose(view: StridedView, dim0: int, dim1: int) -> StridedView:
    """Transpose: swap strides and shape entries for dim0 and dim1."""
    new_shape = list(view.shape)
    new_strides = list(view.strides)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
    new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_permute(view: StridedView, dims: list[int]) -> StridedView:
    """Permute: reorder shape and strides according to dims."""
    new_shape = [view.shape[d] for d in dims]
    new_strides = [view.strides[d] for d in dims]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_expand(view: StridedView, new_shape: list[int]) -> StridedView:
    """Expand: set stride to 0 for expanded dimensions."""
    new_strides = list(view.strides)
    for i in range(len(new_shape)):
        if i >= len(view.shape) or view.shape[i] == 1:
            if new_shape[i] != 1:
                new_strides[i] = 0  # Broadcast dimension
    return StridedView(view.buffer_id, list(new_shape), new_strides, view.offset)


def resolve_squeeze(view: StridedView, dim: int | None) -> StridedView:
    """Squeeze: remove dimensions of size 1."""
    if dim is not None:
        if view.shape[dim] != 1:
            return view  # No-op if dimension is not 1
        new_shape = view.shape[:dim] + view.shape[dim + 1 :]
        new_strides = view.strides[:dim] + view.strides[dim + 1 :]
    else:
        new_shape = [s for s in view.shape if s != 1]
        new_strides = [view.strides[i] for i, s in enumerate(view.shape) if s != 1]
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_unsqueeze(view: StridedView, dim: int) -> StridedView:
    """Unsqueeze: insert a dimension of size 1."""
    new_shape = list(view.shape)
    new_strides = list(view.strides)
    # Stride for the new dim: product of shape[dim:] * strides[dim] would
    # give correct value, but for a size-1 dim the stride is irrelevant
    # for data layout. Use the stride of the next inner dimension.
    insert_stride = view.strides[dim] * view.shape[dim] if dim < len(view.shape) else 1
    new_shape.insert(dim, 1)
    new_strides.insert(dim, insert_stride)
    return StridedView(view.buffer_id, new_shape, new_strides, view.offset)


def resolve_slice(view: StridedView, dim: int, start: int,
                  end: int, step: int = 1) -> StridedView:
    """Slice: adjust offset and shape, keep strides (if step==1)."""
    assert step == 1, "Strided slice (step != 1) requires OP_COPY"
    new_shape = list(view.shape)
    new_shape[dim] = end - start
    new_offset = view.offset + start * view.strides[dim]
    return StridedView(view.buffer_id, new_shape, list(view.strides), new_offset)


def resolve_t(view: StridedView) -> StridedView:
    """Matrix transpose (.t()): swap the last two dimensions."""
    assert len(view.shape) == 2, ".t() only valid for 2D tensors"
    return resolve_transpose(view, 0, 1)
```

**When a copy is forced:**

The schedule compiler checks whether the next consuming op requires contiguous input. If the current `StridedView` is non-contiguous and the consumer requires contiguity, an `OP_COPY` task is emitted to materialize the data in a new contiguous buffer.

Ops that require contiguous input:
- `OP_MATMUL` (CUTLASS expects contiguous or specific stride patterns)
- `OP_ATTENTION` (expects contiguous Q, K, V)
- `OP_REDUCE` when reducing along a non-innermost dimension

Ops that tolerate non-contiguous input (they use strides directly):
- `OP_ELEMENTWISE` (all sub-ops)
- `OP_EMBEDDING` (lookup is gather, stride-agnostic)

### 20.8 torch.export Integration

The schedule compiler's main entry point. This drives the full pipeline from a PyTorch model to a `.schedule` file.

```python
import torch
from torch.export import export
from torch.export import default_decompositions

from megabake.schedule_compiler.op_table import ATEN_OP_MAP, OpType
from megabake.schedule_compiler.shape_ops import (
    StridedView, resolve_reshape, resolve_transpose,
    resolve_permute, resolve_expand, resolve_squeeze,
    resolve_unsqueeze, resolve_slice, resolve_t,
)
from megabake.schedule_compiler.buffer_planner import plan_buffers, Placement
from megabake.schedule_compiler.serializer import write_schedule
# TaskDesc, BufferDesc, etc. from data structures module


# Shape ops that get resolved via stride manipulation, no task emitted
SHAPE_OPS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.t.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.unbind.int,
}


def compile_schedule(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    sm_version: int,
    dtype: torch.dtype = torch.float16,
    batch_range: tuple[int, int] = (1, 1),
    seq_range: tuple[int, int] = (1, 2048),
) -> bytes:
    """
    Compile a torch.nn.Module into a megabake schedule.

    Args:
        model: The PyTorch model (e.g., from HuggingFace transformers)
        example_input: A concrete input tensor for tracing
        sm_version: Target SM architecture (e.g., 90 for H100)
        dtype: Compute dtype (torch.float16 or torch.bfloat16)
        batch_range: (min, max) batch size for this schedule bucket
        seq_range: (min, max) sequence length for this schedule bucket

    Returns:
        Serialized schedule as bytes (write to .schedule file)
    """
    # ----------------------------------------------------------------
    # Phase 1: Export with SDPA preserved as a single node
    # ----------------------------------------------------------------
    decomp_table = default_decompositions()
    # Remove SDPA decompositions so it stays as one node -> OP_ATTENTION
    preserve_ops = [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
    ]
    for op in preserve_ops:
        decomp_table.pop(op, None)

    ep = export(model, (example_input,), strict=False)
    ep = ep.run_decompositions(decomp_table)

    graph = ep.graph_module.graph

    # ----------------------------------------------------------------
    # Phase 2: Walk FX graph, build task list
    # ----------------------------------------------------------------
    tasks: list[TaskDesc] = []
    buffer_map: dict[str, int] = {}       # FX node name -> buffer_id
    buffer_sizes: dict[int, int] = {}     # buffer_id -> size in bytes
    weight_buffers: set[int] = set()      # buffer IDs that are model weights
    weight_names: dict[int, str] = {}     # buffer_id -> state_dict key
    view_map: dict[str, StridedView] = {} # FX node name -> current strided view
    next_buffer_id = 0

    def alloc_buffer(node, shape: list[int], dt: torch.dtype) -> int:
        nonlocal next_buffer_id
        buf_id = next_buffer_id
        next_buffer_id += 1
        buffer_map[node.name] = buf_id
        buffer_sizes[buf_id] = _numel(shape) * _dtype_bytes(dt)
        # Initialize contiguous stride view
        strides = _contiguous_strides(shape)
        view_map[node.name] = StridedView(buf_id, shape, strides)
        return buf_id

    unsupported_nodes: list = []

    for node in graph.nodes:
        if node.op == "placeholder":
            # Input tensor or lifted parameter
            meta = node.meta.get("val")
            if meta is None:
                continue
            shape = list(meta.shape)
            dt = meta.dtype
            buf_id = alloc_buffer(node, shape, dt)

            # Check if this is a lifted parameter (weight)
            param_name = _get_param_name(ep, node)
            if param_name is not None:
                weight_buffers.add(buf_id)
                weight_names[buf_id] = param_name

        elif node.op == "call_function":
            target = node.target

            # Check op table
            mapping = ATEN_OP_MAP.get(target)

            if mapping == "STRIDE_CHANGE":
                # Zero-cost: compute new strides, no task emitted
                input_view = view_map[node.args[0].name]
                new_view = _resolve_shape_op(target, input_view, node.args)
                if new_view is None:
                    # Reshape of non-contiguous requires a copy
                    copy_buf = alloc_buffer(node, list(node.meta["val"].shape),
                                            node.meta["val"].dtype)
                    tasks.append(_make_copy_task(
                        input_view.buffer_id, copy_buf,
                        _numel(input_view.shape)))
                    view_map[node.name] = StridedView(
                        copy_buf,
                        list(node.meta["val"].shape),
                        _contiguous_strides(list(node.meta["val"].shape)),
                    )
                else:
                    view_map[node.name] = new_view
                    buffer_map[node.name] = new_view.buffer_id

            elif mapping is None:
                # Unsupported op -> graph split point
                unsupported_nodes.append(node)

            elif isinstance(mapping, tuple):
                op_type, op_code = mapping

                # Allocate output buffer
                out_meta = node.meta.get("val")
                if out_meta is None:
                    continue
                out_shape = list(out_meta.shape)
                out_buf = alloc_buffer(node, out_shape, out_meta.dtype)

                # Build task descriptor
                task = TaskDesc(op_type=op_type, op_code=op_code)
                task.buffer_indices[0] = out_buf  # output

                # Fill input buffer indices from args
                for slot, arg in enumerate(node.args):
                    if hasattr(arg, "name") and arg.name in buffer_map:
                        task.buffer_indices[slot + 1] = buffer_map[arg.name]
                    if slot + 1 >= 7:
                        break

                # Fill dimensions from metadata
                task.dimensions = _extract_dimensions(op_type, node, out_shape)

                # Fill strides if any input is non-contiguous
                for slot, arg in enumerate(node.args):
                    if hasattr(arg, "name") and arg.name in view_map:
                        v = view_map[arg.name]
                        if not v.is_contiguous():
                            # Pack strides into the strides array
                            for j, s in enumerate(v.strides[:8]):
                                task.strides[j] = s

                # Compute tile count for this task
                task.num_tiles = _compute_tiles(op_type, task.dimensions,
                                                 sm_version)

                tasks.append(task)

        elif node.op == "output":
            pass  # Output node handled by buffer_map

    # ----------------------------------------------------------------
    # Phase 3: RMSNorm pattern matching (post-pass)
    # ----------------------------------------------------------------
    tasks = _fuse_rmsnorm_patterns(tasks, graph)

    # ----------------------------------------------------------------
    # Phase 4: Buffer planning
    # ----------------------------------------------------------------
    placements, total_workspace = plan_buffers(tasks, buffer_sizes,
                                                weight_buffers)

    # ----------------------------------------------------------------
    # Phase 5: Serialize
    # ----------------------------------------------------------------
    dtype_code = {torch.float16: 0x00, torch.bfloat16: 0x01,
                  torch.float32: 0x02}[dtype]

    return write_schedule(
        tasks=tasks,
        placements=placements,
        weight_names=weight_names,
        workspace_bytes=total_workspace,
        sm_version=sm_version,
        compute_dtype=dtype_code,
        batch_range=batch_range,
        seq_range=seq_range,
    )


# ---- Helper functions ----

def _numel(shape: list[int]) -> int:
    result = 1
    for s in shape:
        result *= s
    return result

def _dtype_bytes(dt: torch.dtype) -> int:
    return {
        torch.float16: 2, torch.bfloat16: 2, torch.float32: 4,
        torch.float8_e4m3fn: 1, torch.int8: 1, torch.int32: 4,
        torch.int64: 8, torch.bool: 1,
    }[dt]

def _contiguous_strides(shape: list[int]) -> list[int]:
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides

def _get_param_name(ep, node) -> str | None:
    """Get the state_dict key for a lifted parameter, or None."""
    # In torch.export, lifted parameters appear as placeholders
    # with entries in ep.graph_signature.inputs_to_parameters
    sig = ep.graph_signature
    if hasattr(sig, "inputs_to_parameters"):
        return sig.inputs_to_parameters.get(node.name)
    return None

def _resolve_shape_op(target, view: StridedView, args) -> StridedView | None:
    """Dispatch to the appropriate shape resolution function."""
    if target in (torch.ops.aten.reshape.default, torch.ops.aten.view.default):
        return resolve_reshape(view, list(args[1]))
    elif target == torch.ops.aten.transpose.int:
        return resolve_transpose(view, args[1], args[2])
    elif target == torch.ops.aten.permute.default:
        return resolve_permute(view, list(args[1]))
    elif target == torch.ops.aten.expand.default:
        return resolve_expand(view, list(args[1]))
    elif target == torch.ops.aten.t.default:
        return resolve_t(view)
    elif target == torch.ops.aten.unsqueeze.default:
        return resolve_unsqueeze(view, args[1])
    elif target in (torch.ops.aten.squeeze.default, torch.ops.aten.squeeze.dim):
        dim = args[1] if len(args) > 1 else None
        return resolve_squeeze(view, dim)
    elif target == torch.ops.aten.slice.Tensor:
        dim = args[1] if len(args) > 1 else 0
        start = args[2] if len(args) > 2 else 0
        end = args[3] if len(args) > 3 else view.shape[dim]
        return resolve_slice(view, dim, start, end)
    elif target in (torch.ops.aten.split.Tensor, torch.ops.aten.unbind.int):
        # These produce multiple outputs; handled separately
        return view  # Pass through, consumers use slice
    return view

def _make_copy_task(src_buf: int, dst_buf: int, num_elements: int) -> TaskDesc:
    """Create an OP_COPY task."""
    task = TaskDesc(op_type=OpType.COPY, op_code=0)
    task.buffer_indices[0] = dst_buf
    task.buffer_indices[1] = src_buf
    task.dimensions[0] = num_elements
    task.num_tiles = max(1, num_elements // (1024 * 256))  # ~256K elements per SM
    return task

def _extract_dimensions(op_type: int, node, out_shape: list[int]) -> list[int]:
    """
    Extract dimensions for a task from the FX node metadata.
    Returns a list of 8 uint32 values.
    """
    dims = [0] * 8
    if op_type == OpType.MATMUL:
        # dims: [M, N, K, batch, ...]
        args = node.args
        a_shape = list(args[0].meta["val"].shape)
        b_shape = list(args[1].meta["val"].shape)
        dims[0] = a_shape[-2]          # M
        dims[1] = b_shape[-1]          # N
        dims[2] = a_shape[-1]          # K
        if len(a_shape) > 2:
            dims[3] = _numel(a_shape[:-2])  # batch
    elif op_type == OpType.ATTENTION:
        q_shape = list(node.args[0].meta["val"].shape)
        dims[0] = q_shape[0]           # batch
        dims[1] = q_shape[1]           # num_heads
        dims[2] = q_shape[2]           # seq_len_q
        dims[3] = q_shape[3]           # head_dim
        k_shape = list(node.args[1].meta["val"].shape)
        dims[4] = k_shape[2]           # seq_len_k
    elif op_type == OpType.ELEMENTWISE:
        dims[0] = _numel(out_shape)    # total elements
    elif op_type == OpType.REDUCE:
        dims[0] = _numel(out_shape)    # output elements
        # dims[1] = reduction dimension size (from input)
        in_shape = list(node.args[0].meta["val"].shape)
        dims[1] = _numel(in_shape)     # input elements
        # dims[2] = reduction axis
        if len(node.args) > 1 and isinstance(node.args[1], (list, int)):
            axis = node.args[1] if isinstance(node.args[1], int) else node.args[1][0]
            dims[2] = axis
    elif op_type == OpType.EMBEDDING:
        # dims: [num_indices, embed_dim, vocab_size]
        dims[0] = _numel(out_shape) // out_shape[-1]  # num indices
        dims[1] = out_shape[-1]                         # embed_dim
        table_shape = list(node.args[0].meta["val"].shape)
        dims[2] = table_shape[0]                        # vocab_size
    elif op_type == OpType.COPY:
        dims[0] = _numel(out_shape)
    elif op_type == OpType.ROPE:
        dims[0] = out_shape[0]  # batch
        dims[1] = out_shape[1]  # seq_len
        dims[2] = out_shape[2]  # num_heads
        dims[3] = out_shape[3]  # head_dim
    return dims

def _compute_tiles(op_type: int, dims: list[int], sm_version: int) -> int:
    """Compute the number of SM tiles for a task."""
    max_sms = {80: 108, 90: 132, 100: 144}.get(sm_version, 132)
    if op_type == OpType.MATMUL:
        M, N, K = dims[0], dims[1], dims[2]
        # Tile M and N: 128x128 tiles
        tiles_m = (M + 127) // 128
        tiles_n = (N + 127) // 128
        return min(tiles_m * tiles_n, max_sms)
    elif op_type == OpType.ATTENTION:
        batch, num_heads = dims[0], dims[1]
        return min(batch * num_heads, max_sms)
    elif op_type == OpType.ELEMENTWISE:
        total = dims[0]
        # ~4096 elements per SM (1024 threads * 4 elements each)
        return min((total + 4095) // 4096, max_sms)
    elif op_type == OpType.REDUCE:
        # One SM per output row (for row-wise reductions like RMSNorm)
        return min(dims[0], max_sms)
    elif op_type == OpType.EMBEDDING:
        return min(dims[0], max_sms)  # One SM per index
    elif op_type == OpType.COPY:
        total = dims[0]
        return min((total + 4095) // 4096, max_sms)
    elif op_type == OpType.ROPE:
        batch, seq = dims[0], dims[1]
        return min(batch * seq, max_sms)
    return min(32, max_sms)
```

### 20.9 Build System

#### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.24)
project(megabake_cuda LANGUAGES CXX CUDA)

# Require CUDA 12.0+ for cooperative groups and Device LTO
find_package(CUDAToolkit 12.0 REQUIRED)

# CUTLASS (for matmul task)
# Set CUTLASS_DIR to your CUTLASS installation or use FetchContent
if(NOT DEFINED CUTLASS_DIR)
    include(FetchContent)
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG        v3.7.0
    )
    FetchContent_MakeAvailable(cutlass)
    set(CUTLASS_DIR ${cutlass_SOURCE_DIR})
endif()

# SM architectures to build for
set(MEGABAKE_SM_ARCHS "80;90;100" CACHE STRING "SM architectures to compile for")

# Task source files -- each compiled separately for Device LTO
set(TASK_SOURCES
    src/cuda/tasks/matmul.cu
    src/cuda/tasks/attention.cu
    src/cuda/tasks/elementwise.cu
    src/cuda/tasks/reduce.cu
    src/cuda/tasks/embedding.cu
    src/cuda/tasks/index.cu
    src/cuda/tasks/copy.cu
    src/cuda/tasks/rope.cu
)

# Main kernel source
set(KERNEL_SOURCES
    src/cuda/scheduler.cu
    src/cuda/megakernel.cu
)

# Build a cubin for each SM architecture
foreach(SM_ARCH ${MEGABAKE_SM_ARCHS})
    set(TARGET_NAME megabake_sm_${SM_ARCH})

    add_library(${TARGET_NAME} OBJECT
        ${TASK_SOURCES}
        ${KERNEL_SOURCES}
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        src/cuda
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/tools/util/include
    )

    set_target_properties(${TARGET_NAME} PROPERTIES
        CUDA_ARCHITECTURES ${SM_ARCH}
        CUDA_SEPARABLE_COMPILATION ON      # Required for Device LTO
    )

    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -dlto                          # Device Link-Time Optimization
            --use_fast_math                # Fast math for performance
            --maxrregcount=255             # Allow maximum registers per thread
            -Xptxas -v                     # Print register usage during compilation
            --expt-relaxed-constexpr       # Allow constexpr in device code
        >
    )

    # Link into a single cubin with Device LTO
    add_executable(megabake_cubin_${SM_ARCH} $<TARGET_OBJECTS:${TARGET_NAME}>)
    set_target_properties(megabake_cubin_${SM_ARCH} PROPERTIES
        CUDA_ARCHITECTURES ${SM_ARCH}
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        SUFFIX ".cubin"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cubin"
    )

    target_link_options(megabake_cubin_${SM_ARCH} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -dlto                          # Device LTO at link time
        >
    )

    target_link_libraries(megabake_cubin_${SM_ARCH} PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )
endforeach()
```

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "megabake"
version = "0.1.0"
description = "torch.export to megakernel: compile once, distribute everywhere"
readme = "README.md"
license = {text = "BSD-3-Clause"}
requires-python = ">=3.10"
dependencies = [
    "torch>=2.4",
    "safetensors>=0.4",
    "huggingface_hub>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "transformers>=4.40",
]

[project.scripts]
megabake = "megabake.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

### 20.10 Testing Strategy

#### Level 1: Per-Task Unit Tests

Each task type is tested in isolation by launching a minimal megakernel with a single task and comparing against PyTorch reference output. A helper harness wraps the launch:

```python
import torch
import ctypes

def run_single_task(
    task: TaskDesc,
    input_tensors: list[torch.Tensor],
    output_shape: list[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Launch the megakernel with a single task, return the output tensor.
    Used for per-task unit testing.
    """
    output = torch.empty(output_shape, dtype=output_dtype, device="cuda")

    # Build buffer pointer table: [output, input0, input1, ...]
    buffer_ptrs = [output.data_ptr()] + [t.data_ptr() for t in input_tensors]
    # Pad to 8 entries
    while len(buffer_ptrs) < 8:
        buffer_ptrs.append(0)

    # Set buffer_indices to identity mapping for this test
    task.buffer_indices = list(range(len(buffer_ptrs)))
    while len(task.buffer_indices) < 8:
        task.buffer_indices.append(0xFFFFFFFF)

    # Upload task + buffer pointers to GPU and launch
    _launch_single_task(task, buffer_ptrs, output)
    return output
```

**Test cases per task type:**

```python
# test_tasks/test_matmul.py
class TestMatmulTask(TestCase):
    def test_square(self):
        A = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        B = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        ref = A @ B
        task = TaskDesc(op_type=OpType.MATMUL, dimensions=[4096, 4096, 4096] + [0]*5)
        result = run_single_task(task, [A, B], [4096, 4096], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-3, rtol=1e-3))

    def test_rectangular(self):
        for M, N, K in [(1, 4096, 4096), (128, 11008, 4096), (32, 4096, 11008)]:
            A = torch.randn(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(K, N, device="cuda", dtype=torch.float16)
            ref = A @ B
            task = TaskDesc(op_type=OpType.MATMUL, dimensions=[M, N, K] + [0]*5)
            result = run_single_task(task, [A, B], [M, N], torch.float16)
            self.assertTrue(torch.allclose(ref, result, atol=1e-3, rtol=1e-3))

# test_tasks/test_elementwise.py
class TestElementwiseTask(TestCase):
    def test_add(self):
        A = torch.randn(4096, device="cuda", dtype=torch.float16)
        B = torch.randn(4096, device="cuda", dtype=torch.float16)
        ref = A + B
        task = TaskDesc(op_type=OpType.ELEMENTWISE, op_code=ElemCode.ADD,
                        dimensions=[4096] + [0]*7)
        result = run_single_task(task, [A, B], [4096], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-5))

    def test_silu(self):
        x = torch.randn(4096, device="cuda", dtype=torch.float16)
        ref = torch.nn.functional.silu(x)
        task = TaskDesc(op_type=OpType.ELEMENTWISE, op_code=ElemCode.SILU,
                        dimensions=[4096] + [0]*7)
        result = run_single_task(task, [x], [4096], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-5))

# test_tasks/test_reduce.py
class TestReduceTask(TestCase):
    def test_rmsnorm(self):
        x = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
        w = torch.randn(4096, device="cuda", dtype=torch.float16)
        eps = 1e-5
        ref = torch.nn.functional.rms_norm(x, (4096,), w, eps)
        task = TaskDesc(op_type=OpType.REDUCE, op_code=ReduceCode.RMSNORM,
                        dimensions=[8, 4096, 0, 0, 0, 0, 0, 0])
        # eps stored as float bits in dimensions[2]
        result = run_single_task(task, [x, w], [8, 4096], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-3))

    def test_softmax(self):
        x = torch.randn(8, 128, device="cuda", dtype=torch.float16)
        ref = torch.softmax(x, dim=-1)
        task = TaskDesc(op_type=OpType.REDUCE, op_code=ReduceCode.SOFTMAX,
                        dimensions=[8, 128, 1, 0, 0, 0, 0, 0])
        result = run_single_task(task, [x], [8, 128], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-3))

# test_tasks/test_attention.py
class TestAttentionTask(TestCase):
    def test_basic(self):
        B, H, S, D = 1, 32, 128, 128
        Q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        task = TaskDesc(op_type=OpType.ATTENTION,
                        dimensions=[B, H, S, D, S, 0, 0, 0])
        result = run_single_task(task, [Q, K, V], [B, H, S, D], torch.float16)
        self.assertTrue(torch.allclose(ref, result, atol=1e-3))
```

#### Level 2: Schedule Compiler Tests

```python
# test_schedule_compiler/test_graph_walker.py
class TestGraphWalker(TestCase):
    def test_llama_graph_analysis(self):
        """Export LLaMA-1B, verify all nodes map to tasks."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.float16
        ).cuda().eval()
        x = torch.randint(0, 32000, (1, 128), device="cuda")
        schedule_bytes = compile_schedule(model, x, sm_version=90)
        header = ScheduleHeader.from_bytes(schedule_bytes[:52])
        # LLaMA-1B should produce 150-300 tasks (rough estimate)
        self.assertGreater(header.num_tasks, 50)
        # Verify no unsupported ops (all nodes mapped)
        # (implementation checks for graph split points)

    def test_shape_ops_eliminated(self):
        """Verify reshape/transpose produce no tasks."""
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                x = x.reshape(2, 8, 64)
                x = x.transpose(1, 2)
                x = x.reshape(2, 512)
                return x + 1.0
        model = SimpleModel().cuda().eval()
        x = torch.randn(2, 512, device="cuda")
        schedule_bytes = compile_schedule(model, x, sm_version=90)
        header = ScheduleHeader.from_bytes(schedule_bytes[:52])
        # Only the add should produce a task; reshapes/transpose are free
        self.assertEqual(header.num_tasks, 1)

# test_schedule_compiler/test_buffer_planner.py
class TestBufferPlanner(TestCase):
    def test_no_overlaps(self):
        """Verify no buffer offset ranges overlap for simultaneously live buffers."""
        tasks = [...]  # Construct a test task list
        sizes = {0: 1024, 1: 2048, 2: 512, 3: 4096}
        placements, total = plan_buffers(tasks, sizes, weight_buffers=set())
        # Check no two placements overlap in both arena and time
        for i, p1 in enumerate(placements):
            for j, p2 in enumerate(placements):
                if i >= j:
                    continue
                arena_overlap = (p1.offset < p2.offset + p2.size
                                 and p2.offset < p1.offset + p1.size)
                if arena_overlap:
                    # Must not have overlapping live intervals
                    # (test the invariant the planner guarantees)
                    pass  # detailed check omitted for brevity

    def test_workspace_reasonable(self):
        """Verify total workspace is within expected bounds for LLaMA-1B."""
        # Should be ~2x model activation memory, not 10x
        pass
```

#### Level 3: End-to-End Model Tests

```python
# test_e2e/test_llama_1b.py
class TestLLaMA1B(TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM
        cls.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B", torch_dtype=torch.float16
        ).cuda().eval()

    def test_decode_correctness(self):
        x = torch.randint(0, 32000, (1, 128), device="cuda")
        with torch.no_grad():
            ref = self.model(x).logits

        schedule = compile_schedule(self.model, x, sm_version=90)
        mk_output = run_megakernel(schedule, self.model.state_dict(), x)

        self.assertTrue(torch.allclose(ref, mk_output, atol=1e-4, rtol=1e-3))

    def test_different_seq_lengths(self):
        for seq_len in [1, 32, 128, 512]:
            x = torch.randint(0, 32000, (1, seq_len), device="cuda")
            with torch.no_grad():
                ref = self.model(x).logits
            schedule = compile_schedule(self.model, x, sm_version=90)
            mk_output = run_megakernel(schedule, self.model.state_dict(), x)
            self.assertTrue(torch.allclose(ref, mk_output, atol=1e-4, rtol=1e-3))
```

#### Level 4: Performance Benchmarks

```python
# test_perf/bench_decode_latency.py
def bench_decode_latency(model_id: str, seq_len: int = 128):
    """Compare megabake vs torch.compile vs AOTI+CUDAGraphs."""
    import time

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda().eval()
    x = torch.randint(0, 32000, (1, seq_len), device="cuda")

    # Warmup + benchmark: PyTorch eager
    with torch.no_grad():
        for _ in range(10):
            model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(x)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / 100 * 1000

    # torch.compile
    compiled = torch.compile(model)
    with torch.no_grad():
        for _ in range(10):
            compiled(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            compiled(x)
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) / 100 * 1000

    # Megabake
    schedule = compile_schedule(model, x, sm_version=90)
    # Warmup
    for _ in range(10):
        run_megakernel(schedule, model.state_dict(), x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        run_megakernel(schedule, model.state_dict(), x)
    torch.cuda.synchronize()
    megabake_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"Eager:        {eager_ms:.2f} ms")
    print(f"torch.compile: {compile_ms:.2f} ms")
    print(f"Megabake:     {megabake_ms:.2f} ms")
    print(f"Speedup vs compile: {compile_ms / megabake_ms:.2f}x")
```

### 20.11 Implementation Order

The exact sequence of what to build and validate at each step, with decision points.

**Step 1 (Week 1-2): Critical prototype -- prove matmul works as `__device__` inside a cooperative kernel.**

GOAL: A minimal megakernel that runs a single matmul task and produces correct output.

BUILD:
- `data_types.cuh` with `TaskDesc` struct
- `tasks/matmul.cu` with a basic tiled matmul using shared memory (no CUTLASS yet)
- `scheduler.cu` with the BSP loop (just the for loop + dispatch + grid.sync)
- `megakernel.cu` that includes everything
- A Python test harness that hardcodes a `TaskDesc`, uploads it to GPU, launches the megakernel via `cudaLaunchCooperativeKernel`, and compares against `torch.mm`

VALIDATE:
```bash
python tests/test_tasks/test_matmul.py -k test_square
python tests/test_tasks/test_matmul.py -k test_rectangular
```
Must pass for M/N/K in: (4096, 4096, 4096), (1, 4096, 4096), (128, 11008, 4096).
Tolerance: `atol=1e-3, rtol=1e-3` (FP16 matmul).

DECISION POINT: If the cooperative kernel launch works but matmul accuracy is poor, switch to CUTLASS mid-level components. If cooperative launch itself fails (driver version, occupancy limits), evaluate `cudaGraphs` as an alternative execution model.

**Step 2 (Week 3-4): Add all simple tasks.**

GOAL: Every task type except matmul and attention working inside the megakernel.

BUILD:
- `tasks/elementwise.cu` -- all 19 op_codes (ADD through POW)
- `tasks/reduce.cu` -- SUM, MEAN, SOFTMAX, RMSNORM, LAYERNORM, ARGMAX
- `tasks/embedding.cu` -- table lookup
- `tasks/rope.cu` -- rotary position embedding
- `tasks/copy.cu` -- memcpy with stride support
- `tasks/index.cu` -- gather, scatter, index_select, index_put
- Per-task unit tests for every op_code

VALIDATE:
```bash
pytest tests/test_tasks/ -v
```
All per-task tests pass. Each test compares against the equivalent PyTorch op.

**Step 3 (Week 5-6): Schedule compiler MVP.**

GOAL: `torch.export` of a simple model produces a valid task schedule that runs through the megakernel.

BUILD:
- `schedule_compiler/op_table.py` -- ATen op mapping table
- `schedule_compiler/graph_walker.py` -- FX graph traversal
- `schedule_compiler/shape_ops.py` -- stride resolution for zero-cost ops
- `schedule_compiler/buffer_planner.py` -- arena allocation
- `schedule_compiler/tiling.py` -- tile count computation
- `schedule_compiler/serializer.py` -- binary .schedule writer and reader

VALIDATE:
```bash
# Test with a 2-layer MLP (no attention)
python -c "
import torch
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4096, 11008)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(11008, 4096)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = MLP().cuda().half().eval()
x = torch.randn(1, 4096, device='cuda', dtype=torch.float16)
from megabake.schedule_compiler import compile_schedule
schedule = compile_schedule(model, x, sm_version=90)
# Run through megakernel and compare
"
```

**Step 4 (Week 7-8): Attention task.**

GOAL: SDPA working inside the megakernel.

BUILD:
- `tasks/attention.cu` -- decode attention (batch=1, seq_len=1 query against full KV)
- Update `graph_walker.py` to preserve SDPA and map to OP_ATTENTION
- Update decomposition table to keep SDPA as single node

VALIDATE:
```bash
python tests/test_tasks/test_attention.py
```
Test against `F.scaled_dot_product_attention` for shapes:
- (1, 32, 1, 128) query against (1, 32, 128, 128) KV -- decode
- (1, 32, 128, 128) query against (1, 32, 128, 128) KV -- prefill

**Step 5 (Week 9-12): First full model -- LLaMA-1B.**

GOAL: LLaMA-3.2-1B runs end-to-end as one megakernel with correct output.

BUILD:
- Complete schedule compiler for the LLaMA architecture (all ops mapped)
- `runtime/weight_mapper.py` -- map safetensors keys to buffer indices
- `runtime/loader.py` -- load .schedule + cubin + weights, allocate arena
- `runtime/launcher.py` -- `cudaLaunchCooperativeKernel` wrapper
- RMSNorm pattern matching in the graph walker

VALIDATE:
```bash
python tests/test_e2e/test_llama_1b.py
```
LLaMA-1B decode output matches HuggingFace transformers output: `atol=1e-4, rtol=1e-3`. Test with 5 different random prompts.

**Step 6 (Week 13-16): Performance optimization + LLaMA-8B.**

GOAL: Benchmarkable performance on LLaMA-8B batch-1 decode.

BUILD:
- Replace naive matmul with CuTe DSL or CUTLASS mid-level matmul
- Profile with Nsight Compute: identify bottlenecks (register spills, bank conflicts, HBM bandwidth)
- Shape bucketing: generate schedules for seq_len ranges (1-128, 128-512, 512-2048)
- `runtime/shape_selector.py` -- pick best schedule for input shape

VALIDATE:
```bash
python tests/test_perf/bench_decode_latency.py --model meta-llama/Llama-3-8B
```
Target: 1.5x+ speedup over `torch.compile` on batch-1 decode.

**Step 7 (Week 17-20): Validation suite + second hardware target.**

GOAL: Production-quality correctness on H100 and A100.

BUILD:
- Full numerical validation suite (all Level 1-3 tests)
- NaN/Inf detection at runtime (check output after megakernel completion)
- A100 (sm_80) cubin: recompile with SM 80 target, adjust shared memory limits
- Prefill schedule: separate task ordering with different tiling (full sequence)

VALIDATE:
```bash
# Run full test suite on H100
pytest tests/ -v --hardware h100
# Run full test suite on A100
pytest tests/ -v --hardware a100
```

**Step 8 (Week 21-24): Distribution + packaging.**

GOAL: `pip install megabake` works. Users can load and run pre-compiled models.

BUILD:
- `distribution/hub.py` -- upload/download schedules via `huggingface_hub`
- `distribution/index.py` -- read/write `index.json` for schedule discovery
- `cli.py` -- `megabake bake <model_id>` and `megabake load <model_id>`
- Python package with `load()` and `bake()` API in `__init__.py`
- HF Kernels Hub repo for `megabake-runtime` (cubin files)
- Pre-compiled schedules for: LLaMA-3.2-1B, LLaMA-3-8B, Mistral-7B, Qwen-2.5-7B

VALIDATE:
```bash
pip install megabake
megabake load meta-llama/Llama-3-8B --prompt "The meaning of life is"
```
User gets correct output in under 60 seconds (download + first inference).


## 20. References

### Core Megakernel Systems
- [Hazy Research: Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) -- Original Llama-1B megakernel
- [Hazy Research: Llama-70B TP Megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) -- Throughput-optimized tensor parallel megakernel
- [MPK: Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219) -- Compiler and runtime for mega-kernelizing tensor programs (OSDI 2026)
- [Luminal: Compiling Models to Megakernels](https://blog.luminal.com/p/compiling-models-to-megakernels) -- E-graph based megakernel compiler
- [AutoMegaKernel](https://arxiv.org/html/2606.09682) -- Agent-driven megakernel synthesis for HuggingFace models
- [Together AI: Inside the Kernels Team](https://www.together.ai/blog/inside-the-together-ai-kernels-team) -- Production megakernel deployment
- [Kog.ai: Single-Kernel Engine on MI300X](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/) -- AMD megakernel challenges
- [Smallest.ai: Limits of Large Fused Kernels](https://smallest.ai/blog/the-limits-of-large-fused-kernels-on-nvidia-gpus-why-real-time-ai-inference-needs-more) -- Analysis of fusion limitations
- [Ada-MK: Adaptive MegaKernel](https://arxiv.org/html/2605.11581) -- Baidu production megakernel

### Infrastructure & Distribution
- [HuggingFace Kernels Hub](https://huggingface.co/docs/hub/kernels) -- Pre-compiled operator distribution
- [vLLM RFC #22201: MPK Integration](https://github.com/vllm-project/vllm/issues/22201) -- Attempted and closed
- [vLLM Feature Request #18939](https://github.com/vllm-project/vllm/issues/18939) -- Megakernel support request

### Cross-Domain Research: RTOS
- [AUTOSAR DAG scheduling](https://www.sciencedirect.com/science/article/abs/pii/S1383762126002110)
- [RTAS 2024 Execution Groups](https://daes.cs.tu-dortmund.de/storages/daes-cs/r/Bilder/Beschaeftigte/M._Sc._Mario_Guenzel/publications/shi24rtas_group.pdf)
- [ReDAGRT](https://arxiv.org/pdf/2603.18238)

### Cross-Domain Research: Dataflow & Spatial Computing
- [MIT TTDA](https://ieeexplore.ieee.org/document/48862/)
- [Kitsune](https://dl.acm.org/doi/10.1145/3777466)
- [Graphcore IPU BSP](https://arxiv.org/pdf/2311.04417)
- [SambaNova RDU](https://sambanova.ai/hubfs/23945802/SambaNova_Accelerated-Computing-with-a-Reconfigurable-Dataflow-Architecture_Whitepaper_English-1.pdf)
- [TileLoom](https://arxiv.org/html/2512.22168v1)

### Cross-Domain Research: FPGA & CGRA
- [TAPA](https://dl.acm.org/doi/10.1145/3609335)
- [Dynamic Loop Fusion](https://arxiv.org/html/2501.09231v1)
- [CGRA Survey (NUS)](https://www.comp.nus.edu.sg/~tulika/CGRA-Survey.pdf)
- [Space-time decoupling](https://arxiv.org/pdf/2512.02859)
- [DRESC](https://www.semanticscholar.org/paper/DRESC/2026190e9ddbc75016881cbdd0edef2409fd856e)

### Cross-Domain Research: GPU Systems
- [Megakernels Considered Harmful (HPG 2013)](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- [GPU Coroutines (SIGGRAPH Asia 2024)](https://dl.acm.org/doi/10.1145/3687766)
- [Crystal (GPU databases)](https://github.com/anilshanbhag/crystal)
- [Kernel Weaver](https://ieeexplore.ieee.org/document/6493612/)
- [GPU DB characterization (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p441-cao.pdf)

### Compiler & Optimization
- [NVIDIA Device LTO](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/)
- [nvJitLink JIT LTO](https://developer.nvidia.com/blog/cuda-12-0-compiler-support-for-runtime-lto-using-nvjitlink-library/)
- [PERKS](https://arxiv.org/pdf/2204.02064)
- [Memory-constrained scheduling NP-completeness](https://ieeexplore.ieee.org/document/11078551/)
- [WELDER (OSDI 2023)](https://www.usenix.org/conference/osdi23/presentation/shi)
- [Halide autoscheduler](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf)

### Tooling
- [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [mKernel](https://uccl-project.github.io/posts/mkernel/)
