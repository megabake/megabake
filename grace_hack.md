# GraCE, opaque CUDA kernels, and the path to a composable persistent GEMM

Research status: 2026-09-05, revised for the traffic-first MegaBake V2.2 architecture. This document
supersedes the earlier drafts of this file. It is based on the local
[GraCE paper](./GraceCE_OSDI_26_CudaGraphPytorchCompile%20%281%29.pdf), the
[official OSDI page](https://www.usenix.org/conference/osdi26/presentation/ghosh), current and
archived CUDA documentation, the current MegaBake implementation, and the Mirage MPK/CUTLASS
implementation paths.

## Executive conclusion

The most important correction is this:

> **GraCE does not change the signature or code of a cuBLAS kernel.** It changes bytes in the
> launch-parameter buffer of an already captured cuBLAS kernel node immediately before that node
> runs.

GraCE has two genuinely different implementations:

| Kernel kind | What changes | Is the real kernel ABI changed? |
|---|---|---|
| Triton/JIT kernel | An LLVM/PTX pass changes selected `T*` parameters to `T**` and inserts loads that dereference them | **Yes** |
| cuBLAS or another immutable vendor kernel | A preceding graph node copies new values into existing ABI slots of the captured vendor node | **No** |

For a vendor kernel, the prelude is therefore an **in-graph launch-frame editor**, not a wrapper
function in the C/C++ sense. It cannot add or remove parameters, change their sizes, replace the
kernel body, or make the kernel callable as a device function. It can update bytes in existing
parameter slots, and CUDA also exposes device updates for the node's grid dimension and enabled
state. It cannot device-update the function, block dimension, dynamic shared-memory size, or an
arbitrary launch attribute.

This leads to three distinct goals which should not be conflated:

1. **Use different A/B/C pointers on every graph replay while retaining the exact cuBLAS kernel.**
   GraCE can do this, subject to recovering the correct node and parameter offsets.
2. **Call the exact internal cuBLAS kernel without making a cuBLAS host API call.** This may be
   possible from the host after tracing the live `CUfunction`, launch configuration, argument bytes,
   attributes, workspace, and every kernel in the selected algorithm. It is private and brittle,
   but it is a real experimental path.
3. **Execute cuBLAS's implementation inside MegaBake's existing persistent grid.** GraCE does not
   help with this. A CUDA kernel entry point is not a device subroutine. Dynamic parallelism or a
   device-launched CUDA Graph still creates another grid. True one-grid composition requires a
   device-callable implementation such as cuBLASDx, CUTLASS/CuTe collectives, or an MPK-style task
   function—or a research-scale SASS lifter.

For MegaBake, the highest-probability route to cuBLAS-class performance in an owned persistent
entry is:

- use tracing of cuBLAS as an **oracle** to learn which algorithm and tiling win for each fixed
  shape;
- implement/autotune those shapes with CUTLASS/CuTe or cuBLASDx as callable device code;
- schedule exact-shape output-tile work with a compact phase program, as opposed to an arbitrary
  per-SM busy-wait DAG;
- split resource-incompatible task families into separate entrypoints and join them with a CUDA
  Graph when the recovered occupancy outweighs the extra grid boundary;
- use GraCE only as a hybrid fallback or as a diagnostic reference path.

The project README reports that GEMM is 84.8% of Gemma-2B's MegaBake time. That makes this a kernel
quality and task-mapping problem, not principally a launch-parameter problem.

There is a second, more fundamental conclusion. Batch-one decode GEMV streams almost every model
weight once per token. A persistent grid does not keep a multi-gigabyte model in registers, shared
memory, or the MIG partition's L2. After reference-precision kernel quality is fixed, the largest
remaining levers are weight-only W8/W4 layouts, reuse of each weight tile across continuously
batched tokens, and transformer composites that remove activation materialization and joins. Those
are distinct numerical/service contracts and must be compared with equivalent baselines.

## 1. What “signature” means here

There are four layers that are easy to mix up:

1. **Host library API:** for example `cublasGemmEx(handle, transA, ..., A, ..., C, ...)`.
2. **Private dispatch plan:** cuBLAS chooses one or more internal GPU kernels, workspace, launch
   geometry, attributes, and sometimes preprocessing or reduction steps.
3. **GPU entry ABI:** a selected kernel receives a packed parameter block whose positional fields
   have fixed offsets and sizes.
4. **Kernel body:** SASS/PTX loads those fields and performs the computation.

GraCE's vendor path changes only layer 3's **values** for a captured launch. It does not expose the
host API's semantic signature, does not rerun layer 2's algorithm selection, and does not alter
layer 4.

Suppose a private entry has the effective layout:

```text
offset  size  unknown semantic meaning
0x00      8   A pointer
0x08      8   B pointer
0x10      8   C pointer
0x18      8   workspace pointer
0x20      4   internal dimension/stride
...
```

The prelude can write eight new bytes at `0x00`. The vendor kernel still has exactly the same ABI
and still executes exactly the same instructions. Calling this a “changed function signature” is a
useful description of the graph's new *external interface*, but is technically false for the
vendor kernel itself.

## 2. Exact reconstruction of GraCE's mechanism

### 2.1 The JIT path really changes the ABI

For a generated Triton kernel, GraCE knows which parameters correspond to external graph inputs.
Its custom LLVM pass:

1. changes each selected parameter from `T*` to `T**`;
2. inserts an entry load equivalent to `T *p = *pp`;
3. rewrites subsequent uses to the loaded `p`;
4. leaves the rest of the computation unchanged.

The graph captures the address of a stable device pointer cell. Before each replay the CPU places
the invocation's actual device address into that cell with a tiny H2D copy. The rewritten kernel
loads that address. This is a real kernel-signature and body transformation.

### 2.2 Why the vendor path needs a prelude

GraCE cannot make the same compiler edit to a private cuBLAS binary. Instead, it records two graph
nodes with an ordering edge:

```text
CPU writes new A/B/C addresses into stable device cells
                     |
                     v
       prelude node: patch target-node parameter bytes
                     |
              full dependency edge
                     |
                     v
       original captured cuBLAS kernel node executes
```

The target node is unchanged. The prelude must be an ancestor of every node it edits. Merely adding
an independent graph root is not enough; without dependencies, the target could run first. GraCE
re-records the graph with the prelude first so stream ordering creates the required dependencies.

### 2.3 What `cudaGraphKernelNodeSetParam` actually does

CUDA 12.4 added device-side graph-node updates. The raw overload is conceptually:

```cpp
__device__ cudaError_t cudaGraphKernelNodeSetParam(
    cudaGraphDeviceNode_t node,
    size_t offset,
    const void *value,
    size_t size);
```

It copies `size` bytes from device-accessible storage at `value` into the target node's launch
parameter buffer at `offset`. The current API also supplies a typed overload and a batched
`cudaGraphKernelNodeUpdatesApply`. See the [CUDA 12.8 graph API used by the paper](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__GRAPH.html)
and the [current update structure](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaGraphKernelNodeUpdate.html).

For a stable device cell `void **a_slot` whose eight bytes contain the current A address, the
essential prelude is:

```cpp
__global__ void patch_one(cudaGraphDeviceNode_t target,
                          size_t a_offset,
                          void **a_slot,
                          cudaError_t *status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Copies the pointer VALUE stored at a_slot, not the contents of matrix A.
    *status = cudaGraphKernelNodeSetParam(
        target, a_offset, a_slot, sizeof(void *));
  }
}
```

This is the paper's “dereference.” It does not dereference A and copy matrix data. It dereferences
one level of indirection by copying the pointer-sized value held in `a_slot`.

Per replay, the host-side operation is approximately:

```cpp
void *next_a = d_actual_a;
cudaMemcpyAsync(d_a_slot, &next_a, sizeof(next_a),
                cudaMemcpyHostToDevice, stream);
cudaGraphLaunch(exec, stream);
```

Because the pointer copy and graph launch are enqueued on one stream, no CPU/GPU synchronization is
needed between them. The graph launch then executes the prelude before the target.

The prelude can patch ordinary scalar fields too. That is usually unsafe for an opaque GEMM: changing
M/N/K may invalidate the selected algorithm, grid, workspace size, edge predicates, or launch
attributes. GraCE applies indirection to external data pointers while keeping shapes and algorithms
fixed.

### 2.4 Obtaining the device-updatable node handle

`cudaGraphKernelNodeSetParam` takes a `cudaGraphDeviceNode_t`, not the ordinary host graph-node
handle. The target must opt into device updates before graph instantiation.

If the launch site is controlled, set
`cudaLaunchAttributeDeviceUpdatableKernelNode` on `cudaLaunchKernelEx` while the stream is being
captured. CUDA writes the device handle into the launch-attribute value.

For a node created by an opaque library, current APIs permit opting in after graph construction:

```cpp
cudaKernelNodeAttrValue attr{};
attr.deviceUpdatableKernelNode.deviceUpdatable = 1;
attr.deviceUpdatableKernelNode.devNode = nullptr;

cudaGraphKernelNodeSetAttribute(
    target_node,
    cudaKernelNodeAttributeDeviceUpdatableKernelNode,
    &attr);

cudaGraphDeviceNode_t target_dev =
    attr.deviceUpdatableKernelNode.devNode;
```

The important detail is that `devNode` is a value field filled by CUDA; it is not a pointer to an
output variable. NVIDIA documents the launch attribute and restrictions in the
[driver execution API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html). A
[developer-forum experiment](https://forums.developer.nvidia.com/t/how-to-use-the-device-side-cuda-graph-apis-how-to-get-hold-of-cudagraphdevicenode-t/346795)
demonstrates both launch-time and post-capture forms. The forum code marks a user kernel, not a
cuBLAS node, so the cuBLAS case still must be tested on the exact driver/toolkit combination.

After adding the prelude and all dependency edges:

1. instantiate the graph once;
2. upload it with `cudaGraphUpload`/`cuGraphUpload` before first launch;
3. re-upload after any host-side executable update to a device-updatable node.

Documented restrictions include:

- an opted-in node cannot opt out and cannot be removed;
- its attributes cannot be copied to or from another node;
- a graph containing one cannot be multiply instantiated;
- neither graph nor executable may be passed to `cudaGraphExecUpdate`;
- concurrent host and device updates are not a synchronization mechanism;
- graph upload is required before device updates are used.

### 2.5 Finding the parameter offset

Given a `CUfunction` and positional argument index, the public API is now unambiguous:

```cpp
size_t count;
cuFuncGetParamCount(func, &count);             // current toolkits

size_t offset, size;
cuFuncGetParamInfo(func, index, &offset, &size);
```

`cuFuncGetParamInfo` returns the positional field's offset and size in the device-side parameter
layout. It returns neither semantic type/name nor a parameter-buffer pointer. The current
[execution-control documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
explicitly says its output can be used with device-side graph updates.

The remaining problem is mapping “this captured value equals the known A pointer” to the positional
index/offset. If `cuGraphKernelNodeGetParams` succeeds, its `kernelParams[i]` entries point to the
captured values. Compare their copied bytes against known, distinct, valid A/B/C/workspace
allocations, and use `cuFuncGetParamInfo(func, i, ...)` to translate `i` to the packed offset.

### 2.6 A real reproducibility gap in the paper

The paper says that `cuFuncGetParamInfo` obtains a handle to a kernel parameter buffer and that
GraCE then byte-matches placeholder pointers inside it. That is not the documented contract of
`cuFuncGetParamInfo`; it returns only an offset and optional size. The paper does not state how it:

- obtains the private `CUfunction` for each vendor node;
- obtains readable captured argument bytes if graph-node parameter queries fail;
- associates an Inductor-level A/B/C with a private positional argument;
- handles `kernelParams` versus packed `extra` launch forms.

This matters because a 2022–2025 NVIDIA forum reproducer found that
`cudaGraphKernelNodeGetParams` succeeded for a user kernel but returned “invalid device function”
for a captured internal cuBLAS kernel. NVIDIA did not explain the behavior and warned that editing
a private argument pack is risky. This is an old stack-specific observation, not a documented
current limitation, but it is directly relevant. See the
[full reproducer and discussion](https://forums.developer.nvidia.com/t/stream-capture-of-cublas-gemm/216148).

As of this research, the official GraCE page links the paper but no implementation that resolves
this ambiguity. Therefore the accurate conclusion is:

> The node-update mechanism is public and reproducible for a known kernel. The paper does not give
> enough implementation detail to guarantee recovery of a cuBLAS node's live parameter buffer on
> every CUDA stack.

The practical recovery paths in section 3 close much of that gap experimentally.

### 2.7 Performance scope

The paper's Figure 9 shows the prelude approach scaling poorly with argument count and exceeding
roughly 10 microseconds at the high end. Direct JIT rewriting is substantially faster, which is why
GraCE reserves preludes for immutable kernels. Therefore one prelude per whole graph, using the
batched update API, is preferable to one prelude per argument or target.

This mechanism preserves the target kernel's execution performance, but adds pointer H2D copies and
a prelude launch. It is useful when it replaces large device-to-device placeholder copies; it is not
automatically a win for tiny tensors. GraCE profiles variants and selects among no graph, graph
without indirection, and graph with indirection.

## 3. How far can an opaque cuBLAS launch be recovered?

There are two different extraction targets:

- **live launch extraction:** retain the already loaded `CUfunction` and copy its launch state;
- **file extraction:** recover a cubin/fatbin/PTX image that can be loaded in another process.

Live extraction is easier and preserves module-global state. It should be attempted first.

### 3.1 Recovery ladder

| Evidence recovered | Supported mechanism | What it proves | What it does not prove |
|---|---|---|---|
| Kernel name, duration, grid/block, registers/shared memory | Nsight/CUPTI activity | Which GPU work ran | Argument values or semantics |
| Live function handle and launch call | CUPTI driver callback or interposition | Exact loaded entry and launch form | Meaning of each private field |
| Positional count, offsets, sizes | `cuFuncGetParamCount/Info` | Raw ABI layout | C/C++ types, names, invariants |
| Captured parameter bytes | Graph node params or launch callback | Exact invocation values | Which values may safely change |
| Loaded cubin bytes | CUPTI module resource callback | Executed module image | Source code or a stable ABI |
| SASS and ELF metadata | `nvdisasm`/`cuobjdump` | Machine instructions/resources | Maintainable CUDA source |
| Reproducible direct launch | `cuLaunchKernel(Ex)` with traced state | Host API can be bypassed for that frozen case | Portability, one-grid composition |

### 3.2 Path A: inspect the captured graph node

Enumerate graph nodes, select kernel nodes, and call the driver API
`cuGraphKernelNodeGetParams`. Current `CUDA_KERNEL_NODE_PARAMS` can contain a legacy `func`
(`CUfunction`), a newer `kern` (`CUkernel`), grid and block dimensions, dynamic shared memory,
`kernelParams` or `extra`, context, and launch metadata depending on structure version. The returned argument storage
is owned by the graph node; copy it and do not edit it directly. The
[driver graph documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)
specifies both parameter representations.

If `func` is available, current driver APIs allow:

```text
cuFuncGetName                    exact symbol name
cuFuncGetModule                  owning live module
cuFuncGetParamCount/Info         positional raw ABI
cuModuleGetFunctionCount         number of entries in that live module
cuModuleEnumerateFunctions       every function handle in that module
cuFuncGetAttribute               registers, static shared memory, binary target, etc.
```

These module-enumeration APIs are real and have existed since CUDA 12.4; see
[Module Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).
If a `CUkernel` is supplied instead, the Library Management API exposes `cuKernelGetFunction` and
`cuKernelGetParamInfo` for code loaded as a `CUlibrary`.

This is not guaranteed for a cuBLAS node because of the failure described in section 2.6. Test both
runtime and driver APIs on the target system. Do not treat the 2022 failure as proof that all current
drivers fail, or current documentation as proof that every private library node succeeds.

### 3.3 Path B: trace the driver launch that cuBLAS performs

CUPTI's callback API invokes user code for driver/runtime calls. For launch callbacks,
`CUpti_CallbackData` supplies:

- `symbolName`, the launched kernel symbol;
- `functionParams`, a pointer to the generated metadata structure for that CUDA API call;
- context and correlation information.

The callback data and nested pointers are valid only during the callback, so all required bytes must
be deep-copied immediately. See the current
[CUPTI callback API](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html) and
[`CUpti_CallbackData`](https://docs.nvidia.com/cupti/13.0.0/api/structCUpti__CallbackData.html).

Subscribe at least to all relevant variants, not only `cuLaunchKernel`:

- `cuLaunchKernel` and versioned callbacks;
- `cuLaunchKernelEx` and versioned callbacks;
- cooperative launch variants if observed;
- graph-node creation/update callbacks for capture diagnostics.

At callback entry, cast `functionParams` using the installed
`generated_cuda_meta.h`. For a `cuLaunchKernel` call, this yields the `CUfunction`, geometry,
shared-memory bytes, stream, `kernelParams`, and `extra`. Cache the function's parameter layout, then
deep-copy each positional value. Use a reentrancy guard if the tracer itself makes driver queries;
the installed CUPTI version and headers are the source of truth.

There are two launch encodings, and a useful tracer must handle both:

```text
kernelParams != NULL
    query count/offset/size from CUfunction
    for each i: copy paramSize[i] bytes from kernelParams[i]

extra != NULL
    walk the tag/value list
    CU_LAUNCH_PARAM_BUFFER_POINTER -> packed buffer address
    CU_LAUNCH_PARAM_BUFFER_SIZE    -> total byte count
    copy that entire packed buffer while the callback is active
```

Do not copy only the `void **kernelParams` array: its elements point at the actual value storage,
whose contents may disappear when the library call returns. Likewise, do not retain the packed
buffer pointer. A replay record must own the bytes themselves. For pointer-valued arguments those
bytes are the GPU virtual address—not the memory to which the address points. Any referenced
descriptor or workspace whose contents affect execution needs a separate, explicitly bounded copy
or a retained lifetime.

CUPTI activity records are excellent for timings and correlations but do **not** contain kernel
argument bytes. API callbacks are the relevant facility. Also, replaying a CUDA Graph generally
produces a graph-launch API call rather than a fresh user-visible `cuLaunchKernel` call for every
node. Trace the original ordinary/capture-time cuBLAS call, not only later graph replays.

An `LD_PRELOAD` wrapper around driver launch entry points can record the same information, but modern
libraries can obtain entry points through `cuGetProcAddress`, so naïve symbol interposition may miss
calls. CUPTI is the supported observation interface.

### 3.4 Path C: capture the module bytes that CUDA actually loaded

CUPTI resource callbacks include `CUPTI_CBID_RESOURCE_MODULE_LOADED`. The associated
`CUpti_ModuleResourceData` exposes `pCubin`, `cubinSize`, and `moduleId`; the storage is callback-
scoped and must be copied. NVIDIA explicitly documents using this binary with `nvdisasm` for SASS
correlation. See [`CUpti_ModuleResourceData`](https://docs.nvidia.com/cupti/12.8.1/api/structCUpti__ModuleResourceData.html)
and the [CUPTI SASS-correlation description](https://docs.nvidia.com/cuda/archive/10.0/cupti/index.html#sass-source-correlation).

This route is stronger than blindly scanning `libcublas.so`:

- it captures a module actually loaded for the current architecture;
- it also sees modules loaded dynamically or originating in `libcublasLt.so`;
- it avoids guessing which embedded fatbin contains the selected kernel.

Correlate module ID, function name, context, and launch activity. Hash every dumped image and retain
toolkit, driver, GPU UUID, compute capability, cuBLAS version, math mode, and environment settings.

### 3.5 Path D: inspect installed libraries statically

NVIDIA's official binary utilities can inspect CUDA code embedded in an executable, object, archive,
or shared library:

```bash
cuobjdump --list-elf      /path/to/libcublas*.so
cuobjdump --extract-elf all /path/to/libcublas*.so
cuobjdump --list-ptx      /path/to/libcublas*.so
cuobjdump --dump-sass     extracted.sm_90.cubin
cuobjdump --dump-elf      extracted.sm_90.cubin
cuobjdump --dump-elf-symbols extracted.sm_90.cubin
cuobjdump --dump-resource-usage extracted.sm_90.cubin
nvdisasm extracted.sm_90.cubin
```

The exact supported switches are listed in the
[CUDA Binary Utilities manual](https://docs.nvidia.com/cuda/cuda-binary-utilities/contents.html).
Possible outcomes include no useful embedded image, architecture-specific cubins only, stripped
symbols, or code loaded from a different library at runtime. PTX is often absent; SASS is not CUDA
C++ source.

`cuLibraryLoadFromFile` is not a magic parser for an arbitrary ELF host shared object. It loads a
CUDA code object supported by the Library API. Once a cubin/fatbin/PTX code object has been obtained,
the correct APIs include `cuLibraryLoadData/FromFile`, `cuLibraryGetKernelCount`,
`cuLibraryEnumerateKernels`, `cuKernelGetName`, and `cuKernelGetFunction`. See
[Library Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html).

### 3.6 Recovering a raw ABI is not recovering semantics

`cuFuncGetParamInfo` can reveal this:

```text
arg 0 -> offset 0,  size 8
arg 1 -> offset 8,  size 8
arg 2 -> offset 16, size 4
...
```

It cannot reveal whether arg 0 is A, B, workspace, a pointer to an internal descriptor, or an opaque
token. Recover semantics with controlled differential experiments:

1. Freeze GPU, driver, toolkit, cuBLAS version, dtype, layouts, math mode, workspace, shape, and
   algorithm.
2. Give A/B/C distinct **valid** allocations and record their pointer values.
3. Trace the full launch sequence and deep-copy every argument byte.
4. Change exactly one host-level input while preserving dispatch; trace again.
5. Diff function, launch geometry, attributes, argument values, pointed-to descriptor memory where
   safely identifiable, and output.
6. Repeat with swapped buffers of identical size/alignment.
7. Validate guard regions, numerical results, and Compute Sanitizer before classifying a field.

Do not pass fake addresses such as `0xDEADBEEF` merely to make byte matching convenient. cuBLAS may
validate alignment, create derived descriptors, or execute the pointer during capture. Distinct real
allocations give safe recognizable values.

Binary metadata may contain positional parameter information—unofficial parsers recognize
`.nv.info.<kernel>` records such as `EIATTR_KPARAM_INFO`—and SASS constant-bank loads can show where
fields enter address arithmetic. This is reverse-engineered, architecture/version-dependent
information. [`cudaparsers`](https://github.com/VivekPanyam/cudaparsers),
[`denvdis`](https://github.com/redplait/denvdis/blob/master/test/nvd.cc), and
[`CuAssembler`](https://github.com/gpuocelot/cuasm) are research aids, not NVIDIA-supported ABIs.

### 3.7 The most promising exact-kernel hack: reuse the live `CUfunction`

If the trace supplies a valid live `CUfunction`, a file extraction is unnecessary. Keep the owning
cuBLAS handle/module alive and issue `cuLaunchKernel` or `cuLaunchKernelEx` from the host with the
same:

- entry handle;
- grid, block, cluster and other launch attributes;
- dynamic shared memory and carveout;
- copied argument layout and values;
- user-owned workspace and initialized module state;
- stream ordering and every auxiliary kernel in the algorithm.

The driver accepts either an array of pointers to per-argument values or a correctly aligned packed
buffer; the [Driver API launch guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html)
documents both forms.

This can bypass the cuBLAS **host dispatch** after one discovery/warm-up phase while executing the
same loaded SASS. It is the best candidate for obtaining the same isolated kernel runtime.

It is not automatically equivalent to `cublasGemmEx`. A library call may launch multiple kernels,
use internal global state, select split-K reductions, create/initialize descriptors, or depend on a
workspace lifetime. `cublasLtMatmul` explicitly exposes algorithm/workspace concepts; an opaque
classic API may hide them. Capture and replay the whole observed plan, not just the longest kernel.

Freeze this path by a key such as:

```text
(GPU UUID, SM, driver, toolkit, cuBLAS build hash, operation, dtype,
 layouts, M/N/K/batch, alignments, math mode, workspace bytes, algorithm,
 function/module hashes, launch attributes, raw ABI hash)
```

Any mismatch should fall back to cuBLAS or an owned kernel. Never ship a guessed private ABI as a
general GEMM interface.

## 4. Can the recovered entry run from inside a persistent kernel?

### 4.1 A `CUfunction` is a host handle, not a device-callable pointer

`cuLaunchKernel` consumes a host-side opaque `CUfunction`. Device code cannot call the Driver API.
A CUDA `__global__` entry is also not a normal `__device__` function: it expects grid/block special
registers, launch parameter constant space, entry/exit semantics, and independently allocated
register/shared-memory resources.

Passing the numeric bits of a `CUfunction` into device code is unsupported.

### 4.2 Dynamic parallelism exists, but does not solve composition

CUDA exposes low-level device launch functions:

```cpp
extern "C" __device__ void *cudaGetParameterBuffer(size_t alignment,
                                                     size_t size);
extern "C" __device__ cudaError_t cudaLaunchDevice(
    void *func, void *parameterBuffer,
    dim3 grid, dim3 block, unsigned sharedMem, cudaStream_t stream);
```

The [Dynamic Parallelism documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
specifies a maximum 4 KiB parameter buffer and requires `cudadevrt`/device linking.

The blocker is `func`. NVIDIA's separate-compilation documentation states that device function
addresses cannot cross independent device executables; the device link must see both caller and
callee. A `CUfunction` obtained by loading cuBLAS is not documented as a valid
`cudaLaunchDevice` function pointer. See [NVCC separate compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#potential-separate-compilation-issues).

Even if a relocatable vendor entry could be linked into the same device executable and its pointer
resolved, dynamic parallelism launches a **child grid**. It does not inline work into the parent.
Current CDP2 also does not let the continuing parent explicitly synchronize and consume child
results; dependent continuation is expressed as a tail launch. It adds device-runtime tracking and
launch overhead. A long-lived cooperative parent occupying all SMs is an especially poor host for
child grids because resource availability and concurrency are not guaranteed.

So dynamic launch is a low-probability experiment for GPU-side dispatch, not a route to MegaBake's
single persistent grid.

### 4.3 Device-launched CUDA Graphs are the best hybrid GPU-resident controller

CUDA can instantiate a graph with `cudaGraphInstantiateFlagDeviceLaunch`, upload it, and launch it
from a kernel using fire-and-forget, tail, or sibling streams. Device-launchable graphs may contain
kernel, memcpy, memset, and child-graph nodes; their kernel nodes cannot themselves use dynamic
parallelism. The [CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#device-graph-launch)
lists the complete requirements.

This permits a GPU-controlled state machine such as:

```text
controller graph node
  -> prelude patches cuBLAS-node pointers
  -> exact captured cuBLAS nodes
  -> tail graph continues the pipeline
```

Conditional IF/WHILE/SWITCH graph nodes can add device-controlled loops and branches. This can
remove repeated CPU dispatch while retaining exact vendor kernels. Every vendor node is still a
separate grid; it is not a fused megakernel.

cuBLAS capture can create allocation/free nodes, which are not allowed in device-launched graphs.
NVIDIA recommends supplying user-owned workspace with `cublasSetWorkspace`; coefficient capture also
depends on host versus device pointer mode. See
[cuBLAS CUDA Graph support](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support).

### 4.4 `cuLibraryGetUnifiedFunction` is speculative, not a solution

The current Driver API has `cuLibraryGetUnifiedFunction(void **fptr, CUlibrary, const char *symbol)`
and a unified-function-pointer device attribute. Public documentation does not establish that:

- an ordinary `__global__` entry is a “unified function”;
- arbitrary internal cuBLAS kernels are exported through this namespace;
- the returned pointer is accepted by `cudaLaunchDevice`;
- it crosses independently loaded device executables.

It is worth a small capability probe on the target H200/H100, but it should not be an architectural
dependency. Failure to find the symbol or `CUDA_ERROR_NOT_SUPPORTED` is an expected outcome.

### 4.5 Why “turn the cubin entry into a device function” is research-scale

Inlining an extracted SASS entry into MegaBake would require all of the following:

1. recover and legally use the code image;
2. translate entry-parameter constant-space loads to MegaBake task arguments;
3. replace `blockIdx/gridDim` assumptions with the persistent scheduler's tile identity;
4. convert `EXIT` and reconvergence behavior into callable returns;
5. relocate constant/global symbols, tensor maps, textures, and workspace references;
6. reconcile registers, predicates, barriers, dynamic/shared memory, cluster semantics, and warp roles;
7. merge control-flow metadata, relocations, and scheduling control words into MegaBake's cubin;
8. retune because changed register allocation and instruction addresses can change performance.

NVIDIA does not provide a supported SASS assembler or SASS-to-PTX/CUDA decompiler. Official tools
disassemble. [NVBit](https://github.com/NVlabs/nvbit) can inspect and instrument loaded SASS and is
useful for tracing address/parameter use, but it is a research prototype and does not convert an
entry into a callable subroutine. Third-party assemblers are architecture-limited: for example
[TuringAs](https://github.com/daadaada/turingas) targets Volta/Turing/Ampere, while
[cubit](https://github.com/kacper-daftcode/cubit) currently targets SM120, not MegaBake's H200 SM90.

Even a technically successful lift no longer preserves the original kernel unchanged, so “same
binary performance” cannot be assumed.

## 5. Supported ways to get cuBLAS-class math inside one grid

### 5.1 cuBLASDx: the direct answer to “BLAS inside my kernel”

cuBLASDx is NVIDIA's header-only, device-callable BLAS extension. It provides block descriptors,
Tensor Core/TMA use, shared-memory or register-accumulator GEMM APIs, and pipelined full-device GEMM
examples. The register API is specifically intended for custom epilogues and fusion. See
[Using cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/using_cublasdx.html) and
[performance guidance](https://docs.nvidia.com/cuda/cublasdx/performance.html).

Important limitations:

- it is still Early Access and covers a subset of full cuBLAS;
- A and B tiles must fit the per-block register/shared-memory execution model;
- it does not expose or reuse private cuBLAS host-library kernels;
- a complete global GEMM still needs a tile scheduler and K loop;
- performance depends on alignment, layouts, pipeline depth, and enough CTAs;
- the latest docs require CUDA 13.0+, C++17, and recent CUTLASS, although archived cuBLASDx 0.4.1
  supported CUDA 12.0+. Pin and validate a known MathDx/toolkit combination rather than mixing
  headers. See [current requirements](https://docs.nvidia.com/cuda/cublasdx/requirements_func.html)
  and [archived 0.4.1 requirements](https://docs.nvidia.com/cuda/archive/13.0.1/cublasdx/0.4.1/requirements_func.html).

For MegaBake, cuBLASDx should be benchmarked as a task-body generator, not called through a host
adapter. One persistent CTA computes one or more output tiles, looping over K and fusing the epilogue
before releasing its task.

### 5.2 CUTLASS 3.x/CuTe: use the layer below the host launcher

CUTLASS 3.x separates GEMM into Atom, Tiled MMA/Copy, Collective, Kernel, and Device layers. The
Device adapter is a host launcher. The Collective mainloop/epilogue and the Kernel layer's
`operator()` are device code. NVIDIA explicitly describes the collective epilogue as a fusion point
and `GemmUniversal::operator()` as a device function. See the
[CUTLASS 3.x design article](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
and [GEMM API documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api_3x.md).

For a persistent runtime, it is usually easier to reuse/port the Collective primitives than call an
unchanged whole `GemmUniversal`, because MegaBake already owns block identity and tile scheduling.
The task body must supply the same warp specialization, shared storage, TMA descriptors, pipeline
state, and epilogue expected by the selected collective.

### 5.3 Mirage MPK is a concrete open implementation of the desired architecture

Mirage MPK does not extract cuBLAS. It compiles each operator into a device task and executes those
tasks from persistent worker blocks. Its current Hopper task headers include CuTe warp-specialized
GEMM implementations, and its GEMM code implements TMA producers, WGMMA consumers, pipelines, K
iteration, and the epilogue directly inside a `CUTLASS_DEVICE` function. See the
[MPK repository and branch instructions](https://github.com/mirage-project/mirage),
[Hopper task header](https://raw.githubusercontent.com/mirage-project/mirage/mpk/include/mirage/persistent_kernel/tasks/hopper/task_header.cuh), and
[`gemm_ws_mpk.cuh`](https://raw.githubusercontent.com/mirage-project/mirage/mpk/include/mirage/persistent_kernel/tasks/cute/hopper/gemm_ws_mpk.cuh).

This is much closer to MegaBake's goal than GraCE:

```text
opaque cuBLAS plan                  MPK/MegaBake plan
-----------------                  -----------------
host dispatch chooses entry        compiler/autotuner chooses task variant
new grid owns block IDs             persistent worker owns tile task
private entry consumes ABI          device function consumes TaskDesc
kernel writes global output         task may fuse epilogue/next work
```

The OSDI'26 [Mirage Persistent Kernel paper](https://arxiv.org/abs/2512.22219) and public `mpk`
branch are an actionable source base. Porting its Hopper linear task or its scheduler choices is a
higher-value experiment than reverse engineering a cuBLAS cubin.

### 5.4 Resource facts a persistent GEMM must respect

A monolithic kernel pays constraints that independent cuBLAS kernels do not:

- **block shape is global:** every task executes under MegaBake's fixed block dimension;
- **register allocation is compile-time:** a heavy WGMMA path can raise the persistent kernel's
  register count and reduce occupancy even while other tasks run;
- **shared memory is per resident block:** storage should be overlaid with a union across mutually
  exclusive tasks, not summed, but the maximum still determines residency;
- **warp roles are structural:** Hopper TMA/WGMMA kernels often require dedicated producer and
  consumer warp groups and mbarrier state;
- **grid mapping is owned by MegaBake:** a cuBLAS/CUTLASS tile scheduler cannot independently assume
  `blockIdx.x` enumerates its whole problem;
- **global dependencies need a protocol:** excessive full-grid BSP barriers erase concurrency;
- **skinny decode GEMMs may be bandwidth-bound:** copying a large generic square-GEMM design is not
  necessarily the correct solution for M=1–4.

These explain why copying source-level math is insufficient. The task mapping and resource envelope
must be co-designed with the persistent scheduler.

The present MegaBake implementation makes the worst case global: every cooperative launch requests
about 220 KiB dynamic shared memory, so Hopper can retain only one 256-thread CTA per SM. That is an
eight-warp, 12.5% theoretical active-warp ceiling on a 64-warp SM. One CTA per SM can be correct for
a carefully designed warp-specialized TMA/WGMMA kernel; it should not be the universal launch
contract for pointwise, reduction, attention, and skinny-streaming code. V2.2 therefore compiles
resource-class entrypoints and measures the single-grid versus CUDA-Graph-segmented result.

### 5.5 What “same performance” requires for decode

For a fixed reference-precision `M=1..4` shape, matching the vendor body is primarily an achieved-
bandwidth, work-mapping, and resource problem:

1. specialize M at compile time, including accumulator-array bounds;
2. prepack/transpose weights once into the exact access layout;
3. compare tuned SIMT GEMV with transposed-small-N WGMMA and MPK/CuTe TMA pipelines;
4. tune active worker count and shared memory on the actual MIG instance;
5. reject spills and dynamically indexed local arrays in hot paths;
6. separate release code from task-profiling instrumentation;
7. measure every scheduler, reset, binding, copy, and output operation around the body.

After reference-precision parity, the traffic floor can be changed only by changing what is moved or
how often it is reused:

- W8A16 roughly halves weight bytes relative to FP16/BF16;
- W4A16 roughly quarters weight bytes before scale metadata;
- continuous batching applies one streamed weight tile to several activation rows;
- speculative or multi-token verification can create another form of multi-row work;
- FP8/INT8 KV caches reduce long-context attention traffic;
- fused gate/up and grouped QKV projections remove intermediate traffic and phase joins;
- `RECOMPUTE_PER_WORKER` can duplicate a cheap RMSNorm and avoid global normalized-activation
  materialization.

These are not consequences of extracting cuBLAS. They are architecture choices that an opaque
vendor entry cannot provide inside MegaBake's grid.

## 6. Recommended MegaBake program

### Track A — reproduce GraCE cleanly as a hybrid baseline

Start with a user kernel, not cuBLAS:

1. Build a two-node graph explicitly: prelude -> target.
2. Query known target offsets with `cuFuncGetParamInfo`.
3. Mark target device-updatable before instantiation and retain `devNode`.
4. Allocate global device pointer cells and an error-status cell.
5. Have the one-thread prelude batch-patch A/B/C.
6. Upload, replay with changing buffers, and validate results and error statuses.
7. Benchmark pointer copies + prelude separately from the target.

Then repeat with cuBLAS capture:

1. provide a fixed user-owned workspace;
2. use distinct valid A/B/C allocations;
3. enumerate all graph nodes and dependencies;
4. try both runtime and driver get-params APIs;
5. trace capture-time launch callbacks concurrently as an independent source of truth;
6. mark only the confirmed target node device-updatable;
7. patch one pointer at a time and verify guarded outputs;
8. preserve every other node and edge from the library plan.

Success criterion: exact numerical equivalence across at least 10,000 alternating replays, no
Compute Sanitizer findings, and an end-to-end win over ordinary static-input graph replay plus
copies. This track will not enter the single MegaBake kernel.

### Track B — build an exact private-kernel host launcher lab

This is the highest-value “hack” experiment:

1. Add a CUPTI tracer for module-load and all launch variants.
2. Run one frozen cuBLAS/cuBLASLt operation outside graph replay.
3. Record the whole ordered launch plan, raw per-argument values, launch attributes, and cubin hashes.
4. Keep cuBLAS and its modules alive.
5. Reissue the observed plan with live `CUfunction` handles.
6. Replace only confirmed A/B/C fields with same-shaped, same-aligned allocations.
7. Compare output, isolated kernel duration, full-plan duration, and hardware counters.
8. Repeat in a fresh process using the dumped cubin only after the live-handle version works.

Stop if the plan includes unexplained pointer-bearing descriptors or hidden initialization. Treat a
working launcher as version-locked research infrastructure, not a distributable backend.

### Track C — fix the real one-grid GEMM path

For production MegaBake, use this order:

1. make the benchmark timeline truthful: include every kernel, memcpy, memset, allocation, pointer
   update, scheduler reset, and output copy;
2. remove the current per-invocation dependency/tile-counter clones and hidden output clone by
   introducing preallocated in-entry/epoch state and `RUN_INTO`/borrowed-output contracts;
3. classify real model linears by `(M,N,K,dtype,layout,epilogue,precision policy)` and time each body
   against cuBLASLt or the equivalent quantized baseline;
4. generate distinct M=1, M=2, and M=4 implementations so the compiler never carries an unused
   `float acc[64]` or runtime-M loop through the skinny path;
5. compare a tuned coalesced SIMT GEMV, a transposed-small-N WGMMA form, and one pinned MPK/CuTe
   Hopper implementation;
6. tune active/resident workers, vector width, tile shape, stages, and dynamic shared memory on the
   actual H200 MIG partition;
7. replace the universal maximum-SMEM entry with resource-class entries, while retaining a measured
   strict single-grid artifact;
8. use a compact static phase program with static work ranges and only measured atomic cursors;
9. inspect SASS and counters for TMA/WGMMA issue, achieved bandwidth, memory stalls, registers,
   local memory, spills, shared memory, and achieved occupancy;
10. keep a hybrid cuBLAS graph fallback for shapes the owned task does not yet match.

A realistic acceptance ladder is:

```text
correctness
  -> isolated owned GEMM >= 80% of cuBLASLt for each hot shape
  -> >= 95% after variant tuning
  -> fused task beats cuBLASLt + separate epilogue
  -> whole persistent model beats torch.compile on the target workload
```

Exact isolated parity is not always necessary. The owned task can win end-to-end by eliminating
intermediate global traffic and launch/barrier overhead.

Once one exact-shape body is competitive, add traffic-removing composites in this order:

1. RMSNorm recomputed per output worker and retained locally for the following linear;
2. gate and up projections computed together with a SiLU-multiply epilogue that stores only the
   product;
3. grouped/concatenated QKV projection with GQA-aware output routing;
4. RoPE plus paged KV-cache append;
5. paged and split-KV decode attention selected by sequence/parallelism thresholds.

Quantized W8A16/W4A16 and continuous batching are separate acceptance classes. They should be added
after the reference path is understood, but their precision/layout and service contracts belong in
the compiler from the beginning.

### Track D — low-probability research probes

Run these only after Tracks A–C have instrumentation:

1. test `cuLibraryGetUnifiedFunction` on an owned cubin with an explicitly exported symbol;
2. test whether its pointer is accepted by `cudaLaunchDevice` when caller/callee share one JIT link;
3. feed an owned relocatable cubin and wrapper into `nvJitLink`;
4. repeat with a CUPTI-dumped vendor cubin only if licensing permits and inspect linker diagnostics;
5. use NVBit to log which raw parameter offsets feed global address calculations;
6. evaluate SASS lifting only for one frozen architecture and one kernel after calculating the
   register/shared-memory/control-flow rewrite cost.

`nvJitLink` accepts cubin/PTX/LTO-IR/fatbin inputs, but that does not imply a fully linked private
entry becomes a relocatable device subroutine. Only compatible relocatable device code can resolve
cross-module calls. See [nvJitLink](https://docs.nvidia.com/cuda/nvjitlink/index.html) and the
[NVCC compatibility rules](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#potential-separate-compilation-issues).

## 7. Decision table

| Desired outcome | Feasible? | Best mechanism | Single persistent grid? | Stability |
|---|---:|---|---:|---|
| Change A/B/C on a captured cuBLAS replay | Yes, if node/offset recovery works | GraCE prelude + device node updates | No | Medium, version-test |
| Preserve exact cuBLAS SASS and remove repeated host dispatch | Possibly | Trace once, direct host `CUfunction` launch or graph replay | No | Low/private |
| Let GPU choose and run exact captured cuBLAS work | Yes in compatible cases | Device-launched graph + tail graph | No | Medium |
| Launch recovered entry with dynamic parallelism | Experimental | Same-device-link kernel pointer + `cudaLaunchDevice` | No | Very low |
| Inline unchanged cuBLAS SASS into MegaBake | No supported path | SASS lifting/binary rewriting research | Nominally | Extremely low |
| Put optimized BLAS math in MegaBake | Yes | cuBLASDx or CUTLASS/CuTe/MPK task | **Yes** | High for owned code |
| Match/exceed cuBLAS on fixed hot shapes | Plausible, not guaranteed | Trace as oracle + autotuned specialized task + fusion | **Yes** | High after validation |
| Reduce batch-one weight traffic | Yes, under a quantized numerical policy | W8A16/W4A16 prepack + mixed-dtype task | **Yes** | High for owned code; quality-gated |
| Amortize weight traffic across requests | Yes | Continuous batching + grouped persistent GEMM | **Yes** for owned tasks | High; changes service objective |
| Avoid normalized-activation materialization | Often | Recompute pure RMSNorm per worker/cluster | **Yes** | Shape/resource dependent |
| Preserve exact cuBLAS while using resource-specific owned segments | Yes | Top-level graph with persistent and vendor nodes | No for whole graph | High with public recipes |

## 8. Corrections to the previous draft

The earlier `grace_hack.md` contained several claims that should not guide implementation:

- **Wrong:** GraCE changes a cuBLAS function signature.
  **Correct:** it changes values in the existing captured launch ABI.

- **Wrong:** `cuFuncGetParamInfo` returns a parameter-buffer handle.
  **Correct:** it returns an argument's offset and size given a function and index.

- **Wrong:** the post-capture attribute's `devNode` should point to an output variable.
  **Correct:** it is a value field populated by CUDA.

- **Overstated:** a private cuBLAS kernel can simply be extracted from `libcublas.so`, loaded, and
  relaunched.
  **Correct:** useful code may be elsewhere or dynamically loaded; direct relaunch additionally
  needs the full private plan, ABI, state, attributes, and workspace. CUPTI module callbacks and live
  function handles are stronger discovery routes.

- **Wrong API:** `cuKernelGetFunctionCount`.
  **Correct:** use `cuLibraryGetKernelCount` for a `CUlibrary`, or
  `cuModuleGetFunctionCount`/`cuModuleEnumerateFunctions` for a `CUmodule`.

- **Overstated:** CUPTI activity exposes all kernel arguments.
  **Correct:** activity exposes execution metadata; driver API callbacks expose launch-call
  parameters, which must be deep-copied during the callback.

- **Wrong implication:** a recovered host `CUfunction` can be passed to device launch code.
  **Correct:** it is an opaque host handle; device function pointers cannot cross independent device
  executables under the documented model.

- **Wrong implication:** child launch or device graph launch fuses the vendor kernel into the
  persistent kernel.
  **Correct:** both execute a separate grid.

- **Wrong performance implication:** a persistent model kernel keeps model weights resident on chip.
  **Correct:** batch-one decode still streams model-scale weights each token; persistence primarily
  removes orchestration/materialization and enables controlled composition.

- **Wrong metric:** “one kernel” after filtering memcpy, memset, and copy-like profiler events.
  **Correct:** report host submissions, GPU grids, copies/memsets/allocations, and full invocation
  time separately. The current runtime's scheduler-state and output clones are real work.

- **Overstated architecture:** one maximum-shared-memory entry is inherently faster than segmented
  entries.
  **Correct:** one heavy path constrains the whole entry's occupancy and spills. Compare a strict
  grid with resource-specific entries connected by a CUDA Graph.

- **Incomplete fusion model:** values must either be materialized or forwarded once on chip.
  **Correct:** for tiny pure producers such as decode RMSNorm, recomputation per consuming CTA or
  cluster can be cheaper than global communication.

Any architecture document derived from the prior draft's “extract vendor fatbin and call it as a
device task” assumption should be re-audited against this correction.

## 9. Legal and operational boundary

The technical observation APIs above are public, but using them to reverse engineer, modify, or
redistribute proprietary SDK binaries is a separate licensing question. The current
[CUDA Toolkit EULA](https://docs.nvidia.com/cuda/pdf/EULA.pdf) restricts reverse engineering,
decompilation, disassembly, modification, and distribution of portions of the SDK. Applicable law
and separate NVIDIA agreements may affect the result. Obtain legal review before shipping or sharing
extracted cuBLAS cubins or derived SASS. Keeping a private experimental trace is not automatically a
license to redistribute it.

For production, owned CUTLASS/CuTe/cuBLASDx source is both technically more stable and legally much
cleaner than embedding a private cuBLAS image.

## 10. Source map

Primary sources:

- [GraCE paper and OSDI page](https://www.usenix.org/conference/osdi26/presentation/ghosh)
- [CUDA Runtime Graph API, CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__GRAPH.html)
- [CUDA Driver execution control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
- [CUDA Driver graph management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)
- [CUDA Driver module management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
- [CUDA Driver library management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html)
- [CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA programmatic dependent launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)
- [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA L2 cache control and MIG restriction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [NVCC separate compilation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/contents.html)
- [CUPTI callbacks](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html)
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [cuBLAS CUDA Graph support](https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support)
- [cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/)
- [CUTLASS 3.x design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
- [CUTLASS grouped persistent scheduler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
- [CUTLASS Hopper INT4/BF16 example](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu)
- [Mirage MPK repository](https://github.com/mirage-project/mirage)
- [TensorRT-LLM paged attention and in-flight batching](https://nvidia.github.io/TensorRT-LLM/features/paged-attention-ifb-scheduler.html)
- [TensorRT-LLM attention/XQA](https://nvidia.github.io/TensorRT-LLM/features/attention.html)
- [NVIDIA matrix-multiplication performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

Useful empirical/research sources, treated as non-contractual:

- [Device-updatable graph-node examples](https://forums.developer.nvidia.com/t/how-to-use-the-device-side-cuda-graph-apis-how-to-get-hold-of-cudagraphdevicenode-t/346795)
- [Captured cuBLAS get-params failure](https://forums.developer.nvidia.com/t/stream-capture-of-cublas-gemm/216148)
- [NVBit](https://github.com/NVlabs/nvbit)
- [CuAssembler](https://github.com/gpuocelot/cuasm)
- [cudaparsers](https://github.com/VivekPanyam/cudaparsers)

## Final answer in one paragraph

GraCE's prelude makes an immutable cuBLAS node *look* as if it accepted pointer-to-pointer arguments
by storing changing pointers in stable device cells and copying those values into the captured
node's original parameter slots immediately before execution. The cuBLAS function, signature, SASS,
resources, and grid never change. A live internal entry and raw ABI may be recovered with graph
queries or CUPTI and replayed as a frozen host-side research plan, but there is no supported bridge
from that opaque `CUfunction` to an inline MegaBake device call. A production persistent path must
use owned cuBLASDx/CUTLASS/CuTe/MPK-style code, specialize M/layout/precision, and tune resources
against the traced vendor oracle. The larger architecture must then attack the real decode costs:
streamed weight bytes, resource-wide occupancy, intermediate materialization, phase joins, KV
traffic, per-call resets/copies, and—under a distinct service contract—lack of multi-token weight
reuse. GraCE remains a strong hybrid graph-binding technique, not the mechanism that creates
persistent composability.
