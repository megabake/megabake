"""Emit the one-entry source selected by the bounded physical search."""

from __future__ import annotations

import hashlib

from .bodies.registry import BodyTacticSpec
from .physical import PhysicalCandidate


def emit_entry_source(candidate: PhysicalCandidate, tactic: BodyTacticSpec, *,
                      runtime_api: bool = False) -> str:
    if candidate.tactic_id != tactic.tactic_id or candidate.body_source_hash != tactic.source_hash:
        raise ValueError("physical candidate and body tactic identity/source differ")
    if not tactic.compatible or not tactic.device_callable or tactic.source_text is None:
        raise ValueError("selected source requires a compatible retained device-callable body")
    if hashlib.sha256(tactic.source_text.encode()).hexdigest() != tactic.source_hash:
        raise ValueError("body source hash changed after physical search")
    block = candidate.block_threads
    tile = candidate.outputs_per_cta
    if tactic.provider == "simt":
        if tile * 32 != block:
            raise ValueError("SIMT CTA must own one output per warp")
        try:
            vector_width = int(tactic.tactic_id.rsplit(".v", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("SIMT tactic has no declared vector width") from exc
        kernel = f"""
extern \"C\" __global__ __launch_bounds__({block})
void megabake_v3_entry(Mb3Contraction p) {{
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t output = static_cast<int64_t>(blockIdx.x) * {tile} +
                         static_cast<int64_t>(threadIdx.x >> 5);
  simt_output<{vector_width}>(p, output, lane);
}}
"""
    elif tactic.provider == "generic":
        types = {"float16": "__half", "bfloat16": "__nv_bfloat16", "float32": "float"}
        inputs = tactic.shape_layout_guard.get("inputs", ())
        output = tactic.shape_layout_guard.get("output", {})
        if len(inputs) != 3:
            raise ValueError("the initial generic entry adapter supports the three-input addmm ABI")
        input_types = [types.get(item.get("dtype")) for item in inputs]
        output_type = types.get(output.get("dtype"))
        if any(item is None for item in input_types) or output_type is None:
            raise ValueError("generic body has an unsupported pointer dtype")
        args = ", ".join(
            [*(f"static_cast<const {dtype} *>(p.{name})" for dtype, name in zip(
                input_types, ("bias", "x", "weight"))),
             f"static_cast<{output_type} *>(p.output)",
             "begin", "end"]
        )
        kernel = f"""
extern \"C\" __global__ __launch_bounds__({block})
void megabake_v3_entry(Mb3Contraction p) {{
  const int64_t begin = static_cast<int64_t>(blockIdx.x) * {tile};
  const int64_t end = begin + {tile} < p.m * p.n ? begin + {tile} : p.m * p.n;
  {tactic.source_symbol}({args});
}}
"""
    else:
        raise ValueError(f"no owner-entry source adapter for provider {tactic.provider!r}")

    preamble = "#include <cuda_runtime.h>\n#include <stdint.h>\n"
    if tactic.provider == "generic":
        preamble += '#include "body_harness.h"\n'
    body_source = tactic.source_text.replace("#pragma once\n", "", 1)
    source = preamble + body_source + "\n" + kernel
    if runtime_api:
        source += _runtime_source(block)
    return source


def _runtime_source(block_threads: int) -> str:
    return f"""
#include <cuda_runtime_api.h>
#include <stddef.h>

struct V3EntryResources {{
  int device_sms;
  int cooperative_launch;
  int block_threads;
  int requested_ctas;
  int active_ctas_per_sm;
  int resident_ctas;
  int registers_per_thread;
  size_t static_shared_bytes;
  size_t local_bytes;
  int function_max_threads;
  int max_dynamic_shared_bytes;
  size_t constant_bytes;
  int binary_version;
  int ptx_version;
  size_t parameter_bytes;
  int admitted;
}};

extern \"C\" int mb3_entry_resources(int requested_ctas, V3EntryResources *out) {{
  if (!out || requested_ctas <= 0) return -1;
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaDeviceProp properties{{}};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  int cooperative = 0;
  status = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaFuncAttributes attributes{{}};
  status = cudaFuncGetAttributes(&attributes, megabake_v3_entry);
  if (status != cudaSuccess) return static_cast<int>(status);
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, megabake_v3_entry, {block_threads}, 0);
  if (status != cudaSuccess) return static_cast<int>(status);
  const int resident = active * properties.multiProcessorCount;
  *out = V3EntryResources{{properties.multiProcessorCount, cooperative,
      {block_threads}, requested_ctas, active, resident, attributes.numRegs,
      attributes.sharedSizeBytes, attributes.localSizeBytes,
      attributes.maxThreadsPerBlock, attributes.maxDynamicSharedSizeBytes,
      attributes.constSizeBytes,
      attributes.binaryVersion, attributes.ptxVersion,
      static_cast<int>(sizeof(Mb3Contraction)),
      cooperative && requested_ctas <= resident}};
  return 0;
}}

extern \"C\" int mb3_launch_selected_entry(const Mb3Contraction *problem,
                                           int requested_ctas, uintptr_t stream_value) {{
  if (!problem) return -1;
  V3EntryResources resources{{}};
  const int queried = mb3_entry_resources(requested_ctas, &resources);
  if (queried != 0) return queried;
  if (!resources.admitted) return -2;
  void *arguments[] = {{const_cast<Mb3Contraction *>(problem)}};
  return static_cast<int>(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(megabake_v3_entry), dim3(requested_ctas),
      dim3({block_threads}), arguments, 0,
      reinterpret_cast<cudaStream_t>(stream_value)));
}}
"""


__all__ = ["emit_entry_source"]
