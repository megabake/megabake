/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Based on the cuBLASDx pipelined GEMM API shown in NVIDIA MathDx's
 * 01_gemm_introduction/introduction_pipeline.cu example. See the installed
 * MathDx LICENSE.txt for the full license text.
 */
#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace {
constexpr int kTileM = 16;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kThreads = 128;
constexpr int kPipelineDepth = 1;

using Half = __half;
using BLAS = decltype(cublasdx::Size<kTileM, kTileN, kTileK>() +
                      cublasdx::Precision<Half, Half, float>() +
                      cublasdx::Function<cublasdx::function::MM>() +
                      cublasdx::Arrangement<cublasdx::row_major,
                                            cublasdx::col_major,
                                            cublasdx::row_major>() +
                      cublasdx::Block() + cublasdx::BlockDim<kThreads>() +
                      cublasdx::Alignment<8, 8, 8>() + cublasdx::WithPipeline() +
                      cublasdx::SM<900, cublasdx::sm_modifier::arch_specific>());
using GlobalA = decltype(cublasdx::make_gmem_tensor<cublasdx::row_major>(
    static_cast<const Half *>(nullptr), 1, 1, 1));
using GlobalB = decltype(cublasdx::make_gmem_tensor<cublasdx::col_major>(
    static_cast<const Half *>(nullptr), 1, 1, 1));
using PipelineResult = decltype(cublasdx::suggest_pipeline<kPipelineDepth, BLAS>(
    std::declval<GlobalA>(), std::declval<GlobalB>()));
using Pipeline = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<PipelineResult &>().value())>>;
using DevicePipeline = decltype(std::declval<Pipeline const &>().get_device_handle());
using GlobalC = decltype(cublasdx::make_gmem_tensor<cublasdx::row_major>(
    static_cast<Half *>(nullptr), 1, 1, 1));

struct Context {
  Pipeline pipeline;
  int m;
  int n;
  int k;
  size_t shared_bytes;
  dim3 block;
  dim3 grid;
};

template<class DP>
__launch_bounds__(kThreads, 1) __global__ void pipeline_body(
    float alpha, float beta, GlobalC global_c,
    __grid_constant__ DP const device_pipeline) {
  CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(BLAS);
  extern __shared__ __align__(device_pipeline.buffer_alignment()) cublasdx::byte smem[];
  auto tile_pipeline = device_pipeline.get_tile(smem, blockIdx.x, blockIdx.y);
  auto tile_c = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);
  tile_pipeline.execute([&](auto &accumulator) {
    accumulator.axpby(alpha, beta, tile_c);
  });
}

template<class DP>
int launch(Context *ctx, Half *output, bool owner, uintptr_t stream_value) {
  auto device_pipeline = ctx->pipeline.get_device_handle();
  GlobalC global_c = cublasdx::make_gmem_tensor<cublasdx::row_major>(
      output, ctx->m, ctx->n, ctx->n);
  auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  cudaError_t status;
  if (owner) {
    float alpha = 1.0f;
    float beta = 0.0f;
    void *args[] = {&alpha, &beta, &global_c, &device_pipeline};
    status = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(pipeline_body<DP>), ctx->grid, ctx->block, args,
        ctx->shared_bytes, stream);
  } else {
    pipeline_body<DP><<<ctx->grid, ctx->block, ctx->shared_bytes, stream>>>(
        1.0f, 0.0f, global_c, device_pipeline);
    status = cudaGetLastError();
  }
  return static_cast<int>(status);
}
}  // namespace

extern "C" int mb3_cublasdx_profile(void *opaque, int *device_sms,
                                    int *cooperative, int *active_per_sm, int *registers,
                                    int *shared_bytes, int *static_shared_bytes,
                                    int *local_bytes,
                                    int *block_threads, int *tile_m,
                                    int *tile_n, int *tile_k) {
  auto *ctx = static_cast<Context *>(opaque);
  if (!ctx || !device_sms || !cooperative || !active_per_sm || !registers ||
      !shared_bytes || !static_shared_bytes || !local_bytes ||
      !block_threads || !tile_m || !tile_n || !tile_k)
    return static_cast<int>(cudaErrorInvalidValue);
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) return static_cast<int>(status);
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, pipeline_body<DevicePipeline>);
  if (status != cudaSuccess) return static_cast<int>(status);
  const size_t smem = ctx->shared_bytes;
  status = cudaFuncSetAttribute(pipeline_body<DevicePipeline>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(smem));
  if (status != cudaSuccess) return static_cast<int>(status);
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, pipeline_body<DevicePipeline>, ctx->block.x * ctx->block.y * ctx->block.z, smem);
  if (status != cudaSuccess) return static_cast<int>(status);
  *device_sms = properties.multiProcessorCount;
  *cooperative = properties.cooperativeLaunch;
  *active_per_sm = active;
  *registers = attributes.numRegs;
  *shared_bytes = static_cast<int>(smem);
  *static_shared_bytes = static_cast<int>(attributes.sharedSizeBytes);
  *local_bytes = attributes.localSizeBytes;
  *block_threads = ctx->block.x * ctx->block.y * ctx->block.z;
  *tile_m = kTileM;
  *tile_n = kTileN;
  *tile_k = kTileK;
  return 0;
}

extern "C" int mb3_cublasdx_create(const void *x, const void *weight,
                                   int m, int n, int k, void **opaque,
                                   int *shared_bytes, int *grid_ctas) {
  if (!x || !weight || !opaque || !shared_bytes || !grid_ctas || m <= 0 ||
      n <= 0 || k <= 0)
    return static_cast<int>(cudaErrorInvalidValue);
  if (m % kTileM || n % kTileN || k % kTileK)
    return static_cast<int>(cudaErrorInvalidConfiguration);
  auto a = cublasdx::make_gmem_tensor<cublasdx::row_major>(
      static_cast<const Half *>(x), m, k, k);
  auto b = cublasdx::make_gmem_tensor<cublasdx::col_major>(
      static_cast<const Half *>(weight), k, n, k);
  auto result = cublasdx::suggest_pipeline<kPipelineDepth, BLAS>(a, b);
  if (!result) return static_cast<int>(cudaErrorNotSupported);
  Pipeline pipeline = std::move(result.value());
  const size_t smem = cublasdx::make_shared_storage_calculator()
                          .add(pipeline.buffer_alignment(), pipeline.buffer_size())
                          .get();
  dim3 block = pipeline.get_block_dim();
  dim3 grid(m / kTileM, n / kTileN, 1);
  auto *ctx = new (std::nothrow) Context{std::move(pipeline), m, n, k, smem, block, grid};
  if (!ctx) return static_cast<int>(cudaErrorMemoryAllocation);
  *shared_bytes = static_cast<int>(smem);
  *grid_ctas = static_cast<int>(grid.x * grid.y);
  *opaque = ctx;
  return 0;
}

extern "C" int mb3_cublasdx_launch(void *opaque, void *output, int owner,
                                   uintptr_t stream_value) {
  auto *ctx = static_cast<Context *>(opaque);
  if (!ctx || !output) return static_cast<int>(cudaErrorInvalidValue);
  return launch<DevicePipeline>(ctx, static_cast<Half *>(output), owner != 0,
                                stream_value);
}

extern "C" void mb3_cublasdx_destroy(void *opaque) {
  delete static_cast<Context *>(opaque);
}
