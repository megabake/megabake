#include "body_harness.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int kSerial = 0;
constexpr int kSimt = 1;
constexpr int kTensorCore = 2;
constexpr int kFloat16 = 0;
constexpr int kBfloat16 = 1;
constexpr int kGridRejected = -1001;
constexpr int kMaxWarps = 32;

__device__ __forceinline__ void serial_output(const Mb3Contraction &p,
                                               int64_t output_index) {
  const int64_t count = p.m * p.n;
  if (output_index >= count) return;
  const int64_t row = output_index / p.n;
  const int64_t col = output_index - row * p.n;
  const auto *x = static_cast<const __half *>(p.x);
  const auto *w = static_cast<const __half *>(p.weight);
  const auto *bias = static_cast<const __half *>(p.bias);
  float acc = 0.0f;
  for (int64_t k = 0; k < p.k; ++k) {
    acc = fmaf(__half2float(x[row * p.x_m_stride + k * p.x_k_stride]),
               __half2float(w[col * p.weight_n_stride + k * p.weight_k_stride]),
               acc);
  }
  const float c = bias ? __half2float(bias[output_index]) : 0.0f;
  static_cast<__half *>(p.output)[output_index] =
      __float2half_rn(p.alpha * acc + p.beta * c);
}

#include "simt_body.cuh"

template <int Warps>
__global__ void serial_standalone(Mb3Contraction p) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  serial_output(p, first);
}

template <int Warps>
__global__ void serial_owner(Mb3Contraction p) {
  __shared__ uint32_t tile_base[kMaxWarps];  // 128 bytes live scheduler scratch
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) tile_base[warp] = blockIdx.x * blockDim.x + warp * 32;
  __syncthreads();
  for (int64_t first = tile_base[warp] + lane; first < p.m * p.n;
       first += static_cast<int64_t>(gridDim.x) * blockDim.x)
    serial_output(p, first);
}

template <int Warps, int VectorWidth>
__global__ void simt_standalone(Mb3Contraction p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * Warps + warp;
  simt_output<VectorWidth>(p, output_index, lane);
}

template <int Warps, int VectorWidth>
__global__ void simt_owner(Mb3Contraction p) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t first = static_cast<int64_t>(blockIdx.x) * Warps + warp;
  for (int64_t output_index = first; output_index < p.m * p.n;
       output_index += static_cast<int64_t>(gridDim.x) * Warps)
    simt_output<VectorWidth>(p, output_index, lane);
}

__device__ __forceinline__ float as_float(__half value) {
  return __half2float(value);
}

__device__ __forceinline__ float as_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <int Warps, int Depth, class T>
__global__ void output_major_mma(Mb3Contraction p) {
  constexpr int kTile = 16;
  constexpr int kTileElements = kTile * kTile;
  constexpr int kWeightElementsPerStage = Warps * kTileElements;
  __shared__ __align__(32) T weight_tile[Depth * kWeightElementsPerStage];
  __shared__ __align__(32) T activation_tile[Depth * kTileElements];
  __shared__ __align__(32) float result_tile[Warps * kTileElements];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t output_tiles = (p.n + kTile - 1) / kTile;
  const int64_t worker_groups = (output_tiles + Warps - 1) / Warps;
  const auto *x = static_cast<const T *>(p.x);
  const auto *weight = static_cast<const T *>(p.weight);
  const auto *bias = static_cast<const T *>(p.bias);
  auto *output = static_cast<T *>(p.output);

  // Y^T = W X^T. The batch axis is padded to one WMMA tile; stores retain
  // only live [M,N] values. Every warp owns one output-channel tile.
  for (int64_t group = blockIdx.x; group < worker_groups;
       group += gridDim.x) {
    const int64_t output_tile = group * Warps + warp;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kTile, kTile, kTile,
                           float> accumulator;
    nvcuda::wmma::fill_fragment(accumulator, 0.0f);

    for (int64_t k_base = 0; k_base < p.k; k_base += Depth * kTile) {
      for (int index = threadIdx.x;
           index < Depth * kWeightElementsPerStage;
           index += blockDim.x) {
        const int stage = index / kWeightElementsPerStage;
        const int stage_index = index % kWeightElementsPerStage;
        const int tile_warp = stage_index / kTileElements;
        const int within_tile = stage_index % kTileElements;
        const int row = within_tile / kTile;
        const int k = within_tile % kTile;
        const int64_t channel = (group * Warps + tile_warp) * kTile + row;
        const int64_t reduction = k_base + stage * kTile + k;
        weight_tile[index] = channel < p.n && reduction < p.k
            ? weight[channel * p.weight_n_stride + reduction * p.weight_k_stride]
            : from_float<T>(0.0f);
      }

      // Stage several K tiles before the MMA phase to amortize CTA barriers.
      if (warp == 0 && lane < kTile) {
        for (int stage = 0; stage < Depth; ++stage) {
          const int64_t reduction = k_base + stage * kTile + lane;
          for (int batch = 0; batch < kTile; ++batch) {
            const bool active = batch < p.m && reduction < p.k;
            activation_tile[stage * kTileElements + batch * kTile + lane] =
                active ? x[batch * p.x_m_stride + reduction * p.x_k_stride]
                       : from_float<T>(0.0f);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int stage = 0; stage < Depth; ++stage) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kTile, kTile, kTile,
                               T, nvcuda::wmma::row_major> a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kTile, kTile, kTile,
                               T, nvcuda::wmma::col_major> b;
        nvcuda::wmma::load_matrix_sync(
            a, weight_tile + stage * kWeightElementsPerStage +
                   warp * kTileElements,
            kTile);
        nvcuda::wmma::load_matrix_sync(
            b, activation_tile + stage * kTileElements, kTile);
        nvcuda::wmma::mma_sync(accumulator, a, b, accumulator);
      }
      __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(
        result_tile + warp * kTileElements, accumulator, kTile,
        nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < kTileElements; index += 32) {
      const int channel_offset = index / kTile;
      const int batch = index % kTile;
      const int64_t channel = output_tile * kTile + channel_offset;
      if (channel < p.n && batch < p.m) {
        const int64_t output_index = static_cast<int64_t>(batch) * p.n + channel;
        const float old_value = bias && p.beta != 0.0f
            ? as_float(bias[output_index])
            : 0.0f;
        output[output_index] = from_float<T>(
            p.alpha * result_tile[warp * kTileElements + index] +
            p.beta * old_value);
      }
    }
  }
}

template <class Kernel>
int profile(Kernel kernel, int threads, Mb3Resources *result) {
  cudaDeviceProp prop{};
  int device = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e != cudaSuccess) return e;
  e = cudaGetDeviceProperties(&prop, device);
  if (e != cudaSuccess) return e;
  cudaFuncAttributes attr{};
  e = cudaFuncGetAttributes(&attr, kernel);
  if (e != cudaSuccess) return e;
  int active = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads,
                                                    0);
  if (e != cudaSuccess) return e;
  result->device_sms = prop.multiProcessorCount;
  result->cooperative_launch = prop.cooperativeLaunch;
  result->active_ctas_per_sm = active;
  result->resident_ctas = active * prop.multiProcessorCount;
  result->registers_per_thread = attr.numRegs;
  result->static_shared_bytes = attr.sharedSizeBytes;
  result->local_bytes = attr.localSizeBytes;
  return 0;
}

template <int Warps, int VectorWidth>
int launch_simt(const Mb3Contraction &p, int strategy, int grid,
                cudaStream_t stream) {
  const dim3 block(Warps * 32);
  if (strategy == kSerial) {
    const int64_t outputs = p.m * p.n;
    if (grid == -1) {
      grid = static_cast<int>((outputs + block.x - 1) / block.x);
      if (grid <= 0) return kGridRejected;
      serial_standalone<Warps><<<grid, block, 0, stream>>>(p);
    } else {
      if (grid <= 0) return kGridRejected;
      void *args[] = {const_cast<Mb3Contraction *>(&p)};
      return cudaLaunchCooperativeKernel(
          reinterpret_cast<void *>(serial_owner<Warps>), dim3(grid), block,
          args, 0, stream);
    }
  } else {
    if (grid == -1) {
      const int64_t outputs = p.m * p.n;
      const int64_t blocks = (outputs + Warps - 1) / Warps;
      simt_standalone<Warps, VectorWidth><<<blocks, block, 0, stream>>>(p);
    } else {
      void *args[] = {const_cast<Mb3Contraction *>(&p)};
      return cudaLaunchCooperativeKernel(
          reinterpret_cast<void *>(simt_owner<Warps, VectorWidth>), dim3(grid),
          block, args, 0, stream);
    }
  }
  return cudaGetLastError();
}

template <int Warps, int VectorWidth>
int profile_simt(int strategy, Mb3Resources *resources) {
  if (strategy == kSerial)
    return profile(serial_owner<Warps>, Warps * 32, resources);
  return profile(simt_owner<Warps, VectorWidth>, Warps * 32, resources);
}

template <int Warps, int VectorWidth>
int profile_standalone(int strategy, Mb3Resources *resources) {
  if (strategy == kSerial)
    return profile(serial_standalone<4>, 128, resources);
  return profile(simt_standalone<Warps, VectorWidth>, Warps * 32, resources);
}

template <int Warps, int Depth, class T>
int launch_mma(const Mb3Contraction &p, int grid, cudaStream_t stream) {
  const int64_t output_tiles = (p.n + 15) / 16;
  const int64_t groups = (output_tiles + Warps - 1) / Warps;
  if (grid == -1) {
    grid = static_cast<int>(groups);
    if (grid <= 0) return kGridRejected;
    output_major_mma<Warps, Depth, T><<<grid, Warps * 32, 0, stream>>>(p);
    return cudaGetLastError();
  }
  if (grid <= 0 || grid > groups) return kGridRejected;
  void *args[] = {const_cast<Mb3Contraction *>(&p)};
  return cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(output_major_mma<Warps, Depth, T>), dim3(grid),
      dim3(Warps * 32), args, 0, stream);
}

template <int Warps, int Depth>
int profile_mma(int dtype, Mb3Resources *resources) {
  if (dtype == kFloat16)
    return profile(output_major_mma<Warps, Depth, __half>, Warps * 32, resources);
  if (dtype == kBfloat16)
    return profile(output_major_mma<Warps, Depth, __nv_bfloat16>, Warps * 32, resources);
  return kGridRejected;
}

template <int Warps, int Depth>
int profile_mma_standalone(int dtype, Mb3Resources *resources) {
  return profile_mma<Warps, Depth>(dtype, resources);
}

int dispatch_profile(int strategy, int warps, int vector_width, int mainloop_depth,
                     int dtype,
                     Mb3Resources *resources) {
  if (!resources) return kGridRejected;
#define MB3_PROFILE_MMA(W, D) \
  if (warps == W && mainloop_depth == D) \
    return profile_mma<W, D>(dtype, resources)
  if (strategy == kTensorCore && vector_width == 1) {
    MB3_PROFILE_MMA(1, 1);
    MB3_PROFILE_MMA(1, 2);
    MB3_PROFILE_MMA(1, 4);
    MB3_PROFILE_MMA(2, 1);
    MB3_PROFILE_MMA(2, 2);
    MB3_PROFILE_MMA(2, 4);
    MB3_PROFILE_MMA(4, 1);
    MB3_PROFILE_MMA(4, 2);
    MB3_PROFILE_MMA(4, 4);
    return kGridRejected;
  }
#undef MB3_PROFILE_MMA
  if ((strategy != kSerial && strategy != kSimt) || dtype != kFloat16 ||
      mainloop_depth != 1)
    return kGridRejected;
  if (strategy == kSerial) return profile_simt<4, 1>(strategy, resources);
#define MB3_PROFILE(W, V) \
  if (warps == W && vector_width == V) \
    return profile_simt<W, V>(strategy, resources)
  MB3_PROFILE(1, 1);
  MB3_PROFILE(1, 4);
  MB3_PROFILE(2, 1);
  MB3_PROFILE(2, 4);
  MB3_PROFILE(4, 1);
  MB3_PROFILE(4, 4);
#undef MB3_PROFILE
  return kGridRejected;
}

int dispatch_standalone_profile(int strategy, int warps, int vector_width,
                                int mainloop_depth, int dtype,
                                Mb3Resources *resources) {
  if (!resources) return kGridRejected;
#define MB3_PROFILE_STANDALONE_MMA(W, D) \
  if (warps == W && mainloop_depth == D) \
    return profile_mma_standalone<W, D>(dtype, resources)
  if (strategy == kTensorCore && vector_width == 1) {
    MB3_PROFILE_STANDALONE_MMA(1, 1);
    MB3_PROFILE_STANDALONE_MMA(1, 2);
    MB3_PROFILE_STANDALONE_MMA(1, 4);
    MB3_PROFILE_STANDALONE_MMA(2, 1);
    MB3_PROFILE_STANDALONE_MMA(2, 2);
    MB3_PROFILE_STANDALONE_MMA(2, 4);
    MB3_PROFILE_STANDALONE_MMA(4, 1);
    MB3_PROFILE_STANDALONE_MMA(4, 2);
    MB3_PROFILE_STANDALONE_MMA(4, 4);
    return kGridRejected;
  }
#undef MB3_PROFILE_STANDALONE_MMA
  if ((strategy != kSerial && strategy != kSimt) || dtype != kFloat16 ||
      mainloop_depth != 1)
    return kGridRejected;
  if (strategy == kSerial) return profile_standalone<4, 1>(strategy, resources);
#define MB3_PROFILE(W, V) \
  if (warps == W && vector_width == V) \
    return profile_standalone<W, V>(strategy, resources)
  MB3_PROFILE(1, 1);
  MB3_PROFILE(1, 4);
  MB3_PROFILE(2, 1);
  MB3_PROFILE(2, 4);
  MB3_PROFILE(4, 1);
  MB3_PROFILE(4, 4);
#undef MB3_PROFILE
  return kGridRejected;
}

int dispatch_launch(const Mb3Contraction &p, int strategy, int warps,
                    int vector_width, int mainloop_depth, int grid,
                    cudaStream_t stream) {
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || !p.x || !p.weight || !p.output ||
      (p.beta != 0.0f && !p.bias) ||
      (p.input_dtype != kFloat16 && p.input_dtype != kBfloat16) ||
      (strategy != kSerial && strategy != kSimt && strategy != kTensorCore))
    return kGridRejected;
  if (strategy == kTensorCore) {
    if (p.m > 16 || vector_width != 1) return kGridRejected;
    if (warps == 1 && mainloop_depth == 1)
      return p.input_dtype == kFloat16
          ? launch_mma<1, 1, __half>(p, grid, stream)
          : launch_mma<1, 1, __nv_bfloat16>(p, grid, stream);
    if (warps == 1 && mainloop_depth == 2)
      return p.input_dtype == kFloat16
          ? launch_mma<1, 2, __half>(p, grid, stream)
          : launch_mma<1, 2, __nv_bfloat16>(p, grid, stream);
    if (warps == 1 && mainloop_depth == 4)
      return p.input_dtype == kFloat16
          ? launch_mma<1, 4, __half>(p, grid, stream)
          : launch_mma<1, 4, __nv_bfloat16>(p, grid, stream);
    if (warps == 2 && mainloop_depth == 1)
      return p.input_dtype == kFloat16
          ? launch_mma<2, 1, __half>(p, grid, stream)
          : launch_mma<2, 1, __nv_bfloat16>(p, grid, stream);
    if (warps == 2 && mainloop_depth == 2)
      return p.input_dtype == kFloat16
          ? launch_mma<2, 2, __half>(p, grid, stream)
          : launch_mma<2, 2, __nv_bfloat16>(p, grid, stream);
    if (warps == 2 && mainloop_depth == 4)
      return p.input_dtype == kFloat16
          ? launch_mma<2, 4, __half>(p, grid, stream)
          : launch_mma<2, 4, __nv_bfloat16>(p, grid, stream);
    if (warps == 4 && mainloop_depth == 1)
      return p.input_dtype == kFloat16
          ? launch_mma<4, 1, __half>(p, grid, stream)
          : launch_mma<4, 1, __nv_bfloat16>(p, grid, stream);
    if (warps == 4 && mainloop_depth == 2)
      return p.input_dtype == kFloat16
          ? launch_mma<4, 2, __half>(p, grid, stream)
          : launch_mma<4, 2, __nv_bfloat16>(p, grid, stream);
    if (warps == 4 && mainloop_depth == 4)
      return p.input_dtype == kFloat16
          ? launch_mma<4, 4, __half>(p, grid, stream)
          : launch_mma<4, 4, __nv_bfloat16>(p, grid, stream);
    return kGridRejected;
  }
  if (p.input_dtype != kFloat16 || mainloop_depth != 1) return kGridRejected;
  if (strategy == kSerial) return launch_simt<4, 1>(p, strategy, grid, stream);
#define MB3_LAUNCH(W, V) \
  if (warps == W && vector_width == V) \
    return launch_simt<W, V>(p, strategy, grid, stream)
  MB3_LAUNCH(1, 1);
  MB3_LAUNCH(1, 4);
  MB3_LAUNCH(2, 1);
  MB3_LAUNCH(2, 4);
  MB3_LAUNCH(4, 1);
  MB3_LAUNCH(4, 4);
#undef MB3_LAUNCH
  return kGridRejected;
}

}  // namespace

extern "C" int mb3_owner_profile(int strategy, int warps_per_cta,
                                  int vector_width, int mainloop_depth,
                                  int input_dtype,
                                  Mb3Resources *resources) {
  return dispatch_profile(strategy, warps_per_cta, vector_width, mainloop_depth,
                          input_dtype, resources);
}

extern "C" int mb3_standalone_profile(int strategy, int warps_per_cta,
                                       int vector_width, int mainloop_depth,
                                       int input_dtype,
                                       Mb3Resources *resources) {
  return dispatch_standalone_profile(strategy, warps_per_cta, vector_width,
                                     mainloop_depth, input_dtype, resources);
}

extern "C" int mb3_validate_owner_grid(int strategy, int warps_per_cta,
                                       int vector_width, int mainloop_depth,
                                       int input_dtype,
                                       int grid_ctas) {
  if (grid_ctas <= 0) return kGridRejected;
  Mb3Resources resources{};
  const int e = dispatch_profile(strategy, warps_per_cta, vector_width,
                                 mainloop_depth, input_dtype, &resources);
  if (e) return e;
  if (!resources.cooperative_launch || !resources.active_ctas_per_sm ||
      grid_ctas > resources.resident_ctas)
    return kGridRejected;
  return 0;
}

extern "C" int mb3_launch(const Mb3Contraction *problem, int strategy,
                           int warps_per_cta, int vector_width,
                           int mainloop_depth, int grid_ctas,
                           uintptr_t stream) {
  if (!problem) return kGridRejected;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (grid_ctas == -1)
    return dispatch_launch(*problem, strategy, warps_per_cta, vector_width,
                           mainloop_depth, -1, cuda_stream);
  const int e = mb3_validate_owner_grid(strategy, warps_per_cta, vector_width,
                                        mainloop_depth, problem->input_dtype,
                                        grid_ctas);
  if (e) return e;
  return dispatch_launch(*problem, strategy, warps_per_cta, vector_width,
                         mainloop_depth, grid_ctas, cuda_stream);
}

extern "C" int mb3_launch_admitted(
    const Mb3Contraction *problem, int strategy, int warps_per_cta,
    int vector_width, int mainloop_depth, int grid_ctas, uintptr_t stream) {
  if (!problem || grid_ctas <= 0) return kGridRejected;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  // The caller has already admitted this exact target/entry/grid. Avoid
  // repeating cudaFuncGetAttributes and occupancy queries in timed invocations.
  return dispatch_launch(*problem, strategy, warps_per_cta, vector_width,
                         mainloop_depth, grid_ctas, cuda_stream);
}

extern "C" int mb3_capture(const Mb3Contraction *problem, int strategy,
                            int warps_per_cta, int vector_width,
                            int mainloop_depth, int grid_ctas,
                            int launches, uintptr_t stream, void **graph_exec) {
  if (!problem || !graph_exec || launches <= 0) return kGridRejected;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e = cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeThreadLocal);
  if (e != cudaSuccess) return e;
  for (int i = 0; i < launches; ++i) {
    const int launch_error = mb3_launch(problem, strategy, warps_per_cta,
                                         vector_width, mainloop_depth, grid_ctas,
                                         reinterpret_cast<uintptr_t>(cuda_stream));
    if (launch_error) {
      cudaGraph_t discarded{};
      cudaStreamEndCapture(cuda_stream, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      return launch_error;
    }
  }
  cudaGraph_t graph{};
  e = cudaStreamEndCapture(cuda_stream, &graph);
  if (e != cudaSuccess) return e;
  cudaGraphExec_t executable{};
  e = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  if (e != cudaSuccess) return e;
  *graph_exec = reinterpret_cast<void *>(executable);
  return 0;
}

extern "C" int mb3_graph_launch(void *graph_exec, uintptr_t stream) {
  if (!graph_exec) return kGridRejected;
  return cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec),
                         reinterpret_cast<cudaStream_t>(stream));
}

extern "C" int mb3_graph_destroy(void *graph_exec) {
  if (!graph_exec) return 0;
  return cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph_exec));
}

extern "C" const char *mb3_error_string(int code) {
  if (code == kGridRejected) return "unsupported body arguments or owner grid";
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
