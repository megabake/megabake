#include "body_harness.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int kSerial = 0;
constexpr int kSimt = 1;
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

template <int VectorWidth>
__device__ __forceinline__ float simt_scalar_dot(
    const Mb3Contraction &p, int64_t row, int64_t col, int lane) {
  const auto *x = static_cast<const __half *>(p.x);
  const auto *w = static_cast<const __half *>(p.weight);
  float acc = 0.0f;
  for (int64_t base = 0; base < p.k; base += 32 * VectorWidth) {
#pragma unroll
    for (int item = 0; item < VectorWidth; ++item) {
      const int64_t k = base + lane + item * 32;
      if (k < p.k) {
        acc = fmaf(__half2float(x[row * p.x_m_stride + k * p.x_k_stride]),
                   __half2float(w[col * p.weight_n_stride + k * p.weight_k_stride]),
                   acc);
      }
    }
  }
  return acc;
}

template <int FixedK>
__device__ __forceinline__ void simt_half2_accumulate(
    const Mb3Contraction &p, const __half *x, const __half *w, int64_t pair,
    float &even_acc, float &odd_acc) {
  bool valid;
  if constexpr (FixedK > 0)
    valid = pair < FixedK / 2;
  else
    valid = pair < p.k / 2;
  if (valid) {
    const int64_t k = pair * 2;
    const float2 xv = __half22float2(
        *reinterpret_cast<const __half2 *>(x + k));
    const float2 wv = __half22float2(
        *reinterpret_cast<const __half2 *>(w + k));
    even_acc = fmaf(xv.x, wv.x, even_acc);
    odd_acc = fmaf(xv.y, wv.y, odd_acc);
  }
}

template <int FixedK, int VectorWidth>
__device__ __forceinline__ float simt_half2_dot(
    const Mb3Contraction &p, int64_t row, int64_t col, int lane) {
  const auto *x = static_cast<const __half *>(p.x) + row * p.x_m_stride;
  const auto *w = static_cast<const __half *>(p.weight) + col * p.weight_n_stride;
  float even_acc = 0.0f;
  float odd_acc = 0.0f;
  if constexpr (FixedK > 0) {
    constexpr int kPairs = FixedK / 2;
    constexpr int kIterations = (kPairs + 32 * VectorWidth - 1) /
                                (32 * VectorWidth);
#pragma unroll
    for (int iter = 0; iter < kIterations; ++iter) {
#pragma unroll
      for (int item = 0; item < VectorWidth; ++item)
        simt_half2_accumulate<FixedK>(
            p, x, w, iter * 32 * VectorWidth + lane + item * 32,
            even_acc, odd_acc);
    }
  } else {
    for (int64_t base = 0; base < p.k / 2; base += 32 * VectorWidth) {
#pragma unroll
      for (int item = 0; item < VectorWidth; ++item)
        simt_half2_accumulate<FixedK>(
            p, x, w, base + lane + item * 32, even_acc, odd_acc);
    }
  }
  return even_acc + odd_acc;
}

template <int VectorWidth>
__device__ __forceinline__ void simt_output(const Mb3Contraction &p,
                                             int64_t output_index,
                                             int lane) {
  const int64_t count = p.m * p.n;
  if (output_index >= count) return;  // warp-uniform tail
  int64_t row;
  int64_t col;
  if (p.m == 1) {
    row = 0;
    col = output_index;
  } else {
    row = output_index / p.n;
    col = output_index - row * p.n;
  }
  const auto *x = static_cast<const __half *>(p.x);
  const auto *w = static_cast<const __half *>(p.weight);
  const auto *bias = static_cast<const __half *>(p.bias);
  float acc;
  if constexpr (VectorWidth == 4) {
    const auto *x_row = x + row * p.x_m_stride;
    const auto *w_row = w + col * p.weight_n_stride;
    const bool half2_legal = p.x_k_stride == 1 && p.weight_k_stride == 1 &&
        (p.k & 1) == 0 &&
        (reinterpret_cast<uintptr_t>(x_row) & (alignof(__half2) - 1)) == 0 &&
        (reinterpret_cast<uintptr_t>(w_row) & (alignof(__half2) - 1)) == 0;
    if (half2_legal && p.k == 576)
      acc = simt_half2_dot<576, VectorWidth>(p, row, col, lane);
    else if (half2_legal && p.k == 1536)
      acc = simt_half2_dot<1536, VectorWidth>(p, row, col, lane);
    else if (half2_legal)
      acc = simt_half2_dot<0, VectorWidth>(p, row, col, lane);
    else
      acc = simt_scalar_dot<VectorWidth>(p, row, col, lane);
  } else {
    acc = simt_scalar_dot<VectorWidth>(p, row, col, lane);
  }
  for (int offset = 16; offset; offset >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  if (lane == 0) {
    const float c = bias ? __half2float(bias[output_index]) : 0.0f;
    static_cast<__half *>(p.output)[output_index] =
        __float2half_rn(p.alpha * acc + p.beta * c);
  }
}

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

int dispatch_profile(int strategy, int warps, int vector_width,
                     Mb3Resources *resources) {
  if (!resources || (strategy != kSerial && strategy != kSimt)) return kGridRejected;
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
                                Mb3Resources *resources) {
  if (!resources || (strategy != kSerial && strategy != kSimt)) return kGridRejected;
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
                    int vector_width, int grid, cudaStream_t stream) {
  if (p.m <= 0 || p.n <= 0 || p.k <= 0 || !p.x || !p.weight || !p.output ||
      (p.beta != 0.0f && !p.bias) || (strategy != kSerial && strategy != kSimt))
    return kGridRejected;
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
                                  int vector_width, Mb3Resources *resources) {
  return dispatch_profile(strategy, warps_per_cta, vector_width, resources);
}

extern "C" int mb3_standalone_profile(int strategy, int warps_per_cta,
                                       int vector_width,
                                       Mb3Resources *resources) {
  return dispatch_standalone_profile(strategy, warps_per_cta, vector_width,
                                     resources);
}

extern "C" int mb3_validate_owner_grid(int strategy, int warps_per_cta,
                                       int vector_width, int grid_ctas) {
  if (grid_ctas <= 0) return kGridRejected;
  Mb3Resources resources{};
  const int e = dispatch_profile(strategy, warps_per_cta, vector_width, &resources);
  if (e) return e;
  if (!resources.cooperative_launch || !resources.active_ctas_per_sm ||
      grid_ctas > resources.resident_ctas)
    return kGridRejected;
  return 0;
}

extern "C" int mb3_launch(const Mb3Contraction *problem, int strategy,
                           int warps_per_cta, int vector_width, int grid_ctas,
                           uintptr_t stream) {
  if (!problem) return kGridRejected;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (grid_ctas == -1)
    return dispatch_launch(*problem, strategy, warps_per_cta, vector_width,
                           -1, cuda_stream);
  const int e = mb3_validate_owner_grid(strategy, warps_per_cta, vector_width,
                                        grid_ctas);
  if (e) return e;
  return dispatch_launch(*problem, strategy, warps_per_cta, vector_width,
                         grid_ctas, cuda_stream);
}

extern "C" int mb3_capture(const Mb3Contraction *problem, int strategy,
                            int warps_per_cta, int vector_width, int grid_ctas,
                            int launches, uintptr_t stream, void **graph_exec) {
  if (!problem || !graph_exec || launches <= 0) return kGridRejected;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t e = cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeThreadLocal);
  if (e != cudaSuccess) return e;
  for (int i = 0; i < launches; ++i) {
    const int launch_error = mb3_launch(problem, strategy, warps_per_cta,
                                         vector_width, grid_ctas,
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
