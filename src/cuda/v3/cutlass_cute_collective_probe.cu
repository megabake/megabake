// Isolated V3R-008 CUTLASS 4.x CuTe collective body probe.
// CUTLASS headers are consumed from the installed MathDx distribution.

#include <cuda_runtime.h>

#include <limits>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cute/tensor.hpp>

using Element = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using TileShape = cute::Shape<cute::_128, cute::_64, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using Fusion = cutlass::epilogue::fusion::LinearCombination<
    Element, float, Element, float>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    Element, LayoutC, 8,
    Element, LayoutC, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    Fusion>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Element, LayoutA, 8,
    Element, LayoutB, 8,
    float,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, Mainloop, Epilogue,
    cutlass::gemm::PersistentScheduler>;

constexpr int kTileN = 64;
constexpr int kTileK = 64;

__global__ void cutlass_cute_collective_body(
    const __grid_constant__ Kernel::Params params) {
  extern __shared__ __align__(16) unsigned char storage[];
  Kernel op;
  op(params, reinterpret_cast<char *>(storage));
}

static cudaError_t make_arguments(
    const void *x, const void *weight, const void *bias, void *output,
    int64_t m, int64_t n, int64_t k, float alpha, float beta,
    Kernel::Arguments *arguments) {
  if (!x || !weight || !output || m <= 0 || n <= 0 || k <= 0 ||
      m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max() ||
      (beta != 0.0f && !bias) || n % kTileN != 0 || k % kTileK != 0) {
    return cudaErrorInvalidValue;
  }
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  StrideA stride_a = cute::make_stride(
      k, cute::_1{}, static_cast<int64_t>(m) * k);
  StrideB stride_b = cute::make_stride(
      k, cute::_1{}, static_cast<int64_t>(n) * k);
  StrideC stride_c = cute::make_stride(
      n, cute::_1{}, static_cast<int64_t>(m) * n);
  StrideD stride_d = cute::make_stride(
      n, cute::_1{}, static_cast<int64_t>(m) * n);
  cutlass::KernelHardwareInfo hardware{};
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return error;
  hardware.device_id = device;
  hardware.sm_count = properties.multiProcessorCount;
  *arguments = Kernel::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      {static_cast<Element const *>(x), stride_a,
       static_cast<Element const *>(weight), stride_b},
      {{}, bias ? static_cast<Element const *>(bias)
                : static_cast<Element const *>(output), stride_c,
       static_cast<Element *>(output), stride_d},
      hardware};
  arguments->epilogue.thread.alpha = alpha;
  arguments->epilogue.thread.beta = beta;
  return Kernel::can_implement(*arguments) ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" int cutlass_probe_profile(int *device_sms, int *cooperative,
                                     int *active_per_sm, int *resident_ctas,
                                     int *registers, int *shared_bytes,
                                     int *local_bytes) {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return static_cast<int>(error);
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return static_cast<int>(error);
  cudaFuncAttributes attributes{};
  error = cudaFuncGetAttributes(&attributes, cutlass_cute_collective_body);
  if (error != cudaSuccess) return static_cast<int>(error);
  if (Kernel::SharedStorageSize >= 48 * 1024) {
    error = cudaFuncSetAttribute(
        cutlass_cute_collective_body,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Kernel::SharedStorageSize));
    if (error != cudaSuccess) return static_cast<int>(error);
  }
  int active = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cutlass_cute_collective_body, Kernel::get_block_shape().x,
      Kernel::SharedStorageSize);
  if (error != cudaSuccess) return static_cast<int>(error);
  *device_sms = properties.multiProcessorCount;
  *cooperative = properties.cooperativeLaunch;
  *active_per_sm = active;
  *resident_ctas = active * properties.multiProcessorCount;
  *registers = attributes.numRegs;
  *shared_bytes = static_cast<int>(Kernel::SharedStorageSize);
  *local_bytes = attributes.localSizeBytes;
  return 0;
}

extern "C" int cutlass_probe_grid(
    const void *x, const void *weight, const void *bias, void *output,
    int64_t m, int64_t n, int64_t k, float alpha, float beta,
    int *grid_ctas) {
  if (!grid_ctas) return static_cast<int>(cudaErrorInvalidValue);
  Kernel::Arguments arguments{};
  cudaError_t error = make_arguments(
      x, weight, bias, output, m, n, k, alpha, beta, &arguments);
  if (error != cudaSuccess) return static_cast<int>(error);
  Kernel::Params params = Kernel::to_underlying_arguments(arguments, nullptr);
  const dim3 grid = Kernel::get_grid_shape(params);
  *grid_ctas = static_cast<int>(grid.x) * static_cast<int>(grid.y) *
               static_cast<int>(grid.z);
  return 0;
}

extern "C" int cutlass_probe_launch(
    const void *x, const void *weight, const void *bias, void *output,
    int64_t m, int64_t n, int64_t k, float alpha, float beta,
    int owner_entry, size_t stream_value) {
  Kernel::Arguments arguments{};
  cudaError_t error = make_arguments(
      x, weight, bias, output, m, n, k, alpha, beta, &arguments);
  if (error != cudaSuccess) return static_cast<int>(error);
  Kernel::Params params = Kernel::to_underlying_arguments(arguments, nullptr);
  const dim3 grid = Kernel::get_grid_shape(params);
  const dim3 block = Kernel::get_block_shape();
  const size_t shared_bytes = Kernel::SharedStorageSize;
  if (shared_bytes >= 48 * 1024) {
    error = cudaFuncSetAttribute(
        cutlass_cute_collective_body,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (error != cudaSuccess) return static_cast<int>(error);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  if (owner_entry) {
    void *kernel_arguments[] = {&params};
    error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(cutlass_cute_collective_body), grid, block,
        kernel_arguments, shared_bytes, stream);
  } else {
    cutlass_cute_collective_body<<<grid, block, shared_bytes, stream>>>(params);
    error = cudaGetLastError();
  }
  return static_cast<int>(error);
}
