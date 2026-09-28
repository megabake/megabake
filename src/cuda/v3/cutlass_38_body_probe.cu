// Isolated V3R-008 CUTLASS 3.8 device-callable body experiment.
// CUTLASS source is consumed from the installed distribution and is not copied.

#include <cuda_runtime.h>

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/tensor_ref.h>

using Element = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
    Element, 8, float, float>;
using Gemm = cutlass::gemm::device::Gemm<
    Element, LayoutA, Element, LayoutB, Element, LayoutC, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, OutputOp>;
using Kernel = typename Gemm::GemmKernel;
constexpr int kThreads = 128;
constexpr size_t kSharedBytes = sizeof(typename Kernel::SharedStorage);

__global__ void cutlass_device_callable(
    const __grid_constant__ typename Kernel::Params params) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto &shared = *reinterpret_cast<typename Kernel::SharedStorage *>(storage);
  Kernel op;
  op(params, shared);
}

extern "C" int cutlass_probe_profile(int *device_sms, int *cooperative,
                                     int *active_per_sm, int *resident,
                                     int *registers, int *shared_bytes,
                                     int *local_bytes) {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return error;
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) return error;
  cudaFuncAttributes attributes{};
  error = cudaFuncGetAttributes(&attributes, cutlass_device_callable);
  if (error != cudaSuccess) return error;
  int active = 0;
  error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, cutlass_device_callable, kThreads, kSharedBytes);
  if (error != cudaSuccess) return error;
  *device_sms = properties.multiProcessorCount;
  *cooperative = properties.cooperativeLaunch;
  *active_per_sm = active;
  *resident = active * properties.multiProcessorCount;
  *registers = attributes.numRegs;
  *shared_bytes = static_cast<int>(kSharedBytes);
  *local_bytes = attributes.localSizeBytes;
  return 0;
}

extern "C" int cutlass_probe_launch(
    const void *x, const void *weight, const void *bias, void *output,
    int64_t m, int64_t n, int64_t k, float alpha, float beta,
    int owner_entry, uintptr_t stream_value) {
  if (!x || !weight || !output || m <= 0 || n <= 0 || k <= 0 ||
      (beta != 0.0f && !bias)) return static_cast<int>(cudaErrorInvalidValue);
  const cutlass::gemm::GemmCoord problem(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  const cutlass::gemm::GemmCoord grid_shape(
      (static_cast<int>(m) + 127) / 128,
      (static_cast<int>(n) + 63) / 64, 1);
  const dim3 grid = Kernel::ThreadblockSwizzle::get_grid_shape(grid_shape);
  using RefA = typename Kernel::Mma::IteratorA::TensorRef;
  using RefB = typename Kernel::Mma::IteratorB::TensorRef;
  using RefC = typename Kernel::Epilogue::OutputTileIterator::TensorRef;
  using RefAElement = typename RefA::Element;
  using RefBElement = typename RefB::Element;
  using RefCElement = typename RefC::Element;
  auto ref_a = RefA(
      const_cast<RefAElement *>(static_cast<RefAElement const *>(x)),
      typename RefA::Layout(static_cast<int>(k)));
  auto ref_b = RefB(
      const_cast<RefBElement *>(static_cast<RefBElement const *>(weight)),
      typename RefB::Layout(static_cast<int>(k)));
  // beta==0 permits a null C pointer; CUTLASS's epilogue does not load C then.
  auto ref_c = RefC(
      bias ? const_cast<RefCElement *>(static_cast<RefCElement const *>(bias))
           : static_cast<RefCElement *>(output),
      typename RefC::Layout(static_cast<int>(n)));
  auto ref_d = RefC(static_cast<RefCElement *>(output),
                    typename RefC::Layout(static_cast<int>(n)));
  typename Kernel::Params params(
      problem, grid_shape, ref_a, ref_b, ref_c, ref_d,
      OutputOp::Params(alpha, beta));
  const dim3 block(kThreads, 1, 1);
  auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  cudaError_t error;
  if (owner_entry) {
    void *arguments[] = {&params};
    error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(cutlass_device_callable), grid, block,
        arguments, kSharedBytes, stream);
  } else {
    cutlass_device_callable<<<grid, block, kSharedBytes, stream>>>(params);
    error = cudaGetLastError();
  }
  return static_cast<int>(error);
}
