// V3R-009 diagnostic only: empty-kernel regular/cooperative launch floor.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void mb3_empty_kernel() { asm volatile(""); }

extern "C" int mb3_launch_floor(int cooperative, int grid_ctas, int threads,
                                uintptr_t stream_value) {
  if (grid_ctas <= 0 || threads <= 0) return cudaErrorInvalidValue;
  const dim3 grid(static_cast<unsigned>(grid_ctas), 1, 1);
  const dim3 block(static_cast<unsigned>(threads), 1, 1);
  auto stream = reinterpret_cast<cudaStream_t>(stream_value);
  if (cooperative) {
    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(mb3_empty_kernel), grid, block, nullptr, 0,
        stream);
  }
  mb3_empty_kernel<<<grid, block, 0, stream>>>();
  return cudaGetLastError();
}
