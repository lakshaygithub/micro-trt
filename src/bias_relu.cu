#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace {

// The unfused epilogue: a separate pass over the whole output matrix that reads
// every element, adds the bias, applies ReLU, and writes it back.
//
// This kernel is the thing fusion eliminates. It performs almost no arithmetic
// -- one add and one comparison per element -- while moving 2 * M * N * 4 bytes
// through global memory. It is purely memory-bound, and every byte of that
// traffic is avoidable.
__global__ void bias_relu_kernel(float* __restrict__ C,
                                 const float* __restrict__ bias,
                                 int M, int N) {
    const std::size_t total = static_cast<std::size_t>(M) * N;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    // Grid-stride loop: the kernel works correctly for any grid size, so the
    // launch configuration can be chosen for occupancy rather than to exactly
    // cover the data. Standard pattern for elementwise kernels.
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {
        const int col = static_cast<int>(idx % N);
        const float v = C[idx] + bias[col];
        C[idx] = v > 0.0f ? v : 0.0f;
    }
}

}  // namespace

void bias_relu_inplace(Tensor& C, const Tensor& bias) {
    const int M = C.rows();
    const int N = C.cols();

    const int threads = 256;
    const std::size_t total = static_cast<std::size_t>(M) * N;

    // Cap the grid rather than sizing it to the data; the grid-stride loop
    // handles the rest. 1024 blocks is plenty to saturate 40 SMs.
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 1024) {
        blocks = 1024;
    }

    bias_relu_kernel<<<blocks, threads>>>(C.data(), bias.data(), M, N);
    CUDA_CHECK_LAST();
}