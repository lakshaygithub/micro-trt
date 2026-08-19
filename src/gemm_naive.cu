#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace {

// __global__ = runs on the GPU, launched from the CPU. Must return void;
// results come back through pointers you passed in.
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // Map this thread to exactly one element of C.
    //
    // blockIdx  = which block this thread's block is, within the grid
    // blockDim  = how many threads per block
    // threadIdx = this thread's position inside its block
    //
    // x maps to columns and y to rows (not the reverse) on purpose -- threadIdx.x
    // is the fastest-varying dimension, so consecutive threads get consecutive
    // columns, which makes their writes to C contiguous. That is memory
    // coalescing: the hardware merges 32 adjacent accesses into one transaction.
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Grids are launched in whole blocks, so if M or N is not a multiple of the
    // block size we launch more threads than there are output elements. Those
    // extra threads must not write anywhere.
    if (row >= M || col >= N) {
        return;
    }

    // Accumulate in a register. Writing straight into C[...] inside the loop
    // would be a global-memory round trip per iteration.
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // Row-major indexing: element (r, c) of an X-by-Y matrix is at r * Y + c.
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

}  // anonymous namespace -- gives the kernel internal linkage to this file

void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();

    // 16x16 = 256 threads per block. Blocks are scheduled as 32-thread warps,
    // so keep the count a multiple of 32; 128-512 is the usual sweet spot.
    const dim3 block(16, 16);

    // Ceiling division: we need enough blocks to cover every output element,
    // rounding up. (a + b - 1) / b is integer ceil(a / b).
    const dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);

    gemm_naive_kernel<<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K);

    // Launches are async and do not return an error code -- ask explicitly.
    CUDA_CHECK_LAST();
}
