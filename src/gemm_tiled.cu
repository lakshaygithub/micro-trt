#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace {

// Tile edge length. Also the block dimension, so each block is TILE x TILE
// threads and every thread loads exactly one element of each tile.
//
// Shared memory used per block: 2 * TILE * TILE * sizeof(float)
//   TILE=16 -> 2 KB   TILE=32 -> 8 KB
// The T4 has 64 KB of shared memory per SM, so at 16 this is not the limit on
// how many blocks can be resident.
constexpr int TILE = 8;

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // On-chip scratchpad, shared by every thread in this block. Roughly 30-cycle
    // latency versus 400-600 for global memory. Unlike a CPU cache, what lands
    // here is entirely our decision.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // Walk the K dimension one tile at a time. Each iteration stages a TILE-wide
    // slab of A and a TILE-tall slab of B into shared memory, then consumes them.
    const int num_tiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * TILE + tx;  // column of A this thread fetches
        const int b_row = t * TILE + ty;  // row of B this thread fetches

        // Cooperative load: TILE*TILE threads each grab one element, so the whole
        // block fetches both tiles in a single coordinated step.
        //
        // Out-of-range entries are padded with zero rather than skipped. Zeros
        // contribute nothing to the dot product, which makes ragged matrix sizes
        // fall out for free with no special-case code in the inner loop.
        //
        // Both reads are coalesced: consecutive tx gives consecutive addresses.
        As[ty][tx] = (row < M   && a_col < K) ? A[row * K + a_col]   : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N)   ? B[b_row * N + col]   : 0.0f;

        // Barrier: no thread proceeds until every thread in the block has
        // finished writing. Without it, fast threads would read tile entries
        // their neighbours have not stored yet -- a race that shows up as
        // intermittent wrong answers, not a crash.
        __syncthreads();

        // The payoff. Every value read here came from shared memory, and each
        // one is reused TILE times across the block.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Second barrier, guarding the *other* direction: a thread racing ahead
        // to the next iteration must not overwrite As/Bs while slower threads are
        // still reading this iteration's values. Dropping this one is a classic
        // and very hard-to-reproduce bug.
        __syncthreads();
    }

    // Note the bounds check lives here, not at the top of the kernel. Threads
    // outside the matrix still had to participate in the loads and, critically,
    // in __syncthreads(). __syncthreads() must be reached by every thread in the
    // block; letting some return early makes the barrier undefined behaviour.
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

}  // namespace

void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();

    // Block dims must equal TILE in both directions: the kernel assumes one
    // thread per tile element.
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE,
                    (M + TILE - 1) / TILE);

    gemm_tiled_kernel<<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K);
    CUDA_CHECK_LAST();
}