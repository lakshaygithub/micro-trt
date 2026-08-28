#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

namespace {

// ---------------------------------------------------------------------------
// Tile geometry
//
// One block computes a BM x BN patch of C, marching along K in steps of BK.
// One *thread* computes a TM x TN patch of that, held entirely in registers.
//
// The whole point of this kernel is the ratio inside the innermost loop:
//   tiled kernel     -> 2 shared loads per FMA
//   this kernel      -> (TM + TN) shared loads per (TM * TN) FMAs
//                     = 8 loads per 16 FMAs = 0.5 loads per FMA, a 4x drop.
// ---------------------------------------------------------------------------
constexpr int BM = 128;  // block tile rows of C
constexpr int BN = 128;  // block tile cols of C
constexpr int BK = 16;  // depth of the K slab staged per iteration

constexpr int TM = 8;   // per-thread rows
constexpr int TN = 8;   // per-thread cols

constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

// These relationships are load-bearing: the loading loops below assume the
// thread count divides the tile sizes evenly. Assert them at compile time
// rather than discovering a silent mis-load at runtime.
static_assert(BM % TM == 0, "BM must divide evenly into TM");
static_assert(BN % TN == 0, "BN must divide evenly into TN");
static_assert((BM * BK) % THREADS == 0, "A tile must divide evenly among threads");
static_assert((BK * BN) % THREADS == 0, "B tile must divide evenly among threads");
static_assert(THREADS == 256, "block is launched as 16x16");

__global__ __launch_bounds__(THREADS)
void gemm_regtiled_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    // As is stored TRANSPOSED: [BK][BM] rather than [BM][BK].
    //
    // Reason: the compute loop reads a column of the A tile (fixed k, varying
    // row). Storing transposed turns that into a contiguous row read, which is
    // both faster and simpler to reason about for bank behaviour.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;  // first row of C this block owns
    const int block_col = blockIdx.x * BN;  // first col of C this block owns

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0 .. 255

    // ---- Global -> shared load index mapping -------------------------------
    // The A tile is BM x BK = 1024 floats and we have 256 threads, so each
    // thread fetches 4 elements, walking down in strides.
    const int a_inner_row = tid / BK;              // 0 .. 15
    const int a_inner_col = tid % BK;              // 0 .. 15
    constexpr int a_row_stride = THREADS / BK;     // 16 rows per pass

    // The B tile is BK x BN = 1024 floats. Consecutive tid maps to consecutive
    // columns, so these global reads are perfectly coalesced.
    const int b_inner_row = tid / BN;              // 0 .. 3
    const int b_inner_col = tid % BN;              // 0 .. 63
    constexpr int b_row_stride = THREADS / BN;     // 4 rows per pass

    // ---- Accumulators ------------------------------------------------------
    // TM*TN = 16 floats per thread, living in registers. This array is only
    // register-resident because every index into it is a compile-time constant
    // after the #pragma unroll directives below. If the compiler cannot prove
    // the indices, it spills this to local memory (which is global memory in
    // disguise) and the entire optimization evaporates.
    float acc[TM][TN] = {0.0f};

    // Staging registers for the operands of each rank-1 update.
    float regA[TM];
    float regB[TN];

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * BK;

        // ---- Cooperative load, A tile (written transposed) -----------------
        #pragma unroll
        for (int off = 0; off < BM; off += a_row_stride) {
            const int r = block_row + a_inner_row + off;  // row of A
            const int c = k_base + a_inner_col;           // col of A
            // Zero-pad out of range so ragged sizes need no special case in
            // the compute loop; zeros contribute nothing to a dot product.
            As[a_inner_col][a_inner_row + off] =
                (r < M && c < K) ? A[static_cast<std::size_t>(r) * K + c] : 0.0f;
        }

        // ---- Cooperative load, B tile (natural orientation) ----------------
        #pragma unroll
        for (int off = 0; off < BK; off += b_row_stride) {
            const int r = k_base + b_inner_row + off;     // row of B
            const int c = block_col + b_inner_col;        // col of B
            Bs[b_inner_row + off][b_inner_col] =
                (r < K && c < N) ? B[static_cast<std::size_t>(r) * N + c] : 0.0f;
        }

        __syncthreads();

        // ---- Compute: BK rank-1 updates of the thread's TM x TN patch ------
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Pull this thread's slice of the k-th column of A and k-th row of B
            // into registers: TM + TN = 8 shared loads.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[k][threadIdx.y * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k][threadIdx.x * TN + j];
            }

            // Outer product: TM * TN = 16 FMAs, all operands already in
            // registers. This is where the 4x ratio improvement is realised.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        // Guard the tile buffers before the next iteration overwrites them.
        __syncthreads();
    }

    // ---- Write back --------------------------------------------------------
    // Bounds checked here, after every barrier has been reached uniformly by
    // every thread in the block.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = block_row + threadIdx.y * TM + i;
            const int c = block_col + threadIdx.x * TN + j;
            if (r < M && c < N) {
                C[static_cast<std::size_t>(r) * N + c] = acc[i][j];
            }
        }
    }
}

}  // namespace

void gemm_regtiled(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();

    const dim3 block(BN / TN, BM / TM);            // (16, 16) = 256 threads
    const dim3 grid((N + BN - 1) / BN,
                    (M + BM - 1) / BM);

    gemm_regtiled_kernel<<<grid, block>>>(A.data(), B.data(), C.data(), M, N, K);
    CUDA_CHECK_LAST();
}