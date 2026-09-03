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
// The ratio that matters is inside the innermost loop:
//   tiled kernel  -> 2 shared loads per FMA
//   this kernel   -> (TM + TN) loads per (TM * TN) FMAs
//                  = 16 loads per 64 FMAs = 0.25 per FMA
// ---------------------------------------------------------------------------
constexpr int BM = 128;  // block tile rows of C
constexpr int BN = 128;  // block tile cols of C
constexpr int BK = 16;   // depth of the K slab staged per iteration

constexpr int TM = 8;    // per-thread rows
constexpr int TN = 8;    // per-thread cols

constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

static_assert(BM % TM == 0, "BM must divide evenly into TM");
static_assert(BN % TN == 0, "BN must divide evenly into TN");
static_assert((BM * BK) % THREADS == 0, "A tile must divide evenly among threads");
static_assert((BK * BN) % THREADS == 0, "B tile must divide evenly among threads");
static_assert(THREADS == 256, "block is launched as 16x16");

// ---------------------------------------------------------------------------
// Epilogues
//
// An epilogue is whatever happens to a value between "the dot product is
// finished" and "store it to global memory". Templating the kernel over this
// means a fused variant costs one small struct instead of a duplicated kernel.
//
// This is the same idea CUTLASS uses, for the same reason: the expensive part
// of a fused GEMM is the GEMM, and it should not be copy-pasted per activation.
//
// operator() is __device__ (it runs on the GPU) and __forceinline__, so after
// inlining the abstraction costs nothing at runtime.
// ---------------------------------------------------------------------------

// Plain GEMM: store the accumulator unchanged.
struct Identity {
    __device__ __forceinline__ float operator()(float v, int /*col*/) const {
        return v;
    }
};

// Y = ReLU(A*B + bias), with bias broadcast across rows -- the standard shape
// of a fully connected layer. bias has one entry per output column.
struct BiasRelu {
    const float* __restrict__ bias;

    __device__ __forceinline__ float operator()(float v, int col) const {
        const float s = v + bias[col];
        return s > 0.0f ? s : 0.0f;
    }
};

// Y = A*B + bias, no activation. Needed for the final layer of a classifier,
// where a softmax follows and applying ReLU first would destroy the negative
// logits it depends on.
//
// Note what adding this variant cost: one struct, and one more instantiation
// at the bottom of the file. No kernel was duplicated. That is the payoff of
// templating the epilogue rather than copy-pasting the kernel.
struct BiasOnly {
    const float* __restrict__ bias;

    __device__ __forceinline__ float operator()(float v, int col) const {
        return v + bias[col];
    }
};

template <typename Epilogue>
__global__ __launch_bounds__(THREADS)
void gemm_regtiled_kernel(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K,
                          Epilogue epilogue) {
    // As is stored TRANSPOSED: [BK][BM] rather than [BM][BK], because the
    // compute loop reads a column of the A tile at fixed k. Transposing at load
    // time turns that strided read into a contiguous one, and costs nothing.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0 .. 255

    // ---- Global -> shared load index mapping -------------------------------
    // A tile is BM x BK = 2048 floats over 256 threads, so 8 elements each.
    const int a_inner_row = tid / BK;              // 0 .. 15
    const int a_inner_col = tid % BK;              // 0 .. 15
    constexpr int a_row_stride = THREADS / BK;     // 16 rows per pass

    // B tile is BK x BN = 2048 floats. Consecutive tid maps to consecutive
    // columns, so these global reads are perfectly coalesced.
    const int b_inner_row = tid / BN;              // 0 .. 1
    const int b_inner_col = tid % BN;              // 0 .. 127
    constexpr int b_row_stride = THREADS / BN;     // 2 rows per pass

    // ---- Accumulators ------------------------------------------------------
    // TM*TN = 64 floats per thread in registers. Register-resident only because
    // every index below is a compile-time constant after unrolling; without
    // that the array lands in local memory and the optimization inverts.
    float acc[TM][TN] = {0.0f};

    float regA[TM];
    float regB[TN];

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * BK;

        // ---- Cooperative load, A tile (written transposed) -----------------
        // Out-of-range entries are zero-padded, so ragged sizes need no special
        // case in the compute loop -- zeros contribute nothing to a dot product.
        #pragma unroll
        for (int off = 0; off < BM; off += a_row_stride) {
            const int r = block_row + a_inner_row + off;
            const int c = k_base + a_inner_col;
            As[a_inner_col][a_inner_row + off] =
                (r < M && c < K) ? A[static_cast<std::size_t>(r) * K + c] : 0.0f;
        }

        // ---- Cooperative load, B tile (natural orientation) ----------------
        #pragma unroll
        for (int off = 0; off < BK; off += b_row_stride) {
            const int r = k_base + b_inner_row + off;
            const int c = block_col + b_inner_col;
            Bs[b_inner_row + off][b_inner_col] =
                (r < K && c < N) ? B[static_cast<std::size_t>(r) * N + c] : 0.0f;
        }

        // No thread may read the tiles until every thread has finished writing.
        __syncthreads();

        // ---- Compute: BK rank-1 updates of this thread's TM x TN patch -----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[k][threadIdx.y * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k][threadIdx.x * TN + j];
            }

            // Outer product: TM + TN loads feed TM * TN multiply-adds.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        // Guards the other direction: a thread racing ahead must not overwrite
        // the tiles while slower threads are still reading them.
        __syncthreads();
    }

    // ---- Write back, through the epilogue ----------------------------------
    // Bounds checked here rather than at the top of the kernel: every thread
    // must reach every __syncthreads(), so out-of-range threads participate
    // fully and are filtered only at the store.
    //
    // The epilogue runs on a value already sitting in a register. Whatever it
    // does costs no extra global memory traffic at all.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = block_row + threadIdx.y * TM + i;
            const int c = block_col + threadIdx.x * TN + j;
            if (r < M && c < N) {
                C[static_cast<std::size_t>(r) * N + c] = epilogue(acc[i][j], c);
            }
        }
    }
}

// Shared launch configuration for every instantiation.
template <typename Epilogue>
void launch(const Tensor& A, const Tensor& B, Tensor& C, Epilogue epilogue) {
    const int M = A.rows();
    const int K = A.cols();
    const int N = B.cols();

    const dim3 block(BN / TN, BM / TM);   // (16, 16) = 256 threads
    const dim3 grid((N + BN - 1) / BN,
                    (M + BM - 1) / BM);

    gemm_regtiled_kernel<<<grid, block>>>(A.data(), B.data(), C.data(),
                                          M, N, K, epilogue);
    CUDA_CHECK_LAST();
}

}  // namespace

void gemm_regtiled(const Tensor& A, const Tensor& B, Tensor& C) {
    launch(A, B, C, Identity{});
}

void gemm_bias_relu_fused(const Tensor& A, const Tensor& B,
                          const Tensor& bias, Tensor& C) {
    launch(A, B, C, BiasRelu{bias.data()});
}

void gemm_bias(const Tensor& A, const Tensor& B,
               const Tensor& bias, Tensor& C) {
    launch(A, B, C, BiasOnly{bias.data()});
}