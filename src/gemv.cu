#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

#include <cstddef>

namespace {

// ---------------------------------------------------------------------------
// GEMV: y = x * W (+ bias), where x is ONE row (1 x K) and W is (K x N).
//
// During decode every projection has exactly one row of input. The tiled GEMM
// still computes it correctly, but its 128-row tiles do 128 rows of
// multiply-adds for the single row that exists: 127/128 of its arithmetic
// multiplies zero padding.
//
// A GEMV has no padding and nothing to reuse: each weight is read once and
// used for exactly one multiply-add. Its speed is therefore set by how fast W
// streams in from memory, and the whole design is about bandwidth -- wide,
// coalesced loads, with enough of them in flight to keep the bus busy.
//
// Work layout. W is stored (in x out), row-major, so each row of W is
// contiguous along the output dimension. One block owns a strip of COLS = 32
// output columns and walks every row of it:
//
//   - LANES = 8 threads cover the strip's 32 columns, 4 each, with one
//     16-byte float4 load apiece: together, one 128-byte read per row.
//   - the block's 256 threads form GROUPS = 32 such teams; team g takes rows
//     g, g + 32, g + 64, ... so the block always has 32 rows in flight.
//
// Each team ends with partial sums over its own rows. A shared-memory pass then
// adds the 32 teams' partials for each column, in a fixed order, so the result
// is identical from run to run.
// ---------------------------------------------------------------------------

constexpr int COLS    = 32;               // output columns per block
constexpr int VEC     = 4;                // floats per load: one float4
constexpr int LANES   = COLS / VEC;       // threads sharing one row: 8
constexpr int THREADS = 256;
constexpr int GROUPS  = THREADS / LANES;  // rows in flight per block: 32

static_assert(THREADS % LANES == 0, "block must split into whole row teams");
static_assert(COLS <= THREADS, "the final pass needs one thread per column");

__global__ __launch_bounds__(THREADS)
void gemv_kernel(const float* __restrict__ x,
                 const float* __restrict__ W,
                 const float* __restrict__ bias,  // nullptr for no bias
                 float* __restrict__ y,
                 int K, int N) {
    // float4 stores into this array need 16-byte alignment.
    __shared__ __align__(16) float partial[GROUPS][COLS];

    const int lane  = threadIdx.x % LANES;   // which 4 columns of the strip
    const int group = threadIdx.x / LANES;   // which team of rows
    const int col   = blockIdx.x * COLS + lane * VEC;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // N is a multiple of 4 (checked by the launcher), so each float4 is either
    // wholly inside the matrix or wholly past its last column.
    if (col < N) {
        #pragma unroll 4
        for (int k = group; k < K; k += GROUPS) {
            const float xk = x[k];
            const float4 w = *reinterpret_cast<const float4*>(
                W + static_cast<std::size_t>(k) * N + col);
            acc.x += xk * w.x;
            acc.y += xk * w.y;
            acc.z += xk * w.z;
            acc.w += xk * w.w;
        }
    }

    // Every thread stores and reaches the barrier, including any past column
    // N: __syncthreads() must be reached by the whole block.
    *reinterpret_cast<float4*>(&partial[group][lane * VEC]) = acc;
    __syncthreads();

    if (threadIdx.x < COLS) {
        const int c = blockIdx.x * COLS + threadIdx.x;
        if (c < N) {
            float sum = 0.0f;
            #pragma unroll 8
            for (int g = 0; g < GROUPS; ++g) {
                sum += partial[g][threadIdx.x];
            }
            y[c] = bias != nullptr ? sum + bias[c] : sum;
        }
    }
}

void launch(const Tensor& x, const Tensor& W, const float* bias, Tensor& y) {
    const int K = W.rows();
    const int N = W.cols();
    require(x.rows() == 1 && x.cols() == K, "gemv: x must be 1 x K");
    require(y.rows() == 1 && y.cols() == N, "gemv: y must be 1 x N");
    require(N % VEC == 0, "gemv: N must be a multiple of 4 for float4 loads");

    const int blocks = (N + COLS - 1) / COLS;
    gemv_kernel<<<blocks, THREADS>>>(x.data(), W.data(), bias, y.data(), K, N);
    CUDA_CHECK_LAST();
}

}  // namespace

void gemv(const Tensor& x, const Tensor& W, Tensor& y) {
    launch(x, W, nullptr, y);
}

void gemv_bias(const Tensor& x, const Tensor& W, const Tensor& bias, Tensor& y) {
    require(bias.size() == static_cast<std::size_t>(W.cols()),
            "gemv_bias: bias needs one entry per output column");
    launch(x, W, bias.data(), y);
}
