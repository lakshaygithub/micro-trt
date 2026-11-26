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
// coalesced loads, with enough of them in flight to keep the bus busy. That is
// also why storing W in fp16 pays: half the bytes to stream for the same work.
//
// Work layout. W is stored (in x out), row-major, so each row of W is
// contiguous along the output dimension. One block owns a strip of COLS output
// columns and walks every row of it. Every thread reads 16 bytes at a time:
// four fp32 weights or eight fp16 weights. LANES threads cover one row of the
// strip, the block's 256 threads form GROUPS such teams, and team g takes rows
// g, g + GROUPS, g + 2*GROUPS, ...
//
//                               fp32     fp16     fp16, wide W
//   columns per block, COLS      32       32          64
//   weights per load, VEC         4        8           8
//   threads per row, LANES        8        4           8
//   bytes read per row          128       64         128
//   rows in flight, GROUPS       32       64          32
//
// The strip width trades two needs against each other. A team should read
// whole 128-byte rows: profiled on a T4, fp16 with 64-byte rows streamed the
// LM head at 57% of peak bandwidth, where fp32 with 128-byte rows reached 73%.
// But every SM needs a block to run, and the narrow projections already launch
// too few blocks to fill a large GPU, so halving the count everywhere would
// cost more than it saves. fp16 therefore uses 64-column strips only where
// they still give every SM at least one block -- on a T4, the MLP's gate and
// up projections and the LM head -- and 32-column strips everywhere else.
//
// Each team ends with partial sums over its own rows. A shared-memory pass then
// adds the teams' partials for each column, in a fixed order, so the result is
// identical from run to run. All arithmetic is fp32 in every version; fp16
// weights are widened to fp32 in registers before they are used.
// ---------------------------------------------------------------------------

constexpr int THREADS = 256;

// Reads the 16 bytes at p as weights, widened to fp32: four floats as they are,
// or eight fp16 values converted.
__device__ __forceinline__ void load_weights(const float* p, float (&w)[4]) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    w[0] = v.x;
    w[1] = v.y;
    w[2] = v.z;
    w[3] = v.w;
}

__device__ __forceinline__ void load_weights(const __half* p, float (&w)[8]) {
    // One 16-byte load, then the same 16 bytes viewed as four __half2 pairs.
    // The lower half of each pair is the lower address, i.e. the left column.
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    const __half2* pairs = reinterpret_cast<const __half2*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(pairs[i]);
        w[2 * i]     = f.x;
        w[2 * i + 1] = f.y;
    }
}

template <typename WT, int COLS>  // weight type (float or __half), strip width
__global__ __launch_bounds__(THREADS)
void gemv_kernel(const float* __restrict__ x,
                 const WT* __restrict__ W,
                 const float* __restrict__ bias,  // nullptr for no bias
                 float* __restrict__ y,
                 int K, int N) {
    constexpr int VEC    = static_cast<int>(16 / sizeof(WT));  // weights per load
    constexpr int LANES  = COLS / VEC;                         // threads per row
    constexpr int GROUPS = THREADS / LANES;                    // rows in flight
    static_assert(COLS % VEC == 0 && THREADS % LANES == 0,
                  "block must split into whole row teams");
    static_assert(COLS <= THREADS, "the final pass needs one thread per column");

    // float4 stores into this array need 16-byte alignment.
    __shared__ __align__(16) float partial[GROUPS][COLS];

    const int lane  = threadIdx.x % LANES;   // which VEC columns of the strip
    const int group = threadIdx.x / LANES;   // which team of rows
    const int col   = blockIdx.x * COLS + lane * VEC;

    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) {
        acc[v] = 0.0f;
    }

    // N is a multiple of VEC (checked by the launcher), so each 16-byte load is
    // either wholly inside the matrix or wholly past its last column.
    if (col < N) {
        #pragma unroll 4
        for (int k = group; k < K; k += GROUPS) {
            const float xk = x[k];
            float w[VEC];
            load_weights(W + static_cast<std::size_t>(k) * N + col, w);
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                acc[v] += xk * w[v];
            }
        }
    }

    // Every thread stores and reaches the barrier, including any past column
    // N: __syncthreads() must be reached by the whole block.
    #pragma unroll
    for (int v = 0; v < VEC; v += 4) {
        *reinterpret_cast<float4*>(&partial[group][lane * VEC + v]) =
            make_float4(acc[v], acc[v + 1], acc[v + 2], acc[v + 3]);
    }
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

// Streaming multiprocessors on the current GPU, read once.
int sm_count() {
    static const int count = [] {
        int dev = 0;
        int n = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
        return n;
    }();
    return count;
}

// W points at a K x N matrix of fp32 or fp16 weights; the kernel version is
// picked from its type, and for fp16 also from the matrix's width.
template <typename WT>
void launch(const Tensor& x, const WT* W, int K, int N, const Tensor* bias, Tensor& y) {
    constexpr int VEC = static_cast<int>(16 / sizeof(WT));
    require(x.rows() == 1 && x.cols() == K, "gemv: x must be 1 x K");
    require(y.rows() == 1 && y.cols() == N, "gemv: y must be 1 x N");
    require(N % VEC == 0, VEC == 4 ? "gemv: N must be a multiple of 4 for float4 loads"
                                   : "gemv: N must be a multiple of 8 for 16-byte fp16 loads");
    require(bias == nullptr || bias->size() == static_cast<std::size_t>(N),
            "gemv_bias: bias needs one entry per output column");
    const float* b = bias != nullptr ? bias->data() : nullptr;

    // fp16 takes 64-column strips, and so 128-byte rows, wherever that still
    // leaves every SM at least one block (see the table at the top).
    if constexpr (sizeof(WT) == 2) {
        if (N / 64 >= sm_count()) {
            gemv_kernel<WT, 64><<<(N + 63) / 64, THREADS>>>(x.data(), W, b, y.data(), K, N);
            CUDA_CHECK_LAST();
            return;
        }
    }
    gemv_kernel<WT, 32><<<(N + 31) / 32, THREADS>>>(x.data(), W, b, y.data(), K, N);
    CUDA_CHECK_LAST();
}

}  // namespace

void gemv(const Tensor& x, const Tensor& W, Tensor& y) {
    launch(x, W.data(), W.rows(), W.cols(), nullptr, y);
}

void gemv_bias(const Tensor& x, const Tensor& W, const Tensor& bias, Tensor& y) {
    launch(x, W.data(), W.rows(), W.cols(), &bias, y);
}

void gemv(const Tensor& x, const HalfTensor& W, Tensor& y) {
    launch(x, W.data(), W.rows(), W.cols(), nullptr, y);
}

void gemv_bias(const Tensor& x, const HalfTensor& W, const Tensor& bias, Tensor& y) {
    launch(x, W.data(), W.rows(), W.cols(), &bias, y);
}
