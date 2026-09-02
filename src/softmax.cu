#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"

#include <cfloat>

namespace {

constexpr int BLOCK  = 256;
constexpr int NWARPS = BLOCK / 32;   // 8

static_assert(BLOCK % 32 == 0, "block must be a whole number of warps");

// ---------------------------------------------------------------------------
// Warp-level reductions
//
// A warp is 32 threads executing in lockstep, and they can exchange register
// values directly with each other -- no shared memory, no barrier. That is what
// __shfl_xor_sync does: every thread swaps its value with the thread whose lane
// index differs in one bit, given by `offset`.
//
// Five rounds of this (offsets 16, 8, 4, 2, 1) is a butterfly pattern that
// leaves the fully reduced result in EVERY lane, which saves a separate
// broadcast step. The 0xffffffff mask says all 32 lanes participate.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Block-level reductions
//
// Two stages: reduce within each warp using shuffles, then have one thread per
// warp write its result to shared memory and let the first warp reduce those.
//
// `smem` must have room for NWARPS + 1 floats: NWARPS staging slots plus one
// broadcast slot. Keeping the broadcast slot separate is what makes it safe to
// call these functions twice in a row without an extra barrier in between.
// ---------------------------------------------------------------------------

__device__ float block_reduce_max(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();

    // All 32 lanes of warp 0 must call the shuffle, so out-of-range lanes get
    // an identity value rather than skipping the call.
    v = (threadIdx.x < NWARPS) ? smem[threadIdx.x] : -FLT_MAX;
    if (wid == 0) {
        v = warp_reduce_max(v);
        if (lane == 0) smem[NWARPS] = v;
    }
    __syncthreads();
    return smem[NWARPS];
}

__device__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        smem[wid] = v;
    }
    __syncthreads();

    v = (threadIdx.x < NWARPS) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
        if (lane == 0) smem[NWARPS] = v;
    }
    __syncthreads();
    return smem[NWARPS];
}

// ---------------------------------------------------------------------------
// v1: one thread per row.
//
// Deliberately bad, in two separate ways:
//
//  1. Three full passes over the row -- one to find the max, one to sum the
//     exponentials, one to write the result.
//  2. Catastrophically uncoalesced. Thread 0 walks row 0 while thread 1 walks
//     row 1, so at any instant the 32 threads of a warp are reading addresses
//     N floats apart. Every access becomes its own memory transaction.
// ---------------------------------------------------------------------------
__global__ void softmax_naive_kernel(const float* __restrict__ X,
                                     float* __restrict__ Y,
                                     int M, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const float* xr = X + static_cast<std::size_t>(row) * N;
    float*       yr = Y + static_cast<std::size_t>(row) * N;

    float m = -FLT_MAX;
    for (int j = 0; j < N; ++j) {
        m = fmaxf(m, xr[j]);
    }

    float s = 0.0f;
    for (int j = 0; j < N; ++j) {
        s += expf(xr[j] - m);
    }

    const float inv = 1.0f / s;
    for (int j = 0; j < N; ++j) {
        yr[j] = expf(xr[j] - m) * inv;
    }
}

// ---------------------------------------------------------------------------
// v2: one block per row, cooperative reduction.
//
// Now the 256 threads of a block share a single row and stride across it, so
// consecutive threads touch consecutive addresses -- fully coalesced. The max
// and the sum each become a block-wide reduction.
//
// Still three passes over global memory, but each one is now efficient.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK)
void softmax_block_kernel(const float* __restrict__ X,
                          float* __restrict__ Y,
                          int M, int N) {
    __shared__ float smem[NWARPS + 1];

    const int row = blockIdx.x;
    if (row >= M) return;

    const float* xr = X + static_cast<std::size_t>(row) * N;
    float*       yr = Y + static_cast<std::size_t>(row) * N;

    // Pass 1: row maximum.
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        local_max = fmaxf(local_max, xr[j]);
    }
    const float row_max = block_reduce_max(local_max, smem);

    // Pass 2: sum of exp(x - max).
    //
    // Subtracting the maximum first is not an optimization, it is a correctness
    // requirement. expf(90.0f) already overflows to infinity in fp32. Shifting
    // by the row max makes the largest exponent exactly exp(0) = 1, so nothing
    // can overflow, and the shift cancels in the division below.
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        local_sum += expf(xr[j] - row_max);
    }
    const float row_sum = block_reduce_sum(local_sum, smem);

    // Pass 3: normalise and write.
    const float inv = 1.0f / row_sum;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        yr[j] = expf(xr[j] - row_max) * inv;
    }
}

// ---------------------------------------------------------------------------
// v3: online (single-pass) softmax.
//
// The two statistics softmax needs -- the maximum and the sum of shifted
// exponentials -- look like they require two passes, because you cannot start
// summing until you know the maximum.
//
// You can, if you are willing to correct the running sum whenever the maximum
// changes. Given a running pair (m, s) and a new value x:
//
//     m' = max(m, x)
//     s' = s * exp(m - m') + exp(x - m')
//
// The factor exp(m - m') rescales everything summed so far into the new
// reference point. When the max does not change, m' == m and the factor is 1.
//
// This is associative, so it also serves as the combining rule for the parallel
// reduction: two partial pairs merge the same way. This recurrence is the core
// idea behind FlashAttention.
//
// Result: one pass to compute (max, sum), one pass to write. Three passes
// become two.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void combine_online(float& m, float& s,
                                               float m_other, float s_other) {
    const float m_new = fmaxf(m, m_other);
    s = s * __expf(m - m_new) + s_other * __expf(m_other - m_new);
    m = m_new;
}

__device__ __forceinline__ void warp_reduce_online(float& m, float& s) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_xor_sync(0xffffffffu, m, off);
        const float s_o = __shfl_xor_sync(0xffffffffu, s, off);
        combine_online(m, s, m_o, s_o);
    }
}

__device__ void block_reduce_online(float& m, float& s, float* sm, float* ss) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;

    warp_reduce_online(m, s);
    if (lane == 0) {
        sm[wid] = m;
        ss[wid] = s;
    }
    __syncthreads();

    // Identity element for this reduction: the most negative finite float with
    // a zero sum. Using -INFINITY here would produce (-inf) - (-inf) = NaN when
    // two empty slots combine; -FLT_MAX keeps the arithmetic finite.
    m = (threadIdx.x < NWARPS) ? sm[threadIdx.x] : -FLT_MAX;
    s = (threadIdx.x < NWARPS) ? ss[threadIdx.x] : 0.0f;

    if (wid == 0) {
        warp_reduce_online(m, s);
        if (lane == 0) {
            sm[NWARPS] = m;
            ss[NWARPS] = s;
        }
    }
    __syncthreads();
    m = sm[NWARPS];
    s = ss[NWARPS];
}

__global__ __launch_bounds__(BLOCK)
void softmax_online_kernel(const float* __restrict__ X,
                           float* __restrict__ Y,
                           int M, int N) {
    __shared__ float sm[NWARPS + 1];
    __shared__ float ss[NWARPS + 1];

    const int row = blockIdx.x;
    if (row >= M) return;

    const float* xr = X + static_cast<std::size_t>(row) * N;
    float*       yr = Y + static_cast<std::size_t>(row) * N;

    // Single fused pass: track running max and running sum together.
    float m = -FLT_MAX;
    float s = 0.0f;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        combine_online(m, s, xr[j], 1.0f);   // a lone value has sum exp(x-x) = 1
    }

    block_reduce_online(m, s, sm, ss);

    const float inv = 1.0f / s;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        yr[j] = __expf(xr[j] - m) * inv;
    }
}

}  // namespace

void softmax_naive(const Tensor& X, Tensor& Y) {
    const int M = X.rows();
    const int N = X.cols();
    const int grid = (M + BLOCK - 1) / BLOCK;
    softmax_naive_kernel<<<grid, BLOCK>>>(X.data(), Y.data(), M, N);
    CUDA_CHECK_LAST();
}

void softmax_block(const Tensor& X, Tensor& Y) {
    const int M = X.rows();
    const int N = X.cols();
    softmax_block_kernel<<<M, BLOCK>>>(X.data(), Y.data(), M, N);
    CUDA_CHECK_LAST();
}

void softmax_online(const Tensor& X, Tensor& Y) {
    const int M = X.rows();
    const int N = X.cols();
    softmax_online_kernel<<<M, BLOCK>>>(X.data(), Y.data(), M, N);
    CUDA_CHECK_LAST();
}