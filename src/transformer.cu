#include "kernels.h"
#include "tensor.h"
#include "cuda_utils.h"
#include "reduce.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>

// ---------------------------------------------------------------------------
// Transformer building blocks for a Qwen2 / Llama-family decoder.
//
// Each kernel is the GPU version of one function in tests/check_qwen.cpp, the
// CPU forward pass that matches Hugging Face at every layer. That file is the
// specification; tests/check_qwen_gpu.cu holds these kernels to it.
//
// None of these is the expensive part. At the sequence lengths used here the
// projections -- plain GEMMs from the earlier stages -- are almost all of the
// arithmetic and memory traffic. These kernels only have to be correct and
// cheap; fusing them into GEMM epilogues and FlashAttention come later.
// ---------------------------------------------------------------------------

namespace {

constexpr int WARP = 32;

// Elementwise kernels use grid-stride loops; capping the grid keeps a huge
// tensor from launching millions of blocks that each touch one element.
int grid_for(std::size_t n, int threads) {
    const std::size_t blocks = (n + threads - 1) / threads;
    return static_cast<int>(std::min<std::size_t>(blocks, 4096));
}

// ---- Embedding --------------------------------------------------------------
// One block per token, copying row tokens[t] of the table. Pure data movement.
__global__ void embedding_kernel(const float* __restrict__ table,
                                 const int* __restrict__ tokens,
                                 float* __restrict__ out, int cols) {
    const int t = blockIdx.x;
    const float* src = table + static_cast<std::size_t>(tokens[t]) * cols;
    float*       dst = out + static_cast<std::size_t>(t) * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ---- RMSNorm ----------------------------------------------------------------
// One warp per row. A row is only `hidden` floats (896 for Qwen2.5-0.5B), so a
// warp covers it in 28 strided steps and the sum of squares needs nothing
// beyond warp shuffles -- no shared memory, no __syncthreads.
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ weight,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP;
    if (row >= rows) return;  // row is the same for all 32 lanes: whole warp exits

    const float* xr = x + static_cast<std::size_t>(row) * cols;
    float*       yr = y + static_cast<std::size_t>(row) * cols;

    float ss = 0.0f;
    for (int i = lane; i < cols; i += WARP) {
        ss += xr[i] * xr[i];
    }
    ss = warp_reduce_sum(ss);

    // Same operation order as Hugging Face: weight * (x * rsqrt(mean + eps)).
    const float inv = rsqrtf(ss / static_cast<float>(cols) + eps);
    for (int i = lane; i < cols; i += WARP) {
        yr[i] = weight[i] * (xr[i] * inv);
    }
}

// ---- Rotary position embedding --------------------------------------------
// One thread per rotated pair. Dimension i is paired with i + head_dim/2
// ("rotate_half"), the convention Qwen and Llama were trained with -- not the
// adjacent pairs (2i, 2i+1) of the original RoPE paper.
//
// The angle is formed the way Hugging Face forms it, as an fp32 product of the
// position and a precomputed fp32 inverse frequency. Computing it in higher
// precision would be "more accurate" and would disagree with the model's own
// reference by a phase error that grows with position.
__global__ void rope_kernel(float* __restrict__ x,
                            const float* __restrict__ inv_freq,
                            int rows, int n_heads, int head_dim, int pos) {
    const int half = head_dim / 2;
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * n_heads * half) return;

    const int i    = idx % half;   // which pair within the head
    const int head = idx / half;   // flattened (row, head) index
    const int row  = head / n_heads;

    float* v = x + static_cast<std::size_t>(head) * head_dim;
    float s, c;
    sincosf(static_cast<float>(pos + row) * inv_freq[i], &s, &c);

    const float a = v[i];
    const float b = v[i + half];
    v[i]        = a * c - b * s;
    v[i + half] = b * c + a * s;
}

// ---- Attention ----------------------------------------------------------------
// Causal grouped-query attention over the KV cache, one warp per (query, head).
//
// Lane l owns dimensions l, l+32, ... of the head. For each cached key the warp
// forms the q.k dot product cooperatively -- each lane multiplies its own
// dimensions, then warp_reduce_sum hands every lane the full score -- so the
// K and V reads are contiguous across the warp and fully coalesced.
//
// The softmax is the online form from the softmax stage: a running max m, a
// running denominator l, and a running weighted sum of V, all rescaled by
// exp(m_old - m_new) whenever the max grows. One pass over the keys and no
// buffer of scores: the recurrence FlashAttention is built on, at the scale of
// a single warp.
//
// Query head h reads KV head h / (heads / kv_heads): consecutive groups of
// query heads share one cached K/V head, matching Hugging Face's repeat_kv.
template <int HEAD_DIM>
__global__ void attention_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k_cache,
                                 const float* __restrict__ v_cache,
                                 float* __restrict__ out,
                                 int rows, int pos, int heads, int kv_heads,
                                 float scale) {
    static_assert(HEAD_DIM % WARP == 0, "head_dim must be a multiple of 32");
    constexpr int PER_LANE = HEAD_DIM / WARP;

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP;
    if (warp >= rows * heads) return;  // warp-uniform, like the RMSNorm exit

    const int t      = warp / heads;
    const int h      = warp % heads;
    const int kv_h   = h / (heads / kv_heads);
    const int kv_dim = kv_heads * HEAD_DIM;

    // The 1/sqrt(head_dim) scale is folded into q. For head_dim 64 it is 1/8, a
    // power of two, so the product is exact and no rounding changes.
    const float* qr = q + (static_cast<std::size_t>(t) * heads + h) * HEAD_DIM;
    float qv[PER_LANE];
    #pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        qv[j] = qr[lane + j * WARP] * scale;
    }

    float m = -FLT_MAX;   // running max of the scores
    float l = 0.0f;       // running sum of exp(score - m)
    float acc[PER_LANE];  // running sum of exp(score - m) * v
    #pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        acc[j] = 0.0f;
    }

    // Causal: the query at absolute position pos + t sees keys 0 .. pos + t.
    const int n_keys = pos + t + 1;
    for (int s = 0; s < n_keys; ++s) {
        const std::size_t base = static_cast<std::size_t>(s) * kv_dim + kv_h * HEAD_DIM;

        float dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            dot += qv[j] * k_cache[base + lane + j * WARP];
        }
        const float score = warp_reduce_sum(dot);

        const float m_new      = fmaxf(m, score);
        const float correction = expf(m - m_new);   // 1 unless the max grew
        const float p          = expf(score - m_new);
        l = l * correction + p;
        #pragma unroll
        for (int j = 0; j < PER_LANE; ++j) {
            acc[j] = acc[j] * correction + p * v_cache[base + lane + j * WARP];
        }
        m = m_new;
    }

    float* o = out + (static_cast<std::size_t>(t) * heads + h) * HEAD_DIM;
    const float inv = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < PER_LANE; ++j) {
        o[lane + j * WARP] = acc[j] * inv;
    }
}

// ---- SwiGLU -------------------------------------------------------------------
// gate = silu(gate) * up, where silu(g) = g * sigmoid(g) = g / (1 + exp(-g)).
// For very negative g, exp(-g) overflows to +inf and g / inf is a correct -0.0
// rather than a NaN, so no clamping is needed.
__global__ void silu_mul_kernel(float* __restrict__ gate,
                                const float* __restrict__ up, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float g = gate[i];
        gate[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// ---- Residual -----------------------------------------------------------------
__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y,
                           std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        x[i] += y[i];
    }
}

}  // namespace

void embedding_lookup(const Tensor& table, const int* d_tokens, Tensor& out) {
    require(out.cols() == table.cols(), "embedding_lookup: out must have the table's width");
    embedding_kernel<<<out.rows(), 256>>>(table.data(), d_tokens, out.data(), out.cols());
    CUDA_CHECK_LAST();
}

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    require(weight.rows() == 1 && weight.cols() == x.cols(), "rmsnorm: weight must be 1 x cols");
    require(y.rows() == x.rows() && y.cols() == x.cols(), "rmsnorm: y must match x");

    constexpr int threads = 128;  // 4 warps: 4 rows per block
    const int blocks = (x.rows() * WARP + threads - 1) / threads;
    rmsnorm_kernel<<<blocks, threads>>>(x.data(), weight.data(), y.data(),
                                        x.rows(), x.cols(), eps);
    CUDA_CHECK_LAST();
}

void rope_inplace(Tensor& x, int n_heads, int pos, const Tensor& inv_freq) {
    const int head_dim = x.cols() / n_heads;
    require(head_dim * n_heads == x.cols() && head_dim % 2 == 0,
            "rope: cols must be n_heads * an even head_dim");
    require(inv_freq.size() == static_cast<std::size_t>(head_dim / 2),
            "rope: inv_freq needs head_dim / 2 entries");

    constexpr int threads = 256;
    const int pairs = x.rows() * n_heads * (head_dim / 2);
    rope_kernel<<<(pairs + threads - 1) / threads, threads>>>(
        x.data(), inv_freq.data(), x.rows(), n_heads, head_dim, pos);
    CUDA_CHECK_LAST();
}

void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               int pos, int heads, int kv_heads, Tensor& out) {
    const int head_dim = q.cols() / heads;
    require(head_dim * heads == q.cols(), "attention: q must be T x heads*head_dim");
    require(kv_heads > 0 && heads % kv_heads == 0,
            "attention: heads must be a multiple of kv_heads");
    require(k_cache.cols() == kv_heads * head_dim && v_cache.cols() == k_cache.cols(),
            "attention: caches must be max_seq x kv_heads*head_dim");
    require(v_cache.rows() == k_cache.rows() && pos >= 0 && pos + q.rows() <= k_cache.rows(),
            "attention: positions exceed the cache");
    require(out.rows() == q.rows() && out.cols() == q.cols(), "attention: out must match q");

    constexpr int threads = 128;  // 4 warps, one (query, head) pair each
    const int blocks = (q.rows() * heads * WARP + threads - 1) / threads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // head_dim is a template parameter so each lane's slice of q, and its
    // accumulators, are fixed-size arrays the compiler can keep in registers.
    switch (head_dim) {
    case 64:
        attention_kernel<64><<<blocks, threads>>>(q.data(), k_cache.data(), v_cache.data(),
                                                  out.data(), q.rows(), pos, heads,
                                                  kv_heads, scale);
        break;
    case 128:
        attention_kernel<128><<<blocks, threads>>>(q.data(), k_cache.data(), v_cache.data(),
                                                   out.data(), q.rows(), pos, heads,
                                                   kv_heads, scale);
        break;
    default:
        require(false, "attention: head_dim must be 64 or 128");
    }
    CUDA_CHECK_LAST();
}

void silu_mul_inplace(Tensor& gate, const Tensor& up) {
    require(gate.rows() == up.rows() && gate.cols() == up.cols(),
            "silu_mul: gate and up must have the same shape");
    constexpr int threads = 256;
    silu_mul_kernel<<<grid_for(gate.size(), threads), threads>>>(gate.data(), up.data(),
                                                                 gate.size());
    CUDA_CHECK_LAST();
}

void add_inplace(Tensor& x, const Tensor& y) {
    require(x.rows() == y.rows() && x.cols() == y.cols(), "add: x and y must have the same shape");
    constexpr int threads = 256;
    add_kernel<<<grid_for(x.size(), threads), threads>>>(x.data(), y.data(), x.size());
    CUDA_CHECK_LAST();
}
