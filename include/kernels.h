#pragma once

class Tensor;

// C = A * B, where A is (M x K), B is (K x N), C is (M x N), all row-major.
//
// Every variant computes the same result; they differ only in how they use the
// GPU memory hierarchy. Comparing them is the point of this project.

// All plain GEMM kernels share this signature, so the benchmark harness can
// treat them interchangeably.
using GemmFn = void (*)(const Tensor& A, const Tensor& B, Tensor& C);

// v1: one thread per output element, reading A and B straight from global
// memory. Correct, and deliberately slow -- each element of A is re-read N times
// and each element of B M times.
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// v2: same work, staged through shared memory in TILE x TILE blocks. Cuts global
// memory traffic by a factor of TILE.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C);

// v3: adds register blocking on top of shared-memory tiling. Each thread
// computes an 8x8 patch of C held in registers, cutting the inner loop from
// 2 shared-memory loads per multiply-add down to 0.25.
void gemm_regtiled(const Tensor& A, const Tensor& B, Tensor& C);

// ---------------------------------------------------------------------------
// Fused epilogue
//
// A fully connected layer computes Y = ReLU(X * W + b). Done as three separate
// kernels, the bias and activation passes each read and rewrite the entire
// output matrix through global memory while doing almost no arithmetic.
//
// Fusing applies both to the accumulator while it is still in a register, so
// they cost no global memory traffic at all.
// ---------------------------------------------------------------------------

// Unfused: a standalone pass over C. This is the cost fusion removes.
// `bias` is a 1 x N tensor, broadcast across rows.
void bias_relu_inplace(Tensor& C, const Tensor& bias);

// Fused: C = ReLU(A * B + bias), in a single kernel with a single pass over C.
void gemm_bias_relu_fused(const Tensor& A, const Tensor& B,
                          const Tensor& bias, Tensor& C);

// Fused: C = A * B + bias, with no activation. Needed for the output layer of a
// classifier, where a softmax follows and ReLU would clamp away the negative
// logits it operates on.
void gemm_bias(const Tensor& A, const Tensor& B,
               const Tensor& bias, Tensor& C);

// ---------------------------------------------------------------------------
// Row-wise softmax:  Y[r][c] = exp(X[r][c] - max_r) / sum_c exp(X[r][c] - max_r)
//
// Unlike GEMM, this is memory-bound: it touches every element twice and does
// almost no arithmetic, so the metric that matters is achieved bandwidth rather
// than FLOP/s.
//
// It also needs something none of the GEMM kernels did -- threads cooperating
// to compute a value (the row max, the row sum) that spans the whole row.
// ---------------------------------------------------------------------------

using UnaryFn = void (*)(const Tensor& X, Tensor& Y);

// v1: one thread per row. Three passes over the row, and badly uncoalesced --
// threads in a warp read addresses N floats apart.
void softmax_naive(const Tensor& X, Tensor& Y);

// v2: one block per row. Coalesced access, with the max and sum computed by
// block-wide reductions built on warp shuffles. Still three passes.
void softmax_block(const Tensor& X, Tensor& Y);

// v3: online (single-pass) reduction. Tracks the running max and running sum
// together, rescaling the sum whenever the max changes, so the statistics need
// one pass instead of two. Same recurrence FlashAttention is built on.
void softmax_online(const Tensor& X, Tensor& Y);

// ---------------------------------------------------------------------------
// Transformer building blocks (Qwen2 / Llama-family decoder)
//
// Each is the GPU counterpart of one function in tests/check_qwen.cpp, the CPU
// forward pass that matches Hugging Face layer by layer and serves as their
// specification. Activations are (tokens x features), one row per token.
// ---------------------------------------------------------------------------

// out[t] = table[tokens[t]] for every row t of out. `d_tokens` is a device
// array of out.rows() ids, already checked by the caller to be < table.rows().
void embedding_lookup(const Tensor& table, const int* d_tokens, Tensor& out);

// RMSNorm, per row: y = x / sqrt(mean(x^2) + eps) * weight. weight is 1 x cols.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& y);

// Rotary position embedding, in place. x is (T x n_heads*head_dim), and row t
// is the token at position pos + t. inv_freq is 1 x head_dim/2.
void rope_inplace(Tensor& x, int n_heads, int pos, const Tensor& inv_freq);

// Causal grouped-query attention over a KV cache.
//   q:                (T x heads*head_dim), row t at position pos + t
//   k_cache, v_cache: (max_seq x kv_heads*head_dim), rows 0 .. pos+T-1 filled
//   out:              (T x heads*head_dim)
// Each query attends to every cached position up to and including its own.
void attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
               int pos, int heads, int kv_heads, Tensor& out);

// gate = silu(gate) * up, elementwise: the SwiGLU activation of the MLP.
void silu_mul_inplace(Tensor& gate, const Tensor& up);

// x += y, elementwise: a residual connection.
void add_inplace(Tensor& x, const Tensor& y);
