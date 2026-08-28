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