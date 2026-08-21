#pragma once

class Tensor;

// C = A * B, where A is (M x K), B is (K x N), C is (M x N), all row-major.
//
// Every variant computes the same result; they differ only in how they use the
// GPU memory hierarchy. Comparing them is the point of this project.

// All kernels share this signature, so the benchmark harness can treat them
// interchangeably.
using GemmFn = void (*)(const Tensor& A, const Tensor& B, Tensor& C);

// v1: one thread per output element, reading A and B straight from global
// memory. Correct, and deliberately slow -- each element of A is re-read N times
// and each element of B M times.
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

// v2: same work, staged through shared memory in TILE x TILE blocks. Cuts global
// memory traffic by a factor of TILE.
void gemm_tiled(const Tensor& A, const Tensor& B, Tensor& C);