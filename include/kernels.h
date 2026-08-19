#pragma once
 
class Tensor;

// v1: one thread per output element, every thread reads its full row of A and
// column of B straight from global memory. Correct, and deliberately slow.
void gemm_naive(const Tensor& A, const Tensor& B, Tensor& C);

