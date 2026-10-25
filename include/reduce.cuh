#pragma once

// ---------------------------------------------------------------------------
// Warp-level reductions, shared by every kernel whose threads must agree on a
// value: softmax (row max and sum), RMSNorm (sum of squares), attention (the
// q.k dot product).
//
// A warp is 32 threads executing in lockstep, and they can exchange register
// values directly with each other -- no shared memory, no barrier. That is what
// __shfl_xor_sync does: every thread swaps its value with the thread whose lane
// index differs in one bit, given by `offset`.
//
// Five rounds of this (offsets 16, 8, 4, 2, 1) is a butterfly pattern that
// leaves the fully reduced result in EVERY lane, which saves a separate
// broadcast step. The 0xffffffff mask says all 32 lanes participate, so every
// lane of the warp must reach the call.
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
