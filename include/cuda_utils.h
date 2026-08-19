#pragma once
 
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
 
// This macro wraps a call, checks the status, and aborts loudly with the exact
// file and line if something went wrong.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n  -> %s\n",        \
                         cudaGetErrorName(err__), __FILE__, __LINE__,          \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
 
// Kernel launches (`kernel<<<...>>>(...)`) do NOT return an error code.
// Launch failures are reported by the *next* CUDA call, so you must ask
// explicitly. Call this immediately after every kernel launch.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
