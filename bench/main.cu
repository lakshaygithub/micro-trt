#include "tensor.h"
#include "kernels.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>

namespace {

// Straightforward triple-nested-loop GEMM on the CPU. This is the ground truth:
// if a GPU result does not match this, the kernel is wrong, no matter how fast
// it is. Never benchmark a kernel you have not first proven correct.
void gemm_cpu_reference(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

// Float comparison must be tolerant. The GPU sums in a different order than the
// CPU, and floating-point addition is not associative, so bit-exact equality is
// the wrong test.
bool allclose(const std::vector<float>& got,
              const std::vector<float>& want,
              float rel_tol) {
    if (got.size() != want.size()) {
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - want[i]);
        const float scale = std::fmax(1.0f, std::fabs(want[i]));
        if (diff / scale > rel_tol) {
            std::printf("\n  mismatch at %zu: got %f, want %f\n", i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

std::vector<float> random_matrix(int rows, int cols, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(static_cast<std::size_t>(rows) * cols);
    for (float& v : out) {
        v = dist(rng);
    }
    return out;
}

// Times any kernel matching GemmFn, using CUDA events rather than a CPU clock.
//
// Why events: kernel launches are asynchronous. A CPU timer around the launch
// measures how long it took to *queue* the work, which is meaningless. Events
// are recorded in the GPU's own stream and measure device execution time.
float time_kernel_ms(GemmFn fn, const Tensor& A, const Tensor& B, Tensor& C,
                     int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up: the first launch pays one-time costs (context setup, JIT of PTX
    // for this exact GPU) that would otherwise pollute the measurement.
    fn(A, B, C);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn(A, B, C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iters);
}

struct Result {
    const char* name;
    float ms;
    double gflops;
    bool correct;
};

Result evaluate(const char* name, GemmFn fn,
                const Tensor& dA, const Tensor& dB, Tensor& dC,
                const std::vector<float>& cpu_result,
                int M, int N, int K, int iters) {
    // Correctness first, always.
    fn(dA, dB, dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    const bool ok = allclose(dC.download(), cpu_result, 1e-3f);

    const float ms = time_kernel_ms(fn, dA, dB, dC, iters);

    const double flops = 2.0 * M * N * K;  // one multiply + one add per MAC
    const double gflops = (flops / (ms / 1000.0)) / 1e9;

    return Result{name, ms, gflops, ok};
}

void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("GPU: %s (compute capability %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("Shared memory per block: %zu KB | Peak memory bandwidth: %.0f GB/s\n\n",
                prop.sharedMemPerBlock / 1024,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
    int size = 1024;
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }
    const int M = size;
    const int N = size;
    const int K = size;

    print_device_info();
    std::printf("GEMM: (%d x %d) * (%d x %d)\n\n", M, K, K, N);

    std::mt19937 rng(42);  // fixed seed -> reproducible runs
    const std::vector<float> hA = random_matrix(M, K, rng);
    const std::vector<float> hB = random_matrix(K, N, rng);

    Tensor dA(M, K);
    Tensor dB(K, N);
    Tensor dC(M, N);
    dA.upload(hA);
    dB.upload(hB);

    std::printf("Computing CPU reference... ");
    std::fflush(stdout);
    std::vector<float> cpu_result(static_cast<std::size_t>(M) * N);
    gemm_cpu_reference(hA, hB, cpu_result, M, N, K);
    std::printf("done\n\n");

    const int iters = 20;
    const Result results[] = {
        evaluate("gemm_naive", gemm_naive, dA, dB, dC, cpu_result, M, N, K, iters),
        evaluate("gemm_tiled", gemm_tiled, dA, dB, dC, cpu_result, M, N, K, iters),
    };

    std::printf("%-14s %10s %14s %10s %8s\n",
                "kernel", "time (ms)", "GFLOP/s", "vs naive", "correct");
    std::printf("--------------------------------------------------------------\n");

    const double baseline_ms = results[0].ms;
    for (const Result& r : results) {
        std::printf("%-14s %10.3f %14.2f %9.2fx %8s\n",
                    r.name, r.ms, r.gflops,
                    baseline_ms / r.ms,
                    r.correct ? "yes" : "NO");
    }

    // Every kernel must be correct; a fast wrong answer is a failure.
    for (const Result& r : results) {
        if (!r.correct) {
            std::printf("\n%s produced incorrect results.\n", r.name);
            return 1;
        }
    }

    return 0;
}