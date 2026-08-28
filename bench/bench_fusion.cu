#include "tensor.h"
#include "kernels.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>

namespace {

// Ground truth: Y = ReLU(A*B + bias), bias broadcast across rows.
void fused_cpu_reference(const std::vector<float>& A,
                         const std::vector<float>& B,
                         const std::vector<float>& bias,
                         std::vector<float>& C,
                         int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[m * K + k] * B[k * N + n];
            }
            const float v = acc + bias[n];
            C[m * N + n] = v > 0.0f ? v : 0.0f;
        }
    }
}

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

std::vector<float> random_vector(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (float& v : out) {
        v = dist(rng);
    }
    return out;
}

// Times an arbitrary callable with CUDA events. Templated so it can time either
// a single kernel or a multi-kernel pipeline; the pipeline case is exactly what
// we need, since the unfused path is two launches.
template <typename Op>
float time_op_ms(Op&& op, int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    op();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        op();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iters);
}

template <typename Op>
void warm_up_device(Op&& op, double seconds = 1.5) {
    std::printf("Warming up GPU to steady-state clocks (%.1fs)... ", seconds);
    std::fflush(stdout);

    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 20; ++i) {
            op();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::printf("done\n");
}

struct Result {
    const char* name;
    float median_ms;
    float min_ms;
    float max_ms;
    double spread_pct;
    bool correct;
};

template <typename Op>
Result measure(const char* name, Op&& op,
               Tensor& dC, const std::vector<float>& cpu_result,
               int iters, int reps) {
    op();
    CUDA_CHECK(cudaDeviceSynchronize());
    const bool ok = allclose(dC.download(), cpu_result, 1e-3f);

    std::vector<float> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        samples.push_back(time_op_ms(op, iters));
    }
    std::sort(samples.begin(), samples.end());

    const float median_ms = samples[samples.size() / 2];
    return Result{name, median_ms, samples.front(), samples.back(),
                  100.0 * (samples.back() - samples.front()) / median_ms, ok};
}

void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (%d SMs, %.0f GB/s peak bandwidth)\n\n",
                prop.name, prop.multiProcessorCount,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

}  // namespace

int main(int argc, char** argv) {
    int size = 1024;
    int reps = 15;
    if (argc > 1) size = std::stoi(argv[1]);
    if (argc > 2) reps = std::stoi(argv[2]);

    const int M = size;
    const int N = size;
    const int K = size;

    print_device_info();
    std::printf("Y = ReLU(A*B + bias),  A(%d x %d) * B(%d x %d)\n", M, K, K, N);

    std::mt19937 rng(42);
    const std::vector<float> hA = random_vector(static_cast<std::size_t>(M) * K, rng);
    const std::vector<float> hB = random_vector(static_cast<std::size_t>(K) * N, rng);
    const std::vector<float> hBias = random_vector(static_cast<std::size_t>(N), rng);

    Tensor dA(M, K);
    Tensor dB(K, N);
    Tensor dBias(1, N);
    Tensor dC(M, N);
    dA.upload(hA);
    dB.upload(hB);
    dBias.upload(hBias);

    std::printf("Computing CPU reference... ");
    std::fflush(stdout);
    std::vector<float> cpu_result(static_cast<std::size_t>(M) * N);
    fused_cpu_reference(hA, hB, hBias, cpu_result, M, N, K);
    std::printf("done\n");

    // The two pipelines being compared. Both must produce identical output.
    auto unfused = [&]() {
        gemm_regtiled(dA, dB, dC);      // kernel 1: the matrix multiply
        bias_relu_inplace(dC, dBias);   // kernel 2: a full extra pass over C
    };
    auto fused = [&]() {
        gemm_bias_relu_fused(dA, dB, dBias, dC);  // one kernel, one pass
    };

    warm_up_device(fused);

    const int iters = 50;
    std::printf("Timing: median of %d reps x %d iterations each\n\n", reps, iters);

    const Result results[] = {
        measure("unfused (2 kernels)", unfused, dC, cpu_result, iters, reps),
        measure("fused   (1 kernel)",  fused,   dC, cpu_result, iters, reps),
    };

    std::printf("%-22s %11s %10s %8s %8s\n",
                "pipeline", "median ms", "vs unfused", "spread", "correct");
    std::printf("---------------------------------------------------------------\n");

    const double baseline = results[0].median_ms;
    for (const Result& r : results) {
        std::printf("%-22s %11.4f %9.2fx %7.1f%% %8s\n",
                    r.name, r.median_ms, baseline / r.median_ms,
                    r.spread_pct, r.correct ? "yes" : "NO");
    }

    // The bias+ReLU pass reads C and writes it back: 2 * M * N * 4 bytes of
    // traffic that fusion removes entirely.
    const double bytes_saved = 2.0 * M * N * sizeof(float);
    const double saved_ms = results[0].median_ms - results[1].median_ms;
    std::printf("\nGlobal traffic avoided: %.1f MB per call\n",
                bytes_saved / (1024.0 * 1024.0));
    if (saved_ms > 0.0) {
        std::printf("Time saved: %.4f ms  (implied bandwidth %.0f GB/s)\n",
                    saved_ms, (bytes_saved / (saved_ms / 1000.0)) / 1e9);
    }

    for (const Result& r : results) {
        if (!r.correct) {
            std::printf("\n%s produced incorrect results.\n", r.name);
            return 1;
        }
    }
    return 0;
}