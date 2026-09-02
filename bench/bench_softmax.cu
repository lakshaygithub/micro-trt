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
#include <cfloat>

namespace {

// Ground truth. Accumulates in double so the reference is comfortably more
// accurate than the thing it is checking.
void softmax_cpu_reference(const std::vector<float>& X, std::vector<float>& Y,
                           int M, int N) {
    for (int r = 0; r < M; ++r) {
        const float* x = X.data() + static_cast<std::size_t>(r) * N;
        float* y = Y.data() + static_cast<std::size_t>(r) * N;

        float m = -FLT_MAX;
        for (int j = 0; j < N; ++j) m = std::fmax(m, x[j]);

        double s = 0.0;
        for (int j = 0; j < N; ++j) s += std::exp(static_cast<double>(x[j]) - m);

        for (int j = 0; j < N; ++j) {
            y[j] = static_cast<float>(std::exp(static_cast<double>(x[j]) - m) / s);
        }
    }
}

bool allclose(const std::vector<float>& got, const std::vector<float>& want,
              float rel_tol) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::printf("\n  non-finite value at %zu: %f\n", i, got[i]);
            return false;
        }
        const float diff = std::fabs(got[i] - want[i]);
        // Softmax outputs are all <= 1, so flooring the scale at 1.0 would make
        // the test vacuous. Use a small absolute floor instead.
        const float scale = std::fmax(1e-6f, std::fabs(want[i]));
        if (diff / scale > rel_tol) {
            std::printf("\n  mismatch at %zu: got %g, want %g\n", i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

// Independent structural check: every row of a softmax must sum to 1. This
// catches whole classes of reduction bug that an elementwise comparison against
// a reference computed the same way might not.
double worst_row_sum_error(const std::vector<float>& Y, int M, int N) {
    double worst = 0.0;
    for (int r = 0; r < M; ++r) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += Y[static_cast<std::size_t>(r) * N + j];
        worst = std::fmax(worst, std::fabs(s - 1.0));
    }
    return worst;
}

float time_kernel_ms(UnaryFn fn, const Tensor& X, Tensor& Y, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    fn(X, Y);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) fn(X, Y);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
}

void warm_up_device(UnaryFn fn, const Tensor& X, Tensor& Y, double seconds = 1.5) {
    std::printf("Warming up GPU to steady-state clocks (%.1fs)... ", seconds);
    std::fflush(stdout);
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 20; ++i) fn(X, Y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::printf("done\n");
}

struct Result {
    const char* name;
    float median_ms;
    double gbs;          // achieved effective bandwidth
    double spread_pct;
    double row_sum_err;
    bool correct;
};

Result evaluate(const char* name, UnaryFn fn,
                const Tensor& X, Tensor& Y,
                const std::vector<float>& cpu_result,
                int M, int N, int iters, int reps) {
    fn(X, Y);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = Y.download();
    const bool ok = allclose(got, cpu_result, 2e-3f);
    const double rs = worst_row_sum_error(got, M, N);

    std::vector<float> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) samples.push_back(time_kernel_ms(fn, X, Y, iters));
    std::sort(samples.begin(), samples.end());

    const float median_ms = samples[samples.size() / 2];

    // Minimum traffic any correct implementation must move: read every element
    // once, write every element once. Comparing achieved bandwidth against the
    // card's peak is the right yardstick for a memory-bound kernel -- GFLOP/s
    // would be meaningless here.
    const double bytes = 2.0 * M * N * sizeof(float);
    const double gbs = (bytes / (median_ms / 1000.0)) / 1e9;
    const double spread = 100.0 * (samples.back() - samples.front()) / median_ms;

    return Result{name, median_ms, gbs, spread, rs, ok};
}

double peak_bandwidth_gbs() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    return 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
}

}  // namespace

int main(int argc, char** argv) {
    // usage: ./bench_softmax [rows] [cols] [reps]
    int M = 4096, N = 1024, reps = 15;
    if (argc > 1) M = std::stoi(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) reps = std::stoi(argv[3]);

    const double peak = peak_bandwidth_gbs();
    std::printf("Peak memory bandwidth: %.0f GB/s\n\n", peak);
    std::printf("Row-wise softmax over %d x %d (%.1f MB in, %.1f MB out)\n",
                M, N,
                static_cast<double>(M) * N * sizeof(float) / (1024.0 * 1024.0),
                static_cast<double>(M) * N * sizeof(float) / (1024.0 * 1024.0));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> hX(static_cast<std::size_t>(M) * N);
    for (float& v : hX) v = dist(rng);

    Tensor dX(M, N);
    Tensor dY(M, N);
    dX.upload(hX);

    std::printf("Computing CPU reference... ");
    std::fflush(stdout);
    std::vector<float> cpu_result(static_cast<std::size_t>(M) * N);
    softmax_cpu_reference(hX, cpu_result, M, N);
    std::printf("done\n");

    warm_up_device(softmax_block, dX, dY);

    const int iters = 50;
    std::printf("Timing: median of %d reps x %d iterations each\n\n", reps, iters);

    const Result results[] = {
        evaluate("softmax_naive",  softmax_naive,  dX, dY, cpu_result, M, N, iters, reps),
        evaluate("softmax_block",  softmax_block,  dX, dY, cpu_result, M, N, iters, reps),
        evaluate("softmax_online", softmax_online, dX, dY, cpu_result, M, N, iters, reps),
    };

    std::printf("%-16s %11s %11s %10s %8s %8s\n",
                "kernel", "median ms", "GB/s", "% of peak", "spread", "correct");
    std::printf("---------------------------------------------------------------------\n");
    for (const Result& r : results) {
        std::printf("%-16s %11.4f %11.2f %9.1f%% %7.1f%% %8s\n",
                    r.name, r.median_ms, r.gbs, 100.0 * r.gbs / peak,
                    r.spread_pct, r.correct ? "yes" : "NO");
    }

    std::printf("\nrow-sum error (every row must sum to 1):\n");
    for (const Result& r : results) {
        std::printf("  %-16s %.3g\n", r.name, r.row_sum_err);
    }

    std::printf("\nvs naive: ");
    for (std::size_t i = 1; i < sizeof(results) / sizeof(results[0]); ++i) {
        std::printf("%s %.2fx   ", results[i].name,
                    results[0].median_ms / results[i].median_ms);
    }
    std::printf("\n");

    for (const Result& r : results) {
        if (!r.correct) {
            std::printf("\n%s produced incorrect results.\n", r.name);
            return 1;
        }
    }
    return 0;
}