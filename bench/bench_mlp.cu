#include "tensor.h"
#include "kernels.h"
#include "mlp.h"
#include "cuda_utils.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>

namespace {

bool allclose(const std::vector<float>& got, const std::vector<float>& want,
              float tol) {
    if (got.size() != want.size()) return false;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::printf("\n  non-finite output at %zu: %f\n", i, got[i]);
            return false;
        }
        const float diff = std::fabs(got[i] - want[i]);
        const float scale = std::fmax(1e-6f, std::fabs(want[i]));
        if (diff / scale > tol) {
            std::printf("\n  mismatch at %zu: got %g, want %g\n", i, got[i], want[i]);
            return false;
        }
    }
    return true;
}

double worst_row_sum_error(const std::vector<float>& Y, int rows, int cols) {
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += Y[static_cast<std::size_t>(r) * cols + c];
        worst = std::fmax(worst, std::fabs(s - 1.0));
    }
    return worst;
}

template <typename Op>
float time_op_ms(Op&& op, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    op();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) op();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / static_cast<float>(iters);
}

template <typename Op>
void warm_up_device(Op&& op, double seconds = 1.5) {
    std::printf("Warming up GPU to steady-state clocks (%.1fs)... ", seconds);
    std::fflush(stdout);
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 20; ++i) op();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::printf("done\n");
}

void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);
}

}  // namespace

int main(int argc, char** argv) {
    // usage: ./bench_mlp [batch] [reps]
    int batch = 1024;
    int reps = 15;
    if (argc > 1) batch = std::stoi(argv[1]);
    if (argc > 2) reps = std::stoi(argv[2]);

    // A classifier-shaped network: three hidden layers, then 1000 classes.
    const std::vector<int> dims = {1024, 1024, 1024, 1024, 1000};

    print_device_info();

    std::printf("Network: ");
    for (std::size_t i = 0; i < dims.size(); ++i) {
        std::printf("%d%s", dims[i], i + 1 < dims.size() ? " -> " : "");
    }
    std::printf("   (batch %d)\n", batch);

    // Total multiply-adds across all layers, x2 for the flop convention.
    double flops = 0.0;
    for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
        flops += 2.0 * batch * dims[i] * dims[i + 1];
    }
    std::printf("Arithmetic: %.2f GFLOP per forward pass\n\n", flops / 1e9);

    MLP net(dims, batch, 1234u);

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> hX(static_cast<std::size_t>(batch) * dims.front());
    for (float& v : hX) v = dist(rng);

    Tensor dX(batch, dims.front());
    Tensor dY(batch, dims.back());
    dX.upload(hX);

    auto run = [&]() { net.forward(dX, dY); };

    // ---- correctness ----
    run();
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> gpu = dY.download();

    std::printf("Computing CPU reference (this is slow by design)... ");
    std::fflush(stdout);

    // Re-derive the weights exactly as the MLP constructor did, so the
    // reference sees the same network.
    std::vector<std::vector<float>> Ws, bs;
    {
        std::mt19937 wrng(1234u);
        for (std::size_t i = 0; i + 1 < dims.size(); ++i) {
            const int in = dims[i], out = dims[i + 1];
            const float scale = std::sqrt(2.0f / static_cast<float>(in));
            std::normal_distribution<float> wd(0.0f, scale);
            std::vector<float> W(static_cast<std::size_t>(in) * out);
            for (float& v : W) v = wd(wrng);
            std::vector<float> b(static_cast<std::size_t>(out));
            for (float& v : b) v = wd(wrng) * 0.1f;
            Ws.push_back(std::move(W));
            bs.push_back(std::move(b));
        }
    }
    const std::vector<float> cpu = mlp_cpu_reference(dims, batch, hX, Ws, bs);
    std::printf("done\n");

    const bool ok = allclose(gpu, cpu, 5e-3f);
    const double rs = worst_row_sum_error(gpu, batch, dims.back());
    std::printf("Correct vs CPU reference: %s\n", ok ? "yes" : "NO");
    std::printf("Worst row-sum error (should be ~0): %.3g\n\n", rs);

    // ---- timing ----
    warm_up_device(run);

    const int iters = 50;
    std::printf("Timing: median of %d reps x %d iterations each\n\n", reps, iters);

    std::vector<float> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) samples.push_back(time_op_ms(run, iters));
    std::sort(samples.begin(), samples.end());

    const float median = samples[samples.size() / 2];
    const float lo = samples.front();
    const float hi = samples.back();
    const double gflops = (flops / (median / 1000.0)) / 1e9;
    const double per_sample_us = median * 1000.0 / batch;

    std::printf("micro-trt forward pass\n");
    std::printf("  median          : %.4f ms  (min %.4f, max %.4f, spread %.1f%%)\n",
                median, lo, hi, 100.0 * (hi - lo) / median);
    std::printf("  throughput      : %.1f GFLOP/s\n", gflops);
    std::printf("  per sample      : %.2f us\n", per_sample_us);
    std::printf("  samples/second  : %.0f\n\n", batch / (median / 1000.0));

    // ---- dump for independent verification ----
    const std::string path = "mlp_dump.bin";
    net.dump(path, dX, dY);
    std::printf("Wrote %s\n", path.c_str());
    std::printf("Run 'python3 tools/compare_pytorch.py' to verify the output\n"
                "against PyTorch and time the same network on this GPU.\n");

    if (!ok) {
        std::printf("\nForward pass produced incorrect results.\n");
        return 1;
    }
    return 0;
}