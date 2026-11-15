// Decode speed for Qwen2.5-0.5B: the GEMM path against the GEMV path.
//
// Decode runs one token through the model per step, and every step must read
// all of the weights: about 2 GB in fp32. Memory bandwidth is therefore the
// ceiling, so besides tokens per second this reports the bandwidth each path
// reaches as a share of the GPU's peak.
//
// Usage:  ./bench_qwen [weights_prefix] [reference_prefix] [decode_steps] [reps]
//         defaults: models/qwen  models/ref  64  7

#include "cuda_utils.h"
#include "qwen.h"
#include "weights.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kMaxSeq = 1024;
constexpr int kWarmupMs = 2000;

// Prefills the prompt, then times `steps` decode steps on the GPU's clock and
// returns milliseconds per step. Each step feeds back the same token: the work
// per step does not depend on which token it is, and skipping the argmax
// avoids copying 600 KB of scores back to the CPU every step.
float decode_ms_per_token(Qwen& model, const std::vector<int>& prompt,
                          Activations& prefill, Activations& step, int steps) {
    model.forward(prompt, 0, prefill);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::vector<int> token = {prompt.back()};
    int pos = static_cast<int>(prompt.size());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < steps; ++i) {
        model.forward(token, pos++, step);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(steps);
}

struct Result {
    float median_ms;
    float min_ms;
    float max_ms;
};

// Same method as the kernel benchmarks: warm the GPU up for a fixed stretch of
// wall-clock time so every rep sees steady clocks, then report the median of
// several reps along with their spread.
Result measure(Qwen& model, bool use_gemv, const std::vector<int>& prompt,
               Activations& prefill, Activations& step, int steps, int reps) {
    model.set_decode_gemv(use_gemv);

    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(kWarmupMs);
    while (std::chrono::steady_clock::now() < until) {
        decode_ms_per_token(model, prompt, prefill, step, steps);
    }

    std::vector<float> samples;
    for (int r = 0; r < reps; ++r) {
        samples.push_back(decode_ms_per_token(model, prompt, prefill, step, steps));
    }
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(), samples.back()};
}

}  // namespace

int main(int argc, char** argv) {
    const std::string wprefix = argc > 1 ? argv[1] : "models/qwen";
    const std::string rprefix = argc > 2 ? argv[2] : "models/ref";
    const int steps = argc > 3 ? std::atoi(argv[3]) : 64;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 7;
    require(steps > 0 && reps > 0, "bench_qwen: decode_steps and reps must be positive");

    const WeightFile W(wprefix);
    const WeightFile R(rprefix);
    const std::vector<int>& prompt = R.ints("tokens");
    require(static_cast<int>(prompt.size()) + steps <= kMaxSeq,
            "bench_qwen: prompt plus decode steps exceed the KV cache");

    Qwen model(W, kMaxSeq);
    const QwenConfig& c = model.config();

    // Bytes of weights every decode step reads: the seven projections of each
    // layer plus the LM head. Norm weights and biases add under 0.02%.
    const double H = c.hidden, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    const double per_layer = H * QD + 2.0 * H * KVD + QD * H + 3.0 * H * I;
    const double bytes = (c.layers * per_layer + H * c.vocab) * sizeof(float);

    // Peak bandwidth from the hardware itself: memory clock x 2 transfers per
    // clock (double data rate) x bus width in bytes. 320 GB/s on a T4.
    int dev = 0;
    int mem_khz = 0;
    int bus_bits = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    const double peak_gbs = 2.0 * mem_khz * 1e3 * (bus_bits / 8.0) / 1e9;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s, peak memory bandwidth %.0f GB/s\n", prop.name, peak_gbs);
    std::printf("Each decode step reads %.2f GB of weights, so no kernel can beat %.0f tokens/s here\n\n",
                bytes / 1e9, peak_gbs * 1e9 / bytes);
    std::printf("%zu-token prompt, then %d decode steps; median of %d reps after a %.0f s warm-up\n\n",
                prompt.size(), steps, reps, kWarmupMs / 1000.0);

    Activations prefill(c, static_cast<int>(prompt.size()));
    Activations step(c, 1);

    const Result gemm = measure(model, false, prompt, prefill, step, steps, reps);
    const Result gemv = measure(model, true, prompt, prefill, step, steps, reps);

    std::printf("%-5s %10s %10s %16s %10s %8s\n",
                "path", "ms/token", "tokens/s", "weights, GB/s", "% of peak", "spread");
    auto row = [&](const char* name, const Result& r) {
        const double gbs = bytes / (r.median_ms * 1e-3) / 1e9;
        std::printf("%-5s %10.3f %10.1f %16.1f %9.1f%% %7.1f%%\n",
                    name, r.median_ms, 1000.0 / r.median_ms, gbs,
                    100.0 * gbs / peak_gbs,
                    100.0 * (r.max_ms - r.min_ms) / r.median_ms);
    };
    row("GEMM", gemm);
    row("GEMV", gemv);
    std::printf("\nGEMV speedup over GEMM: %.2fx\n", gemm.median_ms / gemv.median_ms);
    return 0;
}
