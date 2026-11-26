// Decode speed for Qwen2.5-0.5B: the GEMM path against the GEMV path, each with
// the projection weights stored in fp32 and in fp16.
//
// Decode runs one token through the model per step, and every step must read
// all of the weights: about 2 GB in fp32, half that in fp16. Memory bandwidth
// is therefore the ceiling, so besides tokens per second this reports the
// bandwidth each configuration reaches as a share of the GPU's peak.
//
// Usage:  ./bench_qwen [weights_prefix] [reference_prefix] [decode_steps] [reps] [path] [precision]
//         defaults: models/qwen  models/ref  64  7  both  both
//         path: both, gemm or gemv        precision: both, fp32 or fp16
//         One path at one precision keeps a profile focused.

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

// One measured configuration: a decode path and a weight precision.
struct Row {
    bool gemv;
    WeightPrecision precision;
    Result result;
};

}  // namespace

int main(int argc, char** argv) {
    const std::string wprefix = argc > 1 ? argv[1] : "models/qwen";
    const std::string rprefix = argc > 2 ? argv[2] : "models/ref";
    const int steps = argc > 3 ? std::atoi(argv[3]) : 64;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 7;
    const std::string path = argc > 5 ? argv[5] : "both";
    const std::string precision = argc > 6 ? argv[6] : "both";
    require(steps > 0 && reps > 0, "bench_qwen: decode_steps and reps must be positive");
    require(path == "both" || path == "gemm" || path == "gemv",
            "bench_qwen: path must be both, gemm or gemv");
    require(precision == "both" || precision == "fp32" || precision == "fp16",
            "bench_qwen: precision must be both, fp32 or fp16");

    std::vector<WeightPrecision> precisions;
    if (precision != "fp16") precisions.push_back(WeightPrecision::fp32);
    if (precision != "fp32") precisions.push_back(WeightPrecision::fp16);
    std::vector<bool> use_gemv_options;
    if (path != "gemv") use_gemv_options.push_back(false);
    if (path != "gemm") use_gemv_options.push_back(true);

    const WeightFile W(wprefix);
    const WeightFile R(rprefix);
    const std::vector<int>& prompt = R.ints("tokens");
    require(static_cast<int>(prompt.size()) + steps <= kMaxSeq,
            "bench_qwen: prompt plus decode steps exceed the KV cache");
    const QwenConfig c = QwenConfig::from(W);

    // Weights every decode step reads: the seven projections of each layer plus
    // the LM head. Norm weights and biases add under 0.02%.
    const double H = c.hidden, QD = c.q_dim(), KVD = c.kv_dim(), I = c.intermediate;
    const double per_layer = H * QD + 2.0 * H * KVD + QD * H + 3.0 * H * I;
    const double weights_read = c.layers * per_layer + H * c.vocab;
    auto bytes_per_step = [&](WeightPrecision p) {
        return weights_read * (p == WeightPrecision::fp16 ? 2.0 : 4.0);
    };

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
    const double fp32_bytes = bytes_per_step(WeightPrecision::fp32);
    const double fp16_bytes = bytes_per_step(WeightPrecision::fp16);
    std::printf("GPU: %s, peak memory bandwidth %.0f GB/s\n", prop.name, peak_gbs);
    std::printf("Each decode step reads %.2f GB of weights in fp32, %.2f GB in fp16, so no kernel\n"
                "can beat %.0f tokens/s with fp32 weights or %.0f tokens/s with fp16 weights here\n\n",
                fp32_bytes / 1e9, fp16_bytes / 1e9,
                peak_gbs * 1e9 / fp32_bytes, peak_gbs * 1e9 / fp16_bytes);
    std::printf("%zu-token prompt, then %d decode steps; median of %d reps after a %.0f s warm-up\n\n",
                prompt.size(), steps, reps, kWarmupMs / 1000.0);

    std::vector<Row> rows;
    for (WeightPrecision p : precisions) {
        // One model per precision, freed before the next one loads.
        std::printf("Loading weights in %s ...\n", precision_name(p));
        Qwen model(W, kMaxSeq, p);
        Activations prefill(c, static_cast<int>(prompt.size()));
        Activations step(c, 1);
        for (bool use_gemv : use_gemv_options) {
            rows.push_back({use_gemv, p,
                            measure(model, use_gemv, prompt, prefill, step, steps, reps)});
        }
    }

    // GEMM rows first, then GEMV, each fp32 before fp16.
    std::stable_sort(rows.begin(), rows.end(),
                     [](const Row& a, const Row& b) { return a.gemv < b.gemv; });

    std::printf("\n%-5s %-8s %10s %10s %12s %10s %8s\n",
                "path", "weights", "ms/token", "tokens/s", "read, GB/s", "% of peak", "spread");
    for (const Row& r : rows) {
        const double gbs = bytes_per_step(r.precision) / (r.result.median_ms * 1e-3) / 1e9;
        std::printf("%-5s %-8s %10.3f %10.1f %12.1f %9.1f%% %7.1f%%\n",
                    r.gemv ? "GEMV" : "GEMM", precision_name(r.precision),
                    r.result.median_ms, 1000.0 / r.result.median_ms, gbs,
                    100.0 * gbs / peak_gbs,
                    100.0 * (r.result.max_ms - r.result.min_ms) / r.result.median_ms);
    }

    // Speedups, wherever both sides of a comparison were measured.
    auto find = [&](bool gemv, WeightPrecision p) -> const Result* {
        for (const Row& r : rows) {
            if (r.gemv == gemv && r.precision == p) return &r.result;
        }
        return nullptr;
    };
    std::printf("\n");
    for (WeightPrecision p : precisions) {
        const Result* gemm = find(false, p);
        const Result* gemv = find(true, p);
        if (gemm != nullptr && gemv != nullptr) {
            std::printf("GEMV speedup over GEMM, %s weights: %.2fx\n",
                        precision_name(p), gemm->median_ms / gemv->median_ms);
        }
    }
    for (bool use_gemv : use_gemv_options) {
        const Result* fp32 = find(use_gemv, WeightPrecision::fp32);
        const Result* fp16 = find(use_gemv, WeightPrecision::fp16);
        if (fp32 != nullptr && fp16 != nullptr) {
            std::printf("fp16 weights over fp32, %s path: %.2fx\n",
                        use_gemv ? "GEMV" : "GEMM", fp32->median_ms / fp16->median_ms);
        }
    }
    return 0;
}
