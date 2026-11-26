// The GPU forward pass, checked against Hugging Face the way
// tests/check_qwen.cpp checks the CPU reference: every recorded activation
// during prefill, then the whole greedy continuation, token for token, through
// the KV cache. One decode step is also computed by both projection paths,
// GEMM and GEMV, which must agree to within rounding.
//
// The projection weights can be held in fp32 or fp16; `make check-qwen-gpu`
// runs both. Hugging Face's reference is fp32 either way, so the fp16 run is
// held to the same tolerance as the fp32 one.
//
// Usage:  ./check_qwen_gpu [weights_prefix] [reference_prefix] [precision]
//         defaults: models/qwen  models/ref  fp16      precision: fp16 or fp32

#include "cuda_utils.h"
#include "qwen.h"
#include "reference_check.h"
#include "tensor.h"
#include "weights.h"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const std::string wprefix = argc > 1 ? argv[1] : "models/qwen";
    const std::string rprefix = argc > 2 ? argv[2] : "models/ref";
    const std::string precision_arg = argc > 3 ? argv[3] : "fp16";
    require(precision_arg == "fp16" || precision_arg == "fp32",
            "check_qwen_gpu: precision must be fp16 or fp32");
    const WeightPrecision precision =
        precision_arg == "fp16" ? WeightPrecision::fp16 : WeightPrecision::fp32;

    const WeightFile W(wprefix);
    const WeightFile R(rprefix);
    ReferenceCheck check(R);

    std::printf("Uploading weights, projections in %s ...\n", precision_name(precision));
    Qwen model(W, /*max_seq=*/1024, precision);
    const QwenConfig& c = model.config();
    std::printf("Qwen2: %d layers, hidden %d, %d/%d heads, head_dim %d, ffn %d, vocab %d\n\n",
                c.layers, c.hidden, c.heads, c.kv_heads, c.head_dim, c.intermediate, c.vocab);

    // ---- prefill: every activation Hugging Face recorded ----
    const std::vector<int>& prompt = R.ints("tokens");
    std::printf("Prefill, %zu tokens:\n", prompt.size());
    Activations act(c, static_cast<int>(prompt.size()));
    model.forward(prompt, 0, act, [&](const std::string& name, const Tensor& t) {
        if (R.has(name)) check.compare(name, t.download());
    });

    // ---- one decode step, GEMV against the GEMM path ----
    // The same step computed both ways should differ only by rounding. Both
    // runs write KV-cache row T, each from its own keys and values.
    const int T = static_cast<int>(prompt.size());
    const std::vector<int> first = {R.ints("generated").front()};
    Activations one(c, 1);
    model.set_decode_gemv(false);
    model.forward(first, T, one);
    const std::vector<float> by_gemm = one.logits.download();
    model.set_decode_gemv(true);
    model.forward(first, T, one);
    std::printf("\nOne decode step, GEMV against the GEMM path:\n");
    check.compare("decode_logits", one.logits.download(), by_gemm);

    // ---- decode: the full greedy continuation, through the KV cache ----
    // generate() uses the GEMV for every single-token step.
    const std::vector<int>& want = R.ints("generated");
    const std::vector<int> got = model.generate(prompt, static_cast<int>(want.size()));

    std::size_t matched = 0;
    while (matched < want.size() && got[matched] == want[matched]) ++matched;
    std::printf("\nGreedy decode: %zu / %zu tokens match Hugging Face", matched, want.size());
    if (matched < want.size()) {
        std::printf(" -- first difference at token %zu: %d vs %d",
                    matched, got[matched], want[matched]);
        check.record_failure();
    }

    std::printf("\nWorst relative error: %.2e (%s)\n", check.worst(), check.worst_name().c_str());
    if (check.failures()) {
        std::printf("FAILED, %s weights\n", precision_name(precision));
    } else {
        std::printf("GPU forward pass matches Hugging Face, %s weights\n",
                    precision_name(precision));
    }
    return check.failures() ? 1 : 0;
}
