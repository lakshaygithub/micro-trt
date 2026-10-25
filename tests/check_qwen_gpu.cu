// The GPU forward pass, checked against Hugging Face the way
// tests/check_qwen.cpp checks the CPU reference: every recorded activation
// during prefill, then the whole greedy continuation, token for token, through
// the KV cache.
//
// Usage:  ./check_qwen_gpu [weights_prefix] [reference_prefix]
//         defaults: models/qwen  models/ref

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
    const WeightFile W(wprefix);
    const WeightFile R(rprefix);
    ReferenceCheck check(R);

    std::printf("Uploading weights ...\n");
    Qwen model(W, /*max_seq=*/1024);
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

    // ---- decode: the full greedy continuation, through the KV cache ----
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
    std::printf("%s\n", check.failures() ? "FAILED" : "GPU forward pass matches Hugging Face");
    return check.failures() ? 1 : 0;
}
