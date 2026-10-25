// A plain CPU implementation of the Qwen2 forward pass, checked against the
// activations Hugging Face recorded for the same prompt.
//
// This is the specification the GPU kernels will be held to. Every operation in
// the transformer block appears here once, written as simply as possible, and is
// compared against Hugging Face at the point it happens -- so a misread detail of
// the architecture (GQA head sharing, the RoPE pairing, which projections carry
// a bias) fails at the exact step it affects, before any CUDA exists to blame.
//
// Usage:  ./check_qwen [weights_prefix] [reference_prefix]
//         defaults: models/qwen  models/ref

#include "qwen_config.h"
#include "reference_check.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using Mat = std::vector<float>;  // row-major; shape tracked by the caller

// y = x * W (+ b).  x is (T x K), W is (K x N) -- already transposed at export.
Mat linear(const Mat& x, int T, int K, const Mat& W, int N, const Mat* b = nullptr) {
    Mat y(static_cast<std::size_t>(T) * N, 0.0f);
    for (int t = 0; t < T; ++t) {
        float* yr = &y[static_cast<std::size_t>(t) * N];
        for (int k = 0; k < K; ++k) {
            // i-k-j order: the inner loop walks a row of W and a row of y
            // contiguously instead of striding down a column.
            const float xv = x[static_cast<std::size_t>(t) * K + k];
            const float* wr = &W[static_cast<std::size_t>(k) * N];
            for (int n = 0; n < N; ++n) yr[n] += xv * wr[n];
        }
        if (b) for (int n = 0; n < N; ++n) yr[n] += (*b)[n];
    }
    return y;
}

// RMSNorm: x / sqrt(mean(x^2) + eps) * weight. No mean subtraction and no bias,
// which is the whole difference from LayerNorm.
Mat rmsnorm(const Mat& x, int T, int H, const Mat& w, double eps) {
    Mat y(x.size());
    for (int t = 0; t < T; ++t) {
        const float* xr = &x[static_cast<std::size_t>(t) * H];
        double ss = 0.0;
        for (int i = 0; i < H; ++i) ss += static_cast<double>(xr[i]) * xr[i];
        const float inv = static_cast<float>(1.0 / std::sqrt(ss / H + eps));
        for (int i = 0; i < H; ++i) y[static_cast<std::size_t>(t) * H + i] = xr[i] * inv * w[i];
    }
    return y;
}

// Rotary position embedding, in place. x is (T x n_heads*head_dim).
//
// Qwen and Llama pair dimension i with dimension i + head_dim/2 ("rotate_half"),
// NOT the adjacent pair (2i, 2i+1) described in the original RoPE paper. Both
// are valid rotations, the weights were trained with this one, and using the
// other produces plausible-looking numbers that are simply wrong.
void rope(Mat& x, int T, int n_heads, int head_dim, double theta) {
    const int half = head_dim / 2;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* v = &x[(static_cast<std::size_t>(t) * n_heads + h) * head_dim];
            for (int i = 0; i < half; ++i) {
                const double inv_freq = std::pow(theta, -2.0 * i / head_dim);
                const double angle = t * inv_freq;  // position == token index
                const float c = static_cast<float>(std::cos(angle));
                const float s = static_cast<float>(std::sin(angle));
                const float a = v[i];
                const float b = v[i + half];
                v[i] = a * c - b * s;
                v[i + half] = b * c + a * s;
            }
        }
    }
}

// Causal attention with grouped-query heads. q is (T x heads*hd); k and v are
// (T x kv_heads*hd). Each group of heads/kv_heads query heads shares one K/V
// head -- for Qwen2.5-0.5B that is 7 query heads per KV head, which is why its
// KV cache is 7x smaller than the query count suggests.
Mat attention(const Mat& q, const Mat& k, const Mat& v, int T,
              int heads, int kv_heads, int hd) {
    const int group = heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    Mat ctx(static_cast<std::size_t>(T) * heads * hd, 0.0f);
    std::vector<float> p(T);

    for (int h = 0; h < heads; ++h) {
        const int kvh = h / group;
        for (int t = 0; t < T; ++t) {
            const float* qv = &q[(static_cast<std::size_t>(t) * heads + h) * hd];

            float m = -INFINITY;
            for (int s = 0; s <= t; ++s) {  // causal: only positions up to t
                const float* kv = &k[(static_cast<std::size_t>(s) * kv_heads + kvh) * hd];
                float dot = 0.0f;
                for (int i = 0; i < hd; ++i) dot += qv[i] * kv[i];
                p[s] = dot * scale;
                m = std::max(m, p[s]);
            }
            float sum = 0.0f;
            for (int s = 0; s <= t; ++s) { p[s] = std::exp(p[s] - m); sum += p[s]; }

            float* out = &ctx[(static_cast<std::size_t>(t) * heads + h) * hd];
            for (int s = 0; s <= t; ++s) {
                const float* vv = &v[(static_cast<std::size_t>(s) * kv_heads + kvh) * hd];
                const float w = p[s] / sum;
                for (int i = 0; i < hd; ++i) out[i] += w * vv[i];
            }
        }
    }
    return ctx;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string wprefix = argc > 1 ? argv[1] : "models/qwen";
    const std::string rprefix = argc > 2 ? argv[2] : "models/ref";
    const WeightFile W(wprefix);
    const WeightFile R(rprefix);
    ReferenceCheck check(R);

    const QwenConfig c = QwenConfig::from(W);
    const int H = c.hidden;
    const int QD = c.q_dim();
    const int KVD = c.kv_dim();
    const int I = c.intermediate;

    std::printf("Qwen2: %d layers, hidden %d, %d/%d heads, head_dim %d, ffn %d, vocab %d\n",
                c.layers, H, c.heads, c.kv_heads, c.head_dim, I, c.vocab);

    // ---- every weight present, with the shape the forward pass will assume ----
    auto expect = [&](const std::string& n, int r, int cl) {
        if (!W.has(n) || W.rows(n) != r || W.cols(n) != cl) {
            std::printf("  weight %s: expected %d x %d\n", n.c_str(), r, cl);
            check.record_failure();
        }
    };
    expect("embed", c.vocab, H);
    expect("final_norm", 1, H);
    expect("lm_head", H, c.vocab);
    for (int l = 0; l < c.layers; ++l) {
        const std::string L = "L" + std::to_string(l) + ".";
        expect(L + "attn_norm", 1, H);
        expect(L + "wq", H, QD);   expect(L + "bq", 1, QD);
        expect(L + "wk", H, KVD);  expect(L + "bk", 1, KVD);
        expect(L + "wv", H, KVD);  expect(L + "bv", 1, KVD);
        expect(L + "wo", QD, H);
        expect(L + "mlp_norm", 1, H);
        expect(L + "w_gate", H, I);
        expect(L + "w_up", H, I);
        expect(L + "w_down", I, H);
    }
    std::printf("Weight shapes: %s\n\n", check.failures() ? "MISMATCH" : "all as expected");

    const std::vector<int>& tokens = R.ints("tokens");
    const int T = static_cast<int>(tokens.size());
    std::printf("Prompt: %d tokens\n", T);

    // ---- embedding lookup: should be bit-exact, it is only a copy ----
    Mat x(static_cast<std::size_t>(T) * H);
    {
        const Mat embed = W.host("embed");
        for (int t = 0; t < T; ++t)
            std::copy_n(&embed[static_cast<std::size_t>(tokens[t]) * H], H,
                        &x[static_cast<std::size_t>(t) * H]);
    }
    check.compare("embed_out", x);

    for (int l = 0; l < c.layers; ++l) {
        const std::string L = "L" + std::to_string(l) + ".";
        const bool detail = (l == 0);  // HF internals were recorded for layer 0

        // Attention sub-block:  x = x + o_proj(attn(rope(q), rope(k), v))
        const Mat a = rmsnorm(x, T, H, W.host(L + "attn_norm"), c.rms_eps);
        if (detail) check.compare("L0.attn_norm_out", a);

        const Mat bq = W.host(L + "bq"), bk = W.host(L + "bk"), bv = W.host(L + "bv");
        Mat q = linear(a, T, H, W.host(L + "wq"), QD, &bq);
        Mat k = linear(a, T, H, W.host(L + "wk"), KVD, &bk);
        const Mat v = linear(a, T, H, W.host(L + "wv"), KVD, &bv);
        if (detail) { check.compare("L0.q", q); check.compare("L0.k", k); check.compare("L0.v", v); }

        rope(q, T, c.heads, c.head_dim, c.rope_theta);
        rope(k, T, c.kv_heads, c.head_dim, c.rope_theta);
        if (detail) { check.compare("L0.q_rope", q); check.compare("L0.k_rope", k); }

        const Mat ctx = attention(q, k, v, T, c.heads, c.kv_heads, c.head_dim);
        if (detail) check.compare("L0.attn_ctx", ctx);

        const Mat o = linear(ctx, T, QD, W.host(L + "wo"), H);
        if (detail) check.compare("L0.attn_out", o);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += o[i];

        // MLP sub-block:  x = x + down(silu(gate(m)) * up(m))   -- SwiGLU
        const Mat m = rmsnorm(x, T, H, W.host(L + "mlp_norm"), c.rms_eps);
        if (detail) check.compare("L0.mlp_norm_out", m);

        Mat g = linear(m, T, H, W.host(L + "w_gate"), I);
        const Mat u = linear(m, T, H, W.host(L + "w_up"), I);
        for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        const Mat d = linear(g, T, I, W.host(L + "w_down"), H);
        if (detail) check.compare("L0.mlp_out", d);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += d[i];

        check.compare(L + "out", x);
    }

    const Mat f = rmsnorm(x, T, H, W.host("final_norm"), c.rms_eps);
    check.compare("final_norm_out", f);
    const Mat logits = linear(f, T, H, W.host("lm_head"), c.vocab);
    check.compare("logits", logits);

    // The prediction for the token after the prompt is the argmax of the last
    // row -- which must be the first token Hugging Face generated greedily.
    const float* last = &logits[static_cast<std::size_t>(T - 1) * c.vocab];
    const int next = static_cast<int>(std::max_element(last, last + c.vocab) - last);
    const int hf_next = R.ints("generated").front();
    const bool next_ok = (next == hf_next);
    if (!next_ok) check.record_failure();
    std::printf("\nNext token: %d (Hugging Face: %d)  %s\n", next, hf_next, next_ok ? "ok" : "FAIL");

    std::printf("Worst relative error: %.2e (%s)\n", check.worst(), check.worst_name().c_str());
    std::printf("%s\n", check.failures() ? "FAILED" : "CPU reference matches Hugging Face");
    return check.failures() ? 1 : 0;
}
