#include "qwen.h"
#include "kernels.h"
#include "cuda_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace {

Tensor load(const WeightFile& w, const std::string& name) {
    Tensor t(w.rows(name), w.cols(name));
    t.upload(w.host(name));
    return t;
}

std::string layer_name(int layer, const char* name) {
    return "L" + std::to_string(layer) + "." + name;
}

// Writes src's rows into dst starting at row `row`. In a row-major buffer those
// rows are one contiguous block, so appending to the KV cache is one copy.
void store_rows(const Tensor& src, Tensor& dst, int row) {
    require(src.cols() == dst.cols() && row >= 0 && row + src.rows() <= dst.rows(),
            "store_rows: rows do not fit in the destination");
    CUDA_CHECK(cudaMemcpyAsync(dst.data() + static_cast<std::size_t>(row) * dst.cols(),
                               src.data(), src.bytes(), cudaMemcpyDeviceToDevice));
}

// Greedy decoding needs only the last position's logits, so only that row
// crosses the bus: ~600 KB instead of T times that.
int argmax_last_row(const Tensor& logits) {
    std::vector<float> row(static_cast<std::size_t>(logits.cols()));
    const float* last = logits.data() + static_cast<std::size_t>(logits.rows() - 1) * logits.cols();
    CUDA_CHECK(cudaMemcpy(row.data(), last, row.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // max_element returns the first maximum, as torch.argmax does, so an exact
    // tie resolves the same way Hugging Face resolves it.
    return static_cast<int>(std::max_element(row.begin(), row.end()) - row.begin());
}

}  // namespace

Activations::Activations(const QwenConfig& c, int n)
    : tokens(n),
      x(n, c.hidden), norm(n, c.hidden),
      q(n, c.q_dim()), k(n, c.kv_dim()), v(n, c.kv_dim()),
      ctx(n, c.q_dim()), proj(n, c.hidden),
      gate(n, c.intermediate), up(n, c.intermediate),
      logits(n, c.vocab) {}

QwenLayer::QwenLayer(const WeightFile& w, int layer, int max_seq, int kv_dim)
    : attn_norm(load(w, layer_name(layer, "attn_norm"))),
      wq(load(w, layer_name(layer, "wq"))),
      bq(load(w, layer_name(layer, "bq"))),
      wk(load(w, layer_name(layer, "wk"))),
      bk(load(w, layer_name(layer, "bk"))),
      wv(load(w, layer_name(layer, "wv"))),
      bv(load(w, layer_name(layer, "bv"))),
      wo(load(w, layer_name(layer, "wo"))),
      mlp_norm(load(w, layer_name(layer, "mlp_norm"))),
      w_gate(load(w, layer_name(layer, "w_gate"))),
      w_up(load(w, layer_name(layer, "w_up"))),
      w_down(load(w, layer_name(layer, "w_down"))),
      k_cache(max_seq, kv_dim),
      v_cache(max_seq, kv_dim) {}

Qwen::Qwen(const WeightFile& w, int max_seq)
    : cfg_(QwenConfig::from(w)),
      max_seq_(max_seq),
      embed_(load(w, "embed")),
      final_norm_(load(w, "final_norm")),
      lm_head_(load(w, "lm_head")),
      inv_freq_(1, cfg_.head_dim / 2) {
    require(max_seq > 0, "Qwen: max_seq must be positive");
    require(cfg_.kv_heads > 0 && cfg_.heads % cfg_.kv_heads == 0,
            "Qwen: heads must be a multiple of kv_heads");

    // RoPE inverse frequencies 1 / theta^(2i / head_dim), computed in fp32 the
    // way Hugging Face computes them.
    std::vector<float> inv(static_cast<std::size_t>(cfg_.head_dim / 2));
    for (int i = 0; i < cfg_.head_dim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(cfg_.head_dim);
        inv[i] = 1.0f / std::pow(static_cast<float>(cfg_.rope_theta), exponent);
    }
    inv_freq_.upload(inv);

    layers_.reserve(cfg_.layers);
    for (int l = 0; l < cfg_.layers; ++l) {
        layers_.emplace_back(w, l, max_seq, cfg_.kv_dim());
    }

    CUDA_CHECK(cudaMalloc(&d_tokens_, static_cast<std::size_t>(max_seq) * sizeof(int)));
}

Qwen::~Qwen() {
    // No CUDA_CHECK in a destructor, for the reason given in Tensor::~Tensor.
    if (d_tokens_ != nullptr) {
        cudaFree(d_tokens_);
    }
}

void Qwen::forward(const std::vector<int>& tokens, int pos, Activations& a,
                   const Tap& tap) {
    const int T = static_cast<int>(tokens.size());
    require(T == a.tokens, "forward: Activations were sized for a different token count");
    require(pos >= 0 && pos + T <= max_seq_, "forward: positions exceed max_seq");
    for (int id : tokens) {
        require(id >= 0 && id < cfg_.vocab, "forward: token id out of range");
    }

    CUDA_CHECK(cudaMemcpy(d_tokens_, tokens.data(), T * sizeof(int),
                          cudaMemcpyHostToDevice));
    embedding_lookup(embed_, d_tokens_, a.x);
    if (tap) tap("embed_out", a.x);

    const float eps = static_cast<float>(cfg_.rms_eps);

    for (int l = 0; l < cfg_.layers; ++l) {
        QwenLayer& L = layers_[l];
        const std::string prefix = tap ? "L" + std::to_string(l) + "." : std::string();
        auto probe = [&](const char* name, const Tensor& t) {
            if (tap) tap(prefix + name, t);
        };

        // ---- attention:  x += o_proj(attention(rope(q), rope(k), v)) ----
        rmsnorm(a.x, L.attn_norm, eps, a.norm);
        probe("attn_norm_out", a.norm);

        gemm_bias(a.norm, L.wq, L.bq, a.q);
        gemm_bias(a.norm, L.wk, L.bk, a.k);
        gemm_bias(a.norm, L.wv, L.bv, a.v);
        probe("q", a.q);
        probe("k", a.k);
        probe("v", a.v);

        rope_inplace(a.q, cfg_.heads, pos, inv_freq_);
        rope_inplace(a.k, cfg_.kv_heads, pos, inv_freq_);
        probe("q_rope", a.q);
        probe("k_rope", a.k);

        // Keys are cached after RoPE: a position's rotation never changes, so
        // later steps reuse them as they are.
        store_rows(a.k, L.k_cache, pos);
        store_rows(a.v, L.v_cache, pos);

        attention(a.q, L.k_cache, L.v_cache, pos, cfg_.heads, cfg_.kv_heads, a.ctx);
        probe("attn_ctx", a.ctx);

        gemm_regtiled(a.ctx, L.wo, a.proj);
        probe("attn_out", a.proj);
        add_inplace(a.x, a.proj);

        // ---- MLP (SwiGLU):  x += down(silu(gate(m)) * up(m)) ----
        rmsnorm(a.x, L.mlp_norm, eps, a.norm);
        probe("mlp_norm_out", a.norm);

        gemm_regtiled(a.norm, L.w_gate, a.gate);
        gemm_regtiled(a.norm, L.w_up, a.up);
        silu_mul_inplace(a.gate, a.up);
        gemm_regtiled(a.gate, L.w_down, a.proj);
        probe("mlp_out", a.proj);
        add_inplace(a.x, a.proj);

        probe("out", a.x);
    }

    rmsnorm(a.x, final_norm_, eps, a.norm);
    if (tap) tap("final_norm_out", a.norm);

    gemm_regtiled(a.norm, lm_head_, a.logits);
    if (tap) tap("logits", a.logits);
}

std::vector<int> Qwen::generate(const std::vector<int>& prompt, int n_new) {
    require(!prompt.empty(), "generate: empty prompt");
    require(n_new >= 0, "generate: n_new must be non-negative");

    // One set of buffers sized for the prompt, one for single tokens.
    Activations prefill(cfg_, static_cast<int>(prompt.size()));
    Activations step(cfg_, 1);

    forward(prompt, 0, prefill);
    int next = argmax_last_row(prefill.logits);
    int pos = static_cast<int>(prompt.size());

    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(n_new));
    for (int i = 0; i < n_new; ++i) {
        out.push_back(next);
        if (i + 1 < n_new) {
            forward({next}, pos++, step);
            next = argmax_last_row(step.logits);
        }
    }
    return out;
}
