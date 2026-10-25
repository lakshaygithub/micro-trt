#pragma once

#include "qwen_config.h"
#include "tensor.h"
#include "weights.h"

#include <functional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Qwen2.5 inference on micro-trt's own kernels.
//
// Every projection is a GEMM from the earlier stages -- gemm_bias for q, k and
// v, which carry biases in Qwen2, and gemm_regtiled for the rest -- and every
// operation between them is in src/transformer.cu. Nothing calls cuBLAS or
// cuDNN.
//
// One forward() serves both phases of generation:
//   prefill -- the whole prompt at once:   forward(prompt, pos = 0, ...)
//   decode  -- one new token at a time:    forward({token}, pos = tokens so far, ...)
// Each call appends its keys and values to the per-layer KV cache and attends
// over everything cached before it, so decode never recomputes the past.
// ---------------------------------------------------------------------------

// Scratch buffers for forward() over a fixed number of tokens. Allocated once
// and reused, so a forward pass never calls cudaMalloc.
struct Activations {
    Activations(const QwenConfig& c, int n);

    int tokens;       // tokens per forward()
    Tensor x;         // residual stream             T x hidden
    Tensor norm;      // RMSNorm output              T x hidden
    Tensor q;         //                             T x heads*head_dim
    Tensor k, v;      // this call's keys / values   T x kv_heads*head_dim
    Tensor ctx;       // attention output            T x heads*head_dim
    Tensor proj;      // o_proj / down_proj output   T x hidden
    Tensor gate, up;  // MLP                         T x intermediate
    Tensor logits;    //                             T x vocab
};

struct QwenLayer {
    QwenLayer(const WeightFile& w, int layer, int max_seq, int kv_dim);

    Tensor attn_norm;
    Tensor wq, bq, wk, bk, wv, bv, wo;  // projection weights stored (in x out)
    Tensor mlp_norm;
    Tensor w_gate, w_up, w_down;
    Tensor k_cache, v_cache;            // max_seq x kv_heads*head_dim, post-RoPE
};

// Called at each point where Hugging Face's activations were recorded, with the
// same names ("embed_out", "L3.q_rope", "logits", ...), so a test can compare
// the forward pass against them while it runs. Unset in normal use.
using Tap = std::function<void(const std::string& name, const Tensor& value)>;

class Qwen {
public:
    // Uploads every weight (~2.5 GB for Qwen2.5-0.5B in fp32) and allocates a
    // KV cache for max_seq positions.
    Qwen(const WeightFile& weights, int max_seq);
    ~Qwen();

    Qwen(const Qwen&) = delete;
    Qwen& operator=(const Qwen&) = delete;

    // Runs `tokens` at positions pos .. pos + T - 1, leaving logits for every
    // one of them in act.logits.
    void forward(const std::vector<int>& tokens, int pos, Activations& act,
                 const Tap& tap = nullptr);

    // Greedy decoding: prefill the prompt, then repeatedly feed back the
    // highest-scoring token. Returns the n_new generated tokens.
    std::vector<int> generate(const std::vector<int>& prompt, int n_new);

    const QwenConfig& config() const { return cfg_; }

private:
    QwenConfig cfg_;
    int max_seq_;
    Tensor embed_;       // vocab x hidden: a lookup table, not transposed
    Tensor final_norm_;
    Tensor lm_head_;     // hidden x vocab
    Tensor inv_freq_;    // 1 x head_dim/2, RoPE inverse frequencies
    std::vector<QwenLayer> layers_;
    int* d_tokens_ = nullptr;  // device copy of the token ids being processed
};
