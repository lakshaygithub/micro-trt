#pragma once

#include "qwen_config.h"
#include "tensor.h"
#include "weights.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Qwen2.5 inference on micro-trt's own kernels.
//
// Every projection goes through project(): the register-tiled GEMM when a call
// carries several tokens (prefill), and a GEMV when it carries one (decode).
// Both read the projection weights in fp32 or fp16, whichever the model was
// loaded with, and compute in fp32 either way. Every operation between
// projections is in src/transformer.cu. Nothing calls cuBLAS or cuDNN.
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

// How the projection weights are stored on the GPU: the seven matrices of each
// layer plus the LM head, nearly all of the bytes a decode step reads. Decode
// reads every one of them for each token, so their size sets its speed limit,
// and fp16 halves it. Activations, biases, norm weights, the embedding table
// and all arithmetic stay fp32.
//
// For Qwen2.5 the conversion is close to lossless: the checkpoint is bf16, and
// every bf16 value in fp16's normal range is exactly representable in fp16.
enum class WeightPrecision { fp32, fp16 };

inline const char* precision_name(WeightPrecision p) {
    return p == WeightPrecision::fp16 ? "fp16" : "fp32";
}

// One projection's weight matrix, stored (in x out) in either precision.
// Exactly one of the two is filled.
struct Weight {
    std::optional<Tensor> fp32;
    std::optional<HalfTensor> fp16;
};

struct QwenLayer {
    QwenLayer(const WeightFile& w, int layer, int max_seq, int kv_dim, WeightPrecision p);

    Tensor attn_norm;
    Weight wq;  Tensor bq;   // projection weights, stored (in x out)
    Weight wk;  Tensor bk;
    Weight wv;  Tensor bv;
    Weight wo;
    Tensor mlp_norm;
    Weight w_gate, w_up, w_down;
    Tensor k_cache, v_cache; // max_seq x kv_heads*head_dim, post-RoPE
};

// Called at each point where Hugging Face's activations were recorded, with the
// same names ("embed_out", "L3.q_rope", "logits", ...), so a test can compare
// the forward pass against them while it runs. Unset in normal use.
using Tap = std::function<void(const std::string& name, const Tensor& value)>;

class Qwen {
public:
    // Uploads every weight and allocates a KV cache for max_seq positions. For
    // Qwen2.5-0.5B the weights take ~2.5 GB of GPU memory in fp32 and ~1.5 GB
    // with fp16 projections.
    Qwen(const WeightFile& weights, int max_seq,
         WeightPrecision precision = WeightPrecision::fp16);
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
    WeightPrecision precision() const { return precision_; }

    // Single-token projections use the GEMV (the default) or, when switched
    // off, the same tiled GEMM as prefill. The switch exists so the benchmark
    // and the GPU check can run both paths on identical inputs.
    void set_decode_gemv(bool on) { decode_gemv_ = on; }

private:
    // y = x * W (+ bias), choosing GEMM or GEMV by the number of rows in x,
    // and the fp32 or fp16 version of each by how W is stored.
    void project(const Tensor& x, const Weight& W, const Tensor* bias, Tensor& y) const;

    QwenConfig cfg_;
    int max_seq_;
    WeightPrecision precision_;
    Tensor embed_;       // vocab x hidden: a lookup table, not transposed
    Tensor final_norm_;
    Weight lm_head_;     // hidden x vocab
    Tensor inv_freq_;    // 1 x head_dim/2, RoPE inverse frequencies
    std::vector<QwenLayer> layers_;
    int* d_tokens_ = nullptr;  // device copy of the token ids being processed
    bool decode_gemv_ = true;
};
