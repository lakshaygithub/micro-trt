#pragma once

#include "weights.h"

// Hyperparameters of a Qwen2-family model, read from the index written by
// tools/export_qwen.py. Shared by the CPU reference and the GPU engine, so the
// two can never disagree about the shape of the model.
struct QwenConfig {
    int hidden;        // width of the residual stream
    int layers;
    int heads;         // query heads
    int kv_heads;      // key/value heads; heads / kv_heads query heads share each
    int head_dim;
    int intermediate;  // MLP width
    int vocab;
    double rope_theta;
    double rms_eps;

    int q_dim() const { return heads * head_dim; }
    int kv_dim() const { return kv_heads * head_dim; }

    static QwenConfig from(const WeightFile& w) {
        QwenConfig c{};
        c.hidden = static_cast<int>(w.config("hidden"));
        c.layers = static_cast<int>(w.config("layers"));
        c.heads = static_cast<int>(w.config("heads"));
        c.kv_heads = static_cast<int>(w.config("kv_heads"));
        c.head_dim = static_cast<int>(w.config("head_dim"));
        c.intermediate = static_cast<int>(w.config("intermediate"));
        c.vocab = static_cast<int>(w.config("vocab"));
        c.rope_theta = w.config("rope_theta");
        c.rms_eps = w.config("rms_eps");
        return c;
    }
};
