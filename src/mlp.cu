#include "mlp.h"
#include "kernels.h"
#include "cuda_utils.h"

#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

MLP::MLP(const std::vector<int>& dims, int batch_size, unsigned seed)
    : dims_(dims), batch_(batch_size) {
    if (dims.size() < 2) {
        std::fprintf(stderr, "MLP needs at least an input and an output dim\n");
        std::exit(EXIT_FAILURE);
    }

    const int n_layers = static_cast<int>(dims.size()) - 1;

    // reserve() before push_back matters here: Tensor is move-only, and while
    // its move constructor is noexcept (so vector *can* relocate), reserving
    // avoids the reallocation entirely.
    layers_.reserve(n_layers);
    activations_.reserve(n_layers);
    host_W_.reserve(n_layers);
    host_b_.reserve(n_layers);

    std::mt19937 rng(seed);

    for (int i = 0; i < n_layers; ++i) {
        const int in = dims[i];
        const int out = dims[i + 1];
        const bool is_last = (i == n_layers - 1);

        layers_.emplace_back(in, out, !is_last);
        activations_.emplace_back(batch_, out);

        // He initialisation: scale by sqrt(2 / fan_in). Without something like
        // this, activations either vanish or explode as depth grows, and a
        // 4-layer network would produce meaningless output that makes the
        // correctness check useless.
        const float scale = std::sqrt(2.0f / static_cast<float>(in));
        std::normal_distribution<float> dist(0.0f, scale);

        std::vector<float> hW(static_cast<std::size_t>(in) * out);
        for (float& v : hW) v = dist(rng);

        std::vector<float> hb(static_cast<std::size_t>(out));
        for (float& v : hb) v = dist(rng) * 0.1f;

        layers_[i].W.upload(hW);
        layers_[i].b.upload(hb);

        host_W_.push_back(std::move(hW));
        host_b_.push_back(std::move(hb));
    }
}

void MLP::forward(const Tensor& X, Tensor& Y) {
    const int n = static_cast<int>(layers_.size());

    // `cur` walks the chain: it starts at the input and then points at each
    // layer's output buffer in turn. Using a pointer avoids copying tensors
    // between layers.
    const Tensor* cur = &X;

    for (int i = 0; i < n - 1; ++i) {
        // Hidden layer: matrix multiply, bias and ReLU in a single kernel.
        gemm_bias_relu_fused(*cur, layers_[i].W, layers_[i].b, activations_[i]);
        cur = &activations_[i];
    }

    // Output layer: bias but no activation. Applying ReLU here would clamp
    // every negative logit to zero and destroy the distribution softmax is
    // about to produce.
    gemm_bias(*cur, layers_[n - 1].W, layers_[n - 1].b, activations_[n - 1]);

    // Row-wise softmax turns logits into probabilities.
    softmax_online(activations_[n - 1], Y);
}

void MLP::dump(const std::string& path, const Tensor& X, const Tensor& Y) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "could not open %s for writing\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }

    const std::int32_t n_layers = static_cast<std::int32_t>(layers_.size());
    const std::int32_t batch = batch_;
    std::fwrite(&n_layers, sizeof(std::int32_t), 1, f);
    std::fwrite(&batch, sizeof(std::int32_t), 1, f);

    for (const Linear& L : layers_) {
        const std::int32_t in = L.in_features;
        const std::int32_t out = L.out_features;
        const std::int32_t relu = L.relu ? 1 : 0;
        std::fwrite(&in, sizeof(std::int32_t), 1, f);
        std::fwrite(&out, sizeof(std::int32_t), 1, f);
        std::fwrite(&relu, sizeof(std::int32_t), 1, f);
    }

    const std::vector<float> hX = X.download();
    std::fwrite(hX.data(), sizeof(float), hX.size(), f);

    for (std::size_t i = 0; i < host_W_.size(); ++i) {
        std::fwrite(host_W_[i].data(), sizeof(float), host_W_[i].size(), f);
        std::fwrite(host_b_[i].data(), sizeof(float), host_b_[i].size(), f);
    }

    const std::vector<float> hY = Y.download();
    std::fwrite(hY.data(), sizeof(float), hY.size(), f);

    std::fclose(f);
}

// ---------------------------------------------------------------------------
// CPU reference for the entire network.
//
// Deliberately simple: nested loops, double accumulation, no attempt at speed.
// Its only job is to be obviously correct.
// ---------------------------------------------------------------------------
std::vector<float> mlp_cpu_reference(const std::vector<int>& dims,
                                     int batch,
                                     const std::vector<float>& X,
                                     const std::vector<std::vector<float>>& Ws,
                                     const std::vector<std::vector<float>>& bs) {
    const int n_layers = static_cast<int>(dims.size()) - 1;
    std::vector<float> cur = X;

    for (int L = 0; L < n_layers; ++L) {
        const int in = dims[L];
        const int out = dims[L + 1];
        const bool is_last = (L == n_layers - 1);

        std::vector<float> next(static_cast<std::size_t>(batch) * out);

        for (int m = 0; m < batch; ++m) {
            for (int o = 0; o < out; ++o) {
                double acc = 0.0;
                for (int k = 0; k < in; ++k) {
                    acc += static_cast<double>(cur[static_cast<std::size_t>(m) * in + k])
                         * static_cast<double>(Ws[L][static_cast<std::size_t>(k) * out + o]);
                }
                acc += bs[L][o];
                if (!is_last && acc < 0.0) acc = 0.0;   // ReLU on hidden layers
                next[static_cast<std::size_t>(m) * out + o] = static_cast<float>(acc);
            }
        }
        cur = std::move(next);
    }

    // Final softmax, row-wise, with the max subtracted for stability.
    const int out = dims.back();
    for (int m = 0; m < batch; ++m) {
        float* row = cur.data() + static_cast<std::size_t>(m) * out;

        float mx = -FLT_MAX;
        for (int j = 0; j < out; ++j) mx = std::fmax(mx, row[j]);

        double s = 0.0;
        for (int j = 0; j < out; ++j) s += std::exp(static_cast<double>(row[j]) - mx);

        for (int j = 0; j < out; ++j) {
            row[j] = static_cast<float>(std::exp(static_cast<double>(row[j]) - mx) / s);
        }
    }

    return cur;
}