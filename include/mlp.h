#pragma once

#include "tensor.h"

#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// A multi-layer perceptron forward pass, assembled from the kernels built in
// earlier stages.
//
// For each hidden layer:   H = ReLU(H_prev * W + b)      one fused kernel
// For the output layer:    logits = H * W + b            bias, no activation
//                          Y = softmax(logits)           row-wise
//
// The point of this stage is that no new kernels are needed. Everything here is
// composition: a fused GEMM epilogue for the hidden layers, a bias-only variant
// for the last one, and the online softmax on top.
// ---------------------------------------------------------------------------

struct Linear {
    Tensor W;      // (in_features x out_features), row-major
    Tensor b;      // (1 x out_features), broadcast across rows
    int in_features;
    int out_features;
    bool relu;     // hidden layers apply ReLU; the output layer does not

    Linear(int in, int out, bool apply_relu)
        : W(in, out), b(1, out),
          in_features(in), out_features(out), relu(apply_relu) {}
};

class MLP {
public:
    // `dims` lists layer widths including input and output:
    //   {784, 512, 256, 10} is a three-layer network.
    // Every layer but the last applies ReLU; the last is followed by softmax.
    MLP(const std::vector<int>& dims, int batch_size, unsigned seed);

    // Runs the network. `X` is (batch x dims.front()); `Y` receives
    // (batch x dims.back()) of softmax probabilities.
    //
    // Allocates nothing: every intermediate buffer was created in the
    // constructor. A forward pass that calls cudaMalloc would be measuring the
    // allocator as much as the kernels.
    void forward(const Tensor& X, Tensor& Y);

    // Writes weights, input and output to a binary file so an independent
    // implementation can verify the result and time the same network.
    void dump(const std::string& path, const Tensor& X, const Tensor& Y) const;

    int layer_count() const { return static_cast<int>(layers_.size()); }
    const std::vector<int>& dims() const { return dims_; }

private:
    std::vector<int> dims_;
    int batch_;
    std::vector<Linear> layers_;
    std::vector<Tensor> activations_;  // one buffer per layer output
    std::vector<std::vector<float>> host_W_;  // kept for dumping
    std::vector<std::vector<float>> host_b_;
};

// CPU reference for the whole network, used to validate the GPU result.
std::vector<float> mlp_cpu_reference(const std::vector<int>& dims,
                                     int batch,
                                     const std::vector<float>& X,
                                     const std::vector<std::vector<float>>& Ws,
                                     const std::vector<std::vector<float>>& bs);