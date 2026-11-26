#include "tensor.h"
#include "cuda_utils.h"

#include <cstdio>

Tensor::Tensor(int rows, int cols) : rows_(rows), cols_(cols) {
    // cudaMalloc takes a void** -- it writes the allocated address *into* your
    // pointer, so you pass the address of the pointer itself.
    CUDA_CHECK(cudaMalloc(&d_data_, bytes()));
}

Tensor::~Tensor() {
    // Deliberately NOT using CUDA_CHECK here. CUDA_CHECK calls std::exit on
    // failure, and aborting the process from inside a destructor -- which may
    // itself be running during stack unwinding -- is a bad idea. Destructors
    // should not throw and should not kill the program.
    if (d_data_ != nullptr) {
        cudaFree(d_data_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : d_data_(other.d_data_), rows_(other.rows_), cols_(other.cols_) {
    // Critical: null out the source. Both objects now believe they own the same
    // pointer; if we skip this, the moved-from destructor frees memory the new
    // owner is still using (a double free).
    other.d_data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {          // self-assignment guard: `t = std::move(t)`
        if (d_data_ != nullptr) {  // release what we currently own first
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        rows_ = other.rows_;
        cols_ = other.cols_;

        other.d_data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;  // return by reference so `a = b = c` chains
}

void Tensor::upload(const std::vector<float>& host) {
    if (host.size() != size()) {
        std::fprintf(stderr,
                     "[Tensor::upload] size mismatch: host=%zu, tensor=%zu\n",
                     host.size(), size());
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host.data(), bytes(), cudaMemcpyHostToDevice));
}

std::vector<float> Tensor::download() const {
    std::vector<float> host(size());
    CUDA_CHECK(cudaMemcpy(host.data(), d_data_, bytes(), cudaMemcpyDeviceToHost));
    return host;  // moved out, not copied (guaranteed since C++11 via NRVO/move)
}

// ---------------------------------------------------------------------------
// HalfTensor
// ---------------------------------------------------------------------------

namespace {

// dst[i] = src[i] rounded to the nearest fp16, ties to even (the IEEE default
// rounding). A grid-stride loop, so any grid size covers any n. It runs once
// per weight matrix while the model loads, so it is written for clarity, not
// speed.
__global__ void round_to_half_kernel(const float* __restrict__ src,
                                     __half* __restrict__ dst, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2half_rn(src[i]);
    }
}

}  // namespace

HalfTensor::HalfTensor(int rows, int cols) : rows_(rows), cols_(cols) {
    CUDA_CHECK(cudaMalloc(&d_data_, bytes()));
}

HalfTensor::HalfTensor(const Tensor& src) : HalfTensor(src.rows(), src.cols()) {
    const std::size_t n = size();
    if (n == 0) {
        return;  // a launch with zero blocks is an error, and there is nothing to do
    }
    const int threads = 256;
    std::size_t blocks = (n + threads - 1) / threads;
    if (blocks > 4096) {
        blocks = 4096;  // plenty to fill any current GPU; the loop covers the rest
    }
    round_to_half_kernel<<<static_cast<unsigned>(blocks), threads>>>(src.data(), d_data_, n);
    CUDA_CHECK_LAST();
}

HalfTensor::~HalfTensor() {
    // No CUDA_CHECK in a destructor, for the reason given in Tensor::~Tensor.
    if (d_data_ != nullptr) {
        cudaFree(d_data_);
    }
}

HalfTensor::HalfTensor(HalfTensor&& other) noexcept
    : d_data_(other.d_data_), rows_(other.rows_), cols_(other.cols_) {
    other.d_data_ = nullptr;  // the moved-from object must not free it too
    other.rows_ = 0;
    other.cols_ = 0;
}

HalfTensor& HalfTensor::operator=(HalfTensor&& other) noexcept {
    if (this != &other) {
        if (d_data_ != nullptr) {
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        rows_ = other.rows_;
        cols_ = other.cols_;

        other.d_data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}
