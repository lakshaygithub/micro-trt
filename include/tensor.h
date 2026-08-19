#pragma once

#include <cstddef>
#include <vector>

// A 2-D, row-major, float32 tensor whose storage lives in GPU global memory.
//
// Ownership model: a Tensor *owns* its device allocation. Construction acquires
// it (cudaMalloc), destruction releases it (cudaFree). This is RAII, and it is
// the reason this class cannot leak GPU memory even if the caller returns early
// or throws.
//
// The type is move-only: copying would require a second cudaMalloc plus a
// device-to-device copy, which is expensive and almost never what you meant to
// write. Deleting the copy operations turns an accidental expensive copy into a
// compile error instead of a silent performance bug.
class Tensor {
public:
    Tensor(int rows, int cols);
    ~Tensor();

    // Copy operations: deleted (see note above).
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move operations: transfer ownership of the device pointer.
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Host <-> device transfers. `upload` copies CPU data in, `download` copies
    // the current device contents back out into a fresh host vector.
    void upload(const std::vector<float>& host);
    std::vector<float> download() const;

    // Raw device pointer, for handing to kernels.
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // size_t, not int: a 30000x30000 tensor overflows a 32-bit int.
    std::size_t size() const {
        return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
    }
    std::size_t bytes() const { return size() * sizeof(float); }

private:
    float* d_data_ = nullptr;  // device pointer -- NOT dereferenceable from host code
    int rows_ = 0;
    int cols_ = 0;
};
