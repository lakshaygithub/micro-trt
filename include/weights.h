#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Reads a dataset written by tools/mtrt_format.py: a text index <prefix>.idx
// plus raw float32 values in <prefix>.bin. Used for both the exported model
// weights and the Hugging Face reference activations.
//
// Deliberately host-only, with no CUDA in it: getting a tensor onto the GPU is
// one line with the existing Tensor class,
//
//     Tensor t(wf.rows(name), wf.cols(name));
//     t.upload(wf.host(name));
//
// and keeping the loader free of CUDA lets the CPU reference model that
// validates it build and run on a machine with no GPU.
class WeightFile {
public:
    explicit WeightFile(const std::string& prefix);

    bool has(const std::string& name) const;
    int rows(const std::string& name) const;
    int cols(const std::string& name) const;

    // Reads one tensor from disk. Nothing is cached: a 0.5B model is 2.5 GB in
    // fp32, and holding a host copy of every weight would double that.
    std::vector<float> host(const std::string& name) const;

    double config(const std::string& key) const;
    const std::vector<int>& ints(const std::string& key) const;

private:
    struct Entry {
        std::size_t offset;  // bytes into the .bin file
        int rows;
        int cols;
    };

    const Entry& entry(const std::string& name) const;

    std::string bin_path_;
    std::unordered_map<std::string, Entry> tensors_;
    std::unordered_map<std::string, double> config_;
    std::unordered_map<std::string, std::vector<int>> ints_;
};
