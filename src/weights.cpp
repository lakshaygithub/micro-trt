#include "weights.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace {

[[noreturn]] void fail(const std::string& msg) {
    std::fprintf(stderr, "[WEIGHTS ERROR] %s\n", msg.c_str());
    std::exit(EXIT_FAILURE);
}

}  // namespace

WeightFile::WeightFile(const std::string& prefix) : bin_path_(prefix + ".bin") {
    const std::string idx_path = prefix + ".idx";
    std::ifstream idx(idx_path);
    if (!idx) fail("cannot open " + idx_path + " -- run tools/export_qwen.py first");

    std::string line;
    int line_no = 0;
    while (std::getline(idx, line)) {
        ++line_no;
        if (line.empty()) continue;

        std::istringstream in(line);
        std::string kind, name;
        in >> kind >> name;

        if (kind == "tensor") {
            Entry e{};
            in >> e.offset >> e.rows >> e.cols;
            if (!in) fail(idx_path + ":" + std::to_string(line_no) + ": bad tensor line");
            tensors_[name] = e;
        } else if (kind == "config") {
            double v = 0.0;
            in >> v;
            if (!in) fail(idx_path + ":" + std::to_string(line_no) + ": bad config line");
            config_[name] = v;
        } else if (kind == "ints") {
            std::vector<int> vals;
            for (int v; in >> v;) vals.push_back(v);
            ints_[name] = std::move(vals);
        } else {
            fail(idx_path + ":" + std::to_string(line_no) + ": unknown entry '" + kind + "'");
        }
    }

    std::ifstream bin(bin_path_, std::ios::binary | std::ios::ate);
    if (!bin) fail("cannot open " + bin_path_);
    const auto file_bytes = static_cast<std::size_t>(bin.tellg());

    // Catch a truncated download or a mismatched .idx/.bin pair now, rather
    // than as garbage weights three layers into a forward pass.
    for (const auto& [n, e] : tensors_) {
        const std::size_t end = e.offset + static_cast<std::size_t>(e.rows) * e.cols * sizeof(float);
        if (end > file_bytes) fail(n + " extends past the end of " + bin_path_);
    }
}

bool WeightFile::has(const std::string& name) const {
    return tensors_.count(name) != 0;
}

const WeightFile::Entry& WeightFile::entry(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("no tensor named '" + name + "'");
    return it->second;
}

int WeightFile::rows(const std::string& name) const { return entry(name).rows; }
int WeightFile::cols(const std::string& name) const { return entry(name).cols; }

std::vector<float> WeightFile::host(const std::string& name) const {
    const Entry& e = entry(name);
    std::vector<float> out(static_cast<std::size_t>(e.rows) * e.cols);

    std::ifstream bin(bin_path_, std::ios::binary);
    bin.seekg(static_cast<std::streamoff>(e.offset));
    bin.read(reinterpret_cast<char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!bin) fail("short read for '" + name + "'");
    return out;
}

double WeightFile::config(const std::string& key) const {
    auto it = config_.find(key);
    if (it == config_.end()) fail("no config value '" + key + "'");
    return it->second;
}

const std::vector<int>& WeightFile::ints(const std::string& key) const {
    auto it = ints_.find(key);
    if (it == ints_.end()) fail("no int list '" + key + "'");
    return it->second;
}
