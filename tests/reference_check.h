#pragma once

#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Compares computed activations against the ones Hugging Face recorded, and
// keeps score. Shared by the CPU reference check and the GPU check.
//
// Error is reported relative to the largest magnitude in the reference rather
// than elementwise: a per-element relative error explodes on values near zero
// and would fail correct code. Qwen's residual stream also carries a few very
// large outlier channels, so this is the scale a reader of the numbers cares
// about.
//
// The pass threshold sits between two measured populations. Code that only
// sums in a different order from Hugging Face stays below 1e-4 at every
// checkpoint (worst: the attention output, 6e-5 on the CPU reference). Four
// real architecture bugs injected into the CPU reference -- adjacent-pair RoPE,
// the wrong query-to-KV head mapping, a dropped k bias, sqrt missing from the
// attention scale -- are each off by 0.2 to 1.0 at the first checkpoint they
// reach, and each changes the predicted next token. 1e-3 leaves headroom for
// fp32 reordering on another device and still catches all four.
//
// Downstream of a bug the error can shrink to ~1e-3, because layers 2-20 carry
// outlier channels near 1,600 that dominate max|ref|. That is why the layer-0
// internals and the generated tokens are checked as well, not only the layer
// outputs.
class ReferenceCheck {
public:
    static constexpr double kTolerance = 1e-3;

    explicit ReferenceCheck(const WeightFile& ref) : ref_(ref) {}

    void compare(const std::string& name, const std::vector<float>& got) {
        const std::vector<float> want = ref_.host(name);
        if (want.size() != got.size()) {
            std::printf("  %-18s SHAPE MISMATCH (%zu vs %zu values)\n",
                        name.c_str(), got.size(), want.size());
            ++failures_;
            return;
        }

        // A NaN never compares greater than anything, so std::max would quietly
        // skip it and a NaN-filled tensor would pass. Count them separately.
        std::size_t non_finite = 0;
        double max_diff = 0.0, max_ref = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) {
                ++non_finite;
                continue;
            }
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(got[i]) - want[i]));
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(want[i])));
        }
        if (non_finite > 0) {
            std::printf("  %-18s %zu NON-FINITE VALUES\n", name.c_str(), non_finite);
            ++failures_;
            return;
        }

        const double rel = max_diff / std::max(max_ref, 1e-12);
        const bool ok = rel < kTolerance;
        if (!ok) ++failures_;
        if (rel > worst_) {
            worst_ = rel;
            worst_name_ = name;
        }
        std::printf("  %-18s max|diff| %.3e   max|ref| %9.3f   rel %.2e   %s\n",
                    name.c_str(), max_diff, max_ref, rel, ok ? "ok" : "FAIL");
    }

    // For a check that is not a tensor comparison, such as a generated token.
    void record_failure() { ++failures_; }

    int failures() const { return failures_; }
    double worst() const { return worst_; }
    const std::string& worst_name() const { return worst_name_; }

private:
    const WeightFile& ref_;
    int failures_ = 0;
    double worst_ = 0.0;
    std::string worst_name_;
};
