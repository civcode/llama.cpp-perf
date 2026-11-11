#pragma once

// Host-side opt-in comparison of two existing tiled flash-attention GQA layouts.
// Default behavior is unchanged. This is not a general attention optimization.

#include <climits>
#include <cstring>

namespace ggml_cuda_fattn_tune {

// Limit GQA columns per tile only for the HIP 256x256 attention kernels.
// Returns INT_MAX when not opted in, retaining the existing kernel dispatch.
inline int max_gqa_columns(const int dkq, const int dv, const bool hip, const char * value) {
    return hip && dkq == 256 && dv == 256 && value && std::strcmp(value, "2") == 0 ? 2 : INT_MAX;
}

} // namespace ggml_cuda_fattn_tune
