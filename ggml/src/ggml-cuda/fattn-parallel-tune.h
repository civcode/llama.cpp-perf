#pragma once

// Experimental HIP RDNA tile-splitting override for flash-attention 256x256.
// This is a *fixed-work performance experiment*, never enabled by default.
// All choices use the existing multi-block partial-result combination.
// One and two blocks are deliberately unavailable: 32K/98K deterministic
// long-KV tests in Report 0006 failed the CPU-reference tolerance.

#include <algorithm>
#include <cstring>

namespace ggml_cuda_fattn_parallel_tune {

inline int select_parallel_blocks(
        const int existing,
        const int ntiles_kv,
        const int dkq,
        const int dv,
        const bool is_hip_rdna,
        const char * setting) {
    if (!is_hip_rdna || dkq != 256 || dv != 256 || !setting || ntiles_kv <= 0) {
        return existing;
    }

    int requested = 0;
    if      (std::strcmp(setting, "4"  ) == 0) requested = 4;
    else if (std::strcmp(setting, "8"  ) == 0) requested = 8;
    else if (std::strcmp(setting, "16" ) == 0) requested = 16;
    else if (std::strcmp(setting, "32" ) == 0) requested = 32;
    else if (std::strcmp(setting, "64" ) == 0) requested = 64;
    else if (std::strcmp(setting, "96" ) == 0) requested = 96;
    else if (std::strcmp(setting, "128") == 0) requested = 128;
    else if (std::strcmp(setting, "192") == 0) requested = 192;
    else if (std::strcmp(setting, "256") == 0) requested = 256;

    if (requested == 0) {
        return existing;
    }
    return std::min(requested, ntiles_kv);
}

} // namespace ggml_cuda_fattn_parallel_tune
