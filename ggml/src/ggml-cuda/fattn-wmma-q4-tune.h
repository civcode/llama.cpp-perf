#pragma once

// Opt-in: compare existing RDNA WMMA kernel vs. tile kernel for 256x256,
// 4-query GQA4 flash attention. The default selector is untouched.

#include <cstdint>
#include <cstring>

namespace ggml_cuda_fattn_wmma_q4_tune {

inline bool should_force(const bool rdna, const bool wmma_available,
                         const int64_t dkq, const int64_t dv, const int64_t dk,
                         const int64_t nq, const int64_t nseq, const int gqa,
                         const bool gqa_opt_applies, const bool f16_kv,
                         const char * env) {
    return env && std::strcmp(env, "1") == 0 && rdna && wmma_available &&
           dkq == 256 && dv == 256 && dk == 256 && nq == 4 && nseq == 1 &&
           gqa == 4 && gqa_opt_applies && f16_kv;
}

} // namespace ggml_cuda_fattn_wmma_q4_tune
