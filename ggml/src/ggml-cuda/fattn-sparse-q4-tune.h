#pragma once

// Optional RDNA sparse flash attention for 256x256, Q=4, GQA4, f16 K/V.
// Default off: sparse index bounds and model-level correctness need validation.

#include <cstdint>
#include <cstring>

namespace ggml_cuda_fattn_sparse_q4 {

inline bool should_use(const bool rdna, const bool wmma, const int64_t dkq, const int64_t dv,
                       const int64_t dk, const int64_t n_queries, const int64_t n_sequences,
                       const int gqa, const bool gqa_opt, const bool f16_kv,
                       const bool has_mask, const int64_t n_kv, const int32_t max_per_query,
                       const char * env) {
    if (!env || std::strcmp(env, "1") != 0 || !rdna || !wmma || !gqa_opt || !f16_kv || !has_mask) {
        return false;
    }
    if (dkq != 256 || dv != 256 || dk != 256 || n_queries != 4 || n_sequences != 1 || gqa != 4) {
        return false;
    }
    return max_per_query > 0 && n_kv >= 4096 && n_kv >= int64_t(8)*max_per_query;
}

} // namespace ggml_cuda_fattn_sparse_q4
