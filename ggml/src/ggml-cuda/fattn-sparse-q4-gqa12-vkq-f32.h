#pragma once

// Strict CPU-testable configuration gate for a diagnostic-only QSA GQA12
// experiment: FP32 accumulation of attention-weighted values (VKQ).
//
// This header parses the REQUEST ONLY. It does NOT implement the GPU math.
// The local HIP implementer must separately prove the expected kernel variant
// compiled and actually launched; never treat a valid opt-in as evidence that
// F32 arithmetic ran. Unset option leaves all existing behavior unchanged.
//
// The first experiment is deliberately restricted to only layer 11, with
// sparse GQA12 explicitly enabled and the mask audit disabling GPU graphs.
// Values other than exactly "1" fail closed; no silent fallback on typo.
#include <cstring>

namespace ggml_cuda_qsa_gqa12_vkq_f32 {

struct request {
    bool valid;
    bool enabled;
};

static inline request parse(
        const char * const fp32_env,
        const char * const sparse_gqa12_env,
        const char * const mask_audit_env,
        const char * const layer_filter_env) {
    if (fp32_env == nullptr) {
        return {true, false};
    }
    if (std::strcmp(fp32_env, "1") != 0 ||
            sparse_gqa12_env == nullptr ||
            std::strcmp(sparse_gqa12_env, "1") != 0 ||
            mask_audit_env == nullptr ||
            layer_filter_env == nullptr ||
            std::strcmp(layer_filter_env, "only:11") != 0) {
        return {false, false};
    }
    return {true, true};
}

// Exact narrow eligible operation for the first pilot. This is a minimum
// restriction in addition to the project's existing GQA12 sparse should_use
// predicate and actual layer label; it does NOT replace those guards.
static inline bool eligible(
        bool rdna, bool wmma, bool f16_kv, bool f16_mask,
        int dkq, int dv, int dk, int n_queries, int n_sequences,
        int gqa_ratio, int layer, int n_kv, int max_per_query) {
    return rdna && wmma && f16_kv && f16_mask &&
           dkq == 256 && dv == 256 && dk == 256 &&
           n_queries == 4 && n_sequences == 1 &&
           gqa_ratio == 12 && layer == 11 &&
           max_per_query > 0 && n_kv >= 4096 &&
           n_kv >= 8 * static_cast<int64_t>(max_per_query);
}

} // namespace ggml_cuda_qsa_gqa12_vkq_f32
