#pragma once

// CPU-testable grammar for a diagnostic-only, explicit HIP QSA GQA12 layer
// selector. Production sparse enablement is unchanged unless the caller opts
// in with BOTH the existing experimental sparse flag and mask audit.
//
// Filter grammar: "only:11", "except:11" (no spaces or leading zeros).
// An unexpected layer name must be an error in the guarded eligible path,
// never a silent assumption that the layer is 11.
namespace ggml_cuda_qsa_gqa12_layer_filter {

struct decision {
    bool valid;
    bool selected;
};

static inline bool parse_layer(const char * text, int & layer) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    if (text[0] == '0' && text[1] != '\0') {
        return false;
    }
    int n = 0;
    for (const char * p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        const int digit = *p - '0';
        if (n > (9999 - digit) / 10) {
            return false;
        }
        n = 10*n + digit;
    }
    layer = n;
    return true;
}

static inline decision evaluate(const char * filter, const char * tensor_name) {
    if (filter == nullptr) {
        return {true, true};
    }
    constexpr const char * only   = "only:";
    constexpr const char * except = "except:";
    const char * digits = nullptr;
    bool include_only = false;
    if (filter[0]=='o' && filter[1]=='n' && filter[2]=='l' && filter[3]=='y' && filter[4]==':') {
        digits = filter + 5;
        include_only = true;
    } else if (filter[0]=='e' && filter[1]=='x' && filter[2]=='c' &&
               filter[3]=='e' && filter[4]=='p' && filter[5]=='t' && filter[6]==':') {
        digits = filter + 7;
    } else {
        return {false, false};
    }
    // Avoid a dynamic allocation on the flash-attention hot path.
    constexpr const char prefix[] = "qsa_flash_attn_il";
    if (tensor_name == nullptr) {
        return {false, false};
    }
    for (unsigned i = 0; i < sizeof(prefix)-1; ++i) {
        if (tensor_name[i] == '\0' || tensor_name[i] != prefix[i]) {
            return {false, false};
        }
    }
    int requested = -1;
    int actual = -1;
    if (!parse_layer(digits, requested) ||
            !parse_layer(tensor_name + sizeof(prefix)-1, actual)) {
        return {false, false};
    }
    const bool matches = requested == actual;
    return {true, include_only ? matches : !matches};
}

} // namespace ggml_cuda_qsa_gqa12_layer_filter
