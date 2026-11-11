#include "fattn-gqa-tune.h"

#include <cassert>
#include <climits>

int main() {
    using ggml_cuda_fattn_tune::max_gqa_columns;
    assert(max_gqa_columns(256, 256, true, nullptr) == INT_MAX);
    assert(max_gqa_columns(256, 256, true, "0") == INT_MAX);
    assert(max_gqa_columns(256, 256, true, "4") == INT_MAX);
    assert(max_gqa_columns(256, 256, true, "2") == 2);
    assert(max_gqa_columns(256, 256, false, "2") == INT_MAX);
    assert(max_gqa_columns(128, 128, true, "2") == INT_MAX);
    assert(max_gqa_columns(256, 128, true, "2") == INT_MAX);
    return 0;
}
