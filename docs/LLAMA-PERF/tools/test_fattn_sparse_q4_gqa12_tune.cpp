#include "fattn-sparse-q4-gqa12-tune.h"

#include <cassert>

int main() {
    using ggml_cuda_fattn_sparse_q4_gqa12::should_use;
    const auto f = [&](bool rdna, bool wmma, int d, int nq, int nseq, int gqa,
                       bool gqa_opt, bool f16, bool mask, int kv, int selected, const char * env) {
        return should_use(rdna, wmma, d, d, d, nq, nseq, gqa, gqa_opt, f16, mask, kv, selected, env);
    };
    assert(f(true, true, 256, 4, 1, 12, true, true, true, 33024, 2051, "1"));
    assert(f(true, true, 256, 4, 1, 12, true, true, true, 98304, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 4, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 8, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 16, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 32768, 2051, nullptr));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 32768, 2051, "0"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 32768, 2051, "true"));
    assert(!f(false, true, 256, 4, 1, 12, true, true, true, 32768, 2051, "1"));
    assert(!f(true, false, 256, 4, 1, 12, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 128, 4, 1, 12, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 3, 1, 12, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 2, 12, true, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, false, true, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, false, true, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, false, 32768, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 8192, 2051, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 32768, 0, "1"));
    assert(!f(true, true, 256, 4, 1, 12, true, true, true, 32768, 4097, "1"));
    return 0;
}
