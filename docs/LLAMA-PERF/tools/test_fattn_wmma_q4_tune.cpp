#include "fattn-wmma-q4-tune.h"
#include <cassert>

int main() {
    using ggml_cuda_fattn_wmma_q4_tune::should_force;
    const auto f = [&](bool rdna, bool wmma, int dkq, int dv, int dk,
                       int nq, int nseq, int gqa, bool gqa_opt, bool f16,
                       const char * env) {
        return should_force(rdna, wmma, dkq, dv, dk, nq, nseq, gqa, gqa_opt, f16, env);
    };
    assert(f(true, true, 256, 256, 256, 4, 1, 4, true, true, "1"));
    assert(!f(true, true, 256, 256, 256, 4, 1, 4, true, true, nullptr));
    assert(!f(true, true, 256, 256, 256, 4, 1, 4, true, true, "0"));
    assert(!f(true, true, 256, 256, 256, 4, 1, 4, true, true, "2"));
    assert(!f(false, true, 256, 256, 256, 4, 1, 4, true, true, "1"));
    assert(!f(true, false, 256, 256, 256, 4, 1, 4, true, true, "1"));
    assert(!f(true, true, 128, 256, 256, 4, 1, 4, true, true, "1"));
    assert(!f(true, true, 256, 128, 256, 4, 1, 4, true, true, "1"));
    assert(!f(true, true, 256, 256, 128, 4, 1, 4, true, true, "1"));
    assert(!f(true, true, 256, 256, 256, 3, 1, 4, true, true, "1"));
    assert(!f(true, true, 256, 256, 256, 4, 2, 4, true, true, "1"));
    assert(!f(true, true, 256, 256, 256, 4, 1, 8, true, true, "1"));
    assert(!f(true, true, 256, 256, 256, 4, 1, 4, false, true, "1"));
    assert(!f(true, true, 256, 256, 256, 4, 1, 4, true, false, "1"));
    return 0;
}
