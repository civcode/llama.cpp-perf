// CPU-only contract tests for strict diagnostic FP32 VKQ accumulation request.
// Run in the local repository with:
// c++ -std=c++17 -O2 -Wall -Wextra -Werror docs/LLAMA-PERF/tools/test_qsa_gqa12_vkq_f32.cpp -o /tmp/test-qsa-vkq-f32
// /tmp/test-qsa-vkq-f32
//
// This tests request validation and shape guards, NOT GPU arithmetic.
#include "../../../ggml/src/ggml-cuda/fattn-sparse-q4-gqa12-vkq-f32.h"

#include <cassert>
#include <climits>
#include <initializer_list>

using namespace ggml_cuda_qsa_gqa12_vkq_f32;

int main() {
    // Backwards-compatible: absent opt-in must not enable a new kernel.
    assert(parse(nullptr, nullptr, nullptr, nullptr).valid);
    assert(!parse(nullptr, "1", "100", "only:11").enabled);
    assert(!parse(nullptr, "0", "100", "only:11").enabled);
    assert(!parse(nullptr, "1", nullptr, nullptr).enabled);

    const request enabled = parse("1", "1", "100", "only:11");
    assert(enabled.valid && enabled.enabled);

    for (const char * invalid : {"", "0", "true", "01", "2", " 1", "1 ",
                                 "-1", "fp32", "11"}) {
        const request r = parse(invalid, "1", "100", "only:11");
        assert(!r.valid && !r.enabled);
    }
    for (const char * wrong_sparse : {"0", "", "true", "01"}) {
        const request r = parse("1", wrong_sparse, "100", "only:11");
        assert(!r.valid && !r.enabled);
    }
    for (const char * wrong_filter : {"only:12", "except:11", "only:011",
                                      "", "all", "only:11 "}) {
        const request r = parse("1", "1", "100", wrong_filter);
        assert(!r.valid && !r.enabled);
    }
    assert(!parse("1", nullptr, "100", "only:11").valid);
    assert(!parse("1", "1", nullptr, "only:11").valid);
    assert(!parse("1", "1", "100", nullptr).valid);

    // Only the existing exactly eligible audited layer-11 sparse Q4/GQA12
    // operation can be evaluated in this first precision pilot.
    const auto eligible_default = [](const int dkq, const int dv,
                                         const int dk, const int nq,
                                         const int nseq, const int gqa,
                                         const int layer, const int nkv,
                                         const int maximum) -> bool {
        return eligible(true, true, true, true, dkq, dv, dk, nq,
                        nseq, gqa, layer, nkv, maximum);
    };
    assert(eligible_default(256,256,256,4,1,12,11,33024,3537));
    assert(!eligible_default(256,256,256,4,1,12,15,33024,3537));
    assert(!eligible_default(256,256,256,4,1,12,12,33024,3537));
    assert(!eligible_default(256,256,256,4,1,4,11,33024,3537));
    assert(!eligible_default(256,256,256,1,1,12,11,33024,3537));
    assert(!eligible_default(256,256,256,4,2,12,11,33024,3537));
    assert(!eligible_default(128,256,256,4,1,12,11,33024,3537));
    assert(!eligible_default(256,128,256,4,1,12,11,33024,3537));
    assert(!eligible_default(256,256,128,4,1,12,11,33024,3537));
    assert(!eligible_default(256,256,256,4,1,12,11,33024,0));
    assert(!eligible_default(256,256,256,4,1,12,11,4096,1024));
    assert(!eligible_default(256,256,256,4,1,12,11,2048,100));
    assert(!eligible_default(256,256,256,4,1,12,11,33024,INT_MAX));
    assert(!eligible(false, true, true, true, 256,256,256,4,1,12,11,33024,3537));
    assert(!eligible(true, false, true, true, 256,256,256,4,1,12,11,33024,3537));
    assert(!eligible(true, true, false, true, 256,256,256,4,1,12,11,33024,3537));
    assert(!eligible(true, true, true, false, 256,256,256,4,1,12,11,33024,3537));
    return 0;
}
