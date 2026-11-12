// Standalone CPU test for the diagnostic-only HIP GQA12 layer filter.
// Build/run locally, no CUDA, HIP, GPU, or GitHub Actions:
// c++ -std=c++17 -O2 -Wall -Wextra -Werror -o /tmp/test-qsa-layer-filter \
//   docs/LLAMA-PERF/tools/test_qsa_gqa12_layer_filter.cpp && /tmp/test-qsa-layer-filter
#include "../../../ggml/src/ggml-cuda/fattn-sparse-q4-gqa12-layer-filter.h"

#include <cassert>
#include <cstring>

using ggml_cuda_qsa_gqa12_layer_filter::decision;
using ggml_cuda_qsa_gqa12_layer_filter::evaluate;
using ggml_cuda_qsa_gqa12_layer_filter::parse_layer;

static void check(const char * filter, const char * name, bool valid, bool selected) {
    const decision r = evaluate(filter, name);
    assert(r.valid == valid);
    assert(r.selected == selected);
}

int main() {
    int id = -1;
    assert(parse_layer("0", id) && id == 0);
    assert(parse_layer("11", id) && id == 11);
    assert(parse_layer("9999", id) && id == 9999);
    for (const char * s : {"", "01", "-1", "11 ", " 11", "1x", "+1", "10000",
                           "99999999999999"}) {
        assert(!parse_layer(s, id));
    }
    assert(!parse_layer(nullptr, id));

    check(nullptr, "qsa_flash_attn_il11", true, true);
    check("only:11", "qsa_flash_attn_il11", true, true);
    check("only:11", "qsa_flash_attn_il15", true, false);
    check("only:11", "qsa_flash_attn_il32", true, false);
    check("except:11", "qsa_flash_attn_il11", true, false);
    check("except:11", "qsa_flash_attn_il15", true, true);
    check("except:11", "qsa_flash_attn_il32", true, true);
    check("only:0", "qsa_flash_attn_il0", true, true);
    check("only:9999", "qsa_flash_attn_il9999", true, true);

    for (const char * s : {"", "11", "only", "only:", "except:", "exclude:11",
                           "only:01", "except:-11", "only:10000", "except: 11",
                           "ONLY:11", "except:11junk", "only:11\n"}) {
        check(s, "qsa_flash_attn_il11", false, false);
    }
    for (const char * s : {"", "qsa_flash_attn_il", "qsa_flash_attn_il011",
                           "qsa_flash_attn_ilx", "qsa_flash_attn_il-1",
                           "qsa_flash_attn_il10000", "flash_attn_il11",
                           "qsa_flash_attn_il11_suffix"}) {
        check("only:11", s, false, false);
    }
    check("only:11", nullptr, false, false);
    return 0;
}
