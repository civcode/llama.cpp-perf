#include "fattn-parallel-tune.h"

#include <cassert>

int main() {
    using ggml_cuda_fattn_parallel_tune::select_parallel_blocks;

    assert(select_parallel_blocks(6, 1500, 256, 256, true, nullptr) == 6);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "") == 6);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "3") == 6);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "16") == 16);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "8") == 8);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "4") == 4);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "2") == 2);
    assert(select_parallel_blocks(6, 1500, 256, 256, true, "1") == 1);
    assert(select_parallel_blocks(6, 3, 256, 256, true, "16") == 3);
    assert(select_parallel_blocks(6, 2, 256, 256, true, "2") == 2);
    assert(select_parallel_blocks(6, 0, 256, 256, true, "4") == 6);
    assert(select_parallel_blocks(6, 1500, 256, 256, false, "2") == 6);
    assert(select_parallel_blocks(6, 1500, 256, 128, true, "2") == 6);
    assert(select_parallel_blocks(6, 1500, 128, 256, true, "2") == 6);
    return 0;
}
