#include "upload-profile.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>

int main() {
    using namespace ggml_cuda_upload_profile;
    collector c;
    c.add("set_tensor", 0, "first", 128, 3000, 2000);
    c.add("set_tensor", 0, "first", 256, 6000, 4000);
    c.add("set_tensor", 1, "other", 1024, 5000, 1000);
    FILE * tmp = std::tmpfile();
    assert(tmp);
    c.print(tmp);
    std::rewind(tmp);
    std::string output;
    char buf[2048];
    while (std::fgets(buf, sizeof buf, tmp)) output += buf;
    std::fclose(tmp);
    assert(output.find("GGML_CUDA_UPLOAD_PROFILE\tset_tensor\t0\tfirst\t2\t384\t9\t6\t256\t4") != std::string::npos);
    assert(output.find("GGML_CUDA_UPLOAD_PROFILE\tset_tensor\t1\tother\t1\t1024\t5\t1\t1024\t1") != std::string::npos);
    return 0;
}
