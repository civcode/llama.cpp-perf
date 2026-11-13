// Bounded gfx12 WMMA fragment-layout fixture; not an attention kernel test.
// hipcc --offload-arch=gfx1201 -O2 test_rdna4_vkq_wmma.hip -o /tmp/test-rdna4-vkq
// /tmp/test-rdna4-vkq <HIP device index>
// One wave computes 32x16 via two independent FP32 16x16 accumulator
// fragments. Inputs and output use the layout documented by AMD's RDNA4
// dense WMMA compute-policy example. Verify FP32 rescaling and second MAC.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

using h8 = __fp16 __attribute__((ext_vector_type(8)));
using f8 = float __attribute__((ext_vector_type(8)));

__global__ void fixture(float * out) {
    (void) out;
#if defined(__gfx1201__)
    const int lane = threadIdx.x;
    const int col = lane % 16;
    const int kbase = 8 * (lane / 16);
    h8 a[2], b;
    for (int k = 0; k < 8; ++k) {
        const int kk = kbase + k;
        b[k] = (__fp16) (((col * 5 + kk * 7) % 11 - 5) * 0.0625f);
        for (int p = 0; p < 2; ++p) {
            a[p][k] = (__fp16) ((((p * 16 + col) * 3 + kk * 5) % 13 - 6) * 0.125f);
        }
    }
    for (int p = 0; p < 2; ++p) {
        f8 acc = {};
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[p], b, acc);
        acc *= 0.5f;
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[p], b, acc);
        for (int e = 0; e < 8; ++e) {
            const int row = 8 * (lane / 16) + e;
            // WMMA's A row is the output column in this transposed storage.
            out[(p * 16 + row) * 16 + col] = acc[e];
        }
    }
#endif
}

#define HIP_CHECK(call) do { hipError_t e = (call); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e)); return 1; } } while (0)

int main(int argc, char ** argv) {
    if (argc != 2 || (argv[1][0] != '0' && argv[1][0] != '1') || argv[1][1]) return 2;
    HIP_CHECK(hipSetDevice(argv[1][0] - '0'));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, argv[1][0] - '0'));
    if (std::strncmp(prop.gcnArchName, "gfx1201", 7) != 0) return 2;
    float * device = nullptr;
    HIP_CHECK(hipMalloc(&device, 32 * 16 * sizeof(float)));
    hipLaunchKernelGGL(fixture, dim3(1), dim3(32), 0, 0, device);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    float out[32 * 16];
    HIP_CHECK(hipMemcpy(out, device, sizeof(out), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(device));
    int bad = 0;
    for (int row = 0; row < 32; ++row) for (int col = 0; col < 16; ++col) {
        float expected = 0;
        for (int k = 0; k < 16; ++k) {
            const float a = ((row * 3 + k * 5) % 13 - 6) * 0.125f;
            const float b = ((col * 5 + k * 7) % 11 - 5) * 0.0625f;
            expected += a * b;
        }
        expected *= 1.5f;
        if (out[row * 16 + col] != expected) {
            if (bad < 8) std::fprintf(stderr, "[%d,%d] got %.9g expected %.9g\n", row, col, out[row * 16 + col], expected);
            ++bad;
        }
    }
    std::printf("gfx1201 WMMA FP32 two-fragment fixture: %d mismatches\n", bad);
    return bad ? 1 : 0;
}
