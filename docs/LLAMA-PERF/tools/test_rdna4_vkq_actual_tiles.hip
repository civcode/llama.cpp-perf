// Test the exact gfx12 VKQ transposed LDS loader and two-F32-fragment helper.
// hipcc -std=c++17 -O2 --offload-arch=gfx1201 -DGGML_USE_HIP -Iggml/include -Iggml/src docs/LLAMA-PERF/tools/test_rdna4_vkq_actual_tiles.hip -o /tmp/test-vkq-tiles
// /tmp/test-vkq-tiles 0   (or 1; one block per invocation)
#include "ggml-cuda/mma.cuh"
#include <cstdio>
#include <cstring>
using namespace ggml_cuda_mma;
#define CHECK(c) do { hipError_t e_ = (c); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #c, hipGetErrorString(e_)); return 1; } } while (0)

__global__ void test_tiles(float * output) {
#if defined(AMD_WMMA_128B_AVAILABLE)
    constexpr int stride = 16;
    __shared__ half2 v[16*stride];
    // V[k][feature], using exactly the kernel's half2 and swizzled LDS layout.
    for (int t = threadIdx.x; t < 16*16; t += 32) {
        int k = t / 16, pair = t % 16;
        float x = (((2*pair)*3 + k*5) % 13 - 6) * 0.125f;
        float y = (((2*pair+1)*3 + k*5) % 13 - 6) * 0.125f;
        v[swizzle<stride,half2>(t,k)] = __floats2half2_rn(x,y);
    }
    __syncthreads();
    tile<32,8,half2,DATA_LAYOUT_I_MAJOR> a;
    tile<16,8,half2,DATA_LAYOUT_I_MAJOR> b;
    tile<16,16,float,DATA_LAYOUT_I_MAJOR> c[2];
    load_ldmatrix_trans_swizzled<stride>(a,v,0);
    for (int l = 0; l < b.ne; ++l) {
        int q = b.get_i(l), k = 2*b.get_j(l);
        float x = ((q*5 + k*7) % 11 - 5) * 0.0625f;
        float y = ((q*5 + (k+1)*7) % 11 - 5) * 0.0625f;
        b.x[l] = __floats2half2_rn(x,y);
    }
    mma_vkq_gfx12_fp32(c[0],c[1],a,b);
    for (int p = 0; p < 2; ++p) for (int l = 0; l < c[p].ne; ++l) c[p].x[l] *= 0.5f;
    mma_vkq_gfx12_fp32(c[0],c[1],a,b);
    __shared__ half combined[16*32];
    for (int p = 0; p < 2; ++p) for (int l = 0; l < c[p].ne; ++l) {
        int q = c[p].get_i(l), feature = 2*c[p].get_j(l)+p;
        combined[q*32+feature] = __float2half_rn(c[p].x[l]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 512; i += 32) output[i] = __half2float(combined[i]);
#else
    (void) output;
#endif
}
int main(int argc,char **argv) {
    if (argc != 2 || (std::strcmp(argv[1],"0") && std::strcmp(argv[1],"1"))) return 2;
    int id = argv[1][0]-'0'; CHECK(hipSetDevice(id));
    hipDeviceProp_t prop{}; CHECK(hipGetDeviceProperties(&prop,id));
    if (std::strncmp(prop.gcnArchName,"gfx1201",7)) return 2;
    float * d; CHECK(hipMalloc(&d,512*sizeof(float)));
    hipLaunchKernelGGL(test_tiles,dim3(1),dim3(32),0,0,d);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    float got[512]; CHECK(hipMemcpy(got,d,sizeof(got),hipMemcpyDeviceToHost)); CHECK(hipFree(d));
    int bad=0;
    for (int q=0;q<16;++q) for (int f=0;f<32;++f) {
        float ref=0; for(int k=0;k<16;++k) ref += ((f*3+k*5)%13-6)*0.125f*((q*5+k*7)%11-5)*0.0625f;
        ref *= 1.5f;
        if (got[q*32+f] != ref) { if(bad<8) std::fprintf(stderr,"q=%d f=%d got=%g expected=%g\n",q,f,got[q*32+f],ref); ++bad; }
    }
    std::printf("actual LDS VKQ loader and two FP32 fragments: %d mismatches\n",bad);
    return bad != 0;
}
