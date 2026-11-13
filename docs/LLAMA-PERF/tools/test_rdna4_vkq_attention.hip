// Direct sparse FP32 attention-kernel fixture, no environment dispatch bypass.
// Build with hipcc --offload-arch=gfx1201 -O2 -std=c++17 -DGGML_USE_HIP
// -Iggml/include -Iggml/src docs/LLAMA-PERF/tools/test_rdna4_vkq_attention.hip
// -o /tmp/test-vkq-attention; invoke once per device: /tmp/test-vkq-attention 0|1
#include "ggml-cuda/fattn-mma-f16.cuh"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(c) do { hipError_t e_=(c); if(e_ != hipSuccess) { std::fprintf(stderr,"%s: %s\n",#c,hipGetErrorString(e_)); return 1; } } while(0)

int main(int argc, char **argv) {
    if (argc != 2 || (std::strcmp(argv[1],"0") && std::strcmp(argv[1],"1"))) return 2;
    int dev=argv[1][0]-'0'; CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop{}; CHECK(hipGetDeviceProperties(&prop,dev));
    if (std::strncmp(prop.gcnArchName,"gfx1201",7)) return 2;
    constexpr int D=256, KV=64, HEAD=24, Q=4, KVHEAD=2;
    std::vector<float> q(D*Q*HEAD,0.0f), out(D*Q*HEAD);
    std::vector<half> k(D*KV*KVHEAD), v(D*KV*KVHEAD), mask(KV*Q);
    std::vector<int> indices(KV+1);
    for(int i=0;i<KV;i++) indices[i]=i;
    indices[KV]=KV;
    for(int h=0;h<KVHEAD;h++) for(int t=0;t<KV;t++) for(int f=0;f<D;f++) {
        k[(h*KV+t)*D+f]=__float2half(0.0f);
        v[(h*KV+t)*D+f]=__float2half(((f*3+t*5+h*7)%29-14)*0.03125f);
    }
    for(int x=0;x<Q;x++) for(int t=0;t<KV;t++)
        mask[x*KV+t]=__float2half((t<32 ? -3.0f : -0.25f) - ((t+x)%7)*0.0625f);
    float *dq=nullptr,*dout=nullptr; half *dk=nullptr,*dv=nullptr,*dm=nullptr; int *di=nullptr;
    CHECK(hipMalloc(&dq,q.size()*sizeof(float))); CHECK(hipMalloc(&dout,out.size()*sizeof(float)));
    CHECK(hipMalloc(&dk,k.size()*sizeof(half))); CHECK(hipMalloc(&dv,v.size()*sizeof(half)));
    CHECK(hipMalloc(&dm,mask.size()*sizeof(half))); CHECK(hipMalloc(&di,indices.size()*sizeof(int)));
    CHECK(hipMemcpy(dq,q.data(),q.size()*sizeof(float),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dk,k.data(),k.size()*sizeof(half),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dv,v.data(),v.size()*sizeof(half),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dm,mask.data(),mask.size()*sizeof(half),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(di,indices.data(),indices.size()*sizeof(int),hipMemcpyHostToDevice));
    auto kernel=flash_attn_ext_f16<256,256,4,4,false,false,true,true>;
    const uint3 ne01=init_fastdiv_values(Q);
    // Six whole output tiles (one per 4 GQA heads and KV head), two wave32 warps.
    kernel<<<dim3(6),dim3(32,2),32768>>>(
        (char*)dq,(char*)dk,(char*)dv,(char*)dm,nullptr,di,nullptr,dout,nullptr,
        1.0f,0.0f,1.0f,1.0f,16,0.0f,
        D,ne01,HEAD,1,D*4,D*Q*4,int64_t(D)*Q*HEAD*4,
        D,KV,KVHEAD,1,D*2,D*KV*2,int64_t(D)*KV*KVHEAD*2,
        D*2,D*KV*2,int64_t(D)*KV*KVHEAD*2,
        Q,1,1,KV*2,KV*Q*2,int64_t(KV)*Q*2);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(),dout,out.size()*sizeof(float),hipMemcpyDeviceToHost));
    int bad=0; double maxerr=0;
    for(int x=0;x<Q;x++) for(int h=0;h<HEAD;h++) for(int f=0;f<D;f++) {
        double sum=0, den=0;
        for(int t=0;t<KV;t++) { double w=std::exp(double(__half2float(mask[x*KV+t]))); den+=w; sum+=w*__half2float(v[((h/12)*KV+t)*D+f]); }
        double ref=sum/den, val=out[(x*HEAD+h)*D+f], err=std::abs(val-ref);
        if(!std::isfinite(val) || err>0.001) { if(bad<8) std::fprintf(stderr,"q=%d h=%d f=%d got=%g ref=%g\n",x,h,f,val,ref); ++bad; }
        if(err>maxerr) maxerr=err;
    }
    CHECK(hipFree(dq)); CHECK(hipFree(dout)); CHECK(hipFree(dk)); CHECK(hipFree(dv)); CHECK(hipFree(dm)); CHECK(hipFree(di));
    std::printf("full sparse FP32 attention fixture: %d mismatches max_abs=%.9g\n",bad,maxerr);
    return bad ? 1 : 0;
}
