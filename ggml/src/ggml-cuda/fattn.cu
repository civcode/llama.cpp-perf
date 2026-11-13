#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "fattn-wmma-q4-tune.h"
#include "fattn-sparse-q4-tune.h"
#include "fattn-sparse-q4-gqa12-tune.h"
#include "fattn-sparse-q4-gqa12-layer-filter.h"
#include "fattn-sparse-q4-gqa12-vkq-f32.h"

#include <cstdlib>
#include <atomic>
#include <chrono>

#if defined(GGML_USE_HIP)
// Diagnostic-only layer isolation. The default is the original experimental
// GQA12 selector (no layer restriction); this does not enable sparse by itself.
// Stable layer names require the opt-in graph-disabling QSA mask audit. The
// caller uses this identical guard in both supported() and launch selection.
static const char * ggml_cuda_qsa_sparse_gqa12_layer_option(const ggml_tensor * dst) {
    const char * sparse_enabled = std::getenv("GGML_HIP_QSA_SPARSE_Q4_GQA12");
    const char * filter = std::getenv("GGML_HIP_QSA_GQA12_LAYER_FILTER");

    // Explicit fail-closed scaffold: the HIP WMMA accumulator variant is NOT
    // implemented yet. A user opting in must never silently run the existing
    // half2 path and mistake it for an FP32 VKQ precision experiment.
    // The local GPU implementer removes only the unimplemented abort AFTER
    // wiring a distinct compiled FP32 VKQ path and verifying actual launches.
    const auto vkq_f32 = ggml_cuda_qsa_gqa12_vkq_f32::parse(
        std::getenv("GGML_HIP_QSA_GQA12_VKQ_F32"), sparse_enabled,
        std::getenv("GGML_QSA_MASK_AUDIT"), filter);
    if (!vkq_f32.valid) {
        GGML_ABORT("GGML_HIP_QSA_GQA12_VKQ_F32 requires exactly value 1, GQA12=1, GGML_QSA_MASK_AUDIT and layer filter only:11");
    }
    if (vkq_f32.enabled) {
        GGML_ABORT("GGML_HIP_QSA_GQA12_VKQ_F32 GPU implementation not built; refuse to silently run half2 VKQ");
    }
    if (filter == nullptr) {
        return sparse_enabled;
    }
    if (sparse_enabled == nullptr || strcmp(sparse_enabled, "1") != 0 ||
            std::getenv("GGML_QSA_MASK_AUDIT") == nullptr) {
        GGML_ABORT("GGML_HIP_QSA_GQA12_LAYER_FILTER requires experimental GQA12=1 and GGML_QSA_MASK_AUDIT");
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    // Ignore other attention shapes, which are ineligible for sparse Q4 in
    // the existing selector and may not have stable QSA layer names.
    if (q == nullptr || k == nullptr || k->ne[2] <= 0 ||
            q->ne[1] != 4 || q->ne[2] / k->ne[2] != 12 ||
            ggml_get_op_params_i32(dst, 4) <= 0) {
        return nullptr;
    }
    const auto decision = ggml_cuda_qsa_gqa12_layer_filter::evaluate(filter, dst->name);
    if (!decision.valid) {
        GGML_ABORT("invalid GGML_HIP_QSA_GQA12_LAYER_FILTER or missing audited layer name: %s at %s",
                   filter, dst->name);
    }
    return decision.selected ? sparse_enabled : nullptr;
}
#endif // GGML_USE_HIP

#if !defined(GGML_USE_MUSA)

// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33, const bool report_overflow) {
    ggml_cuda_pdl_sync();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int values_per_lane = 8;
    using lane_mask_t = std::conditional_t<warp_size == 64, uint64_t, uint32_t>;
    constexpr uint64_t ballot_mask = ~(lane_mask_t)0;
    const auto popc = [](const lane_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        lane_mask_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = (lane_mask_t)__ballot_sync(ballot_mask, (int) selected);
            warp_count += popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const lane_mask_t lane_mask = lane == 0 ? 0 : ((lane_mask_t)1 << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & ((lane_mask_t)1 << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = report_overflow ? row_count : count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream, bool report_overflow) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream, report_overflow);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // A Q=4 verification batch can share a sparse index list across its queries.
    GGML_ASSERT(ncols1 == 1 || ncols1 == 4 || ncols1 == 8);
    const auto kernel = ncols1 == 1 ? flash_attn_mask_to_sparse_indices<1, false> :
                        ncols1 == 4 ? (n_queries % 4 != 0 ? flash_attn_mask_to_sparse_indices<4, true> :
                                                               flash_attn_mask_to_sparse_indices<4, false>) :
                                      (n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true> :
                                                               flash_attn_mask_to_sparse_indices<8, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33, report_overflow);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // On CDNA only the DKQ=576 kernel variant with ncols2=16 is compiled with sparse support.
    if (amd_mfma_available(cc) && !ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(int(Q->ne[0]), int(V->ne[0]), 1, 16)) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    bool sparse_q4 = false;
#if defined(GGML_USE_HIP)
    if (ncols1 == 4 && ncols2 == 4 && mask != nullptr && K->ne[2] > 0 && Q->ne[2] % K->ne[2] == 0) {
        bool gqa_opt = max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
        for (const ggml_tensor * t : {Q, K, V, mask}) {
            if (ggml_is_quantized(t->type)) {
                gqa_opt = false;
            }
            for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
                if (t->nb[i] % 16 != 0) {
                    gqa_opt = false;
                }
            }
        }
        // Keep the proven GQA4 guard intact. The GQA12 prototype has a
        // separate opt-in flag and reuses the same 4x4 sparse specialization
        // across three GQA-head tiles; GPU correctness is still to be tested.
        sparse_q4 = ggml_cuda_fattn_sparse_q4::should_use(
            GGML_CUDA_CC_IS_RDNA(cc), amd_wmma_available(cc), Q->ne[0], V->ne[0], K->ne[0],
            Q->ne[1], Q->ne[3], Q->ne[2]/K->ne[2], gqa_opt,
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16,
            mask->type == GGML_TYPE_F16, K->ne[1], n_kv_max,
            std::getenv("GGML_HIP_QSA_SPARSE_Q4")) ||
            ggml_cuda_fattn_sparse_q4_gqa12::should_use(
            GGML_CUDA_CC_IS_RDNA(cc), amd_wmma_available(cc), Q->ne[0], V->ne[0], K->ne[0],
            Q->ne[1], Q->ne[3], Q->ne[2]/K->ne[2], gqa_opt,
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16,
            mask->type == GGML_TYPE_F16, K->ne[1], n_kv_max,
            ggml_cuda_qsa_sparse_gqa12_layer_option(dst));
    }
#endif
    if (ncols1 == 4 && ncols2 == 4 && !sparse_q4) {
        return false;
    }

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return ((GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc)) || amd_mfma_available(cc) || sparse_q4) && dst->src[5] == nullptr &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, GGML_CUDA_CC_IS_NVIDIA(cc) ? 2*n_gather : 2LL*n_kv_max);
#endif // !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = dst->src[5] ? dst->src[5]->ne[0] : K->ne[1];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 40:
            GGML_ASSERT(V->ne[0] == 40);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 40,  40>(ctx, dst);
            break;
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 72:
            GGML_ASSERT(V->ne[0] == 72);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 72,  72>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, BF16)

    FATTN_VEC_CASES_ALL_D(F16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q8_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(BF16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F8_E4M3)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F8_E4M3:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * kv_rows = dst->src[5];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = kv_rows ? kv_rows->ne[0] : K->ne[1];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // only the MMA kernel reads K/V through kv_rows
    if (kv_rows) {
        return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask && Q->ne[0] != 40 && Q->ne[0] != 72 ?
            BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
    }

#if defined(GGML_USE_HIP)
    // Opt-in sparse gather reuses the existing 256x256 WMMA kernel.
    // Only the QSA n_kv_max shape can activate it.
    const int32_t n_kv_max_sparse = ggml_get_op_params_i32(dst, 4);
    const bool sparse_q4_gqa4 = ggml_cuda_fattn_sparse_q4::should_use(
            GGML_CUDA_CC_IS_RDNA(cc), amd_wmma_available(cc),
            Q->ne[0], V->ne[0], K->ne[0], Q->ne[1], Q->ne[3], gqa_ratio,
            gqa_opt_applies,
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16,
            mask != nullptr && mask->type == GGML_TYPE_F16, K->ne[1],
            n_kv_max_sparse, std::getenv("GGML_HIP_QSA_SPARSE_Q4"));
    const bool sparse_q4_gqa12 = ggml_cuda_fattn_sparse_q4_gqa12::should_use(
            GGML_CUDA_CC_IS_RDNA(cc), amd_wmma_available(cc),
            Q->ne[0], V->ne[0], K->ne[0], Q->ne[1], Q->ne[3], gqa_ratio,
            gqa_opt_applies,
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16,
            mask != nullptr && mask->type == GGML_TYPE_F16, K->ne[1],
            n_kv_max_sparse, ggml_cuda_qsa_sparse_gqa12_layer_option(dst));
    if (sparse_q4_gqa4 || sparse_q4_gqa12) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }
#endif

#if defined(GGML_USE_HIP)
    // Opt-in comparison for the 4-query RDNA GQA4 verification boundary.
    // Default kernel selection is unchanged.
    static const char * const wmma_q4_env = std::getenv("GGML_HIP_FATTN_FORCE_WMMA_Q4");
    if (ggml_cuda_fattn_wmma_q4_tune::should_force(
            GGML_CUDA_CC_IS_RDNA(cc), amd_wmma_available(cc),
            Q->ne[0], V->ne[0], K->ne[0], Q->ne[1], Q->ne[3], gqa_ratio,
            gqa_opt_applies,
            K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16,
            wmma_q4_env)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }
#endif

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc)) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // On CDNA, the sparse MMA kernel is used for DKQ=576 regardless of the head size heuristics.
    if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

// Opt-in, bounded, synchronous audit of the final attention-consumer mask.
static long ggml_cuda_qsa_audit_mask(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const char * option = std::getenv("GGML_QSA_MASK_AUDIT");
    if (!option) return -1;
    const long limit = std::strtol(option, nullptr, 10);
    if (limit <= 0 || limit > 100000) GGML_ABORT("GGML_QSA_MASK_AUDIT must be 1..100000");
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * mask = dst->src[3];
    const int bound = ggml_get_op_params_i32(dst, 4);
    if (!mask || bound <= 0 || q->ne[1] != 4) return -1;
    static std::atomic<unsigned long> invocation{0};
    const auto id = invocation.fetch_add(1);
    if (id >= (unsigned long) limit) {
        if (id == (unsigned long) limit) fprintf(stderr, "QSA_MASK_AUDIT limit reached; subsequent masks not inspected\n");
        return -1;
    }
    if (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(ggml_fp16_t) || mask->ne[0] > 262144 ||
        mask->ne[1] < 4 || mask->ne[2] != 1 || mask->ne[3] > 8 || mask->ne[3] != q->ne[3]) {
        GGML_ABORT("QSA mask audit: unsupported mask layout");
    }
    const size_t kv = mask->ne[0];
    const int64_t capacity = std::min<int64_t>(kv, int64_t(4)*bound);
    std::vector<ggml_fp16_t> data(kv*4);
    cudaStream_t stream = ctx.stream();
    for (int64_t s = 0; s < mask->ne[3]; ++s) {
        const char * src = (const char *) mask->data + s*mask->nb[3];
        CUDA_CHECK(cudaMemcpy2DAsync(data.data(), kv*sizeof(ggml_fp16_t), src, mask->nb[1],
                                     kv*sizeof(ggml_fp16_t), 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int counts[4] = {0, 0, 0, 0};
        int nans = 0, posinf = 0, neginf = 0, total = 0;
        for (size_t i = 0; i < kv; ++i) {
            bool any = false;
            for (int j = 0; j < 4; ++j) {
                const uint16_t x = data[j*kv + i];
                if ((x & 0x7c00u) != 0x7c00u) { ++counts[j]; any = true; }
                else if (x & 0x03ffu) ++nans;
                else if (x & 0x8000u) ++neginf;
                else ++posinf;
            }
            total += any;
        }
#ifdef GGML_USE_HIP
        const char * backend = "HIP";
#else
        const char * backend = "CUDA";
#endif
        // Optional, bounded raw contiguous [4, KV] F16 samples per backend/device.
        // Keep the original first-sample filename when DUMP_LIMIT is not set.
        // This is a synchronous diagnostic, never for performance measurements.
        const char * dump_dir = std::getenv("GGML_QSA_MASK_AUDIT_DIR");
        const char * dump_limit_option = std::getenv("GGML_QSA_MASK_DUMP_LIMIT");
        long dump_limit = 1;
        if (dump_limit_option) {
            char * end = nullptr;
            dump_limit = std::strtol(dump_limit_option, &end, 10);
            if (!end || *end != '\0' || dump_limit < 1 || dump_limit > 256 || !dump_dir) {
                GGML_ABORT("GGML_QSA_MASK_DUMP_LIMIT requires GGML_QSA_MASK_AUDIT_DIR and a value 1..256");
            }
        }
        static std::atomic<unsigned long> dumps[32]{};
        if (dump_dir && ctx.device >= 0 && ctx.device < 32) {
            const unsigned long dump_id = dumps[ctx.device].fetch_add(1);
            if (dump_id < (unsigned long) dump_limit) {
                char path[4096];
                const int n = dump_limit_option
                    ? snprintf(path, sizeof(path), "%s/%s%d-id%lu-s%lld-kv%zu.f16", dump_dir, backend,
                               ctx.device, id, (long long) s, kv)
                    : snprintf(path, sizeof(path), "%s/%s%d-mask.f16", dump_dir, backend, ctx.device);
                if (n < 0 || size_t(n) >= sizeof(path)) GGML_ABORT("QSA mask audit dump path too long");
                FILE * file = fopen(path, "wb");
                if (!file) GGML_ABORT("QSA mask audit cannot open dump %s", path);
                const size_t written = fwrite(data.data(), sizeof(data[0]), data.size(), file);
                const int close_result = fclose(file);
                if (written != data.size() || close_result != 0) GGML_ABORT("QSA mask audit cannot write dump %s", path);
                fprintf(stderr, "QSA_MASK_DUMP backend=%s device=%d id=%lu stream=%lld kv=%zu path=%s\n",
                        backend, ctx.device, id, (long long) s, kv, path);
            } else if (dump_id == (unsigned long) dump_limit) {
                fprintf(stderr, "QSA_MASK_DUMP limit=%ld reached backend=%s device=%d; subsequent masks not dumped\n",
                        dump_limit, backend, ctx.device);
            }
        }
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        fprintf(stderr, "QSA_MASK_AUDIT mono_ns=%lld backend=%s id=%lu device=%d tensor=%s kv=%zu q=4 qheads=%lld kvheads=%lld stream=%lld bound=%d counts=%d,%d,%d,%d union=%d capacity=%lld nan=%d posinf=%d neginf=%d\n",
                      ns, backend, id, ctx.device, dst->name, kv, (long long) q->ne[2], (long long) dst->src[1]->ne[2],
                      (long long) s, bound, counts[0], counts[1], counts[2], counts[3], total,
                      (long long) capacity, nans, posinf, neginf);
        if (nans || posinf || total > capacity || counts[0] > bound || counts[1] > bound || counts[2] > bound || counts[3] > bound) {
            GGML_ABORT("QSA mask audit detected an invalid or over-bound mask");
        }
    }
    return static_cast<long>(id);
}

// Opt-in synchronous F32 output dump for the same QSA node as the mask audit.
// Pair this with its mask dump by the shared invocation id, backend and device.
// Do not use this diagnostic for benchmarks: it synchronizes after the kernel.
static void ggml_cuda_qsa_dump_attn_output(
        ggml_backend_cuda_context & ctx, const ggml_tensor * dst, long audit_id) {
    const char * dir = std::getenv("GGML_QSA_ATTN_OUT_DIR");
    if (!dir) return;
    if (!std::getenv("GGML_QSA_MASK_AUDIT")) {
        GGML_ABORT("GGML_QSA_ATTN_OUT_DIR requires GGML_QSA_MASK_AUDIT");
    }
    // Other attention operators are not QSA. Do not abort when the model
    // contains a mixture of regular and QSA attention layers.
    if (audit_id < 0) return;
    const char * limit_option = std::getenv("GGML_QSA_ATTN_OUT_LIMIT");
    long limit = 1;
    if (limit_option) {
        char * end = nullptr;
        limit = std::strtol(limit_option, &end, 10);
        if (!end || *end != '\0' || limit < 1 || limit > 256) {
            GGML_ABORT("GGML_QSA_ATTN_OUT_LIMIT must be 1..256");
        }
    }
    if (ctx.device < 0 || ctx.device >= 32) GGML_ABORT("QSA output audit: invalid device");
    static std::atomic<unsigned long> dumped[32]{};
    const unsigned long ordinal = dumped[ctx.device].fetch_add(1);
    if (ordinal >= (unsigned long) limit) {
        if (ordinal == (unsigned long) limit) {
            fprintf(stderr, "QSA_ATTN_OUT limit=%ld reached device=%d\n", limit, ctx.device);
        }
        return;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        ggml_nelements(dst) <= 0 || ggml_nelements(dst) > 1 << 20) {
        GGML_ABORT("QSA output audit: unsupported output layout or size");
    }
    const size_t n = ggml_nelements(dst);
    std::vector<float> values(n);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemcpyAsync(values.data(), dst->data, n*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#ifdef GGML_USE_HIP
    const char * backend = "HIP";
#else
    const char * backend = "CUDA";
#endif
    char path[4096];
    const int used = snprintf(path, sizeof(path), "%s/%s%d-id%ld-qsa-out.f32",
                              dir, backend, ctx.device, audit_id);
    if (used < 0 || size_t(used) >= sizeof(path)) {
        GGML_ABORT("QSA output audit: path too long");
    }
    FILE * file = fopen(path, "wb");
    if (!file) GGML_ABORT("QSA output audit: cannot open %s", path);
    const size_t written = fwrite(values.data(), sizeof(float), values.size(), file);
    const int closed = fclose(file);
    if (written != values.size() || closed != 0) {
        GGML_ABORT("QSA output audit: cannot write %s", path);
    }
    fprintf(stderr,
        "QSA_ATTN_OUT backend=%s device=%d id=%ld tensor=%s ne=%lld,%lld,%lld,%lld path=%s\n",
        backend, ctx.device, audit_id, dst->name,
        (long long) dst->ne[0], (long long) dst->ne[1],
        (long long) dst->ne[2], (long long) dst->ne[3], path);
}

// One bounded capture of the first HIP0 layer-11 QSA invocation for input
// identity diagnosis. Tensor spans may contain padding; use ne/nb to interpret.
static void ggml_cuda_qsa_dump_first_inputs(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, long audit_id) {
    const char * dir = std::getenv("GGML_QSA_INPUT_DIR");
    if (!dir) return;
    if (!std::getenv("GGML_QSA_MASK_AUDIT")) GGML_ABORT("QSA input capture requires mask audit");
#ifdef GGML_USE_HIP
    if (audit_id < 0 || ctx.device != 0 || strcmp(dst->name, "qsa_flash_attn_il11") != 0) return;
    static std::atomic<bool> captured{false};
    if (captured.exchange(true)) return;
    // The CPU-reference replay must use the *actual* layer attention scale,
    // and must reject ALiBi/softcap/sink variants it does not implement.
    float scale = 0.0f;
    float max_bias = 0.0f;
    float softcap = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    fprintf(stderr,
            "QSA_INPUT_META backend=HIP device=%d id=%ld tensor=%s scale=%.9g max_bias=%.9g softcap=%.9g sinks=%d\n",
            ctx.device, audit_id, dst->name, scale, max_bias, softcap, dst->src[4] != nullptr ? 1 : 0);
    for (int i = 0; i < 3; ++i) {
        const ggml_tensor * t = dst->src[i];
        const size_t bytes = ggml_nbytes(t);
        if (!t->data || bytes == 0 || bytes > size_t(128)*1024*1024) GGML_ABORT("QSA input capture: invalid tensor span");
        std::vector<unsigned char> data(bytes);
        CUDA_CHECK(cudaMemcpyAsync(data.data(), t->data, bytes, cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        char path[4096];
        const int n = snprintf(path, sizeof(path), "%s/HIP0-id%ld-%c.bin", dir, audit_id, "qkv"[i]);
        if (n < 0 || size_t(n) >= sizeof(path)) GGML_ABORT("QSA input capture: path too long");
        FILE * f = fopen(path, "wb");
        if (!f) GGML_ABORT("QSA input capture: cannot open %s", path);
        const size_t written = fwrite(data.data(), 1, bytes, f);
        const int closed = fclose(f);
        if (written != bytes || closed != 0) GGML_ABORT("QSA input capture: write failed");
        fprintf(stderr, "QSA_INPUT id=%ld name=%c type=%s ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu bytes=%zu path=%s\n",
                audit_id, "qkv"[i], ggml_type_name(t->type),
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3], bytes, path);
    }
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst); GGML_UNUSED(audit_id);
#endif
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const long qsa_audit_id = ggml_cuda_qsa_audit_mask(ctx, dst);
    ggml_cuda_qsa_dump_first_inputs(ctx, dst, qsa_audit_id);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
    ggml_cuda_qsa_dump_attn_output(ctx, dst, qsa_audit_id);
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
