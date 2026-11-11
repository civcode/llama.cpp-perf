# Report 0012

Status: **Partial — synthetic operator passes and is faster; full-model gate remains blocked.** No model inference or three-GPU benchmarks were run. The structural QSA bound is strong, but actual 32K/98K masks and the relevant live graph/rollback behavior were not measured. Do not enable this experiment for full-model inference until that remaining check is complete. Defaults are unchanged.

## Build and CPU reference

Synced to `1ab5f9261ef9a9e74b48204ee05c50fec4d79868`. The strict C++17 selector test passed. The mixed CUDA/HIP build of `test-backend-ops`, `llama-server`, and `llama-bench` passed with `-j 28`, including the 4x4 HIP and CUDA instantiations. With both experimental flags unset, the Report 0011 dense fixtures reproduced its 32K/98K NMSEs (8.24800058e-6 / 2.52415994e-5) on both ROCm devices.

For each QSA-like fixture, device, and flag state, the filter selected exactly one test; all outputs were finite and CPU-reference NMSE was below 5e-4. Results were identical on both devices:

| KV | Baseline (flag unset) | Sparse (flag=1) |
| --- | ---: | ---: |
| 32768 | 7.68722127e-7 | 2.71365931e-7 |
| 98304 | 6.60394904e-7 | 2.38422177e-7 |

These are synthetic masks, not a model logits or GDN-state check. Mask union is 2048, while per-query visibility is ~1890; production unions can be substantially larger.

## Dispatch and timing

Kernel traces, one correctness process per condition, confirmed the switch on both devices at both lengths. Baseline launched `flash_attn_tile<256,256,4,4,false>` and `flash_attn_combine_results`; sparse launched `flash_attn_mask_to_sparse_indices<4,false>`, `flash_attn_ext_f16<256,256,4,4,false,false,true>` and `flash_attn_stream_k_fixup_uniform<256,4,4>`. ROCprofiler reports work-item grid / workgroup sizes respectively: dense tile `(32,512,2)/(32,8,1)`, combine `(1024,8,1)/(256,1,1)`; sparse compactor `(256,1,1)/(256,1,1)`, gather `(7168,2,1)/(32,2,1)`, fixup `(512,4,4)/(256,1,1)`. These are trace grid sizes, not block counts.

Unprofiled whole-op perf, including compaction, three alternating baseline/sparse process pairs; microseconds/run, median (range):

| Device | KV | Dense | Sparse |
| --- | ---: | ---: | ---: |
| ROCm0 | 32768 | 113.64 (113.60–113.71) | 85.44 (85.44–85.51) |
| ROCm0 | 98304 | 342.54 (342.49–342.69) | 135.14 (134.72–204.53) |
| ROCm1 | 32768 | 113.60 (113.43–113.67) | 83.34 (83.29–83.43) |
| ROCm1 | 98304 | 342.98 (342.83–343.06) | 132.04 (131.96–132.12) |

The ROCm0 98K sparse outlier is unexplained; all three samples nevertheless beat their matched dense runs. Each perf run initialized the same synthetic Q=4, GQA4, two-KV-head F16 workload. Baseline flag was genuinely unset and the older WMMA flag was unset. These numbers do not establish a model speedup.

The separate, perturbed correctness traces gave illustrative per-kernel times in microseconds: at 32K dense tile/combine 143–144/8, sparse compact/gather/fixup 37–39/50–52/34–37; at 98K dense 467–468/8, sparse 85–95/53–57/37. These single profiled launches are not comparable as A/B whole-op measurements to the unprofiled table. Compaction grows with KV length.

## Bounds audit and remaining gate

`build_inp_kpool` sets `n_sel = kpool*min(n_pool, top_k/kpool) + kpool - 1`. In `build_qsa_sel`, each query has at most that many candidate cell slots: `n_top_pool` selected pools of `kpool` cells plus `kpool-1` tail slots. The fused GPU `top_k_mask_kernel` starts each mask row at -infinity and only writes those pool/tail cells; duplicates, invalid entries, invisible pool scores and tail sentinels cannot increase the number of finite cells. The unfused graph scatters exactly `n_sel` slots into an initially -infinity mask, with dead slots directed outside the viewed KV rows. The subsequent QSA reshape partitions rows into streams, and `build_attn_mha` passes `n_sel` as `n_kv_max`. Thus, assuming these operations execute as represented, a four-query union has at most `4*n_sel` finite positions, even if all four selections differ; an incomplete tail, mixed sequences, and an MTP rollback can alter indices/visibility but not the number of slots in a freshly constructed mask. `fattn-common.cuh` allocates `min(n_kv, 4*n_sel)` indices per group, sufficient for this structural bound.

This is a cardinality argument, **not a live verification** of actual model masks, stale/reused graph state, or all producer implementations and rollback transitions. The compactor silently clips any excess, and the opt-in selector checks shapes and a supplied bound, not QSA provenance or the actual count. No 32K/98K real-mask sample or overflow detector was obtained. Therefore the requested actual-mask safety gate is not considered satisfied, despite the source argument and synthetic win. Never treat a synthetic pass as evidence that arbitrary masks respect `n_kv_max`.

Next: instrument the dense/default model path to count finite cells per query and exact four-query unions against the passed bounds without enabling sparse attention, covering both context depths, MTP acceptance/rollback and sequence transitions; also add a non-silent overflow diagnostic/fallback for the experimental path. Only after verifying those actual masks should matched three-GPU MTP runs be attempted; before promotion validate fixed-token logits and GDN state, not just sampled output. The upstream [group-union sparse FA PR 28770](https://github.com/ggml-org/llama.cpp/pull/28770) and the distinct single-token [gather PR 28213](https://github.com/ggml-org/llama.cpp/pull/28213) were consulted as references.

No GitHub Actions or PRs were used. Bounded runs and profiler traces are under `/home/chris/workspace/llama-perf-results/iteration-0012/`. No source fixes were made.
