# Report 0017 — captured attention FP64 replay

**Status: Partial; numerical gate NO-GO.** Started at `ad262c43aceb6d6e607b4bf6394d1a63be595da6`. The mixed build and exact-input replay now work; on this operation sparse has lower overall FP64-reference NMSE than dense, but neither is exact and sparse has a slightly larger maximum absolute error. This does not establish model-level correctness. Layer-15 score margins, fixed-token logits and actual GDN state through acceptance/rollback remain unmeasured. No default promotion, kernel fix or full-model performance A/B was made.

## Build and reproduction

`cmake --build build-perf -j 28 --target llama-server test-backend-ops llama-bench` passed, compiling `fattn.cu` for both CUDA and HIP. Python syntax checks and replay, mask comparator and output comparator suites passed (6, 6 and 7 tests). Initially four replay tests failed: the synthetic K/V views deliberately have a feature stride larger than the element size, which the reader incorrectly rejected. Fixed the reader to accept element-aligned larger strides while retaining span bounds checks; also report all heads and the first nonzero error coordinate. The existing synthetic stride test exercises the fix. No project-wide Python dependency changes.

Three fresh bounded 32K/64-token processes used the Report 0016 IQ3_XXS + MTP Q8 fixture, CUDA0/ROCm0/ROCm1 split 4/7/7, F16 KV, context 196608, max draft 3 and seed 42. All had mask, input and output captures enabled with distinct directories; only sparse had the GQA12 selector enabled (dense explicitly set it to 0), and old GQA4/WMMA flags were unset. Auditing synchronizes and disables graph replay; these are not performance runs. Each emitted exactly one HIP0 layer-11 `QSA_INPUT_META`, with scale 0.0625, zero bias/softcap and no sinks. A separate short no-audit/no-experimental-flags server request completed with no metadata line. Process-group cleanup left no server. Raw data, logs, JSON comparisons, traces and bounded runner are in `/home/chris/workspace/llama-perf-results/iteration-0017/`, outside Git. No GPU reset, profiler or GitHub Actions was used.

Dense A/B matched all 273 raw mask groups and all 273 outputs bitwise. Replay also verifies the complete captured Q/K/V spans, their type/dimensions/strides, and raw F16 mask bytes are identical between dense A/B and dense A/sparse. The HIP0 layer-11 operation has KV 33024, union 3537, Q F32 [256,4,24,1], K/V F16 [256,33024,2,1]; the output is [1,4,24,256] in stream/query/head/feature order. The tool rejects unsupported bias/softcap/sinks and uses the logged scale and mask, grouped-query head mapping, and strided tensor views. The local `build_attn_mha` source passes these parameters to flash attention; this is an FP64 mathematical reference, not an emulation of GPU rounding.

## Exact-input errors

| Comparison | NMSE over reference energy | Max abs | Max location [stream,query,head,feature] |
|---|---:|---:|---|
| dense vs FP64 | 9.99118e-7 | 0.00801139 | [0,2,14,124] |
| sparse vs FP64 | 4.44979e-7 | 0.00825457 | [0,2,14,124] |
| sparse vs dense | 8.97579e-7 | 0.00535035 | [0,0,10,189] |

The first nonzero error for all three comparisons is [0,0,0,0]: dense minus reference +1.75464e-5, sparse minus reference +3.51298e-5, sparse minus dense +1.75834e-5. Exact FP64 disagreement is not itself a failure criterion. Per-head NMSE and max absolute error (dense / sparse):

| Head | NMSE dense | NMSE sparse | max abs dense | max abs sparse |
|---:|---:|---:|---:|---:|
|0|3.148e-7|3.012e-7|0.0017511|0.0019858|
|1|3.824e-7|4.024e-7|0.0029277|0.0019371|
|2|4.293e-7|4.405e-7|0.0027001|0.0022297|
|3|7.312e-7|3.312e-7|0.0021952|0.0022919|
|4|1.771e-6|3.917e-7|0.0035470|0.0023432|
|5|9.001e-8|7.670e-8|0.0013288|0.0010062|
|6|1.535e-6|4.564e-7|0.0030447|0.0016319|
|7|6.822e-7|2.633e-7|0.0017449|0.0015071|
|8|2.154e-7|1.624e-7|0.0015919|0.0018977|
|9|6.442e-7|6.224e-7|0.0034479|0.0031083|
|10|8.552e-7|5.652e-7|0.0056183|0.0020098|
|11|1.882e-7|8.743e-8|0.0015707|0.0010170|
|12|6.044e-7|3.521e-7|0.0024142|0.0016543|
|13|1.662e-6|9.268e-7|0.0044862|0.0059071|
|14|2.259e-6|6.776e-7|0.0080114|0.0082546|
|15|1.264e-6|4.344e-7|0.0032687|0.0022047|
|16|1.117e-6|5.961e-7|0.0039368|0.0044211|
|17|1.629e-6|6.348e-7|0.0079638|0.0044390|
|18|3.204e-7|2.836e-7|0.0028104|0.0024907|
|19|7.144e-7|2.350e-7|0.0026806|0.0033082|
|20|1.102e-6|3.284e-7|0.0046055|0.0022704|
|21|1.283e-6|6.306e-7|0.0051530|0.0058134|
|22|1.676e-6|2.180e-7|0.0033305|0.0042502|
|23|8.725e-7|5.688e-7|0.0064368|0.0041186|

Local `fattn-mma-f16.cuh` uses half operands and some half intermediates; [upstream source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh) also illustrates this precision distinction. Differences are consistent with mixed-precision arithmetic, but these results do not quantify individual rounding contributions or establish a model tolerance. No specific sparse kernel defect is confirmed by this evidence. The Report 0014 both-device/32K/98K/boundary operator results were not rerun because no kernel change was made; they remain separate synthetic evidence.

## Remaining work and limits

The layer-15 code was inspected: `build_qsa_sel` produces `indexer_score` before `ggml_top_k`, and maps pool selections via `pool_idxs`. The score tensor, pool IDs and cutoff were not captured, so near-tie margins, changed pool IDs and whether cache state or perturbation caused the selection difference remain unknown. Report 0016's layer-11-before-layer-15 pre-draft chronology and changed KV positions are the existing evidence, not a causal proof.

The bounded logical-token/logits and sequence/cell/position-aware GDN observer has **not** been implemented. In particular, there is no matched teacher-forced acceptance/rejection/rollback comparison and no justified logits or recurrent-state tolerance. Existing debug callbacks would perturb scheduler execution and do not supply those logical identities; using them without a validated observer would not establish the requested comparison. No 98K model numerical replay was attempted. GPU graph replay, mixed sequences and graph-safe overflow are additional separate gates. Continue with a bounded, synchronized, logically labelled score/top-k observer and token/state capture; retain dense-repeat control and the NO-GO until all required gates are measured.
