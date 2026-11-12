# Report 0022 — fixed-reference QSA predictive-fidelity pilot

**Status: Completed for this bounded pilot; numerical equivalence and production promotion remain NO-GO.** Started at `d771f75f3942d019b12939764060482309e77634`. This is exploratory evidence, not a quality threshold, a full-model performance comparison, or a statistical acceptance study.

## Integration, alignment and activation

The mixed CUDA/HIP Release build succeeded for `llama-qsa-reference`, `llama-server`, `test-backend-ops`, and `llama-bench`; `llama-tokenize` also built. Python compileall and all 67 `test_*qsa*.py` tests passed. Fixed two runner integration hazards: reject a nonempty output directory (not merely existing manifests), and reject logical **or physical** batch sizes below four, including the actual initialized context's `n_ubatch`. A nonempty directory and `--ubatch-size 1` were rejected; the rebuilt runner reproduced all four short dense logits byte-for-byte. A fresh bounded server request with diagnostic and sparse flags unset succeeded without QSA diagnostics. The example is an explicitly invoked CMake target, not an automatic server operation. Production selectors, math, and fail-closed overflow checks were not changed.

Each scored batch contains four consecutive tokens at positions `p..p+3`, all four output flags are set, and row `i` uses `llama_get_logits_ith(ctx,i)` and labels `tokens[p+i+1]`. The regular decoder's causal attention supplies the prefix through `p+i`. Independent `llama-tokenize --no-escape --ids` on the identical long input produced 32,757 tokens; **all 32 consumed IDs and all 32 next-token labels**, including the final label, match its independently produced token list at the stated positions. All three long manifests and label files match, including prefix hashes and group positions. This independently checks indexing, not a second tokenizer implementation.

On the short fixture (55 tokens), replacing only the fourth query token, position 53, left the first three input-prefix fingerprints and their full logits byte-identical while changing the fourth logits row. This directly checks future-query isolation in that group. An external diagnostic-only Q=1 variant scored the same four prefixes and labels; all four top choices agreed, but logits were not bit-identical (max absolute differences 0.31–1.76), so it does not establish numerical equality or a tolerance. The Q=1 variant was not used for the sparse experiment. This limited mutation check does not independently prove causal masking for every position and model architecture.

At 32K, all three runs recorded 96 Q=4 masks each: 16 CUDA and 80 HIP, with 24 query / 2 KV heads. A separate otherwise-matched sparse four-row run under the previously used ROCm kernel tracer recorded **10 actual** `flash_attn_ext_f16<256,256,4,4,false,false,true>` launches, five per HIP device. Audit timing/layers match the ten Q4 HIP operations (layers 11–47); the final template parameter is `use_sparse_kernel=true` in `fattn-mma-f16.cuh`. The 32-row measurement itself was not profiled: its 80 HIP Q4 audits have the same eligible shapes, selector, KV=32768 and bound=2051, and source dispatch implies sparse activation, but 80 launches were not independently traced. No legacy GQA4 or forced-WMMA flag was set. All runs disabled graphs with `GGML_QSA_MASK_AUDIT=1000` and `GGML_PERF_DISABLE_CUDA_GRAPHS=1`. For example, the first layer-15 Q4 group had union 3028 in both dense runs versus 3040 sparse, capacity 8204; this is a selection-count observation, not a causal attribution or a raw-index comparison.

## Fixture and results

Target was the existing two-shard `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` GGUF, vocabulary 248,320, embedded GPT-2 tokenizer with `qwen35` pretokenizer; exact model upstream revision is not recorded in the local GGUF. All three processes used the same local files, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 K/V, context 65536, Flash Attention on, no draft model or sampling. The long fixed text is the user-message content of the pre-existing `20261111-143704/server/mtp-controlled-32k-request.json` fixture: synthetic repetitive background originally created for context-performance testing, **not held out**, not a benchmark dataset. Its UTF-8 SHA-256 is `bbed1c433ad8557994b0c46603997d6ba75ce7aca9deae540b851fed667ba8ef`. Reference labels are text tokens, not model predictions. Rows cover positions 32724–32755 and predict positions 32725–32756. Raw text, tokens, captures, logs, comparisons and trace are outside Git at `/home/chris/workspace/llama-perf-results/iteration-0022/`.

| Metric, 32 long rows | Dense A vs B | Sparse vs dense A |
|---|---:|---:|
| Mean reference NLL, nats | 1.705491 / 1.705491 | 1.705491 / 1.702720 |
| Mean / median delta NLL, nats | 0 / 0 | -0.00277154 / -0.000001508 |
| Min / max delta NLL, nats | 0 / 0 | -0.103140 / +0.053949 |
| Reference top-1 hits | 22 / 22 | 22 / 22 |
| Top-1 changes / top-5 set changes | 0 / 0 | 0 / 4 |
| Max KL(dense||other) / KL(other||dense), nats | 0 / 0 | 0.0135723 / 0.0141867 |
| Max Jensen–Shannon divergence, nats | 0 | 0.00345615 |

Dense A/B F32 logits are byte-identical for all 32 rows. Sparse NLL rose in 16 rows and fell in 16. Reference ranks changed at position 32732 (486→470), 32734 (9→8), and 32752 (120→116). The largest regression was position 32746: label rank 5 in both, NLL 2.671491→2.725440, top token ID 321 in both. The largest improvement was position 32752, -0.103140 nats. Maximum KL occurs at position 32729: top token ID 24 in both, probability 0.448994→0.434446. Zero top-1 flips does not imply equality. Metrics use full-vocabulary softmax, deterministic ID tie-breaking for ranks, and no arbitrary acceptance tolerance.

A second fixed short input is the existing `mtp-context-short-request.json` user message (SHA-256 `42b925dcf01b719348515fb99c56ddea9ca32ed7345abba918088df60953ddc4`). Dense A/B and selector-enabled sparse yielded four byte-identical rows, NLL 3.044238 and 2/4 top-1 hits. Its KV=256 fails the sparse selector's KV>=4096 guard; **this is not a sparse fidelity result**.

## Decision and limits

The tiny non-held-out sample has a slightly lower mean sparse NLL, with meaningful individual changes in both directions. It does not establish product quality, equivalence, or a sparse performance benefit. The narrowly justified next step is a larger independently held-out model/task-quality calibration with pre-agreed statistical measures and explicit user approval of any acceptance criterion. Headwise arithmetic, GDN R/S, natural MTP acceptance, graph replay, mixed sequences, and graph-safe overflow remain separately measured or blocked. No 98K replay, full-model performance A/B, GitHub Actions, graph-enabled sparse run, driver change, or default promotion was performed.
