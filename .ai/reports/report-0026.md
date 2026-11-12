# Report 0026 — counterfactual source grounding

**Status: Completed for the bounded synthetic probe. Numerical correctness and production promotion remain NO-GO.** Before sparse, dense A/B solved both variants of all **6/6 pairs**, including 2/2 pairs in each source-position band and 3/3 of each response type. Sparse solved all six complete pairs on the unchanged manifest. This is not a numerical-equivalence result or an approved quality tolerance.

## Validation and frozen manifest

Starting commit `a0b3c9b199ae83f3e83fe4030e7f89442925b9fb`; ending commit is the commit containing this report (identified by the pushed completion message). Mixed CUDA/HIP Release `build-perf` targets `llama-server` and `llama-tokenize` built successfully. Compileall passed; 116 `test_*qsa*.py` tests passed, including 13 direct counterfactual tests. Initially two new tests failed because they assumed the first sorted pair was the first generated pair; fixed them to select by pair ID. Added duplicate-variant and changed-prefix negative tests and made the analyzer reject a changed prompt prefix. Bad gold, unrelated-row mutation and sparse token mismatch tests also pass. No production numerical code changed.

Generator SHA-256 `fb9e20432db475d29e71034ad611ff983e0b6e1fa4c15d42fc37ecfd060b4edb`; analyzer SHA-256 `fb1d8df4403a9a4ad8996436e15a748fd5c3cd24b610c50669d7f6e691926668`; tests SHA-256 `075324aeb9dd384f06de7e6d9ca50d2ceda4f273eed4361f46bc39970febc041`. Frozen manifest SHA-256 `93b307bb0948895fd194dce0b287a724bcb31a9048ab5926612baea47dffd19f`, seed 20261114, 520 records, 12 tasks. A separate literal-ledger parser checked every row, target ID and position, answer, identical query/prefix, and exactly one changed field in exactly one row per pair. Targets are zero-based rows 3, 260 and 516. Before inference, target tokenizer counts were 21,536–21,641 user tokens and prompt sizes 44,696–44,836 bytes; safely above 16,408 and below context 65,536. Q=4/GQA12 eligibility was subsequently confirmed by HIP audit (24/2 heads, bound 2051, padded KV 21760). No tasks or labels were modified after generation.

## Dense-first and sparse results

Separate fresh, bounded servers used the same Qwen3.8-Flash-Next two-shard IQ3_XXS target and Q8 MTP draft, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, context 65536, maximum draft 3, reasoning off, temperature 0, top-p 1, seed 42, no prompt cache, and graph-disabled mask audit. Legacy sparse and forced-WMMA flags were unset. The dense-only analyzer with `--require-baseline-pair` succeeded on 6/6 before the sparse process was started. Sparse used only `GGML_HIP_QSA_SPARSE_Q4_GQA12=1` in addition to the shared settings and ran all 12 cases.

All 12 dense A, dense B and sparse answers exactly matched their actual independently reconstructed source references, with `stop` finishes. Text and finish reasons matched across all conditions; prompt token counts matched per task across all three. Each counterfactual variant produced its distinct correct answer: no pair returned the same response for two different references. No dense-repeat instability, token-count incomparability, sparse regression, or sparse improvement was observed. The task-level outcomes and actual launch attribution are:

| Task | Server prompt tokens (all conditions) | Dense A / B / sparse vs actual reference | Natural Q4 verification groups / sparse HIP launches |
|---|---:|---|---:|
| retrieval-early-a | 21648 | pass / pass / pass | 3 / 30 |
| retrieval-early-b | 21646 | pass / pass / pass | 2 / 20 |
| retrieval-middle-a | 21629 | pass / pass / pass | 3 / 30 |
| retrieval-middle-b | 21629 | pass / pass / pass | 2 / 20 |
| retrieval-late-a | 21586 | pass / pass / pass | 2 / 20 |
| retrieval-late-b | 21589 | pass / pass / pass | 3 / 30 |
| instruction_following-early-a | 21691 | pass / pass / pass | 5 / 50 |
| instruction_following-early-b | 21691 | pass / pass / pass | 5 / 50 |
| instruction_following-middle-a | 21655 | pass / pass / pass | 5 / 50 |
| instruction_following-middle-b | 21655 | pass / pass / pass | 5 / 50 |
| instruction_following-late-a | 21644 | pass / pass / pass | 5 / 50 |
| instruction_following-late-b | 21643 | pass / pass / pass | 5 / 50 |

For each of early, middle and late, sparse preserved both answers in **2/2 baseline-solved pairs**, one retrieval and one strict JSON pair. No pair was incomparable; none lacked kernel attribution. The two variants are correlated and are not independent benchmark samples.

ROCm kernel tracing found **570 actual** `flash_attn_ext_f16<256,256,4,4,false,false,true>` sparse HIP launches, including **450 within 45** natural `target_process` spans with batch tokens 4 and draft tokens 3. Each span contained exactly five launches on each of HIP agents 1 and 2. Server monotonic span times were correlated to profiler timestamps and to per-request monotonic client boundaries; every request had attributed groups on both HIP devices. This is launch evidence, not a timing benchmark. Raw prompts, gold, answers, manifest, local runners, independent parser, trace summary, logs and profiler CSV are outside Git under `/home/chris/workspace/llama-perf-results/iteration-0026/`.

This matched synthetic sample is not held-out certified, statistically representative, numerical parity, or evidence of graph replay, mixed-sequence, or graph-safe overflow correctness. Known logits/GDN-state differences remain. No sparse default, user quality threshold, full-model throughput A/B, GitHub Actions, GPU reset or driver change was introduced. Processes were cleaned up.
