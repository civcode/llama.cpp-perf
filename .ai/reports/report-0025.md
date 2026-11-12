# Report 0025 — baseline-calibrated natural-MTP task pilot

**Status: Completed for the bounded pilot. Numerical equivalence and production promotion remain NO-GO.** Dense A/B stably solved eight of 12 long-context tasks, so the predeclared feasibility stop did not trigger; sparse was then run on the **entire unchanged manifest**. The eight dense successes also passed sparse. This small synthetic sample does not establish quality parity or a product tolerance.

## Validation and frozen inputs

The mixed CUDA/HIP Release `llama-server` and `llama-tokenize` build succeeded. Python compileall passed, as did the direct calibration suite (9 tests), quality suite (17 tests), and all 103 `test_*qsa*.py` tests. Independent ledger parsing reconstructed all 12 actual frozen gold answers and checked anchored facts against their source records without executing generated code. Backward-compatible `unclassified` difficulty tests also passed. Fixed the scorer's `stable_dense_solved` tier count: it now depends on the two dense controls and their eligible context, not on whether sparse happens to report a matching prompt-token count. A regression test checks that sparse mismatch still blocks attribution while preserving the dense count.

Before any model answer, generated the requested seed `20261113`, 520-record, 12-task manifest and froze SHA-256 `031fb952638125bd4aab756e5e763267a716482fcff245e20237397baa634877`. User prompt bytes were 44,826–45,019; the target tokenizer counted 21,562–21,638 user-prompt tokens, below the 65,536 context and above the prior 16,408 sparse-KV eligibility floor. Actual server prompt counts including chat formatting are below. The same manifest, prompts, gold, order and output budgets were used throughout. Tasks are synthetic and not certified held out. Anchored cases repeat necessary facts next to the question and are **not distant retrieval tests**; lookup cases require one ledger record.

## Dense-only feasibility before sparse

Fresh dense A and dense B server processes produced identical output text, finish reasons and prompt-token counts on 12/12 tasks, with all counts eligible. `calibrate_qsa_dense_tasks.py --require-solvable` returned success: **8/12 stable dense-solved**, comprising 4/6 anchored and 4/6 lookup. Both dense controls solved all four retrieval and all four JSON instruction tasks, but none of the four simple arithmetic/code-reasoning tasks. Both anchored arithmetic outputs violated the exact requested literal form; both lookup arithmetic outputs were decimal literals but incorrect. All responses finished `stop`; no clipping or refusal was recorded. This feasibility decision preceded the sparse process and is not a quality threshold.

## Sparse execution and per-task outcomes

All processes used the same existing Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS target and Q8 MTP draft, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, context 65536, max draft 3, `--reasoning off`, temp 0, top-p 1, seed 42, no prompt caching, and graph-disabled mask audit. Only sparse enabled `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`; legacy sparse and forced-WMMA flags stayed unset. A locally orchestrated sparse server was traced with ROCm's previously used kernel tracer; no performance inference is made from traced execution.

Direct trace identified **610 actual** HIP `flash_attn_ext_f16<256,256,4,4,false,false,true>` sparse launches across all requests, of which **490** occurred inside 49 server `target_process` spans with four batch tokens and three natural MTP draft tokens. Audit recorded Q=4, 24 Q / 2 KV heads, bound 2051 and padded KV 21760. The following launch counts are inside the corresponding task's MTP verification spans, correlated using client request boundaries and server monotonic trace. Every group has ten launches, five on each HIP device; thus every task reached the actual experimental sparse path, not just an environment selector or prompt prefill.

| Task | Prompt tokens (all three) | Q4 MTP groups / sparse launches | Dense A/B and sparse | Sparse answer changed? |
|---|---:|---:|---|---|
| retrieval-anchored-1 | 21612 | 2 / 20 | pass / pass / pass | no |
| retrieval-anchored-2 | 21619 | 2 / 20 | pass / pass / pass | no |
| retrieval-lookup-1 | 21648 | 2 / 20 | pass / pass / pass | no |
| retrieval-lookup-2 | 21642 | 2 / 20 | pass / pass / pass | no |
| code_reasoning-anchored-1 | 21656 | 5 / 50 | fail / fail / fail | no |
| code_reasoning-anchored-2 | 21665 | 5 / 50 | fail / fail / fail | no |
| code_reasoning-lookup-1 | 21681 | 1 / 10 | fail / fail / fail | **yes** |
| code_reasoning-lookup-2 | 21664 | 1 / 10 | fail / fail / fail | no |
| instruction_following-anchored-1 | 21642 | 7 / 70 | pass / pass / pass | no |
| instruction_following-anchored-2 | 21688 | 7 / 70 | pass / pass / pass | no |
| instruction_following-lookup-1 | 21626 | 7 / 70 | pass / pass / pass | no |
| instruction_following-lookup-2 | 21628 | 8 / 80 | pass / pass / pass | no |

By category, dense A/B/sparse each scored retrieval **4/4**, code reasoning **0/4**, and instruction following **4/4**. By tier each scored anchored **4/6** and lookup **4/6**. Dense repeats were stable on 12/12, prompt usage aligned on 12/12, and no task was below the usage floor. There were **zero genuine sparse correctness regressions and zero improvements**; one lookup arithmetic answer changed, but both versions were incorrect. No raw prompts, answer keys, output text, server logs or kernel traces were committed; these and the local bounded runners remain under `/home/chris/workspace/llama-perf-results/iteration-0025/`. The initial sparse startup hit an unused port in TIME_WAIT; changing only the local port allowed the fresh sparse server to start. All processes were cleaned up.

## Interpretation

Four successful lookup cases demonstrate this specific single-record synthetic workload is not entirely floor-limited, but they are not a representative long-distance retrieval benchmark or evidence of statistical parity. Four arithmetic cases remain floor-limited, so sparse correctness changes on those cannot be assessed. Prior measured logits and GDN-state drift persists regardless of identical answers. Request user-selected real deployment workloads, contamination policy and an explicitly approved rubric before wider evaluation or any quality GO decision. Graph replay, mixed production sequences and graph-safe overflow remain separate gates. No GitHub Actions, 98K replay, graph-enabled sparse execution, full-model throughput A/B, default promotion, GPU reset or driver change was performed.
