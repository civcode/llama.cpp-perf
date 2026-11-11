# Handoff 0015 — QSA first-divergence, fixed-token logits and GDN-state gate

## Objective

Identify the earliest numerical divergence between default dense and opt-in GQA12 sparse QSA during real Qwen3.8-Flash-Next MTP, then measure its impact on fixed-token logits and Gated DeltaNet (GDN) recurrent state across acceptance/rollback. Do **not** run end-to-end performance A/B or promote sparse attention before this correctness gate is met.

## Starting point

Repository civcode/llama.cpp-perf; branch docs/llama-perf; synchronize the **exact handoff commit in Chat AI's transfer message** and take exclusive Coding Agent ownership. Read .ai/WORKFLOW.md, .ai/reports/report-0013.md, .ai/reports/report-0014.md, and docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md.

Report 0014 verified Q4/GQA12 (24 query heads / 2 KV heads) operator correctness, sparse dispatch, overflow rejection and synthetic whole-op speedups on both R9700s at 32K/98K. In a 32K real-model 64-output-token run, dense and sparse yielded identical sampled output hashes, 42/62 accepted/drafted, and identical 21 MTP verification outcomes, yet **181 of 273 QSA union counts differed** (maximum difference 52). Dense-vs-dense repeated all union counts exactly. That is *not* proof of numerical model equivalence. The audit disabled GPU graph replay. GPU replay, mixed sequences, fixed-token logits and GDN rollback-state equality remain unverified. No production defaults changed.

Chat AI has now implemented:
- Opt-in, bounded multi-mask binary16 dumps in ggml/src/ggml-cuda/fattn.cu: set GGML_QSA_MASK_AUDIT=1000, GGML_QSA_MASK_AUDIT_DIR to a unique empty external directory per run, and GGML_QSA_MASK_DUMP_LIMIT=256 (1..256 groups per backend/device). Without DUMP_LIMIT, the prior first-sample behavior remains.
- Standard-library comparator docs/LLAMA-PERF/tools/compare_qsa_mask_dumps.py: reads audit and dump logs, verifies F16 counts, compares mask membership and finite mask values, identifies first differences per backend/device and also count differences beyond captured dump limit.
- Host-only tests at docs/LLAMA-PERF/tools/test_compare_qsa_mask_dumps.py and updated audit documentation.

These newest changes are **not yet compiled or executed on the agent host**. The previous Q4/GQA12 sparse selector/overflow gates must remain intact.

## Ordered tasks

### A. Validate new diagnostics

1. Run python3 docs/LLAMA-PERF/tools/test_compare_qsa_mask_dumps.py and Python syntax checks; fix any comparator test, parsing or binary byte-order errors before model runs. Ensure identical, membership-changed with equal per-query counts, finite-mask-value-only differences, and unexpected nonfinite mask cases behave correctly.
2. Build with cmake --build build-perf -j 28 --target llama-server test-backend-ops llama-bench using the existing mixed HIP/CUDA toolchain. The new dump branch must compile for both CUDA and HIP; no-env dispatch and model behavior must be unchanged. Do not use GitHub Actions.

### B. Identify the first real-model mask-index divergence

1. Repeat the bounded Report 0014 Qwen3.8-Flash-Next IQ3_XXS target + MTP Q8 model, CUDA0/ROCm0/ROCm1 with 4/7/7 split, F16 KV, 196608 context, max draft3, same seed/prompt, 32K and up to 64 generated tokens. Run three **fresh** processes: dense A, dense B and sparse. Use the audit and unique dump directories in **all** three; both original GQA4 and old WMMA tuning flags remain **unset**; the GQA12 sparse run alone has GGML_HIP_QSA_SPARSE_Q4_GQA12=1. The diagnostic and sparse experiment both disable GPU graph replay, so their timings are invalid for performance conclusions.
2. Save actual token IDs, acceptance/rejection/rollback event traces, all mask-audit logs and raw F16 dumps outside Git at /home/chris/workspace/llama-perf-results/iteration-0015/. Compare dense A vs B first, then dense A vs sparse using the new tool. Confirm the raw F16 counts and union match audit records; report the earliest per-device changed selected KV indices and changed values at common selected positions. A same-size union does not imply identical indices.
3. Associate each first changed mask with a **logical** target step, attention layer, stream/sequence identity where genuinely known, and MTP state transition. Audit per-device ordinal is *not* a synchronized cross-GPU clock or inherently stable layer identifier. If dense A/B disagree in raw masks, diagnose that baseline nondeterminism first. If sparse differs before any sparse kernel could influence the indexer, investigate model inputs/graph/backend ordering instead of assuming the gather is at fault.
4. Once a specific divergence is localized, capture a small matched Q/K/V/mask or attention-output snapshot at the relevant layer/step and, if practical, test the *actual saved mask* under both kernels with per-head CPU-reference error. Compare indexer inputs/top-k score margins when selection is near tied. Distinguish different sparse attention accumulation on **identical masks** from differing later indexer selection. Avoid broad 98K profiling while 32K is unresolved.

### C. Fixed-token logits and recurrent-state comparisons

1. Implement a strictly **opt-in, bounded** teacher-forced/identical-token diagnostic through several MTP verification steps. Record exact prompt and candidate token IDs, cache positions, target/draft verifier inputs and acceptance outcomes, including fully accepted and rejected/rollback cases. Ensure both runs consume **identical logical tokens**; if forcing a speculative acceptance schedule for diagnosis, label it diagnostic-only and never claim normal MTP semantics.
2. Capture matched pre-sampling logits and **the actual relevant GDN recurrent state** (not just a sampled output or hash), with layer, sequence, position, shapes/dtypes and explicit checkpoint boundary: before verification, after accept/reject, after rollback/reprocessing. Use stable state ownership and stream-safe synchronization. Compute finite checks, max absolute and relative differences, normalized errors, and first element/layer divergence; derive interpretation/tolerances using dense-vs-dense control. Never relax normal math or acceptance rules to create agreement.
3. At the first divergence, capture enough adjacent indexer/attention/logit/state intermediates to establish whether QSA top-k sensitivity, a sparse attention numerical error, cache/rollback mismatch, or backend timing is responsible. Report what is **observed** versus **hypothesized**. Same sampled text or union cardinality alone does not pass this gate.
4. Start with 32K. If 32K numerical differences are understood and the harness is reliable, perform a *bounded* 98K correctness replay. If full GDN state cannot be inspected or token trajectories cannot be aligned, return Partial/Blocked with the specific reason rather than claiming correctness.

### D. Explicit decisions

This is a **correctness/causality iteration**, not a 512-token model-throughput iteration. Do not run the three-condition production-graph-on/dense-graph-off/sparse-model performance A/B. Even a verified fixed-token result would leave GPU graph replay, mixed-sequence support and graph-safe fail-closed overflow as separate promotion gates.

Do not remove the existing stream-synchronized overflow detection, weaken either GQA4 or GQA12 selector, change default attention/KV formats, scheduler synchronization, MTP behavior or production numerical math. No GitHub Actions, PRs, merges, force pushes, GPU resets, privileged changes, unsupported profilers, nvprof or unbounded processes. Preserve the original /home/chris/workspace/llama.cpp/, model weights and build-mixed/. Use bounded readiness/execution timeouts and process-group cleanup as in earlier reports.

## Report and ownership transfer

Keep binary dumps, raw logs, local runners and snapshots outside Git at /home/chris/workspace/llama-perf-results/iteration-0015/. Commit tested source fixes and compact analysis utilities. Write .ai/reports/report-0015.md with exact toolchain/commit, build and tool tests, dense-replicate vs sparse first mask indices/values and trace correlation, fixed-token logits and GDN states across rollback, explicit pass/fail/unknown gates and the narrow next action. Commit and push the report, then return:

Iteration: 0015 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0015.md | Summary: <1-2 sentences>
