# Handoff 0013 — live QSA mask bounds, fail-closed sparse overflow, then gated model A/B

## Objective

Close the actual-mask correctness blocker from Report 0012 **before** the opt-in RDNA4 sparse QSA operator is ever used in full-model inference. Observe real Qwen3.8-Flash-Next QSA masks on the dense/default path at ~32K and ~98K, ensure the selected-index bounds remain valid through MTP verification/acceptance/rollback and relevant sequence transitions, and make an over-capacity sparse union impossible to silently truncate. Only after these gates are met, perform bounded matched three-GPU MTP A/B runs and report whether an end-to-end benefit actually exists.

This is a safety and integration milestone, **not** an invitation to retune kernels or enable sparse attention by default.

## Starting point

- Repository: `civcode/llama.cpp-perf`; branch: `docs/llama-perf`. Start from the exact handoff commit supplied in the Chat AI transfer prompt. Work under exclusive Coding Agent ownership.
- Read `.ai/WORKFLOW.md`, `.ai/reports/report-0012.md`, `.ai/handoffs/handoff-0012.md`, `docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md`, and the benchmark methods in Reports 0008–0010.
- Report 0012 established HIP/CUDA build and test success on both R9700s and CPU-reference Q=4/GQA4 sparse-operator correctness at KV=32768 and 98304. Unprofiled synthetic whole-op 98K medians were dense/sparse 342.54/135.14 us (ROCm0) and 342.98/132.04 us (ROCm1); there was an unexplained ROCm0 204.53 us sparse outlier. **No real QSA masks, full-model inference, fixed-token logits, or GDN-state checks were verified.**
- Current opt-in selection: `GGML_HIP_QSA_SPARSE_Q4=1` only for a narrow RDNA WMMA FP16 Q=4/GQA4 shape. `GGML_HIP_FATTN_FORCE_WMMA_Q4` is independent and must remain unset.
- Source chain: `src/models/qwen4exp.cpp::build_inp_kpool` computes `n_sel = kpool*min(n_pool, top_k/kpool) + kpool - 1`; `build_qsa_sel` builds a mask with at most `n_sel` finite positions per query, by source-level argument; `build_attn_qsa` reshapes the mask into streams and supplies `n_sel` to `build_attn_mha` as `n_kv_max`. `ggml/src/ggml-cuda/fattn-common.cuh::launch_fattn` allocates `min(KV, ncols1*n_kv_max)` selected indices per query group. `ggml/src/ggml-cuda/fattn.cu::flash_attn_mask_to_sparse_indices` currently drops index writes when `dst >= n_kv_max` and clamps `row_count`: an unverified real mask might produce silently incomplete attention.

## Tasks, in order

### A. Establish real-model mask cardinalities on the **default dense path**

1. Introduce a *disabled-by-default, bounded* diagnostic at the actual QSA mask-to-attention boundary. Inspect the final F16 mask as the attention consumer sees it, not only the model's candidate slot list. Instrument without enabling sparse QSA and without changing attention results. Prefer an opt-in trace/capture with explicit capacity and controlled host synchronization; never silently assume debug tensor reads are cheap or that graph capture/reuse is irrelevant.
2. For each observed invocation and distinct stream/group, log enough context to audit the real bound: model/graph phase, attention layer, KV length, Q row count, stream/sequence identity as available, MTP verification iteration/accept/reject/rollback transition where observable, supplied `n_sel` (`n_kv_max`), per-query count of finite mask cells, exact union cardinality of each four-query group, and allocated index capacity `min(KV, 4*n_sel)`. Record any NaNs or unexpected non-finite mask values separately; do not treat NaN as a valid selected position.
3. Capture representative *actual* 32768 and 98304-KV runs (or the exact 32K/98K depths of the prior model fixtures) with MTP Q=4, including accepted drafts, rejected drafts/rollback, incomplete tail, and any relevant mixed/transitioned sequence scenario supported by the workflow. Distinguish observed cases from source-proven cases; if a transition is unreachable in the test setup, document why rather than inventing coverage. Keep logs and binary mask samples **outside Git** in `/home/chris/workspace/llama-perf-results/iteration-0013/` and write small analysis summaries/scripts to Git only when useful.
4. Prove or refute `finite_count(query) <= n_sel` and `four_query_union <= min(KV, 4*n_sel)` **for every measured group**, including graph reuse and rollback. Document the actual maximum and distribution of group union sizes, not only pass/fail. Be mindful of per-stream tiling and producer/consumer layout; do not accidentally combine or split streams incorrectly.
5. If an actual mask violates its declared bound, is stale/reused unexpectedly, or cannot be inspected reliably, **stop before enabling sparse full-model inference**. Diagnose the producer and report the blocker; synthetic correctness cannot waive this gate.

### B. Eliminate silent overflow in the *experimental* gather path

1. Make the actual pre-clamp compactor `row_count` observable as an overflow status for the opt-in Q4 variant; a union exceeding capacity must never silently proceed with only the clipped list. Implement the least disruptive correct fail-closed behavior. It is acceptable for this prototype to reject the sparse run with an explicit, deterministic overflow error **before gathered attention consumes truncated indices**. A genuinely correct dense fallback is also acceptable if it is designed and tested. Do not merely print a warning after the bad output or assume a device assertion propagates safely.
2. Respect HIP streams, asynchronous launches, graph capture/replay, pointer lifetime, and the dependent-kernel ordering. If host synchronization is required for a debug-only fail-closed check, isolate it under the opt-in experiment/diagnostic and disclose its timing impact; never modify default dispatch or scheduler behavior. If a safe check is impractical in the live execution path, keep full-model sparse testing blocked and explain the exact limitation.
3. Add targeted boundary tests: union strictly below the allocated bound, exactly equal to it, and intentionally larger (four queries selecting substantially different positions). Confirm both ROCm devices preserve CPU-reference results for valid cases and either correctly fall back or fail explicitly for overflow. Test the no-flag baseline remains unchanged. Ensure the over-capacity test does **not** report an apparently successful but incorrect output.
4. Build with the local mixed HIP/CUDA toolchain and re-run the strict C++17 selector test plus focused `test-backend-ops` tests. Do not weaken shape guards, finite-result checking, or the existing NMSE <= 5e-4 gate.

### C. Gated three-GPU MTP integration and measurement

**Only if A and B pass** and the experimental path is safe to execute: perform the matched 32K/98K uncached Qwen3.8-Flash-Next IQ3_XXS+MTP benchmark setup described in Reports 0008–0010: CUDA0/ROCm0/ROCm1, 4/7/7 split, 196608 context, max draft 3, F16 KV, unchanged prompt/output length, sampler/seed and acceptance settings. Use separate fresh processes with `GGML_HIP_QSA_SPARSE_Q4` fully unset versus exactly `1`; keep `GGML_HIP_FATTN_FORCE_WMMA_Q4` unset.

1. First run short bounded safety/correctness checks with the real-mask gate enabled, checking QSA use actually reaches the opt-in sparse compactor/gather, that no group overflows, and that outputs/acceptance trajectories match the baseline. If they do not, stop and investigate before measuring.
2. Where feasible implement a *fixed-token/teacher-forced* comparison of logits and Gated DeltaNet recurrent state across multiple verification and rollback events. Report errors/tolerances and any differences. Matching text or hashes alone do not establish numeric equivalence.
3. With diagnostics/profilers off for timing, alternate 2–3 clean baseline/sparse process pairs per depth. Capture tokens/s, per-verification-step latency, distinct draft/target/sync/reprocess phase medians, counts of accepted/drafted tokens and acceptance distribution, complete output hashes, and the actual sparse activation frequency. Record startup and resource limits. Compare like with like and avoid claiming a speedup if paths or draft histories differ.
4. If needed use **separate, small** profiling to attribute compaction, gather and waiting, never combine profiled whole-op numbers with unprofiled A/B. Explain the 98K ROCm0 sparse outlier from Report 0012 if evidence is available; do not make that an open-ended diversion.
5. Even if there is a full-model speedup, do not promote the new kernel to the default path in this iteration. A production proposal requires the fixed-token logits/GDN-state gate and adequate real-mask coverage.

## Constraints

- No GitHub Actions, PRs, merges, force pushes, automatic workflow triggers, upstream submissions, GPU reset, reboot, privileged operations or driver/security changes. Run local tests/builds; prefer `cmake --build build-perf -j 28` for changed targets.
- No `nvprof`, long/unbounded profiling, orphan servers or broad stress loops. Use preflight/liveness checks, explicit timeouts and process-group cleanup from prior iterations.
- Preserve `/home/chris/workspace/llama.cpp/`, the model files, and the reference `build-mixed/`. Use the fork's `docs/llama-perf` branch and only modify this project's agreed files.
- No changes to default attention selection, production KV formats, scheduler/stream synchronization policy, speculative MTP acceptance semantics, or model math. Trace and safety controls must be opt-in and isolated.
- A true safety-blocked result is valid; do **not** force full-model experiments merely to claim completion.

## Completion criteria and reporting

Write `.ai/reports/report-0013.md` with: (1) exact tested commit/toolchain, (2) true real-mask samples and cardinality bounds with phase/transition coverage, (3) overflow injection and fail-closed/fallback evidence, (4) per-device builds and focused correctness, (5) if permitted, matched three-GPU MTP numerical/performance results and their caveats, and (6) explicit go/no-go with the next narrow engineering action. Preserve raw data outside Git. Commit/push changes and the report, then transfer ownership back with exactly:

`Iteration: 0013 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0013.md | Summary: <1-2 sentences>`
