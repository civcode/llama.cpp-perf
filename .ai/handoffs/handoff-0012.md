# Handoff 0012 — opt-in RDNA sparse QSA for four-token MTP verification

## Goal

Validate one **existing-upstream-inspired, disabled-by-default** sparse flash-attention extension on Radeon AI PRO R9700 at 32K and 98K KV, for Qwen3.8-Flash-Next QSA's Q=4/GQA4, 256x256, FP16 K/V shape. Unlike Iteration 0011's failed dense-WMMA experiment, this path first compacts the union of the four query masks into selected K/V indices, then gathers **only those K/V rows** in the existing WMMA kernel.

**The objective is to eliminate unnecessary full-context K/V reads while preserving each query's exact mask.** A ~2048-position selected set can be distributed across 32768 or 98304 KV rows. Speedups are a hypothesis, not a forecast. No change should affect normal inference unless the new experiment env flag is explicitly set.

## Branch and references

Repo `civcode/llama.cpp-perf`, branch `docs/llama-perf`, handoff revision specified in transfer prompt. Read `.ai/WORKFLOW.md`, Reports 0010 and 0011 and `docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md`. Work under exclusive Coding Agent ownership and return it after the report is pushed.

Relevant community work is linked and analyzed in the audit. In particular:
- Upstream **merged** #28770 implements CUDA sparse FA mask union/index gather for Qwen4 (do not reimplement the whole graph): https://github.com/ggml-org/llama.cpp/pull/28770
- Upstream #28213 demonstrates a *different* compact gathered-K/V QSA decode algorithm, but only single-token: https://github.com/ggml-org/llama.cpp/pull/28213
- Issue #28734 profiles dense mask and pooled-key context growth: https://github.com/ggml-org/llama.cpp/issues/28734
- Sparse compactor performance regression #29281: https://github.com/ggml-org/llama.cpp/issues/29281
- JohnTDI-cpu R9700 fusion/graph experiments, same model family but different inference setup: https://github.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4

These are sources of architecture and testing ideas, not directly comparable speed measurements. Do **not** cherry-pick unrelated upstream PRs or copy whole backends. The current `qwen4exp.cpp` already has QSA kpool changes; do not assume the 2026-09 version is identical.

## Chat AI implementation (already committed)

1. `ggml/src/ggml-cuda/fattn-sparse-q4-tune.h` pure selector and `docs/LLAMA-PERF/tools/test_fattn_sparse_q4_tune.cpp`. Must have `GGML_HIP_QSA_SPARSE_Q4=1` exactly, RDNA, WMMA, Q/K/V 256, 4 queries, 1 sequence, GQA4, existing GQA optimization, F16 K/V, F16 mask, positive per-query max selected and at least 8 times that number in KV rows. Host-only C++17 selector test compiled and passed using `-Wall -Wextra -Werror -pedantic` on Chat AI runtime.
2. `ggml/src/ggml-cuda/fattn.cu`: opt-in 256x256 Q4/GQA4 chooses existing MMA/WMMA; extends `flash_attn_mask_to_sparse_indices` to 4-query union by instantiating the existing index compactor for ncols1=4; extends `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` to permit 4x4 on RDNA *only with the new env guard*. Other kernel selection/defaults stay as before. The old `GGML_HIP_FATTN_FORCE_WMMA_Q4` experiment must remain **unset**.
3. `ggml/src/ggml-cuda/fattn-mma-f16.cuh` marks 256x256, ncols1=4, ncols2=4 sparse specialization available; existing template launch and KV gather code are reused, not rewritten.
4. `tests/test-backend-ops.cpp` adds `test_flash_attn_ext_qsa_sparse` under `GGML_TEST_QSA_SPARSE_FA=1`, at KV32768 and KV98304 in `test` and `perf`; Q=4, 2 KV heads, GQA4, D=256, F16 K/V, 2048 maximum visible positions **per query** with a synthetic deterministic mask selected across the entire KV range. Adjacent queries have slightly different visible subsets; the union has 2048 positions in these synthetic fixtures. Finite-result and NMSE checks (<= 5e-4) use the existing MTP test harness. Normal test suite is unaffected.
5. The exact sparse guard was compiled locally and the synthetic mask was verified mathematically: KV32K and KV98K each select 1890-1891 positions per query, union=2048, so it fits the allocated maximum of 4x2048. **The real HIP kernel and graph are not compiled/tested by Chat AI**.

## Essential correctness concern

The upstream `flash_attn_mask_to_sparse_indices` function **silently truncates** the generated union at its `n_kv_max` buffer capacity (source at `fattn.cu`: `dst < n_kv_max`, `min(row_count,n_kv_max)`). For real QSA, `build_inp_kpool` computes `n_sel = kpool*min(n_pool,top_k/kpool) + kpool - 1`; `build_attn_mha` passes that as an upper bound for **each query**. Q4 sparse compaction allocates 4 times that per-query bound.

**Before full-model testing**, inspect real 32K/98K QSA masks and bound semantics (including the incomplete tail, mixed sequences and MTP acceptance/rollback) to ensure each per-query mask has <= declared live positions and union <= 4*n_sel. If the bound is violated or cannot be verified, **do not enable the new flag in a full model**. A source-level proof plus a bounded live instrumented check is preferable to unsafe inference. Do not silently drop selected rows.

The synthetic mask is designed to check sparse attention against the dense CPU reference; it does not prove production mask invariants or Gated DeltaNet state correctness.

## Tasks

1. **Build and audit first:** Sync exact revision; run host C++17 selector test, build `test-backend-ops`, `llama-server`, `llama-bench` with `cmake --build build-perf -j 28`. Validate the new HIP sparse compaction specialization and WMMA kernel instantiation with current toolchain (ROCm0 and ROCm1). Fix any compilation issues without weakening the strict guard. Confirm default unset kernel choice and baseline output remain identical to Report 0011. No production changes.
2. **Gate CPU-reference correctness:** `GGML_TEST_QSA_SPARSE_FA=1` for the two opt-in sparse-mask cases under both `test-backend-ops test` and `perf`. Compare flag **unset** vs `GGML_HIP_QSA_SPARSE_Q4=1` in separate processes, both ROCm0 and ROCm1, both KV32768 and KV98304. Example:
   `GGML_TEST_QSA_SPARSE_FA=1 GGML_HIP_QSA_SPARSE_Q4=1 ./build-perf/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=98304.*nb=4.*qsa_sparse=1'`.
   Verify the filter selects precisely one fixture and the experiment genuinely changes dispatch. Require all finite outputs and CPU NMSE <= 5e-4; do not change precision or tolerance to hide errors. Compare `flash_attn_tile` vs **two kernels**, `flash_attn_mask_to_sparse_indices<4>` plus `flash_attn_ext_f16<256,256,4,4,...true>` (plus fixup as present). If no true sparse kernel is used, diagnose before timing.
3. **Evaluate selected-index bounds:** Inspect GPU compactor, QSA mask construction and `n_kv_max` plumbing to show that no real QSA selected positions would be silently truncated. Include at least one 32K/98K real-mask sample or precise source proof covering incomplete tail, N=4 speculative verification and potential distinct per-query selected positions. If uncertain, keep full-model test blocked even if synthetic operator tests pass.
4. **Measure whole-op time incl. compactor:** Repeat matched 32K/98K operator timings at least three times each with the flag *unset* vs *1*, alternating short processes, no other HIP tuning flags. Capture medians/ranges on both GPUs. ROCprofiler once per condition to attribute mask compaction, gathered MMA, fixup and dense tile separately; do not compare profiled and unprofiled numbers as the same A/B. Context growth should ideally be flatter with sparse; mask compaction still scans O(KV). If sparse fails correctness or is slower at 98K on either device, **stop** and close this candidate without expensive mixed-GPU runs.
5. **Only after a proven device-local win and mask-bound validity:** Run matched unprofiled three-device IQ3_XXS+MTP 32K/98K uncached requests from Reports 0008-0010, CUDA0/ROCm0/ROCm1 split 4/7/7, 196608 context, max draft3, F16 KV and identical sampling/seed. Alternate clean process baseline/flag1; 2-3 repeats. Preserve output checksums, actual verification counts, accepted/drafted, latency per verification step and phase medians; verify output and acceptance trajectory identical before interpreting throughput. Add fixed-token logit/GDN recurrent-state validation before proposing production promotion; a sampled-token match alone is insufficient.
6. **If the sparse candidate fails** due to mask compaction overhead or WMMA slowdown, report exactly which phase fails and prefer either the upstream #28213 compact-K/V path (which needs a Q4/MTP adaptation), optimizing mask compaction, or independent indexer pooling/cache analysis as a later **separate** experiment. No blind tile, GQA or parallel block retuning.
7. **Deliver:** Write `.ai/reports/report-0012.md` with compiler/test outcome, operator correctness/NMSE, sparse vs dense kernel names/launch shapes, baseline-vs-sparse timings, bounds proof and go/no-go, any legitimate full-model result if all gates pass, and next narrow engineering direction. Keep raw traces and logs outside Git at `/home/chris/workspace/llama-perf-results/iteration-0012/`. Commit/push report and needed prototype fixes, return the standard completion pointer.

## Safety and process constraints

- No `nvprof`, no 5-hour timeout, no unbounded health polls, no orphan processes. Use small profiler preflight and bounded startup/execution with process-group cleanup. Avoid any superuser, GPU reset, reboot, PCIe/driver or security changes. If CUDA health fails again, record and stop heavy mixed runs.
- No GitHub Actions, PRs, merges, force-pushes or upstream contributions. Direct commits to fork's `docs/llama-perf` under `.ai/WORKFLOW.md`, no repeated upstream acknowledgement gate.
- **28 parallel build jobs**, preserve original checkout `/home/chris/workspace/llama.cpp/`, model weights and reference `build-mixed/`.
- Disabled-by-default experiment only. Do not change production numerical behavior, scheduler synchronizations, or default kernel selection.

## Completion message

`Iteration: 0012 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0012.md | Summary: <1-2 sentences>`
