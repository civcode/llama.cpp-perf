# Handoff 0016 — pinpoint pre-MTP QSA numerical divergence, then establish token/state correctness

## Objective

Determine the **first causal dense-vs-sparse discrepancy in real Qwen3.8-Flash-Next QSA** with aligned, layer-labelled mask and post-attention F32 outputs. The first divergent selected-index set in Report 0015 occurred in the **final four-token prompt batch, before any speculative draft**; rolling back MTP cannot explain that initial difference. Distinguish (a) already different Q/K/V/input/cache state, (b) dense vs sparse attention accumulation on *identical* inputs and mask, and (c) downstream QSA indexer/top-k sensitivity. Only then establish fixed-token logits and actual GDN recurrent-state correctness through subsequent acceptance/rejection/rollback. **No full-model throughput A/B until these numerical gates are met.**

## Repository and evidence

- Repository: `civcode/llama.cpp-perf`; branch `docs/llama-perf`; start at the **exact transfer commit** in the Chat AI message and take exclusive Coding Agent ownership.
- Read `.ai/WORKFLOW.md`, `.ai/reports/report-0014.md`, `.ai/reports/report-0015.md`, and `docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md`.
- Report 0014: GQA12 sparse dispatch, all 24 query-head CPU-reference checks, overflow rejection and 32K/98K **synthetic** inclusive speedups passed on both R9700s; a 32K real-model probe showed same sampled token hash/acceptance history but differing QSA mask unions.
- Report 0015: in three repeatable graph-disabled fresh runs, dense A vs B matched **273/273 raw F16 masks**. Dense vs sparse differed on **210/273 raw masks**, including **181/273 four-query union counts**; per-query selected counts remained equal. Earliest different mask was HIP0 ordinal 1, KV 33024, in the 4-token final prompt batch (slot 0, no draft/rollback). HIP0 ordinal 0 precedes that difference, and a sparse output could change subsequent indexer selection. Per-device ordinals and graph node numbers are **not** globally synchronized model step/layer identifiers.
- Experimental flag **only** `GGML_HIP_QSA_SPARSE_Q4_GQA12=1` on the sparse process. Leave `GGML_HIP_QSA_SPARSE_Q4` and `GGML_HIP_FATTN_FORCE_WMMA_Q4` **unset** for all runs. Never change production defaults, GQA guards, overflow check or graph policy. Opt-in audit disables GPU graphs and adds host synchronizations, so runs are *diagnostic only*.
- Chat AI already added two **unverified** diagnostics and host comparison tools:
  1. In `src/llama-graph.cpp::build_attn_mha`, when the mask audit is set and n_kv_max > 0, the exact flash-attention node is labelled `qsa_flash_attn_il<layer>`. This should disambiguate model layers without relying on `node_NNNN`.
  2. In `ggml/src/ggml-cuda/fattn.cu`, `GGML_QSA_ATTN_OUT_DIR=<external-dir>` plus `GGML_QSA_ATTN_OUT_LIMIT=256` (1..256 per device, defaults to one) synchronously dumps the **post-kernel F32 QSA output** with the same backend/device/audit invocation ID as its F16 input mask; shape is [value_dim, Q_heads, Q_positions, streams]. The output dump requires `GGML_QSA_MASK_AUDIT` and skips non-QSA attention. No-env model behavior is intended to stay unchanged. No HIP/CUDA build or actual GPU run yet.
  3. `docs/LLAMA-PERF/tools/compare_qsa_attn_out.py` compares labelled F32 outputs by device and ordinal, validates audit identity/shape, rejects nonfinite results and reports first changed feature/head/query, max abs/relative and head-wise normalized errors. `test_compare_qsa_attn_out.py` adds six host-only synthetic regressions. Existing `compare_qsa_mask_dumps.py` compares raw selected KV indices. These new Python files also remain **unexecuted** by Chat AI.

## Tasks in order

### A. Verify newly committed instrumentation

1. Run host-only Python tests for both `test_compare_qsa_mask_dumps.py` and `test_compare_qsa_attn_out.py`, plus syntax checks for both comparators. Fix any actual bugs in byte order, F32 shape parsing, identical-mask/aligned-layer matching, per-head indices and finite-output handling. Verify the tool can compare later ops even if their mask union counts differ. Use only Python standard library.
2. Build the mixed CUDA/HIP `llama-server test-backend-ops llama-bench` targets with `cmake --build build-perf -j 28 --target ...`. Check that layer naming, F32 copying and two dump env flags compile on **both** backends. Check existing GQA12 sparse synthetic CPU-reference tests and overflow protection remain unchanged. Do a short flag-unset sanity run. Fix build/runtime issues autonomously; keep all diagnostics disabled by default.
3. Confirm the output dump is made **after** the correct attention calculation and any stream-k fixup, and that its ID refers to the **same** F16 mask invocation. Abort/diagnose any mismatched tensor shapes, unreadable dumps or nonfinite outputs. Diagnostics must never be silently interpreted as production timing.

### B. Establish the first attention-boundary difference

1. Use original 32K final-prompt fixture, IQ3_XXS target + MTP Q8, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, 196608 context, max draft3, seed42, up to 64 output tokens, process-group cleanup and deadlines. Run three fresh processes: dense A, dense B, sparse, with distinct external output directories for masks and F32 outputs and all of:
   - `GGML_QSA_MASK_AUDIT=1000`
   - `GGML_QSA_MASK_AUDIT_DIR=<unique mask directory>`, `GGML_QSA_MASK_DUMP_LIMIT=256`
   - `GGML_QSA_ATTN_OUT_DIR=<unique F32 output directory>`, `GGML_QSA_ATTN_OUT_LIMIT=256`
   - Only sparse adds `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`.
2. Run both comparators dense A/B and dense A/sparse, saving compact JSON. Check dense A/B output stability **as well as** mask stability. Align by verified model layer and actual final-prompt processing step, not only per-device ordinal. Find the first QSA layer/stream where **inputs' selected indices match but F32 output changes**, and distinguish bitwise rounding differences from tolerance-relevant changes. Report per-head max abs, normalized error, changed features, and earliest later changed index set. If outputs already differ in dense A/B, diagnose baseline nondeterminism first.
3. If F32 attention differs on an identical mask, capture the smallest bounded, identical logical Q and relevant K/V rows at that layer (or a canonical minimal subgraph/fixture) and compare those inputs and per-head attention outputs under both kernels and a CPU reference. A difference with **identical Q/K/V and mask** is an accumulation/numerical-kernel issue; a difference in Q/K/V before that operator implies an earlier input/state issue. Capture indexer input scores/top-k margin immediately before the *first changed mask* to test near-tie selection sensitivity. Do not assume raw union differences are a kernel bug or accept tiny output differences as harmless without downstream evidence.
4. Preserve stable layer/phase labels, exact prompt token IDs and KV/position metadata. GPU log timestamps and ordinals alone are not a full logical execution trace.

### C. Actual fixed-token logits and Gated DeltaNet recurrent state

1. Implement a **disabled-by-default bounded** comparison harness that feeds identical predetermined target/draft token IDs through the two modes, capturing logits **before sampling** and the actual relevant recurrent-state cells before/after QSA target processing and across at least one fully accepted and one rejected/rollback MTP verification. When divergence in selected indices makes normal draft candidate IDs differ, use a controlled teacher-forced diagnostic for alignment and **identify this as a forced schedule**, not proof of ordinary MTP behavior.
2. Serialize or directly snapshot the **logical GDN state**, including pending rollback/snapshot row interpretation, with stable layer/sequence/cache-position identity and synchronized device copies. See `src/llama-memory-recurrent.cpp::state_write/state_write_data` and `tools/server/server-context.cpp` checkpoint/verification handling for possible existing mechanisms. **Avoid serializing the entire 32K KV cache** merely to inspect recurrent state; use a bounded targeted snapshot. A state hash or sampled tokens alone is insufficient.
3. Compare dense A/B and dense/sparse logits and GDN state with explicit finite checks, first changed layer/index, max absolute/relative differences and normalized errors; justify tolerances from dense-repeat controls and any relevant known CPU accuracy gates. Report first divergent tensor and likely cause. Do not weaken production numerical math or scheduler/rollback semantics to force agreement.
4. Only if the 32K discrepancy is localized and controlled comparisons are reliable, attempt a bounded 98K correctness replay. GPU graph replay and mixed-sequence checks remain separate promotion gates.

## Operational safety and scope

- **No three-condition 512-token full-model performance A/B in this iteration.** Do not enable/promote sparse attention by default.
- No GitHub Actions, PRs, merges, force pushes, GPU resets, privileged commands, unsupported profilers, driver changes, `nvprof`, unbounded loops or orphan processes. Preserve `/home/chris/workspace/llama.cpp/`, `build-mixed/` and model weights.
- Use local `build-perf`, 28 build jobs, bounded server startup/request timeouts and process-group cleanup. Store raw masks, F32 outputs, scores, logits and GDN state only under `/home/chris/workspace/llama-perf-results/iteration-0016/` (outside Git).
- Fix source and tests locally when needed. If exact token/state alignment cannot be made safe, report its specific blocker and retain no-go. A constrained partial result is better than an invented numerical pass.

## Completion

Write `.ai/reports/report-0016.md` containing exact revision/toolchain, Python tests and HIP/CUDA build, dense-repeat vs sparse first **layer-labelled attention output** and mask-index differences, Q/K/V or indexer-score evidence where necessary, fixed-token pre-sampling logits and logical GDN state after accepted/rejected MTP transitions (with exact coverage/tolerances), and explicit go/no-go for full-model A/B. Commit and push code, tools and report to `docs/llama-perf`; return:

`Iteration: 0016 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0016.md | Summary: <1-2 sentences>`
