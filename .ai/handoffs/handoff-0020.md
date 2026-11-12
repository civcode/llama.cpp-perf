# Handoff 0020 — exercise GDN source-cell indirection and checkpoint restoration

## Objective and context

Use small local fixtures to close the two remaining unvalidated GDN observer paths: real nontrivial cell.src source-row indirection and the actual checkpoint-restore/replay branch, distinct from recurrent snapshot rollback. Independently verify captured **bytes** against recurrent-only serialization on the same logical sequence and checkpoint. Do not repeat the full 5.3-GiB capture just to test coverage. Preserve numerical NO-GO and do not run production-performance A/B.

Repository: civcode/llama.cpp-perf
Branch: docs/llama-perf
Starting commit: exact transfer commit supplied by Chat AI
Ownership: Coding Agent exclusively until report is pushed.

Read .ai/WORKFLOW.md, Reports 0017-0019, and docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md.

### Evidence from Report 0019

Three naturally aligned 32K mixed GPU Qwen3.8-Flash-Next target + MTP runs used CUDA0/ROCm0/ROCm1 4/7/7, IQ3_XXS target + Q8 MTP, F16 KV, context196608, max draft3 and seed42. All 16 verification groups had matched histories and decisions. Dense A/B matched exactly all 3504 GDN snapshots; dense/sparse differed in 2592 (R/S; P unchanged). Max R/S row NMSE .00195039/.00154951. The first state difference preceded sampling (layer 12). Dense/sparse logits differed in 64/64 rows, although top-1 did not flip; top-5 changed in 5/64 and max KL(dense||sparse) .0106312 nats, JS .00260841. No numerical equivalence proven, and no tolerance established. These data are not model-performance measurements.

The existing logical GDN observer checked its selected offsets against recurrent state_write range metadata on 36 R, 36 S and one P layer per checkpoint, including planes 0, 2 and 3. But all observed source/physical cells were zero. No live nontrivial source indirection, separate emitted-byte comparison or checkpoint-restore path was verified. Snapshot rollback followed by target reprocessing WAS observed.

### New code committed by Chat AI; unverified on Coding Agent host

- src/llama-memory-recurrent.cpp: GDN tensor records now include source_cell = cell.src and cache_size, in addition to existing physical_cell, logical_source_row, rs_plane and logical_cell_pos. No cache or math changes.
- tools/server/server-context.cpp: GDN events now include rollback_mode = none/snapshot/checkpoint_restore, based on the existing use_ckpt_tgt condition; no MTP decision logic changes. Optional LLAMA_QSA_GDN_GROUP_LIMIT=1..16 (default16, parsed only when LLAMA_QSA_GDN_DIR set) allows one or two bounded verification groups rather than 5+ GiB.
- docs/LLAMA-PERF/tools/audit_qsa_gdn_coverage.py: validates logged row = rs_plane*cache_size + (source_cell >= 0 ? source_cell : physical_cell), state metadata consistency across all layers, R/S completeness and which source/rollback cases actually occurred. Optional --require-source-indirection and --require-checkpoint-restore fail with exit 2 when absent. It does not read tensor data and cannot validate byte transfers.
- docs/LLAMA-PERF/tools/test_audit_qsa_gdn_coverage.py: five synthetic cases for ordinary/indirected source cells, pending plane, checkpoint-restore tag, invalid rows and missing S.
- Updated documentation. No mixed HIP/CUDA compile or Python test yet by Chat AI. New fields do not exist in old 0019 manifests; collect fresh small traces.

## Ordered tasks

### 1. Validate the new tooling

Run Python syntax checks and the new test_audit_qsa_gdn_coverage.py, plus existing test_compare_qsa_model_state.py and focused regressions. Fix ordinary script errors autonomously. Build locally with cmake --build build-perf -j 28 --target llama-server test-backend-ops llama-bench. Confirm server builds with use_ckpt_tgt computed once before diagnostics and that flag-unset behavior is unchanged. No GitHub Actions.

### 2. Verify a truly redirected source cell

Construct the smallest supported live test that causes source_cell >= 0 and source_cell != physical_cell, preferably via a sequence copy/shared sequence and subsequent independent continuation or a bounded llama_memory_recurrent fixture exercising seq_cp, find_slot, and ownership. Do not falsify cell.src in normal model inference. Use a fresh LLAMA_QSA_GDN_DIR, LLAMA_QSA_GDN_GROUP_LIMIT=1 (or 2) and the coverage analyzer with --require-source-indirection. Record the actual owner sequence, physical/source cell and rollback plane. A synthetic unit test is useful but not a substitute for live coverage.

On the same small state, independently emit the recurrent-only serialized R/S/P bytes via llama_memory_recurrent::state_write / a bounded llama_io_write_i writer, then compare byte-by-byte to the observer dump for the same layer, logical row, type and checkpoint. The current range-only writer does not suffice as an independent byte check. Validate device ownership and negative paths (bad row, absent owner, mismatched bytes). Keep the fixture compact; do not serialize huge long-context attention/indexer KV caches.

### 3. Validate the real checkpoint-restore path

Find the smallest supported natural rejection that takes use_ckpt_tgt true, such as a supported draft length exceeding llama_n_rs_seq or documented FULL sequence-removal mode. Do not change MTP acceptance/math to force a production path. Capture a genuine restore + sequence removal, followed by replay/reprocessing; verify rollback_mode records and actual state. Use --require-checkpoint-restore to prevent an untested path from being reported as covered. Check restored recurrent R/S/P against a pre-checkpoint serialized reference and inspect ownership/plane after restore. If there is temporarily no active logical cell after restoration, implement a bounded, correct observation point and fail closed rather than inventing zero state.

Avoid repeating the full 32K three-process 5.3-GiB capture. Prefer 1-2 small groups and short prompts where possible. Use existing bounded server startup/request deadlines and process-group cleanup. Keep all raw data and local runners outside Git in /home/chris/workspace/llama-perf-results/iteration-0020/.

### 4. Decision

Report exactly what was exercised and what remains untested. Even if the observer now passes both missing paths, the measured sparse-vs-dense recurrent and logits drift remains; numerical equivalence is NOT established. Summarize whether exact dense/sparse identity is plausible given different WMMA accumulation arithmetic, and outline two possible next directions without silently choosing a product tolerance: (a) justified fidelity/quality criteria or (b) change experimental sparse arithmetic for closer dense matching. Do not launch long full-model A/B, 98K replay, or default promotion.

Keep GGML_HIP_QSA_SPARSE_Q4_GQA12=1 experimental only; do not weaken GQA4/GQA12 selector guards or fail-closed overflow host check. Do not change production graph policy, KV precision, recurrent state semantics, or MTP acceptance. No Actions, PRs, merges, force pushes, driver/GPU resets, privileged changes, nvprof, unbounded loops or abandoned servers. Preserve /home/chris/workspace/llama.cpp, build-mixed and model weights.

## Deliverable

Write .ai/reports/report-0020.md documenting exact build/suites, live nontrivial source-cell and restore coverage, independent serializer byte results and negative cases, limitations, and a concise recommendation on the model-fidelity decision. Commit/push source, tests, compact findings and report to docs/llama-perf.

Return the standard completion line:
Iteration: 0020 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0020.md | Summary: <1-2 sentences>
