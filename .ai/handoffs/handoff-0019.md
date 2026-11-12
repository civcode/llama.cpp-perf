# Handoff 0019 — validate logical GDN snapshots and numerical effect of logits drift

## Objective

Complete the next real-model numerical-correctness gate for experimental Q4/GQA12 sparse QSA: capture and compare **actual logical Gated DeltaNet (GDN) recurrent-state R/S/P values** before MTP verification, after acceptance decisions, and after commit or rollback with subsequent reprocessing. Interpret already-observed dense-versus-sparse F32 logits differences at the probability distribution level. No full-model performance A/B or default promotion until evidence warrants it.

## Starting point and evidence

Repository: civcode/llama.cpp-perf; branch docs/llama-perf; synchronize the exact starting commit supplied in Chat AI's transfer message. Read .ai/WORKFLOW.md, Reports 0016/0017/0018 and docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md. Take exclusive Coding Agent ownership until reporting.

Report 0017: exact real-model byte-identical Q/K/V/mask layer-11 inputs; sparse NMSE to FP64 reference 4.44979e-7 vs dense 9.99118e-7, but slightly larger sparse maximum absolute error. This does **not** establish an acceptable model tolerance. Report 0018: layer-15 pool 1040 vs 6316 changed across narrow top-k cutoff (dense margin .0034008; sparse .0010815; paired scores shift by up to .016921); pool-to-KV mapping identical. More importantly, 64 full-vocabulary target-verification logit vectors had **identical natural token/draft/acceptance histories** and were bitwise identical dense A/B, but **all 64** differed dense/sparse, max abs .9186411, max per-row NMSE .00462375. Eight fully accepted and eight rejected/rollback-pending groups were aligned. No GDN state was measured. Sparse numerical equivalence is thus **NO-GO**.

Experimental sparse activation only GGML_HIP_QSA_SPARSE_Q4_GQA12=1. Old GGML_HIP_QSA_SPARSE_Q4 (GQA4) and GGML_HIP_FATTN_FORCE_WMMA_Q4 must be unset. Keep fail-closed union overflow host synchronization, graph-disable policies and all production defaults.

## Chat AI implementation already committed (not built or run on agent host yet)

1. Internal diagnostic header src/llama-qsa-debug.h and a libllama bridge in src/llama-context.cpp exposing llama_qsa_debug_dump_gdn for the server's opt-in observer. It synchronizes the context, obtains the hybrid-indexed recurrent memory by base-class cast and reads recurrent state **without** serializing the massive long-context attention/indexer KV caches.
2. src/llama-memory-recurrent.h/.cpp now implements qsa_debug_dump_state: finds the sequence-owned active tail, resolves the exact same logical source row as state_write, namely rs_idx[seq] * size + (cell.src >= 0 ? cell.src : tail), and synchronously reads actual GDN R/S/P tensor rows per present layer via ggml_backend_tensor_get. Records type, layer, logical cell position, physical cell, source row and snapshot plane. Raw files are F32/F16/BF16, capped at 8 MiB each and 512 MiB total/process; unsupported or missing state fails closed; no cache mutation.
3. tools/server/server-context.cpp: new independent LLAMA_QSA_GDN_DIR=<fresh external directory> opt-in. First 16 text-only slot-0 verification groups emit chronologically numbered JSONL event records with exact target-token histories, draft candidate IDs, accepted tokens, rollback/replay metadata and sequence/position; snapshot phases are pre_verification (post target-decode, before sampling), post_verification (after sampling, before any cache edit), and post_accept or post_rollback after corresponding cache transition. Each state reader appends R/S/P tensor records. The existing LLAMA_QSA_LOGITS_DIR capture from Report 0018 is separate; use both on all three processes. No forced sampling, no default effects.
4. docs/LLAMA-PERF/tools/compare_qsa_model_state.py now rejects misaligned logical_cell_pos, supports up to 16384 records/32 MiB manifest, and retains strict target/draft/accepted/rollback/replay event equality; its tests include new rollback and logical cell position cases.
5. New docs/LLAMA-PERF/tools/compare_qsa_logit_distributions.py plus test_compare_qsa_logit_distributions.py (six synthetic tests) take already captured full-vocabulary F32 logits with strictly aligned natural histories, reporting top-1/top-5 changes, dense and sparse choice probabilities, KL in each direction, JS in nats and centered-logit differences after removing a constant shift. Stdlib only; at most 64 rows by default. These measures are NOT correctness gates on their own.
6. Updated docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md. All code is pushed but was statically reviewed only by Chat AI. Local mixed HIP/CUDA compilation and tests remain unverified.

## A. Build, run tests, validate logical recurrent row semantics

1. Run Python syntax checks plus test_compare_qsa_model_state.py and test_compare_qsa_logit_distributions.py, and focused previous tests (top-k, mask, attention-output, FP64 replay). Fix ordinary script failures, including BF16/F16 parsing, manifest chronology, KL/JS numerical behavior, missing GDN fields.
2. Build locally: cmake --build build-perf -j 28 --target llama-server test-backend-ops llama-bench with established mixed CUDA/HIP. Verify the new libllama bridge is exported and server linked; fix RTTI, C++ scope, backend read and header issues autonomously. No audit/no sparse/no GDN flags must preserve normal model behavior and emit no new records.
3. **Before trusting any GDN measurement**, independently validate the row resolution against actual llama_memory_recurrent::state_write/state_write_data or another well-grounded recurrent-only reference for a small live sequence. Test a normal cell and pending rollback plane, cell.src indirection, correct layer/type/size, and per-device reads. Never confuse a physical row with its logical sequence snapshot. If a checkpoint/seq_rm stage has no active cell, report and resolve the boundary instead of silently comparing zeroes.
4. Audit total dump limits, manifest file lifetimes, logical cell position alignment, supported tensor layouts, stream synchronization and multi-GPU ownership. Ensure GDN capture is truly opt-in and cannot force different MTP choices.

## B. Reproduce 32K natural MTP acceptance/rollback with GDN + logits

1. Three fresh separate bounded runs: dense A, dense B and sparse with Report 0018's Qwen3.8-Flash-Next IQ3_XXS target/MTP Q8, CUDA0/ROCm0/ROCm1 4/7/7 layers, F16 KV, 196608 context, max draft3, seed42 and 32K prompt/64 sampled tokens. Use distinct empty external GDN directories (LLAMA_QSA_GDN_DIR) and logits directories (LLAMA_QSA_LOGITS_DIR) on each run. Use GGML_QSA_MASK_AUDIT=1000 in all to match graph-disabled diagnostics. Only sparse sets GQA12 flag.
2. Verify actual full prompt, candidate IDs, accepted/rejected decisions, logical checkpoints and target logits align across all three fresh processes. If they do not align, keep numerical status blocked; any forced schedule must be explicitly diagnostic-only.
3. Verify GDN snapshots cover pre_verification, post_verification, post_accept and post_rollback including subsequent target reprocessing. Capture logical row/rs_plane/pos before and after an actual rejection and both recurrent snapshot-based rollback and any checkpoint restore path that occurs. If a phase's state cannot be captured safely, specify exactly which one, fix it if feasible and do not claim equivalent state.
4. Compare dense A/B GDN manifests first with compare_qsa_model_state.py. Then compare dense A/sparse and report first differing layer/element/checkpoint, max absolute/relative error, NMSE, finite status per R/S/P type. Do not invent a blanket tolerance. Also rerun previous logits comparator to confirm same-history and bitwise dense-repeat controls.
5. Run compare_qsa_logit_distributions.py for dense repeat and dense/sparse logits. Report top-1 and top-5 changes, max/typical KL and JS divergence, probability of first differing top token, and examples with steps/positions. Quantify whether raw logit error implies meaningful softmax differences without presuming it is harmless. The observer's R/S/P state interpretation must be independently established before downstream inference is made.

## C. Decision and constraints

No long full-model 512-token throughput A/B in Iteration 0019 and **no default promotion**. A bounded 98K correctness-only replay is appropriate only after the 32K observer has proven reliable and its differences are characterized. GPU graph replay, mixed-sequence correctness and graph-safe overflow remain separate production gates. Do not weaken selectors, overflow checks, model math, scheduler semantics or acceptance behavior.

Use build-perf/28 jobs, bounded server startup/request deadlines and process-group cleanup. Do not use GitHub Actions/runs, PRs, merges, force pushes, privilege escalation, GPU resets/reboots, driver changes, nvprof, unbounded loops or abandoned servers. Preserve /home/chris/workspace/llama.cpp/, build-mixed/ and model weights. All raw GDN/logits binary data, manifests, logs, model prompts and runners belong **outside Git**, under /home/chris/workspace/llama-perf-results/iteration-0019/. Commit only code, tests, compact summaries and report.

## Deliverable

Write .ai/reports/report-0019.md with compilation/tests, independently checked logical recurrent-row semantics, dense-repeat vs sparse checkpoint-aligned GDN accuracy across accepted/rejected/reprocessed transitions, probability-level logits impact, explicit remaining numerical GO/NO-GO, and narrow next action. If remaining genuinely Partial/Blocked, identify precisely the unresolved state/phase or numeric issue without claiming success. Commit and push to docs/llama-perf.

Return:
Iteration: 0019 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0019.md | Summary: <1-2 sentences>
