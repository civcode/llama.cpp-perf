# Handoff 0022 — fixed-reference predictive-fidelity pilot for QSA GQA12

## Goal

With the recurrent observer largely validated in Reports 0020–0021, run a **small, deterministic, teacher-forced target-only** experiment to measure the effect of experimental sparse QSA on reference next-token negative log likelihood (NLL), rather than inferring quality from bitwise identity or just max logit error. Verify the actual experimental **Q=4 GQA12** kernel executes. Gather evidence only: **no automatic numerical pass criterion, product tolerance, default sparse promotion or long full-model throughput A/B**.

## Repository and ownership

Repository: civcode/llama.cpp-perf. Branch: docs/llama-perf. Starting revision: **the exact commit in the Chat AI transfer prompt**. Read .ai/WORKFLOW.md, .ai/reports/report-0019.md through report-0021.md, docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md and docs/LLAMA-PERF/QSA-SPARSE-NUMERICAL-FIDELITY.md. Coding Agent has exclusive write ownership after the user relays the transfer prompt; commit/push Report 0022 before returning.

Report 0021 mixed HIP/CUDA build and all 56 Python tests passed. It validated real serializer-vs-observer R/S/P byte equality on CUDA0/HIP0/HIP1 selected layers through a natural FULL checkpoint restore/replay; pre-checkpoint/post-restore bytes matched on layer 1. A true seq_cp/find_slot two-sequence fixture confirmed staged cross-cell src0=0 feeding seq1 cell1 while stable serialized src=1, and checked same-token/different-token continuations without donor corruption. Runtime bad-row injection and nontrivial stable serialized src remain untested; the latter is normally normalized by find_slot.

These validations increase confidence in the previously **measured** sparse numerical drift: in Reports 0017–0019, dense-repeat was bit-identical, layer 11 had identical Q/K/V/masks but different F32 attention, layer 15 top-k changed at near ties, GDN R/S diverged in 2592/3504 rows and naturally aligned full-vocabulary logits differed in 64/64 rows (max abs .9186411, max row NMSE .00462375, KL max .0106312 nats, zero top-1 flips in that limited sample). The sparse kernel was locally closer to FP64 on layer-11 NMSE, but this does NOT prove model fidelity. Numerical gate remains NO-GO.

## Chat AI implementation now committed — needs local validation

1. examples/qsa-reference/qsa-reference.cpp, examples/qsa-reference/CMakeLists.txt and examples/CMakeLists.txt register **llama-qsa-reference**, a manually invoked C++ target-only reference runner using the repo's standard common_params CLI and llama batch/process APIs. It reads fixed reference text from LLAMA_QSA_REFERENCE_TEXT_FILE, requires a **fresh existing** LLAMA_QSA_REFERENCE_OUT_DIR, and captures LLAMA_QSA_REFERENCE_ROWS=4..64 in multiples of four (default32). It tokenizes the **same fixed text** under the target vocab and feeds it deterministically without MTP, sampling or modifying target math.
2. It pre-fills the earlier context in bounded chunks, then decodes the last 4/8/.../64 input positions as **four-token query groups**, requesting all four causal F32 target output logits. This is essential: **single-token decode would not exercise the Q4 experimental sparse kernel**. For each scored position, it emits the full F32 vocab row, strict JSONL event with target token, position, per-prefix FNV64 input fingerprint and group-first position, and an independent next-reference-token label from the text at the next input position. It produces manifest.jsonl, references.jsonl and logits-N.f32 files under the external directory. Every reference label has origin=teacher_forced_reference and must match the actual causal logit row, not a sampled token.
3. docs/LLAMA-PERF/tools/compare_qsa_reference_nll.py and test_compare_qsa_reference_nll.py (11 synthetic host tests) compare dense and sparse manifests only when every event/token/candidate/position matches and each row has an independently specified reference label. Report signed sparse-minus-dense NLL, reference ranks, top-1 hit counts, worst and median NLL changes and examples. There is **no pass/fail quality threshold**; the evaluator reports evidence, not production correctness.
4. docs/LLAMA-PERF/QSA-SPARSE-NUMERICAL-FIDELITY.md describes separating single-op FP64 fidelity, fixed-token predictive fidelity and natural MTP/task fidelity. Existing tools compare_qsa_logit_distributions.py and compare_qsa_model_state.py can operate on the same run manifests.
5. Updated audit notes. All new source, CMake and Python files were committed and pushed but have **not been compiled or executed in Chat AI's GPU-less workspace**.

## Task A — local build/tests and ensure genuine Q=4 sparse activation

1. Run Python syntax checks and new test_compare_qsa_reference_nll.py. Run relevant existing suites; fix ordinary bugs autonomously. Compile with the established mixed ROCm/CUDA toolchain, including new target:
   cmake --build build-perf -j 28 --target llama-qsa-reference llama-server test-backend-ops llama-bench
   Reconfigure existing build-perf if the newly registered example target is not found. Avoid changing drivers, build-mixed or model weights.
2. Review the new runner for correct llama_batch_ext_add_token, set_pos, set_output_logits, llama_process and llama_get_logits_ith use. Verify that each four-query group's row i really predicts the next token at the stated reference position, with causal masking. Validate a tiny known token sequence before long-context use. Detect malformed prompts, size/IO errors, duplicate manifests and mismatched tokenization. Do not silently fall back to Q=1.
3. Verify graph-disabled dense A/dense B and experimental sparse process actually reach the Q4/GQA12 path. Set GGML_QSA_MASK_AUDIT=1000 on all and only sparse sets GGML_HIP_QSA_SPARSE_Q4_GQA12=1; the legacy GGML_HIP_QSA_SPARSE_Q4 and GGML_HIP_FATTN_FORCE_WMMA_Q4 flags stay **unset**. Inspect mask audit and backend/model layer trace. A successful command with the sparse selector but no Q=4 QSA invocation is **not a valid optimization-fidelity test**; fix fixture/batching rather than making claims.
4. Confirm all flags unset yields the original no-diagnostic server behavior and that the reference example is never automatically run.

## Task B — small three-process fixed-token reference experiment

1. Reuse the same IQ3_XXS Qwen3.8-Flash-Next target, F16 KV, context and three-device split 4/7/7 when appropriate; do NOT load an MTP draft model in this controlled target-only measurement. Choose a short reference text and a *relevant* long-context fixture from local available data; record provenance and explicitly mark any non-held-out fixture as exploratory. Do not invent a benchmark dataset or label existing sampled MTP logits as teacher-forced.
2. Use an external local folder under /home/chris/workspace/llama-perf-results/iteration-0022/. For each fixture run **fresh** dense A, dense B, sparse with identical text/tokenization and target CLI settings, unique empty output dirs, LLAMA_QSA_REFERENCE_ROWS=32 (or smaller divisible by four). Graph disabled in all three for operator comparability. Pre-existing model weights, build folders and unrelated files must be preserved.
3. Verify references.jsonl from dense A and sparse are identical, and that all F32 event and prefix fingerprint records align; compare dense A/B first. Example use:
   python3 docs/LLAMA-PERF/tools/compare_qsa_reference_nll.py dense-A/manifest.jsonl dense-B/manifest.jsonl dense-A/references.jsonl
   python3 docs/LLAMA-PERF/tools/compare_qsa_reference_nll.py dense-A/manifest.jsonl sparse/manifest.jsonl dense-A/references.jsonl
   Optionally use compare_qsa_logit_distributions.py on each pair to report top-1, top-5, KL and JS changes for the same deterministic rows.
4. Report exact source of reference tokens and tokenizer/model revision, number of scored rows and Q4 kernel activations, dense-repeat numerical variance, sparse-minus-dense mean/median/max/worst reference NLL, reference token rank/accuracy, top-token and softmax divergence. Inspect representative changed rows around actual QSA mask selections. If small short-context fixture has no sparse activation, do not count it as a sparse fidelity test.
5. This exploratory sample is **not** a statistical product-quality acceptance study. Distinguish a genuine fixed-token teacher-forced schedule from natural MTP verification and acknowledge that differences in R/S, GQA Q4, graph replay and MTP acceptance are separately measured/blocked.

## Task C — decision and next research choice

- Retain **NO-GO** for sparse numerical equivalence and production promotion, regardless of whether a 32-row reference NLL pilot improves. No user-approved product tolerance exists yet.
- If reference NLL shows a material or systematic deterioration, propose a **reversible opt-in sparse arithmetic variant closer to dense accumulation**, with follow-up headwise FP64 and fixed-token regression, rather than weakening numerical thresholds. If small-sample NLL improves or is neutral, propose a **larger held-out model/task-quality calibration** with agreed statistical measures and explicit user approval of acceptance thresholds. In either case do not infer equality from no top-1 changes.
- Do not perform long 512-token full-model throughput A/B or 98K correctness replay, GPU graphs, mixed production sequences or a default sparse promotion. Keep fail-closed union overflow host check and all original selector/mathematical guards.
- No GitHub Actions, PRs, merges, force pushes, privileged operations, driver changes, GPU resets, unsupported profilers or orphan servers. Use bounded startup/request deadlines and process-group cleanup; 28 build jobs. Keep all raw prompts, tokens, weights references, logits and probe logs outside Git.

## Deliverable

Write .ai/reports/report-0022.md with compilation and tests, actual verified four-query sparse activation, exact labelled next-token indexing, three-run small deterministic reference NLL and distribution metrics, explicit limitations and narrow justified next task. If no valid sparse Q4 run can be obtained after reasonable autonomous debugging, return Partial with the precise blocker; do not report a meaningless zero-difference result as evidence for sparse quality.

Commit and push code/test fixes and report to docs/llama-perf, then return:

Iteration: 0022 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0022.md | Summary: <1-2 sentences>
