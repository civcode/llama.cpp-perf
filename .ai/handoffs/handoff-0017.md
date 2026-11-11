# Handoff 0017 — exact layer-11 QSA CPU reference and model numerical correctness

## Objective

Identify whether the dense tile or opt-in Q4/GQA12 sparse flash-attention path is more accurate on **the exact identical Q/K/V/mask inputs at layer 11**, determine why a different layer-15 top-k selection follows, and establish bounded fixed-token logits and actual Gated DeltaNet recurrent-state comparisons across speculative acceptance and rollback. Numerical correctness, not throughput, is the milestone. No full-model performance A/B until the gate is met.

## Starting point and evidence

- Repository: civcode/llama.cpp-perf; branch: docs/llama-perf; start from the exact transfer commit in the Chat AI message. Take exclusive Coding Agent ownership. Read .ai/WORKFLOW.md, Reports 0014–0016, and docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md.
- Report 0016: in the final four-token prompt batch **before any MTP drafting**, the first changed F32 attention result was HIP0 layer 11. Dense vs opt-in sparse had byte-identical Q, K, V spans and F16 mask, union=3537. All 24576 output values differed, max absolute 0.00535035, dense-energy NMSE 8.97579e-7. Layer 15 subsequently selected different KV blocks. Dense A and B reproduced all 273 F16 masks and F32 outputs bitwise. These facts prove a kernel-path numerical difference but **not which path is correct**, its downstream impact or GDN/logit equivalence.
- Separate GQA12 selector: GGML_HIP_QSA_SPARSE_Q4_GQA12=1 on sparse only. Both GGML_HIP_QSA_SPARSE_Q4 and GGML_HIP_FATTN_FORCE_WMMA_Q4 remain unset. Q4 sparse compactor still fails closed on overflow with a host stream sync, and experimental HIP graphs remain disabled.
- Chat AI just added **unverified on-device** source changes:
  - In ggml/src/ggml-cuda/fattn.cu, the existing first HIP0 layer-11 Q/K/V dump now logs QSA_INPUT_META: precise attention scale, max_bias, logit_softcap and sink presence, with layer/device/invocation ID.
  - docs/LLAMA-PERF/tools/replay_qsa_captured_attention.py requires NumPy and replays real strided Q(F32)/K,V(F16) and captured F16 mask through an FP64 scaled masked-softmax attention reference. It rejects differing Q/K/V/mask bytes, wrong shapes, invalid scalars and unsupported ALiBi/softcap/sinks. It reports dense/reference, sparse/reference, and dense/sparse errors including per-head NMSE and max absolute error.
  - docs/LLAMA-PERF/tools/test_replay_qsa_captured_attention.py contains six synthetic tests; source has not yet been executed in the agent's environment. Chat AI ran only a small independent NumPy FP64 attention computation locally.
- Source documentation: docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md.

## Task A — CPU reference on the real first-divergent attention operation

1. Sync the handoff commit and run local Python tests: test_replay_qsa_captured_attention.py, test_compare_qsa_mask_dumps.py and test_compare_qsa_attn_out.py plus Python syntax checks. Use local NumPy or an isolated local environment; do not change project-wide dependency setup. Fix all ordinary test/parsing/stride bugs autonomously.
2. Build mixed HIP/CUDA using cmake --build build-perf -j 28 --target llama-server test-backend-ops llama-bench. Verify the new QSA_INPUT_META line is properly emitted for first HIP0 layer 11 only when the input capture flag and mask audit are enabled. Flag-unset behavior must be unchanged.
3. Repeat the bounded 32K/64-token Qwen3.8-Flash-Next IQ3_XXS + MTP Q8 fixture from Report 0016 (CUDA0/ROCm0/ROCm1 4/7/7, context 196608, F16 KV, max draft3, seed42) with **fresh** dense A, dense B, sparse processes. Enable on all three: GGML_QSA_MASK_AUDIT=1000, GGML_QSA_MASK_AUDIT_DIR, GGML_QSA_MASK_DUMP_LIMIT=256, GGML_QSA_INPUT_DIR, GGML_QSA_ATTN_OUT_DIR, GGML_QSA_ATTN_OUT_LIMIT=256. Use distinct external directories. Only sparse adds GGML_HIP_QSA_SPARSE_Q4_GQA12=1. GPU graphs are disabled and timing is not meaningful.
4. Compare dense A vs B and A vs sparse using replay_qsa_captured_attention.py. Verify identical actual Q/K/V and mask, metadata, scale and tensor strides. Report **per-head dense-vs-FP64 and sparse-vs-FP64 errors** and the first bad head/feature; do not infer equivalence from an NMSE number alone. If the simple scaled attention reference does not match model semantics, stop and make the reference exact rather than fudging expected data.
5. Determine whether one implementation is unexpectedly inaccurate, or both differ from high precision through expected FP16 accumulation. If a real sparse kernel defect exists, fix and test it without changing production defaults; rerun targeted GQA12 CPU-reference tests on **both** R9700s at 32K and 98K, including all 24 heads and 8191/8192/8193 overflow boundaries. Prefer a small reproducible captured-input regression instead of repeated full-model captures once the problem is localized. Do not relax any gate.

## Task B — prove or refute top-k near-tie propagation

1. At layer 15, capture indexer_score just before ggml_top_k, together with actual selected pool IDs and enough metadata for mapping a pool ID to the observed four-token KV block. The construction is in src/models/qwen4exp.cpp::build_qsa_sel. Make this a bounded, disabled-by-default diagnostic with layer/phase/sequence/position identity, correct stream synchronization, and no numerical/scheduler changes.
2. Compare scores and ordering dense A/B vs A/sparse; report the top-k cutoff, score margin between last included and first excluded pool, near-tie count, first changed selected pool and associated KV indices. Distinguish a tiny score perturbation that crosses a near tie from an incorrect indexer/cache state. The fact that attention layer 11 differs does not alone prove it caused the layer-15 selection change.
3. Establish the logical step and layer chronology from verified trace metadata. A per-device mask ordinal is not a cross-device clock and the old node_NNN labels are not stable layer names.

## Task C — fixed-token logits and actual GDN state through MTP

1. Implement a **disabled-by-default bounded** diagnostic that captures all relevant *actual token IDs* and feeds matched predetermined tokens to dense and sparse target verification. Capture pre-sampling logits with sequence, position, target step and exact draft IDs. If a verification acceptance schedule must be forced, mark it diagnostic-only rather than claiming normal speculative correctness.
2. Inspect and capture relevant *logical Gated DeltaNet recurrent-state values* rather than only sampled token hashes or draft counts. Use src/llama-memory-recurrent.cpp state_write/state_write_data row and rollback-snapshot semantics plus the server checkpoint logic where safe. Bound the copies: **do not serialize the entire long-context KV cache**. Include cell owner, sequence, layer, position and dtype. Synchronize device reads only under diagnostic flags.
3. Compare dense A/B baseline and dense/sparse after the final prompt, before verification, after at least one fully accepted and one rejected/rollback/reprocessing MTP transition with **identical logical token/cache positions**. Report finite status, per-layer/global NMSE, max absolute and relative error, and first divergent element. Justify numerical tolerances from observed dense-repeat stability and CPU-reference context; do not assume acceptance from sampled text alone.
4. After 32K correctness is understood, a bounded 98K numerical replay is appropriate. GPU graph replay, mixed sequences and graph-safe overflow remain **separate production promotion gates** even if correctness here passes.

## Constraints

- **No three-condition 512-token full-model performance A/B in this iteration.** No default promotion; the strong Report 0014 operator-speedups are not model-speedup evidence.
- No GitHub Actions/runs, PRs, merges, force pushes, driver changes, GPU resets, privileged commands, unsupported profilers, nvprof, unbounded runs or abandoned processes.
- Preserve /home/chris/workspace/llama.cpp/, build-mixed/, model weights. Use build-perf with 28 build jobs; bounded server health/startup/request deadlines and process-group cleanup. Keep raw masks/Q/K/V, logits, scores and GDN state in /home/chris/workspace/llama-perf-results/iteration-0017/ **outside Git**.
- Resolve reasonable build/test/instrumentation problems autonomously. Return Partial/Blocked if required numerical/state evidence remains genuinely inaccessible, and never claim that an unmeasured gate passed.

## Reporting

Write .ai/reports/report-0017.md with build/tests, FP64 reference verdict by head, relevant fixes and GPU regression evidence, layer-15 indexer score margins/changed selected IDs, fixed-token logits and GDN-state comparisons through acceptance and rollback, exact limits, and explicit numerical GO/NO-GO. Commit and push all changes to docs/llama-perf.

Return: Iteration: 0017 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0017.md | Summary: <1-2 sentences>
