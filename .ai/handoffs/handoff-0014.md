# Handoff 0014 — verify separately gated RDNA4 Q4/GQA12 sparse QSA

## Objective

Validate or reject the newly wired, **disabled-by-default** GQA12 variant of Q4 sparse flash attention on the actual two-R9700 Qwen3.8-Flash-Next workload, starting with synthetic CPU-reference operators and progressing to real-model checks **only if safety and correctness permit**. The prior GQA4-only kernel passed synthetic tests but could not activate on this model: Report 0013 measured `24 Q heads / 2 KV heads = GQA12`. Keep production behavior unchanged and do not circumvent correctness/overflow gates to gain speed.

## Repository and required reading

- Repo: `civcode/llama.cpp-perf`; branch: `docs/llama-perf`; starting commit is the exact transfer commit in the Chat AI message. Work under exclusive Coding Agent ownership, and return it after report/changes are pushed.
- Read `.ai/WORKFLOW.md`, Reports 0011–0013, `docs/LLAMA-PERF/QSA-SPARSE-RDNA4-AUDIT.md`, and benchmark setup in Reports 0008–0010.
- Prior evidence: at 32K/98K, 2652/2665 **real dense-path** mask groups passed per-query bound 2051 and Q4 union capacity 8204, with max observed unions 5493/5863. This was **graph-disabled, one request/slot**, not replay or multi-sequence proof. Q4 sparse compactor overflow now fails closed through a **host stream synchronization before gather**; this incurs overhead and disables graph replay for the experiment.
- Source audit: `ggml/src/ggml-cuda/fattn.cu` picks ncols2=4 on RDNA when GQA12 is divisible by four, then `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` picks ncols1=4 at Q=4. `ggml/src/ggml-cuda/fattn-common.cuh::launch_fattn` uses `ntiles_z_gqa=ceil(GQA/ncols2)`: expected **three GQA-head tiles per KV head** at GQA12, sharing each Q4 group's sparse index union. This is a source-level hypothesis, **not a GPU-verified equivalence**.
- New `GGML_HIP_QSA_SPARSE_Q4_GQA12=1` is independent from original `GGML_HIP_QSA_SPARSE_Q4=1` (GQA4-only), and the older `GGML_HIP_FATTN_FORCE_WMMA_Q4` should remain unset. Do not broaden either original guard.
- Chat AI added `ggml/src/ggml-cuda/fattn-sparse-q4-gqa12-tune.h`, strict host selector test `docs/LLAMA-PERF/tools/test_fattn_sparse_q4_gqa12_tune.cpp`, new flag branches in `fattn.cu`, GPU graph disabling for the GQA12 experiment in `ggml-cuda.cu`, and test/perf fixtures `GGML_TEST_QSA_GQA12_FA=1`. At 32768/98304 KV, fixtures have a 2048-position union (`wide_union=0`) or a **5120-position four-query union** (`wide_union=1`) with 2048 visible/query and bound 2051. The test-mode `GGML_TEST_QSA_GQA12_BOUND_FA=1` covers 8191/8192/8193 selected union counts. Chat AI's local strict C++17 selector check and independent cardinality checks passed, but **HIP/CUDA and GPU correctness have not yet been verified**.
- New `GGML_PERF_DISABLE_CUDA_GRAPHS=1` is an opt-in graph-off dense control (affects CUDA and HIP) to isolate the graph suppression overhead; no production baseline should silently inherit it.

## Ordered tasks

### A. Build, verify exact dispatch, and reject invalid outputs

1. Sync starting revision. Run both strict C++17 selector tests using `-std=c++17 -Wall -Wextra -Werror -pedantic`, then `cmake --build build-perf -j 28 --target test-backend-ops llama-server llama-bench` on mixed CUDA/HIP. Resolve compilation issues without weakening either GQA guard or changing normal dispatch. Verify GQA4 original selector and defaults remain unchanged.
2. For both ROCm0/ROCm1, both KV=32768 and KV=98304, and **both union widths** (2048/5120), run `GGML_TEST_QSA_GQA12_FA=1` with GQA12 sparse flag *unset* vs exactly `1` in separate clean processes, one matching fixture per filter. For example `./build-perf/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=98304.*qsa_gqa12=1.*wide_union=1'`. Confirm full CPU-reference comparison: finite outputs, finite NMSE <= 5e-4. Compare actual dense vs sparse outputs, not just each result in isolation if an additional direct comparison is easy. Both original sparse/Wmma flags must be unset.
3. Trace kernel dispatch on both AMD devices at both KV depths for representative narrow/wide cases. Confirm `flash_attn_mask_to_sparse_indices<4,...>`, sparse `flash_attn_ext_f16<256,256,4,4,...true>` and fixup, plus the actual head-tile geometry. Prove all 24 query heads are processed, including all three 4-head GQA tiles for both KV heads, and mask semantics remain per-query. Compare against true flag-unset dense tile path. If no sparse dispatch or head coverage is incomplete, **stop before model work**.
4. Run GQA12 8191/8192/8193 union tests under `GGML_TEST_QSA_GQA12_BOUND_FA=1`: valid cases must pass the same CPU-reference gate; the intentionally overbound 8193 case must fail explicitly **before gather** (as in Report 0013). Confirm flag-unset dense cases pass. Audit the existing fail-closed check against GQA12 and graph suppression. Do not weaken overflow tests or suppress aborts to make runs complete.

### B. Device-local performance including all overhead

1. If A passes, run `GGML_TEST_QSA_GQA12_FA=1` in perf mode, whole-op inclusive of mask compaction, sync/copy, sparse gather and fixup, at both 32K/98K and both union widths. Alternate at least three short dense/sparse **separate process** pairs per AMD GPU. Use the actual GQA12 flag only; keep all other HIP tuning flags unset. Provide medians/ranges, counts and any outliers. The prior Report 0012 speedups are invalid comparisons after the Report 0013 sync change.
2. If convenient, use bounded ROCprofiler only to confirm dispatch and timing attribution, separately from unprofiled A/B. The compactor scans full KV while sparse WMMA work may be three head tiles per KV head.
3. If sparse is slower at 98K on either card or numerically unreliable, **no full-model sparse benchmark**. Report the failure phase and a narrow next direction: graph-safe overflow handling, faster compaction, or different gather strategy. Never bypass safety checks simply to produce a speedup.

### C. Only if A and B are good: staged full-model correctness, then performance

1. Begin with a short bounded opt-in safety/correctness probe on the same Qwen3.8-Flash-Next IQ3_XXS+MTP Q8, CUDA0/ROCm0/ROCm1 4/7/7 split, 196608 context and max-draft3 as Reports 0008–0010. Verify **real** GQA12 sparse launches on both R9700s, no union overflow, matching accepted/drafted trajectory and fixed-seed output. Capture coverage of varying accept/reject/rollback events. GPU graph replay is still deliberately disabled for sparse; **do not claim** its correctness from these tests. If the requested model run fails, stop and diagnose.
2. Establish fixed-token/teacher-forced logits and Gated DeltaNet recurrent-state comparisons through multiple verification/acceptance/rollback transitions before interpreting any model performance as a production candidate. Report numerical tolerance, observed differences, and where direct state comparisons are unavailable. If this cannot be established, mark model-level correctness blocked rather than treating sampled text hashes as sufficient.
3. Only after the required correctness gate, run matched uncached 32K and 98K 512-output-token A/B requests with identical seed, sampling, model split and settings; use fresh separate server processes, alternate short runs and keep logs raw outside Git. Compare **three conditions**: (i) true production-default dense (graph enabled), (ii) dense with `GGML_PERF_DISABLE_CUDA_GRAPHS=1`, and (iii) sparse with GQA12 flag=1 (which disables GPU graph replay). Measure complete hashes, matched draft/acceptance history, tokens/s and per-step/phase medians. Graph-off dense vs sparse isolates graph suppression, whereas true-default dense vs sparse determines current end-to-end benefit. A graph-off win by itself is **not** a production speedup. No default promotion in this iteration.
4. Do not add background/unbounded tests. If mixed-sequence/replay correctness remains unverified, identify it as a separate promotion blocker even if a short single-slot run succeeds.

## Constraints

- No GitHub Actions, PRs, merges, force pushes, upstream submissions, privileged changes, GPU resets/reboots, driver changes, unsupported profilers, `nvprof`, 5-hour timeouts or orphan processes.
- Preserve `/home/chris/workspace/llama.cpp/`, `build-mixed/`, and model weights. Use `build-perf` and 28 parallel build jobs. Raw logs under `/home/chris/workspace/llama-perf-results/iteration-0014/`.
- Use bounded health preflights, server startup/request deadlines and process-group cleanup as in prior reports.
- Do not change scheduler synchronization, model semantics, KV formats, MTP draft/acceptance parameters or default kernel dispatch. Keep all debug and sparse flags opt-in.
- A genuine partial/blocked safety or performance gate is a valid result. Do not manufacture a passing full-model benchmark.

## Report and completion

Write `.ai/reports/report-0014.md` including exact commit/toolchain, HIP/CUDA build, selector and original-guard regression, 32K/98K both-width GQA12 correctness/NMSE, trace-confirmed tiles/head coverage, overflow tests, whole-op perf inclusive of host sync and outliers, and (if permitted) gated full-model correctness/performance, with explicit go/no-go. Commit/push the report and necessary fixes. Return:

`Iteration: 0014 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0014.md | Summary: <1-2 sentences>`
