# Handoff 0038 — finish offline dense TOP_K order audit on saved 0037 / 0018 captures

## Objective

The user requested an investigation of Report 0037's **1,962/2,048 dense A/B raw top-k rank-order changes** despite identical layer15 score bytes, pool/KV membership and model-level dense controls. Chat AI has already traced and explained the likely cause in source: for 8,256 pool scores × four query rows, the **HIP radix top-k** gather writes strictly-above-threshold picks to unordered positions using atomicAdd(), while threshold ties are handled in deterministic low-pool-ID order. The fused TOP_K_MASK consumes selected **pool IDs**, not rank order, and maps all selected pools/extra cells to KV mask positions, making the mask semantically set-based. Report 0018 independently noted the same dense order difference before the FP32 experiment.

**Remaining local-only milestone:** Run Chat AI's new fail-closed CPU analyzer on the **actual saved 0037 dense A/B QSA_TOPK_TRACE arrays** (and optional separately labeled 0018 dense A/B), verify the exact discrepancy is confined to the nondeterministic atomic prefix, and report a final evidence-based classification. NO new GPU run is needed or authorized.

## Repository and ownership

Repository civcode/llama.cpp-perf, branch docs/llama-perf; starting revision exactly as in the Chat AI transfer prompt. Read .ai/WORKFLOW.md, .ai/reports/report-0037.md, .ai/reports/report-0018.md, docs/LLAMA-PERF/QSA-TOPK-ORDER-INVESTIGATION.md and the source top-k.cu, top-k-mask.cu. Once user forwards handoff, Coding Agent has exclusive write ownership until report-0038.md is committed/pushed. Chat AI makes no concurrent repo changes.

## Already implemented

- **docs/LLAMA-PERF/tools/analyze_qsa_topk_order.py**: CPU-only, imports existing compare_qsa_topk_trace.read_log; compares exact raw bytes of score/blocks/extra maps, validates each selected index and the exact CPU reproduction of GPU uint32 score rank (including signed zero), tests lower-pool-ID cutoff-tie policy, computes difference counts in the unordered atomic prefix versus deterministic tie tail, checks mapped KV sets, returns bounded summary JSON. Output must be a fresh external file; raw captures remain unmodified outside Git. This tool **fails closed** if scores/maps/membership/ties differ.
- **docs/LLAMA-PERF/tools/test_analyze_qsa_topk_order.py**: 14 synthetic CPU-only tests, including realistic 8,256-pool × four-query shape, passed locally in Chat AI's temporary container copy. Local integration against existing comparator parser and original workstation data is still required.
- **docs/LLAMA-PERF/QSA-TOPK-ORDER-INVESTIGATION.md** explains the source-level mechanism and why rank-order variability alone should not change the fused sparse mask. No GPU arithmetic or production selector code was changed in this investigation.

## Offline-only tasks

1. Verify exact branch/commit and inspect the local source. Run Python compileall and the 14 new unit tests; also run the existing focused compare_qsa_topk_trace tests. Fix ordinary parser/import/array-data issues in the utility, not the production GPU kernel, if needed. NO GitHub Actions.
2. Locate saved **0037** raw dense A and dense B server logs that include **QSA_TOPK_GROUP/QSA_TOPK_PART** records and their referenced raw score/top-k/blocks/extra binaries in /home/chris/workspace/llama-perf-results/iteration-0037/. Check model/fixture and group identity from Report 0037; do not guess log paths or mix cohorts. Preserve all originals.
3. Run analyzer with both valid 0037 logs, choose a NEW external JSON output path, e.g. /home/chris/workspace/llama-perf-results/iteration-0038/dense-topk-order-0037.json. Confirm groups, four query rows, n_pool=8256, k=512 and **1,962/2,048 raw order mismatches** for the relevant first layer15 group. If multiple groups are in the captures, report separately the first-group tally and full analyzed aggregate; don't misleadingly equate the two.
4. **Decisive checks:** (a) dense A/B score/blocks/extra bytes equal; (b) both selected sets equal the CPU reference exact top512 with deterministic low-pool-ID boundary ties; (c) identical mapped KV sets and any separately saved layer15 mask bytes; (d) changed raw positions ONLY in the unordered above-threshold atomic prefix, **zero** in the deterministic tie tail; (e) whether the result lists are actually score sorted (they needn't be). Report each query's changed-order count, number of threshold ties, above-cutoff count and descending-score inversions. This distinguishes expected atomic ordering from a genuine selection or tie policy defect.
5. Optionally run a separately labeled audit on 0018 dense A/B raw captures, if still available, to confirm the phenomenon predates the FP32 specialization. Do not merge different groups, models or token histories. If those files are missing, mark historical report 0018 as previous aggregate evidence only.
6. If the tool fails because of mismatched raw scores, mappings, membership or a nonzero tie-tail difference: **do not force a passing interpretation**, correct only a confirmed parser bug or explain the counterexample and source line. Do not run GPU tests to resolve it in this iteration. Note that duplicate writes to the same mask cell write the same source value but can be a separate concurrency issue; do not claim general race freedom or graph safety.
7. Report the exact conclusion and what is not proved: Expected HIP radix ordering is not a QSA mask/accuracy defect on unchanged inputs; it is NOT the explanation for CUDA Xid79, nor proof of numerical equivalence, graph correctness or sparse release tolerance. No modifications to production top-k sorting, sparsity defaults, PCIe/driver settings.

## Hard boundaries

**Zero CUDA, HIP, ROCm, nvidia-smi, device probing, llama-server, model inference, tracing, stress, GPU resets, reboots, hardware changes, sudo, GitHub Actions.** This iteration is entirely CPU and existing external artifact reads. Do not put raw tensors, model paths, prompts, system hardware identifiers, full telemetry/logs or private data in Git.

Write .ai/reports/report-0038.md with concise actual per-query findings, exact source revision and test result, and a clear pass/fail separation between raw order and semantic selection. Commit/push only small utility fixes and report to docs/llama-perf.

Return:
Iteration: 0038 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0038.md | Summary: <1-2 sentences>
