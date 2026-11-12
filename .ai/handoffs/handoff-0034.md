# Handoff 0034 — offline FP64 headwise decomposition before changing sparse kernel arithmetic

## Objective

Report 0033 completed the matched five-condition 32K ablation. The key intervention proved that enabling sparse only on layer 11 is sufficient to reproduce the first layer-15 top-k selection and layer-12 GDN R/S drift, but omitting layer 11 does **not** remove all later logit drift. More importantly, on identical layer-11 Q/K/V/mask, the sparse output had *lower* normalized squared error against the FP64 mathematical reference than dense. We must NOT blindly edit sparse arithmetic to match dense. The user authorized investigating the numerical differences, not changing a production quality tolerance.

Next coherent milestone: **CPU-only, head-by-head and query-by-query reference accuracy analysis** of the raw capture data that already exist. Identify whether sparse is closer overall but worse in selected heads/positions, and which arithmetic hypothesis is most worthwhile to examine. Prepare an evidence-based recommendation for a precisely scoped optional precision variant, or explicitly explain why no kernel change is yet justified. **Do not run GPU inference or implement a kernel change in this iteration**; use existing data and code inspection.

## Repository and ownership

Repository civcode/llama.cpp-perf; branch docs/llama-perf; starting SHA exactly as in Chat AI's user-relayed transfer prompt. Read .ai/WORKFLOW.md and Reports 0016, 0017, 0018, 0019, 0033, and docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md. Coding Agent owns repository changes exclusively after transfer; Chat AI stops writing until Report 0034 returns.

Relevant Report 0033 facts:
- Five sequential 32K matched runs with exact dense A/B controls, aligned natural histories, real HIP sparse Q4 launches per expected layer, no observed GPU fault on PCIe Gen3 x4.
- All/only11 first-layer-11 output differs from dense by max abs 0.0184302 and dense-energy NMSE 1.05615e-6 on **identical QKV/mask**; except11 is identical to dense.
- FP64 reference layer-11 NMSE: **dense 1.18147e-6**, **sparse 3.68239e-7**. No mathematical or quality acceptance threshold is established.
- Top-k selection changed in 3/4 layer-15 queries in sparse-all/only11; layer-12 GDN R/S changed 12/12 in those conditions but not except11. All three sparse conditions changed 34/34 verification logits rows; no top1 flips.
- The 0016–0019 results were separate older cohorts. Do not combine their raw rows as if one matched process history.
- Original model/device settings and raw 0033 captures are outside Git at /home/chris/workspace/llama-perf-results/iteration-0033/. Keep them there, unchanged.

## Chat AI implementation (committed, not run on agent host)

1. New **docs/LLAMA-PERF/tools/analyze_qsa_headwise_precision.py**: CPU-only reference comparison for **three exact captured HIP0 layer-11 Q4 operations** (fresh dense A, dense B, one sparse variant). Imports the previously validated replay_qsa_captured_attention.py functions; requires matching layer, scale, audited KV and shape, and **bitwise identical Q/K/V spans and F16 masks** across the three; requires byte-identical dense A/B F32 output. Builds FP64 scaled attention using the actual captured tensors/mask, reports dense and sparse squared-error and NMSE, max-abs errors for all **24 heads × four query positions**, counts sparse/dense winners for heads and 96 query-head pairs, and leaves policy NO-GO.
2. New **docs/LLAMA-PERF/tools/test_analyze_qsa_headwise_precision.py** (16 synthetic tests) exercises query/head aggregation, mixed winners, ties, mismatched shapes, NaNs, zero-energy, raw QKV and signed-zero-mask bitwise mismatches, raw dense A/B signed zero difference, metadata mismatches and duplicate files.
3. Updated QSA-SPARSE-TASK-QUALITY.md documenting Report 0033 outcomes and why matching dense is not necessarily an accuracy improvement. No production numerical kernel, selector, GPU graph, model, CUDA/HIP driver, sparse default or hardware configuration was changed.
4. A bounded pre-read size check charges the three raw copies of each Q/K/V input to 512 MiB before loading. The existing FP64 replay performs mathematical reference attention, not hardware WMMA rounding emulation. This is descriptive analysis, not GPU arithmetic source diagnosis or regression acceptance.

## Tasks — offline only

- Run Python compileall, the direct new analyzer tests, existing test_replay_qsa_captured_attention.py and focused QSA test discovery. Fix ordinary import, shape, memory-bound or numerical issues, preserving exact dense/mask requirements and no silent fallbacks. No GitHub Actions.
- Resolve actual saved **0033** dense A, dense B, sparse-all, sparse-only11 (and if available sparse-except11) server log/capture filenames, and run the analyzer in separate fresh external output files for dense A/B vs sparse-all and dense A/B vs sparse-only11. The two sparse sets should have identical layer-11 outputs if prior evidence is correctly aligned, but verify this rather than assume. If sparse-except11 layer11 data are present, check it is bit-identical to dense and compute the same reference statistics as a negative control; do not mislabel dense fallback as actual sparse launch.
- Compare each of the 24 head-level NMSE and max-abs results; report winner counts, which heads/queries show the **largest relative loss of reference accuracy under sparse**, and whether aggregate sparse improvement is driven by many heads or a few large ones. Choose robust values (absolute/nmse) and avoid misleading near-zero relative error. Distinguish per-head NMSE from per-query/head NMSE and 96 correlated samples from independent observations.
- If exact captured QKV or mask bytes are not truly identical, FAIL CLOSED and describe this difference. Do not normalize or silently change inputs to force comparison; inspect actual source paths/hashes and use a valid matching triple. If 0033 raw tensor files are absent, optionally use existing 0017 exact captures as separately labeled fallback, but do not present that as the 0033 matched result.
- Inspect kernel source **ggml/src/ggml-cuda/fattn-mma-f16.cuh** around RDNA WMMA 256×256 tile configuration, half2 versus float VKQ accumulation and online softmax/combination order, and source gating in fattn.cu. Propose at most **one** precisely bounded *future* diagnostic-only arithmetic experiment, stating what accuracy metric (FP64 headwise), error/fallback behavior, build/trace validation and model-level drift outcomes would support or refute its benefit. Do not assume FP32 is always more accurate on all heads, or change kernel code without such evidence. A mathematical reference is not an arbitrary release acceptance tolerance.
- Preserve old captures, code and hashes. Raw tensors, model paths, prompts, GPU identifiers, full logs and reports remain outside Git. Commit only small utility fixes and aggregate evidence.
- Write .ai/reports/report-0034.md with direct local test results, exact capture provenance, per-head/position error summaries, how 0033 results compare with 0017 if useful, and an evidence-grounded decision on whether to attempt one diagnostic precision variant next. Explicitly mark **numerical equivalence NOT_ESTABLISHED and production NO-GO**.

## Hard boundaries

**NO GPU inference, kernels dispatched, tracing, stress testing, telemetry probes, reboots, resets, CUDA/HIP driver changes, PCIe Gen4/Auto changes, sudo, GPU graph enabled experiments, throughput tests, source model/kernel arithmetic changes, promotion of sparse default, GitHub Actions or forced PR/merge.** No raw benchmark data in Git. If required artifact bytes unavailable, do the available offline reasoning and report Partial or Blocked with precise limitations; do not fabricate metrics.

Commit/push completed host-side test fixes and report on docs/llama-perf and return:

Iteration: 0034 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0034.md | Summary: <1-2 sentences>
