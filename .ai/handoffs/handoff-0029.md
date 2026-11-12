# Handoff 0029 — safe Xid 79/154 incident continuation, offline preflight first

## Objective

Report 0028 is PARTIAL, not a sparse quality result. During the fourth sparse LongBench-v2 request CUDA0 reported unspecified launch failure and NVIDIA logged Xid 79 (GPU fallen off bus) and Xid 154 (node reboot required). nvidia-smi then failed to access the device. Root cause is unknown and must not be attributed to the HIP sparse path without independent evidence. Preserve the completed dense controls and frozen data. Run CPU-only integrity verification now. Proceed to a FRESH entire 24-question sparse quality comparison ONLY if an authorized human/operator has ALREADY restored the host and independently confirmed recovery. The Coding Agent MUST NEVER reboot, reset, or perform GPU recovery.

## Repository and ownership

Repository civcode/llama.cpp-perf, branch docs/llama-perf, starting SHA exactly as in Chat AI's handoff transfer prompt. Read .ai/WORKFLOW.md, Reports .ai/reports/report-0027.md and report-0028.md, docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md. Coding Agent has exclusive repo ownership after receiving the transfer. Commit/push the 0029 report and any small fixes before returning.

## Frozen 0028 facts (must not change)

- Official LongBench-v2 source (0027 local cached data.json), revision 12bc220a4e257d069e9005f2df762cac4755dc54; SHA-256 15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2.
- Prior 0027 task manifest SHA-256 d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1, 12 publisher IDs.
- Current 0028 task manifest SHA-256 ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5: 24 disjoint publisher IDs, seed20261116, context>=66,000 bytes, prompt<=138,000 bytes, chat 17,051–45,158 tokens, four source domains, source difficulty easy. No gold/prompt edit, truncation, or reselection.
- Dense A/B completed 24/24 byte-exact output/finish/token repeats; 14/24 scored correct. The first sparse process completed ONLY 3/24 (2 correct), with 60 real HIP GQA12 Q4 sparse kernel launches inside six natural target-verification groups across three requests; fourth request failed. The incomplete sparse file is historical diagnostics, NOT a subset that can be joined with fresh results. No complete 0028 sparse regression count or evidence bound exists.

## Chat AI implementation — CPU-only, not yet run on agent host

- New docs/LLAMA-PERF/tools/preflight_qsa_0028_resume.py requires the original external dataset, previous/current frozen manifests and both dense JSONL files. It validates pinned SHA256 digests, old/new source-ID disjointness, frozen 24-task MCQ identity, full dense A/B exact stability, 14 gold successes and sparse context floor. Optional --failed-sparse validates the historical *incomplete prefix* by exact task order and request hashes, leaving it explicitly unmergeable. Writes an external summary; never touches GPU, downloads data, starts inference or grants authorization.
- New docs/LLAMA-PERF/tools/test_preflight_qsa_0028_resume.py has 11 CPU-only synthetic regression tests for the pinned evidence and malformed sparse prefixes. Updated QSA-SPARSE-TASK-QUALITY.md with incident, limits and continuation policy. Neither new script has been executed by Chat AI; fix ordinary test issues locally. No backend numerical code, MTP, model settings or CUDA/HIP driver modified.

## A. Do offline work even if NVIDIA remains unavailable

1. Run local Python compileall and the direct preflight tests. Run the established LongBench/QSA Python tests if reasonable. No GPU activity needed. Correct host-tool issues autonomously.
2. Resolve actual file paths under /home/chris/workspace/llama-perf-results/iteration-0027/ and iteration-0028/; never overwrite or re-download the 465 MB dataset. Run the read-only pinned preflight, selecting a fresh external JSON output under iteration-0029/. Include the interrupted sparse JSONL only if it exists in the expected historical form.
3. If any SHA/source/manifest/dense-control mismatch exists, STOP and report the artifact blocker. Never rewrite old answers or regenerate the benchmark to make evidence pass. Keep old logs/data outside Git.

## B. STRICT GPU safety gate — no autonomy to recover

1. **Do not reboot, reset any GPU, reload drivers, rescan PCIe, use sudo, alter PCIe/power state, run stress tests, kill unrelated system processes, or attempt privileged hardware recovery.** Xid 154 states node reboot required, and that action is reserved to the human/operator.
2. If an authorized human/operator has NOT ALREADY restored and independently confirmed host health, **STOP after offline work and report Status: Blocked** with "NVIDIA Xid 79/154, operator recovery not confirmed." Do not launch a CUDA/HIP model to test recovery, loop/retry or automatically switch to HIP-only configuration.
3. Only after operator confirmation, perform bounded nonprivileged READ-ONLY enumeration, e.g. one invocation of timeout 10s nvidia-smi -L plus basic PCI/device identity query; inspect NVIDIA CUDA/driver readiness without hardware stress. If unavailable or errors persist, STOP and report Blocked. Device listing alone does not prove reliable model operation; follow conservative local host checks only once, after human confirmation. Record changed driver/model/GPU mappings and never conceal configuration changes.

## C. CONDITIONAL full rerun only on a recovered healthy host

1. Preserve exactly the Qwen3.8-Flash-Next two-shard IQ3_XXS target, Q8 MTP draft, CUDA0/ROCm0/ROCm1 4/7/7 split, F16 KV, context65536, max draft3, --reasoning off, greedy temp0/top-p1, seed42, no cache, identical chat system/user template and graph-disabled QSA mask audit. Sparse-only flag GGML_HIP_QSA_SPARSE_Q4_GQA12=1; keep legacy GQA4 sparse and forced WMMA override flags unset.
2. If the operator recovery changed driver version, weights, GPU mapping or mixed build, the old dense controls are NOT automatically comparable. Establish fresh bounded matched dense A/B controls if authorized on the healthy host, or STOP Blocked. If all settings are verified identical, retain the old 24/24 stable dense data as the comparison baseline.
3. Run the ENTIRE frozen 24-item sparse sample FROM TASK 1 with a fresh sparse server and a NEW empty result file, never the previous incomplete prefix. Bound startup and per-request deadlines, correlate task boundaries with actual HIP sparse Q4 target-verification launch trace on both HIP GPUs. Real GQA12 kernel launches inside natural MTP verification spans are necessary for quality attribution; the sparse flag/eligible KV alone do not count. If CUDA device failure recurs, STOP immediately; do not retry, reset or merge partial results. Clean up only owned process groups.
4. ONLY if all 24 new sparse responses complete, run score_qsa_quality_tasks.py with --require-stable-controls and analyze_qsa_quality_evidence.py on that full result. Report correctness, exact answer changes, stable baseline passes, true sparse regressions/improvements, per-domain breakdown, incomparability and per-task verified sparse launches. The illustrative one-sided iid regression upper bound is not a valid population confidence bound or accepted release margin; do not calculate or claim a 0028 full-sample value when sparse data are partial.

## Hard boundaries and deliverable

This is an incident-first guarded diagnostic. No default sparse enablement or product GO, numerical tolerance, full-model 512-token throughput A/B, 98K replay, GPU graph-enabled experiment, mixed production load, graph-safe overflow change, GitHub Actions, PR/merge/force push, driver reset or node reboot. The device incident cannot be proven caused by HIP sparse without a separate root-cause study. Keep every raw prompt, model output, frozen manifest, dataset file, process log and GPU trace OUTSIDE Git.

Write .ai/reports/report-0029.md with offline preflight and tests, exact frozen hashes, current operator-confirmed recovery/host-health state, actions actually taken, bounded GPU evidence ONLY if authorization and health gate passed, and explicit quality NO-GO. If the host is not independently recovered, return Blocked after committing/pushing the CPU test results and report. Do not waste additional GPU time trying to fix an Xid 79/154 host.

Return:
Iteration: 0029 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0029.md | Summary: <1-2 sentences>
