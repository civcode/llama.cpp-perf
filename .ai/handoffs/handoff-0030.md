# Handoff 0030 — controlled PCIe Gen3 mixed-GPU retest after operator recovery

## Objective

The operator reports they have **powered off/rebooted** the host, switched the NVIDIA CUDA card's PCIe link setting to **Gen3**, and can see all three GPUs again. In Report 0029, GPU work was blocked because the authorized recovery was not confirmed. That blocker is now explicitly addressed by the user, but GPU visibility is NOT proof of stability: there were two earlier NVIDIA CUDA GPU dropouts/Xid79/154 on this same project day after weeks of stable ordinary llama.cpp inference. Take a conservative, staged, bounded approach to detect the issue's recurrence **without** automatic recovery or stress testing.

The chosen PCIe Gen3 setting is a **different host configuration** from the previous 0028 runs. Consequently, do **not** use the pre-incident dense controls as the matched baseline for the new sparse run. Create **new fresh dense A and dense B full controls under the Gen3 setting** first, and score the new sparse condition **only against those new controls**, on the same already-frozen original 24 questions. Preserve all old outputs.

## Repository / ownership

- Repository: civcode/llama.cpp-perf
- Branch: docs/llama-perf
- Starting commit: exact SHA in Chat AI's user-relayed message.
- Read .ai/WORKFLOW.md, .ai/reports/report-0028.md, .ai/reports/report-0029.md and docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md.
- Coding Agent owns repository changes exclusively after transfer. No Chat AI concurrent file edits. Write .ai/reports/report-0030.md, commit/push and return the standard message.

## Frozen data and previous incident

- Official 0027 cached dataset from zai-org/LongBench-v2 revision 12bc220a4e257d069e9005f2df762cac4755dc54, SHA-256 15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2.
- Previous 0027 12-item manifest SHA-256 d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1.
- 0028 frozen 24-item manifest SHA-256 ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5; user prompt+system tokens 17,051–45,158, within ctx65536, >=16408 sparse selector floor. Entire original source-easy sample disjoint from the earlier 12 IDs; source texts/gold must not change.
- Pre-incident dense A/B ran all 24, exact text/finish/token repeat, 14/24 correct. Historical sparse process completed ONLY first 3/24, with matching text on 3/3, two correct, 60 real Q4 HIP launches inside six natural MTP verification groups. It failed at CUDA prompt processing of question 4 with NVIDIA Xid79 followed by Xid154 and GPU no longer accessible. A second incident reportedly also occurred that day. No causal component is proven.
- 0029 CPU-only preflight passed on all pinned files, 152 focused QSA tests passed; the local Coding Agent appropriately returned Blocked without GPU activity because there was no operator confirmation yet.
- The **new** 0030 matched comparison must never reselect benchmark cases, merge incomplete sparse prefix or overwrite old 0028/0029 files.

## Chat AI implementation — not yet tested on local host

1. New docs/LLAMA-PERF/tools/compare_qsa_recovered_runs.py consumes the exact frozen 0028 manifest; *old* dense A/B files; *new* recovered-host dense A/B files; and optionally a *complete new* sparse JSONL. It checks distinct artifact paths, pinned manifest SHA/count, strict 24-row result alignment, exact dense A/B repeats in both cohorts, matching token counts/context window, historical old dense 14/24 score and new sparse-vs-NEW dense comparisons. It separately reports any old-to-new dense answer/finish drift and new sparse regressions/improvements. Without --sparse it produces **only a fresh-dense review** and cannot claim sparse completion. It is CPU-only and explicitly does not authenticate GPU health, run provenance or HIP sparse kernel launches. Distinct paths alone cannot prove files came from different processes; corroborate by local log timestamps and process arguments.
2. New docs/LLAMA-PERF/tools/test_compare_qsa_recovered_runs.py has 11 synthetic CPU-only tests for full/partial cohort behavior, new-vs-old drift, genuine sparse regression classification, missing files, old score pin, token usage and unstable controls. Validate and autonomously correct ordinary bugs. The existing preflight_qsa_0028_resume.py and its tests remain unchanged.
3. Updated docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md to record the operator Gen3 change and staged continuation. No production kernel, selector, MTP behavior, GPU graphs, CUDA driver, model weights or source gold changed.

## A. Offline preflight, host identity and limited smoke

1. Run python3 -m compileall -q docs/LLAMA-PERF/tools; python3 docs/LLAMA-PERF/tools/test_compare_qsa_recovered_runs.py; and proportionate existing focused QSA tests. No GitHub Actions. Verify prior 0029 offline report and frozen artifact hashes remain unchanged. Use a NEW external output directory under /home/chris/workspace/llama-perf-results/iteration-0030/ for every new result/log, never overwrite old data.
2. User has stated an authorized operator performed a power cycle and PCIe Gen3 adjustment. This is sufficient to permit **bounded nonprivileged read-only checks**, NOT permission to perform any further hardware recovery. Take a small set of passive snapshots of NVIDIA device enumeration, PCIe current and maximum link generations/width, NVIDIA driver version, visible AMD GPUs, temperatures/power if available, and recent current-boot NVIDIA Xid and PCIe AER/DPC errors. Prefer a single bounded invocation per check (e.g. timeout 10s nvidia-smi -L, then a bounded nvidia-smi --query-gpu read-only output) and nonprivileged kernel journal inspection. Verify CUDA GPU truly negotiated **Gen3 currently**, not merely that firmware requested Gen3. Avoid exposing GPU UUIDs/serial numbers in the public report; small non-identifying model/driver/link summaries suffice.
3. If any GPU disappeared again, NVIDIA init fails, new Xid79/154 appears or even passive checks indicate a serious continuing PCIe fault, **STOP, do not retry/load inference**. Report Blocked with evidence. The visibility of all three GPUs after reboot is a prerequisite, not a stability guarantee.
4. If passive checks pass, perform **ONE** low-footprint, bounded normal mixed CUDA/HIP llama.cpp dense inference smoke (short prompt, few generated tokens, existing target weights and same CUDA0/HIP0/HIP1 hardware), with sparse flags and ROCm kernel tracing OFF. This is only a readiness sanity check, not a performance benchmark or numerical equivalence test. Capture whether it completed and check for fresh NVIDIA Xid/PCIe errors afterward. Abort immediately on any GPU error; do not retry automatically. Don't change GPU clocks, power limits or device mapping.

## B. Fresh frozen 24-question dense A/B on Gen3

Only after A succeeds:

1. Use the exact frozen 0028 task manifest and existing verified dataset, no new selection, truncation or gold changes. Run **fresh dense A and fresh dense B** sequentially in separate bounded llama-server processes on the current host. Restore original 0028 inference settings: Qwen3.8-Flash-Next two-shard IQ3_XXS target and Q8 MTP draft, CUDA0/ROCm0/ROCm1 layer split4/7/7, F16 KV, context65536, max draft3, reasoning OFF, temp0/top-p1, seed42, no prompt caching, identical system/user templates and graph-disabled mask auditing. Set NO sparse flags and no forced WMMA flags. Do not use GPU graphs or start background services.
2. Run all 24 requests per process through the existing run_qsa_quality_tasks.py, writing new files in iteration-0030/ (e.g. dense-a.jsonl and dense-b.jsonl), plus exact separate server logs and process argument/config notes. Use bounded startup/per-request deadlines and process-group cleanup.
3. After each phase, inspect for fresh device errors/Xid, check bounded device visibility, and STOP immediately on anomalous CUDA errors. Do not auto-retry or reset GPUs. Preserve any partial file separately; never call it a complete control.
4. Run compare_qsa_recovered_runs.py **without** --sparse against original 0028 dense files and **new** 0030 dense files. Inspect old-to-new dense output drift (a different score is allowed, but must be reported) and verify all 24 exact new dense A/B text, finish and token repeats. If new A/B are inconsistent or fail the full 24, stop before sparse; report Blocked/Partial. Do not silently compare sparse against old pre-Gen3 controls.

## C. Conditional entire frozen sparse run on healthy Gen3 host

Only if B is fully complete, stable, and host remains healthy:

1. Run exactly **one fresh sparse server** with all the same Gen3 model/target/draft/device/KV/settings, only adding GGML_HIP_QSA_SPARSE_Q4_GQA12=1. Legacy GQA4 sparse flag and forced WMMA stay unset. Use a new empty sparse JSONL path; start at question 1, run **all 24** original frozen questions in original order. Never merge with the three historical 0028 sparse answers. Request timeout/cleanup bounded. Stop on any CUDA/HIP error, new NVIDIA Xid79/154, disappearance or other dangerous condition, and DO NOT automatically retry.
2. Independently confirm actual experimental HIP GQA12 Q4 kernel launches in **natural MTP target-process verification** on both AMD devices and attribute launches to EVERY completed task included in quality evidence. Use already established bounded ROCm tracing and client/server monotonic markers; trace instrumentation may change load pattern, so record accurately if it is enabled and treat this as a possible triggering factor. If tracing itself provokes instability, STOP rather than silently disabling trace then claiming unverified sparse activation.
3. **ONLY on complete 24/24 new sparse outputs**, run compare_qsa_recovered_runs.py with --sparse and the existing strict score_qsa_quality_tasks.py --require-stable-controls using NEW dense A/B. Run analyze_qsa_quality_evidence.py on the complete strict score report. Report old-vs-new dense drift separately from new-dense-vs-new-sparse differences; give 24-item benchmark accuracy, per-domain results, baseline-correct preserved/lost, genuine regressions/improvements, changed-but-both-wrong/changed-but-both-right, and exact trace attribution. A nominal iid upper confidence figure, if reported, MUST be labeled an illustrative sensitivity value, not a valid population guarantee, release threshold or automatic GO decision.
4. Keep raw benchmark data, outputs, kernel traces, power snapshots and system logs OUTSIDE Git. Link only compact findings and original public dataset IDs/hashes in the report. Do not claim Gen3 fixed the fault from one healthy run. One or two recurrent incidents plus a changed link speed can support a PCIe link hypothesis, but do not prove root cause.

## Hard boundaries and completion

- User/operator already performed the reboot and Gen3 change. The agent is NOT permitted to reboot, reset NVIDIA/AMD GPUs, reload drivers, rescan PCIe, use sudo, alter power/PCIe/firmware settings, run GPU stress tests, or retry after a serious error. Do not run profiler with elevated privilege, and never leave orphan model or tracing processes.
- No full-model 512-token throughput A/B, 98K stress replay, graph-enabled sparse run, changes to kernel arithmetic/MTP selectors/overflow, default sparse enablement, GitHub Actions, PR/merge/force push or branch changes.
- Sparse numerical equivalence remains NO-GO, production promotion NO-GO and the experimental path stays disabled by default, regardless of results.
- If a new serious hardware error occurs at ANY stage: **stop the entire GPU experiment immediately**, preserve relevant logs and report Blocked/Partial. A new GPU failure is not permission for automatic recovery.
- If complete, report the full 24-task fresh matched quality comparison and whether the Gen3 run remained device-stable through every phase.

Write .ai/reports/report-0030.md with exact commits, CPU tests, pinned artifact verification, independent observed NVIDIA current PCIe generation and GPU health checks, timestamps of any new Xids, smoke, fresh dense A/B comparison vs 0028, complete or partial new sparse results, HIP launch evidence and quality NO-GO. Commit/push tested code, fixes and report to docs/llama-perf, keep raw artifacts external.

Return:
Iteration: 0030 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0030.md | Summary: <1-2 sentences>
