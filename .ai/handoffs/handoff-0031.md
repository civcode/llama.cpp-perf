# Handoff 0031 — offline incident forensics after successful PCIe Gen3 recovery

## Objective

After two NVIDIA Xid 79/154 fall-off-bus incidents on a single project day, the operator powered off/rebooted and forced the CUDA card to PCIe Gen3. Report 0030 then completed an entire mixed CUDA/HIP dense and sparse quality pilot without a new device fault. This is **one successful test**, not proof the PCIe generation was the fault's sole cause or a permanent fix. Before more GPU execution, perform a **read-only, CPU-only retrospective comparison** of historical system/experiment logs for both failures against the now successful Gen3 run. Seek diagnostic evidence of PCIe AER/DPC, NVIDIA GPU fault sequencing, CUDA failures, tracer concurrence and link settings. Do not generate another GPU workload.

Additionally, make the location and read-only viewing of ordinary llama.cpp server console/timing/tokens-per-second logs clear for the user. They previously asked to watch normal terminal performance output while the local Coding Agent owns a run.

## Repository and exclusive ownership

Repository civcode/llama.cpp-perf; branch docs/llama-perf. Use exactly the starting commit in Chat AI transfer prompt. Read .ai/WORKFLOW.md and Reports 0028, 0029, 0030 plus docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md. Coding Agent takes exclusive repository write ownership and returns it only after report 0031 is pushed.

Verified 0030 evidence:
- RTX 4070 Ti SUPER NVIDIA driver 580.178.04, Gen1 x4 idle and negotiated **Gen3 x4 during inference**, device maximum Gen3 x16. AMD HIP agents both visible; the original NVIDIA PCIe link width/topology and pre-failure negotiated width were not established. An x4 link is not itself a proven defect.
- An ordinary dense mixed-GPU smoke, separate fresh dense A/B for all **24 frozen** LongBench-v2 cases, and a complete new sparse run succeeded with no observed CUDA/HIP faults or new NVIDIA Xids/AER/DPC through testing.
- Fresh dense A/B 24/24 byte-exact repeat, **14/24** correct, zero old-to-new dense answer drift; sparse **14/24**, preserved all 14 dense-correct answers, **one** changed but still-correct response; **470 actual** HIP GQA12 Q4 sparse launches inside **47** natural MTP 4-token verification spans on both HIP GPUs across every case.
- Earlier 0028 sparse run had stopped after **3/24** with a CUDA prefill failure on question four, NVIDIA Xid79/154 and lost device access. The user reported one other similar incident that day, after many weeks of normal stable llama.cpp inference. Both incidents may be workload-triggered; neither the HIP Q4 kernel nor PCIe Gen4 is proven root cause.
- No production promotion, GPU graph, 98K stress replay, 512-token throughput benchmark or acceptable quality degradation criterion has been approved.

## Chat AI changes already committed — validate locally

1. New **docs/LLAMA-PERF/tools/analyze_qsa_gpu_incidents.py** is an offline, nonprivileged parser of text log files, preferably exported as kernel-journal short-ISO with monotonic source labels. Classifies NVIDIA Xid codes 79/154, PCIe AER/DPC/link problems, NVIDIA accessibility errors, selected CUDA failures, AMD ring/VM/reset failures and IOMMU errors. Reports event counts, available timestamps, Xid79→154 sequences and the preceding classified events without raw log text, GPU UUIDs or serial numbers. Limits input file size (32 MiB each), file count (6), matched lines (10,000) and output JSON; it does not read system journals by itself or access devices.
2. New **docs/LLAMA-PERF/tools/test_analyze_qsa_gpu_incidents.py** includes 10 CPU-only tests covering common Xid log shapes, ISO offsets, AER/DPC classification, sequencing, two-boot separation, no-logs caveats, missing paths and malformed lines. Both source and tests were written by Chat AI but have not been run on the local host. Fix ordinary bugs autonomously.
3. Updated QSA-SPARSE-TASK-QUALITY.md with evidence from 0030, incident hypotheses/limits and instructions for following normal llama-server console logs. No inference code or GPU device settings changed.

## A. CPU-only validation and evidence collection

- Run python3 -m compileall -q docs/LLAMA-PERF/tools, python3 docs/LLAMA-PERF/tools/test_analyze_qsa_gpu_incidents.py and focused QSA tests as appropriate. CPU-only; no CUDA/ROCm imports or model process. Fix ordinary test/format bugs.
- Locate preserved historical failed sparse server logs and tracer/client monotonic logs in /home/chris/workspace/llama-perf-results/iteration-0028/ and the operator's SECOND crash event logs if available. Inspect local journal's available boot records with read-only journalctl --list-boots and kernel journal (if already accessible without sudo) in short-ISO form for each relevant boot; identify the two actual incident times and, if possible, independently see the same PCI address/Xid pair for each. If previous-boot kernel journal was not persisted, it is an evidence GAP; do not elevate privilege, modify journald persistence, restart logging services or infer no errors. Copy only bounded relevant existing logs to a NEW external /home/chris/workspace/llama-perf-results/iteration-0031/ folder, not Git.
- Run analyze_qsa_gpu_incidents.py with each available relevant log as a distinct source label (first incident, second incident, Gen3 healthy boot if available) and a new external output file. Document total observed Xid79/154 counts, timestamps, PCIe AER/DPC and any relevant event preceding Xid79. Do NOT misinterpret a message absent from incomplete logs as proof it did not occur.
- Correlate each known incident's timestamps to the corresponding historical llama.cpp request/phase (CUDA prompt prefill, MTP verify, tracer startup etc.) only when grounded in preserved monotonic/client/server logs. Compare with successful 0030 phase; retain that 0030 had ROCm tracing enabled during sparse and still succeeded.
- Distinguish physical-link failure mechanism, potential workload trigger and diagnostic confidence. It is legitimate to say "root cause remains unknown." Note x4 negotiated width relative to maximum x16; don't recommend touching GPU hardware/BIOS automatically. If evidence suggests a physical lane issue, mention that a human can inspect slot/riser topology separately, not on this agent turn.

## B. Ordinary llama.cpp terminal/tokens-per-second output visibility

- From 0030 external artifacts, identify if the server stdout/stderr logs were retained and their naming convention, and whether normal llama.cpp timing/tokens-per-second messages actually appear. Report exact existing log filenames or a stable *pattern*, not invented paths.
- Show a **read-only** monitoring command suitable for the user's second terminal (e.g. tail -F /actual/path/log once known). A running server PID may disclose stdout/stderr targets through ls -l /proc/PID/fd/1 /proc/PID/fd/2. If the target is a file, tail that file; if a terminal or agent-owned pipe, do NOT attach a competing pipe reader, use the original agent terminal or plan proper future logging. Do not restart models simply to see tps.
- If the existing 0030 logs contain timing lines, summarize only the relevant examples; do NOT infer a full-model sparse-vs-dense throughput gain from unpaired/incidental logs. If no timing lines, say so and suggest a consistent log filename for any later separately approved run. Preserving server logs should not create another benchmark by itself.

## Safety constraints and completion

- **Offline-only:** NO model inference, no GPU stress or load, no GPU reset/reboot, no PCIe/device probing beyond existing saved records, no driver changes, sudo, privileged journal access, graph replay, 98K/512-token tests, production defaults, source kernel modifications, GitHub Actions, new workflows, PR/merge or force push.
- Keep raw host journals, paths/identifiers, model answers, hardware serial numbers, performance logs and benchmark texts **outside Git**. Report small aggregate facts and sanitized timestamps only. Historical incident analysis can be partial if the logs are genuinely unavailable; report exact limits.
- Write .ai/reports/report-0031.md with local Python tests, what historical log data is available, observed Xid/PCIe event chronology for each crash, current Gen3 evidence, level of causal uncertainty, and concrete terminal-log viewing instructions with grounded file names. No numerical equivalence/quality or production GO; sparse remains disabled by default.
- Commit/push ordinary fixes and the report to docs/llama-perf; then stop writing.

Return:
Iteration: 0031 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0031.md | Summary: <1-2 sentences>
