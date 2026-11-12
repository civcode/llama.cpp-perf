# Report 0029 — offline continuation after NVIDIA incident

**Status: Blocked.** NVIDIA Xid 79/154 was recorded during Iteration 0028; operator recovery is not confirmed. No operator confirmation was supplied in this handoff. Current GPU/driver health is **unverified**, not asserted healthy or unhealthy. In accordance with the safety gate, no device enumeration, CUDA/HIP initialization or inference was attempted. Numerical correctness and production promotion remain **NO-GO**; sparse remains disabled by default.

## CPU-only validation

Starting commit: `2eb3b634d24f5d42d88aaf763f8ae1c128e0ded8`. Local `python3 -m compileall -q docs/LLAMA-PERF/tools` passed. The direct CPU-only preflight suite passed **11 tests**; `python3 -m unittest discover -s docs/LLAMA-PERF/tools -p 'test_*qsa*.py'` passed **152 tests**, including the preflight tests. No fixes were needed.

Ran `preflight_qsa_0028_resume.py` on the original external artifacts, including the historical incomplete sparse JSONL, with a new exclusive-create output at `/home/chris/workspace/llama-perf-results/iteration-0029/offline-preflight.json`. The preflight passed:

- Original cached publisher dataset `iteration-0027/data.json`: SHA-256 `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.
- Previous 12-question manifest `iteration-0027/tasks.json`: SHA-256 `d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1`.
- Frozen 24-question manifest `iteration-0028/tasks.json`: SHA-256 `ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5`; 24 distinct IDs, disjoint from the 12 recorded exclusions.
- Both complete dense files matched in response text, finish reason and prompt token count for all 24 tasks; **14/24** graded correct. Chat-token counts span 17,051–45,158 and meet the preflight context constraints.
- The historical sparse file contains exactly the first **3/24** ordered requests with matching request metadata. All three match dense text and prompt counts; two graded correct. This is an incomplete diagnostic prefix, not a full sparse result, and must never be merged into a future run.

Re-ran the existing CPU-only trace summarizer against the preserved 0028 trace and request boundaries: **90** recorded sparse Q4 kernel launches total, **60** inside six natural batch-4/draft-3 verification spans belonging to the three completed requests; each request had two spans and 10 launches on each HIP agent. This verifies historical trace evidence only; it says nothing about present device health or the remaining 21 answers. The original data, answers, logs and trace were not overwritten or downloaded again.

## Boundary and next step

The historical CUDA failure and NVIDIA Xid 79/154 require independent operator diagnosis; the cause is unknown and cannot be attributed to HIP sparse on this evidence. No reboot, reset, driver reload, PCIe rescan, sudo, hardware recovery, GPU stress test, full-model throughput benchmark or GitHub Actions was performed. There is no complete 0028 sparse comparison, regression count or full-sample evidence bound.

After an authorized operator has already restored and independently confirmed host recovery, the next agent may perform the handoff's bounded read-only checks. If the original mixed-device configuration is healthy and controls remain comparable, rerun **all 24 frozen questions from task 1** in a new output file and verify actual HIP sparse Q4 launches per task before scoring. If configuration changed, establish authorized matched dense controls or stop. Do not reuse or merge the old three responses. All raw artifacts remain outside Git.
