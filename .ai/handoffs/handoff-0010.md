# Handoff 0010 — identify the two HIP waits and the remaining MTP step latency

## Objective

Use the **already captured Iteration 0009 HIP/CUDA traces** to determine which configured backend and GPU work corresponds to each of the two sequential HIP waits inside a target verification step. Quantify the 32K-to-98K change separately for each wait, the overlapping Radeon kernel families, and the CUDA work before the waits. Establish the strongest defensible dependency sequence and the first concrete optimization target, without another blind launch-parameter experiment.

## Evidence and source prepared

- Work only on `civcode/llama.cpp-perf`, branch `docs/llama-perf`. Read `.ai/WORKFLOW.md` and `.ai/reports/report-0009.md`. The Chat AI transfers exclusive write ownership with this handoff.
- Report 0009 obtained working Nsight Systems 2022.4.2 GPU timeline export and correlated all **204 target-sync host windows** in separate matched ROCprofiler runs. **32K → 98K** HIP-profiled target-sync median **29.895 → 37.717 ms** (+7.822 ms). Per-window Radeon busy union increased by **+3.307 ms on agent 1** and **+3.443 ms on agent 2**; flash-attention overlap sums increased **+2.734 / +3.307 ms**. No CUDA kernels or memcpys appeared during target-sync windows on the separate CUDA captures, but CUDA executed elsewhere in the verification cycle.
- In representative HIP step 50, two sequential `hipStreamSynchronize` calls spanned **0.001–13.645 and 13.646–29.930 ms** at 32K vs **0.002–18.129 and 18.129–37.692 ms** at 98K. The final observed HIP kernel was agent-2 `mul_mat_vec_q`, ~0.26–0.31 ms before host sync end, but the kernel itself did not grow substantially. This suggests the **upstream queue and ordered backend waits**, not the final matvec alone, may explain the latency.
- Source inspection: `ggml/src/ggml-backend.cpp::ggml_backend_sched_synchronize()` executes `ggml_backend_synchronize(sched->backends[i])` **sequentially in scheduler backend order**. This is necessary completion and not removable delay. Other relevant paths: `ggml_backend_sched_compute_splits()`, `ggml_backend_sched_copy_input()`, `src/llama-context.cpp::synchronize()`, and `tools/server/server-context.cpp::decode()`.
- Chat AI extended `docs/LLAMA-PERF/tools/rocpd_step_alignment.py` to list **each actual same-thread HIP stream-sync API wait per MTP step** with start offset, duration, clipped per-Radeon busy union, per-kernel-family overlap and last AMD agent/kernel-family. It summarizes medians and tail-agent counts by wait position. Expanded synthetic unit test uses two waits per step. This analysis does **not** know the stream/backend identity; temporal overlap does not establish dependency.
- Chat AI added **disabled-by-default** `GGML_SCHED_SYNC_TRACE=1` diagnostic around the **existing** `ggml_backend_synchronize` calls in `ggml-backend.cpp`, which logs scheduler backend index/name, start steady-clock ns and duration for waits at least 1 ms. It adds **no** new synchronization. Intended only for a **short** mixed-GPU diagnostic request; logging may perturb latency. Both HIP tuning flags remain unset.
- These changes are pushed with the handoff but **have not yet been built or validated on the local machine**.

## Tasks

1. **Validate tools and build.** Update the agreed branch to the exact handoff commit, run `python3 -m unittest discover -s docs/LLAMA-PERF/tools -p 'test_rocpd_step_alignment.py'`, then `cmake --build build-perf -j 28 --target llama-server llama-bench`. Fix ordinary analyzer/C++ errors autonomously; preserve the untouched `build-mixed`, original checkout and GGUF files.
2. **Reanalyze existing traces FIRST.** Reuse the exact local ROCPD databases and host TSVs at `/home/chris/workspace/llama-perf-results/iteration-0009/`. Run the enhanced alignment tool separately at 32K and 98K with the **MONOTONIC clock alignment validated in Report 0009**, e.g. `python3 docs/LLAMA-PERF/tools/rocpd_step_alignment.py host32.tsv trace32_results.db --clock mono --json wait32.json --rows-out wait32.csv` (substitute actual local names). Confirm ~204 steps and the distribution of HIP wait counts. Summarize wait #1 vs #2 median/p90 durations, their 32K→98K deltas, per-agent busy union, flash attention and other major kernel family overlap, last AMD completion per wait and idle/tail gaps. Include representative steps 50/100/150. Do not add overlapped GPU times across devices.
3. **Prove backend identity with a minimal live diagnostic, if necessary.** First consult source/backend order and the already captured CUDA/HIP API timestamps. If the two stream waits still cannot be mapped to actual ROCm0/ROCm1 backends, run **one short, small-output** mixed-GPU MTP request with `GGML_SCHED_SYNC_TRACE=1`, both HIP tuning flags unset, and standard split. Capture scheduler lines and correlate start/end with host step markers to identify backend order. Do **not** rerun both full 32K/98K profile suites just to establish call identity. Diagnostic logging is not a performance benchmark.
4. **Connect target-process and the two waits.** Use the **existing** 0009 Nsight CUDA SQLite exports, host markers, and HIP ROCpd traces: locate CUDA graph/memcpy activity in target_process and any earlier prefetch/copy phases, characterize which CUDA work is complete **before** host target_sync and which Radeon work stretches the first and second HIP waits. Inspect scheduler event, copy, and split sequencing; distinguish **queued work on a device** from **time spent in a host API**. Separate HIP total target-sync increase from the rest of the ~19–21 ms end-to-end per-verification increase (draft, target_process, spec_reprocess). If cross-run clocks and profiling overhead prevent a complete DAG, explicitly state the remaining edges rather than inventing a precise universal timeline.
5. **Decide next code target.** Identify the one source path and behavior most likely to reduce true end-to-end MTP latency (for example R9700 flash attention data movement, split scheduling, or a copy/event handoff) only if supported by the measured phase ordering. Recommend a **specific bounded A/B** and numerical/recurrent-state correctness checks. Do **not** remove synchronizations, change math, promote past GQA/parallel-block experiments, or claim a speedup now.
6. **Report and hand back.** Save derived CSV/JSON and any short diagnostic logs outside Git under `/home/chris/workspace/llama-perf-results/iteration-0010/`. Write `.ai/reports/report-0010.md` with wait-by-wait measurements, validated backend mapping or precise ambiguity, unprofiled vs profiled caveats, dependency diagram in textual form and one next engineering action. Commit/push report and any necessary small fixes, then return the standard completion pointer.

## Execution safeguards

- **Do not use `nvprof`** or repeat unsupported profiling attempts. Start with already saved data, not another expensive profile.
- If a live diagnostic is needed, perform a small process-liveness/health preflight, apply short bounded startup/execution deadlines, and cleanly terminate the entire child process tree. Do not use five-hour timeouts or continue if the profiler/server has already exited. No elevated permissions, resets, reboots or driver/security modifications.
- No GitHub Actions, PRs, merges, or force-pushes. Keep defaults unchanged, build with **28 jobs**, and leave raw profiler data out of Git.

## Completion criteria

- Backend sync diagnostic compiles (or a documented compiler blocker), and the extended alignment tool passes unit tests.
- Existing 0009 traces yield a per-step **wait #1 vs wait #2** 32K/98K comparison, including Radeon kernel overlap and actual host wait duration.
- Identify the backends behind the HIP waits, if supported by the trace and a short live diagnostic, or precisely explain what evidence is still missing.
- Explain as much of the ~20 ms/verification increase as the data genuinely supports and name the next bounded causal A/B.
- `.ai/reports/report-0010.md` pushed to `docs/llama-perf`; exclusive write ownership returned.

## Reporting

`Iteration: 0010 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0010.md | Summary: <1-2 sentences>`
