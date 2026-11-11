# Report 0010

Status: Completed. Analysis and build completed. No inference or synchronization behavior was changed in this iteration.

## Validation and identity

Synced to `c000ea8c08a0e7429c9312259b6c41908f10ef9f`. The ROCPD analyzer's two synthetic tests passed; `cmake --build build-perf -j 28 --target llama-server llama-bench` passed, including the opt-in scheduler diagnostic. The build's UI download fell back from the requested revision to the cached latest archive; no UI behavior was evaluated.

Reanalyzed the existing HIP traces using the Report 0009 validated MONOTONIC domain and zero offset: 204 paired target-sync windows at each depth, all with exactly two same-thread HIP stream waits. The enhanced analyzer reports time overlaps, not dependencies. Its CSV and JSON outputs are under `/home/chris/workspace/llama-perf-results/iteration-0010/`.

The captured HIP API records do not identify the stream's backend. A separate short 32-output-token mixed-GPU diagnostic with `GGML_SCHED_SYNC_TRACE=1`, standard 4/7/7 split and both HIP tuning flags unset identified scheduler backend 1 as ROCm0 and backend 2 as ROCm1. All eight speculative target-sync windows logged ROCm0 followed by ROCm1; for example step 2 began these waits at 0.001 and 10.403 ms after the marker, lasting 10.399 and 11.953 ms. The diagnostic logs waits of at least 1 ms and is not a benchmark. Together with the sequential scheduler loop and the two observed HIP calls, this maps wait 1 to ROCm0 and wait 2 to ROCm1 for this configuration. Agent 1 work almost exclusively overlaps wait 1 and agent 2 work wait 2; this supports, but does not independently prove, the agent-to-backend mapping.

## HIP wait measurements

Milliseconds; p90 uses the sorted sample at floor(0.9 * (n - 1)). Busy is a clipped per-agent kernel interval union; family values are sums of clipped kernel durations. Overlaps across devices or families must not be added as a critical-path decomposition.

| Measure | Wait 1, 32K | Wait 1, 98K | Wait 2, 32K | Wait 2, 98K |
| --- | ---: | ---: | ---: | ---: |
| HIP API duration median | 13.709 | 18.126 | 16.205 | 19.609 |
| HIP API duration p90 | 13.783 | 18.221 | 16.297 | 19.714 |
| Agent 1 busy union median | 10.048 | 13.355 | 0 | 0 |
| Agent 2 busy union median | 0 | 0 | 12.047 | 15.490 |
| Relevant agent flash-attention median | 1.424 | 4.158 | 1.819 | 5.126 |
| Relevant agent quantized matmul median | 4.827 | 4.958 | 6.110 | 5.945 |
| Relevant agent floating matmul median | 1.786 | 1.867 | 1.912 | 1.883 |
| Last observed AMD kernel to wait end median | 0.015 | 0.016 | 0.276 | 0.273 |

Median durations grow +4.417 ms and +3.404 ms respectively. The 98K first wait had negligible agent 2 overlap in two windows; its last observed AMD agent was 1 in 202 and 2 in two. At 32K its last agent was 1 in all 204; the second wait's last agent was 2 in all windows at both depths. The gap is only from last observed kernel to host API end, not proof of device idle or a measured synchronization overhead.

Representative wait durations and last-kernel gaps, in ms (wait 1 / wait 2):

| Step | 32K waits | 32K gaps | 98K waits | 98K gaps |
| --- | --- | --- | --- | --- |
| 50 | 13.644 / 16.284 | 0.015 / 0.303 | 18.127 / 19.563 | 0.016 / 0.263 |
| 100 | 13.722 / 16.276 | 0.015 / 0.308 | 18.031 / 19.686 | 0.015 / 0.282 |
| 150 | 13.778 / 16.182 | 0.015 / 0.260 | 18.220 / 19.636 | 0.015 / 0.298 |

The last observed family was `other` in the first wait and quantized matmul in the second for these steps. Longer flash-attention overlap is a strong lead, not a demonstrated causal share of either wait increase.

## Phase ordering and next action

Observed sequence: target processing queues split work and handles inputs/copies -> target-sync enters scheduler backend order (CUDA0, ROCm0, ROCm1, then CPU) -> the two measured HIP waits finish -> subsequent speculative reprocessing. In `ggml_backend_sched_compute_splits`, copies and prefetch precede asynchronous split graph submission, with events protecting reuse; unsupported async copy falls back to synchronizing the source and copying. These source-level ordering constraints do not establish every cross-device event edge in the traces.

The separate Nsight runs have no CUDA kernel or memcpy overlap in any of the 204 calibrated target-sync windows at either depth (Report 0009). In target-process windows, median CUDA kernel overlap was zero at both depths, but median memcpy counts were 16 at each depth, with 0.643/1.100 MB median bytes and host phase medians 7.512/11.675 ms. Some windows do contain kernels including `quantize_q8_1`, `mul_mat_vec_q`, and `rms_norm_f32`; their aggregate occurrence is not evidence they occur in a typical window. Copies are temporal overlap, not proof of a specific split handoff. The independent HIP and CUDA executions do not provide a common per-step cross-device DAG, and activity outside these windows is not ruled out.

HIP-profiled total target-sync median rises 29.895 to 37.717 ms (+7.822 ms). Unprofiled host-trace phase median changes from Report 0008 were draft +2.023, target processing +4.313, target sync +8.277, and speculative reprocessing +3.251 ms. These medians cannot be summed into an exact decomposition; HIP profiling materially perturbs phases, particularly speculative reprocessing. The rest of the approximately 19-21 ms per-verification increase is not explained by these GPU overlaps.

Next: a bounded A/B on the Radeon flash-attention implementation in `ggml/src/ggml-cuda/fattn*`, targeting reduced KV data movement at long context while retaining the same scheduler and math. Compare baseline and one isolated change with matched 32K/98K fixed-token workloads and unprofiled end-to-end phase timings; use targeted operator profiling only if needed. Check fixed-token logits and recurrent state across multiple verification/acceptance sequences before considering promotion. No speedup is claimed and no waits should be removed.

The diagnostic used the existing liveness/health checks and process-tree cleanup with a 60-second startup and 120-second total deadline. Diagnostic log, host TSV, response and derived files are outside Git in the iteration-0010 results directory. No GitHub Actions or unsupported profiler were used.
