# Report 0008

Status: Partial. Matched benchmarks and host phase traces were obtained, but GPU activity was not correlated to individual verification steps and no CUDA timeline was recovered. Therefore the device-level critical path and the cause of the additional time remain unproven. Source revision: `043dbffb3dd4a3f1e3eff96ba2054754c51c62ea`. No production changes were made.

## CUDA and build

`nvidia-smi -L` and `llama-server --list-devices` showed CUDA0, an RTX 4070 Ti SUPER. `test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'kv=32768.*long_kv_deterministic=1'` passed against the CPU reference. CUDA0 also completed repeated fixed-work attention at 32K and 98K, and the mixed-device requests below completed. This establishes execution for these runs, not permanent PCIe stability. `dmesg` was inaccessible without privilege; `journalctl -k -b -g 'NVRM: Xid'` returned no entries, which is not proof that all kernel logs were accessible. No resets or system changes were made.

`cmake --build build-perf -j 28 --target llama-server llama-bench test-backend-ops` passed. The three host analyzer synthetic tests passed using `python3 -m unittest discover -s docs/LLAMA-PERF/tools -p 'test_server_step_trace_analysis.py'`. The first invocation using a path as a unittest module failed due to the import path; discovery succeeded.

## Unprofiled matched runs

Separate fresh server processes used the saved uncached 512-token requests and the Report 0004 model, MTP draft, CUDA0/ROCm0/ROCm1 device list, 4/7/7 requested tensor split, 196608 context, F16 KV, flash attention and sampling settings. Both HIP experimental settings and the host trace setting were unset. As before, `--fit-target 768` was requested but fitting aborted because the split was supplied; it was not effective. The non-verbose logs do not establish per-tensor placement or per-run memory headroom. Prompt time is server-reported prefill time, not an independently measured network TTFT.

| Prompt tokens | Rep | Prefill ms | Generation ms | tokens/s | Verification steps | ms/step | accepted/drafted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32809 | 1 | 15660.455 | 8662.810 | 58.99 | 204 | 42.46 | 306/611 |
| 32809 | 2 | 15853.998 | 8563.867 | 59.67 | 204 | 41.98 | 306/611 |
| 98275 | 1 | 59047.634 | 12874.410 | 39.69 | 204 | 63.11 | 306/612 |
| 98275 | 2 | 59643.324 | 12499.537 | 40.88 | 204 | 61.27 | 306/612 |

All returned length finish, 512 output tokens and zero cached prompt tokens. SHA256 of the complete JSON message, including reasoning content, was identical between repetitions at each depth: 32K `687e1904b3ce98f11909c0f7ac80f88c8dba86c5bfb9f8ea4f2feba303affd7f`; 98K `c6c72dc2ce772be9b8bd75609bb626cb7b9ea6c0444a73219c72c0ef4b1e45ab`. This is not a logits or recurrent-state correctness test. The between-depth difference is approximately 19-21 ms per verification, measured as whole generation time divided by the metrics counter.

## Host phase markers

A short request with `LLAMA_SERVER_STEP_TRACE` unset produced no trace at the designated path; an enabled request produced a TSV on clean server shutdown. Full enabled 32K and 98K requests produced paired marker intervals with zero unpaired events. The analyzer selected 204 speculative intervals per phase. Values below are median/p90 milliseconds, with median differences; these are host durations, not independent additive GPU durations.

| Phase | 32K median/p90 | 98K median/p90 | Median delta |
| --- | ---: | ---: | ---: |
| draft | 5.895/6.090 | 7.918/8.676 | +2.023 |
| target_process | 7.156/7.784 | 11.470/12.366 | +4.313 |
| target_sync | 26.649/26.776 | 34.926/35.102 | +8.277 |
| spec_reprocess | 0.949/1.397 | 4.199/5.493 | +3.251 |
| post_decode | 0.712/0.918 | 0.830/1.003 | +0.118 |
| verification_sample | 0.260/0.500 | 0.303/0.520 | +0.043 |

Representative step 50: 32K target_process 7.057, target_sync 26.665, spec_reprocess 0.957 ms; 98K 11.687, 34.895, 3.871 ms respectively. These labels indicate host boundaries, not proof of device ownership. In particular, target synchronization may wait for previously submitted work. Draft may execute on a different host thread. Marker overhead and phase overlap prevent treating summed median deltas as a measured device critical-path decomposition.

## GPU profiling and unresolved attribution

ROCprofiler 1.4.1 collected HIP kernel, copy and API CSVs for both complete requests. Profiled generation throughput was 51.42 and 33.75 tokens/s at 32K and 98K respectively, showing substantial profiler perturbation. Unfortunately the profiler shutdown finalized after signal 2 without flushing the server's host marker files. Thus those CSVs cannot be safely correlated to individual marker-defined verification steps; they also omit CUDA. Report 0003's approximate HIP flash-attention increase remains a hypothesis, not a measured share of the current elapsed delta.

CUDA timeline capture was not successful. The first `nsys` launch failed due to an ambiguous command-line separator; the second tried to execute `env` and returned `Not executable`. Both launch scripts incorrectly waited for server health after the profiler had already exited, wasting time; they should have checked process liveness and enforced a short readiness deadline. A direct `nsys` backend smoke executed, but the installed Nsight Systems 2022.4.2 reported `The importer binary and its dependencies were not found` and emitted only a `.qdstrm` stream, which the installed export/stats tools did not accept. `nvprof` reported that compute capability 8.0 and newer is unsupported. No CUDA GPU kernel/memcpy timeline was obtained. These failures are profiler/tooling limitations, not evidence of GPU inactivity during the successful inference runs.

The host evidence ranks the extra synchronization wait first, then target processing, speculative reprocessing and drafting by median duration increase. It does not identify the device operations responsible or establish that any wait can be removed. The observed ordering is host draft, target processing and synchronization, speculative reprocessing, post-decode and verification sampling/commit; exact GPU dependencies and overlap remain unmeasured. No causal optimization or speedup is claimed.

Next priority: obtain a working unprivileged Nsight Systems importer matching the capture tool, then repeat bounded CUDA and HIP captures with process-liveness checks and graceful delivery of SIGINT to the actual server, verifying that both host TSV and profiler data are present. Calibrate clock domains before associating events with marker windows. Inspect target execution and dependency ordering in `tools/server/server-context.cpp` and the backend scheduler before proposing any change; preserve synchronization and require fixed-token logits and recurrent-state validation for an A/B.

Raw logs, responses and profiler CSVs remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0008/`. No GitHub Actions were run.
