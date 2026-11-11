# Report 0009

Status: Partial. The matching Nsight importer and usable CUDA timelines were recovered, and HIP kernels were aligned to individual 32K/98K MTP steps. Longer Radeon flash-attention work is a measured contributor to the extra target-sync interval; a complete cross-device dependency DAG and a full explanation of the 19-21 ms per-step increase remain unproven. No production code or inference defaults changed.

## Validation and profiling safety

The checkout was synced to `b7cec5024b88666c6919cf8fecd676ee65e9f800`. The mixed CUDA/HIP build passed with `cmake --build build-perf -j 28 --target llama-server llama-bench test-backend-ops`. The host trace analyzer tests passed (4 tests), as did the ROCPD alignment tests (2 tests).

Before full-model profiling, a CUDA0 ADD test was run under Nsight with a 55-second deadline. The matching `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` (2022.4.2.50, no missing `ldd` dependencies) manually converted its stream; the SQLite export contained one GPU kernel and 23 GPU memcpys. Nsight's automatic importer discovery still reports that the importer is missing, but direct conversion works. No `nvprof` command was run.

The external `iteration-0009/run-profile.py` uses a fresh server, verifies the listening PID is a descendant of the launched profiler, checks process liveness, and enforces 90-second readiness and 300-second total deadlines. Its cleanup tracks descendant process groups and sends SIGINT, SIGTERM, then SIGKILL as needed. The first Nsight attempt correctly refused an ownership check based solely on process group because Nsight put the server in a separate group; the script was changed to verify ancestry and track descendant groups before retrying. No server or profiler was left running. The HIP tuning flags were unset.

A short MTP request produced the request-finished and final post-decode markers while its server was alive; the file survived SIGINT shutdown. Trace-enabled and disabled short requests had identical complete message SHA256 `4648f7d7545b8a4c655d8dcd5d2fd350ca4b2f1087393227d8d5199b9b9bfd89` and 321/565 accepted/drafted. This is not a logits test. A reader observed a zero-length file during a concurrent snapshot rewrite; the final file was intact. The current snapshot is therefore not an atomic replacement, and a reader must wait for a complete snapshot before consuming it.

## Matched captures and clocks

Each profile used a fresh server, CUDA0/ROCm0/ROCm1, CUDA0 MTP draft, requested 4/7/7 split, F16 KV, 196608 context and the saved uncached 512-token fixtures. HIP and CUDA captures are separate executions, not one simultaneous multi-vendor trace. Both returned 32809 or 98275 prompt tokens, 512 generated tokens and 306/611 or 306/612 accepted/drafted. Per-depth output hashes matched Report 0008 and between profilers. HIP-profiled throughput was 51.04 and 32.54 tokens/s, CUDA-profiled 58.20 and 40.87 tokens/s; profiling, especially HIP, perturbs timings.

ROCPD actual `kernels` and `regions_and_samples` views matched the analyzer schema. On the same marker thread, step 50's two `hipStreamSynchronize` calls began 0.001/0.002 ms after `target_sync_begin` and ended at the marker end for 32K/98K, respectively. `CLOCK_MONOTONIC_RAW` was approximately 135/136 ms behind MONOTONIC and did not match these boundaries. This supports MONOTONIC with zero offset for these captures, rather than merely relying on process-range overlap. Both traces contained 204 paired speculative target-sync windows and no unpaired markers.

Nsight SQLite uses session-relative timestamps. Its session UTC epoch was converted to monotonic using the locally sampled realtime-minus-monotonic offset, then independently checked against same-run CUDA runtime calls: the first `cudaStreamSynchronize` began a median 0.0007/0.0006 ms after `target_sync_begin` in all 204 windows. This supports the selected origin at these boundaries; it does not join clocks across separate runs. The CUDA exports have 79598/222777 kernels and 19268/23992 memcpys (32K/98K), including `mul_mat_q`, `quantize_mmq_q8_1` and `rms_norm_f32`.

## Per-step evidence

HIP-profiled medians, in milliseconds, for the target-sync windows:

| Measurement | 32K | 98K | Change |
| --- | ---: | ---: | ---: |
| Host target sync | 29.895 | 37.717 | +7.822 |
| Radeon agent 1 kernel busy union | 10.049 | 13.356 | +3.307 |
| Radeon agent 2 kernel busy union | 12.047 | 15.490 | +3.443 |
| Agent 1 flash-attention kernel overlap sum | 1.424 | 4.158 | +2.734 |
| Agent 2 flash-attention kernel overlap sum | 1.819 | 5.126 | +3.307 |

Busy is the union of clipped kernel intervals per agent; flash is the sum of clipped matching kernel durations, not a union. Neither should be added across devices or treated as an independently measured critical-path contribution. Median time from the last observed AMD kernel end to host sync end was about 0.277/0.274 ms. HIP target-process medians were 8.572/12.935 ms, with agent 1 busy overlap 1.274/1.699 ms; HIP spec-reprocess medians were 1.842/11.577 ms and had no AMD kernel overlap. The latter differs substantially from unprofiled host tracing and illustrates profiler perturbation.

At steps 50, 100 and 150 the HIP 32K sync windows were 29.931, 30.000 and 29.963 ms; 98K windows were 37.692, 37.721 and 37.859 ms. At step 50, the two host HIP synchronizations occupied 0.001-13.645 and 13.646-29.930 ms at 32K, versus 0.002-18.129 and 18.129-37.692 ms at 98K. In all six representative windows, the last observed AMD kernel was agent 2 `mul_mat_vec_q` and ended about 0.26-0.31 ms before host sync end. Its near-tail position does not prove it caused the depth-dependent increase: its individual duration was about 0.84-0.93 ms at both depths.

In the separate CUDA profiles, no CUDA kernels or memcpys overlapped any of the 204 target-sync windows at either depth. Representative CUDA-profiled step 50/100/150 target-sync intervals were 26.573/26.543/26.487 ms at 32K and 34.733/34.826/34.877 ms at 98K. This excludes observed CUDA device activity during those calibrated windows, not earlier CUDA work or inter-device dependencies. It does not establish a common clock or exact dependencies between the separate CUDA and HIP captures.

## Decision and next investigation

The +7.8 ms HIP-profiled host target-sync increase coincides with materially longer AMD execution, particularly flash attention; the final observed AMD work is on agent 2. The two HIP waits and near-tail AMD completion make the Radeon path the strongest measured target-sync lead, but overlap and profiling overhead prevent assigning all of the increase to attention or proving the full chain. Report 0008's unprofiled host phase changes remain draft +2.023, target processing +4.313, target sync +8.277 and speculative reprocessing +3.251 ms; those medians overlap and are not an additive decomposition.

Next inspect the backend ordering at `tools/server/server-context.cpp:4761`, `src/llama-context.cpp:972`, and `ggml/src/ggml-backend.cpp:2482`, then correlate HIP graph/event dependencies and copies on agent 2 with attention and the final matvec. Do not remove synchronization or retune attention based on these observations alone. Fixed-token logits and recurrent-state checks remain necessary for any future behavioral change.

Raw logs, responses, ROCPD databases, imported Nsight reports, SQLite exports, per-step CSV/JSON, and the external scripts are in `/home/chris/workspace/llama-perf-results/iteration-0009/`. No GitHub Actions were run.
