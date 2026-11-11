# Report 0001

Status: Partial. Branch `docs/llama-perf`, starting commit `6c8fd40e59a7d2812ac8572e02fb1620baac7632` (reference binaries built from `1648d641174927634dfb72135b220e4052393729`). No scheduling change was made.

## Existing short trace

The committed analyzer ran unchanged against `/home/chris/workspace/llama-perf-results/20261111-143704/hip/short-0/iq3_short_results.db`; its three fixture tests passed. The JSON summary is `/tmp/sync-short.json` (temporary). In the 6.000-9.371 s window, 3,981 synchronizations on TID 331789 totaled 2,769.204 ms, 82.15% of the window; mean 695.6 us, p95 7.806 ms, maximum 12.658 ms. This is blocking time, not established removable overhead.

The immediate same-thread neighbor pattern `hipMemcpyAsync -> sync -> hipEventSynchronize` accounts for 2,698 calls but only 134.95 ms. `hipMemcpyAsync -> sync -> sync` accounts for 129 calls and 997.50 ms; `sync -> sync -> sync` accounts for another 254 calls and 1,418.92 ms. Neighbors do not identify callsites or dependencies. The window contains 128 agent-2-to-host copies of 993,280 bytes (9.524 ms total copy-engine duration), consistent with logits but not proof. Agent 1 and 2 accumulated 876.693 and 953.767 ms of kernel durations respectively; durations can overlap and cannot be summed as wall time. CUDA activity is absent from this trace.

## Source attribution and next step

`ggml_backend_sched_copy_input` in `ggml/src/ggml-backend.cpp` synchronizes the source backend on unsupported asynchronous copies, and may synchronize or wait on the destination before reusing storage. `ggml_backend_sched_compute_splits` can synchronize across splits for allocator safety; `ggml_backend_sched_synchronize` synchronizes every backend. The HIP build's `ggml_backend_cuda_synchronize` in `ggml/src/ggml-cuda/ggml-cuda.cu` calls stream synchronize. `src/llama-context.cpp` schedules logits readback sized `n_outputs*n_vocab*sizeof(float)`. These are plausible paths, not measured callsite attribution; other direct sync paths can exist.

Most promising next action: on the GPU host, add opt-in aggregated callsite and backend counters/timers in an independent `build-perf/`, especially around scheduler fallback and full scheduler synchronization, without adding device synchronization. Correlate decode windows with both GPU timelines and compare fixed-placement depth 0 versus 32K target-only, then validate any proposal with unprofiled end-to-end A/B and correctness tests. The expensive repeated sync patterns justify investigation, not deletion; event and buffer lifetime dependencies must remain intact.

## Limitations

This execution environment exposes neither `/dev/kfd`, `/dev/dri`, nor NVIDIA device nodes; `nvidia-smi -L` failed. Its PID namespace also prevents checking host server processes. Therefore no new GPU trace, prefill/decode isolation, call stacks, placement confirmation, trace overhead measurement, long-context sync comparison, MTP run, or inference correctness test was possible. Existing MTP 32K versus 98K results (59.10 versus 39.28 t/s at similar acceptance) demonstrate a depth-related end-to-end slowdown but say nothing about how synchronization changes. Existing target-only depth results in `docs/LLAMA-PERF/BENCHMARK-RESULTS.md` likewise do not measure sync. The raw trace and model files were not modified; `build-mixed/` and the original checkout were preserved. No GitHub Actions were used.
