# LLAMA-PERF: IQ3_XXS performance engineering

Profiling and benchmark record for Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS on mixed CUDA + HIP inference using [civcode/llama.cpp-perf](https://github.com/civcode/llama.cpp-perf).

**Status (2026-11-11):** Reports 0008–0010 reproduced mixed CUDA0/ROCm0/ROCm1 MTP context degradation at ~59 t/s (32K) versus ~40 t/s (98K) and resolved the two ordered GPU waits: ROCm0 wait +4.417 ms, ROCm1 +3.404 ms per verification step (HIP-profiled medians). Prior GQA/parallel-block tuning did not help. **Report 0011 rejected the opt-in dense WMMA Q=4 dispatch**: numerically correct but 15.5–25.4% slower than tile on two R9700s at 32K/98K. Handoff 0012 now tests whether the existing Qwen4 sparse-mask compaction and K/V gather architecture can be extended to **RDNA4 Q=4/GQA4** to avoid full-context K/V reads; it is guarded, off by default, and subject to strict CPU-reference, mask-bound and fixed-work performance gates. No proven production speedup or resolved whole-step causal decomposition yet.

## Documents

- [Profiling plan](PROFILING-PLAN.md) - phase-by-phase procedures, commands, measurements, instrumentation targets, and completion gates.
- [MTP and HIP trace measurements (2026-11-11)](MEASUREMENTS-20261111.md) - collected server measurements, short-trace findings, and current uncertainty.
- [ROCPD synchronization analyzer](tools/rocpd_sync_analysis.py) - read-only per-thread HIP sync and GPU activity analysis; real-device validation assigned in handoff 0001.
- [Collaborative handoff 0001](../../.ai/handoffs/handoff-0001.md) - hardware-dependent synchronization root-cause investigation.
- [Report 0001](../../.ai/reports/report-0001.md) - root-cause diagnosis, source call stacks and 32K comparison.
- [Collaborative handoff 0002](../../.ai/handoffs/handoff-0002.md) - validate the opt-in upload profiler and determine if safe transfer batching is worthwhile.
- [Report 0002](../../.ai/reports/report-0002.md) - upload attribution and on-device short/32K/MTP diagnostics, with no safe transfer optimization proven.
- [ROCPD operator comparator](tools/rocpd_operator_compare.py) - read-only kernel-family and readback-window comparison; validated on synthetic SQLite data.
- [Collaborative handoff 0003](../../.ai/handoffs/handoff-0003.md) - identify why MTP verification costs increase between 32K and 98K context.
- [Report 0003](../../.ai/reports/report-0003.md) - matched 32K/98K MTP throughput and phase-selected HIP trace identifying flash-attention as a hypothesis.
- [Collaborative handoff 0004](../../.ai/handoffs/handoff-0004.md) - test the reversible HIP 256x256 GQA-grouping A/B with end-to-end MTP and GPU evidence.
- [GQA tile experiment helper](../../ggml/src/ggml-cuda/fattn-gqa-tune.h) - opt-in `GGML_HIP_FATTN_TILE_MAX_GQA=2`; disabled by default.
- [Report 0004](../../.ai/reports/report-0004.md) - experimental GQA tile changes MTP acceptance and output, without faster time per verification.
- [Collaborative handoff 0005](../../.ai/handoffs/handoff-0005.md) - long-KV deterministic validation and fixed-work target-only attention A/B.
- [Report 0005](../../.ai/reports/report-0005.md) - long-KV CPU-reference passes, slower GQA2 device-local work, and NVIDIA PCIe Xid 79/154 blocker.
- [Collaborative handoff 0006](../../.ai/handoffs/handoff-0006.md) - AMD-only experiment varying HIP 256x256 flash-attention KV parallel blocks.
- [KV split-work diagnostic helper](../../ggml/src/ggml-cuda/fattn-parallel-tune.h) - opt-in `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=4|8|16|32|64|96|128|192|256` on ROCm; 1/2 rejected for long-KV numerical failures. See [Report 0006](../../.ai/reports/report-0006.md).
- [Collaborative handoff 0007](../../.ai/handoffs/handoff-0007.md) - bounded AMD-only fixed-work test of larger-than-auto KV partitions with CPU-reference correctness.
- [Report 0007](../../.ai/reports/report-0007.md) - 96/128/192/256 blocks all slower than auto64 on both R9700s at 32K/98K; block-count tuning closed.
- [Long-KV scan / bandwidth estimate](ATTENTION-KV-ROOFLINE.md) - computed logical K/V scan rates compared with R9700 rated bandwidth; hypothesis only, not measured DRAM counters.
- [Server MTP step trace collector](../../tools/server/server-step-trace.h) - disabled-by-default host phase markers via `LLAMA_SERVER_STEP_TRACE=/absolute/path.tsv`; flush on clean server exit, no device waits.
- [Host phase comparison tool](tools/server_step_trace_analysis.py) - paired CPU timeline durations and monotonic interval exports; GPU critical path requires separate correlation.
- [Collaborative handoff 0008](../../.ai/handoffs/handoff-0008.md) - resume post-reboot mixed CUDA/ROCm MTP 32K/98K critical-path investigation.
- [Report 0008](../../.ai/reports/report-0008.md) - matched 32K/98K MTP reproduction and host stage timings; full GPU timeline unresolved.
- [Host step/ROCPD alignment tool](tools/rocpd_step_alignment.py) - read-only, clock-checked Radeon kernel overlap per MTP verification window; does not cover CUDA.
- [Collaborative handoff 0009](../../.ai/handoffs/handoff-0009.md) - preserve profiler host markers at request completion, recover NVIDIA importer and correlate HIP/CUDA work with MTP steps.
- [Report 0009](../../.ai/reports/report-0009.md) - Nsight importer recovery, 204-step HIP alignment and longer Radeon attention execution during target waits.
- [Collaborative handoff 0010](../../.ai/handoffs/handoff-0010.md) - split two HIP stream synchronizations per verification step and associate with scheduler/backend ordering.
- [Report 0010](../../.ai/reports/report-0010.md) - confirmed ordered ROCm0 then ROCm1 target waits; relative wait and kernel-family growth.
- [Collaborative handoff 0011](../../.ai/handoffs/handoff-0011.md) - opt-in 256x256 GQA4 Q=4 WMMA-vs-tile correctness and fixed-work A/B.
- [Report 0011](../../.ai/reports/report-0011.md) - WMMA no-go despite CPU-reference correctness; tile faster on both R9700s.
- [QSA sparse-attention RDNA4 audit](QSA-SPARSE-RDNA4-AUDIT.md) - relevant prior art, source gaps, sparse mask bounds and experiment gates.
- [Collaborative handoff 0012](../../.ai/handoffs/handoff-0012.md) - verify guarded 4-query RDNA sparse FlashAttention against CPU reference and fixed-work baseline.
- [Q4 RDNA sparse guard](../../ggml/src/ggml-cuda/fattn-sparse-q4-tune.h) - exact opt-in `GGML_HIP_QSA_SPARSE_Q4=1`; default dispatch unchanged.
- [WMMA Q4 diagnostic guard](../../ggml/src/ggml-cuda/fattn-wmma-q4-tune.h) - strict `GGML_HIP_FATTN_FORCE_WMMA_Q4=1`; no default changes.
- [MTP-shaped flash-attention tests](../../tests/test-backend-ops.cpp) - `GGML_TEST_MTP_FA=1` adds deterministic Q=4 CPU reference/performance cases at 32K and 98K.
- [K/V scan-rate estimator](tools/kv_attention_bandwidth.py) - logical byte model only, **not** measured DRAM counters.
- [Benchmark results](BENCHMARK-RESULTS.md) - measured throughput, context-depth scaling, limitations, and initial hypotheses.
- [Original auto-fit benchmark](raw/results-20261111-140511.md) - unmodified uploaded Markdown table (24 tests).
- [Original fixed-split benchmark](raw/results-fixed-split-20261111-142812.md) - unmodified uploaded Markdown table (6 tests).

## Reference configuration

| Item | Configuration |
| --- | --- |
| Model | Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (GGUF) |
| Inference engines | llama.cpp, CUDA + HIP dynamic backends |
| GPU order | CUDA0 / ROCm0 / ROCm1 |
| GPUs | NVIDIA RTX 4070 Ti Super (16 GiB), 2 x AMD Radeon AI PRO R9700 (32 GiB each) |
| Split | layer, requested proportions 4 / 7 / 7 |
| CPU / RAM | Ryzen 9 9950X, 64 GiB DDR5-5600 |
| Context | 196,608 tokens in the actual server baseline |
| KV cache | F16 K and V |
| Flash attention | enabled |
| MTP | Q8_0 draft on CUDA0, up to 3 draft tokens in the server baseline |
| Source | `1648d641174927634dfb72135b220e4052393729` |
| Build | `1648d6411 (12296)`, shown in both benchmark results |
| Build directory | `/home/chris/workspace/llama.cpp-perf/build-mixed` |

## Key observed result

At 163,840 tokens of prefilled depth, the auto-fit target-only benchmark reports:

- `pp1024`: **794.88 ± 1.76 t/s**, down **65.3%** from depth zero.
- `tg512`: **32.46 ± 0.01 t/s**, down **29.6%** from depth zero.

This is evidence of context-length sensitivity, **not** evidence that any one kernel (QSA, experts, transfers, or scheduling) is the cause.

**Next checkpoint:** complete [handoff 0012](../../.ai/handoffs/handoff-0012.md): validate sparse QSA mask union + gathered K/V for RDNA4 Q=4/GQA4 using CPU-reference 32K/98K cases and repeated device-local A/B; verify real model selection bounds before any mixed-GPU inference. All production defaults unchanged; use **28 build jobs** and no GitHub Actions.

The original `/home/chris/workspace/llama.cpp` checkout is not part of this experiment and should remain unchanged.
