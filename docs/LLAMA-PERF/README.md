# LLAMA-PERF: IQ3_XXS performance engineering

Profiling and benchmark record for Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS on mixed CUDA + HIP inference using [civcode/llama.cpp-perf](https://github.com/civcode/llama.cpp-perf).

**Status (2026-11-11):** two target-only llama-bench baselines, MTP-enabled server measurements at 8K/32K/98K context, and an initial ROCprofiler HIP trace have been captured. Handoff 0001 identified benchmark-driven HIP waits; handoff 0002 validated the mixed-GPU upload profiler and found no safe transfer optimization. Handoff 0003 is directed at the 32K-to-98K MTP verification-step slowdown. No speedup or numerical-correctness claim has been established.

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

**Next checkpoint:** complete [handoff 0003](../../.ai/handoffs/handoff-0003.md), quantifying the additional MTP verification-step work at 98K versus 32K before selecting a kernel or scheduler optimization. Use **28 build threads** for local builds; do not dispatch GitHub Actions.

The original `/home/chris/workspace/llama.cpp` checkout is not part of this experiment and should remain unchanged.
