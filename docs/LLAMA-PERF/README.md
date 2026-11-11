# LLAMA-PERF: IQ3_XXS performance engineering

Profiling and benchmark record for Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS on mixed CUDA + HIP inference using [civcode/llama.cpp-perf](https://github.com/civcode/llama.cpp-perf).

**Status (2026-11-11):** two target-only llama-bench baselines captured; MTP-enabled server measurement and GPU kernel profiling are pending. No optimization or correctness result is claimed from the current benchmarks.

## Documents

- [Profiling plan](PROFILING-PLAN.md) - phase-by-phase procedures, commands, measurements, instrumentation targets, and completion gates.
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

**Next checkpoint:** collect an MTP-enabled server baseline with timing and speculative-acceptance counters, then acquire short HIP and CUDA traces. All profiling and tests run locally; do not dispatch GitHub Actions.

The original `/home/chris/workspace/llama.cpp` checkout is not part of this experiment and should remain unchanged.
