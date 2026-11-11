# LLAMA-PERF: IQ3_XXS performance engineering

Profiling and benchmark record for Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS on mixed CUDA + HIP inference using [civcode/llama.cpp-perf](https://github.com/civcode/llama.cpp-perf).

**Status (2026-11-11):** two target-only llama-bench baselines, MTP-enabled server measurements at 8K/32K/98K context, and an initial ROCprofiler HIP trace have been captured. Handoff 0001 identified benchmark-driven HIP waits; handoff 0002 found no safe tensor-upload optimization; handoff 0003's matched 32K/98K MTP traces identified flash-attention as the leading incremental GPU-time hypothesis, but not a proven causal bottleneck. Handoff 0004 verified that the alternate HIP 256x256 GQA tile grouping changes kernel selection and MTP acceptance, but does not improve time per verification step; the apparent throughput gain is confounded by different generated tokens. Handoff 0005 validated deterministic long-KV 256x256 flash-attention cases on both R9700s and measured the experimental GQA2 tile as 6–8% slower on matching device-local fixed work. Mixed three-device inference is blocked by the NVIDIA PCIe Xid 79/154 failure. Report 0006 shows that tested HIP RDNA attention KV partition overrides are slower than the automatic policy, and 1 and 2 blocks fail selected long-KV correctness checks; both experiments remain disabled by default. No validated model-level speedup or long-context numerical-equivalence claim has been established.

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
- [KV split-work diagnostic helper](../../ggml/src/ggml-cuda/fattn-parallel-tune.h) - opt-in `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=1|2|4|8|16` for ROCm; 1 and 2 fail selected long-KV CPU-reference tests, and all tested overrides are slower than the default. See [Report 0006](../../.ai/reports/report-0006.md).
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

**Next checkpoint:** investigate long-KV attention K/V reuse and the reduction cost around the default work distribution. Verify NVIDIA health before resuming three-GPU inference and model-level numerical checks. Use **28 build jobs** locally; do not dispatch GitHub Actions.

The original `/home/chris/workspace/llama.cpp` checkout is not part of this experiment and should remain unchanged.
