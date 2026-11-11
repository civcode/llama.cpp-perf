# Flash-attention long-KV scan estimate after handoff 0007

**Status:** analytical hypothesis, **not** a measured memory-bandwidth counter or a model-level speedup. Source: [Report 0007](../../.ai/reports/report-0007.md), [Report 0006](../../.ai/reports/report-0006.md), and `ggml/src/ggml-cuda/fattn-tile.cuh` / `ggml/src/ggml-cuda/fattn-common.cuh`.

## Measured device-local baseline

Each Radeon AI PRO R9700 (gfx1201) ran a deterministic `FLASH_ATTN_EXT` operation with Q=3, two KV heads, GQA4, K/V head dimensions 256, FP16 K and V, and a deterministic F16 mask. Baseline dispatch used `flash_attn_tile<256,256,4,4,false>` and an automatic 64-way KV partition. Both R9700s showed similar results:

| Device | KV positions | Baseline median µs/op | Modelled K+V bytes if read once | Nominal scan rate |
| --- | ---: | ---: | ---: | ---: |
| ROCm0 | 32,768 | 115.04 | 64 MiB | 583.35 GB/s |
| ROCm0 | 98,304 | 343.23 | 192 MiB | 586.56 GB/s |
| ROCm1 | 32,768 | 115.56 | 64 MiB | 580.73 GB/s |
| ROCm1 | 98,304 | 344.83 | 192 MiB | 583.84 GB/s |

The computed *logical* K+V bytes for a single scan are

```text
KV positions × 256 elements/head × 2 bytes/FP16 × 2 (K and V) × 2 KV heads
```

The *nominal scan rate* is simply that byte count divided by the recorded whole-operation latency; 1 GB = 10^9 bytes. These are calculated values, **not** physical DRAM transfer rates. The launch may reuse cached K/V values or read data more than once, and the time includes kernel launch and reduction overhead. The actual traffic, cache-hit rate, bandwidth, arithmetic bottlenecks and cross-device dependencies have **not** been measured.

AMD lists **640 GB/s peak external memory bandwidth** for the R9700: [AMD product specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). The modelled logical scan rate is **~91–92% of this rated bandwidth**. *Conditional on a single off-chip read of every K and V element*, the theoretical streaming memory floor would be ~105 µs at 32K and ~315 µs at 98K, versus measured ~115 and ~343 µs. This is a **reason to investigate a memory-bandwidth limit**, not proof that external DRAM is actually saturated: cache reuse, clock variation and actual memory transactions can invalidate the simple floor.

## Interpretation and follow-up

- Reports 0006 and 0007 established that forcing the existing 64-block policy to 1/2/4/8/16 or 96/128/192/256 is slower for these shapes. The 1-block and 2-block settings also failed selected CPU-reference correctness checks and are now rejected by the diagnostic selector.
- The controlled baseline latency scales approximately linearly with KV length (3× data and ~3× time). With a nominal scan rate close to the rated external bandwidth, **additional block-count search is not warranted**. The 6–8% slower GQA grouping from Report 0005 is also not a production optimization.
- The next possible AMD-only measurement is actual VRAM/cache transactions, sustained clocks and occupancy using read-only profiling counters **if supported and available**; do not conflate estimated traffic with counters. Pursue implementation only if it reveals a concrete reducible data movement or computation path.
- The stronger next **model-level** decision depends on demonstrating that CUDA0 is stable after the Xid 79/154 PCIe incident. Then rerun matched 32K/98K three-device target-only benchmarks and phase-aligned MTP verification, with actual CUDA+HIP critical-path attribution and numerical/state checks. CUDA enumeration alone is not a recovery test.
- **No production changes** are justified by the block-count A/B experiments. Both HIP tuning switches remain disabled by default. Keep source and baseline artifacts within the original fork/workflow; local builds use 28 jobs, never GitHub Actions.
