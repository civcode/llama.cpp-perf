# Report 0011

Status: Completed. **No-go for this WMMA dispatch:** it passes the targeted CPU-reference gate but is consistently slower on both R9700s. No three-GPU MTP benchmark was run. Production dispatch defaults remain unchanged; the override remains an opt-in diagnostic.

## Build and correctness

Synced to `b54d5082769b3ffad044d1200aa440660669e02c`. The mixed CUDA/HIP build of `test-backend-ops`, `llama-server` and `llama-bench` passed with 28 jobs; the C++17 selector test and all three bandwidth-tool unit tests passed. The UI download fell back to the cached latest archive, as in Report 0010. CUDA0 and ROCm0 ADD preflights passed.

The existing comparison accepts matching infinities and could silently accept a non-finite NMSE. The test harness now explicitly rejects non-finite FLASH_ATTN_EXT outputs and NMSE when `GGML_TEST_MTP_FA` is set and prints NMSE; normal test behavior and tolerances are unchanged. After rebuilding, each of the eight separate 32K/98K, baseline/WMMA, ROCm0/ROCm1 runs selected exactly one deterministic Q=4 test, passed, and had finite outputs. The CPU-reference NMSEs, identical on both devices, were:

| KV | Tile | WMMA | Limit |
| --- | ---: | ---: | ---: |
| 32768 | 8.2480e-6 | 6.7205e-7 | 5e-4 |
| 98304 | 2.52416e-5 | 1.94954e-6 | 5e-4 |

These operator tests are not model-level logits or recurrent-state validation.

## Dispatch and performance

ROCm profiler kernel traces of the 32K test on **each** device show baseline `flash_attn_tile` plus `flash_attn_combine_results`; with the flag, `flash_attn_ext_f16<256, 256, 4, 4, ...>` (defined in `fattn-mma-f16.cuh`) plus `flash_attn_stream_k_fixup_uniform`. Thus the actual switch is confirmed. The 32K profiled perf runs also used the flag truly unset for baseline.

Unprofiled whole-op microseconds per run, three successive baseline/WMMA process pairs, with `GGML_HIP_FATTN_FORCE_WMMA_Q4` unset versus exactly `1`. Both other HIP tuning flags were unset. Values are medians with ranges; percentages are WMMA median latency increase, not throughput gain.

| Device | KV | Tile median (range) | WMMA median (range) | Change |
| --- | ---: | ---: | ---: | ---: |
| ROCm0 | 32768 | 115.20 (115.04–115.20) | 144.51 (144.15–144.88) | +25.4% |
| ROCm0 | 98304 | 344.64 (344.52–344.75) | 399.27 (398.15–399.81) | +15.9% |
| ROCm1 | 32768 | 115.34 (115.03–115.52) | 142.25 (141.03–142.63) | +23.3% |
| ROCm1 | 98304 | 345.01 (344.73–345.13) | 398.49 (397.24–398.85) | +15.5% |

Each process used the same Q=4, GQA4, two KV heads, 256 dimensions, FP16 K/V and mask; the perf harness chose its own iteration count. A prior three-pair run using an empty rather than unset baseline setting yielded the same qualitative result. For comparison, one profiled 32K perf run per configuration measured ROCm0 118.21/147.83 and ROCm1 118.14/146.89 us tile/WMMA, roughly 2–3% higher than the unprofiled medians; these profiled timings are not the A/B baseline.

The next direction should target actual KV data movement or another independently justified bottleneck, with a new bounded, fixed-work hypothesis; do not promote this dispatch or blindly retune GQA/block count. No production defaults changed, no GitHub Actions ran, and no server or profiler was started for model tests. All runs were bounded with timeout/kill deadlines; raw logs and profiler CSVs are under `/home/chris/workspace/llama-perf-results/iteration-0011/`.
