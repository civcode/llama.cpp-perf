# Report 0006

Status: Completed. No production change is justified: every tested parallel-block override is slower than the automatic policy on both R9700s, and 1 block fails the CPU-reference threshold at both lengths; 2 blocks fails at 98K. Both experimental options remain unset and disabled by default. Source revision tested: `79d305f2098f488ad583817d0998b8c81863045f`.

## Setup and validation

- Started from the requested revision on `docs/llama-perf`. Built with `cmake --build build-perf -j 28 --target test-backend-ops`; it passed. The host-only selector test passed:
  `c++ -std=c++17 -Wall -Wextra -pedantic -Iggml/src/ggml-cuda docs/LLAMA-PERF/tools/test_fattn_parallel_tune.cpp -o /home/chris/workspace/llama-perf-results/iteration-0006/test_fattn_parallel_tune && /home/chris/workspace/llama-perf-results/iteration-0006/test_fattn_parallel_tune`.
- GPUs: ROCm0 and ROCm1 are AMD Radeon AI PRO R9700, gfx1201, 64 compute units each. All tests were single-device. The GQA override `GGML_HIP_FATTN_TILE_MAX_GQA` was unset in every run. Each baseline also had `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS` unset; each variant used a new process.
- `nvidia-smi -L` initially reported that it could not communicate with the driver, as in the prior blocker; later during this iteration it listed the RTX 4070 Ti SUPER and PCI bus `00000000:04:00.0`. No reset, reboot or system change was attempted. Visibility alone does not establish that CUDA inference or the earlier PCIe fault is resolved. No CUDA or mixed-GPU inference was run.
- For each GPU, KV length and setting, the test command was `env -u GGML_HIP_FATTN_TILE_MAX_GQA GGML_TEST_LONG_FA=1 GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=<1|2|4|8|16> ./build-perf/bin/test-backend-ops test -b <ROCm0|ROCm1> -o FLASH_ATTN_EXT -p 'kv=<32768|98304>.*long_kv_deterministic=1'`. For baseline, replace the assignment with `-u GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS` before the environment assignments. Each process selected exactly one test.
- A temporary test-runner diagnostic printed the NMSE and counted nonfinite GPU/CPU output pairs; its patch and all logs are saved outside Git. It was reverted, and the final target was rebuilt with 28 jobs. The uninstrumented binary independently reproduced the 98K baseline and 4-block passes and 1/2-block failures on both GPUs. No production source fix was required.

## CPU-reference error

256x256, GQA4, two KV heads, three queries, F16 K/V and deterministic F16 mask. NMSE threshold is `5e-4`; values below are identical on both GPUs. Every run had zero nonfinite pairs among 6,144 output elements. Tests compare each output to the CPU reference, not model logits or recurrent state.

| Blocks | KV 32768 NMSE | Result | KV 98304 NMSE | Result |
| --- | ---: | --- | ---: | --- |
| Auto | 7.88595e-6 | pass | 2.48310e-5 | pass |
| 1 | 5.68898e-4 | **fail** | 2.13888e-3 | **fail** |
| 2 | 2.80523e-4 | pass | 9.31584e-4 | **fail** |
| 4 | 1.43104e-4 | pass | 4.05297e-4 | pass |
| 8 | 6.93208e-5 | pass | 2.19967e-4 | pass |
| 16 | 3.18227e-5 | pass | 9.91058e-5 | pass |

A pass does not prove accuracy for other attention distributions, shapes or model state. Smaller partitions increase the accumulation length per block and change rounding/reduction order; this is consistent with the observed errors but is not a proven quantitative root cause. The overrides are experimental and must not be treated as generally numerically safe.

## Fixed-work device-local timings

Used the same deterministic shapes with `env -u GGML_HIP_FATTN_TILE_MAX_GQA GGML_TEST_LONG_FA=1 GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=<setting> ./build-perf/bin/test-backend-ops perf -b <device> -o FLASH_ATTN_EXT -p 'long_kv_deterministic=1'`; baseline again unsets the parallel setting. Each process selected both KV lengths, warmed up once and repeated a fixed attention graph for at least one second. Each entry is median milliseconds per operation over three fresh processes, with the range in parentheses; the percentage is median latency change relative to that device/length baseline. Order was `auto,1,2,4,8,16`, then reversed, then `4,auto,8,1,16,2` for each device. This measures graph wall time per synthetic attention op, including execution/launch overhead, not isolated kernel time, MTP steps or tokens. Failed-correctness variants are shown only to characterize the experiment.

| Device | KV | Blocks | Median ms/op (range) | Change |
| --- | ---: | --- | ---: | ---: |
| ROCm0 | 32768 | auto | 0.11450 (0.11426-0.11462) | baseline |
| ROCm0 | 32768 | 1 | 2.67326 (2.62158-2.68113) | +2234.7% |
| ROCm0 | 32768 | 2 | 1.51052 (1.51011-1.51095) | +1219.2% |
| ROCm0 | 32768 | 4 | 0.63670 (0.63472-0.64698) | +456.1% |
| ROCm0 | 32768 | 8 | 0.31715 (0.31614-0.31737) | +177.0% |
| ROCm0 | 32768 | 16 | 0.16308 (0.16295-0.16319) | +42.4% |
| ROCm0 | 98304 | auto | 0.34234 (0.34160-0.34261) | baseline |
| ROCm0 | 98304 | 1 | 7.78331 (7.77152-7.83557) | +2173.6% |
| ROCm0 | 98304 | 2 | 4.10164 (4.09790-4.10462) | +1098.1% |
| ROCm0 | 98304 | 4 | 2.26050 (2.25727-2.27907) | +560.3% |
| ROCm0 | 98304 | 8 | 1.05714 (1.05370-1.05959) | +208.8% |
| ROCm0 | 98304 | 16 | 0.53652 (0.53613-0.53707) | +56.7% |
| ROCm1 | 32768 | auto | 0.11458 (0.11351-0.11466) | baseline |
| ROCm1 | 32768 | 1 | 2.61302 (2.61180-2.61653) | +2180.5% |
| ROCm1 | 32768 | 2 | 1.49364 (1.48125-1.49967) | +1203.6% |
| ROCm1 | 32768 | 4 | 0.62387 (0.61999-0.62497) | +444.5% |
| ROCm1 | 32768 | 8 | 0.31087 (0.30977-0.31123) | +171.3% |
| ROCm1 | 32768 | 16 | 0.16054 (0.16034-0.16137) | +40.1% |
| ROCm1 | 98304 | auto | 0.34279 (0.34150-0.34285) | baseline |
| ROCm1 | 98304 | 1 | 7.72451 (7.71399-7.76658) | +2153.4% |
| ROCm1 | 98304 | 2 | 4.06937 (4.06564-4.07450) | +1087.1% |
| ROCm1 | 98304 | 4 | 2.23023 (2.22756-2.25034) | +550.6% |
| ROCm1 | 98304 | 8 | 1.01094 (1.01070-1.02264) | +194.9% |
| ROCm1 | 98304 | 16 | 0.53124 (0.53090-0.53285) | +55.0% |

No allocation failures or obvious memory-pressure errors were reported. Host scheduling, clocks and runtime overhead were not independently isolated; these timings are not a bandwidth or occupancy measurement.

## Launcher and next hypothesis

`rocprofv3 --kernel-trace -f csv` on each test case confirmed `flash_attn_tile<256,256,4,4,false>` launches on both devices. Grid dimensions in **workgroups**, calculated from CSV grid/workgroup thread sizes, were `(1,64,2)` for auto at both lengths and `(1,N,2)` for every requested N. The workgroup was `(32,8,1)`; the trace reports 152 VGPR and 26,112 bytes LDS per block. A separate combine kernel appeared for auto and N > 1; N = 1 wrote directly without it. For example ROCm0 32K test traces measured the auto tile at 145 us and combine at 7.6 us, versus N=16 tile at 193 us and combine at 3.84 us. Single traced launches have profiler overhead and different conditions from the repeated graph timings above.

Auto launches 128 workgroups versus 32 at N=16 and only 2 at N=1, on 64-CU devices; reduced available parallelism and longer per-block KV loops are plausible reasons for the regressions. The combine savings do not compensate in these traces. Grid size is not measured occupancy, and traces do not establish K/V bandwidth, cache reuse or the precise bottleneck. The existing heuristic is preferable for these shapes, not necessarily all shapes. A next code-level hypothesis is to measure K/V traffic and reuse in `fattn-tile.cuh` near the auto 64-block policy, including partial-result reduction cost, before changing math or scheduling.

Keep both tuning options disabled. If NVIDIA health is confirmed separately, still perform the previously blocked matched three-device fixed-work target-only benchmark, fixed-token logits and recurrent-state comparison, and cross-device timeline; neither these single-op results nor device visibility establish end-to-end correctness or speedup. Raw measurements, profiler CSVs, builds and temporary instrumentation patch are in `/home/chris/workspace/llama-perf-results/iteration-0006/`. No GitHub Actions, model changes or mixed-GPU inference were run.
