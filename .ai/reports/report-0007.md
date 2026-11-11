# Report 0007

Status: Completed. No production change is justified: on both R9700 GPUs, larger HIP flash-attention parallel-block overrides (96/128/192/256) are consistently slower than the automatic policy (auto64) at both fixed-work long-KV test lengths, while correctness remains acceptable for all tested settings. This closes the block-count tuning direction for this path. Source revision tested: `c5e37ad1a9ca6fa7f0ef17aba8dab09ba5b2d3e4`.

## Scope and constraints

- Repository/branch: `civcode/llama.cpp-perf`, `docs/llama-perf`.
- Built with 28 jobs: `cmake --build build-perf -j 28 --target test-backend-ops`.
- Selector test rebuilt and passed with C++17 + warnings-as-errors:
  `c++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iggml/src/ggml-cuda docs/LLAMA-PERF/tools/test_fattn_parallel_tune.cpp -o /home/chris/workspace/llama-perf-results/iteration-0007/test_fattn_parallel_tune && /home/chris/workspace/llama-perf-results/iteration-0007/test_fattn_parallel_tune`.
- Experimental settings kept disabled by default. All baseline runs unset both tuning env vars.
- No three-GPU mixed inference, no reset/reboot/driver changes.

## Test matrix

- Operator/test: `FLASH_ATTN_EXT` with `GGML_TEST_LONG_FA=1`, deterministic long-KV shape.
- Devices: ROCm0 and ROCm1 (AMD Radeon AI PRO R9700).
- KV lengths: 32768 and 98304.
- Settings: `auto` (unset), explicit `64` control, and larger candidates `96`, `128`, `192`, `256`.
- Perf command pattern:
  `env -u GGML_HIP_FATTN_TILE_MAX_GQA [GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=<setting>|-u GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS] GGML_TEST_LONG_FA=1 ./build-perf/bin/test-backend-ops perf -b <ROCm0|ROCm1> -o FLASH_ATTN_EXT -p 'long_kv_deterministic=1'`
- Correctness command pattern:
  `... test -b <device> -o FLASH_ATTN_EXT -p 'kv=<32768|98304>.*long_kv_deterministic=1'`

## Fixed-work performance results (3 process reps per setting)

Median us/op with min-max in parentheses. Percent is vs that device/KV auto baseline.

| Device | KV | Blocks | Median us/op (range) | Change vs auto |
| --- | ---: | --- | ---: | ---: |
| ROCm0 | 32768 | auto | 115.04 (113.78-115.18) | baseline |
| ROCm0 | 32768 | 64 | 114.70 (113.79-115.33) | -0.30% |
| ROCm0 | 32768 | 96 | 117.51 (116.49-117.88) | +2.15% |
| ROCm0 | 32768 | 128 | 118.97 (118.40-119.50) | +3.42% |
| ROCm0 | 32768 | 192 | 124.01 (123.70-124.65) | +7.80% |
| ROCm0 | 32768 | 256 | 127.87 (127.18-128.49) | +11.15% |
| ROCm0 | 98304 | auto | 343.23 (341.54-343.63) | baseline |
| ROCm0 | 98304 | 64 | 342.81 (341.55-344.00) | -0.12% |
| ROCm0 | 98304 | 96 | 355.95 (355.65-356.02) | +3.71% |
| ROCm0 | 98304 | 128 | 352.47 (351.97-353.05) | +2.69% |
| ROCm0 | 98304 | 192 | 359.62 (359.19-360.77) | +4.78% |
| ROCm0 | 98304 | 256 | 368.82 (368.57-369.65) | +7.46% |
| ROCm1 | 32768 | auto | 115.56 (113.49-115.95) | baseline |
| ROCm1 | 32768 | 64 | 115.33 (114.06-116.64) | -0.20% |
| ROCm1 | 32768 | 96 | 118.03 (116.39-118.95) | +2.14% |
| ROCm1 | 32768 | 128 | 119.74 (118.55-120.09) | +3.62% |
| ROCm1 | 32768 | 192 | 124.08 (123.87-125.44) | +7.37% |
| ROCm1 | 32768 | 256 | 127.96 (127.12-129.36) | +10.73% |
| ROCm1 | 98304 | auto | 344.83 (341.85-345.21) | baseline |
| ROCm1 | 98304 | 64 | 344.21 (342.30-346.17) | -0.18% |
| ROCm1 | 98304 | 96 | 357.25 (356.88-357.33) | +3.60% |
| ROCm1 | 98304 | 128 | 353.53 (352.35-354.45) | +2.52% |
| ROCm1 | 98304 | 192 | 360.28 (359.54-362.33) | +4.48% |
| ROCm1 | 98304 | 256 | 369.43 (369.37-370.64) | +7.13% |

Observations:
- Auto remains effectively equivalent to explicit 64, consistent with current heuristic.
- 96/128/192/256 are all slower on both GPUs and both KV lengths.
- Degradation grows with block count, strongest at 192/256.

## Correctness gate

- Deterministic CPU-reference long-KV tests passed (`OK`) for auto, 64, 96, 128, 192, 256 on both ROCm0 and ROCm1 at KV 32768 and 98304.
- This binary reports pass/fail against the existing threshold (`NMSE <= 5e-4`) but does not print NMSE values in default output; no failures were observed.
- Additional fallback check with `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=1` and `=2` also passed in this build, and profiling confirmed they map to auto64 launch geometry (see next section), i.e. unsupported values are rejected and fall back.

## Grid and kernel-trace confirmation (rocprofv3)

`rocprofv3 --kernel-trace -f csv` confirmed `flash_attn_tile<256,256,4,4,false>` launch geometry scales as expected for accepted overrides:

- auto and explicit 64: grid workgroups `(1,64,2)`
- 96/128/192/256: grid workgroups `(1,N,2)` for N in {96,128,192,256}
- workgroup shape fixed at `(32,8,1)`
- combine kernel grid remained `(3,8,1)`

Representative ROCm0 flash kernel durations at KV=98304:
- auto: ~469 us
- 64: ~467 us
- 96: ~493 us
- 128: ~472 us
- 192: ~476 us
- 256: ~482 us

Combine kernel duration rises with larger block count (about 7.8 us at auto/64 to about 23 us at 256), matching the direction of end-to-end ms/op regressions.

Fallback confirmation:
- `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=1` -> grid `(1,64,2)`
- `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=2` -> grid `(1,64,2)`

## NVIDIA visibility note

Some runs reported CUDA device discovery (RTX 4070 Ti SUPER) in startup logs, but this is visibility only and not evidence of recovered NVIDIA stability. No CUDA inference and no three-GPU mixed runs were executed.

## Conclusion and next priority

- **Go/No-go:** No-go for larger parallel-block overrides as a tuning path for this 256x256 HIP RDNA flash-attention case.
- 96/128/192/256 do not deliver measurable improvement; they regress latency while staying numerically acceptable.
- Keep defaults unchanged and keep experimental override disabled by default.
- Recommended next narrow target: gather direct K/V memory-traffic and occupancy evidence for the current auto64 path, then revisit only if metrics suggest a concrete bottleneck. Mixed-GPU critical-path validation remains blocked on demonstrably stable CUDA.

Raw logs, traces and summaries are stored outside Git at:
`/home/chris/workspace/llama-perf-results/iteration-0007/`.
