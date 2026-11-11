# Handoff 0006 — AMD-only long-KV flash-attention split-work experiment

## Goal

With NVIDIA unavailable after PCIe Xid 79/154, determine whether the **existing HIP RDNA 256x256 flash-attention kernel** can run long-KV fixed-work attention faster by changing only the number of parallel KV work blocks (gridDim.y). This is a controlled device-local experiment, **not** a substitute for the three-GPU MTP verification critical path. No GPU reset, reboot, PCIe configuration changes, or attempts to benchmark the mixed system are required.

## Prior evidence

- In `.ai/reports/report-0005.md` the opt-in alternative `GGML_HIP_FATTN_TILE_MAX_GQA=2` passed long-KV CPU reference at 4K–98K but was **6–8% slower** than the existing grouping on matching ROCm0/1 microbenchmarks. Keep it unset/default for this experiment. Prior MTP t/s increases stemmed from changed acceptance and sampled output; no fixed-work speedup was established.
- Report 0005 confirmed the NVIDIA GPU dropped off PCIe at 18:44:39 (Xid 79 followed by Xid 154). The mixed-device target-only and full-model logits/state tests remain blocked. Do not claim the hardware has recovered without checking, and do not modify hardware/system state.
- `ggml/src/ggml-cuda/fattn-common.cuh::launch_fattn` currently chooses the number of `parallel_blocks` via occupancy and a wave-efficiency heuristic; flash-attention tile mode uses the non-stream-K branch and combines partial outputs with the existing reduction. It may be useful to test whether varying the number of block partitions changes long-KV performance without modifying math.
- Chat AI added disabled-by-default **`GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=1|2|4|8|16`** in `fattn-common.cuh` and helper `fattn-parallel-tune.h`. This overrides `parallel_blocks` only for 256×256 attention on HIP RDNA using existing combine logic, clamped to actual KV tile count; unset/unknown values and other shapes keep original behavior. A standalone host test is at `docs/LLAMA-PERF/tools/test_fattn_parallel_tune.cpp`. **The full HIP build and device tests have not been performed by Chat AI.** Do not claim a speedup before running them.
- The deterministic backend test cases are now opt-in for both `test` and `perf` modes: `GGML_TEST_LONG_FA=1` adds test-mode 256x256 GQA4 F16 KV at 4096, 16384, 32768, **98304** plus GQA8 8192; perf mode adds only KV 32768 and 98304 GQA4. The deterministic inputs and F16 mask are identical across runs; test mode uses the normal `5e-4` NMSE CPU-reference threshold. The 98K deterministic case passed with temporary source modifications in Report 0005 but is now committed into the opt-in suite. Keep default tests unchanged.

## Tasks

1. **Validate code locally using 28 build jobs.** Sync `civcode/llama.cpp-perf` `docs/llama-perf` to the handoff commit. Read `.ai/WORKFLOW.md`, Reports 0004/0005 and this handoff. Run the host-only C++17 helper test with `-Iggml/src/ggml-cuda`. Build the targeted test with `cmake --build build-perf -j 28 --target test-backend-ops` (or a separate local build if required). Correct compiler/portability issues in the opt-in implementation. Do not modify the original checkout, `build-mixed`, models, or system configuration.
2. **Check deterministic numerical behavior.** On both working Radeon GPUs, compare baseline (unset) to parallel settings `1`, `2`, `4`, `8`, `16` for 256x256 GQA4, KV32768 and KV98304. Run focused CPU-reference tests, for example:
   `GGML_TEST_LONG_FA=1 GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=4 ./build-perf/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p 'long_kv_deterministic=1.*kv=98304'`
   NOTE: exact filter ordering is `vars()` output with `kv=...` **before** `long_kv_deterministic=1`, so use `-p 'kv=98304.*long_kv_deterministic=1'` or just `-p 'long_kv_deterministic=1'`. Verify the filter selects actual cases (not zero tests). Capture NMSE and finite-output results. If the 98K CPU reference cost is high, prune variants after poor 32K results, documenting choices.
3. **Benchmark matched AMD-only fixed work.** Use `test-backend-ops perf -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=32768.*long_kv_deterministic=1'` (and KV98304), with `GGML_TEST_LONG_FA=1` and different `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS` values set per clean process. Repeat for ROCm1. Compare baseline and selected variants in alternating process order with at least three iterations or equivalent stable timing; record us/op, variability and any measurements where runtime overhead or memory pressure is significant. Do not conflate one GPU operation with an MTP verification step or generation token.
4. **Confirm actual launcher behavior when practical.** Obtain minimal ROCprofiler kernel-grid observations or a temporary controlled diagnostic to ensure the intended `gridDim.y` changes. Do not change production default or hardcode a new block count. Verify any difference in outputs is within the CPU-reference tolerance, including at long KV.
5. **Inspect performance contributors.** If one partition count gives a reproducible device-local speedup, examine occupancy, block count, K/V data reuse and reduction overhead to judge whether its cost can translate to a real inference step when NVIDIA recovers. If all variants lose, explain why the existing auto heuristic appears preferable and identify the next code-level hypothesis. Avoid other math changes, scheduler changes, or source changes not supported by the measurements.
6. **Report and hand back.** Create `.ai/reports/report-0006.md` with exact test/build commands, per-device/per-KV NMSE and ms/op, measured baseline vs candidate, code decision and what still needs full mixed-GPU testing. Save raw profiler data outside Git under `/home/chris/workspace/llama-perf-results/iteration-0006/`; commit useful portability/test fixes plus report, push only to `docs/llama-perf`, then send the standard short completion message.

## Constraints and scope

- **Exclusive repo:** `civcode/llama.cpp-perf`, branch `docs/llama-perf`. No PRs, merges, force-push, workflow executions or GitHub Actions.
- **28 parallel jobs** for local builds; no CUDA0 or multi-GPU inference experiments while GPU remains unavailable. Running only independent R9700 backend tests is permitted if those devices are stable.
- No NVIDIA driver reset, reboot, power/PCIe fixes or privileged profiler access. Hardware diagnosis/recovery is for the user to decide separately.
- Existing GQA switch `GGML_HIP_FATTN_TILE_MAX_GQA=2` stays disabled for the split-work experiment. This new split-work switch also remains disabled by default.
- Preserve existing source behavior when experiment options are absent. CPU-reference correctness at 98K does not prove full-model recurrent state or numerical equivalence, and single-op gains do not establish end-to-end gains.

## Completion criteria

- Host test and HIP targeted build pass (or precise documented blocker).
- Focused deterministic 32K/98K attention CPU-reference results and fixed-work timing on at least one R9700, preferably both, showing whether block partitioning is beneficial.
- Verified launcher selection where possible and an explicit go/no-go decision on a production change.
- `.ai/reports/report-0006.md` committed/pushed to `docs/llama-perf` with any fixes, then exclusive write ownership returned.

## Completion message

`Iteration: 0006 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0006.md | Summary: <1–2 sentences>`
