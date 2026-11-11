# Handoff 0007 — test larger HIP RDNA KV partitions against auto64

## Objective

Determine whether allowing **more** parallel KV partitions than the existing automatic 64-block policy improves long-KV 256x256 HIP flash-attention on the two Radeon AI PRO R9700 GPUs. This is a *last bounded block-count experiment*: do not try unlimited values or modify attention arithmetic or general scheduling. Keep production defaults unchanged and avoid the three-GPU model while NVIDIA PCIe stability is unverified.

## Evidence and source changes prepared by Chat AI

- Read `.ai/WORKFLOW.md`, `.ai/reports/report-0005.md`, `.ai/reports/report-0006.md` and this handoff. Work only in `civcode/llama.cpp-perf`, branch `docs/llama-perf`, at the exact handoff commit; you have exclusive write ownership until completion.
- Report 0006: 256x256, GQA4, F16 K/V, Q=3, deterministic mask. The existing automatic HIP RDNA policy launched `flash_attn_tile<256,256,4,4,false>` with **grid workgroups `(1,64,2)`** on each R9700. At KV=32768 / 98304, auto time was ~0.1145 / ~0.3425 ms per fixed-work attention operation. Overrides 1,2,4,8,16 were all slower; 1 failed CPU-reference NMSE at both lengths and 2 failed at 98K. The validated automatic path is preferable to every tested *smaller* partition count. The observed per-block loop length and low workgroup count motivate testing the untested **larger** partitions.
- Chat AI modified `ggml/src/ggml-cuda/fattn-parallel-tune.h`: disabled-by-default `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS` now additionally recognizes **32,64,96,128,192,256** and explicitly **rejects `1` and `2`**, falling back to automatic selection for those values because Report 0006 found long-KV correctness failures. Existing 4/8/16 values remain available solely for diagnostics. This helper is used only on HIP RDNA head 256x256; absent/unrecognized values preserve the existing heuristic. The setting is parsed in the host dispatch and may be process-static; restart between settings.
- Expanded `docs/LLAMA-PERF/tools/test_fattn_parallel_tune.cpp`. The host-only selector test **compiled and passed with C++17 `-Wall -Wextra -Werror -pedantic` locally**. No GPU build or performance test for larger partitions has been performed by Chat AI.
- Retain default GQA grouping by unsetting `GGML_HIP_FATTN_TILE_MAX_GQA`; do not mix two experiments. The opt-in `GGML_TEST_LONG_FA=1` cases in `tests/test-backend-ops.cpp` provide deterministic 32K/98K CPU-reference tests and perf-mode fixed-work measurements. Report 0006 contains exact command patterns and raw data locations.

## Tasks

1. **Build/validate:** Sync fork branch to the handoff commit and build `test-backend-ops` with **28 parallel jobs**, e.g. `cmake --build build-perf -j 28 --target test-backend-ops`. Run the selector test with C++17. Confirm unsupported `1`/`2` now choose auto, not the failed variant. Preserve the original checkout, weights, and `build-mixed`.
2. **Baseline control:** With both options unset, rerun matched deterministic single-R9700 `FLASH_ATTN_EXT` perf cases at KV 32768 and 98304; verify timings align with Report 0006 and that grid is still `(1,64,2)` for the relevant GQA4 shape. `64` should be a useful explicit-control arm if auto remains 64.
3. **A/B only larger partitions:** First test `96` and `128` at KV32768/98304 on ROCm0. If either gives a repeatable improvement, extend to ROCm1, `192` and `256`; if both lose materially, use a small 192/256 probe for a complete trend and stop. Prefer 3 alternating process repetitions for promising candidates, not an exhaustive matrix. Time must be fixed-work attention ms/op, never MTP acceptance-derived t/s. Confirm actual gridDim.y (workgroup count) with a short existing ROCprofiler kernel trace or a temporary local diagnostic if needed.
4. **Correctness gate:** Run focused `GGML_TEST_LONG_FA=1` CPU-reference cases for selected high-block candidates on both R9700s at KV32768/98304; require finite outputs and NMSE <= 5e-4. Test 64/default as comparison. If increased partitions introduce errors or force excessive scratch allocation, stop and document; do not permit any default change. Larger partitions may change reduction order and floating-point results.
5. **Interpret and report:** Identify whether launch/combination overhead, occupancy or K/V access dominates the observed difference, recognizing that grid sizes and summed kernel durations do not prove bandwidth saturation. If higher partitions do not beat auto, **close the block-count tuning direction** and recommend a new evidence-gathering target (K/V memory traffic, occupancy metrics or mixed-GPU critical path once CUDA is demonstrably stable), instead of launching another arbitrary grouping experiment. If a repeatable safe win emerges, retain as opt-in until model-level and three-device validation is possible.
6. **Handoff:** Write `.ai/reports/report-0007.md` with build/test outcomes, exact settings, both device/KV results where meaningful, grid confirmations, NMSE, speed/overhead changes, NVIDIA visibility if observed (not a recovery claim), and a clear go/no-go plus next narrow priority. Keep raw logs/profiler databases outside Git under `/home/chris/workspace/llama-perf-results/iteration-0007/`. Commit and push report and only justified code corrections to `docs/llama-perf`, and return standard completion pointer.

## Constraints

- Work exclusively in our fork and branch. No PRs, merges, force-push, GitHub Actions, CI dispatch or automatic workflow triggers.
- Use **28 parallel build jobs**. Do not run three-GPU mixed inference, GPU resets, reboots or privileged system/driver changes while the NVIDIA Xid 79/154 incident remains unresolved.
- Both tuning options are disabled by default. `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS=1|2` is now rejected; `4|8|16` remains experimental despite already demonstrated regressions. Do not promote a production default based on single-op data.
- Perform the work autonomously and keep scope bounded. The current CUDA-visibility report is **not** proof of CUDA health.

## Completion criteria

- GPU build and selector tests pass; unsupported unsafe settings fall back to auto.
- At least 96/128 and a bounded 192/256 probe are tested against auto64 at 32K/98K, unless a specific blocker prevents it.
- Strongest high-block candidate (if any) passes targeted CPU-reference long-KV checks.
- Explicit conclusion: high-block tuning worth further work or closed; remaining mixed-GPU validation depends on stable CUDA0.
- `.ai/reports/report-0007.md` committed and pushed, with ownership returned.

## Completion message

`Iteration: 0007 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0007.md | Summary: <1–2 sentences>`
