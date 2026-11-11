# Handoff 0011 - RDNA WMMA vs tile for four-query MTP attention

## Objective

Test one specific, reversible flash-attention hypothesis at the **MTP verification batch boundary**: on Radeon AI PRO R9700, 256x256 head size, GQA4 and Q=4, the existing dispatch chooses `flash_attn_tile` because the WMMA threshold is `Q * effective_GQA > 16`; **4 * 4 = 16** falls just below it. The existing WMMA kernel supports 256x256 with 4 query columns and 4 grouped heads. An opt-in experiment forces this already-compiled implementation for **only** that shape. Determine whether it is (a) numerically acceptable and (b) faster on matched long-KV fixed work before any full three-GPU model benchmark. **Leave production selection and defaults unchanged.**

## Background and prior measurements

- Fork: `civcode/llama.cpp-perf`, branch `docs/llama-perf`. Read `.ai/WORKFLOW.md`, Reports 0009 and 0010, and this handoff. The local Coding Agent has exclusive write ownership after transfer.
- Report 0010 mapped the two ordered target synchronizations to ROCm0 and ROCm1, with HIP-profiled 32K-to-98K **median wait increases +4.417 and +3.404 ms**, respectively. Flash-attention kernel overlap grew about +2.734/+3.307 ms inside the corresponding waits. Other GPU kernel work and host/backend dependencies remain present; these are not proven causal latency shares.
- Previous GQA grouping and KV parallel-block experiments were slower on matched fixed work. Do **not** retest those settings. Both `GGML_HIP_FATTN_TILE_MAX_GQA` and `GGML_HIP_FATTN_TILE_PARALLEL_BLOCKS` stay unset.
- Report 0007's device-local 3-query FP16 baseline roughly implied 581-587 GB/s logical K/V scan rate vs 640 GB/s rated R9700 bandwidth, **not physical DRAM traffic**. The new A/B is a WMMA vs scalar tile algorithm choice (different kernel), not an arbitrary block-count change. A WMMA win is not assured when bandwidth dominates.
- The 256-head FP16 **tile** kernel reuses K/V within a GQA group. On the current dispatch path, simply changing the cache to Q8_0 with Q=4 typically requires f16 materialization for the tile/MMA kernels (`ggml/src/ggml-cuda/fattn.cu` `need_f16_K/V`); it is **not** a clean no-code way to measure reduced attention HBM reads. Do not change KV format in this experiment.

## Code already implemented by Chat AI

1. New `ggml/src/ggml-cuda/fattn-wmma-q4-tune.h`: pure guarded selector. Returns true only for HIP RDNA with WMMA support; K/Q/V head size 256; Q=4, sequence count 1, GQA4; existing `gqa_opt_applies`; FP16 K/V; env setting **exactly** `GGML_HIP_FATTN_FORCE_WMMA_Q4=1`. Any other value or shape returns false.
2. `ggml/src/ggml-cuda/fattn.cu`: after the normal kernel-support and kv_rows checks, an `#if defined(GGML_USE_HIP)` override returns `BEST_FATTN_KERNEL_MMA_F16` under the guard above. No kernel math is edited and no extra synchronizations are added. Baseline with the variable **unset** is byte-for-byte the original dispatch policy.
3. Host-only C++17 test `docs/LLAMA-PERF/tools/test_fattn_wmma_q4_tune.cpp`. **Compiled/passed** with `-Wall -Wextra -Werror -pedantic` by Chat AI, exercising active and all major inactive shapes.
4. `tests/test-backend-ops.cpp` now provides opt-in `GGML_TEST_MTP_FA=1` deterministic **Q=4** cases at KV=32768/98304 for both `test` CPU-reference and `perf` fixed-work modes, 2 KV heads, GQA4, F16 K/V and mask. Uses existing `test_flash_attn_ext_long_kv` initializer; normal NMSE tolerance **5e-4**. Existing `GGML_TEST_LONG_FA=1` Q=3 cases and default suite unchanged.
5. `docs/LLAMA-PERF/tools/kv_attention_bandwidth.py` calculates a **logical one-pass K/V scan estimate** from KV length, head count, dimensions, storage bytes and elapsed whole-op time; makes no DRAM counter claim. Host-only 3 unit tests **passed** locally. This model should only be applied to fixed-work attention latency, not the full MTP verification step.
6. Full HIP build, GPU correctness and timing **not available to Chat AI**, delegated here.

## Tasks

1. **Read, build, validate selector:** Sync the exact handoff revision and build `cmake --build build-perf -j 28 --target test-backend-ops llama-server llama-bench`; compile/run C++ selector with host `c++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iggml/src/ggml-cuda docs/LLAMA-PERF/tools/test_fattn_wmma_q4_tune.cpp -o /tmp/test_fattn_wmma_q4_tune && /tmp/test_fattn_wmma_q4_tune`. Run `python3 -m unittest discover -s docs/LLAMA-PERF/tools -p 'test_kv_attention_bandwidth.py'`. Fix ordinary compile/portability issues, retaining guard semantics and production default.
2. **Confirm kernel actually changes:** Execute 32K Q=4 operator tests with baseline env unset and `GGML_HIP_FATTN_FORCE_WMMA_Q4=1` in **separate processes** (process-static variable). Check a brief ROCprofiler kernel name/dispatch on each R9700: default must launch tile, experimental must launch `flash_attn_mma_f16` or its true build-specific name. If the experimental condition does not activate, diagnose whether it is due to `gqa_opt_applies`, WMMA availability, shape, or test masks; **do not silently assume it ran**.
3. **CPU-reference correctness gate FIRST:** For both ROCm0 and ROCm1, test Q=4 KV32768 and KV98304 with baseline and opt-in: `GGML_TEST_MTP_FA=1 GGML_HIP_FATTN_FORCE_WMMA_Q4=1 ./build-perf/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=98304.*nb=4.*long_kv_deterministic=1'`. Verify the pattern selects at least one actual test. Require finite outputs and NMSE <= 5e-4. Any failure is a **no-go** for model-level tests; report the error without altering tolerances or math semantics.
4. **Fixed-work device-local performance:** Benchmark the same `GGML_TEST_MTP_FA=1 ... test-backend-ops perf -b ROCm0/ROCm1 -o FLASH_ATTN_EXT -p 'kv=32768.*nb=4.*long_kv_deterministic=1'` (and KV98304) for **unset vs enabled**, alternating processes and >=3 repetitions for any promising result. Measure whole-op us/op, variance, profiler overhead and actual kernel families. A previous Q=3 baseline is a control, **not** the target MTP verification shape. If the candidate is slower, materially unstable or inaccurate on either Radeon, reject it and **stop without another full-model MTP run**.
5. **Full mixed-GPU model A/B only on a real win:** If the WMMA variant passes long-KV CPU reference and is reproducibly faster on **both** R9700s at the relevant 98K fixed-work shape, run matched **unprofiled** 32K/98K IQ3_XXS+MTP requests on CUDA0/ROCm0/ROCm1 with same 4/7/7 split, F16 KV, 196608 ctx, exact saved uncached 512-token requests, draft CUDA0 max 3, sampling seed/config from Report 0008. Alternate clean-process env unset/enabled, 2-3 repeats, capture **actual verification count**, 512 completion tokens, accepted/drafted, per-step generation time, output hashes, and host phase medians if desired (separate traced runs; never use traced runs as throughput baseline). Flag differences in acceptance/output and do not report throughput-only wins from altered token trajectories. Stronger teacher-forced fixed-token logits and recurrent-state comparisons are **required before considering promotion**. Auto-fit remains ineffective when tensor-split explicitly supplied.
6. **Bounded profiling & safety:** Use earlier process watchdog, liveness and health checks. Small CUDA/HIP preflights; no `nvprof`, no long unbounded jobs, no orphaned profiler/server processes. No privilege/driver/PCIe changes. If CUDA falls off PCIe again, stop heavy runs and document Xid blocker.
7. **Decision and report:** Write `.ai/reports/report-0011.md` with compiler/test results, actual kernel dispatch, exact per-device Q4 32K/98K correctness and us/op, baseline vs WMMA relative differences, and the clearest next optimization direction. If the guarded WMMA is slower or numerically unacceptable, retain disabled-by-default as a diagnostic (or remove if it creates maintenance risk) and **close this algorithm-dispatch idea**. No blind GQA/block tuning. Save raw GPU data outside Git under `/home/chris/workspace/llama-perf-results/iteration-0011/`. Commit/push report and useful code fixes to the branch, return ownership.

## Constraints

- Exclusive fork/branch only, no PRs, merges, force-pushes, GitHub Actions or workflow triggers. Direct commits/pushes follow `.ai/WORKFLOW.md`; do not require the upstream `AGENTS.md` acknowledgment, as the fork-specific files were deliberately removed in commit `83559c8e`.
- Use **28 parallel build jobs**, preserve source checkout `/home/chris/workspace/llama.cpp/`, weights and untouched `build-mixed`.
- All experiment flags **unset** by default. The `GGML_HIP_FATTN_FORCE_WMMA_Q4` switch is an opt-in diagnostic; it may alter numerical results. Do not promote production defaults based on device-local microbenchmarks or one sampled output alone.
- The Q=4 operator case is **verification-shaped**, not identical to the full recurrent/MTP graph, so CPU reference at 98K is necessary but insufficient proof of model-level numerical state correctness.
- After the report is pushed, return exclusive write ownership; no additional acknowledgement gates for direct pushes.

## Completion criteria

- Host tests and targeted ROCm test-backend-ops/llama-server build pass (or concrete reproducible blocker).
- A confirmed baseline-to-WMMA kernel switch on both R9700s (or precise reason not activated).
- CPU-reference and fixed-work timings at Q4, 32K and 98K for both Radeon devices.
- If a real win exists, matched fixed-work and full-model A/B with numerical/state caveats. If not, clear no-go, no expensive full-model run.
- `.ai/reports/report-0011.md` committed/pushed and repo ownership returned.

## Completion message

`Iteration: 0011 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0011.md | Summary: <1-2 sentences>`
