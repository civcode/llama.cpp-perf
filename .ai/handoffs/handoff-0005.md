# Handoff 0005 — isolate flash-attention compute from MTP acceptance

## Objective

Establish whether changing the HIP 256x256 flash-attention GQA grouping is numerically sound at long KV lengths and whether it improves **fixed-work GPU/inference latency** independently of the changing MTP acceptance and generated text. Use the new deterministic long-KV backend tests as the first correctness gate; no default optimization without a validated end-to-end critical-path benefit.

## Context and Chat AI implementation

- Shared fork `civcode/llama.cpp-perf`; branch `docs/llama-perf`. Read `.ai/WORKFLOW.md`, `.ai/reports/report-0003.md`, `.ai/reports/report-0004.md` and this exact handoff. Chat AI transfers exclusive write ownership when this handoff is published.
- Report 0004 confirmed the opt-in HIP switch `GGML_HIP_FATTN_TILE_MAX_GQA=2` changes `flash_attn_tile<256,256,4,4,false>` to `<256,256,4,2,false>` on both R9700s, while the default path is unchanged. 32K MTP throughput increased ~15.6%, 98K increased ~5.9%, but acceptance rose and per-verification-step wall time was slightly *slower*. GPU flash-attention time per observed ~4-logit-sized readback also rose in both cases. These are not established kernel performance gains.
- Repeated matched 512-token MTP samples: 32K default ~58.98 t/s / 204 steps / 306 accepted, experiment ~68.19 t/s / 176 steps / 334 accepted; 98K default ~40.79 t/s / 203 steps / 307 accepted, experiment ~43.20 t/s / 190 steps / 321 accepted. Repeated within-arm output deterministic but **between-arm sampled text diverged**. A plausible response is not a long-context finite-logit or GDN state check.
- The agent's existing CPU-reference `FLASH_ATTN_EXT` tests covered 256x256, GQA4, KV512, batch3, F16 K/V (two permutations), not 32K or 98K KV.
- Chat AI implemented opt-in deterministic **long-KV CPU-reference tests** in `tests/test-backend-ops.cpp`: `GGML_TEST_LONG_FA=1` adds 256x256, 3 query tokens, F16 K/V, masked GQA4 at KV=4096/16384/32768 plus GQA8 at KV=8192. The deterministic initializer generates identical input tensors from their names and indices in independent processes, and a consistent nontrivial F16 attention mask. Cases are tagged in test parameters as `long_kv_deterministic=1`, allowing focused selection. No new cases are added to the default test suite. The Chat AI examined relevant code but **cannot build the full mixed CUDA/HIP backend in its environment**; the agent must run these tests and correct implementation/compiler issues if any. This is source-level test coverage, not a claim of numerical equivalence.

## Tasks — coherent on-device validation and causal comparison

1. **Validate the opt-in tests.** Sync the exact handoff revision, keep original checkout/models/reference `build-mixed` unchanged. Build `test-backend-ops`, `llama-bench` and `llama-server` in `build-perf` with **28 parallel build jobs**, e.g. `cmake --build build-perf -j 28 --target test-backend-ops llama-bench llama-server`. On each R9700 and with separate processes for default and `GGML_HIP_FATTN_TILE_MAX_GQA=2`, run focused CPU-reference cases such as:

   `GGML_TEST_LONG_FA=1 ./build-perf/bin/test-backend-ops test -b ROCm0 -o FLASH_ATTN_EXT -p 'long_kv_deterministic=1'`

   Repeat with ROCm1 and the experimental environment switch. Capture per-case error and CPU reference threshold (`test_flash_attn_ext` uses 5e-4 NMSE); fix genuine test/compiler bugs. If 32K CPU reference is prohibitively expensive, start with 4K and 16K and document why, rather than silently skipping.
2. **Unconfound throughput with target-only fixed work.** Compare env unset vs `GGML_HIP_FATTN_TILE_MAX_GQA=2` using identical target-only `llama-bench` at depth 32768 and 98304 (and optionally depth0) with fixed CUDA0/ROCm0/ROCm1 split 4/7/7, F16 K/V, no draft model, fixed `-n 128` generation, one warmup, and ~3 unprofiled repetitions per arm. Record effective layer/tensor placement and t/s. Unlike MTP, the timed target-only decode steps have no speculative acceptance confound, although benchmark work shape still differs from MTP verification. Restart processes between arms because the option is process-static.
3. **Check numerical stability on the actual model.** Compare a controlled model-evaluation or fixed-token scoring workload (e.g. deterministic prompt and teacher-forced logit/probability scoring) between default and alternative, ideally at both 32K and 98K. Use saved prompt fixtures without committing them. Examine logits/probabilities for numerical differences, non-finite values and sampled-token divergence near small score margins. If direct logits or GDN-state comparison cannot be implemented practically, distinguish that limitation from the achieved CPU-reference checks. MTP temperature-zero runs may be informative but cannot substitute for fixed-token numerical equivalence.
4. **Tie to dependency timing where possible.** Relate any true target-only fixed-work difference to the longer-context `flash_attn_tile` GPU duration, per-device busy union and transfer/event sequencing. Scope GPU traces narrowly and acknowledge profiler overhead. Existing 0004 ROCPD traces lacked a CUDA timeline and precise MTP verification boundaries; don't manufacture causal attribution from summed kernel time.
5. **Make a decision, not a speculative default.** If the alternative is numerically acceptable and shows reproducible *fixed-work* speedups, identify a scoped path to implement/enable it safely and verify MTP/GDN. If it does not, retain the disabled-by-default diagnostic flag and recommend the next plausible flash-attention improvement (KV tiling, tile work distribution, sequence-length masking/occupancy, or cross-device execution), supported by the new tests. Do not remove synchronizations, change attention math or promote the GQA2 flag as a default simply because MTP acceptance increased.
6. **Report results.** Commit any useful test fixes or tightly justified source changes and `.ai/reports/report-0005.md`. Include per-case reference errors/pass/fail; matched 32K/98K fixed-work t/s and any per-kernel comparison; numerical checks/limits; remaining causal gap; the narrowest recommended next code target. Save bulky profiler databases, prompts and logs outside Git under `/home/chris/workspace/llama-perf-results/iteration-0005/`. Push to `docs/llama-perf` and return the standard completion pointer.

## Constraints

- Work exclusively in `civcode/llama.cpp-perf` on `docs/llama-perf`, with exclusive repository write ownership. No PRs, merges, force-pushes, GitHub Actions, workflow changes or running CI.
- Build with **28 parallel jobs** on the inference system. Do not modify `/home/chris/workspace/llama.cpp/`, model weights or reference `build-mixed`.
- Opt-in experiment must remain disabled by default. CPU-reference tolerance alone does not prove full-model trajectory equivalence. Ensure the baseline and experimental branches use the same fixed workload before assigning a speedup.
- Keep tests targeted, not exhaustive; resolve ordinary build/profile problems autonomously. Report genuine blockers and unsupported correctness claims explicitly.

## Completion criteria

- Long-KV deterministic test cases build and run successfully (or a documented concrete blocker), with per-arm/reference checks for both R9700s when available.
- Matched 32K and 98K target-only no-draft fixed-work A/B shows whether changed attention grouping is actually faster outside MTP acceptance effects.
- Useful fixed-token or model-level numerical evidence, or a precise explanation of what's still unverified.
- Evidence-based go/no-go for the tile experiment and next narrow optimization target.
- `.ai/reports/report-0005.md` committed and pushed to `docs/llama-perf`.

## Reporting

`Iteration: 0005 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0005.md | Summary: <1-2 sentences>`
