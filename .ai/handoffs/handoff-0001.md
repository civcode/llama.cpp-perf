# Handoff 0001 — mixed-CUDA/HIP synchronization root cause

## Objective

Determine, with sufficient on-device evidence to guide a safe change, **why a single host thread spends 82.1% of the short-context decode window inside `hipStreamSynchronize`**, and whether there is a practical way to reduce *end-to-end* latency without harming correctness or mixed-device overlap. Include a controlled short-vs-long context comparison if feasible. Do not assume that waiting for actual GPU work is avoidable.

## Context and completed Chat AI work

- Shared project: `civcode/llama.cpp-perf`, working branch **`docs/llama-perf`**. Read `.ai/WORKFLOW.md`, [profiling notes](../../docs/LLAMA-PERF/MEASUREMENTS-20261111.md) and [profiling plan](../../docs/LLAMA-PERF/PROFILING-PLAN.md) before executing. Current base model code comes from llama.cpp commit `1648d6411`.
- Chat AI prepared `docs/LLAMA-PERF/tools/rocpd_sync_analysis.py` and targeted `test_rocpd_sync_analysis.py`. Both were locally syntax-checked and fixture-tested; **the real ROCm trace remains untested by Chat AI**.
- Local hardware: Ryzen 9950X, RTX 4070 Ti Super 16 GiB on CUDA0, two R9700 gfx1201 32 GiB each on ROCm0/1; 80 GiB aggregate VRAM. ROCm Core 10.1 at `/opt/rocm/core-10.1` (HIP and ROCprofiler-SDK 1.4.1). NVIDIA Nsight Systems 2022.4.2 may be too old for full CUDA GPU tracing; its CPU sampling is restricted by perf permissions. Avoid changing system security settings without reason.
- Existing raw HIP trace: `/home/chris/workspace/llama-perf-results/20261111-143704/hip/short-0/iq3_short_results.db` (118.49 MiB). **Do not add it, model weights, server responses containing sensitive content, or giant raw profiler files to Git.** Keep raw data outside the repo and commit only concise summaries / small scripts.
- The 3,981 sync calls were all on CPU thread 331789 during 6.0–9.371 s of the traced target-only `tg128` generation window; total 2,769.20 ms, 82.1% of that window. GPU copy, kernel and graph activity are also present. Details and caveats are in the measurement notes.
- MTP-enabled server results at 8K/32K/98K: generation 75.01/59.10/39.28 t/s; at 32K and 98K draft acceptance stayed ~50% with ~203–204 verification steps, but per-step wall time rose from ~42.47 to ~64.08 ms. The HIP trace above is **target-only depth 0**, not an MTP trace at 98K.

## Tasks — one coherent local investigation

1. **Synchronize and establish baseline.** Verify branch and the exact handoff commit supplied in the transfer message; preserve any unrelated local uncommitted changes. Check the existing `build-mixed/`, model files and profiler versions. Verify the old short ROCPD DB is readable and the server is not unexpectedly running before loading a second model.
2. **Run and improve the committed analyzer on the real trace.** Start with:

   `python3 docs/LLAMA-PERF/tools/rocpd_sync_analysis.py --db /home/chris/workspace/llama-perf-results/20261111-143704/hip/short-0/iq3_short_results.db --start-seconds 6 --json /home/chris/workspace/llama-perf-results/20261111-143704/analysis/sync-short.json`

   Fix any differences in the local ROCPD schema, and preserve the fixture tests or extend them when appropriate. Obtain same-host-thread API neighbor patterns before and after sync, times and distribution, and kernel / memory copy activity on both R9700s. Identify whether the 970 KiB device-to-host readbacks are logits via sizes/callsite evidence if practical, but do not treat size alone as proof.
3. **Source-level attribution.** Inspect `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_copy_input`, `ggml_backend_sched_compute_splits`, scheduler event waits), `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_backend_cuda_synchronize` and device copy functions), and `src/llama-context.cpp`. Use low-overhead, opt-in diagnostic counters/timestamps, debugger call stacks, profiler correlation IDs, or equivalent **only if necessary** to find where synchronizations originate and what GPU/transfer work each waits for. Never insert forced HIP/CUDA synchronization merely for measurement. Keep the production reference build `build-mixed/` unchanged; use independent `build-perf/` for any instrumentation. No source modification is mandatory if trace analysis is sufficient.
4. **Validate across phases / depth.** In the same controlled mixed-GPU placement and with the server stopped before a standalone profiling process, acquire a limited additional trace or phase-isolated sample that can distinguish prefill from decode. Prefer a small 0-depth vs 32K comparison with 128 decode tokens; include 98K only if profiling size/runtime is manageable. Avoid another full startup-to-98K trace if it would be huge. Capture exact actual tensor placement, effective context sizes, trace overhead and whether HIP GPU timelines show concurrent work or gaps. When possible, compare unprofiled t/s, not trace-distorted timings. If practical, correlate with a real MTP request rather than assuming target-only results generalize to speculative verification. NVIDIA CUDA-side correlation is useful but optional if the installed Nsight toolchain cannot do it reliably.
5. **Form and, only if justified, test one safe hypothesis.** Distinguish essential compute wait from redundant sync on the critical path. If a small well-understood, reversible change is supported by evidence, implement it in the fork, build locally, run a focused short-context and longer-context A/B comparison against baseline, and validate finite outputs/GDN state, inference behavior and MTP if touched. If confidence is insufficient, stop before changing scheduling semantics and report a ranked evidence-based optimization proposal instead. An unsupported attempt to delete `hipStreamSynchronize` is explicitly **not** a success criterion.
6. **Deliver concise trace-backed conclusions.** Commit any useful analyzer fixes, optional safe prototype and `.ai/reports/report-0001.md` to the working branch, and push. Include the metrics and local paths (not giant raw trace bytes), C++ callsites supported by evidence, what was tested, what was not, and the single most promising next optimization or unresolved blocker. Return the standard short completion message.

## Constraints

- **Exclusive write ownership transfers to the local Coding Agent for this turn.** Chat AI will make no further repository writes until the completion report is relayed.
- Work only in `/home/chris/workspace/llama.cpp-perf/`, on `docs/llama-perf`, except local experiments/results outside the repo. Do not modify `/home/chris/workspace/llama.cpp/`, the model GGUF files, or the baseline `build-mixed/` cache/configuration.
- Preserve actual server configuration when running server workloads: main model on `CUDA0,ROCm0,ROCm1`, draft on `CUDA0`, split `4,7,7`, F16 K/V, context 196608, `--spec-draft-n-max 3`, `--fit-target 768`; note requested versus fitted GPU placement. Server must be stopped before any separate whole-model benchmark to avoid VRAM pressure.
- Profiling records blocking API time, not directly removable overhead; avoid summing concurrent GPU durations into wall time. ROCPD duration units are nanoseconds. HIP trace does not show all NVIDIA runtime activity.
- Run builds/tests/scripts locally; do **not** run, dispatch or enable GitHub Actions. Do not introduce automatic workflow triggers. Do not force-push, merge or create PRs as part of this handoff. No system-wide privilege changes without a strong need.
- Avoid extensive cross-matrix validation; use targeted cases sufficient to support the diagnosis. Resolve ordinary technical problems autonomously, escalating only a genuine blocker or a consequential user decision.

## Completion criteria

- Analyzer runs successfully on the existing real ROCPD DB, with checked thread/call counts and a useful summary stored outside Git.
- Clear diagnosis or bounded, evidence-backed hypotheses for the observed 3,981 synchronization calls, supported by real SQL trace and specific backend code paths.
- At least one controlled test or justified limitation addresses whether the sync behavior changes between short and longer contexts, without conflating MTP and target-only benchmarks.
- If a code change is attempted, it builds and passes targeted inference correctness and performance comparisons, and has a rollback/revert story. Otherwise provide the most actionable proposed change and the evidence still missing.
- `.ai/reports/report-0001.md` committed and pushed with relevant source/analysis changes; report contains short results, exact commit, branch, remaining limits, and locations of local raw data.

## Reporting

Create `.ai/reports/report-0001.md` with the key synchronization patterns and timing distributions, C++ callsites, at least one controlled comparison and recommended next action. Avoid giant logs and unsupported claims. Then send the user **only** the short completion pointer:

`Iteration: 0001 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0001.md | Summary: <1-2 sentences>`.
