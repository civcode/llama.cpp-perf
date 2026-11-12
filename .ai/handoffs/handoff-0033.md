# Handoff 0033 — layer-11 causal ablation and numerical drift localization

## Objective

The user approved investigating **dense-versus-sparse attention numerical differences**, especially layer-11 arithmetic, later layer-15 top-k cutoff sensitivity, GDN R/S recurrent state drift and verification logits. This is NOT another 24-question LongBench run, not a throughput test, and NOT approval for nonzero quality-loss tolerance or production promotion.

Question to answer with honest evidence: **under identical Q/K/V/mask inputs and matched natural MTP history, is the sparse arithmetic difference at audited HIP0 QSA layer 11 a sufficient/necessary contributor to downstream top-k mask changes and later GDN/logit drift?** Isolating a conditional effect is different from establishing a unique root cause or globally exact equivalence.

Reports to read first: .ai/reports/report-0016.md, report-0017.md, report-0018.md, report-0019.md, report-0021.md, report-0030.md, report-0032.md and docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md. The prior 32K results are from SEPARATE process cohorts and must not be described as one confirmed full causal trace.

## Ownership, starting revision and hardware stability

Repository civcode/llama.cpp-perf; branch docs/llama-perf. Starting commit: exactly as supplied in Chat AI's transfer prompt. Read .ai/WORKFLOW.md. Coding Agent has exclusive write ownership after user transfers the prompt; commits report .ai/reports/report-0033.md and source fixes then returns ownership. No concurrent Chat AI writes.

NVIDIA RTX 4070 Ti SUPER with current operator-selected PCIe **Gen3**, observed Gen3 x4 under load, and two AMD Radeon AI PRO R9700 HIP GPUs. CUDA0/HIP0/HIP1 mixed 4/7/7. The machine had TWO NVIDIA Xid79/154 events on November 12, then completed TWO traced 24-question Gen3 sparse runs (0030/0032) without observed device errors. Those successes do NOT prove permanent recovery; protect hardware. No switching back to Gen4/Auto, node reboots, resets, driver changes or stress tests.

## Prior numeric evidence / hypotheses

1. Report 0016–17: **HIP0 layer 11** is earliest observed changed QSA F32 output, with exact identical Q/K/V+F16 mask bytes before launch. Max dense–sparse abs 0.00535035; dense-energy NMSE 8.97579e-7. Exact-input F64 mathematical reference did NOT show sparse uniformly worse: dense-reference NMSE 9.99118e-7, sparse-reference 4.44979e-7. Thus this is a changed arithmetic path, **not a confirmed implementation defect**.
2. Report 0018: later **HIP0 layer 15** has changed indexer score rankings near top-k cutoff; first swap pool 1040 for pool 6316 with a dense cutoff margin 0.0034008 and max paired score delta 0.016921. Difference may be downstream perturbation; prior records do not establish a unique source.
3. Report 0019: natural aligned MTP dense A/B recurrent/logit dumps were exactly identical; sparse changed GDN R 1296/1728, S 1296/1728, P 0/48, and all64 F32 full-vocab logits (max abs 0.918641; max KL(dense||sparse) 0.0106312 nats, 0 top1 flips/5 top5 changes). Earliest recorded GDN difference layer12 at step0 **after target decode**. Recorded R/S deviations through rejection/replay are not evidence of incorrect rollback; the source may have diverged earlier in that same decode.
4. Reports 0030/0032: fully frozen 24-question sparse outputs repeated exactly, sparse scored 14/24 as dense. This does NOT erase the numerical differences above and does not justify a production threshold.

## Code implemented by Chat AI — local validation required

1. New **docs/LLAMA-PERF/tools/analyze_qsa_gdn_drift_layers.py** with **test_analyze_qsa_gdn_drift_layers.py** (14 tests). Reuses compare_qsa_model_state.read_manifest/values/diff_stats. Requires dense A/B/sparse logical-event equality and complete tensor-key alignment, validates raw dense A/B bytes (signed-zero included) on selected roles/steps/layers, then produces sparse first OBSERVED differing GDN R/S/P row plus by-step/phase/layer/role counts, per-row max abs and NMSE. Selection options --steps e.g. 0 or 0,6,8,9, --layers e.g. 12,32,44, --max-rows; 512 rows and 1-GiB aggregate read cap. Writes external exclusive-create JSON and never accesses GPU. Test locally. Do not confuse the first after-decode observed state with first causal arithmetic error.
2. New **ggml/src/ggml-cuda/fattn-sparse-q4-gqa12-layer-filter.h** strict pure helper and **docs/LLAMA-PERF/tools/test_qsa_gqa12_layer_filter.cpp** standalone C++17 grammar tests. Valid diagnostic strings **GGML_HIP_QSA_GQA12_LAYER_FILTER=only:11** and **except:11**, with numeric audited node names qsa_flash_attn_ilN. Missing/invalid names and malformed filters fail closed in eligible QSA GQA12 paths.
3. Updated **ggml/src/ggml-cuda/fattn.cu** so both GQA12 sparse selector checks use identical diagnostic gate. With the filter ABSENT, original experimental GQA12 selector behavior is unchanged. With filter present, it REQUIRES existing GGML_HIP_QSA_SPARSE_Q4_GQA12=1 AND GGML_QSA_MASK_AUDIT, and selects only or all except the requested eligible QSA layer. Audit is the pre-existing graph-disabling diagnostic. The GQA4 flag/forced WMMA remain unset. This is a **diagnostic-only dispatch selection change**, not a change to sparse kernel numerical code, production default or overflow guards. It has NOT yet been CUDA/HIP compiled by Chat AI.

## A. Host/CPU validation and reuse already captured evidence FIRST

- Sync exact starting revision. Compile/run standalone CPU C++ test with local c++ -std=c++17 -Wall -Wextra -Werror; compileall Python; direct new Python tests and appropriately bounded existing QSA tests. Fix any build/logic mistakes in the new diagnostic code. A mixed CUDA/HIP Release build (llama-server and necessary test targets) is required BEFORE any new GPU experiment, and verify no changes to production dispatch when the layer filter is absent. No GitHub Actions.
- Inspect old external captures/logs/comparator output in /home/chris/workspace/llama-perf-results/iteration-0016 through iteration-0019 (and 0022/0023 if useful). Reuse existing tools compare_qsa_attn_out.py, replay_qsa_captured_attention.py, compare_qsa_topk_trace.py, compare_qsa_logit_distributions.py and compare_qsa_model_state.py. Run the NEW GDN layer-drift localizer on existing 0019 three-way manifests, with bounded selected steps/layers (first step0 and natural rejection/replay if recorded). Preserve exact dense-repeat and logical-event matching; select by checkpoint/role, not by observed outcome. If raw artifacts are absent, document that gap, do not reconstruct or invent them.
- Separate observations by their originating run: 0016/17 exact-input attention, 0018 score selection, 0019 natural GDN/logits. Describe where existing evidence stops being causal. Record raw inputs/output outside Git only.

## B. Conservative fresh matched intervention under current PCIe Gen3 ONLY if useful

A single coherent causal pilot is authorized if A cannot settle the conditional hypotheses (likely). Use one predeclared existing ~32K long-context target/MTP fixture (prefer the exact Report 0019 input, source fingerprint, chat format), with **five separately bounded fresh conditions**, no repetition loops:
  1. dense A (all experimental sparse flags unset);
  2. dense B (separate baseline repeat, same settings);
  3. sparse-all (only GQA12=1; no layer filter);
  4. sparse-only11 (GQA12=1, QSA layer filter only:11);
  5. sparse-except11 (GQA12=1, QSA layer filter except:11).

Keep actual model shards/draft, CUDA0/HIP0/HIP1 4/7/7, F16 KV, appropriate 32K-plus context capacity (avoid 98K), max draft3, seed42, greedy, reasoning setting, no prompt cache and graph-disabled GGML_QSA_MASK_AUDIT absolutely identical. Ensure each condition has a fresh bounded server/process, disjoint output/capture dir and expected env (including explicit UNSET of layer filter in dense A/B and sparse-all). Use the 0032 saved model/binary fingerprints as provenance where appropriate; rebuilt server SHA naturally changes and must be recorded rather than claimed identical. Do NOT reuse a pre-Gen3 dense baseline as a matched run.

**Prefer minimal diagnostic capture**: first natural target 4-query long-context group and, if feasible, one or two accepted/rejected groups; cap observer rows/layers instead of repeating 0019's 5.3-GiB full per-process capture. Existing bounded mask, attention Q/K/V/output and layer-15 top-k score tools should run at the same checkpoints as practical. If capturing GDN, use the existing one-layer filter (from earlier experiments) or a few identified selected checkpoints; don't indiscriminately dump every recurrent row on all five runs. Logits must be actual pre-sampling target output, not a forced teacher prediction. Do NOT force draft acceptance or rollback to manufacture comparable histories. A strict aligned history or bitwise Q/K/V/mask check is needed before claiming per-layer causal effects.

- Verify that NO sparse GQA12 Q4 kernels fire in dense A/B; **all eligible QSA layers** fire in sparse-all; only audited layer11 fires in sparse-only11; audited layer11 does **not** fire in sparse-except11 while other eligible layers DO. Prefer bounded actual HIP ROCm kernel tracing with per-layer mask/audit attribution, not flag-only claims. HIP tracing changes timing and may affect host load; keep instrumentation consistent and bounded. If the new diagnostic filter fails to constrain dispatch or unexpectedly changes graph/supported() selection, fix on CPU/build before reattempt, not through unbounded GPU retries.
- Compare first QSA layer11 F32 output with same Q/K/V and F16 mask bytes for each relevant run, and the per-head F64 mathematical reference; quantify dense-vs-only11 and dense-vs-except11 (not just an assumed transitive causal chain).
- Compare layer15 indexer score arrays, near-tied cutoffs, pool IDs, mapped KV rows and whether swapping remains for the only11 vs except11 intervention. Distinguish identical score mapping, changed scores and changed membership.
- Compare selected logical GDN R/S/P after target decode (especially step0 layer12, downstream 32/44 if safely captured) and natural target logits; use dense A/B bytewise controls and report first observed drift separately for each variant. Do not say post-rollback state discrepancy proves a rollback bug. If candidate/accepted histories diverge, STOP comparable-per-step analysis beyond the first divergence and report the divergence itself without attempting to force identical samples.
- If one intervention removes a downstream difference while another preserves it under identical checkpoint inputs and matched histories, that supports a **conditional contribution** of layer11 arithmetic. Do not call it sole, proven global root cause; other layers/rounding and changed dispatch remain possible confounders. If there is no natural Q4 verification group, or no actual eligible HIP sparse launches, report the coverage gap and do not ascribe model differences to experimental sparse kernels.

## Device safety / environmental boundaries

- Before model load verify PCIe firmware limit Gen3, NVIDIA CUDA card and both AMD HIP devices accessible, no unresolved new Xid79/154; Gen1 at idle is expected power management, under load 0032 observed Gen3 x4. Use bounded read-only checks; no persistent high-rate telemetry.
- **Any new CUDA unspecified launch failure, GPU disappearance, Xid79/154, serious PCIe/AER/DPC, HIP fault or tracing-induced instability: STOP all GPU activity immediately.** Preserve external logs and return Partial/Blocked; no retry, reset, reboot or device swap. Stop on unexpected memory/capture volume too; diagnostic tools must be bounded.
- No NVIDIA/AMD stress tests, PCIe Gen4/Auto switch, driver/power/BIOS adjustments, sudo/privileged profiler, unbounded 98K replay, full-model throughput A/B, GPU graph execution, default selector change, modified arithmetic/overflow guard, speculative acceleration, automatic GitHub Actions, PR/merge/force push. Keep production disabled. All dump files, prompts, GGUF model paths/contents, GPU identifiers, raw logs and traces stay outside Git.

## Completion/report

Write .ai/reports/report-0033.md with:
- exact commit, local CPU/C++/mixed build validation and confirmation original default dispatch remains unchanged when diagnostic flags unset;
- reused old data provenance and separation between cohorts; any missing artifacts;
- the actual number of fresh processes/captured groups and sparse HIP Q4 launches **per layer/condition**, and whether each intervention worked;
- precise per-stage outcomes (layer11 exact-input arithmetic, layer15 top-k cutoff, first observed GDN R/S/P, logits) with dense A/B controls and chronological/causal limitations;
- best supported diagnosis and a focused next action: numerical kernel refinement, fidelity criterion/representative workload decision, or newly revealed bug. **Do not claim exact numerical equivalence or production readiness** without evidence and explicit tolerances.

If the host's device health is not sound, do only offline part and return Blocked with preserved evidence. Commit/push meaningful code fixes and report, no raw artifacts.

Return:
Iteration: 0033 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0033.md | Summary: <1-2 sentences>
