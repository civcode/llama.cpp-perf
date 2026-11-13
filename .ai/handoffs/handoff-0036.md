# Handoff 0036 — finish true RDNA4 gfx1201 FP32 VKQ, then the already-approved matched pilot

## Objective and stop conditions

Continue **the partially completed Iteration 0035** rather than starting a new performance optimization. The user already authorized one diagnostic FP32 VKQ accumulation experiment. Report 0035 is Partial: it did NOT implement or test the FP32 kernel. After the Coding Agent disabled a device-isolating sandbox, both gfx1201 AMD GPUs and the NVIDIA CUDA card became accessible, and the existing mixed Release build passed. The hardware access issue no longer justifies abandoning the kernel implementation. Finish the **real** opt-in, separately identifiable FP32 weighted-value (VKQ) accumulator/rescaling path at audited QSA GQA12 layer11, then perform the one originally authorized bounded comparison **only if the local build, dispatch proof and passive device preflight are sound**.

If the gfx1201 WMMA accumulator/layout/LDS interface is genuinely infeasible without a disproportionate architecture change, retain the existing fail-closed guard, stop before model inference, and write an evidence-grounded Partial/Blocked report with exact compiler, tile/layout or shared-memory limits. Do not claim the sparse FP32 experiment succeeded if no FP32 kernel launched.

## Repository / one developer owns the turn

Repository civcode/llama.cpp-perf, branch docs/llama-perf, starting commit exactly as supplied in Chat AI's transfer prompt. Read .ai/WORKFLOW.md, Reports 0033–0035, docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md and the new **docs/LLAMA-PERF/QSA-GQA12-VKQ-FP32-DESIGN.md**. After the user forwards this handoff, the Coding Agent exclusively edits the repo until report-0036.md is committed/pushed and completion is reported. No concurrent Chat AI repository edits.

## Confirmed starting evidence

- 0033 five-way ~32K natural-MTP ablation under PCIe Gen3 had exactly matched dense A/B and enabled sparse Q4 only in requested layers, with actual HIP launches. Sparse only at layer11 caused the earliest observed downstream layer15 top-k selection differences (3/4 queries) and layer12 R/S differences (12/12 rows); excluding sparse layer11 removed those effects but still changed other logits. No observed device fault.
- 0034 first exact-input HIP0 layer11 Q4 operation: sparse half2 output was closer to FP64 than dense by total NMSE (3.68239e-7 versus 1.18147e-6), winning 22/24 heads and 85/96 correlated query/head positions; dense was better in heads8,9. FP64 closeness is descriptive, not an approval criterion.
- 0035 report: C++ request-gate tests, Python compileall, 215 QSA tests passed. Passive hardware inspection was initially blocked by sandbox device-node isolation, subsequently resolved. AMD devices both gfx1201, mixed Release CUDA sm_89/HIP gfx1201 build passed on **existing scaffold only**. No GPU math or model inference, no traced FP32 variant, no 4-way pilot. UI asset fallback to latest is a build provenance caveat, not a numeric result.
- User previously observed two Xid79/154 NVIDIA fall-off-bus incidents. NVIDIA PCIe remains operator-configured Gen3; observed idle Gen1 is normal power management, loaded Gen3 x4 was seen previously. Successful Gen3 tests are encouraging, not root-cause proof.

## Chat AI completed code and analysis

- Strict pure helper **ggml/src/ggml-cuda/fattn-sparse-q4-gqa12-vkq-f32.h** and CPU C++ test **docs/LLAMA-PERF/tools/test_qsa_gqa12_vkq_f32.cpp**, already corrected/tested on agent in 0035. The opt-in GGML_HIP_QSA_GQA12_VKQ_F32=1 requires GGML_HIP_QSA_SPARSE_Q4_GQA12=1, GGML_QSA_MASK_AUDIT and GGML_HIP_QSA_GQA12_LAYER_FILTER=only:11. The normal flag-unset code path is unchanged; malformed opt-in fails closed. Shape limit layer11, F16 K/V/mask, Q/K/V dimensions 256, q4, GQA12, RDNA/WMMA, context/index bounds, **alongside** existing sparse should_use checks.
- **fattn.cu** wires the gate, but currently contains an explicit "GPU implementation not built" abort for a requested FP32 variant. Remove that temporary abort ONLY when a distinct, valid FP32 VKQ specialization is actually selected; retain validation for unsupported requests.
- **QSA-GQA12-VKQ-FP32-DESIGN.md** documents the observed gfx1201 source issues. Existing non-RDNA3 256×256 half2 WMMA uses a mirrored 32×8 A operand, 16×8 B, and scrambled 16×16 half2 C; it invokes gfx12 FP16 WMMA twice with opsel0/1 to pack the results. FP32 builtin expects a v8float C and has no dual-opsel equivalent, likely needing TWO explicitly mapped accumulator fragments for the 32-feature logical tile. Existing generic float C branch converts to half when combining into tile_Q; merely substituting a float type would leave a substantial half-rounding point. The 4-query, GQA12 shape spans multiple partial warps, so FP32 shared scratch size and synchronization must be validated. The official AMD gfx12 WMMA reference is linked in the design note.
- No Chat AI GPU kernel arithmetic implementation has been attempted because the actual gfx1201 compiler, WMMA builtin instantiation, LDS occupancy and attached AMD devices are all available on the Coding Agent host. This task is specifically delegated to that hardware-capable implementer.

## A. Implement and verify a real, narrow FP32 variant

1. Confirm exact pinned starting commit and clean workspace. Re-run only focused CPU source tests if needed, plus mixed build as implementation evolves; no GitHub Actions. Verify both AMD agents gfx1201 and linked device nodes with passive, nonprivileged checks. A non-sandboxed run is required for GPU tests; do not use sudo, reboot/reset, change host drivers/firmware or PCIe settings to get access.
2. Examine existing gfx12 WMMA instructions and fragment layout in mma.cuh; trace KQ→softmax→F16 weight conversion→half2 VKQ across fattn-mma-f16.cuh, including **both** KQ_max rescale paths, partial warp combines and final write/fixup. Prove correct mapping for new float fragments with a deterministic tiny WMMA multiply fixture or equivalent compiler-verified unit test before the expensive 32K model. The standalone CPU grammar test does not prove device arithmetic.
3. Implement an opt-in **compile-time-distinguishable** FP32 VKQ accumulation/rescale variant for the existing sparse-only11 Q4 GQA12 eligibility. Preserve inputs, selected Q4 index order, F16 mask, KQ/softmax calculation, F16 conversion of softmax weights, attention value semantics, original overflow guards and model/user graph policy. Keep half2 code and GQA4, NVIDIA, other heads/layers, normal dense and default-off dispatch unchanged. Avoid a scalar/different algorithm falsely labeled as WMMA FP32. Any change beyond F32 MAC/rescale (e.g. FP32 intermediate LDS combine versus existing half2) is a potential numerical confounder; document the exact scope and handle layout correctly. If required shared-memory changes exceed supported local LDS or break occupancy, stop or choose a bounded justified partition, with full disclosure and correctness fixtures.
4. Verify a distinct mangled symbol or precise trace/marker plus ROCm profiler correlates an actual **F32 WMMA specialization** with audited HIP QSA layer11, and zero such specialized launches in dense and half2 control or any nonlayer11 operation. A validated env flag alone does NOT prove the FP32 path executed. Bad/unsupported env must produce explicit error, not fallback under the FP32 name.
5. Build mixed CUDA/HIP Release llama-server and necessary targeted backend tests locally; add/execute regression tests of FP32 dispatch, unsupported env/shapes, numerical tile mapping, same-input control and default unchanged. If the CUDA compile cannot build the new HIP-only specialization, fix guards without affecting NVIDIA default. Do not dispatch GitHub Actions.

## B. One already-authorized bounded 32K numerical pilot, only after A and healthy preflight

Fresh external output directory: /home/chris/workspace/llama-perf-results/iteration-0036/. Do not overwrite 0033–0035 data. Use exactly the matched Report0033 fixture and target/draft model fingerprints: Qwen3.8-Flash-Next IQ3_XXS target shards + MTP Q8 draft, mixed CUDA0/HIP0/HIP1 4/7/7, F16 KV, n_ctx65536, max draft3, reasoning off, greedy temp0/top_p1 seed42, no prompt cache, graph-disabled QSA mask audit, original ~32K context and 32 generation token budget. Separate bounded server process per condition; do not force accepted/rejected MTP decisions or alter target/draft weights.

**Exactly four fresh conditions, no repeats/soak:**
- dense A: no sparse/GQA4/forced-WMMA/FP32 options;
- dense B: independent identical dense repeat;
- sparse half2 only11: experimental sparse GQA12=1, layer filter only:11, FP32 flag unset;
- sparse FP32 only11: identical to prior except FP32 opt-in=1 and proven distinct FP32 specialization.

Capture normal server stdout/stderr to new external dense-a-server.log, dense-b-server.log, sparse-half2-server.log and **sparse-fp32-server.log**. For operator second-terminal live timing:
    tail -n 30 -F /home/chris/workspace/llama-perf-results/iteration-0036/sparse-fp32-server.log
Preserve exit status and clean only owned processes. Timings under profiling/capture are NOT throughput measurements.

Use bounded captures of the exact same first HIP0 layer11 Q4 operation: Q/K/V raw spans, F16 mask bytes, selected sparse index sets, float output, actual F32 launch trace; a limited layer15 top-k score/mapping, layer12 recurrent R/S after decode, and aligned natural verification logits. Keep capture storage bounded (do NOT generate >5GiB whole recurrent dumps). Under a healthy preflight, monitor sparse Q4 launch and current-boot Xid/CUDA/AER/DPC/HIP errors. **A serious NVIDIA device-loss/Xid79/154, PCIe fault, CUDA/HIP fault or tracer instability means STOP GPU activity immediately, preserve raw external logs, no retry, reboot, reset or PCIe/power/driver change.**

## C. Required quantitative interpretation

- Dense A/B must be **bitwise identical** for selected controls; all four must have the same initial logical token history, and the captured layer11 Q/K/V and F16 mask must be byte-identical. Sparse half2 and FP32 must also have the exact same selected sparse indices. No silent reordering or mismatched capture accepted.
- Use the existing FP64 exact-input replay plus headwise precision analyzer to report dense/half2/FP32 errors for 24 heads and 96 correlated query/head pairs, aggregate NMSE, maximum absolute errors and the worst loss cases (especially heads8/9/14/17). A lower mean error with a worse tail is not automatic improvement; no arbitrary release tolerance.
- Compare layer15 top-k cutoff scores/selections/mapped KV, first observed layer12 R/S snapshots and actual presampling target logits on identical histories. If natural decisions diverge, record the first divergence and **do not force or compare unaligned later states**.
- Distinguish FP32 WMMA register accumulation, FP32 online rescale, any needed intermediate precision/layout changes, and FP32/f16 final partial combine in the report. Do not infer precision effect merely because outputs differ.
- The original kernel and default sparse-disabled state must remain exactly available. No production promotion, GPU graph test, 98K replay or full-model throughput benchmark.

## Finish

Write .ai/reports/report-0036.md with concise details: exact revision, build and test outcomes, actual host health and Gen3 observation, real specialized FP32 launches or precise blockers, code changes and any confounders, per-head FP64 accuracy versus dense and half2, layer15/GDN/logit results where aligned, and honest limits. Raw captures, model paths, serial/UUIDs, profiler logs and full answers remain outside Git.

**Status Completed** only if real FP32 VKQ GPU code was implemented, compiled, executed with verified sparse layer11 dispatch, and the bounded four-way comparison and report are complete. **Status Partial/Blocked** on a genuine WMMA/LDS feasibility or device-health limit; never silently retry hardware failures or call a fallback "FP32".

Commit/push source, tests and report to docs/llama-perf; stop writing. Return:
Iteration: 0036 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0036.md | Summary: <1-2 sentences>
