# Handoff 0037 — compile and validate Chat AI's FP32 sparse VKQ attention specialization

## Objective

Finish the user-approved FP32 weighted-value (VKQ) precision experiment after Report 0036 returned Partial. **Chat AI has now implemented an actual candidate, compile-time-distinct FP32 WMMA register accumulator inside the attention kernel**. The local Coding Agent's primary task is to **compile it, fix any real compiler/fragment/dispatch errors, prove the actual arithmetic and dispatch, and then run the previously authorized bounded numerical pilot if safe**.

The prior two partial reports 0035/0036 are NOT evidence of FP32 attention performance. The latest 0036 fixture merely establishes that gfx1201 can compute two independent 16x16 FP32 WMMA fragments and rescale them. It was not a model kernel. Do not stop after re-running that already-passing standalone fixture; integrate, debug and validate the actual flash-attention specialization.

Repository civcode/llama.cpp-perf; branch docs/llama-perf. Starting SHA: exactly as supplied by Chat AI in transfer message. Read .ai/WORKFLOW.md; Reports 0033, 0034, 0035, 0036; docs/LLAMA-PERF/QSA-GQA12-VKQ-FP32-DESIGN.md and QSA-SPARSE-TASK-QUALITY.md. Exclusive Coding Agent ownership starts on user relay and ends after report-0037.md commit/push. No concurrent Chat AI repository writes.

## What Chat AI implemented this turn (UNCOMPILED and UNVERIFIED on GPU)

1. **ggml/src/ggml-cuda/mma.cuh**: gfx12-only function mma_vkq_gfx12_fp32 takes two independent tile<16,16,float,DATA_LAYOUT_I_MAJOR> accumulators and the *same* A tile<32,8,half2> / B tile<16,8,half2> operands as baseline. It reuses the existing half2 implementation's A16[0]/A16[1] partition, but invokes the preexisting gfx12 F32 WMMA intrinsic through the standard mma() overload. The earlier 0036 standalone fixture proves F32 builtin behavior on arbitrary matrices, NOT this exact loader/fragment mapping.
2. **ggml/src/ggml-cuda/fattn-mma-f16.cuh**: new template bool use_vkq_f32 propagates from distinct global flash_attn_ext_f16<...,true> kernel through process_tile and iter. The first narrow specialization is compiled only for (DKQ=256,DV=256,ncols1=4,ncols2=4,use_sparse=true,softcap=false) on gfx12 WMMA. It uses twice as many C fragments (two 16x16 F32 fragments per original packed 32-feature F16 tile), invokes the new two-fragment WMMA helper, and uses existing F32 C online rescale branches. Host case selects it only with strict valid env and an actually eligible audited QSA GQA12 layer11 sparse kernel. It logs QSA_VKQ_F32_SELECT, which is a **host selection event only**, not GPU launch proof. Mixed CUDA and all existing dense/half2 defaults are still template use_vkq_f32=false.
3. **ggml/src/ggml-cuda/fattn.cu**: invalid FP32 requests continue to abort; the old "GPU implementation not built" scaffold abort is removed in favor of the new distinct specialization. This is source code, NOT evidence the new variant passes compilation.
4. Updated design and numerical-quality docs. **No HIP compile/model inference was performed by Chat AI** in this turn due to lack of local gfx1201 devices/toolchain.

**Important precision scope**: Current float register VKQ accumulators and KQ_max-scale rescaling are F32; the existing downstream tile_Q shared-memory combine **casts float to F16**, then the final stage converts half2 back to FP32 for partial combination and normalization. This is a *FP32 WMMA register accumulation + online rescale pilot*, NOT full FP32 end-to-end attention or pure accumulator-only numerical effect. The shared cast must be explicitly documented. Do not silently replace that shared combine with FP32 as part of the first pilot; that would confound the single-variable experiment. If tile mapping/float conversion is incorrect or requires a shared layout change, fix correctness and disclose the change.

## A. Complete local implementation/validation before model inference

- Sync exact starting commit, inspect fresh Chat AI edits and preserve unrelated working files. Compile standalone host gate with -std=c++17 -Wall -Wextra -Werror; use existing 0036 HIP dual-fragment fixture on both gfx1201 devices only if needed. Run focused QSA Python tests. Existing device-node isolation was resolved after sandbox deactivation in 0035, so use a device-visible, nonprivileged environment. Do not alter host hardware, drivers, firmware/PCIe configuration or use GitHub Actions.
- **Build mixed CUDA sm_89 / HIP gfx1201 Release llama-server and the relevant backend tests with the NEW specialization.** Fix source errors autonomously. In particular validate the new F32 helper overload and the exact (ncols1=4,ncols2=4) instantiation, type-dependent C tile bounds, register allocation and mangled kernel-symbol distinction. On unsupported graphics/CUDA shapes the false/default branch must build unchanged. A routine compiler error is not a reason to return a Partial report before fixes.
- Add a **small deterministic kernel-level fixture** using the actual A/B ldmatrix fragments consumed by the sparse value path (not simply a separate independent matrix), requiring correctly mapped 16x16 pair output versus scalar/FP64 reference and the F32-to-F16 shared-output conversion. Test both gfx1201 devices under passive healthy device preflight. Check head/query layout, both F32 rescale sites, finite behavior, no OOB and any unexpected HIP register spills/LDS limit. If changes to C fragments, final half storage or wave mapping are needed for correctness, fix them and state precise arithmetic scope.
- Prove negative and positive dispatch: no FP32 specialization in dense or original sparse-half2; user setting FP32=1 with GQA12=1 and layer filter only:11 plus mask audit must select the F32 kernel only at truly eligible QSA layer11. Incorrect flag, other layers/devices/shapes must abort/reject instead of silently labeling half2 FP32. Use source/build symbol and **actual HIP trace** to prove F32 launch; QSA_VKQ_F32_SELECT by itself does NOT prove launch. A populated trace for old sparse Q4 isn't sufficient to prove the F32 variant.

## B. Single bounded four-condition pilot (only after fully passing A and passive healthy GPU preflight)

Continue the user's existing explicit authorization from Handoff0035/0036; do not expand into new stress experiments or a new quality sample. Create new external output root:
  /home/chris/workspace/llama-perf-results/iteration-0037/
Keep all full prompts, raw model tensors, HIP traces, device IDs, logs and files outside Git.

Four fresh sequential processes on the exact original Report 0033 ~32K natural-MTP fixture, same GGUF fingerprints and chat settings, context65536, max32 generated tokens, IQ3_XXS target + Q8 MTP draft, CUDA0/HIP0/HIP1 split4/7/7, F16 KV, MTP draft max3, seed42, temp0/top_p1, reasoning off, cache off, graph-disabled GGML_QSA_MASK_AUDIT:
  1. dense A, no experimental sparse or precision env;
  2. dense B, independent exact same baseline;
  3. sparse GQA12 ONLY at layer 11, original half2 VKQ (GGML_HIP_QSA_GQA12_VKQ_F32 UNSET);
  4. sparse GQA12 ONLY at layer 11, FP32 WMMA register VKQ (GGML_HIP_QSA_GQA12_VKQ_F32=1, strict other flags).

Exact current flags: GGML_HIP_QSA_SPARSE_Q4_GQA12=1, GGML_HIP_QSA_GQA12_LAYER_FILTER=only:11, GGML_QSA_MASK_AUDIT enabled for 3 and 4. GQA4 and forced-WMMA overrides unset. Use exactly the previous proven experiment fixtures and same bounded tracer/capture options. Do NOT force MTP draft acceptance or manufacture an aligned history.

Capture combined ordinary llama-server stdout/stderr in these external files: dense-a-server.log, dense-b-server.log, sparse-half2-server.log, sparse-fp32-server.log. To let operator follow live normal timing while FP32 condition runs:
  tail -n 30 -F /home/chris/workspace/llama-perf-results/iteration-0037/sparse-fp32-server.log
No restart just to display logs, preserve child process exit status, do not infer full-model throughput from incidental timings.

**Host safety**: existing operator-selected NVIDIA PCIe Gen3 stays unchanged (idle Gen1 can be normal); CUDA + both AMD devices must be accessible with nonprivileged read-only check, current-boot journal must have no unresolved new serious fault, memory/capture storage bounded. If ANY serious NVIDIA Xid79/154, CUDA device loss, PCIe AER/DPC, HIP error or tracer instability appears, STOP ALL GPU work immediately; preserve logs and return Partial/Blocked. NO automatic retry, reset, reboot, power/driver changes, sudo, privileged profiling, PCIe Gen4/Auto, stress/soak/98K/graphs, full-model throughput benchmark, extra run or GitHub Actions.

## C. Evidence and interpretation

- Confirm byte-exact dense A/B captured input and attention output, same initial candidate histories, and exact Q/K/V and F16 mask bytes for first layer11 Q4 operation across four conditions. Half2 sparse and FP32 sparse must have identical selected KV index sequence/mask; don't compare different sparse masks as precision intervention.
- Compute FP64 mathematical reference headwise for all 24 heads × 4 query positions, comparing dense, sparse-half2, sparse-FP32. Report aggregate NMSE, headwise winners/losses, worst max-abs and differences especially heads8,9,14,17. The 96 pairs are correlated; no invented population confidence interval. Compare the FP32 kernel's output versus its known F16 shared cast and explicitly distinguish rounding stages.
- If identical MTP histories remain aligned, compare layer15 top-k scores/cutoffs/selected pools and KV maps, GDN R/S after target decode and pre-sampling full-vocab target logits. If histories diverge, report first divergence and stop subsequent aligned comparisons instead of forcing decisions.
- FP64 closeness is NOT a quality threshold, model correctness guarantee or reason to enable sparse by default. Both dense/sparse paths may diverge numerically without being defective; objective is to isolate F32 arithmetic effect and any new outliers.

## Finish and status criteria

Write concise .ai/reports/report-0037.md with implementation changes and exact built revision, local tests/build, actual gfx1201 F32 WMMA specialization symbol/launch evidence, per-layer sparse counts, host health, four matched conditions and numeric/error/downstream results; clearly acknowledge FP16 shared combine as remaining rounding. No raw logs/tensors/model files in Git.

**Completed** requires a real correctly mapped FP32 register WMMA attention kernel compiled/trace-proven on actual layer11 plus all FOUR matched condition analyses. **Partial/Blocked** is appropriate only for a genuine hardware error or documented technical limitation after autonomous compiler/fragment debugging; it is NOT for merely finishing the standalone fixture or encountering a routine build error. Keep invalid FP32 requests fail-closed and default experimental sparse off.

Commit/push to docs/llama-perf, stop repo writes, return:

Iteration: 0037 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0037.md | Summary: <1-2 sentences>
