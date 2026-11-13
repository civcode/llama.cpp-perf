# RDNA4 gfx1201 GQA12 sparse VKQ FP32 design audit (Iteration 0036)

**Status:** source-level design investigation only; this is NOT a compiled, tested,
or working FP32 HIP kernel. Read Report 0035 before modifying code.

## Known constraints from repository and device identification

- AMD Radeon AI PRO R9700 devices are **gfx1201** according to Report 0035's
  post-sandbox-deactivation rocminfo. Existing mixed Release build with
  CMAKE_HIP_ARCHITECTURES=gfx1201 and CUDA sm_89 passed; that build contains
  ONLY the fail-closed FP32 request scaffold.
- The fixed first pilot is a **QSA 256x256, n_queries=4, GQA12**, eligible sparse
  attention at audited layer 11 only. Existing sparse GQA12 is opt-in:
  GGML_HIP_QSA_SPARSE_Q4_GQA12=1, graph-disabled GGML_QSA_MASK_AUDIT and
  GGML_HIP_QSA_GQA12_LAYER_FILTER=only:11.
- Additional exact opt-in GGML_HIP_QSA_GQA12_VKQ_F32=1 currently aborts with
  **"GPU implementation not built"** in fattn.cu. It must remain fail-closed
  until a genuinely compiled FP32 VKQ kernel variant is selected and traced.
- The experimental goal is to change **VKQ accumulation and rescaling**, not
  the selected sparse-index order, QKV inputs, KQ/F32 scores, softmax weight
  half conversion, output shapes, head mapping, or overflow policy. Preserve
  the original half2 sparse kernel and all default/flag-unset behavior.

## Why a type substitution is insufficient

Existing gfx1201 path, inspected in ggml/src/ggml-cuda/fattn-mma-f16.cuh
and ggml/src/ggml-cuda/mma.cuh:

1. The 256x256 non-RDNA3 WMMA configuration uses a
   tile<32,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED> A operand,
   tile<16,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED> B operand, and
   tile<16,16,half2,DATA_LAYOUT_I_MAJOR_SCRAMBLED> VKQ C tile.
2. The existing half2 WMMA intrinsic is invoked **twice**, using distinct
   A fragments and opsel=0 then opsel=1. Both are stored into one packed
   half-precision accumulator representation. It is therefore unsafe to
   replace the C alias with one float tile and retain this calling convention.
3. The float output instruction is genuinely distinct: gfx12
   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 accepts FP16 A/B inputs and a
   vector of eight FP32 accumulator values per lane, returning eight FP32
   values per lane. **There is no same-opsel dual-half accumulator interface.**
   Likely two separately-addressed F32 fragments are required per logical
   32-feature block, with verified row/column mapping.
4. Both per-KV-tile FP32 online rescaling sections in
   fattn-mma-f16.cuh already have a generic float branch, but their loop
   bounds, C tile indexing and WMMA C data layout must be established for
   the actual new fragment pair. The existing float branch was NOT proof
   that the 256x256 gfx1201 VKQ implementation supports the new layout.
5. The current combine writes VKQ_C into **half2** shared memory tile_Q
   (one branch converts float accumulator values to half). Downstream partial
   warp reduction reads those half2 values. Leaving this conversion active
   limits the claimed intervention to FP32 *register* accumulation and
   rescaling, **not** end-to-end F32 partial-combine fidelity. This is a
   material additional rounding step that must be disclosed and ideally
   preserved in F32 for the intended clean experiment.
6. This QSA shape uses several warps/partial accumulators per output query
   tile; a wider F32 shared scratch may increase workgroup LDS requirements.
   Check shared-memory limits and occupancy **at the actual compiled
   specialization**, not by assuming 2x half2 scratch always fits.

AMD reference:
https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/docs-1.0.0/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html
This source documents the gfx12 builtin's FP32 accumulator and wave32
behavior. It does not verify that llama.cpp's existing C tile mapping matches
the new builtin's expected output ordering.

## Recommended precise implementation/validation sequence

A. **Source/host validation first.** Confirm exact gfx1201 preprocessor
   definitions, WMMA overload shapes, mirrored A/B fragment layout and C
   register element ordering from the available toolchain. Compare the
   existing half2 two-opsel tile mapping to the two proposed F32 fragments
   with deterministic tiny 16x16 multiplication fixtures. Ensure each output
   element maps to its correct logical V feature/query head before attempting
   a long 32K model run. A deterministic GPU fixture may be included in the
   one bounded local validation sequence only after passive device-health
   checks; do not run unbounded trial-and-error loads.

B. **Keep specialization distinct.** Propagate a compile-time VKQ_F32
   switch through the actual sparse path (fattn.cu dispatch ->
   ggml_cuda_flash_attn_ext_mma_f16_case -> flash_attn_ext_f16 kernel ->
   process_tile -> iter and combine). Only instantiate it for the eligible
   RDNA4, 256x256, 4-query, GQA12, layer11, F16 K/V and mask sparse path.
   Keep flag-off/GQA4 and other architectures unchanged. Remove the
   temporary explicit abort only after the alternative kernel is genuinely
   wired; bad requests still abort explicitly, not silently use half2.

C. **Numerical scope.** Ideally maintain FP32 throughout VKQ MAC,
   KQ_max_scale rescaling and final warp/tile combination. If F32 combination
   requires a different LDS scratch layout or extra synchronization, document
   those choices, verify shared-memory bound and avoid attributing changed
   outputs solely to FP32 WMMA MAC. Do not change KQ/softmax, F16 weight
   conversion, sparse indices or mask selection to make tests pass. If the
   exact accumulator-only comparison proves infeasible without substantial
   redesign, retain fail-closed behavior and return a precise blocker with
   source/compiled constraints rather than a mislabeled precision run.

D. **Trace attribution.** Distinct mangled specialization, explicit
   diagnostic launch marker AND ROCm trace should show that actual new FP32
   VKQ kernel launches on **layer11 only** in the FP32 condition, while
   the half2 baseline launches the old sparse kernel and dense controls have
   no experimental sparse Q4 kernels.

E. **Single matched 0036 pilot.** After local build and passive
   health checks, use Report 0033's exact ~32K model/MTP fixture and
   graph-disabled audit in only four fresh conditions: dense A, dense B,
   sparse-only11 half2, sparse-only11 FP32. For the first aligned layer11
   Q4 capture, require byte-identical Q/K/V spans, F16 mask and selected
   sparse indices. Compare FP64 mathematical reference per 24 heads and 96
   query/head positions; then layer15 score cutoff, GDN R/S and target
   logits only while natural histories align. Record worse-tail cases and
   stop comparisons after first history divergence. No artificial
   draft-acceptance alignment, throughput claims or quality-release margin.
   Keep all raw captures outside Git.

## Hardware and policy

Operator-configured NVIDIA PCIe Gen3 remains unchanged. Previous Xid79/154
episodes were followed by healthy full sparse repeat runs, but no permanent
root-cause fix is established. A new serious CUDA/PCIe/Xid/HIP fault means
**STOP GPU activity immediately, no automatic retry/reset/reboot**.
No GitHub Actions, sudo, privileged profiling, stress, 98K/GPU graph tests,
model/driver/hardware changes or production sparse promotion.

This design note is an implementation aid, **not** evidence of a working
FP32 kernel, actual arithmetic improvement or numerical equivalence.
