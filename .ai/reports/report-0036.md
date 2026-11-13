# Report 0036 — FP32 WMMA layout fixture; attention variant incomplete

**Status: Partial.** Starting commit `978f0f39acd09952c22d813fa3ae6856b09975f1`. The actual sparse attention FP32 VKQ specialization, dispatch, partial-result combine and tracing were **not implemented**. The existing `GPU implementation not built` abort remains, and the half2 path and production defaults are unchanged. The 32K pilot and all requested FP64/headwise and downstream comparisons were not run. This is unfinished implementation, **not** evidence that gfx1201 FP32 WMMA is infeasible.

## Bounded progress

Added `docs/LLAMA-PERF/tools/test_rdna4_vkq_wmma.hip`, a standalone one-wave gfx1201 layout fixture based on AMD's [RDNA4 WMMA compute-policy example](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html). It computes two independent 16x16 FP32 accumulator fragments from half inputs, applies an FP32 0.5 rescale and a second MAC, then compares all 512 results to exactly representable host references. Compiled using `hipcc --offload-arch=gfx1201 -O2 -Wall -Wextra -Werror`; it passed on each of the two gfx1201 devices. Disassembly of its gfx1201 code object shows `v_wmma_f32_16x16x16_f16` instructions. This establishes the fixture's builtin and output layout for its own operands, **not** the attention kernel's operand loading, packed-half equivalence, sparse dispatch, FP32 combine, or actual layer11 launches. A stronger nonseparable input pattern also passed on both devices. No attention kernel or model was run.

The standalone C++ configuration-gate test passed with `-Wall -Wextra -Werror`; 215 focused Python QSA tests passed. The mixed CUDA/HIP build from 0035 was not rebuilt for a new specialization because none was added. There is no evidence of an FP32 attention symbol or launch.

## Hardware and remaining work

Passive preflight identified both discrete AMD devices as gfx1201 and the NVIDIA device as idle Gen1 with maximum Gen3, 15930 MiB free; idle current Gen1 does not demonstrate loaded Gen3. Current-boot kernel log checks before and after the bounded fixture showed no matching Xid/AER/DPC or AMD GPU fault. No PCIe, driver or power settings were changed; no retries of hardware faults, resets, reboots, stress tests or GitHub Actions occurred.

The substantial unresolved work is to map the existing scrambled half2 VKQ operands and accumulator into the two FP32 fragments, wire both rescale sites and a correctly sized FP32 shared-memory partial combine/fixup, enforce the audited layer11 eligibility at real dispatch, compile the distinct kernel, and verify it against a suitable attention fixture and trace. The standalone test is deliberately not a substitute. Preserve the fail-closed guard until those steps are completed; only then consider the authorized four-condition experiment after full preflight. No raw model captures were created.
