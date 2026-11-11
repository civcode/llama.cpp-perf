# MTP and HIP measurements — 2026-11-11

These observations supplement [the target-only llama-bench baseline](BENCHMARK-RESULTS.md) and provide the starting point for [handoff 0001](../../.ai/handoffs/handoff-0001.md). Reported measurements came from user-run commands on the inference workstation; the Chat AI has **not** rerun them on that hardware.

## Configuration

- Fork: `civcode/llama.cpp-perf`, baseline build `1648d6411 (12296)`.
- Hardware: RTX 4070 Ti SUPER 16 GiB (CUDA0) + two Radeon AI PRO R9700 32 GiB (ROCm0/1), Ryzen 9950X, 64 GiB RAM.
- MTP server: `CUDA0,ROCm0,ROCm1`; draft on CUDA0; split `4,7,7`; `--ctx-size 196608`; F16 K and V; `--spec-type draft-mtp --spec-draft-n-max 3`; `--fit-target 768`; one slot; sampling `temp=1 top-p=.95 top-k=20 min-p=0`; fixed seed 42 during checks.
- Local results root: `/home/chris/workspace/llama-perf-results/20261111-143704/` (outside repository).

## MTP server baselines

All requests below were capped at 512 generated tokens and ended with `finish_reason=length`. Different prompt content can change generated tokens and acceptance; long-context input included synthetic background records and the same final reasoning question. Full requests/responses, where saved, are in the local `server/` directory.

| Experiment | Effective prompt tokens | Cached prompt tokens | Prefill t/s | Generation t/s | Draft acceptance | Verification steps | Mean tokens/verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original coding smoke test | 84 | 0 | 110.35 | 67.80 | 46.26% | 214 | 2.39 |
| Coding workload | 73 | 42 | — | 57.83 | 34.40% | 252 | 2.03 |
| Reasoning workload | 107 | 42 | — | 76.45 | 56.81% | 189 | 2.70 |
| Explanation workload | 81 | 42 | — | 54.59 | 30.58% | 267 | 1.91 |
| Controlled 8K reasoning | 8,236 | 0 | 2,221.77 | 75.01 | 59.74% | 183 | 2.79 |
| Controlled 32K reasoning | 32,809 | 0 | 2,136.11 | 59.10 | 50.08% | 204 | 2.50 |
| Controlled 98K reasoning | 98,275 | 0 | 1,670.23 | 39.28 | 50.41% | 203 | 2.51 |

The repeated warm coding prompt produced `68.53`, `68.50`, `68.39` t/s at constant acceptance (46.26%) and 80 cached prompt tokens. The 8K and 32K controlled prompts used `cache_prompt=false` and reported zero cached tokens. The 98K request likewise reported zero cached tokens. At 32K vs. 98K, acceptance and verification count were almost unchanged while generation fell from 59.10 to 39.28 t/s: estimated generation time per verification step rose from 42.47 to 64.08 ms. This is **end-to-end** time, not a GPU kernel duration or a directly removable cost.

## Existing HIP short-context trace

The server was stopped before profiling to free VRAM. The user ran ROCprofiler-SDK v1.4.1, ROCm 10.1, with `rocprofv3 --runtime-trace --output-format rocpd --summary` around a **target-only** `llama-bench`, `pp1024` and `tg128`, depth 0, requested fixed split `4/7/7`, one repetition, without `--fit-target`. It reported 2,322.21 t/s prefill and 38.61 t/s target-only decode. Profiling overhead is not yet measured. Source build was `1648d6411`.

- DB: `/home/chris/workspace/llama-perf-results/20261111-143704/hip/short-0/iq3_short_results.db` (118.49 MiB; SQLite quick_check `ok`).
- The actual SQLite data tables have unique suffixes. Use the stable views `kernels`, `regions_and_samples`, `rocpd_info_agent`, and `rocpd_memory_copy`. HIP API category is `HIP_RUNTIME_API_EXT` rather than `HIP_API`.
- First AMD kernel to last kernel: **9.371 s**. Model-upload activity dominates seconds 1–4: approximately **34.7 GiB host → R9700**, likely initialization. Do not treat this as recurrent decode traffic.
- AMD GPU agents: `1=gfx1201`, `2=gfx1201`; agent 3 `gfx1036` integrated GPU, idle in this trace. GPU agent 1 accumulated 1,334.972 ms kernel execution (150,678 dispatches); agent 2 accumulated 1,371.476 ms (147,839). BF16 and quantized matvec/matmul dominate many top kernel entries; GDN kernels are present.
- Across whole trace: **131 GPU2 → CPU copies**, each **970 KiB**, during seconds 5.465–9.371; 9.75 ms aggregate GPU copy duration. Their size is consistent with logits readback but not proven. **8 AMD GPU1 → GPU2 copies** totaling 160 MiB, concentrated before sustained decode. CUDA-side copies are **not** covered by ROCprofiler.
- Generation-heavy interval defined as **6.000–9.371 s** after first AMD kernel: 3,981 `hipStreamSynchronize` calls, all from CPU TID `331789`, totaling **2,769.20 ms**, or **82.1%** of the 3,371 ms window. Average sync call 695.61 μs, max 12.658 ms. Same interval included 1,270 `hipGraphLaunch` calls (87.83 ms of API time) and 3,642 `hipMemcpyAsync` calls (10.33 ms of API time).
- This 82.1% is time **inside blocking API calls on one host thread**, not proof that GPU waits are unnecessary. The profiler observes both GPUs, possible overlapped compute, and only the AMD side of the mixed-backend critical path. We must identify the synchronization callsites and the work each call waits for before proposing removal or asynchronous pipelining.

## Immediate next question

Which backend scheduler or device-copy callsites lead to the 3,981 host-side `hipStreamSynchronize` calls, and what share of their latency is essential GPU completion versus avoidable serialization? Existing code hotspots include `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_copy_input`, `ggml_backend_sched_compute_splits`) and `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_backend_cuda_synchronize`). A same-thread neighboring HIP API is a useful clue, **not proof** of the C++ callsite.

A read-only, schema-aware analyzer lives at [`tools/rocpd_sync_analysis.py`](tools/rocpd_sync_analysis.py). It has been validated on a synthetic SQLite fixture, **not yet against the full device trace**.
