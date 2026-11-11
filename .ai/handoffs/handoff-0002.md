# Handoff 0002 — synchronous tensor upload cost and safe optimization decision

## Objective

Quantify the real cost, tensor identities and safety constraints of the **repeated synchronous host-to-device uploads** observed in handoff 0001, on the actual mixed CUDA/HIP system. Determine whether a modest safe batching or staging change can improve end-to-end performance. Do not delete waits whose completion is required to preserve data lifetime or device ordering.

## Context / work already implemented

- GitHub repository: `civcode/llama.cpp-perf`. Exclusive working branch: `docs/llama-perf`. Read `.ai/WORKFLOW.md`, `.ai/reports/report-0001.md`, `docs/LLAMA-PERF/MEASUREMENTS-20261111.md`, and this handoff.
- Chat AI implemented an **opt-in host-side upload profiler**, disabled by default, in `ggml/src/ggml-cuda/upload-profile.h`, integrated at `ggml/src/ggml-cuda/ggml-cuda.cu` in `.set_tensor`, `.set_tensor_2d`, and `.set_tensor_n`. It measures call counts, sizes, device, tensor name, total host API time and time spent in the existing stream synchronization. Each backend emits summary rows to stderr at **normal process exit**, prefixed `GGML_CUDA_UPLOAD_PROFILE`; CUDA vs ROCm is encoded in the `op` field. Enable with `GGML_CUDA_PROFILE_UPLOADS=1`. The code does not add GPU synchronizations or modify outputs. The agent must validate that the source builds and that the log format works for both backends.
- Chat AI independently tested the standard-library collector with `c++ -std=c++17 -Wall -Wextra -Werror -pthread` and the committed `docs/LLAMA-PERF/tools/test_upload_profile.cpp`. This does **not** prove that the entire CUDA/HIP source compiles; hardware build is delegated.
- Handoff 0001 established that 256 `llama-bench`-driven per-token synchronizations consumed ~2.0 s in a separate short trace, so blanket deletion is not justified. The same separate interposer run attributed 3,200 additional calls (~245 ms) to `ggml_backend_cuda_buffer_set_tensor` and 256 (~57 ms) through `ggml_backend_tensor_copy`. The source currently calls `cudaMemcpyAsync` and then `cudaStreamSynchronize(cudaStreamPerThread)` in `.set_tensor` — synchronous contract / source-buffer reuse can require this.
- Target-only baseline: short `tg128` **46.51 t/s**; 32K-depth `tg128` **41.64 t/s**, one unprofiled run each, no `--fit-target`. MTP server (F16 KV, 196608 context, draft on CUDA0, max3) gave 75.01 / 59.10 / 39.28 t/s at ~8K / ~32K / ~98K with differing acceptance. Old HIP trace and interposer details are stored outside Git under `/home/chris/workspace/llama-perf-results/iteration-0001-retry/`.
- Source inspection shows that the backend already has `ggml_backend_cuda_buffer_set_tensor_n`, but existing scheduler paths can use individual synchronous `ggml_backend_tensor_copy` calls. The availability of a batch callback is **not** permission to reorder those copies; event, lifetime, aliasing, and stream dependencies must first be established.

## Tasks — one local hardware-dependent milestone

1. **Build/test the opt-in instrumentation.** Synchronize the handoff commit, preserve any unrelated local changes, run the committed standalone collector test, and compile in separate `build-perf/` using equivalent mixed CUDA/HIP configuration. Fix CUDA/HIP compiler or logging issues, keeping upload diagnostics **disabled by default**. Do not alter `build-mixed/`, original `llama.cpp`, model files, or users' live server setup.
2. **Measure which tensors are uploaded.** Run a controlled small target-only decode comparison at depth 0 and 32,768 with `GGML_CUDA_PROFILE_UPLOADS=1`, after ensuring a conflicting server is not running. Capture process exit stderr to local results files so both backend summaries are visible. Examine names, sizes, per-device bytes, call counts, and timings for `.set_tensor`, `.set_tensor_2d` and `.set_tensor_n`. Map frequent tensors to source `ggml_backend_sched_copy_input`, the graph split and any host-resident weights; explain if uploads are dynamic expert weights, intermediate activations, or input tensors rather than assuming. Record actual placement and host-resident weights.
3. **Control for diagnostic overhead.** In the same `build-perf`, repeat one short and one 32K case with instrumentation **disabled**; compare them with enabled results and known `build-mixed` baseline. Since timed upload logs are CPU-side and wait for GPU work, distinguish measured host blocking from any actually avoidable time. Look for batchable groups with stable host-buffer lifetimes and same stream/device semantics. If practical, sample one representative **MTP-enabled server** coding or reasoning request so conclusions aren't limited to target-only `llama-bench`. For server, preserve all existing settings: CUDA0+ROCm0+ROCm1 target, CUDA0 MTP draft, max3, F16 KV, 196608 context, layer split 4,7,7, fit-target768, parallel1.
4. **Implement a bounded optimization only if supported.** If evidence demonstrates a safe opportunity (e.g., same-device grouped copies with valid lifetimes), prototype it behind a disabled-by-default option in `build-perf/`; compare against unmodified behavior and verify finite logits, valid output/GDN recurrent state, throughput and memory safety. An alternative successful result is a clear *no-go* showing why syncs cannot safely be removed/batched, with exact callsites and evidence. Avoid global changes to scheduler semantics or extra forced synchronization solely for measurement.
5. **Deliver report and sources.** Write `.ai/reports/report-0002.md` with the exact instrumented run settings, measured per-tensor upload groups, meaningful A/B throughput, numerical checks if code changed, effect of instrumentation, and a single prioritized recommendation. Commit and push useful fixes/diagnostics and the report. Keep raw traces, large logs and model files outside Git.

## Constraints

- Local agent has **exclusive write ownership** until it pushes the matching report and returns the completion pointer. Chat AI must not write concurrently.
- Use the designated branch. Do not force-push, create a PR, modify workflows, dispatch GitHub Actions or run CI jobs. Local builds and targeted tests only; avoid unneeded hardware reconfiguration or kernel-level privilege changes.
- The `build-mixed` baseline must remain unchanged. The 32K and 98K MTP performance issue is end-to-end and cannot be directly inferred from target-only kernel timings.
- Understand synchronous buffer-set and copy contracts, copy-stream ordering, host-pointer lifetime and graph reuse *before* attempting asynchronous behavior. A copy followed by stream sync may be required for safety; never claim the measured API duration is automatically removable.
- No blanket numerical-correctness claim from `llama-bench`; if changing transfer behavior, test finite logits/output and GDN state with actual inference.

## Completion criteria

- Instrumented mixed CUDA/HIP source builds locally in `build-perf`, collector passes its targeted test, and both ROCm and CUDA output semantics are verified (or a specific environment limitation documented).
- On-device evidence explains the frequency, tensor identities and size distribution of synchronous uploads, separating load-time from decode-time transfers and describing relevant placement.
- Controlled baseline-vs-instrumented comparison quantifies measurement overhead; at least one depth comparison tests context sensitivity.
- A safe tested optimization **or** a defensible no-change recommendation identifies the next priority, including whether MTP server workloads need additional work.
- Matching `.ai/reports/report-0002.md` committed and pushed with relevant changes.

## Reporting

Use the standard concise completion pointer: iteration `0002`, status, branch `docs/llama-perf`, pushed commit SHA, report `.ai/reports/report-0002.md`, and 1–2 sentence summary.
