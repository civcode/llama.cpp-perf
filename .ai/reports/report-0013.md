# Report 0013 — QSA live masks and overflow protection

**Status: Partial; no-go for sparse full-model inference or A/B.** Live dense-path masks passed the measured bounds, and an experimental Q4 overflow now aborts before gather. However, the model has **24 query heads and 2 KV heads (GQA12)**, whereas the experimental selector explicitly requires GQA4. The requested model cannot activate this kernel. GPU-graph replay, mixed sequences and fixed-token logits/GDN-state equivalence also remain unverified. Do not weaken the selector to manufacture a benchmark. Production defaults remain unchanged.

## Environment and changes

Starting commit: `8d45e3bd277e848cab923167d7d4def71e9d5731`; this report and changes are in its descendant commit. Mixed build: CUDA nvcc 12.0.140, HIP 7.16.26385/AMD clang 24, GCC 13.3.0; RTX 4070 Ti SUPER and two Radeon AI PRO R9700 gfx1201 GPUs. `cmake --build build-perf -j 28 --target test-backend-ops llama-server llama-bench` passed; the strict C++17 sparse selector test passed. One intermediate rebuild failed due to a missing format macro; it was fixed and both CUDA and HIP rebuilt successfully.

`GGML_QSA_MASK_AUDIT=<1..100000>` is an opt-in synchronous consumer-boundary audit of final F16 Q=4 masks. It reports monotonic timestamp, backend/device, tensor identity, KV length, Q/KV heads, stream index, supplied per-query bound, exact finite counts and union, allocation capacity, NaNs and signed infinities; unexpected layouts, NaNs, +infinity or bound violations abort. The invocation limit is explicit and exhaustion is reported. `GGML_QSA_MASK_AUDIT_DIR` optionally writes the first contiguous `[4,KV]` raw F16 mask per backend/device to a caller-provided directory. Audit disables GPU graph capture/replay and changes timing, so it is not a benchmark or evidence about GPU graph replay. The stream index is not a server sequence ID; attention tensor node IDs are not stable layer labels across graphs. Timestamped server step traces permit phase correlation.

For the opt-in HIP Q4 sparse path, the compactor now returns the **unclamped** union count. Before any dependent gather launch, the host copies all counts on the same HIP stream, synchronizes it, and aborts explicitly if any count exceeds allocated capacity. Other sparse variants retain their existing count behavior. HIP graph capture is disabled when the Q4 experiment is enabled; a stream-capture check also rejects unexpected capture. This prototype adds a host synchronization per sparse operation and invalidates Report 0012's sparse performance numbers for the modified implementation. It does not establish a fast production solution. No scheduling, graph, or synchronization policy is changed when the experimental and diagnostic variables are unset.

## Actual model masks — dense/default path

Fresh isolated server processes, `GGML_HIP_QSA_SPARSE_Q4` and `GGML_HIP_FATTN_FORCE_WMMA_Q4` unset, used the Report 0008–0010 IQ3_XXS target plus MTP Q8 model, CUDA0/ROCm0/ROCm1 4/7/7 layer split, F16 KV, context 196608, max draft 3, original uncached requests and seed 42. Each generated 512 tokens. Diagnostic output is not timing evidence. The server reported llama computation-graph reuse (200 and 201), but the audit explicitly disabled **GPU** capture/replay; these are different things.

All reported masks had per-query bound 2051 and Q4 allocation capacity `min(KV, 4*2051) = 8204`. The distribution includes 12 target attention layers per Q4 target pass and one draft/reprocessing mask per verification step; Q=4 only, not a claim about other batch sizes. Percentiles use the sorted sample at index `floor(p*(N-1))`.

| Prompt fixture | Actual padded KV lengths | Groups | ROCm0 / ROCm1 / CUDA0 | Per-query finite range | Union min / median / p90 / p99 / max |
| --- | --- | ---: | --- | --- | --- |
| 32K | 33024, 33280, 33536 | 2652 | 1020 / 1020 / 612 | 2048–2051 | 2393 / 3406 / 4102 / 4734 / 5493 |
| 98K | 98304, 98560, 98816 | 2665 | 1025 / 1025 / 615 | 2048–2051 | 2415 / 3620 / 4314 / 5006 / 5863 |

Every measured group met both bounds; no NaNs or +infinities were observed. Negative infinity was the normal masked value. There were 2448/2460 masks timestamped during target processing and 204/205 during speculative reprocessing. Verification-commit traces for the 32K run show 74/22/39/68 steps accepting respectively 3/2/1/0 of 3 drafts, plus one accepting 1 of 2; the 98K run shows 76/24/30/74 accepting 3/2/1/0 of 3. Thus both fully accepted and rejected drafts, subsequent verification/reprocessing, and varying incomplete-tail counts were observed. These are trace-observed acceptance outcomes, not an independent fixed-token recurrent-state rollback proof.

All six raw F16 samples, one per device and depth, were independently decoded using binary16 exponent bits and matched their logged per-row counts and union. Raw logs, TSV traces, response JSON, samples and bounded runner are outside Git at `/home/chris/workspace/llama-perf-results/iteration-0013/`. The runner uses process groups, readiness/deadline checks and signal escalation; no server or profiler was left running. An intermediate 98K runner had a trace-snapshot `IndexError` after the request completed; the runner was corrected and the final 98K run completed successfully.

Coverage limitations: one server slot/stream and one request per depth; no mixed-sequence transition, no direct sequence identity or binary samples for every group, no GPU graph replay, and no independent GDN-state or logits comparisons. Audit itself uses a synchronized host read and graph disabling, so cannot establish behavior under GPU replay. The observations do not prove all future masks safe.

## Overflow and regression tests

Added opt-in deterministic Q4 boundary fixtures with KV=32768, per-query declared bound 2048 and capacity 8192. Four substantially disjoint rows produce unions 8191, 8192 and 8193. On **both ROCm0 and ROCm1**, sparse mode passed CPU-reference `test-backend-ops` for 8191 and 8192 at the existing NMSE <= 5e-4 tolerance. The 8193 case terminated with exit 134 and `QSA sparse index overflow: group 0 union 8193 exceeds capacity 8192 (KV=32768)` before gather, not a success or silently truncated result. The no-flag dense baseline passed all three fixtures. The audit independently reported correct 8191/8192 counts and rejected 8193 on CUDA0 and both ROCm devices. Core dumps were disabled for intentional aborts.

After the changes, the existing deterministic QSA-shaped CPU-reference 32K and 98K tests passed separately with the sparse flag unset and set on both ROCm devices (one selected test in each run). Selector guards and finite/NMSE checks were not relaxed. The overflow fixture deliberately violates the declared per-query bound in its final row to force a union larger than capacity; it is not presented as a valid QSA mask.

## Decision and next action

The actual model mask log directly reports `qheads=24 kvheads=2`, and GGUF metadata independently confirms head counts 24/2. `fattn-sparse-q4-tune.h::should_use` requires `gqa == 4`, so this model's GQA12 workload cannot exercise the new operator. Three-GPU **sparse** inference and matched sparse A/B were not run; claiming a zero-activation A/B as kernel validation would be misleading. No GitHub Actions or PRs were used.

Next, separately design and review a GQA12-capable sparse implementation and its tiling/dispatch, without relaxing this kernel's guard. First add graph-replay-safe fail-closed protection and extend the dense mask audit to mixed sequences and replay-equivalent observation; then establish fixed-token logits and GDN-state correctness through accept/reject/rollback before any sparse-model benchmark. Keep the new path opt-in.
