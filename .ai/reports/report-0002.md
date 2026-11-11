# Report 0002

Status: Completed for diagnosis; no transfer optimization justified. Handoff base `f9b64194c7115beb2abab0602ba942d3a7f9d464`. A previous attempt in a GPU-less sandbox recorded a blocker in the parent commit; this run had access to all requested GPUs.

## Build and measurements

The standalone collector test passed with C++17, `-Wall -Wextra -Werror -pthread`. The separate `build-perf` mixed CUDA/HIP build produced both backends, `llama-bench` and `llama-server`; enabled runs emitted correctly labeled CUDA and ROCm summaries on normal exit. Disabled runs emitted no upload summaries. No `set_tensor_2d` or `set_tensor_n` rows were observed in these workloads; this is not a general claim about their behavior.

Target-only runs used the IQ3_XXS model, CUDA0/ROCm0/ROCm1, layer split 4/7/7, flash attention, mmap, lazy mode, F16 K/V, no fit target, `-p 0 -n 128`, default warmup and one timed repetition per process. Three enabled and three disabled processes per depth ran sequentially, with no server running. Results in tokens/s:

| Depth | Disabled | Enabled | Means off / on |
| --- | --- | --- | --- |
| 0 | 46.48, 46.44, 46.45 | 46.44, 46.48, 46.44 | 46.46 / 46.45 |
| 32768 | 41.33, 42.21, 41.21 | 40.60, 41.63, 41.06 | 41.58 / 41.10 |

The enabled mean is 1.2% lower at 32K, but variation and the small sample prevent a precise overhead estimate. Earlier `build-mixed` one-run values were 46.51 and 41.64 t/s. These are benchmark comparisons, not MTP throughput or numerical checks. Summary counters cover the entire process, including model load, warmup and, at 32K, depth preparation; they are not isolated timed-decode counters.

## Upload attribution

In the depth-0 enabled run, per-device lifetime totals were CUDA0: 2,089 calls / 9,465,485,664 bytes; ROCm0: 2,286 / 17,005,838,560; ROCm1: 2,136 / 20,298,151,780. These totals include one-off model loading. The recurrent `#1` scheduler-copy rows had 1,778, 1,778 and 1,651 calls respectively, corresponding mostly to 127 copies of each named tensor and 254 of `attn_inp_k_idxs`. Their aggregate sync time was 5.84, 89.91 and 225.52 ms respectively; these are host blocking times, not demonstrated removable cost.

- Repeated small inputs include `inp_pos`, `attn_inp_k_idxs`, `attn_inp_v_idxs`, `rs_s_copy`, unnamed `leaf_*` and reshaped tensors. `rs_s_copy` is explicitly the recurrent-state copy-index input in `src/llama-graph.cpp`; on ROCm1 its 127 four-byte copies accounted for 173.28 ms of sync wait. A four-byte transfer does not establish a 173 ms transfer cost: it can wait for preceding work.
- Repeated larger rows include ROCm0 `node_1541` and `l_last-10`, each 127 x 40,960 bytes, and CUDA0 `model.input_embed (reshaped)` and `ple_embd`, each 127 x 10,240 bytes. The scheduler-generated `backend#source#copy` names identify split copies, but `node_1541` and anonymous leaves do not by themselves establish their semantic identities. No recurrent expert-weight uploads were established by these rows. Single-call named `blk.*.ffn_*_exps.weight` rows are consistent with model loading, not per-token expert traffic.
- At 32K, `#1` rows combine depth preparation with generation: ROCm0 `node_1541` has 144 calls / 340,787,200 bytes, maximum 20,971,520 bytes, and 55.58 ms sync wait; ROCm1 `rs_s_copy` has 144 calls / 576 bytes and 170.92 ms sync wait. These cannot be divided by 128 as pure decode costs.

The previous separate verbose 32K placement check reported 49/49 layers offloaded, 11/19/19 assigned to CUDA0/ROCm0/ROCm1; model buffers were 9,024 / 16,208 / 19,358 MiB and CPU-mapped buffers 260.50 and 27,465.95 MiB. Offloaded layers do not mean all weights reside on GPUs; the placement log does not identify all host-resident tensors. It is a separate placement run, not this benchmark's own startup log.

## MTP sample and decision

A separate instrumented server used the requested target and draft devices, 196608 context, F16 KV, 4/7/7 split, fit target 768, parallel 1 and MTP max 3. One uncached coding request generated 128 tokens at 61.24 t/s, with 72/162 draft tokens accepted. It exhausted the output budget while still in reasoning, so this is not a completed coding answer or a correctness test. ROCm1 `rs_s_copy#3` was called 54 times, four bytes each, with 22.47 ms sync wait; counters also include model load and prefill. This single short request does not establish long-context MTP behavior.

No optimization was implemented. `ggml_backend_sched_copy_input` synchronizes the previous destination-copy event before copying user inputs immediately, because the caller can overwrite them. Other inputs may depend on producer and destination events, and allocator regions can be reused across splits. The CUDA/HIP async tensor-copy implementation accepts device buffers only; fallback copies use synchronous tensor set, and cross-backend fallback may free a temporary host buffer immediately. `set_tensor_n` batches onto the per-thread stream but grouping arbitrary scheduler inputs without proving source lifetime, split ordering and destination dependencies is unsafe. The prominent small-copy wait may include earlier work, so its duration is not a validated speedup opportunity.

**Next priority:** add phase-separated, split/callsite attribution and a CUDA+ROCm timeline for representative long-context MTP requests, especially the work preceding ROCm1 `rs_s_copy`; establish source lifetime and stream dependencies before considering bounded batching. A transfer change would additionally require finite-logit, generated-output and recurrent-state tests, which were not claimed here.

Raw logs, responses and results: `/home/chris/workspace/llama-perf-results/iteration-0002/`. The original checkout, model files and `build-mixed` were preserved. No GitHub Actions were used.
