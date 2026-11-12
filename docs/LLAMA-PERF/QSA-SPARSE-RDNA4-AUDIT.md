# QSA sparse attention on RDNA4: source audit and experiment

Last reviewed: 2026-11-11. This is an experiment on our fork, not an upstream change or a verified speedup.

## Iteration 0016 diagnostic update

[Report 0016](../../.ai/reports/report-0016.md) identifies a first layer-11 HIP attention-output difference with byte-identical Q/K/V and mask, followed by a layer-15 mask divergence. This does not establish acceptable model-level accuracy; fixed-token logits and GDN rollback state remain unverified, so full-model performance A/B remains blocked.

For this specific investigation, `GGML_QSA_INPUT_DIR=<external-dir>` plus `GGML_QSA_MASK_AUDIT` dumps the first audited HIP0 layer-11 Q/K/V tensor spans, capped at 128 MiB per tensor, with type, dimensions and byte strides logged. It is disabled by default and synchronizes the stream. Spans can contain padding or unused cache data; keep them outside Git and interpret only using the logged strides. It is deliberately not a general sequence-aligned state capture.

## Iteration 0014 validation update

[Report 0014](../../.ai/reports/report-0014.md) validates the separately gated GQA12 operator and overflow rejection on both R9700s. A short 32K model probe confirmed sparse launches and matching output/acceptance history, but measured mask-union differences versus a reproducible dense baseline. Fixed-token logits and recurrent-state correctness remain unverified; no end-to-end performance claim or default promotion is justified.

## Iteration 0013 safety and model update

[Report 0013](../../.ai/reports/report-0013.md) records dense-path actual masks at both depths and an opt-in Q4 overflow check which aborts before gather on overflow. The check synchronizes the HIP stream and disables HIP graph capture while the sparse experiment is enabled; it is a safety prototype, not a performance-ready implementation. `GGML_QSA_MASK_AUDIT=<1..100000>` enables bounded, synchronous consumer-mask logging and disables GPU graph capture; see the report for coverage and limitations. Leave both variables unset in normal operation.

The tested model's actual Q/KV head counts are 24/2, i.e. GQA12. The original selector requires GQA4 and cannot activate for that model. Do not infer a model speedup from the GQA4 microbenchmark or broaden its shape guards.

## Iteration 0012 hardware validation update

[Report 0012](../../.ai/reports/report-0012.md) verified the mixed HIP/CUDA build, sparse dispatch and CPU-reference correctness on both R9700s. The opt-in synthetic Q=4/GQA4 whole-op test (including mask compaction) measured 98K medians of 342.54 -> 135.14 us on ROCm0 and 342.98 -> 132.04 us on ROCm1. ROCm0 had an unexplained 204.53 us sparse sample in its three-run range. These are **synthetic operator** timings, not model inference speedups.

At that point the **full-model safety gate remained blocked**: real QSA mask unions had not been observed and the compactor could silently clip an over-capacity union. See the iteration 0013 update above for subsequent work and remaining limitations. Production defaults remain unchanged.

## Iteration 0013 findings and GQA12 prototype

[Report 0013](../../.ai/reports/report-0013.md) audited the *actual* Q=4 final F16 masks with sparse dispatch disabled and GPU graph capture/replay disabled by the audit. At 32K/98K, the measured maximum per-query mask count was <= 2051 and maximum four-query unions were 5493 / 5863, below the 8204 allocated capacity. Accepted and rejected drafts were observed, but these samples do **not** cover GPU replay, mixed sequences or fixed-token logits/GDN-state equivalence. The Q4 experimental compactor now synchronously checks *unclamped* union counts before gather and fails closed on overflow; this adds per-operation synchronization and invalidates Report 0012's performance numbers for the modified path.

The target model's 24 Q heads / 2 KV heads imply **GQA12**, not GQA4. The original `GGML_HIP_QSA_SPARSE_Q4=1` guard remains GQA4-only. A **separate** prototype, `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`, opts into the existing sparse 256x256, (ncols1,ncols2)=(4,4) WMMA specialization for RDNA Q=4/GQA12. Conceptually, GQA12 uses three ncols2=4 tiles per KV head, sharing the one Q4 mask-union index list; see Report 0014 for device validation. The new selector requires exactly the original dimensions, types, alignment/optimization eligibility, one sequence and positive selection bound, except it requires GQA12 rather than GQA4. Both flags are disabled by default and independent.

Host-side C++17 selector tests and static fixture-cardinality checks passed in the Chat AI runtime. These checks were subsequently performed in Report 0014; model-level numerical correctness remains unverified. Opt-in `GGML_TEST_QSA_GQA12_FA=1` adds test/perf fixtures at KV32768 and KV98304 with 2048 and 5120-position unions; `GGML_TEST_QSA_GQA12_BOUND_FA=1` adds 8191/8192/8193 boundary tests in correctness mode. Every sparse model experiment must still pass safety and fixed-token/GDN correctness gates first.

Experimental sparse attention intentionally disables HIP graph capture/replay because of its host-synchronized overflow check. `GGML_PERF_DISABLE_CUDA_GRAPHS=1` is a **separate opt-in, graph-off dense control** (also affects mixed CUDA/HIP backends) for isolating the timing impact of graph suppression. Any eventual full-model comparison must report both production graph-enabled dense behavior and matched graph-off dense-vs-sparse timing; the latter cannot by itself establish a production speedup.

## Iteration 0014 and first-divergence diagnostic

[Report 0014](../../.ai/reports/report-0014.md) confirms **GQA12** sparse dispatch (all 24 heads), CPU-reference correctness, fail-closed overflow tests and inclusive whole-op performance on both R9700s at 32K/98K. With the wide 5120-position union at 98K, dense/sparse medians were 963.26/186.08 us (ROCm0) and 969.62/181.35 us (ROCm1); these are synthetic operator measurements, not model speedups.

A short 32K real-model graph-disabled diagnostic produced identical sampled messages and accepted/drafted outcomes, but **181/273 Q4 mask union counts differed** between dense and sparse processes (maximum count delta 52); a second independent dense process reproduced all dense union counts. This could reflect numerical sensitivity in QSA's indexer or a numerical/state error; matching sampled tokens is insufficient. Do not benchmark or promote sparse full-model inference until the earliest divergence is characterized and fixed-token logits/GDN-state comparisons are completed. Mixed-sequence and GPU graph-replay correctness are not proven.

To inspect *actual selected indices* rather than only union sizes, use the disabled-by-default consumer-mask audit in two separate fresh model processes with the same tokens/seed/configuration:

- `GGML_QSA_MASK_AUDIT=1000`: bounded synchronous audit of Q=4 masks. This intentionally disables GPU graph capture/replay and changes timing.
- `GGML_QSA_MASK_AUDIT_DIR=<empty-external-directory>`: save raw contiguous four-row F16 masks; use separate directories for dense and sparse runs.
- `GGML_QSA_MASK_DUMP_LIMIT=256`: newly added optional upper bound of 1–256 raw groups **per backend/device**; without it, the prior first-sample behavior remains. Multi-sample files are keyed by backend/device/invocation/stream/KV. An exhausted limit is logged. The final tensor F16 values are saved as little-endian binary16, row-major `[4,KV]`. Keep raw files and model traces outside Git.
- `python3 docs/LLAMA-PERF/tools/test_compare_qsa_mask_dumps.py`: host-only standard-library regression tests (not yet run by Chat AI after repository commit).
- `python3 docs/LLAMA-PERF/tools/compare_qsa_mask_dumps.py <dense.log> <sparse.log> > mask-diff.json`: align mask audit rows by backend/device and per-device order, validate logged counts against raw F16 data, and report the first changed position per query/union and the first differing finite mask values. Compare dense-vs-dense first to establish run-to-run stability. The tool **does not validate logits or GDN state**. Ensure trace step/sequence alignment separately: per-device order is not a global causal clock.

This instrumentation is **diagnostic only**. It is disabled when the environment flags are unset; it is neither an inference-speed test nor a GPU graph replay correctness check. See Handoff 0015 for next validation gates.

## Iteration 0015 and the first-attention-output diagnostic

[Report 0015](../../.ai/reports/report-0015.md) compared 273 fully validated raw QSA mask pairs in a graph-disabled, 32K real-model probe. Dense A and dense B matched **273/273 masks byte-for-byte**. Dense versus sparse differed on **210/273 masks**, with **181/273 differing four-query union counts**, while per-query counts remained unchanged. The earliest observed changed mask came from HIP0 during the final prompt-processing batch of four tokens, **before any MTP draft/accept/reject/rollback**. These measurements do not identify whether the upstream cause is sparse attention accumulation, indexer/top-k sensitivity, or another state/ordering difference. Fixed-token logits and recurrent GDN state were not measured.

For the next bounded investigation, Chat AI has added an opt-in, **post-kernel F32 QSA attention output dump** and a stable model-layer name in `src/llama-graph.cpp::build_attn_mha` when `GGML_QSA_MASK_AUDIT` is present. These new diagnostics have **not been validated by a HIP/CUDA build or actual model run**. The existing F16 mask audit and dump may be paired with the new output capture by `backend/device/id`. Each output is a raw little-endian F32 tensor shaped `[value_dim, query_heads, Q_positions, streams]`. Use unique external directories for each fresh process and set:

- `GGML_QSA_MASK_AUDIT=1000` plus `GGML_QSA_MASK_AUDIT_DIR=<mask-dir>` and `GGML_QSA_MASK_DUMP_LIMIT=256` to capture and validate mask inputs.
- `GGML_QSA_ATTN_OUT_DIR=<output-dir>` and `GGML_QSA_ATTN_OUT_LIMIT=256` to synchronously save up to 256 post-kernel QSA outputs per backend/device. This requires the mask audit. With the output directory unset, no new post-kernel synchronization/dumps are performed. The output limit defaults to one if omitted.
- `python3 docs/LLAMA-PERF/tools/test_compare_qsa_attn_out.py` to run the new host-only comparator tests (Chat AI static review only; not yet executed on the Coding Agent host).
- `python3 docs/LLAMA-PERF/tools/compare_qsa_attn_out.py <dense.log> <other.log> > attn-output-diff.json` to report first changed layer/output, maximum absolute and relative difference, normalized error, first changed feature/head/query, and eight worst individual heads. Compare dense A versus B *first*. Separately use `compare_qsa_mask_dumps.py` to prove whether the input mask at that same operation was identical.

These graph-disabled, synchronized output dumps are **diagnostics, not performance measurements**. The comparator checks tensor name, output shape, device order, mask-bound metadata and finiteness, but cannot itself prove identical Q/K/V inputs, aligned token IDs, or equivalent GDN rollback state. Infer causality only after aligning true model steps and comparing surrounding inputs. Full-model performance A/B and production promotion remain blocked; see Handoff 0016.

## Iteration 0016 result and FP64 reference replay

[Report 0016](../../.ai/reports/report-0016.md) established that at the first diverging attention output on **HIP0 layer 11**, in the final four-token prompt batch before any MTP drafts, the dense and experimental GQA12 sparse paths consumed **byte-identical** Q, K, V tensor spans and the same actual F16 mask (3537 selected KV union). All 24,576 layer-11 F32 output values differed: maximum absolute error 0.00535035, dense-energy NMSE 8.97579e-7. The subsequent layer-15 QSA top-k/selection differed. Dense A/B repeated mask and output bytes exactly. **These data isolate a kernel-path numerical difference but do not establish a correct reference, acceptable numerical tolerance, causal top-k sensitivity or recurrent-state/logit equivalence.**

Chat AI added a **disabled-by-default** exact-capture metadata line `QSA_INPUT_META` to `ggml/src/ggml-cuda/fattn.cu`, emitted alongside the existing first HIP0 layer-11 Q/K/V capture. It records the actual `scale`, `max_bias`, `softcap`, and whether attention sinks are present, plus backend/device/layer/invocation ID. The corresponding new analyzer `docs/LLAMA-PERF/tools/replay_qsa_captured_attention.py` uses NumPy (the only external Python dependency) to:

- read raw Q(F32) and strided K/V(F16) spans using the recorded GGML shapes and **byte strides**, respecting noncontiguous views/padding;
- require bit-identical dense/sparse Q/K/V captures and identical F16 masks; verify mask finite counts/union against the logged live audit and verify attention-scale equality;
- reject unsupported ALiBi, logit-softcap and sink variants rather than silently assuming a normal softmax;
- compute an FP64 `softmax((Q K^T)*scale + mask) V` reference using only selected KV rows and the actual 24-query/2-KV-head GQA mapping;
- report dense vs FP64, sparse vs FP64 and sparse vs dense errors, overall and worst per head (max absolute and normalized squared error).

Use **fresh** graph-disabled dense and sparse runs, each with `GGML_QSA_MASK_AUDIT=1000`, `GGML_QSA_MASK_AUDIT_DIR`, `GGML_QSA_MASK_DUMP_LIMIT=256`, `GGML_QSA_INPUT_DIR`, `GGML_QSA_ATTN_OUT_DIR` and `GGML_QSA_ATTN_OUT_LIMIT=256` pointing to distinct external directories; only sparse enables `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`. Run `python3 docs/LLAMA-PERF/tools/test_replay_qsa_captured_attention.py` and `python3 docs/LLAMA-PERF/tools/replay_qsa_captured_attention.py dense.log sparse.log`. The script/test are new and **not yet tested on the agent host**. Chat AI exercised only a small independent synthetic NumPy attention calculation locally; full source build and real captured replay are next-agent work.

The FP64 result is a high-precision mathematical **reference**, not proof of bitwise GPU behavior: even a closer match does not establish acceptable model-logit or GDN-state correctness, which must be measured separately. Do not promote experimental sparse attention or run long model performance A/B until those gates pass. See Handoff 0017.

## Iteration 0017 FP64 result and layer-15/logical-state diagnostics

[Report 0017](../../.ai/reports/report-0017.md) verified the real Qwen3.8-Flash-Next layer-11 attention replay on **byte-identical Q(F32), K/V(F16), F16 mask and scale 0.0625**. The sparse kernel had lower overall FP64-reference NMSE (4.44979e-7) than dense tile (9.99118e-7), but sparse had slightly higher largest absolute difference (0.00825457 vs dense 0.00801139). Sparse-vs-dense NMSE was 8.97579e-7. **No sparse kernel correctness defect was confirmed**; this is a local mathematical-reference comparison, not a validated model tolerance. Layer-15 selection cutoff, fixed-token logits and recurrent GDN state remain unmeasured. Do not run full-model performance A/B or promote sparse by default.

Chat AI subsequently added these **disabled-by-default, not yet GPU-tested** diagnostics:

### A. Bounded HIP0 layer-15 score/top-k and pool-map capture

With the existing `GGML_QSA_MASK_AUDIT=1000` (which disables GPU graph replay), set `GGML_QSA_TOPK_TRACE_DIR=<unique-external-directory>` and optionally `GGML_QSA_TOPK_TRACE_LIMIT=64` (1..128 per capture kind). Only four-query operations are captured; larger prefill batches and other query counts do not consume the limit. When set, the Qwen4-exp graph names layer-15 top-k nodes `qsa_topk_il15` and fused mask nodes `qsa_topk_mask_il15`. The HIP0 CUDA-backend dispatcher synchronously writes raw little-endian `score` (F32), `topk` (I32), and, **for the fused top-k mask path**, `blocks` (I32 pool→KV member map) and `extra` (I32 tail cells). Each part is individually bounded to 32 MiB and combined raw trace output is capped at 512 MiB per process (abort on overflow); logs include backend/device, kind, group ordinal, KV, pool count, top-k count, query count, tensor type, shape, strides and path. This is intentionally synchronous **diagnostic-only**; do not time these runs. A nonfused TOP_K-only trace can compare selected pool IDs and score margins but cannot map them to selected KV cells.

Use separate fresh dense A, dense B and sparse process directories and compare:

- `python3 docs/LLAMA-PERF/tools/test_compare_qsa_topk_trace.py` — six host-only synthetic tests for score margins, swapped pool IDs, pool→KV mapping, nonfused fallback, invalid scores and malformed dumps.
- `python3 docs/LLAMA-PERF/tools/compare_qsa_topk_trace.py dense-A.log dense-B.log --require-identical-selections > topk-control.json`
- `python3 docs/LLAMA-PERF/tools/compare_qsa_topk_trace.py dense-A.log sparse.log > topk-sparse.json`

The comparator compares unordered selected pool *sets*, per-query cutoff margins, changed pool scores and the actual mapped KV cells (including changes when the selected pool IDs are unchanged but their membership maps differ). An all-equal pool-set result is **not** proof that scores, caches or logits are bitwise equal. Group ordinals are per-device/per-kind, **not** logical global step identities: correlate the final prompt batch with the server trace.

### B. Strict bounded logits / recurrent-state comparison contract

`docs/LLAMA-PERF/tools/compare_qsa_model_state.py` is a host-only standard-library comparator; the opt-in C++ observers described below collect logits and GDN state. Its input is a separate JSONL manifest per dense A/dense B/sparse run, with strictly increasing `step` as a **checkpoint ordinal**, not generation count. Each logical event records exact `phase, seq, position, target_ids, draft_ids, accepted, rollback`; each tensor record specifies the same logical event, `layer`, tensor role `logits|gdn_r|gdn_s|gdn_p`, little-endian `dtype=f32|f16|bf16`, `shape`, and external binary `path`. Physical cell and rollback plane can be logged for attribution but are deliberately *not* logical matching keys. The comparator bounds each raw tensor file to 64 MiB and each manifest to 16384 records and 32 MiB. Logits refer to **pre-sampling** target outputs.

An example *format only* (not actual measured model data):

```jsonl
{"kind":"event","step":0,"phase":"after_prompt","seq":0,"position":33024,"target_ids":[71],"draft_ids":[],"accepted":[],"rollback":false}
{"kind":"tensor","step":0,"phase":"after_prompt","seq":0,"position":33024,"layer":-1,"name":"logits","dtype":"f32","shape":[1024],"path":"logits.bin"}
{"kind":"tensor","step":0,"phase":"after_prompt","seq":0,"position":33024,"layer":11,"name":"gdn_s","dtype":"f16","shape":[128],"path":"gdn-s.bin","physical_cell":17,"rs_plane":0}
```

Compare dense A vs B first; then A vs sparse using `python3 docs/LLAMA-PERF/tools/compare_qsa_model_state.py <dense.jsonl> <other.jsonl>` (optional `--abs-tol` and `--rel-tol` are **reporting thresholds only**). The comparator **fails closed** if token IDs, accepted/rollback decisions or checkpoint identities differ; it verifies snapshot shapes and finite content and reports max absolute, relative, normalized errors and first differing elements. The observer itself must synchronize device reads, resolve **logical** GDN row and pending rollback snapshot plane correctly, and avoid serializing an entire long-context KV cache.

Neither new diagnostic nor its tests have been executed in Chat AI's local workspace: GitHub DNS access is unavailable, and the connected code was reviewed but not locally compiled. See Handoff 0018 for mandatory first-step validation, bounded GPU evidence and numerical go/no-go.

## Iteration 0018: top-k and verification-logit diagnostics

[Report 0018](../../.ai/reports/report-0018.md) measures layer-15 score cutoff swaps and captures actual pre-sampling verification logits with matching token/candidate histories. `LLAMA_QSA_LOGITS_DIR=<fresh-empty-external-directory>` enables a diagnostic for the first 16 verification groups on slot 0 (text only): it synchronously copies the full F32 vocabulary row for every verification batch index, and writes an external `manifest.jsonl` with the actual input history, draft IDs, sampler result and pending rollback indicator. It does not force tokens or acceptance, and does not capture logical GDN state or a post-rollback state. Do not use for performance or infer recurrent-state correctness. Compare manifests with `compare_qsa_model_state.py`; it rejects mismatched histories.

## Iteration 0018 findings and new opt-in logical GDN observer

[Report 0018](../../.ai/reports/report-0018.md) confirmed that layer-15 QSA selected-pool changes arise from a **real score-cutoff crossing**: first query swaps pool 1040 for 6316, remapping four KV entries 4160–4163 to 25264–25267, with identical pool-to-KV maps. Dense cutoff margin was only 0.0034008; sparse margin 0.0010815; maximum paired F32 score difference was 0.0169210. This establishes top-k sensitivity, but not that layer-11 arithmetic alone caused it.

The same iteration aligned 64 actual, naturally sampled MTP target-verification logit rows (16 groups: eight fully accepted and eight rejected/rollback pending) with identical token/draft/acceptance histories. Dense A/B were **bitwise identical** across all full-vocabulary vectors. Sparse differed at **all 64 rows**, with max abs logit difference **0.9186411** and maximum per-row NMSE **0.00462375**. This is materially different from the earlier FP64 attention-only NMSE (~1e-6), so **model numerical equivalence is not established**. Do not change tolerance to make the divergence disappear; investigate the causal chain and downstream behavior. Full-model A/B throughput, graph replay and default promotion remain blocked.

### New recurrent-only observer: disabled by default, not yet compiled or GPU-verified

The internal diagnostic-only bridge `src/llama-qsa-debug.h` / `src/llama-context.cpp`, recurrent-cache reader in `src/llama-memory-recurrent.cpp`, and server integration use `LLAMA_QSA_GDN_DIR=<fresh-external-directory>`. See Report 0019 for local validation and remaining limits. It captures slot 0, text-only speculative verification for at most the first 16 verification groups, at the following boundaries:

- `pre_verification`: after the target model computed verification logits, before sampling/acceptance;
- `post_verification`: after sampling decisions, before any cache rollback or commit;
- `post_accept`: after the accepted-token history and cache truncation/commit;
- `post_rollback`: after a rejected group either triggers checkpoint restoration and sequence removal or updates the recurrent rollback state through sequence removal. For pending rollback, snapshot-plane identity is logged; **reprocessing on the next inference step is a separate event**, not implied by a post-rollback snapshot.

For every snapshot, the server appends an event to `manifest.jsonl` with strictly increasing checkpoint ordinal `step`, `phase`, `seq`, logical verification starting `position`, the same actual full target token history and draft IDs, `accepted`, `rollback`, and `replay` fields. The internal bridge synchronizes the model and dynamically locates its hybrid recurrent cache. It resolves the active sequence tail and current logical source row using `rs_idx[seq] * size + (cell.src >= 0 ? cell.src : tail)`, *matching* `llama_memory_recurrent::state_write`. Only current R/S/P row data is read; all existing attention/indexer KV caches are excluded. Each tensor record includes `layer`, `name=gdn_r|gdn_s|gdn_p`, dtype, 1-D row shape, the raw little-endian file path, active physical cell, resolved source row, pending rollback snapshot-plane ID and logical cell position. Reads occur via `ggml_backend_tensor_get` after `llama_context::synchronize`. Unsupported layouts, absent sequence ownership, impossible row or snapshot indices, invalid paths, write failures and missing recurrent tensors cause diagnostic failure instead of silently comparing incorrect state.

The observer enforces an 8 MiB per-row limit and 8 GiB aggregate raw output limit per process; every selected tensor range must also match the recurrent serializer's own range for the sequence. Its separate `LLAMA_QSA_GDN_DIR` keeps the state manifest distinct from `LLAMA_QSA_LOGITS_DIR` (the latter's full-vocabulary logits observer already worked in Report 0018). **Run both flags in each dense and sparse process** for complementary evidence and unique, initially empty external directories. All captures add GPU synchronization and must never be used for benchmarking. Do not commit raw captures or token histories.

The host comparator `docs/LLAMA-PERF/tools/compare_qsa_model_state.py` now allows up to 16,384 manifest records, with a 32 MiB manifest cap, and additionally rejects misaligned `logical_cell_pos` between corresponding tensor records. Physical cell numbers and rollback plane IDs are attribution fields rather than comparison keys. Its existing tests were extended for checkpoint-phase alignment and logical-cell-position mismatches; the new source/tests require local validation.

Use:
```text
python3 docs/LLAMA-PERF/tools/test_compare_qsa_model_state.py
python3 docs/LLAMA-PERF/tools/compare_qsa_model_state.py dense-A-gdn/manifest.jsonl dense-B-gdn/manifest.jsonl
python3 docs/LLAMA-PERF/tools/compare_qsa_model_state.py dense-A-gdn/manifest.jsonl sparse-gdn/manifest.jsonl
```

### Full-vocabulary logit distribution analysis

New host-only `docs/LLAMA-PERF/tools/compare_qsa_logit_distributions.py` reads the **existing, real full-vocabulary** `LLAMA_QSA_LOGITS_DIR` manifests. It requires exact equality of recorded token histories, draft candidates, acceptance/rollback, replay and sequence/position checkpoint IDs and processes at most 64 F32 vocabulary rows by default (max 256). It reports the number of top-1/top-5 set changes, dense and other top-1 probabilities, KL divergence in both directions, Jensen–Shannon divergence (nats), and centered-logit error after removing an irrelevant constant shift. The tool does not require NumPy, infer ground-truth token labels or certify model correctness; it cannot replace recurrent-state validation or a task-quality evaluation. New `test_compare_qsa_logit_distributions.py` includes six synthetic tests. These tools were committed but **not executed in the Chat AI workspace**.

Use `python3 docs/LLAMA-PERF/tools/compare_qsa_logit_distributions.py dense-A-logits/manifest.jsonl dense-B-logits/manifest.jsonl` for repeat-control and then the same command with the sparse manifest for the first probability-level assessment.

**Caution:** Report 0019 checks actual tensor ranges against recurrent-only serialization and dense A/B repeat stability, including pending rollback planes and later verification. It does not validate live `cell.src` indirection or checkpoint restore, and the observer covers only one text slot and the first 16 verification groups. Production mixed sequences and GPU graph replay remain unverified.

## Iteration 0019 results and targeted source-row/restore coverage

[Report 0019](../../.ai/reports/report-0019.md) verified 16 naturally aligned 32K MTP verification groups on each of dense A, dense B and sparse. Each process captured 48 GDN checkpoints (3,504 R/S/P rows) over both accepted and rejected/rollback paths; dense repeats matched all rows bytewise while sparse differed in 2,592/3,504 rows, including 1,296/1,728 R and 1,296/1,728 S rows (P unchanged 0/48). First changed GDN value was at layer 12 pre-verification, *before* first sampling. Maximum dense-energy row NMSE was 0.00195039 (R), 0.00154951 (S). Dense A/B logits were identical; dense/sparse full-vocabulary logits had 0/64 top-1 flips, 5/64 changed top-5 sets, max KL(dense||sparse) 0.0106312 nats, max JS 0.00260841 nats. The full-softmax probability of the same top-ranked token changed from 0.5710 to 0.6347 at one checkpoint. **No justified acceptance criterion or model numerical equivalence has been established. No production promotion or 512-token performance A/B is allowed on this evidence.**

The observer was independently cross-checked against the *range-selection* metadata requested by the recurrent serializer, including active rollback snapshot planes 2/3 and the normal plane 0, over all 36 R, 36 S and one P layers per checkpoint. But **all live source rows were physical cell 0**, so nontrivial `cell.src` indirection and the checkpoint-restore/replay path remain untested. This is an observer-validation gap distinct from the already-observed sparse numerical differences.

Chat AI then added a bounded `LLAMA_QSA_GDN_GROUP_LIMIT=1..16` (default 16) to permit a one- or two-group verification experiment without the full 5.3 GiB capture. The limit is parsed only when `LLAMA_QSA_GDN_DIR` is set and leaves production defaults unchanged.

Chat AI also added minimal **disabled-by-default metadata** to avoid repeated huge captures for coverage discovery:

- Each `LLAMA_QSA_GDN_DIR` GDN tensor record now includes `source_cell` (actual `cell.src`, possibly -1), `cache_size` (recurrent memory cells), plus the existing `physical_cell`, `logical_source_row`, `rs_plane` and `logical_cell_pos`.
- Each GDN checkpoint event now includes `rollback_mode`: `none`, `snapshot` or `checkpoint_restore`, determined by the target's existing rollback decision. **This does not change** sampling, acceptance, rollback semantics or state.
- New host-only `docs/LLAMA-PERF/tools/audit_qsa_gdn_coverage.py` checks every recorded tensor's `logical_source_row = rs_plane * cache_size + (source_cell >= 0 ? source_cell : physical_cell)`, consistency across layers within each checkpoint, completeness of R/S roles and the actual observed source/rollback coverage. It can fail with exit 2 when `--require-source-indirection` or `--require-checkpoint-restore` is requested. Its synthetic test `test_audit_qsa_gdn_coverage.py` covers both source indirection, rollback types and malformed data.
- These new fields **are not present** in Iteration 0019's old manifests. Use new small, bounded targeted runs for this audit; do not relabel old captures as proving the missing cases. Metadata and tests are not yet compiled or executed on the local agent's machine.

For new GDN captures:
```text
python3 docs/LLAMA-PERF/tools/test_audit_qsa_gdn_coverage.py
python3 docs/LLAMA-PERF/tools/audit_qsa_gdn_coverage.py   /home/chris/workspace/llama-perf-results/iteration-0020/<run>/manifest.jsonl
```

**The audit verifies row *selection metadata*, not actual bytes.** Independently compare the chosen row bytes with a bounded recurrent-only serializer on an actual nontrivial source-cell and checkpoint-restore fixture before claiming those cases validated. Use a small 1–2-group test or temporary local fixture, not a repeated 5.3-GiB full-context snapshot. See Handoff 0020.

## Iteration 0020 and narrow serializer-backed verification

[Report 0020](../../.ai/reports/report-0020.md) validated a real natural **checkpoint-restore and subsequent replay**, in addition to Iteration 0019's recurrent snapshot rollback, and fixed a diagnostic-only replay classification error. In a short ngram-simple run, one rejected draft group caused FULL checkpoint restoration, followed by a fully accepted replay group. The mixed CUDA/HIP build and 48 Python tests passed. However, the measured physical/serialized source cell remained 0 and actual serialized **byte contents** were not independently compared. Existing sparse-vs-dense logits and GDN drift still make numerical correctness **NO-GO**.

### Actual serialized payload comparison (opt-in, unverified on hardware)

Chat AI added two independent **capture traversals** within `llama_memory_recurrent::qsa_debug_dump_state`:

- Set `LLAMA_QSA_GDN_SERIAL_REF_LAYER=12` (replace 12 with any present R/S layer) to cause the **real recurrent `state_write` tensor-write callbacks** to produce raw R/S and available P payload files for just that layer. The companion `serializer-reference.jsonl` logs checkpoint, role, tensor type, serialization-selected byte offset, byte count and file path. The serializer visitor independently selects tensor ranges and writes their actual bytes rather than only recording requested offsets.
- Set `LLAMA_QSA_GDN_LAYER_FILTER=12` to make the regular GDN observer dump **only the same layer**. It continues checking `state_write` range metadata for *every* layer, but skips full-state copies, substantially reducing diagnostic I/O. Both flags are disabled by default and accept only existing R/S layer IDs. If both are provided, they must select the same layer.
- Combine with `LLAMA_QSA_GDN_DIR=<new external directory>` and `LLAMA_QSA_GDN_GROUP_LIMIT=1` for a bounded checkpoint, along with normal model audit settings if needed. No extra huge attention/indexer KV serialization is performed.
- Compare `python3 docs/LLAMA-PERF/tools/compare_qsa_gdn_serialized_bytes.py <dir>/manifest.jsonl <dir>/serializer-reference.jsonl --require-identical`. The standard-library comparator checks every checkpoint's R/S/P reference coverage, matching exact tensor roles/types, recorded logical row offsets, bounded file lengths and **every raw byte**; it reports a first mismatched byte and fails if `--require-identical` is requested. `test_compare_qsa_gdn_serialized_bytes.py` supplies seven synthetic regressions.

**Limitation:** both the serializer visitor and the observer read device data using `ggml_backend_tensor_get`. They follow independently selected tensor-offset/callback paths, but this is **not** a second independent GPU transfer implementation. Byte equality validates serializer-vs-observer agreement, not correct attention numerics or end-to-end equivalence. The new C++ code and Python tests have not yet run on the agent's host.

### Staged input source versus serialized state source

Source review of `llama_memory_recurrent::find_slot` identified a distinction that changes the appropriate test: during a shared-sequence split, `cell.src0` retains the *staged graph-input source* (used by `llama_memory_recurrent_context::s_copy`), while the algorithm normalizes `cell.src = cell_index` after preparing the slot. The normal stable-state serializer uses **`cell.src`**, not `cell.src0`. Thus a real sequence copy can involve cross-cell state **without leaving a nontrivial serialized `cell.src`** in the post-decode checkpoint. A test that requires `cell.src != physical_cell` after every normal decode may be unattainable by design.

For coverage, the GDN manifest now also logs `staged_source_cell=cell.src0`, independently of `source_cell=cell.src`. The updated `audit_qsa_gdn_coverage.py` reports `nontrivial_staged_source_rows` and supports `--require-staged-source`. **This only proves the staged index is nontrivial**, not that a genuine shared-sequence copy happened or that the resulting device state bytes are correct. Validate provenance through a supported `seq_cp`/split fixture and compare the resulting R/S bytes to the expected source and serial reference.

Do not forge production `src` values, equate `src0` with the serialized logical row, weaken overflow/math/defaults or run long model performance A/B. The next narrow task is in Handoff 0021.

## Why investigate sparse attention?

Report 0010 measured the ordered target-synchronization waits at 32K vs 98K:
ROCm0 median 13.709 -> 18.126 ms (+4.417 ms); ROCm1 16.205 ->
19.609 ms (+3.404 ms). GPU kernel-family overlap increased primarily for
flash attention, but these overlaps do **not** prove full causal latency shares.

Report 0011 ruled out a dense tile-to-WMMA dispatch swap. On two R9700s,
FP16 Q=4/GQA4 fixed-work attention was 15.5%-25.4% *slower* with WMMA
than with the default tile, despite passing the CPU-reference tolerance.
The correct next question is whether sparse QSA can avoid unnecessary K/V
data reads, not whether denser WMMA is faster.

Qwen3.8-Flash-Next's QSA indexer selects roughly 2048 context positions
per query (plus the incomplete tail). The fork's model graph in
`src/models/qwen4exp.cpp::build_qsa_sel` constructs a sparse selection
but passes an **F16 mask over the full KV cache** into
`build_attn_qsa` -> `build_attn_mha` ->
`GGML_OP_FLASH_ATTN_EXT`. A mask alone does not mean K/V reads are
sparse. The dispatch and launch path determine that.

For the tested Q=4, head dimension 256, GQA4, F16 K/V case, the
unmodified RDNA selector chooses the dense `flash_attn_tile` path.
The existing MMA sparse kernel specialization list has 256x256 cases
for (ncols1,ncols2)=(1,8) and (8,8), not (4,4). The mask
compactor previously accepted only 1 or 8 query rows per group.
Its sparse-eligibility predicate accepted NVIDIA tensor cores or
AMD MFMA, but not RDNA WMMA. This combination leaves the 4-query
RDNA GQA4 MTP path scanning K/V for the complete context.

## Prior art, use as reference (do not copy unrelated changes)

- Upstream llama.cpp issue #28012, "qwen4exp: decode cost grows
  ~linearly with context":
  https://github.com/ggml-org/llama.cpp/issues/28012
- Upstream PR #28770 (merged 2026-09-20), CUDA sparse flash attention
  for Qwen4 and the group-mask union idea:
  https://github.com/ggml-org/llama.cpp/pull/28770
- Upstream PR #28213 (closed), single-token
  gather-based QSA decode and mask-preserving dense attention:
  https://github.com/ggml-org/llama.cpp/pull/28213
- Issue #28734, independent 5x RTX3090 experiments with contextual
  throughput gains when removing dense attention/pooled-key overhead:
  https://github.com/ggml-org/llama.cpp/issues/28734
- Issue #29281, real regression in the sparse index compactor:
  an expensive per-query dynamic inner loop in CUDA sparse FA can be
  slower at small batch sizes. Even a correct gather can lose to dense
  attention if index compaction scans the entire mask.
  https://github.com/ggml-org/llama.cpp/issues/29281
- Upstream PR #28699, QSA pooled indexer key caching; this is a
  separate source of context-length growth and not the present
  experiment:
  https://github.com/ggml-org/llama.cpp/pull/28699
- The independent R9700 fork by JohnTDI-cpu (graph and kernel-fusion
  results on Qwen3.8-Flash-Next) is useful for future bit-exact
  validation and distinct overhead sources, but its 65K-context
  numbers are not comparable directly to our 98K, three-GPU setup:
  https://github.com/JohnTDI-cpu/llama.cpp-flash-next-rdna4

## Experiment implemented on our fork

**Only if** `GGML_HIP_QSA_SPARSE_Q4=1`, enable existing
`BEST_FATTN_KERNEL_MMA_F16` sparse gathering with a newly
instantiated 256x256, (ncols1,ncols2)=(4,4) kernel variant for
a strict RDNA + WMMA + FP16 KV + Q=4 + one sequence + GQA4 mask
shape. The helper additionally requires a positive `n_kv_max`
selection bound and at least 8x that count in KV positions.
The normal `GGML_HIP_FATTN_FORCE_WMMA_Q4` experiment remains
separate and must be unset.

The F16 mask-to-index compactor adds the 4-query case and computes
the union of positions visible to any of the four queries. Every
query still reads its original per-row mask in the MMA kernel, so
union does **not** mean treating a masked position as valid.
The sparse launch bounds read from `n_kv_max`: it allocates up to
4 times the per-query upper bound and gathers only masked-selected
K/V rows. **A correct upper bound for each query is mandatory**;
do not run this path with arbitrary masks or pretend that the
number is merely a tuning parameter.

Existing `GGML_TEST_MTP_FA` dense Q=4 tests remain unchanged.
New `GGML_TEST_QSA_SPARSE_FA=1` fixtures deliberately place up
to 2048 selected positions across the full KV and give each of
four queries a slightly different subset. They test CPU-reference
masked attention at 32768 and 98304 positions with two KV heads
and GQA4, with F16 K/V and the existing finite/NMSE gate.
The synthetic masks are not exact QSA model output.

At the original prototype handoff, HIP compilation and runtime performance
were not yet verified. Report 0012 subsequently passed the hardware
operator gates; **live QSA mask safety and model-level correctness are still
unverified**, so the path remains disabled by default.

## Explicit go/no-go gates

1. HIP/CUDA build succeeds; host C++17 selector test succeeds.
2. Baseline QSA-like tests pass CPU-reference comparison.
3. Sparse flag actually launches a sparse mask compactor and the
   sparse 256x256 (4,4) MMA variant on both R9700 cards.
4. Both 32K/98K sparse flag variants pass CPU reference with all
   finite outputs and NMSE <= 5e-4.
5. The opt-in sparse path has a repeatable device-local whole-op
   latency win on both cards at 98K. Profile its compaction and
   gather kernels together; do not exclude setup overhead.
6. **Only then** run the three-GPU matched MTP 32K/98K A/B,
   with unchanged accepted/drafted counts, output, per-step timings
   and, before promotion, teacher-forced logits/GDN state checks.

Avoid changing scheduler ordering, HIP synchronization, launch block
counts, default KV types, acceptance settings, or model semantics.
If QSA's actual mask union exceeds its advertised bound, stop and
investigate rather than silently truncating the selection.
