# Report 0037 — gfx1201 FP32 VKQ pilot

**Status: Completed for the bounded diagnostic pilot, with an important dense-control ordering caveat.** Starting commit `840d02efe86c4533d93834bb9e09740e02962677`. The distinct FP32 sparse attention kernel was compiled, fixture-tested and traced on HIP, and the four authorized 32K conditions ran once each. FP32 was closer to the exact-input FP64 attention reference at all 96 query/head positions versus sparse half2, but it changed downstream selection and recurrent states; this is **not** model equivalence or a quality threshold. Sparse and the FP32 variant remain opt-in; production promotion remains **NO-GO**.

## Implementation and validation

- Fixed the host/device preprocessor guard that had excluded the new specialization from HIP host dispatch. The gfx1201 code object contains the distinct `<256,256,4,4,false,false,true,true>` kernel and `v_wmma_f32_16x16x16_f16` instructions. Runtime selection is restricted to audited layer11, sparse GQA12, Q4, RDNA4 and the existing F16/shape/bounds guards. A post-pilot selector check also rejects an unsupported requested layer11 Q4 before generic fallback. Invalid opt-in grammar remains fail-closed; other shapes/layers and flag-unset defaults keep their existing paths.
- Corrected a real feature-order bug: the two FP32 accumulators represent **even and odd** V features, not contiguous halves. The shared store now interleaves them. Register WMMA accumulation and online max rescaling are FP32; softmax weights remain F16, and the partial-result LDS store rounds to F16 before the existing FP32 reduction/normalization. This is not end-to-end FP32 attention, and the altered accumulator plus this existing rounding point must be considered together.
- `test_rdna4_vkq_actual_tiles.hip` exercises the actual transposed LDS loader, two FP32 WMMA fragments and the **same shared-store helper as the kernel**; its 512 outputs passed on each gfx1201 device. `test_rdna4_vkq_attention.hip` directly launches the actual sparse FP32 specialization on both devices with a deterministic two-tile changing-mask fixture; all 24 heads/four queries passed the scalar reference, max abs **2.126e-5**. This is bounded fixture coverage, not proof for all masks, split-K paths or graph modes.
- A separate audited QSA GQA12 backend fixture at KV=32768 passed CPU comparison: FP32 NMSE **6.943e-8**, worst head **8.655e-8**; half2 NMSE **7.007e-7**, worst head **8.614e-7**. ROCm traces show the `true,true` FP32 specialization for the FP32 fixture, `true,false` for half2, and no sparse-Q4 specialization in dense. The actual sparse-index capture was separately checked against all 2048 expected indices. C++ request/shape test and 215 Python QSA tests passed. Mixed CUDA sm_89/HIP gfx1201 Release `llama-server` and `test-backend-ops` builds passed with `-j 28`. The UI's pinned download fell back to `latest`.
- Captured the actual compacted sparse index sequence once per sparse process, opt-in and bounded, after the existing overflow check. It does not alter default execution. Post-pilot, added the early unsupported-shape rejection and rebuilt; the pilot used the preceding build, whose eligible layer11 path is unchanged. Exact pilot binary/model hashes and the post-pilot binary hashes are external under `iteration-0037/`.

## Matched four-condition observations

The original ~32K request and the same target/draft model paths as 0033, mixed 4/7/7 CUDA/HIP mapping, F16 KV, context 65536, MTP max 3, greedy seed 42, reasoning off, no prompt cache and 32-token budget were used in four fresh processes: dense A/B, half2 only11 and FP32 only11. Current model hashes were recorded externally; the 0033 artifacts inspected did not contain model hashes to independently verify historical weight identity. Captures and tracing synchronize execution, so timing is not throughput evidence. All four had the same captured logical candidates/accepted histories; the strict GDN and logit comparators aligned all reported rows. The first layer11 raw Q/K/V spans and F16 masks were byte-identical across all four; the actual compacted sparse index bytes matched between half2 and FP32. Dense A/B first attention output, captured attention outputs, masks, GDN and logits were byte-identical.

**Dense-control caveat:** first layer15 score, block and extra-map arrays were byte-identical, and selected pool/KV **sets** matched for all four queries, but the two raw 512-entry top-k arrays were not byte-identical: 1,962 ordered positions differed across the four queries, with identical sets per query. Thus ordered top-k repeatability is **not established**; the selection comparisons below are set-based, not claims of identical sequence order. No repeat was run.

ROCｍ traces attributed all sparse Q4 launches to layer11: dense A/B **0**, half2 **9 half2 / 0 FP32**, FP32 **0 half2 / 9 FP32**, with zero unassigned sparse launches. Each count includes the initial final-prompt operation and eight natural verification groups. Host selection markers alone were not used as launch proof.

## Exact-input FP64 attention accuracy

For the first HIP0 layer11 operation, dense, sparse half2 and sparse FP32 reference-energy NMSE were **1.18147e-6**, **3.68239e-7** and **3.89329e-8**, respectively. Their worst absolute reference errors were **0.0160212**, **0.00439935** and **0.00277848**. FP32 improved squared error and max-absolute error versus half2 in **24/24 aggregate heads and 96/96 query/head pairs**. The smallest squared-error improvement was q2/head5: FP32/half2 NMSE ratio **0.969**. Dense was closer than FP32 in one pair, q3/head5 (NMSE 3.357e-8 versus 4.584e-8), although FP32 had lower max abs there (0.000351 versus 0.000488). The worst FP32 absolute outlier was q1/head13/feature124. These 96 correlated slices are from one attention operation, not independent trials.

Per-head NMSE below is in units of **1e-7**; max abs is the maximum over that head's four queries and 256 features. Full 96-row numeric results, including all three maxima and errors, are external in `all-96-query-head.csv`.

|Head|Dense NMSE|half2 NMSE|FP32 NMSE|half2 max abs|FP32 max abs|
|---:|---:|---:|---:|---:|---:|
|0|5.376|3.650|0.281|0.004109|0.000465|
|1|4.015|3.615|0.308|0.002248|0.000463|
|2|7.812|4.069|0.344|0.002326|0.000962|
|3|5.212|2.775|0.283|0.001671|0.000471|
|4|11.231|2.717|0.170|0.001936|0.000368|
|5|0.702|0.618|0.518|0.000780|0.000777|
|6|20.501|5.086|0.310|0.002334|0.000801|
|7|4.153|2.345|0.319|0.001520|0.000715|
|8|1.747|1.755|0.373|0.001504|0.000511|
|9|4.511|4.646|0.392|0.003411|0.001082|
|10|5.711|4.728|0.335|0.001639|0.000361|
|11|2.072|1.295|0.219|0.001109|0.000490|
|12|2.226|1.241|0.328|0.001279|0.000804|
|13|21.249|7.387|0.642|0.003455|0.002778|
|14|36.040|3.559|0.466|0.002481|0.001830|
|15|6.755|2.412|0.331|0.001625|0.001129|
|16|10.819|3.755|0.318|0.002258|0.001315|
|17|23.892|7.114|0.461|0.004399|0.001674|
|18|8.450|3.392|0.343|0.002048|0.001307|
|19|5.052|1.471|0.463|0.001933|0.001885|
|20|14.865|2.759|0.222|0.001712|0.000655|
|21|11.353|3.440|0.522|0.002288|0.001357|
|22|16.008|1.179|0.181|0.001052|0.000430|
|23|14.083|6.488|0.600|0.002578|0.002198|

## Downstream effects on aligned histories

- At the first layer15 top-k group, half2 versus dense changed pool and mapped-KV sets in **3/4** queries; FP32 versus dense changed **4/4**. For FP32 query 0, 10 pools were removed and 10 added; dense cutoff margin was 0.002595 and maximum paired score delta 0.275641. Half2 versus FP32 also changed 4/4 query sets. This downstream score shift is not an exact-input FP64 comparison at layer15; subsequent QKV/masks can differ. The layer11 second captured mask differed from dense in both sparse conditions.
- Layer12 GDN: half2 and FP32 each changed **12/12** R/S snapshots versus dense. First observed, step 0 pre-verification **after target decode**, R max abs was 0.021261 and 0.027429 respectively. This does not prove an incorrect rollback or isolate a rounding origin.
- All **34/34** aligned full-vocabulary verification logits changed in each sparse condition; max abs versus dense was 0.566130 half2 and 0.546836 FP32. Top1 changed **0/34** for both; top5 sets changed **3/34** and **4/34**. Half2 versus FP32 changed 34/34 logits, max abs 0.603208, with 0 top1 and 1 top5-set change. Better FP64 attention did not eliminate downstream selection/state/logit drift.

## Device safety and PCIe/CUDA timing

A current-boot AMD gfxhub page/protection-fault burst at 14:12:40 local time, attributed by the kernel to `strata`, initially stopped work. The user subsequently identified it as belonging to another project and authorized continuation. Fresh checks found no later matching serious GPU/PCIe event; no new fault was observed in the four runs. No reset, reboot, driver/PCIe/power change, stress test, automatic retry or GitHub Actions was performed. Model process return code was **-2** after the runner's intentional SIGINT following each successful response; profiler traces were flushed.

The external per-condition timeline records UTC and monotonic calibration pairs, server load/ready, request start, observed prefill progress/end, natural MTP step timestamps, QSA audit/selection and request/stop boundaries, plus bounded NVIDIA snapshots and journal checks. Sampled NVIDIA link was **Gen1 x4 idle**, **Gen3 x4 after load and during prefill** in all four conditions; snapshots cannot measure transient electrical or power behavior. Monotonic ROCm trace launch timestamps were bracketed by QSA monotonic audit intervals for layer attribution. Full telemetry and logs remain external. No stage correlated with a new device issue in this pilot. Previously, the first Xid79/154 pair occurred during fourth-task CUDA prefill with HIP tracing active; the second pair's active application/tracer state is unknown. Neither that history nor this clean bounded run proves or disproves a workload-timing, link, power or driver cause; tracing/sampling can themselves perturb scheduling.

Raw tensors, full responses, traces, detailed per-head/query results, model hashes, telemetry and logs remain under `/home/chris/workspace/llama-perf-results/iteration-0037/`. The strict ordered top-k dense-repeat caveat, graph safety, mixed-sequence behavior, overflow limits and a user-approved model-fidelity rubric remain independent gates.
