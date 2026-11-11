# QSA sparse attention on RDNA4: source audit and experiment

Last reviewed: 2026-11-11. This is an experiment on our fork, not an upstream change or a verified speedup.

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
