# Report 0014 — GQA12 sparse QSA validation

**Status: Partial.** Synthetic safety, per-head correctness and device-local performance gates passed on both R9700s. A short three-GPU model probe confirmed actual sparse dispatch and matched sampled output and acceptance history, but its intermediate QSA mask unions differed from a reproducible dense baseline. Fixed-token logits and recurrent GDN state across rollback have not been compared. **Do not treat the model path as numerically validated or run performance A/B yet.** No production defaults or selector guards were changed.

## Build and test fixes

Started at `d6c07d3f345dae3a38981c7e89c7cb96af8de8d5`. Mixed CUDA 12.0.140 and HIP 7.16.26385/AMD clang 24 build passed with `cmake --build build-perf -j 28 --target test-backend-ops llama-server llama-bench`; host GCC 13.3.0. Both GQA4 and GQA12 strict `-std=c++17 -Wall -Wextra -Werror -pedantic` selector tests passed. The initial mixed build failed because the new GQA-bound fixture omitted the `logit_softcap` constructor argument; fixed it to zero. The new GQA12 test flags also did not enable the existing explicit finite/NMSE reporting gate: fixed that, and added a separate NMSE check for each of the 24 heads so a bad tile cannot hide in a whole-output average.

The independent selector headers still require GQA4 and GQA12 respectively. On both GPUs, the original GQA4 fixture passed with its original flag (NMSE 2.71365931e-7); enabling only the GQA12 flag left that fixture on the dense path (NMSE 7.68722127e-7). The older WMMA override was unset. No GitHub Actions were run.

## CPU-reference accuracy, dispatch and overflow

All 16 combinations of ROCm0/ROCm1, KV 32768/98304, narrow/wide mask and GQA12 flag unset/1 passed one selected CPU-reference fixture with finite outputs and finite NMSE <= 5e-4. Narrow fixture union is 2048 with about 1890–1891 visible positions per query; wide union is 5120 with 2048 per query. Bounds are 2051/query and 8204/union. Results were identical across the GPUs:

| KV | Union | Dense NMSE / worst head NMSE | Sparse NMSE / worst head NMSE |
| --- | ---: | --- | --- |
| 32768 | 2048 | 2.08812818e-6 / 2.61123515e-6 | 7.00680932e-7 / 8.61350209e-7 |
| 32768 | 5120 | 2.25236020e-6 / 2.40674334e-6 | 6.88799233e-7 / 7.48263565e-7 |
| 98304 | 2048 | 2.64048358e-6 / 2.86683589e-6 | 5.85612348e-7 / 6.28710419e-7 |
| 98304 | 5120 | 2.22717182e-6 / 2.45810544e-6 | 6.66807941e-7 / 7.58754816e-7 |

Separate bounded ROCprofiler correctness traces for the wide fixture on each device at each depth showed dense `flash_attn_tile` plus combine with the flag unset; with GQA12 flag=1 they showed the Q4 mask compactor, `flash_attn_ext_f16<256,256,4,4,...,true>` and `flash_attn_stream_k_fixup_uniform<256,4,4>`. Gather work-item grid was `(7104,2,1)`, workgroup `(32,2,1)`; fixup work-item grid `(1536,4,4)`, workgroup `(256,1,1)`. The latter corresponds to six 256-thread destination tiles, four query positions and four heads/tile. Source maps these six tiles to three GQA-head tiles for each of two KV heads; the independent 24-head CPU checks cover all of them, including per-query mask differences. This establishes the tested shape, not arbitrary GQA/sequence layouts.

For GQA12 boundary fixtures on both ROCm devices, union 8191 and 8192 passed the CPU-reference gate in dense and sparse modes. An intentionally invalid 8193 union passed dense but sparse terminated with exit 134 and `QSA sparse index overflow: group 0 union 8193 exceeds capacity 8192 (KV=32768)`, before gather. Core dumps were disabled. The Q4 check reads all unclamped compactor counts and synchronizes the same HIP stream before launching gather; capture is disabled for the experiment and unexpected capture is rejected. This is fail-closed, not a dense fallback.

## Unprofiled whole-operation performance

Three alternating separate-process dense/sparse pairs per condition; microseconds/run, median (range), including compaction, device-to-host count copy, host synchronization and gather/fixup for sparse. Dense flag was fully unset; original sparse and old WMMA flags were unset throughout. The sparse experiment disables HIP graph capture; the default dense graph policy was not changed. Perf fixtures were separately audited to confirm actual 2048/5120 unions. These are synthetic results, not end-to-end model timings.

| GPU | KV | Union | Dense | Sparse |
| --- | ---: | ---: | ---: | ---: |
| ROCm0 | 32768 | 2048 | 316.98 (316.87–317.01) | 136.70 (136.64–136.74) |
| ROCm0 | 32768 | 5120 | 316.85 (315.34–317.39) | 136.34 (136.25–136.80) |
| ROCm0 | 98304 | 2048 | 960.58 (959.15–961.99) | 186.99 (186.90–187.38) |
| ROCm0 | 98304 | 5120 | 963.26 (962.98–963.75) | 186.08 (185.89–186.97) |
| ROCm1 | 32768 | 2048 | 316.63 (316.07–317.07) | 133.90 (133.84–134.09) |
| ROCm1 | 32768 | 5120 | 317.63 (317.49–317.87) | 133.89 (133.82–133.93) |
| ROCm1 | 98304 | 2048 | 965.31 (965.06–967.12) | 185.05 (184.79–185.10) |
| ROCm1 | 98304 | 5120 | 969.62 (968.77–970.37) | 181.35 (181.34–181.57) |

Individual perf processes reported about 1034–1045 dense and 5346–5522 sparse timed operations at 98K, and 3168–3200 dense and 7328–7488 sparse at 32K. There was no comparable large outlier in these samples; this does not explain Report 0012's isolated outlier. The almost union-independent timing should not be generalized to arbitrary masks. Profiled kernel durations are perturbed and are not added to or substituted for these unprofiled totals.

## Short real-model probe and blocker

After operator gates passed, ran fresh, bounded 32K 64-output-token diagnostic processes on Qwen3.8-Flash-Next IQ3_XXS+MTP Q8, CUDA0/ROCm0/ROCm1 4/7/7, F16 KV, context 196608, max draft 3, seed 42. Consumer-mask audit was enabled in both dense and sparse probes, so GPU graph replay was disabled in both; no timing conclusion is valid. Both completed with identical message SHA256 `9c19ec72268a0f3ff8de0671d6eb95a6ed61ef996cb1a22a854443101734d224`, 42/62 accepted/drafted and identical 21 verification-commit outcome sequences, including rejection and acceptance. Each audited 273 Q4 masks (105 per HIP device and 63 CUDA), no NaNs or +infinities, max union 5493 < 8204 and no overflow. A separate short ROCprofiler run observed **105 compactor, 105 sparse gather and 105 fixup launches on each AMD agent**, confirming activation on both actual GPUs; profiling was not used for timing.

However, comparing matched audit rows showed **181/273 four-query union cardinalities differed** between the dense and sparse probes, with maximum absolute difference 52; per-query counts, tensor IDs, device order and KV lengths matched. A second independent dense process reproduced every dense union cardinality exactly. This strongly motivates investigating numerical changes propagating into later QSA selection; cardinality changes alone do not identify the exact mask elements or prove a specific error. Matching sampled output and acceptance counts is not fixed-token logits or recurrent-state equivalence. The current diagnostic does not capture all token IDs, every mask bit, or GDN state through rollback. Consequently the model correctness gate is **not met** and the three-condition 512-token model performance A/B was not attempted.

Next: implement a controlled teacher-forced identical-token/rollback harness that compares logits and serialized or directly observed GDN states, plus mask indices at the first divergence; establish tolerances and distinguish expected floating-point/top-k sensitivity from a kernel or state error. Also retain mixed-sequence and GPU-graph-replay checks as separate promotion blockers. Do not promote either sparse flag by default.

Raw logs, timed runs, ROCprofiler CSV, traces, responses and bounded runner are under `/home/chris/workspace/llama-perf-results/iteration-0014/`. The runner checked liveness and port ownership and used process-group cleanup; no server or profiler was left running.
