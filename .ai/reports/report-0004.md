# Report 0004

Status: Partial. The HIP grouping switch works and produces repeatable end-to-end generation throughput gains for these two sampled outputs, but the gains coincide with different sampled text and higher MTP acceptance, not lower time per verification. Long-context numerical equivalence, recurrent-state integrity and the cross-device critical path remain unproven. Do not promote the option as a default optimization. Source revision: `7d52d49188aa7b80cce538c06b7f036eb11adb21`.

## Build and correctness checks

`g++ -std=c++17 -O2 -Iggml/src/ggml-cuda docs/LLAMA-PERF/tools/test_fattn_gqa_tune.cpp -o /tmp/test_fattn_gqa_tune && /tmp/test_fattn_gqa_tune` passed. `cmake --build build-perf -j 28 --target llama-server llama-bench` passed with CUDA and HIP; `test-backend-ops` also built with `-j 28`. On ROCm0, baseline and `GGML_HIP_FATTN_TILE_MAX_GQA=2` each passed two CPU-reference `FLASH_ATTN_EXT` cases: 256x256 head sizes, GQA 4, KV 512, batch 3, F16 K/V, masked, with two tensor permutations. This is not a numerical test at 32K or 98K.

Each arm used a separate process. Short uncached 64-token MTP smoke requests produced identical text, 46/47 accepted/drafted, and length finish in both arms. Long requests returned nonempty, plausible reasoning and a length finish, without reported inference errors or NaNs; all exhausted their token cap before producing a final answer. Repeated runs within each arm generated identical messages and acceptance counts, but the two arms diverged in long-request text and acceptance. No direct long-context logits/NaN comparison or independent GDN state validation was performed, so plausible output is not proof of numerical or state correctness.

## Matched unprofiled MTP A/B

Server command was `build-perf/bin/llama-server --model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --spec-draft-model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-Q8_0.gguf --alias qwen3.8-flash-next-gsq-rco-iq3_xxs-192k --host 127.0.0.1 --port 8081 --device CUDA0,ROCm0,ROCm1 --spec-draft-device CUDA0 --parallel 1 --jinja --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --reasoning-format auto --reasoning auto --reasoning-budget -1 --spec-type draft-mtp --spec-draft-n-max 3 --flash-attn on --split-mode layer --tensor-split 4,7,7 --load-mode mmap --lazy-mode on --fit-target 768 --ctx-size 196608 --cache-type-k f16 --cache-type-v f16 --n-predict 49152 --perf --metrics`. Each arm started clean; the saved 32K and 98K requests used seed 42, `cache_prompt=false`, and 512 output tokens. Arm order was baseline, alternative, alternative, baseline, with a short smoke followed by 32K then 98K in each process. Each long request reported zero cached tokens.

| Prompt tokens | Group | Generation t/s, two runs | Steps | Accepted / drafted | Generation ms / step, two runs |
| ---: | --- | --- | ---: | ---: | --- |
| 32809 | original | 58.87, 59.09 | 204 | 306 / 611 | 42.55, 42.39 |
| 32809 | max 2 | 68.02, 68.36 | 176 | 334 / 528 | 42.69, 42.48 |
| 98275 | original | 40.78, 40.80 | 203 | 307 / 609 | 61.72, 61.70 |
| 98275 | max 2 | 43.23, 43.17 | 190 | 321 / 570 | 62.22, 62.31 |

Steps are deltas of `llamacpp:spec_decode_num_drafts_total`; generation time per step is not GPU kernel latency. The approximate mean generation throughput gains are 15.6% and 5.9%, respectively, but steps fell by 28 and 13 while ms/step stayed flat or increased. This does not establish faster attention or a generalizable MTP gain. Prompt evaluation took 15.30-15.70 s original vs 15.56-15.63 s alternative at 32K, and 58.78-59.89 s vs 59.70-59.71 s at 98K; generation speed is not whole-request speed.

Verbose startup checks in separate processes with the same options showed identical target model buffer sizes: CUDA0 9024.29 MiB, ROCm0 16207.78 MiB, ROCm1 19357.65 MiB, plus CPU-mapped 260.50 and 27465.95 MiB; 49/49 target layers were reported offloaded. Draft model CUDA0 buffer was 3318.05 MiB, with 644.14 MiB CPU-mapped. These checks were separate from the timed processes, not per-tensor placement proofs. `--fit-target 768` was requested but the server reported that fitting aborted because the tensor split was already set; do not interpret 768 as an effective fit guarantee.

## HIP traces and decision

ROCprofiler delayed kernel and copy traces were collected separately for each arm and depth. Both Radeons changed from `flash_attn_tile<256, 256, 4, 4, false>` to `flash_attn_tile<256, 256, 4, 2, false>`, confirming the experiment changes kernel dispatch across process restarts. Approximate first-to-last 3,973,120-byte readback windows gave:

| Depth | Group | Observed copies | Summed flash-attention ms / copy | Radeon 1 / 2 busy-union % |
| --- | --- | ---: | ---: | --- |
| 32K | original | 203 | 3.585 | 23.16 / 24.67 |
| 32K | max 2 | 176 | 3.829 | 23.19 / 24.66 |
| 98K | original | 204 | 10.366 | 22.27 / 23.28 |
| 98K | max 2 | 189 | 11.187 | 22.45 / 23.29 |

The copy-window heuristic is approximate; observed copies are not independently proven verification counts. GPU kernel durations sum across devices and can overlap. Trace overhead reduced generation throughput to 50.63/58.29 t/s at 32K and 36.86/39.04 t/s at 98K, original/alternative. There is no NVIDIA timeline or measured host synchronization attribution in these traces. The increased flash-attention work per observed copy and unchanged or slower unprofiled ms/step argue against claiming a kernel speedup; changed acceptance explains the observed direction better, but is not a proof of all causality.

Keep the existing disabled-by-default switch only as an experiment; no production default or kernel arithmetic change is justified. Next inspect `ggml/src/ggml-cuda/fattn-tile.cuh` sequence-tile work distribution and masking for these 256x256 long-KV shapes, with fixed token trajectories or direct logits/state checks and a cross-device dependency timeline before proposing another change.

Raw responses, logs, scripts and ROCprofiler databases remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0004/`; saved inputs are in `/home/chris/workspace/llama-perf-results/20261111-143704/server/`. The original checkout, models and `build-mixed` were not modified. No GitHub Actions were run.
