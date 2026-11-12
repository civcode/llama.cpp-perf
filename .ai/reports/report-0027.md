# Report 0027 — LongBench-v2 real-document pilot

**Completed bounded pilot; numerical correctness and production promotion remain NO-GO.** Dense A/B stably solved 8/12 questions, so sparse ran on the entire frozen sample. Sparse also scored 8/12, with no changed answers, regressions or improvements. This is descriptive evidence on 12 public items, not an acceptance margin or numerical equivalence.

## Provenance and pre-inference validation

Starting commit `2ee2b9dd362fa9d7b7b7acab902508ea0c35ffa8`; ending commit is the commit containing this report, identified by the pushed completion message. The official [zai-org/LongBench-v2 dataset card](https://huggingface.co/datasets/zai-org/LongBench-v2/blob/12bc220a4e257d069e9005f2df762cac4755dc54/README.md) declares Apache-2.0 and 503 items; underlying source-content rights may vary. Downloaded its public [data.json](https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/12bc220a4e257d069e9005f2df762cac4755dc54/data.json), revision `12bc220a4e257d069e9005f2df762cac4755dc54`, 465,490,535 bytes, SHA-256 `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`, matching the publisher's LFS object hash. No dataset text was committed.

Before any inference, froze seed 20261115, limit 12, source difficulty easy, 56,000 minimum context bytes and 210,000 maximum prompt bytes; 53 candidates met the byte filters, across six domains. Manifest SHA-256 `d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1`. No re-selection, padding, truncation or label edits occurred. An independent parser checked all 503 unique IDs and rebuilt each selected complete prompt from its original document, question and four options, verified answer and row hash, and tokenized all selected user prompts with the target Qwen tokenizer. Before inference, a separate server applied the actual chat template and tokenized each full system/user input: counts appear below, all fit 65,536 including 32 output tokens and exceed 16,408. These predicted counts exactly matched all three actual server usage counts. Full source contexts were 60,188–197,759 bytes. All source length buckets were `short` and all difficulty labels `easy` (dataset classifications, not claims of model solvability).

Fixed the runner before inference to use a benchmark-specific system message referring to supplied **documents**, rather than falsely calling public benchmark material fictional and limiting the requested output to a value or JSON. The synthetic-task system message remains unchanged; all three benchmark conditions used the same revised message and template. Added a regression test. Compileall, 11 direct LongBench adapter tests, and all 127 `test_*qsa*.py` tests passed. Mixed CUDA/HIP Release `build-perf` `llama-server` and `llama-tokenize` built successfully. Adapter SHA-256 `b91d4ba95ebdfa4a1b80854a840273d32d0a88c9c2f13a32c932a3ac19f1fc95`; runner `531c7f49f48b0d7098e0526ea3a4eda2843c379e842d005c1786eb67b355d9c4`; scorer `8a39f0cb729d01d5f452f237832d2ec984d494cb32a7f52e60151a8b931ec5e0`.

## Dense-first and sparse results

Fresh sequential servers used the prior Qwen3.8-Flash-Next IQ3_XXS two-shard target plus Q8 MTP draft, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, context 65536, max draft 3, reasoning off, temperature 0, top-p 1, seed 42, disabled prompt caching and graph-disabled mask audit. Legacy GQA4 and forced-WMMA flags stayed unset. Dense-only `calibrate_qsa_dense_tasks.py --require-solvable` found **8 stable, correctly solved** cases, 12/12 stable dense repeats and 12/12 eligible by prompt usage. Only then did a fresh sparse server run all cases, adding `GGML_HIP_QSA_SPARSE_Q4_GQA12=1` alone.

All completions used the requested unambiguous answer form and stopped normally: none failed because of strict formatting. Dense A, dense B and sparse text, finish reasons and prompt counts were identical for every question. There were **0 changed-but-both-correct, 0 changed-but-both-wrong, 0 genuine regressions, 0 improvements, 0 unstable or incomparable cases**. The per-domain easy accuracy (dense A / dense B / sparse) was:

| Domain | A / B / sparse |
|---|---|
| Multi-Document QA | 3/3 / 3/3 / 3/3 |
| Long In-context Learning | 1/2 / 1/2 / 1/2 |
| Long Structured Data Understanding | 0/1 / 0/1 / 0/1 (floor-limited) |
| Single-Document QA | 1/2 / 1/2 / 1/2 |
| Code Repository Understanding | 2/2 / 2/2 / 2/2 |
| Long-dialogue History Understanding | 1/2 / 1/2 / 1/2 |

The following source ID, source-context byte count, chat-token count and actual natural verification evidence cover the frozen sample. Pass/fail compares all three outputs to the original source answer; no model answers or dataset text are reproduced.

| Dataset ID | Domain abbreviation | Context bytes | Chat tokens | A/B/sparse | Q4 groups; HIP agent 1 / agent 2 launches |
|---|---|---:|---:|---|---|
| 66f95126bb02136c067c5070 | Multi-doc | 103766 | 29328 | pass/pass/pass | 2; 10/10 |
| 66f3edfb821e116aacb2fc91 | In-context | 184132 | 46016 | fail/fail/fail | 2; 10/10 |
| 66ed2c87821e116aacb1f149 | Structured | 60188 | 19556 | fail/fail/fail | 2; 10/10 |
| 66ec2e2a821e116aacb1bb9f | Single-doc | 90144 | 22859 | fail/fail/fail | 2; 10/10 |
| 66f530ce821e116aacb32f09 | Code | 194824 | 50857 | pass/pass/pass | 2; 10/10 |
| 671b08c8bb02136c067d4e19 | Dialogue | 92333 | 29045 | pass/pass/pass | 2; 10/10 |
| 66f0ed5a821e116aacb265ab | Multi-doc | 106290 | 27291 | pass/pass/pass | 2; 10/10 |
| 66ec3a46821e116aacb1c49f | In-context | 99017 | 26329 | pass/pass/pass | 2; 10/10 |
| 66ebc0c95a08c7b9b35de7f3 | Single-doc | 88435 | 29759 | pass/pass/pass | 2; 10/10 |
| 6708a096bb02136c067d1789 | Code | 108845 | 26889 | pass/pass/pass | 3; 15/15 |
| 671b2c05bb02136c067d50c8 | Dialogue | 111001 | 45161 | fail/fail/fail | 2; 10/10 |
| 67078ecabb02136c067d0b89 | Multi-doc | 197759 | 47805 | pass/pass/pass | 2; 10/10 |

ROCm trace observed 370 actual HIP `flash_attn_ext_f16<256,256,4,4,false,false,true>` sparse launches in total, **250 inside 25** natural MTP `target_process` verification spans with batch=4 and draft=3, correlated to request boundaries using monotonic timestamps. Each span contained exactly five launches per HIP agent; all 12 tasks therefore qualify for attribution. Mask audit confirmed Q=4, 24 Q heads / 2 KV heads and bound 2051. Profiling is not a timing benchmark.

Public benchmark contamination is unknown; the GGUF upstream training revision is unknown. This tiny nonrepresentative sample with one decoding configuration does not establish accuracy parity, numerical equivalence, graph replay safety, mixed production sequence correctness or graph-safe overflow. Ask the user to choose deployment workloads and an acceptable quality-loss bound before a powered evaluation. No sparse default, throughput A/B, 98K replay, graph-enabled sparse test, GitHub Actions, GPU reset or driver change occurred. Processes were cleaned up. Raw dataset, prompts, gold, model outputs, manifests, scripts, logs and profiler traces remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0027/`.
