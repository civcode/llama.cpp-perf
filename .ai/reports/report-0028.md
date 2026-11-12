# Report 0028 — LongBench expansion interrupted by GPU failure

**Status: Partial.** Dense A/B completed the frozen 24-question non-overlapping sample and stably solved 14/24. The sparse run completed its first 3/24 questions before the NVIDIA GPU fell off the bus; it could not complete the unchanged sample. No full-sample sparse accuracy, regression count or Iteration 0028 evidence bound can be claimed. Numerical correctness and production promotion remain **NO-GO**; sparse remains disabled by default.

## Validation and provenance

Starting commit `04a0eddfa472a42bb6f54467f85064732d360c6c`; ending commit is the commit containing this report, identified in the completion message. Compileall passed, as did 14 adapter tests, 11 evidence-analyzer tests and all 141 `test_*qsa*.py` tests. Mixed CUDA/HIP Release `build-perf` targets `llama-server` and `llama-tokenize` built. Corrected the adapter docstring's outdated 12-item limit to 36; no runtime logic or numerical code changed.

Reused official [zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) revision `12bc220a4e257d069e9005f2df762cac4755dc54`; dataset SHA-256 `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`, prior 0027 manifest SHA-256 `d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1`. Dataset card states Apache-2.0; underlying source-content rights may vary. Independently checked all 503 source IDs, full document/question/four-choice mapping, gold and source-row hashes for every selected task. The 24 distinct selected IDs and the previous 12 IDs are disjoint; the manifest explicitly records all 12 exclusions.

The initial requested 36-item seed-20261116 selection, byte range 56,000–210,000, had 41 eligible candidates but failed pre-inference target-tokenizer checks: several user prompts were below 16,408 tokens and two exceeded 65,536 even before formatting. Before any model answer, rejected that entire manifest and changed the fixed filter to context >=66,000 bytes and prompt <=138,000 bytes, limit 24. This yielded exactly 24 candidates in four domains; none from the code-repository or structured-data domains remained eligible. The resulting frozen manifest SHA-256 is `ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5`. All are source-easy, source-length-short. Every complete system/user chat template was tokenized by the actual target server before inference: 17,051–45,158 tokens including template, plus 32 output tokens fits 65,536, and exact counts matched both dense runs. No document was truncated, no labels changed, and no sample was selected based on answers.

## Dense-first and incomplete sparse execution

Fresh bounded sequential dense processes used the established Qwen3.8-Flash-Next two-shard IQ3_XXS target and Q8 MTP draft, CUDA0/ROCm0/ROCm1 4/7/7, F16 KV, context 65536, max draft 3, reasoning off, greedy seed 42, prompt cache off, graph-disabled mask audit, and identical 0027 benchmark system/user template. Dense A/B were text-, finish- and prompt-count-identical on **24/24**; all stopped normally. Dense-only `--require-solvable` reported **14/24 stable baseline successes**, so sparse was started on the entire unchanged manifest with only `GGML_HIP_QSA_SPARSE_Q4_GQA12=1` added. All other sparse/forced-WMMA flags stayed unset.

| Domain | Dense A/B correct | Sparse completed |
|---|---:|---:|
| Long-dialogue History Understanding | 2/6 | 1/6 |
| Long In-context Learning | 1/1 | 1/1 |
| Single-Document QA | 9/13 | 1/13 |
| Multi-Document QA | 2/4 | 0/4 |

Sparse's first three outputs matched dense text, finish and prompt usage; two were correct and one wrong in all conditions. No sparse changes, regressions or improvements were observed **among those three only**; this interrupted prefix is not a valid full-sample comparison. ROCm recorded 90 actual HIP `flash_attn_ext_f16<256,256,4,4,false,false,true>` sparse launches, including **60 in six** natural MTP target-process spans with batch=4/draft=3, attributed by monotonic request boundaries. Each of the three completed requests had two groups and 10 launches on each HIP agent. All other 21 tasks lack completed sparse answers and are not attributed.

| Source ID | Domain abbreviation | Chat tokens | Dense correct | Sparse evidence |
|---|---|---:|---|---|
| 671b3d9fbb02136c067d52b2 | Dialogue | 33812 | no | 2 groups; agents 10/10 |
| 66f68b33bb02136c067c2303 | In-context | 30772 | yes | 2 groups; agents 10/10 |
| 66ec3d1d821e116aacb1c622 | Single-doc | 19214 | yes | 2 groups; agents 10/10 |
| 66fc0152bb02136c067c8826 | Multi-doc | 28563 | no | incomplete |
| 671b3e0cbb02136c067d52e5 | Dialogue | 33708 | yes | not run |
| 66f3918f821e116aacb2d8b7 | Single-doc | 18443 | yes | not run |
| 66f96115bb02136c067c541c | Multi-doc | 22870 | no | not run |
| 671b3d1bbb02136c067d5283 | Dialogue | 33875 | no | not run |
| 66f58d6c821e116aacb33e76 | Single-doc | 17877 | no | not run |
| 67062fbfbb02136c067cf66a | Multi-doc | 21371 | yes | not run |
| 671b17d5bb02136c067d4f7d | Dialogue | 45158 | no | not run |
| 6703fc69bb02136c067cd7c2 | Single-doc | 20737 | yes | not run |
| 66fcf2f2bb02136c067c9169 | Multi-doc | 22957 | yes | not run |
| 671b1480bb02136c067d4f14 | Dialogue | 29726 | no | not run |
| 66ebc95f5a08c7b9b35df497 | Single-doc | 30587 | no | not run |
| 671b1335bb02136c067d4e88 | Dialogue | 29145 | yes | not run |
| 66ebd5675a08c7b9b35e06bf | Single-doc | 22728 | no | not run |
| 66f55729821e116aacb3358b | Single-doc | 19915 | yes | not run |
| 66f67c18bb02136c067c21b8 | Single-doc | 24746 | yes | not run |
| 66ec0f3c821e116aacb19d70 | Single-doc | 27082 | yes | not run |
| 66ebd5125a08c7b9b35e0616 | Single-doc | 20511 | no | not run |
| 66eefa2f821e116aacb2284f | Single-doc | 27161 | yes | not run |
| 66ec41d3821e116aacb1c874 | Single-doc | 19348 | yes | not run |
| 66ebd49a5a08c7b9b35e0550 | Single-doc | 17051 | yes | not run |

During prompt processing for the fourth request the server logged CUDA `unspecified launch failure` at `cudaStreamSynchronize`. System log at the same time reports NVIDIA Xid 79, GPU fallen off the bus, and Xid 154, **node reboot required**. `nvidia-smi -L` subsequently reports an unknown error and cannot obtain the GPU handle. The failure's root cause is not established and cannot be attributed to the HIP sparse kernel. Kernel `dmesg` access is denied without privilege. The server and profiler process groups were cleaned up. No reset, reboot, driver change or privileged operation was attempted, as prohibited. After authorized host recovery, rerun the **entire frozen sparse sample** in a fresh bounded server with per-task trace; preserve the failed attempt rather than merging a selective prefix. Recheck the unchanged manifest and dense controls, then run the full scorer and evidence analyzer.

## Evidence analyzer and limitations

The new evidence analyzer's 11 host tests passed. As a real-input integration check only, it processed the **previous 0027 full summary** and reproduced 8 stable baseline successes and 0 regressions. Its illustrative iid one-sided 95% exact binomial upper value was 0.31234 for 0/8; this is **not an Iteration 0028 result**, population confidence bound, approved degradation tolerance or release criterion. IID/representative sampling is not established for this small deterministic domain-balanced public sample, and pretraining contamination is unknown. No 0028 full-sample evidence summary exists because sparse did not complete; the analyzer itself cannot prove kernel dispatch.

Raw data, manifests, model outputs, scripts, logs and trace remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0028/` (source remains in iteration-0027). No full-model throughput A/B, 98K replay, graph-enabled sparse test, default promotion or GitHub Actions was performed. Existing logit/recurrent-state drift, graph replay, mixed-sequence and graph-safe overflow remain unresolved. After recovery and complete measurement, request the user's actual deployment workloads, acceptable quality-loss fraction and a decision on a powered representative evaluation; no margin has been authorized.
