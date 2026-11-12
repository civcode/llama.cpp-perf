# Report 0024 — six-task natural-MTP quality pilot

**Status: Completed for the predefined bounded pilot; task-quality discrimination is blocked by a floor effect.** Started at `65bf85ef0ea615734d8b9677224151eef4109dda`. All six tasks failed under both dense controls and sparse. This is not evidence of sparse quality equivalence or a production acceptance threshold; sparse numerical equivalence and promotion remain **NO-GO**.

## Tool validation and fixes

The mixed CUDA/HIP Release `llama-server` and `llama-tokenize` targets built. Python compileall passed; the direct task suite passed **17 tests**, and all **94** `test_*qsa*.py` tests passed. A loopback mock HTTP server checked the OpenAI-style POST payload/response, nonstreaming deterministic parameters, output records and refusal to overwrite. Actual llama-server responses also provided the expected content, finish reason and `usage.prompt_tokens` fields. All 18 real responses ended with `stop`; none was truncated.

Added client monotonic request-boundary markers solely to correlate GPU audit/kernel events with individual tasks. Fixed the evaluator so missing prompt-token usage cannot establish dense alignment, and even matching counts below the configured sparse floor cannot be attributed as a sparse regression/improvement. Added negative regressions. The scorer still cannot prove kernel dispatch; that was checked independently below. Literal grading strips surrounding whitespace; JSON grading requires exact keys, values and types, rejects duplicate keys/nonfinite constants, and does not execute model output.

Gold values were fixed before any response. A separate ledger parser reconstructed all six stored gold answers from the generated records and predefined arithmetic without executing generated code; the same reconstruction is now a host test. The manifest uses seed `20261112`, 900 distinct records per task and SHA-256 `779fc151b23b544a989a86c19c4a3fe76f0f7a1a125303b1551f2899c787d67a`. Before inference, the target tokenizer counted 46,622–46,821 tokens in the six user prompts, safely beyond the sparse guard. Including chat formatting/system text, actual server counts are below.

## Setup and direct sparse-kernel evidence

Three fresh, sequential, bounded server processes used the existing two-shard Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS target and Q8 MTP draft, CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, context 65536, max draft 3, localhost-only access and identical graph-disabled audit settings. All used greedy temperature 0, top-p 1, seed 42, prompt caching off and `--reasoning off` to request the specified short final answers rather than spend the limited output budget on hidden thinking. Only sparse set `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`; legacy GQA4 and forced-WMMA flags stayed unset. The client did not alter task prompts or gold between conditions. Processes were cleaned up; no full-model throughput comparison was run.

The sparse process ran under the previously used ROCm kernel tracer. Its trace records **430 actual** `flash_attn_ext_f16<256,256,4,4,false,false,true>` launches across the two HIP devices; the final template argument selects the sparse kernel. The server's monotonic target-process trace identifies 37 natural MTP verification batches with `batch_tokens=4`, `draft_tokens=3`; **370** of the sparse launches fall inside these verification spans, ten per batch, five per HIP device. Request boundaries and per-task verify-span counts below establish that this was not just an environment flag or a prompt-prefill Q4 operation. HIP mask audit independently records Q=4, 24/2 heads, bound 2051, and padded KV 46848 or 47104. All satisfy the existing selector eligibility; no overflow was reported. Trace instrumentation changes execution conditions and is not a timing benchmark.

| Task | Prompt tokens, all conditions | Q4 MTP verify groups / actual sparse launches within them | Dense A / dense B / sparse | Sparse text changed? |
|---|---:|---:|---|---|
| retrieval-1 | 46672 | 3 / 30 | fail / fail / fail | no |
| retrieval-2 | 46791 | 2 / 20 | fail / fail / fail | no |
| code_reasoning-1 | 46856 | 2 / 20 | fail / fail / fail | no |
| code_reasoning-2 | 46871 | 2 / 20 | fail / fail / fail | no |
| instruction_following-1 | 46867 | 8 / 80 | fail / fail / fail | no |
| instruction_following-2 | 46814 | 20 / 200 | fail / fail / fail | yes |

Each verification count includes actual launches on **both** HIP devices. All dense A/B outputs, finish reasons and per-task prompt counts matched exactly, and sparse prompt counts matched the controls. Both retrieval tasks and both code tasks gave incorrect literal answers in every condition. Both instruction-following tasks produced JSON with incorrect values in every condition. On instruction_following-2 sparse changed the generated JSON's `ids` and `total`, but both dense and sparse were wrong; this is an observed answer difference, **not** a task-correctness regression or improvement. Therefore retrieval, code and instruction categories each scored 0/2 in all conditions, with **zero genuine sparse regressions and zero improvements** among these six stable, activated cases.

All raw prompts, gold answers, generated answers, manifests, server logs, kernel traces and the bounded local orchestration script remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0024/`.

## Interpretation

This is a floor-limited synthetic probe: common model failures dominate, and strict exact-answer scoring cannot determine whether the numerical differences matter on tasks the model can otherwise solve. It does not establish statistical equivalence, realistic deployment quality, held-out status, graph safety, mixed-sequence safety, or graph-safe overflow. Do not change the predefined gold or cherry-pick results retrospectively. A future separately predeclared calibration should use workload-representative tasks with a range of dense difficulty and user-selected success metrics; any consequential acceptance margin requires explicit user approval. No GitHub Actions, 98K replay, default promotion, graph-enabled sparse run, driver change or GPU reset occurred.
