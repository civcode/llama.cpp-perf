# QSA sparse GQA12 — bounded task-quality pilot

**Status:** Experiment tooling for Iteration 0024, not a product acceptance criterion. Sparse GQA12 remains experimental and disabled by default.

## Motivation

Report 0023 measured 48 teacher-forced next-token losses across Alice in Wonderland, HTTP Semantics RFC 9110 and local C++ source. Dense A/B were bitwise equal and sparse differences were heterogeneous. All 48 labels were the models' top-ranked choice, so these easy positions provided little evidence about retrieval, code reasoning or instruction-following difficulty. The next question is whether sparse attention changes **actual task success** on matched long-context inputs, not just whether its logits differ.

## Six predefined synthetic tasks

The new standard-library-only tools are:

- docs/LLAMA-PERF/tools/generate_qsa_quality_tasks.py builds six long-context prompts, two per task type: two-hop lookup/numerical aggregation in a fictional ledger, static Python code-value reasoning over fields found in the ledger, and exact-schema JSON instruction following. The answer key is computed independently from the generated data, never from a model output. All prompts and answers are fixed by a named PRNG seed. No generated code is executed.
- docs/LLAMA-PERF/tools/run_qsa_quality_tasks.py sends exact same user/system messages to an already-started LOCAL llama-server using its OpenAI-compatible chat endpoint. Runs separately with condition dense-a, dense-b or sparse, temperature zero, top-p one, seed42, bounded output, disabled prompt cache. It records prompt SHA-256, manifest SHA-256, finish reason, usage token count, raw output and local monotonic request-boundary markers for correlating backend diagnostics. It does NOT start a server, choose its backend, benchmark latency or prove sparse GPU dispatch.
- docs/LLAMA-PERF/tools/score_qsa_quality_tasks.py compares all result files against the exact fixed manifest/decoding settings and scores literal responses or strict JSON (wrong types, duplicate/nonfinite/extra fields fail). It reports per-category correct counts and sparse task regressions/improvements only when dense controls are stable and reported prompt tokenization aligns. Distinct sparse-versus-dense prompt-token count disagreement is classified as a prompt usage mismatch, not dense-repeat instability or a sparse quality regression. Missing usage cannot establish control alignment; matching but below-floor counts are excluded from sparse regression/improvement attribution. A token-floor check flags prompts too short for the default sparse selector. This tool cannot certify actual sparse kernel activation.
- docs/LLAMA-PERF/tools/test_qsa_quality_tasks.py has regression tests for deterministic tasks, correct/incorrect grading, stable controls, truncated responses, malformed JSON and mismatched prompt metadata.

All task manifests, raw inputs, outputs, grader summaries and server logs belong OUTSIDE Git, e.g. /home/chris/workspace/llama-perf-results/iteration-0024/. Tools are not yet validated on the local GPU host.

## Local run outline

First generate the private manifest and run host tests:

    python3 docs/LLAMA-PERF/tools/generate_qsa_quality_tasks.py --out /external/tasks.json --records 900 --seed 20261112
    python3 docs/LLAMA-PERF/tools/test_qsa_quality_tasks.py

Start three fresh, separate bounded llama-server processes, one after another, with the established Qwen3.8-Flash-Next IQ3_XXS target, MTP draft Q8, CUDA0/HIP0/HIP1 4/7/7, F16 K/V, context65536 or larger, max draft3, seed42 and identical graph-disabled QSA audit settings. The sparse process alone sets GGML_HIP_QSA_SPARSE_Q4_GQA12=1. Keep legacy GGML_HIP_QSA_SPARSE_Q4 and GGML_HIP_FATTN_FORCE_WMMA_Q4 unset. Choose a local server port and use it as shown:

    python3 docs/LLAMA-PERF/tools/run_qsa_quality_tasks.py /external/tasks.json --condition dense-a --url http://127.0.0.1:8080 --out /external/dense-a.jsonl
    python3 docs/LLAMA-PERF/tools/run_qsa_quality_tasks.py /external/tasks.json --condition dense-b --url http://127.0.0.1:8080 --out /external/dense-b.jsonl
    python3 docs/LLAMA-PERF/tools/run_qsa_quality_tasks.py /external/tasks.json --condition sparse --url http://127.0.0.1:8080 --out /external/sparse.jsonl
    python3 docs/LLAMA-PERF/tools/score_qsa_quality_tasks.py /external/tasks.json /external/dense-a.jsonl /external/dense-b.jsonl /external/sparse.jsonl --out /external/quality-summary.json --require-stable-controls

The runner refuses to overwrite an output file, so each condition's output file must not exist. Clean up every server process between conditions and use bounded start/request deadlines. Verify actual Q4 MTP verification under the sparse selector: mask audit/eligible KV and, for a representative request, real HIP sparse kernel launch evidence. Prompt-token count over 16408 is only one necessary precondition; the flag alone cannot prove experimental sparse execution.

If a task has too few prompt tokens or never reaches a Q4 sparse attention operation, exclude it from sparse quality claims rather than treating a dense fallback as a positive result. If dense A/B outputs differ, mark the task as control-unstable rather than assigning the difference to sparse. Explain answer truncation and strict-format failures. Do not cherry-pick tasks after examining model answers.

## Limitations and decision

These fictional ledger tasks are **predeclared, reproducible functional probes**, not a certified training-held-out sample, natural code-generation benchmark, downstream deployment study or statistically powered quality evaluation. A task may fail because of general model capabilities rather than sparse numerical fidelity; dense controls distinguish these cases. Run no generated code. A six-task pilot can inform the size and composition of a later evaluation, but cannot establish a production-quality degradation tolerance.

Numerical equivalence remains NO-GO. GPU graph replay, mixed production sequence correctness and graph-safe overflow are separate unpassed promotion gates. Do NOT run the previously deferred long 512-token full-model throughput A/B, 98K replay or default-enable sparse. Any consequential quality threshold needs explicit user approval after reviewing the results.
