# Report 0023 — multi-source fixed-reference pilot

**Status: Completed for the bounded descriptive pilot. Sparse numerical equivalence and production promotion remain NO-GO.** Started at `9723d20498f5126d44eec785307b48a580108bd5`. The three inputs were selected by source and content type before observing model results, not by their measured NLL. None is certified training-held-out; all 48 labels in this small sample happen to be the dense and sparse top choice, making it especially unrepresentative of hard prediction cases.

## Analyzer and controls

Mixed CUDA/HIP Release `llama-qsa-reference` and `llama-tokenize` built successfully. Python compileall, the 10 direct aggregator tests, and all 77 `test_*qsa*.py` tests passed. Negative tests cover signed-zero dense differences despite identical numerical values/NLL, mismatched labels, duplicate source bytes under different paths, missing input, invalid provenance and incomplete Q4 groups.

Fixed a substantive analyzer gap: a row count divisible by four did **not** establish actual Q4 group boundaries. It now checks consecutive positions, group starts, row ordinals, sequence, phase, prefix token count and 16-digit fingerprint format. It additionally verifies that each available reference label matches the following input event, and tests reject identically malformed groups or labels across all runs. The final reference label and the correspondence of all inputs to source text still require independent tokenization; metadata equality alone is not proof of either provenance or actual GPU dispatch. Documented this limitation. No numerical quality threshold was introduced.

For every document, independent `llama-tokenize --no-escape --ids` matched **16/16 consumed IDs and 16/16 next-token labels**, including the final label. This is a separate CLI invocation of the same model tokenizer, not an independent tokenizer implementation. The aggregator confirmed matching event histories, prefix hashes/group positions and byte-identical label files across conditions, as well as bitwise-identical dense A/B F32 logits for all 48 rows, including signed-zero distinctions. Its exact-control requirement is experimental hygiene, not a product quality criterion.

## Sources and execution

All inputs are contiguous prefixes ending at the last newline within the first 150,000 bytes; no whitespace or textual normalization, repetition or chat template was applied. Raw sources, selected bytes, tokenizer outputs, fixture JSON, logits, logs and comparisons are in `/home/chris/workspace/llama-perf-results/iteration-0023/`, outside Git. Source digests below are for the **actual selected input**:

| ID | Independently chosen source and category | Input bytes / tokens | SHA-256 | Scored positions |
|---|---|---:|---|---:|
| Alice | Lewis Carroll, [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11), [plain UTF-8 text](https://www.gutenberg.org/cache/epub/11/pg11.txt); public-domain literary text in the US | 149,948 / 36,851 | `6bfaf98678aab5fa286e6833e84fbc97f05f6ab6cbfc796dc1354629d7b1f47f` | 36834–36849 |
| RFC | IETF [RFC 9110, HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.txt), technical standards text; publicly accessible, subject to its stated IETF Trust rights | 149,961 / 36,258 | `24ef8ce24f234cf0b573e0fa103f8946d49d49982dff5b5ba57be390ee4d9b3b` | 36241–36256 |
| Code | This repository's MIT-licensed `src/llama-context.cpp` at the starting commit; C++ implementation source | 149,940 / 41,158 | `a887bb5f676e2d7ab55412dfff8adace3410653760a80f7be5e08a466b18e412` | 41141–41156 |

The code source is a different document/source category from the downloaded literature and standard, but is project-related, not an independent evaluation dataset. Public availability does not establish training exclusion; exact model upstream revision is not recorded in the local GGUF. The model was the same local two-shard Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS GGUF and embedded `qwen35` tokenizer as Report 0022. Each document used three fresh processes, identical CUDA0/ROCm0/ROCm1 layer split 4/7/7, F16 KV, context 65536, Flash Attention, graph-disabled audit, and 16 teacher-forced rows in four Q4 groups. No MTP draft, sampling, GDN observer, forced WMMA or legacy sparse flag was used; only sparse enabled `GGML_HIP_QSA_SPARSE_Q4_GQA12=1`.

For **each** document and each process, the Q4 audit recorded 48 operations: eight CUDA and 40 HIP, across 24 Q / 2 KV heads and four groups. Sparse HIP audits respectively show padded KV 36864, 36352, and 41216, with selection bound 2051 and maximum observed union 4722, 5193, and 5181 (capacity 8204). These satisfy the existing GQA12 RDNA selector's KV>=4096 and KV>=8×2051 guards, alongside the audited Q4/GQA12 shapes and established F16 mixed-device settings. The unchanged selector/dispatch is the path whose actual sparse 4×4 WMMA launches were independently traced on both HIP devices in Report 0022. Thus dispatch is supported by **selector plus eligible-shape audit and previously validated source path** for each fixture; this iteration did not independently profile or count actual kernel launches. The audit itself is a mask observation, not a kernel trace. No performance timing is claimed.

## Paired predictive fidelity

NLL is full-vocabulary next-token loss in nats; delta means sparse minus dense, so positive is worse for the reference token. Each document has 16 correlated neighboring scored tokens, not 16 independent tasks. Dense A/B NLL and distributions matched exactly in every case.

| Document | Dense / sparse mean NLL | Mean / median delta | Best / worst token delta | Worse / better rows | Reference hits dense / sparse | Top-1 flips / changed top-5 sets | Max JS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alice | 0.0102005 / 0.0112096 | +0.00100909 / +0.000004064 | -0.00112759 / +0.00901798 | 10 / 6 | 16 / 16 | 0 / 5 | 0.000673982 |
| RFC | 0.00156232 / 0.00150103 | -0.0000612833 / -0.0000022973 | -0.000460428 / +0.0000176748 | 5 / 11 | 16 / 16 | 0 / 1 | 0.00000309692 |
| Code | 0.0310047 / 0.0297750 | -0.00122973 / +0.0000118242 | -0.0121423 / +0.00183442 | 9 / 7 | 16 / 16 | 0 / 3 | 0.000135648 |

Reference ranks stayed 1 throughout. Maximum KL(dense||sparse), respectively, was 0.00243665, 0.0000128606, and 0.000549313; reverse KL maxima were 0.00307517, 0.0000122807, and 0.000536176. The largest Alice NLL regression was at position 36846; there were no top-token ID flips to enumerate. Despite unchanged top-1 and ranks, top-five set changes and nonzero divergences show distributions changed.

The equal-document macro mean delta and equal-row micro mean delta both equal **-0.0000939732 nats** because each document contributes 16 rows; macro median document mean delta is -0.0000612833. One document's mean is worse and two are better; 24 token rows are worse and 24 better. These are descriptive aggregates, not independent-token confidence estimates or evidence of equivalence. Both the near-perfect reference hit rate and possible source/training overlap are serious limitations.

## Decision

Directions are heterogeneous, rather than consistently regressive. Recommend a larger explicitly designed task-quality evaluation with user-specified deployment use cases, source/contamination policy, metrics and a user-approved acceptance criterion before any GO decision. Do not infer acceptable quality from this sample or no top-1 flips. Natural MTP, graph replay, mixed sequences and graph-safe overflow remain separate gates. No full-model throughput A/B, 98K replay, GPU graph-enabled sparse run, default promotion, GPU reset, driver modification or GitHub Actions was performed.
