# QSA GQA12 sparse attention — numerical-fidelity measurement protocol

**Status:** Research protocol, not a product acceptance criterion. Experimental sparse GQA12 attention remains **NO-GO for production promotion**. This document does not authorize a default change, production full-model performance A/B, relaxed overflow guards or GPU graph replay.

## What we know

Reports 0017–0021 established that the sparse HIP path can be closer to FP64 mathematical attention on identical layer-11 Q/K/V and F16 mask, while producing different F32 attention outputs, crossing narrow layer-15 top-k cutoffs and changing downstream GDN recurrent R/S state and full-vocabulary target logits. In 64 naturally matched MTP verification positions, dense A/B repeated exactly; dense/sparse differed in all rows (max absolute logit difference 0.91864, max row NMSE 0.00462, max KL(dense||sparse) 0.01063 nats, five changed top-5 sets but no top-1 changes). Reports 0020–0021 validated checkpoint restore and serializer-state observer bytes on selected layers and real staged cross-cell source propagation, but not every multi-sequence or graph condition.

The absence of top-1 flips in 64 positions is **not** a numerical acceptance threshold.

## Separate three questions

1. **Operator fidelity:** Do both kernels behave correctly on exactly identical Q/K/V/mask? Measure against the same FP64 reference, preserving the actual scale, type, head mapping and score mask. Different reduction order need not be bitwise identical.
2. **Fixed-token predictive fidelity:** For externally selected next-token labels from a genuinely held-out or teacher-forced sequence, compare dense and sparse full-vocabulary pre-sampling logits on the *same exact token IDs and positions*. Report next-token negative log likelihood (NLL), reference rank/top-1 accuracy, distribution drift and outliers; do not conflate this with stochastic speculative decoding.
3. **Operational/task fidelity:** On tasks relevant to the target deployment, use deterministic scoring and dense/dense controls, *then* separately verify normal MTP acceptance/rejection, mixed sequences, GPU graph replay and graph-safe fail-closed overflow. A teacher-forced sequence cannot certify natural MTP execution.

## Small reference-token experiment

Implement a strictly bounded, disabled-by-default C++ target-decode fixture or equivalent observer that teacher-forces known next tokens, emitting a complete F32 logits vector **before any sampling** for each labelled position, with consistent step/phase/seq/position, exact input IDs, target model, weights and F16 KV settings. The existing LLAMA_QSA_LOGITS_DIR observer captures natural speculative target-verification logits; do **not** label it teacher-forced unless the sequence was actually constrained.

Use an independently chosen reference text/token fixture and document its provenance, same tokenizer and token schedule for dense A/dense B/sparse. Supply a separate JSONL label record with fields step, phase, seq, position, label_token and origin. The origin must be heldout_reference or teacher_forced_reference. Do not take gold labels from the dense model's predicted IDs, sparse predictions or speculative acceptance. Align labels with **the next token actually scored by each logits row**, not just a similar position number.

Example single reference record (illustrative, not real data):

    {"step":0,"phase":"verify_logits","seq":0,"position":32809,"label_token":1234,"origin":"heldout_reference"}

The host-only tools/compare_qsa_reference_nll.py checks strict event/candidate-history alignment and exact reference label coverage; it reports signed sparse-minus-dense next-token NLL and reference token ranks, hit counts and outliers without a hard-coded PASS threshold. Run test_compare_qsa_reference_nll.py and other focused host suites locally first.

For the first experiment capture **no more than 64 logits rows** across a bounded short prompt and one relevant long-context prompt. Run dense A/B controls before sparse, with identical graph-disabled conditions and only the sparse process setting GGML_HIP_QSA_SPARSE_Q4_GQA12=1. Legacy GQA4 and WMMA override flags remain unset. Keep raw logits, text, label/fixture token IDs and all processes/logs outside Git. These initial rows are exploratory, not a statistically powered quality study.

## Interpretation and decision

Report dense A/B variation, row-aligned reference-token NLL difference, top-1 reference hit and rank changes, largest regressions/improvements, and whether the sparse path systematically changes actual labelled prediction errors. Document test-set limitations, normalization, exact tokenization, target positions and any forced-token schedule.

**No numerical/quality threshold is established here.** The user must explicitly approve a consequential model-fidelity and task-quality acceptance criterion. Even if sparse NLL improves on a tiny sample, it does not establish normal MTP, GPU graphs, mixed-sequence or overflow correctness.

If predictive fidelity degrades at near-tied selection cutoffs, another reversible experiment is an opt-in arithmetic variant aiming closer to the dense accumulation behavior, followed by the same fixed-token reference regression. Do not silently choose a tolerance or claim a production speedup from synthetic QSA operator benchmarks.
