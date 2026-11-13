# QSA dense top-k raw-order investigation — source findings after Report 0037

**Status: source-level cause established; original external raw arrays not replayed in Chat AI's container.** This is an offline examination and CPU-only validation. It does not introduce a new GPU kernel, model run, production selector change or quality threshold.

## Observed data

Report 0037's first layer15 HIP0 QSA capture had exactly matching dense A/B score, pool mapping, extra cell mapping, selected **sets** and downstream captured attention, masks, GDN recurrent state and logits. Despite this, the **raw ordered 512-element top-k arrays differed at 1,962 of 2,048 positions** across four query rows. This is **not a new FP32 side effect**. Report 0018 already independently observed nonidentical dense A/B top-k order and identical pool/KV membership, back when the earlier sparse arithmetic had not been replaced.

Report 0018 establishes the relevant group shape: **8,256 candidate pool indices × 4 query rows**, selecting 512 per query. Report 0037 captures the same first-layer15 four-query QSA operation. The raw 0037 files are outside Git in the Coding Agent's external iteration-0037 directory; those bytes cannot be inspected in Chat AI's container. Hence this note cannot independently assert a per-rank/tie-tail count for that exact pair yet.

## Source-level mechanism: unordered HIP radix gather

In the current repository:
- ggml/src/ggml-cuda/top-k.cu, ggml_cuda_op_top_k(), chooses the radix implementation for **8,256 columns × 4 rows** on HIP. The HIP short bitonic threshold is 1,024 columns; nrows=4 also exceeds the small-row threshold. Neither the 1,024-wide bitonic path nor an explicitly descending sorted result applies to this captured operation.
- top_k_radix_gather() writes every score **strictly greater than the kth threshold** to the next output slot using **atomicAdd(&state->greater_count, 1)**. GPU execution timing decides the order in which those index writes claim positions. The output prefix is therefore an **unordered permutation of the correctly chosen above-threshold pool IDs**, even with byte-identical scores and masks.
- The same kernel handles exact-threshold **ties separately**. It calculates the per-row residual rank and writes qualifying equal-score pool IDs in ascending pool-index order into the last slots. The finite score-to-ordered-uint32 conversion distinguishes signed zero and must be matched exactly by a CPU audit. The tie-tail portion should remain stable across two identical inputs, while the atomic prefix need not.
- In ggml/src/ggml-cuda/top-k-mask.cu, top_k_mask_kernel() fills the destination with -Infinity, then for each **selected pool ID**, copies the same source mask value into each valid mapped KV cell. It also copies extra cells independent of selected array order. Since it uses pool IDs and not their rank positions, a permutation of identical selected sets with unchanged score/map/mask inputs cannot change which cells are unmasked. Multiple picks mapping to one cell write the same source value; this is a write-collision caveat but not a score-order semantic requirement.
- The report's exact dense attention/mask/logit/GDN repeats are therefore consistent with **expected top-k output-order nondeterminism with deterministic membership**, not a newly identified kernel arithmetic bug or a CUDA GPU bus fault.

### Implication

Use **exact score bytes, tie policy, selected pool sets, mapped KV sets and ultimately mask bytes**, not raw ordered top-k array byte identity, as the dense-control correctness condition for this HIP radix path. A raw order mismatch is informative about the scheduling of GPU atomics, but does **not** itself imply a changed attention mask. It does **not** establish a CUDA timing or PCIe hardware fault either.

A future code change adding an unconditional sort to the selected top-k indices is **not justified by these observations**. It would add work and could alter launch timing; it would not address the FP32-versus-half2 numerical-fidelity question. This is not an endorsement of graph replay or untested paths.

## New CPU-only capture validator

New **tools/analyze_qsa_topk_order.py**, with 13 synthetic regression tests **tools/test_analyze_qsa_topk_order.py**, performs a fail-closed audit on the **existing** QSA_TOPK_TRACE captures:
- loads each capture through the existing comparator parser, matches HIP0 layer15 event keys, shapes and exact score/blocks/extra **bytes**;
- validates all selected indices are unique, in bounds and exactly equal to the expected top-k **set** with uint32 score ordering and ascending-index threshold tie policy, including raw +/-0 distinction;
- asserts that the deterministic threshold tie-tail is correct, counts raw order-mismatched positions **in the nondeterministic above-threshold prefix versus the tie tail**;
- verifies the reconstructed mapped KV sets are identical regardless of top-k order, counts descending-score inversions, and emits a bounded JSON report outside Git;
- refuses any mismatch in raw scores/maps, membership, threshold tie policy or mapped cell sets rather than silently normalizing or attributing it to atomic scheduling.

**Local Chat AI check:** 13 synthetic tests passed and the script compiled with Python syntax checks in a temporary container copy. **Not validated on the original 0037 raw captures**, because those bytes are on the external Coding Agent workstation.

The offline-only original-capture replay should be performed on the Coding Agent host, without launching llama-server, CUDA/HIP, telemetry or GPU tracing. For example, with the actual saved log paths (do not guess filenames), run:

    python3 docs/LLAMA-PERF/tools/analyze_qsa_topk_order.py \
        /path/to/iteration-0037/dense-a-server.log \
        /path/to/iteration-0037/dense-b-server.log \
        --out /path/to/new-external/dense-order-audit-0037.json

The expected original first-group tally from Report 0037 is **1,962/2,048 changed raw positions**. The decisive additional check is whether the tool attributes **all changed positions to the atomic above-threshold prefix**, with **zero changes in the deterministic tie tail**, while confirming the same exact membership and mapped KV sets. Run the earlier 0018 dense controls as a separate cohort if their raw files remain available; don't pool them as the same model process or logical history. If a mismatch appears in the tie tail, score bits, policy membership or mask cells, preserve it and investigate rather than calling it harmless.

## Open boundaries

This rules out raw top-k rank-order equality as a *required invariant* for the known HIP radix and fused-mask QSA path, but it does not establish that **every consumer of TOP_K** in every model architecture is order-insensitive. No GPU test here isolates NVIDIA PCIe link stability, and no bound for sparse FP32 production fidelity exists. User-approved quality thresholds, graph replay safety, mixed-sequence behavior and sparse overflow policy remain separate NO-GO gates.
