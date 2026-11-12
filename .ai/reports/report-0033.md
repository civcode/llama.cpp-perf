# Report 0033 — sparse layer-11 causal ablation

**Status: Completed for the bounded pilot.** Starting commit `de34b705f199a815ffb2ceffa8c846909b76b8af`. On this matched fixture, enabling sparse **only at layer 11** is sufficient to reproduce the first layer-15 selection changes and layer-12 recurrent-state drift; disabling sparse at layer 11 removes those particular observed differences even while later HIP sparse layers execute. Later layers independently still change verification logits. This is a conditional intervention result, not proof of a unique global cause, a kernel defect, numerical equivalence or an acceptable tolerance. Sparse remains disabled by default and production promotion **NO-GO**.

## Validation and historical evidence

Standalone C++17 layer-filter tests passed with `-Wall -Wextra -Werror`; fixed a backslash-ended comment that initially broke this build. Python compileall passed; the new localizer passed 14 tests, all 199 focused QSA tests and seven FP64 replay tests passed. The mixed CUDA/HIP Release `llama-server` and `test-backend-ops` build succeeded. The build's UI provisioning fell back from its pinned asset to `latest`; this affects bundled UI provenance, not an asserted numerical comparison. Source inspection confirms a missing layer-filter option returns the original GQA12 environment value unchanged; this diagnostic gate does not enable sparse by default. The new build was used for all five fresh conditions, not compared as the same binary as 0032. The rebuilt binary hashes are preserved externally.

Reanalyzed the existing 0019 dense A/B/sparse manifests at steps 0,6,8,9 and layers 12,32,44: all 24 selected dense rows were byte-identical, all 24 sparse rows differed, first observed at step 0 layer-12 R (max abs 0.0248908). This snapshot is **after target decode**, not the first possible cause. Separately, 0016/17 established layer-11 changed arithmetic with identical inputs and a reference that did not uniformly favor dense; 0018 observed near-cutoff selection changes; 0019 captured aligned GDN/logit drift. Those were separate cohorts, not one proved causal trace. Historical captures were available. While replaying new captures, fixed the CPU FP64 analyzer incorrectly parsing capture-limit notices as records and added a regression.

## Fresh matched intervention

Five separate fresh processes, in order dense A, dense B, sparse-all, sparse-only11 and sparse-except11, used the same existing 32K fixture, target/draft and mixed 4/7/7 mapping, F16 KV, context 65536, max draft 3, reasoning off, greedy seed 42 and graph-disabled audit. The original fixture was reused with the same explicit 32-token response budget and greedy settings across conditions; no sampling decisions were forced. Each condition used identical bounded ROCm tracing and captures, including the first Q4 final-prompt operation, eight natural four-token/draft-three verification groups, first layer-15 top-k group, two GDN groups at layer 12 and 34 full-vocabulary verification rows. P and downstream layers 32/44 were not captured in this deliberately bounded fresh pilot. Trace and synchronization perturb timing; this is not a throughput test.

Actual HIP Q4 trace launches, attributed by monotonic QSA audit intervals and GPU agent:

| Condition | Sparse Q4 launches by layer |
|---|---|
| dense A, dense B | 0 each |
| sparse-all | 9 each at layers 11,15,19,23,27,31,35,39,43,47 (90 total) |
| sparse-only11 | 9 at layer 11 only |
| sparse-except11 | 9 each at layers 15,19,23,27,31,35,39,43,47 (81 total); zero at 11 |

The nine launches per enabled layer include one final-prompt Q4 and eight natural Q4 MTP verification groups. All were attributed; no claims rely solely on environment flags. Dense A/B had byte-identical captured attention outputs, layer-15 score/maps, 12 layer-12 R/S snapshots and 34 logits rows; logical histories, candidates and acceptance decisions aligned across all five conditions for the compared records.

## Localization results

- **Layer 11:** Q/K/V span bytes and F16 mask bytes were identical across all five at the first captured operation. Sparse-all and only11 had the same dense-vs-sparse max absolute output difference **0.0184302**, NMSE **1.05615e-6**; except11 and dense B were bit-identical to dense A. Against the actual-input FP64 reference, dense NMSE was **1.18147e-6** and sparse-all/only11 **3.68239e-7**. Per-head reference results are preserved externally; the reference does not establish an acceptance criterion or prove a defect.
- **Layer 15:** For the first Q4 group, all/only11 changed pool and mapped-KV selection in **3/4 queries**, with identical block/tail mapping. Example query 1 removed pools 63 and 2958 and added 2122 and 6807; the dense cutoff margin was 0.000787735 versus maximum paired score delta 0.0212178. Except11 had byte-identical scores and mappings and unchanged selected sets. The only11 intervention therefore conditionally reproduces the observed cutoff crossings without sparse layer 15 itself.
- **Layer 12 GDN:** All/only11 changed **12/12** captured R/S rows; the first observed row was step 0 pre-verification, after target decode, R max abs 0.0212607 and NMSE 1.95339e-5. Except11 had **0/12** changed R/S rows. This isolates a contribution to these snapshots, not proof of an incorrect rollback or of behavior in uncaptured layers/P states.
- **Verification logits:** All three sparse conditions changed **34/34** aligned rows despite unchanged sampled histories. Maximum absolute delta was 0.55040 all, 0.56613 only11, and 0.65486 except11; top-1 changed 0/34 in each, while top-5 sets changed 2, 3 and 2 rows respectively. Thus layer 11 is not necessary for *all* downstream logit drift: other sparse layers can independently contribute. No forced decisions or inference beyond the first divergent history were used; no divergence occurred in captured histories.

The host reported Gen3 x4 under each loaded server and no new matching Xid/PCIe/CUDA/HIP failure was observed. No reset, reboot, driver or hardware changes, stress test, GPU graph test or GitHub Actions was performed. All raw captures, model data, per-head results, traces, configurations and logs remain outside Git in `/home/chris/workspace/llama-perf-results/iteration-0033/`.

**Next action:** investigate a controlled sparse arithmetic refinement against the exact-input FP64/headwise captures and repeat the same layer-11 ablation, while separately defining a user-approved model-fidelity criterion. Other layers, graph safety, mixed-sequence behavior and overflow remain independent unpassed gates; no global sole-cause or release claim follows from this fixture.
