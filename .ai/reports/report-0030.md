# Report 0030 — recovered-host Gen3 matched quality comparison

**Initial gate (subsequently resolved as described below): Blocked.** The operator reports a power cycle and Gen3 firmware setting. At 2026-11-12 20:16:49 +01:00, a bounded read-only `nvidia-smi` query independently observed the RTX 4070 Ti SUPER on driver 580.178.04 with **current PCIe Gen1 x4**, maximum **Gen3 x16**, 35°C and 5.88 W. NVIDIA enumeration succeeded; PCI enumeration also showed both discrete AMD devices. Current Gen1 may reflect idle power management; it is not evidence of a fault, but neither is it verification of an actually negotiated Gen3 link. The nonprivileged `lspci -vv` output did not expose link details. Current-boot kernel journal inspection found no matching NVIDIA Xid or PCIe AER/DPC errors. This is a limited passive snapshot, not a stability certification.

The handoff initially required verification of the **current** Gen3 link before the inference stages. That gate was not met, so no smoke, fresh dense A/B or sparse inference was started, and no new HIP launch or task-quality claim is possible. In particular, there was no observed new serious NVIDIA failure and no recovery action or automatic retry. An operator should independently establish an authorized way to observe Gen3 negotiation under appropriate conditions and provide guidance on proceeding; maximum Gen3 alone is insufficient. No inference was used to force a higher link state.

## CPU validation and preservation

Starting commit `af241d52b64b214c37a6cf4e6ff34bb3f12866af`. Compileall passed; recovered-comparison tests passed 11/11; focused QSA discovery passed 163/163. The new comparator was validated by its synthetic tests, not by new GPU results. CPU-only frozen-artifact preflight passed, including the complete historical dense controls (14/24) and separate incomplete 3/24 sparse prefix. Dataset, prior manifest and frozen 24-question manifest SHA-256 values respectively match `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`, `d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1` and `ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5`. The prior 0029 preflight file remains present, SHA-256 `dc929ffc7b35152fb053143a03207c276703f8852ee608ca3376f5349474c131`; its hash was not previously recorded, so unchanged status cannot be independently established from that digest alone. Frozen questions and gold were not modified.

New passive and preflight logs are outside Git under `/home/chris/workspace/llama-perf-results/iteration-0030/`; previous artifacts were not overwritten.

## Continuation after operator clarification — completed

The user clarified that Gen1 is the idle state and explicitly requested continuation. One short dense mixed-GPU smoke was run; during model loading, `nvidia-smi` independently observed **current Gen3 x4**, maximum Gen3 x16. The smoke completed normally, using the target and draft with the same CUDA0/ROCm0/ROCm1 mapping, no sparse flag and no ROCm tracing. Subsequent phase checks still observed Gen3 x4, and a later idle query returned Gen1 x4. Link width x4 is an observation, not a claim that the card is running at its maximum x16 width.

Fresh dense A and B then completed all 24 frozen questions in separate server processes. Their responses, finish reasons and prompt counts matched on 24/24; each scored **14/24**. The recovered-host comparator found **zero** answer/finish changes versus the historical dense controls. Dense logs finished at 20:25 and 20:30 local time on 2026-11-12; configurations and individual client/server logs are preserved externally. The fresh traced sparse server then completed all 24 original questions, finishing at 20:38, with no reuse of the historical three-answer prefix. Sparse was compared **only against the new dense A/B**:

| Domain | New dense | New sparse |
|---|---:|---:|
| Long-dialogue History Understanding | 2/6 | 2/6 |
| Long In-context Learning | 1/1 | 1/1 |
| Single-Document QA | 9/13 | 9/13 |
| Multi-Document QA | 2/4 | 2/4 |
| **Total** | **14/24** | **14/24** |

Strict scoring found 14/14 baseline-correct preserved, zero lost, zero genuine regressions and zero improvements. One response changed text but both versions were correct (source ID `66f3918f821e116aacb2d8b7`); zero changed responses were wrong in both. All prompt counts aligned. The recovered comparator, strict stable-control scorer and evidence analyzer all completed. The illustrative iid one-sided 95% upper sensitivity value is 0.1926 for 0/14 regressions; iid representativeness is not established, so this is **not** a population guarantee, release threshold or automatic GO.

ROCm kernel tracing was enabled for the entire sparse process and may change the load pattern. Monotonic client request and server target-process boundaries attributed **470 real HIP sparse Q4 GQA12 launches within 47 natural batch-4/draft-3 MTP verification groups**, across all 24 tasks and both HIP agents; each group contained five launches per agent. The trace had 710 such launches total, including those outside these verification spans. The per-task external trace summary verifies at least one group and launches on both agents for every task.

Bounded NVIDIA checks after each phase succeeded; current-boot journal checks found no matching Xid or PCIe AER/DPC errors through completion. No new CUDA/HIP failures were observed. This single successful run does not establish that Gen3 fixed the earlier fault or prove its root cause. Raw traces, results, scripts, process arguments and logs remain outside Git. Sparse remains disabled by default; numerical equivalence and production promotion remain **NO-GO**. No GPU reset, reboot, sudo, stress test, full-model throughput benchmark or GitHub Actions was performed.
