# Handoff 0032 — one controlled Gen3 repeat of the exact Xid79-triggering sparse workload

## Objective

The user explicitly wants to **redo the test during which the CUDA issue first appeared**. That was the original frozen **24-question LongBench-v2 sparse-attention run from Iteration 0028**, interrupted during **the fourth question's CUDA prompt prefill** with NVIDIA Xid79/154. The exact same 24-question manifest subsequently passed **all 24** under the operator-selected NVIDIA PCIe **Gen3 x4** configuration in Iteration 0030. Report 0031 correlated the original two fall-off-bus incidents but could not establish the workload of the second or a unique root cause. The requested next step is **ONE more bounded repeat of the same frozen 24 sparse questions on the current Gen3 setup**, to check reproducibility and capture normal llama.cpp timing output.

This is *not* permission to change PCIe back to the previous generation, force the Xid failure, intentionally load/stress hardware beyond the original bounded run, or run a repeated soak/throughput test. A second healthy run would add evidence of stability but **still would not establish that Gen3 permanently fixed PCIe communication**, numerical equivalence or production readiness.

## Ownership and starting state

Repository: civcode/llama.cpp-perf
Branch: docs/llama-perf
Starting commit: use exact SHA in the user-relayed Chat AI transfer prompt.
Read .ai/WORKFLOW.md, .ai/reports/report-0028.md, report-0030.md and report-0031.md, and docs/LLAMA-PERF/QSA-SPARSE-TASK-QUALITY.md. Coding Agent takes exclusive repo write ownership after handoff and returns it upon commit/push of Report 0032. No concurrent repo changes.

Verified prior results:
- NVIDIA RTX 4070 Ti SUPER, driver 580.178.04, firmware-capped **Gen3** link; observed Gen1 x4 idle and **Gen3 x4 during inference** in Report 0030, with two AMD HIP GPUs. Width x4 vs card max x16 is an observation, not a diagnosed fault.
- Frozen full original 0028 manifest: /home/chris/workspace/llama-perf-results/iteration-0028/tasks.json, SHA-256 ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5. Twenty-four non-overlapping source-easy LongBench-v2 tasks, actual chat inputs 17,051–45,158 tokens, context65536. Original publisher source digest 15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2.
- Iteration 0028: pre-Gen3 dense A/B stably correct 14/24; sparse process completed only first three tasks then NVIDIA CUDA0 disappeared on the fourth. This incomplete prefix is purely historical, never join it into another file.
- Iteration 0030 under Gen3: fresh complete dense A/B (24/24 repeats identical, 14/24 correct, exact same results as original 0028 dense); complete fresh sparse 24/24, 14/24 correct, one text change but still correct; 470 real HIP GQA12 sparse Q4 launches inside 47 natural 4-token MTP target-verification groups, attributed to all 24 tasks; zero observed new NVIDIA Xid/PCIe faults. It used ROCm tracing for the full sparse process.
- Report 0031 located real ordinary llama-server logs in iteration-0030/dense-a-server.log, dense-b-server.log and sparse-server.log. They contain actual slot print_timing prompt/gen tokens/sec. CPU-only analysis of two incident boots found two Xid 79/154 pairs; active workload during second remains unknown. No preceding classified AER/DPC events were observed in accessible logs.

## Chat AI implementation — host validation required

New docs/LLAMA-PERF/tools/compare_qsa_sparse_repeats.py compares two complete, separate sparse-result JSONL files against the EXACT frozen 0028 manifest. It uses strict existing _read_results to enforce 24 complete, correctly ordered requests and matching manifest/prompt/sampling metadata; verifies matching prompt-token counts, grade transitions, exact text and finish changes, pins the previous successful 0030 sparse score at 14/24, and labels all GPU/production claims unverified. Writes only an external summary. Corresponding CPU-only tests: docs/LLAMA-PERF/tools/test_compare_qsa_sparse_repeats.py (10 cases, not locally run yet). Updated QSA-SPARSE-TASK-QUALITY.md for the exact repeat protocol. No inference code, backend numerical kernel, runner, graph or CUDA/HIP driver modified.

## A. Before GPU activity: validate and compare operating conditions

1. Run Python compileall, new direct repeat-comparator tests, and related QSA tests if useful. Fix ordinary Python errors autonomously; no GitHub Actions. Validate pinned frozen dataset/manifest with the previously passing CPU-only 0029 preflight, choosing a fresh output file under iteration-0032/.
2. Verify existing iteration-0030/dense-a.jsonl, dense-b.jsonl and sparse.jsonl were fully completed with original 24 correct task IDs and unchanged source hash, model GGUF files, server executable/build, driver, CUDA0/HIP0/HIP1 mapping, KV/layout, MTP draft and sparse environment. Record exact source and model fingerprints/config from existing 0030 logs. Do NOT overwrite any old results.
3. Perform a bounded, nonprivileged passive device check (NVIDIA accessible and same card/driver, both AMD devices, current PCIe link max Gen3 and expected width) and current-boot journal review for fresh Xid79/154 or serious PCIe/CUDA error before launching. Under idle load, observed Gen1 may be normal power management; do not falsely demand Gen3 *currently* while idle. Device count or read-only enumeration is not a stability certification.
4. If any settings changed materially compared with the successful 0030 cohort (including PCIe generation, CUDA driver, GPUs, target/draft weights, build, context/sampling), STOP before sparse and report the mismatched configuration; do not silently reuse the 0030 dense controls. If current health signals are bad or an unrecovered Xid is evident, STOP. Do not run a model just to "test whether it still crashes."

## B. Exactly ONE bounded complete sparse repeat, unchanged Gen3

Only if A passes and the same healthy Gen3 configuration is established:

1. Create a NEW external directory under /home/chris/workspace/llama-perf-results/iteration-0032/ with unoccupied output names. **Start exactly one** fresh bounded local llama-server with the **same 0030 sparse target/draft weights and flags**, mixed CUDA0/ROCm0/ROCm1 split4/7/7, IQ3_XXS target, Q8 MTP draft, F16 KV, context65536, max draft3, reasoning OFF, temp0/top_p1/seed42, prompt cache OFF and graph-disabled QSA mask audit. Sparse ONLY via GGML_HIP_QSA_SPARSE_Q4_GQA12=1, with other sparse and forced-WMMA overrides unset.
2. Reuse the same bounded, nonprivileged ROCm tracing/monotonic request markers that worked in 0030 to attribute **real** experimental HIP GQA12 Q4 launches in **natural** four-token MTP verification spans on both HIP GPUs. Profiler configuration should match 0030. If tracer crashes or new CUDA/HIP hardware errors appear, STOP instead of silently switching to unverified sparse execution. Never use sudo/privileged hardware profiling.
3. Use exactly the frozen /iteration-0028/tasks.json and the existing run_qsa_quality_tasks.py client, sparse condition, same prompts/order/budgets. Write to a **new** /iteration-0032/sparse.jsonl (exclusive-create, no append to 0028/0030), and capture the running server's **combined ordinary stdout/stderr** at a known file /iteration-0032/sparse-server.log; preserve process exit/failure status, timeout bounds and cleanup only owned process group. Do not run a second new dense A/B or model smoke if unchanged 0030 conditions and controls have been confirmed.
4. Make the normal live server console log accessible to the user from their own terminal while the run happens:
   tail -n 30 -F /home/chris/workspace/llama-perf-results/iteration-0032/sparse-server.log
   This must contain llama-server's normal slot print_timing prompt and generation tokens/sec messages if provided by the build. Do not pretend it streams per-token chat text or is a controlled throughput benchmark. The agent need not wait for user interaction to execute the existing authorized single run.
5. Use bounded passive NVIDIA/PCIe health and current-boot Xid error checks before and after server startup and each client request if feasible. Record actual current Gen3 x4 under model load without modifying device settings. **If any CUDA failure, device disappearance, Xid79/154, serious new PCIe fault, tracer failure or other serious hardware anomaly appears: immediately STOP all GPU activity, preserve logs, do not retry, reset or reboot.** If the run ends partially, report Partial/Blocked; **do not** append old responses or present a full accuracy number.
6. This is a single completed repeat maximum. Do NOT perform loops, additional model runs to seek a crash, GPU stress tests, PCIe Gen4/Auto reconfiguration, 98K context replay, graph-enabled sparse or 512-token full-model throughput benchmark.

## C. Offline attribution after all 24 complete, and report

- Only if all 24 new sparse requests finish normally and every case is trace-confirmed, score against the previously matched complete **0030 dense A/B** using existing score_qsa_quality_tasks.py with --require-stable-controls; run analyze_qsa_quality_evidence.py on that full scorer output. Do not compare against pre-Gen3 0028 dense as the matched condition.
- Independently run new compare_qsa_sparse_repeats.py on the same frozen 0028 manifest plus the **complete** 0030/sparse.jsonl and **complete** 0032/sparse.jsonl. Distinguish answer-text changes from actual correctness transitions, and ensure same token counts. Exact output form, completeness and input alignment must be verified; no output merger or re-run after failure.
- Verify actual HIP Q4 MTP verification launch groups attributable to all 24 tasks on both HIP devices and preserve those per-task counts outside Git. The benchmark score by itself does not prove sparse was active. Report ordinary console tokens/sec *observations* only; do not infer a statistically controlled speedup from these incidental logs.
- Review kernel journal/current host snapshots after the completed run; report any new Xid/CUDA/AER/DPC and state if no matching errors were *observed*. A successful run cannot prove permanent GPU stability or causal responsibility of Gen3.
- Keep raw model inputs, benchmark texts/answers, device identifiers, logs and HIP profiler CSV in the external iteration-0032 directory. Commit only compact report, test fixes and aggregated findings.

## Safety, authority and completion

- No node reboot, GPU reset, driver reload, PCIe link/clock/power change, sudo, privileged profiling, forced hardware rescan, GitHub Actions workflow run or auto triggers, PR/merge/force push, default sparse promotion or kernel numerical changes.
- Sparse remains disabled by default. **Numerical equivalence and production promotion remain NO-GO**, irrespective of one or two successful 24-case results. The user's explicit requested test is for reproducibility/host confidence, not a production tolerance.
- If healthy repeat finishes, write .ai/reports/report-0032.md summarizing exact starting/ending revisions, pinned manifest, confirmed unchanged Gen3 config and model fingerprints, CPU tests, actual 24/24 output and Q4 per-task launches, error-free or failed host state, simple quality comparison to 0030 and genuine limitations.
- If any hardware error recurs or comparable controls cannot be verified, STOP immediately, write a clear Partial/Blocked report, preserve all evidence, and never retry without another explicit user decision.

Commit/push the report on docs/llama-perf and return:
Iteration: 0032 | Status: Completed/Partial/Blocked | Branch: docs/llama-perf | Commit: <pushed-sha> | Report: .ai/reports/report-0032.md | Summary: <1-2 sentences>
