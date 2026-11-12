# Report 0031 — offline NVIDIA incident chronology and server logs

**Status: Completed**, with the second incident's workload and tracer state unresolved. Starting commit `8abab498a56afa8709a6eaadb8a457f998031447`. All investigation was CPU-only and read-only with respect to devices. No inference, GPU probing or recovery, stress test, reboot, driver change, sudo or GitHub Actions was performed.

## Validation and parser correction

Compileall passed; the incident analyzer passed **11 tests** and focused QSA discovery passed **174 tests**. Real boot logs exposed two false positives: PCIe `_OSC` capability negotiation was counted as an AER fault, and `iommu: Default domain` matched `fault` inside `default`. Tightened classification to require diagnostic AER/DPC wording and whole-word IOMMU fault/error, added regression coverage, and reran the analysis. Classification remains a limited pattern-based screen, not an exhaustive hardware diagnosis. Raw journals and both initial and corrected analysis outputs remain outside Git at `/home/chris/workspace/llama-perf-results/iteration-0031/`.

## Chronology and limits

Nonprivileged persistent kernel journals for both incident boots and the subsequent boots were available. Times below are local CET, UTC+01:00, on 2026-11-12; both incident Xid pairs name the **same NVIDIA PCI address**. The corrected parser found two Xid 79 and two Xid 154 events in the two failure boots.

- **19:11:05 — first incident, Iteration 0028:** Xid 79 reports GPU fallen off the bus, followed about 123 microseconds later by Xid 154, recovery action **node reboot required**. The server logged a CUDA unspecified launch failure at `cudaStreamSynchronize` while processing the fourth question's prompt, after three completed answers. Monotonic journal and saved client boundaries place this after the third request and during the fourth server phase. ROCm tracing was active: the first three completed requests had 60 verified HIP sparse Q4 launches in six natural verification groups. This establishes concurrency, not that a particular sparse launch caused the CUDA/PCIe failure.
- **19:22:00 — second incident, next boot:** Xid 79 again reports the GPU fallen off the bus; approximately 96 microseconds later Xid 154 reports recovery action **GPU reset required**, not node reboot required. The available journal records a crash dump creation, but the examined window and result-directory artifacts did not establish the active application, CUDA error, inference phase or ROCm tracing state. Do not infer these were the same as the first incident. No crash-dump collection or reset was attempted.
- **20:06:00 — intermediate boot:** a separate Xid 62 appears during boot. Its cause and relationship, if any, to the prior failures are not established; it is not another observed 79/154 pair.
- **Gen3 comparison:** Report 0030 recorded Gen3 x4 under model load, Gen1 x4 idle, maximum Gen3 x16. One dense smoke, fresh dense A/B and traced sparse runs all completed; available Gen3-boot journal through the investigation contained no classified Xid, PCIe AER/DPC, CUDA or AMD/IOMMU fault messages. No matched AER/DPC error was found preceding either 79 in the supplied failure-boot kernel logs. Absence from accessible logs and the parser's patterns does not prove no link error occurred.

A fallen-off-bus report identifies a failure state, not a unique physical mechanism. Workload, tracing, driver, power and PCIe/link conditions remain possible contributors; the second workload is unknown. One successful Gen3 trial cannot establish a permanent fix. The observed x4 width alone does not establish a defect; a human may separately inspect slot/riser topology if desired. Sparse stays disabled by default; numerical equivalence and production promotion remain **NO-GO**.

## Watching ordinary llama-server output

Iteration 0030 captured server stdout and stderr together in actual files:

- `/home/chris/workspace/llama-perf-results/iteration-0030/dense-a-server.log`
- `/home/chris/workspace/llama-perf-results/iteration-0030/dense-b-server.log`
- `/home/chris/workspace/llama-perf-results/iteration-0030/sparse-server.log`

They contain ordinary `slot print_timing` prompt-progress, prompt-evaluation and generation-evaluation tokens-per-second lines. For example, the first sparse request logged prompt evaluation at 1515.22 tokens/s and generation at 34.02 tokens/s; those incidental numbers are not a controlled full-model throughput comparison. The old processes have ended, so these files will not grow. For a future separately authorized run writing to a known file, watch from another terminal with:

```sh
tail -n 30 -F /home/chris/workspace/llama-perf-results/iteration-0030/sparse-server.log
```

Substitute the **actual new run's** server log path. If the path is unknown and the server PID is known, `ls -l /proc/PID/fd/1 /proc/PID/fd/2` shows whether stdout/stderr target a file, terminal or pipe; replace `PID` with the process ID. Tail a file, but do not read an agent-owned pipe directly because it can steal output; use the original terminal or arrange file capture for a future approved run. Server logs may contain sensitive prompts or outputs and should stay outside Git.
