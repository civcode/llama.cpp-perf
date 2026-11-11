# Report 0005

Status: Partial. The experimental grouping passes the selected CPU-reference tests on both R9700s, but is slower on a matched single-device attention microbenchmark. The requested three-device target-only benchmark and model-level numerical comparison are blocked by the NVIDIA GPU falling off the PCIe bus. Do not promote the experimental option. Source revision: `8923315e98b82bf489137e23cfc2a3e821537cd4`.

## Build and long-KV correctness

`cmake --build build-perf -j 28 --target test-backend-ops llama-bench llama-server` succeeded. Each Radeon ran the four opt-in `GGML_TEST_LONG_FA=1` CPU-reference cases in separate default and `GGML_HIP_FATTN_TILE_MAX_GQA=2` processes: 4/4 passed per arm and device. The baseline instrumentation runs set the tuning variable to `0`, which selects the normal dispatch; separate baseline performance and 98K correctness processes used an unset variable. The threshold was NMSE <= 0.0005; both GPUs produced the same values:

| KV | GQA | Default NMSE | Max 2 NMSE |
| ---: | ---: | ---: | ---: |
| 4096 | 4 | 1.25320e-6 | 1.68346e-6 |
| 16384 | 4 | 5.29149e-6 | 7.07593e-6 |
| 32768 | 4 | 7.88595e-6 | 1.09856e-5 |
| 8192 | 8 | 2.69374e-7 | 2.69374e-7 |
| 98304 | 4 | 2.48310e-5 | 3.27270e-5 |

All are passes. The 98K case was added temporarily for this measurement and is not part of the committed suite. The test runner normally prints errors only on failure; a temporary print recorded each successful comparison. The temporary patch is saved with the logs, was reverted, and the final targets were rebuilt. The cases use 256x256 heads, three queries, two KV heads, F16 K/V, a deterministic mask and strided KV views. They check the tested output against a CPU reference, including NaN/inf checks, not direct baseline-versus-experiment logits or model state. In particular the synthetic inputs and mask do not establish accuracy for every real prompt or attention distribution. The larger GQA4 reference error in the experimental arm is evidence of a numerical difference, not by itself a failure or proof of the cause of prior sampled-token divergence. GQA8 happens to give identical reported errors; no dispatch conclusion is inferred from that alone.

## Fixed-work Radeon attention timing

As a fallback, the test runner's existing `perf` mode was temporarily wired to the deterministic cases, including KV 98304. Each case warmed up once, then repeated its graph for at least one second; the reported number is wall time divided by repeated attention operations, not a profiler-isolated kernel duration. Inputs, shapes, device, and work were matched; each arm used a fresh process. Arm order per device was default, max 2, max 2, default, default, max 2. These are three-query synthetic attention operations, not single-token target-only inference or MTP verification throughput.

| Device | KV | Default us/run, 3 runs | Max 2 us/run, 3 runs | Mean latency change |
| --- | ---: | --- | --- | ---: |
| ROCm0 | 32768 | 114.02, 114.11, 114.25 | 122.52, 122.79, 123.01 | +7.6% |
| ROCm0 | 98304 | 341.93, 342.10, 342.31 | 363.63, 363.69, 365.39 | +6.5% |
| ROCm1 | 32768 | 113.45, 114.14, 114.37 | 122.00, 122.08, 123.29 | +7.5% |
| ROCm1 | 98304 | 341.84, 342.34, 342.83 | 362.57, 363.13, 365.05 | +6.2% |

This controlled device-local workload shows no grouping compute advantage in these shapes; it does not prove the end-to-end effect for all batch sizes or the actual target model. Report 0004's HIP traces independently showed increased summed flash-attention time per approximate readback at both depths, but those traces have no CUDA timeline or proven verification boundaries. No new cross-device trace was collected while CUDA was unavailable.

## Blocked target-only and model-level checks

The NVIDIA driver logged Xid 79, `GPU has fallen off the bus`, followed by Xid 154, `GPU Reset Required`, at 18:44:39. `nvidia-smi -L` cannot obtain a device handle; `llama-bench --list-devices` omits CUDA0. An attempted 32K benchmark with `-dev CUDA0/ROCm0/ROCm1 -ts 4/7/7 -sm layer -ctk f16 -ctv f16 -fa on -p 0 -n 128 -d 32768 -r 3` fails at argument parsing with `invalid device: CUDA0`, before loading the model. Therefore there are no valid 32K or 98K target-only t/s, effective placements, fixed-token logits, GDN-state checks, or sampled-margin comparisons for this iteration. A Radeon-only or CPU-offloaded model run would change placement and is not a substitute for the requested matched split. No GPU reset or reboot was attempted.

After NVIDIA recovery, rerun the target-only benchmarks at both depths, in separate unset/max-2 processes with the stated split and three repetitions; record actual placement and use fixed-token teacher forcing to compare finite logits, score margins, and recurrent state. A full cross-device dependency timeline is still needed before attributing inference wall-time changes to a kernel.

## Decision and next target

Keep `GGML_HIP_FATTN_TILE_MAX_GQA=2` disabled by default. The measured long-KV synthetic CPU-reference errors are within tolerance, but the experimental path is slower on the matched attention workload and model-level correctness and fixed-work end-to-end speed remain unverified. Do not attribute report 0004's MTP throughput difference to a compute improvement.

Next inspect the long-sequence tile work distribution and K/V reuse in `ggml/src/ggml-cuda/fattn-tile.cuh` for the 256x256, GQA4 shapes; use the fixed-work attention test before proposing a scoped change, then validate against CPU reference and the restored three-device critical path. The timing alone does not establish whether bandwidth, occupancy, repeated K/V traffic, or another factor causes the slowdown.

Logs, measurements and the temporary instrumentation patch are under `/home/chris/workspace/llama-perf-results/iteration-0005/`. Production defaults, model weights, the original checkout and `build-mixed` were not changed. No GitHub Actions were run.
