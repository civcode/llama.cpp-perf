# Benchmark results - Qwen3.8-Flash-Next IQ3_XXS

**Captured:** 2026-11-11. **Status:** completed target-only llama-bench runs; no MTP server benchmark, GPU kernel trace, or finite-logit regression test included yet.

## Raw data and provenance

- [Auto-fit raw output](raw/results-20261111-140511.md) - originally named `results-20261111-140511.md`.
- [Fixed-split raw output](raw/results-fixed-split-20261111-142812.md) - originally named `results-fixed-split-20261111-142812.md`.
- Both files report `build: 1648d6411 (12296)`, matching the fork's full source commit `1648d641174927634dfb72135b220e4052393729`.
- Reported model metadata: `qwen4exp A3B IQ3_XXS - 3.0625 bpw`, **70.62 GiB** size, **176.94 B** parameters, `CUDA,ROCm` backend.
- Hardware: Ryzen 9 9950X, 64 GiB DDR5-5600; RTX 4070 Ti Super (16 GiB, OCuLink PCIe 4.0 x4); two Radeon AI PRO R9700 (32 GiB each, PCIe 5.0 x8); ASUS ProArt X870E-Creator WiFi, BIOS revision 2503.
- Common arguments: device `CUDA0/ROCm0/ROCm1`, split mode `layer`, requested tensor split `4/7/7`, flash attention on, mmap, lazy mode on, F16 K/V cache.
- Both runs used 3 repetitions per test. Values below are the original **mean ± reported standard deviation**, in tokens per second.

### Benchmark semantics and limitations

`ppN` measures target-model prompt processing for N tokens; `tgN` measures target-model token-by-token generation for N steps. The suffix `@ dN` means an N-token context depth is prepared before the measured portion. Depth preparation is **not** included in the reported tokens/s.

The benchmark uses synthetic/random tokens and does not exercise MTP draft or target verification, real chat request latency, token sampling quality, or NaN-logit checks. Unlike the server's fixed 196,608-token context, `llama-bench` initializes a context sized for each test's required prompt + generation + depth. Consequently, these are controlled **target-model measurements**, not directly the effective MTP-enabled server speed.

## Run A - automatic fitting (`--fit-target 768`)

Original file: [results-20261111-140511.md](raw/results-20261111-140511.md)

Requested arguments:
- Prompt lengths: 256, 1,024, 8,192
- Generation lengths: 256, 512, 2,048
- Depth: 0, 65,536, 98,304, 163,840
- Fit margin: 768 MiB per device
- Repetitions: 3
- Number of unique test configurations: **24**; 72 timed repetitions, excluding warmups and depth preparation

| Depth (tokens) | pp256 (t/s) | pp1024 (t/s) | pp8192 (t/s) | tg256 (t/s) | tg512 (t/s) | tg2048 (t/s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1313.56 ± 38.81 | 2289.98 ± 17.45 | 3491.91 ± 9.01 | 46.22 ± 0.08 | 46.09 ± 0.13 | 45.86 ± 0.01 |
| 65,536 | 747.04 ± 3.47 | 1283.61 ± 3.81 | 1928.37 ± 0.48 | 38.73 ± 0.71 | 38.91 ± 0.44 | 39.09 ± 0.09 |
| 98,304 | 616.84 ± 5.39 | 1062.68 ± 1.20 | 1581.61 ± 0.76 | 36.58 ± 0.12 | 36.53 ± 0.18 | 36.49 ± 0.07 |
| 163,840 | 463.33 ± 6.43 | 794.88 ± 1.76 | 1166.77 ± 1.82 | 32.41 ± 0.01 | 32.46 ± 0.01 | 32.37 ± 0.02 |

### Context-depth effects

For the consistent `pp1024` shape, throughput changes from **2289.98 t/s** at depth zero to **794.88 t/s** at depth 163,840 (**-65.3%**).

For `tg512`, throughput changes from **46.09 t/s** to **32.46 t/s** (**-29.6%**).

The strongest observation is that the slowdown varies with context depth. It does **not**, by itself, establish whether QSA, attention, recurrent-state processing, memory traffic, graph scheduling, or multi-GPU transfers dominate.

## Run B - no explicit fit-target (requested fixed split)

Original file: [results-fixed-split-20261111-142812.md](raw/results-fixed-split-20261111-142812.md)

Requested arguments:
- Prompt length: 1,024
- Generation length: 512
- Depth: 0, 98,304, 163,840
- No `--fit-target` argument
- Requested tensor split: `4/7/7`
- Repetitions: 3
- Number of unique test configurations: **6**; 18 timed repetitions, excluding warmups and depth preparation

| Depth (tokens) | pp1024 (t/s) | tg512 (t/s) |
| ---: | ---: | ---: |
| 0 | 2321.71 ± 9.98 | 46.44 ± 0.02 |
| 98,304 | 1062.28 ± 29.33 | 36.79 ± 0.15 |
| 163,840 | 721.34 ± 138.47 | 32.58 ± 0.00 |

### Comparison where test shapes overlap

| Depth | Auto-fit pp1024 | Fixed-split pp1024 | Auto-fit tg512 | Fixed-split tg512 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2289.98 ± 17.45 | 2321.71 ± 9.98 | 46.09 ± 0.13 | 46.44 ± 0.02 |
| 98,304 | 1062.68 ± 1.20 | 1062.28 ± 29.33 | 36.53 ± 0.18 | 36.79 ± 0.15 |
| 163,840 | 794.88 ± 1.76 | 721.34 ± 138.47 | 32.46 ± 0.01 | 32.58 ± 0.00 |

**Caution about split equivalence:** the printed `ts=4.00/7.00/7.00` records the requested split. In this build, `--fit-target` invokes the fitting process, which can derive new placement and fitting parameters rather than using those exact proportions. Neither raw file contains an effective placement log, so the run-A and run-B numbers must not be interpreted as a controlled proof of a particular GPU split's effect.

**Variance warning:** the fixed-split `pp1024 @ d163840` result is **721.34 ± 138.47 t/s** (reported standard deviation approximately **19.2%** of the mean). That is too variable for a confident performance comparison with auto-fit's **794.88 ± 1.76 t/s**. Rerun that shape and inspect memory residency, thermal/clock behavior, page faults, and actual placement.

At the same deep context, the target-only `tg512` values are much closer: 32.46 t/s auto-fit versus 32.58 t/s without explicit fit-target. Such small differences are not sufficient to attribute a win to one configuration without a repeatable controlled comparison.

## Initial profiling hypotheses (not measured diagnoses)

1. **Long-context QSA / attention:** inspect `ggml_lightning_indexer`, `ggml_top_k`, top-k masking and the selected attention path.
2. **Multi-GPU scheduling:** inspect CUDA-to-HIP handoffs, buffer copies, asynchronous-event waits and forced synchronizations, especially where compute gaps grow with depth.
3. **Quantized experts:** identify the contribution of IQ3_XXS expert GEMV/GEMM kernels to decode/prefill.
4. **MTP effectiveness:** measure acceptance, verification steps and draft overhead using a real server run; the present tables cannot answer this.
5. **Variability:** explain the high 163,840-depth fixed-split prompt-processing standard deviation before comparing small tuning changes.

No GPU timeline, operator-level breakdown or evidence of the underlying cause has yet been collected.

## Follow-on measurements required

- Run the exact existing server configuration with MTP enabled and record request timing plus `/metrics` deltas.
- Obtain short HIP and NVIDIA CUDA traces for `pp1024` and `tg512` at low and long context where feasible.
- Record actual weight placement, VRAM allocations, clocks and relevant PCIe transfer/synchronization behavior.
- Test finite logits, recurrent-state stability and valid token output separately; `llama-bench` is not a numerical-correctness test.
- Re-run the deep fixed-split prompt-processing case until its variance is understood.

See [PROFILING-PLAN.md](PROFILING-PLAN.md) for the executable sequence and acceptance criteria.
