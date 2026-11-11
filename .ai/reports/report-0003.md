# Report 0003

Status: Partial. Matched MTP baselines and phase-scoped HIP traces identify flash attention as the leading incremental GPU-work hypothesis, but the critical path and causality remain unproven. No optimization was applied. Source revision: `e954b61c2bd53a8ef4302305e03e72cb73d2e144`.

## Controlled baseline

Used the saved 32K and 98K requests and the `build-perf` server, IQ3_XXS target and Q8_0 MTP draft, CUDA0/ROCm0/ROCm1 target, CUDA0 draft, 4/7/7 layer split, 196608 context, F16 KV, fit target 768, one slot, FA on, MTP n-max 3, seed 42 and the requested sampling. Both requests were uncached and generated 512 tokens. Metrics are read after each request; verification counts are differences of `llamacpp:spec_decode_num_drafts_total`.

| Prompt tokens | Generation ms | t/s | Accepted / drafted | Verification steps | Generation ms / step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32809 | 8714.32 | 58.64 | 306 / 611 | 204 | 42.72 |
| 98275 | 12784.96 | 39.97 | 307 / 609 | 203 | 62.98 |

These per-step figures are whole generation time divided by verification steps, not measured GPU kernel latency. A separate matched run gave 58.81 and 39.77 t/s. The full server command, logs, requests (in the prior results directory), metrics and responses are retained outside Git. The current server log does not provide detailed per-tensor placement: report 0002's separate placement check is not evidence that these particular server runs placed every tensor identically or that all weights were GPU resident.

## Profiling and comparator

The comparator's three synthetic tests passed and it read both existing real databases. Its automatic 32K target-only window includes depth preparation, as shown by large prefill kernels; it must not be interpreted as decode. The server MTP traces have almost no 993280-byte GPU-to-host copies; the auto-window assumption is inappropriate here. A common 3973120-byte GPU-to-host copy equals four full FP32 vocab logits and occurs approximately once per verification in these captures, but its identity and exact correspondence were not independently instrumented. The figures below are per observed copy, not measured per verification step.

ROCprofiler was launched with delayed collection. The 32K window is seconds 0.5-8.5 from the first collected kernel; the 98K window is seconds 3-14. Per-second kernel inspection showed prefill-style `flash_attn_ext_f16` in the first two seconds of the 98K capture; the selected windows exclude that region and show sustained `flash_attn_tile` decode. Boundary and phase assumptions remain approximate. Profiled whole-request throughput was 52.74 and 36.85 t/s versus the unprofiled 58.64 and 39.97 t/s, so tracing materially perturbs execution.

| Kernel family | 32K ms / observed copy | 98K ms / observed copy | Delta |
| --- | ---: | ---: | ---: |
| Flash attention | 3.589 | 10.394 | +6.805 |
| Other | 3.084 | 3.400 | +0.316 |
| QSA lightning indexer | 0.219 | 0.498 | +0.279 |
| Other top-k | 0.688 | 0.807 | +0.119 |
| QSA top-k mask | 0.046 | 0.052 | +0.006 |
| Recurrent GDN/SSM | 0.397 | 0.399 | +0.002 |
| Floating matmul | 3.885 | 3.877 | -0.008 |
| Quantized matmul | 11.336 | 11.260 | -0.076 |

Values sum GPU kernel durations across the two Radeon devices and can overlap; they cannot be subtracted from, or directly assigned as a share of, the 20.26 ms/step unprofiled elapsed difference. In the selected 8 s and 11 s windows, GPU 1/2 kernel busy unions were respectively 1.915/2.030 s and 2.467/2.520 s. HIP stream synchronization summed host API durations were 5.259 and 6.359 s; these may wait for other work and are not additive GPU time or proof of removable waits. The observed copy counts were 169 and 162. ROCprofiler does not capture NVIDIA execution, and no CUDA timeline was acquired. Nsight Systems reported CPU profiling unavailable with the existing security policy; no security policy was changed.

The full leading decode kernel template is `flash_attn_tile<256, 256, 4, 4, false>` on both Radeons. The narrowest code starting point is `ggml/src/ggml-cuda/fattn-tile.cuh`, especially the tile implementation and its launch configuration; inspect sequence-length scaling, tile occupancy and masking for these exact shapes before proposing a reversible alternative. This is a workload correlation, not a causal A/B or demonstrated safe optimization. A targeted configuration A/B with identical prompt, placement and output/state checks, and a CUDA+HIP dependency timeline or genuine per-step phase markers, are still required. Do not treat raw sync durations as evidence for removing synchronization.

ROCprofiler attach initially failed because attach support was not enabled; enabling `ROCP_TOOL_ATTACH=1` then failed with `PTRACE_SEIZE` permission denied. Delayed launch tracing provided the evidence above without changing system policy. Original checkout, model files and `build-mixed` were preserved. No build was needed and no GitHub Actions were used.

Raw data: `/home/chris/workspace/llama-perf-results/iteration-0003/`; saved requests: `/home/chris/workspace/llama-perf-results/20261111-143704/server/`. The JSON window summaries and full profiler databases are in the iteration directory. No profiler database or request content was committed.
