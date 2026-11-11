# Profiling plan - llama.cpp-perf

## Objective and fixed constraints

Use measurement to choose the first safe performance improvement for Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS at up to 196,608 tokens. Keep the proven three-GPU server layout and F16 KV cache intact for the primary comparison; do not assume that an alternative layout or Q8_0 KV is faster.

- **Separate repository:** `/home/chris/workspace/llama.cpp-perf`. Do not edit `/home/chris/workspace/llama.cpp`.
- **Reference build:** `build-mixed/`, source commit `1648d641174927634dfb72135b220e4052393729`. Preserve the binary and CMake cache.
- **Experimental build:** `build-perf/` only when source instrumentation is necessary.
- **Existing best server layout:** target on `CUDA0,ROCm0,ROCm1`; MTP draft on `CUDA0`; layer split `4,7,7`; F16 K/V; flash attention; 196,608 context; speculative MTP maximum 3; maximum generation 49,152.
- **Evidence:** [two uploaded target-only benchmark tables](BENCHMARK-RESULTS.md). MTP performance and GPU kernel timelines remain unmeasured.
- **Execution:** local workstation only. Do not configure, enable, dispatch, or run GitHub Actions.

## Stage 0 - Environment and source-of-truth capture

Goal: repeatability across profiling experiments.

~~~bash
cd /home/chris/workspace/llama.cpp-perf

export ROCM_PATH=/opt/rocm/core-10.1
export HIP_PLATFORM=amd
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/usr/local/cuda/bin:$PATH"

export PROFILE_DIR="/home/chris/workspace/llama-perf-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$PROFILE_DIR"/{environment,target-only,server,hip,cuda,analysis}

git rev-parse HEAD > "$PROFILE_DIR/environment/commit.txt"
git status --short > "$PROFILE_DIR/environment/git-status.txt"
cmake -LAH -N build-mixed > "$PROFILE_DIR/environment/cmake-options.txt"

./build-mixed/bin/llama-server --list-devices   > "$PROFILE_DIR/environment/devices.txt" 2>&1
nvidia-smi > "$PROFILE_DIR/environment/nvidia-smi.txt"
command -v rocprofv3 nsys amd-smi rocm-smi   > "$PROFILE_DIR/environment/profiler-paths.txt" || true
~~~

Also retain server startup logs showing actual CUDA/ROCm buffer allocations, layer distribution, GPU memory headroom and any host-resident tensors. Record the installed ROCm and CUDA versions, negotiated PCIe widths/speeds, GPU clocks, power states and thermal limits before comparing runs. The current benchmark files alone do not provide these facts.

**Control variables:** one active inference process at a time; no other GPU workloads; unchanged model files, source revision, CMake settings and request corpus. Warm up before timed runs. Record run order and repeat measurements, particularly at 163,840-token depth.

## Stage 1 - Target-only throughput baseline (captured)

See [BENCHMARK-RESULTS.md](BENCHMARK-RESULTS.md) and both raw files. The provided benchmark uses the special llama-bench slash-separated device and tensor-split syntax, unlike the comma-separated llama-server syntax.

To reproduce **run A**:

~~~bash
cd /home/chris/workspace/llama.cpp-perf

./build-mixed/bin/llama-bench   --model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf   --device CUDA0/ROCm0/ROCm1   --split-mode layer   --tensor-split 4/7/7   --flash-attn on   --load-mode mmap   --lazy-mode on   --fit-target 768   --cache-type-k f16   --cache-type-v f16   --n-prompt 256,1024,8192   --n-gen 256,512,2048   --n-depth 0,65536,98304,163840   --repetitions 3   --output md   > "$PROFILE_DIR/target-only/auto-fit.md"
~~~

To reproduce **run B**, remove `--fit-target 768` and use `--n-prompt 1024 --n-gen 512 --n-depth 0,98304,163840`. Capture stdout and stderr separately where possible.

**Interpretation constraints:**
- `llama-bench` uses random/synthetic input tokens and feeds random generation tokens; it does not exercise MTP or validate output logits.
- Context depth is prepared **outside** the measured interval. The reported `t/s` concerns the prompt or generation section, not the total time to prepare the depth.
- This version sizes each benchmark context based on prompt + generation + depth, unlike the fixed 196,608-token server configuration.
- `--fit-target 768` can override actual placement choices during fitting despite the printed requested split. Use startup placement logs and controlled no-fit tests when comparing split performance.
- Preserve the raw results and avoid conflating target-only `tg` with effective MTP server speed.

**Current status:** completed. The long-context slowdown and one high-variance deep-prompt result determine the next experiments.

## Stage 2 - Real MTP-enabled server measurements

Goal: measure throughput, latency, draft acceptance and verification efficiency under the actual production-style configuration.

Start the server from `build-mixed/` with all established parameters. This is the recorded workload configuration; `--perf` and `--metrics` are profiling-only additions.

~~~bash
cd /home/chris/workspace/llama.cpp-perf

./build-mixed/bin/llama-server   --model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf   --spec-draft-model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Joakimpalm-Zen/Qwen3.8-Flash-Next-MTP-Q8_0.gguf   --alias qwen3.8-flash-next-gsq-rco-iq3_xxs-192k   --host 127.0.0.1   --port 8081   --device CUDA0,ROCm0,ROCm1   --spec-draft-device CUDA0   --parallel 1   --jinja   --temp 1   --top-p 0.95   --top-k 20   --min-p 0   --reasoning-format auto   --reasoning auto   --reasoning-budget -1   --spec-type draft-mtp   --spec-draft-n-max 3   --flash-attn on   --split-mode layer   --tensor-split 4,7,7   --load-mode mmap   --lazy-mode on   --fit-target 768   --ctx-size 196608   --cache-type-k f16   --cache-type-v f16   --n-predict 49152   --perf   --metrics
~~~

Keep the server bound to localhost. Capture the server stdout/stderr to a dated file; do not run a second model process while measuring.

Capture counters before and after a controlled request set:

~~~bash
curl -fsS http://127.0.0.1:8081/metrics   > "$PROFILE_DIR/server/metrics-before.txt"

python3 tools/server/bench/speed-bench/speed_bench.py   --url 127.0.0.1:8081   --bench qualitative   --category coding,reasoning   --osl 512   --concurrency 1   --limit 3   --extra-inputs '{"temperature":1,"top_p":0.95}'   --output "$PROFILE_DIR/server/mtp-baseline.json"

curl -fsS http://127.0.0.1:8081/metrics   > "$PROFILE_DIR/server/metrics-after.txt"
~~~

The included SPEED-Bench client requires its Python dependencies and may download its Hugging Face dataset. Its results include per-request latency, prompt and generation tokens/s, and draft counts. If its workload differs from real usage, create and save a fixed local prompt corpus for coding, reasoning and structured long-context tasks.

Compute changes in `spec_decode_num_draft_tokens_total`, `spec_decode_num_accepted_tokens_total` and `spec_decode_num_drafts_total`, not the cumulative values alone:

- Draft acceptance = accepted draft tokens / drafted tokens.
- Mean effective verification length = 1 + accepted draft tokens / verification steps.
- Effective throughput = emitted completion tokens / total generation elapsed time; record time to first token separately.

Repeat representative requests with MTP disabled as an **isolated diagnostic**, but record changed VRAM placement: removing the draft may change available target-model memory and confound direct comparison. Keep the three-GPU target layout, sampling and context unchanged.

Use the same saved prompts at actual token lengths near 4K, 32K, 98K and 160K. Record actual tokenization, prompt-cache reuse and output length; synthetic depth is not identical to real long-context prompts. Compare effective throughput at equal output and context lengths.

**Exit condition:** a reproducible MTP-enabled latency/acceptance baseline, plus enough data to identify whether draft, verification or target-model work dominates.

## Stage 3 - HIP and NVIDIA CUDA GPU timelines

Goal: identify dominant kernels and transfers without modifying source.

First inspect tool availability and supported flags:

~~~bash
command -v rocprofv3
command -v nsys
rocprofv3 --help
nsys profile --help
~~~

Run AMD and NVIDIA profilers in **separate passes**. Start with small `llama-bench` shapes (`pp1024` and `tg128` at depth 0), then one controlled longer-depth case. Do not collect a full 160K-depth trace until the output size and overhead are understood.

Example AMD trace (verify flags against the installed ROCprofiler-SDK):

~~~bash
rocprofv3   --runtime-trace   --output-format rocpd   --output-directory "$PROFILE_DIR/hip"   -- ./build-mixed/bin/llama-bench      --model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf      --device CUDA0/ROCm0/ROCm1      --split-mode layer      --tensor-split 4/7/7      --flash-attn on      --load-mode mmap      --lazy-mode on      --cache-type-k f16      --cache-type-v f16      --n-prompt 1024      --n-gen 128      --n-depth 0      --repetitions 1      --output md   > "$PROFILE_DIR/hip/short-run.md"   2> "$PROFILE_DIR/hip/short-run.log"
~~~

Example NVIDIA trace using the identical short-run test (run separately):

~~~bash
nsys profile   --trace=cuda,nvtx,osrt   --sample=none   --cpuctxsw=none   --output "$PROFILE_DIR/cuda/short-run"   ./build-mixed/bin/llama-bench     --model /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf     --device CUDA0/ROCm0/ROCm1     --split-mode layer     --tensor-split 4/7/7     --flash-attn on     --load-mode mmap     --lazy-mode on     --cache-type-k f16     --cache-type-v f16     --n-prompt 1024     --n-gen 128     --n-depth 0     --repetitions 1     --output md   > "$PROFILE_DIR/cuda/short-run.md"   2> "$PROFILE_DIR/cuda/short-run.log"
~~~

Collect:
- Per-device kernel-name frequency, GPU execution duration and launch count.
- DMA/memcpy calls and transfer sizes, copy direction and blocked host threads.
- GPU idle gaps and serial handoffs between layer groups.
- Time spent in quantized matmul, QSA indexer, top-k selection, attention and launch/synchronization APIs.

Do not naively sum overlapping kernels across streams or different GPUs. CPU API timestamps are not GPU-execution durations; profile the end-to-end critical path. Tool tracing changes timing, so compare unprofiled runs for true throughput claims.

**Exit condition:** an evidence-based list of the highest-cost kernel families and stalls at both short and long depth.

## Stage 4 - Minimal source instrumentation, only if needed

Use a dedicated `build-perf/` configuration and feature-gated aggregated timers. Preserve `build-mixed/` and do not alter model math or quantization in the profiling phase.

| Area | Existing source | Measure |
| --- | --- | --- |
| IQ3_XXS expert kernels | `ggml/src/ggml-cuda/mmq.cuh`, `mmq.cu` | shape, kernel choice, execution cost, call counts |
| QSA scoring + selection | `src/models/qwen4exp.cpp`, `build_qsa_sel()` | lightning indexer, top-k, mask construction and selected attention |
| GPU graph execution | `src/llama-context.cpp` | graph split/reallocation counts, compute intervals |
| Backend handoffs | `ggml/src/ggml-backend.cpp` | split copies, event waits, fallbacks to blocking synchronization |
| MTP acceptance | `tools/server/server-context.cpp`, `tools/server/server-task.cpp` | drafting, verification, accepted-token counters and wall time |

Use device profilers or nonblocking event measurement for GPU execution time. **Do not add forced `hipDeviceSynchronize()`, `cudaDeviceSynchronize()` or per-node blocking calls just to measure duration**: doing so changes the overlap under investigation. Limit per-token logging and record all tracing overhead.

An additional focused case should investigate the unusually high fixed-split prefill variance at 163,840 depth: check actual device placement, disk page faults, cold/warm model state, PCIe traffic and clocks.

**Exit condition:** quantitative attribution of the top one or two bottlenecks, with instrumentation disabled by default and verified not to alter output validity.

## Stage 5 - Decision report and correctness gate

Create a concise report containing:

1. Hardware, source revision, full runtime settings and effective weight placement.
2. Target-only prefill/decode curves by depth (already captured).
3. MTP-enabled request metrics: TTFT, end-to-end tokens/s, draft acceptance and verification length.
4. Ranked HIP/CUDA kernel durations and relevant cross-device transfer/synchronization stalls.
5. Variability, uncertainties and profiler overhead.
6. An optimization proposal targeting **one** bottleneck, with a rollback path and acceptance criteria.

A proposed optimization must be compared on identical real prompts and controlled settings, must increase **end-to-end** throughput beyond measurement variability, and must pass finite-logit checks, token-generation correctness tests, long-context repeatability and backend numerical-parity tests. No speedup is accepted solely from a kernel microbenchmark.

**Current next action:** complete Stage 2, then use short Stage 3 traces to choose between QSA, expert kernels and mixed-backend scheduling. No GPU kernel modifications are authorized or required by this profiling plan.
