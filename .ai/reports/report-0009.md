# Report 0009

Status: Partial. The checkout was fast-forwarded to `b7cec5024b88666c6919cf8fecd676ee65e9f800`. This execution environment exposes neither NVIDIA nor AMD GPU device nodes: `/dev/nvidiactl`, `/dev/kfd`, and `/dev/dri` are absent, and `nvidia-smi -L` cannot communicate with the driver. This establishes an access limitation here, not a hardware diagnosis. No full-model or small CUDA profiling was launched, and no `nvprof` command was run. The required small CUDA preflight must succeed before any expensive profiling.

The mixed CUDA/HIP build passed with `cmake --build build-perf -j 28 --target llama-server llama-bench test-backend-ops`. The host trace analyzer tests passed (4 tests), as did the ROCPD alignment tests (2 tests). These synthetic tests do not establish actual clock alignment or request-completion flushing. Neither the short MTP trace/output comparison nor the real per-step 32K/98K HIP/CUDA investigation could be run without the GPUs.

## Nsight importer recovery

`nsys --version` reports 2022.4.2.50-32196742v0. Contrary to the previous automatic-import error, `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` exists, reports the exact same version, and `ldd` reports no missing libraries. Directly invoking it on the prior `iteration-0008/cuda-prof/nsys-backend-smoke.qdstrm` produced `/tmp/iteration-0009-prior-smoke.nsys-rep`; `nsys export --type sqlite` successfully exported it to `/tmp/iteration-0009-prior-smoke.sqlite`. This demonstrates offline conversion, not a working new capture: the exported database has no CUDA activity kernel or memcpy tables. The reason automatic importer discovery failed remains unknown. These `/tmp` files are transient; the requested results directory outside the repository is read-only in this environment.

With device access restored, run a small CUDA kernel-and-copy workload under a 30-60 second deadline, convert the resulting stream using the matching importer directly if needed, and verify CUDA kernel and memcpy activity in the exported database. Only then consider a full-model capture. Long profiling runs need bounded startup and total deadlines, child liveness checks, health endpoint ownership verification, and process-group cleanup with graceful then forced termination and reaping.

## Remaining attribution

Report 0008's host phase medians remain the best available evidence: 32K to 98K changes were draft +2.023 ms, target processing +4.313 ms, target synchronization +8.277 ms, and speculative reprocessing +3.251 ms. These are host intervals, potentially overlapping, not an additive GPU critical-path decomposition. The earlier HIP CSVs lack paired host markers, and no new actual GPU timeline or calibrated clock alignment was obtained. The additional latency cannot yet be attributed to a particular device operation or dependency. Start the next source investigation at `tools/server/server-context.cpp:4761`, around `llama_synchronize(ctx_tgt)`, correlating real device and host timelines rather than removing the synchronization.

No production code or inference defaults were changed. The existing untracked `.env` was left untouched. No GitHub Actions were run.
