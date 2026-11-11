# Report 0002

Status: Blocked on GPU access. Checked out handoff commit `f9b64194c7115beb2abab0602ba942d3a7f9d464` on `docs/llama-perf`; preserved the untracked `.env`, `build-mixed/`, and the original checkout.

The standalone collector test passed with `c++ -std=c++17 -Wall -Wextra -Werror -pthread -Iggml/src/ggml-cuda docs/LLAMA-PERF/tools/test_upload_profile.cpp`. A separate `build-perf/` was configured for dynamic mixed CUDA/HIP backends, CUDA architecture 89 and HIP architecture gfx1201; the instrumented `ggml-cuda.cu` compiled successfully as both CUDA and HIP objects. This is not a full build or runtime validation.

This sandbox has no `/dev/kfd`, `/dev/dri`, or `/dev/nvidia*`. `rocminfo` could not open `/dev/kfd`, and `nvidia-smi -L` could not communicate with the driver. Therefore no instrumented backend output, tensor attribution, controlled throughput comparison, placement check, or MTP run was possible. No transfer optimization was made; existing wait times cannot be assumed removable. No GitHub Actions were used.

Next priority: resume handoff 0002 in a session with access to both GPUs, complete the separate mixed build, and run the requested disabled/enabled short and 32K comparisons before deciding whether any upload optimization is safe. Previous measurements in report 0001 are not substitutes for these runs.
