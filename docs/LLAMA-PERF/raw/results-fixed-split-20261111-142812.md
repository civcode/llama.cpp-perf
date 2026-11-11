| model                          |       size |     params | backend    | ngl |  fa | dev          | ts           |         lm | lazy_mode  |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | ------------ | ------------ | ---------: | ---------- | --------------: | -------------------: |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         |          pp1024 |       2321.71 ± 9.98 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         |           tg512 |         46.44 ± 0.02 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         | pp1024 @ d98304 |      1062.28 ± 29.33 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         |  tg512 @ d98304 |         36.79 ± 0.15 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         | pp1024 @ d163840 |      721.34 ± 138.47 |
| qwen4exp A3B IQ3_XXS - 3.0625 bpw |  70.62 GiB |   176.94 B | CUDA,ROCm  |  -1 |   1 | CUDA0/ROCm0/ROCm1 | 4.00/7.00/7.00 |       mmap | on         | tg512 @ d163840 |         32.58 ± 0.00 |

build: 1648d6411 (12296)
