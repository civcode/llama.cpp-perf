# Report 0034 — offline layer-11 FP64 headwise precision

**Status: Completed.** Starting commit `f1d005979d8ad67f5a1b21ed69fded4b94a4e9cf`. The new CPU-only analyzer passed 16 tests; FP64 replay passed seven; compileall and all 215 focused QSA tests passed. No fixes were needed. No GPU execution, telemetry, hardware/driver changes, stress testing or GitHub Actions occurred. Numerical equivalence is **NOT_ESTABLISHED**; production promotion remains **NO-GO** and sparse remains disabled by default.

## Captures and aggregate

Reused the actual 0033 first HIP0 layer-11 Q4 captures: separate dense A/B, sparse-all and sparse-only11. The analyzer required byte-identical Q/K/V spans and F16 masks in each triple, matching metadata and byte-identical dense A/B F32 output before computing the mathematical FP64 reference. Sparse-all and only11 layer-11 output files also matched each other byte-for-byte. The separately captured except11 negative control was byte-identical to dense at layer 11; it is **dense fallback**, not a sparse launch.

Dense reference-energy NMSE was **1.18147e-6**, sparse **3.68239e-7**. Sparse had lower squared error in **22/24 heads** and **85/96 query-head pairs**. By query position 0–3, sparse won 19/24, 21/24, 23/24 and 22/24 respectively; there were no ties. The 96 measurements share one attention operation and are not independent trials. Improvement is widespread, but not evenly distributed: heads 14 and 17 account for **37.7%** of the net squared-error reduction, and the top five improving heads for **64.0%**. This is descriptive for one captured operation, not evidence that a sparse result is always more accurate.

## Per-head comparison

Each row aggregates all four queries. NMSE values below are in units of **1e-7**; max-absolute errors are the maximum across that head's four queries and 256 features. Lower is closer to FP64. Squared-error winners and max-error winners need not coincide.

| Head | Dense NMSE | Sparse NMSE | Dense max abs | Sparse max abs |
|---:|---:|---:|---:|---:|
|0|5.38|3.65|0.002976|0.004109|
|1|4.01|3.61|0.002011|0.002248|
|2|7.81|4.07|0.002708|0.002326|
|3|5.21|2.77|0.001841|0.001671|
|4|11.2|2.72|0.002317|0.001936|
|5|0.702|0.618|0.001000|0.000780|
|6|20.5|5.09|0.002683|0.002334|
|7|4.15|2.34|0.001824|0.001520|
|8|1.75|1.76|0.001363|0.001504|
|9|4.51|4.65|0.003005|0.003411|
|10|5.71|4.73|0.003329|0.001639|
|11|2.07|1.29|0.001401|0.001109|
|12|2.23|1.24|0.001438|0.001279|
|13|21.2|7.39|0.008873|0.003455|
|14|36.0|3.56|0.012040|0.002481|
|15|6.76|2.41|0.005063|0.001625|
|16|10.8|3.75|0.004844|0.002258|
|17|23.9|7.11|0.016021|0.004399|
|18|8.45|3.39|0.012063|0.002048|
|19|5.05|1.47|0.005981|0.001933|
|20|14.9|2.76|0.003760|0.001712|
|21|11.4|3.44|0.003139|0.002288|
|22|16.0|1.18|0.003478|0.001052|
|23|14.1|6.49|0.008427|0.002578|

Heads **8 and 9** are the two aggregate dense winners. At query/head granularity the 11 dense winners are q0 heads 1,8,9,12,18; q1 heads 1,8,10; q2 head 0; q3 heads 5,10. The largest sparse NMSE ratios among these are q1/head10 **1.66×** (maximum absolute errors dense 0.001351, sparse 0.001639) and q3/head5 **1.63×** (0.000488 versus 0.000468). These ratios are not release criteria. Across all 96 pairs sparse has lower maximum absolute error in 77, dense in 19. Full precision per-query/head squared error, reference energy and max error are external in `iteration-0034/`.

## Arithmetic inspection and next experiment

In `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the RDNA WMMA generic 256-dimensional tile uses F32 KQ accumulation but half2 VKQ accumulation; FP32 online softmax scores/rowsums are rescaled and the VKQ accumulator is rescaled in half2. Softmax weights are converted to half2 before the value-product operation, and final tile combination has its own rounding/order. Sparse gathering also changes the number and grouping of processed rows. These are plausible sources of changed error, but the captures do **not** isolate any one rounding mechanism.

A reasonable **future, diagnostic-only** hypothesis test is an opt-in sparse-GQA12/RDNA 256×256 variant using FP32 VKQ accumulation and rescaling while holding selected indices, masks, weight conversion, softmax and the rest of dispatch constant. It must fail closed or fall back to the existing path for unsupported layouts, shapes or devices, without changing the default or overflow protections. Because the WMMA tile/layout and final combine must support the altered accumulator, feasibility needs source/build validation rather than assuming a type substitution works. In a separately authorized run, require mixed-build tests and actual dispatch tracing, identical-input per-head/per-query FP64 errors including worst heads and maximum absolute errors, plus matched dense-repeat controls and natural-history layer-15/GDN/logit comparisons. A consistent accuracy improvement without new outliers would support further investigation; worse or mixed headwise error, changed selection due to an unintended layout, or unchanged downstream drift would refute the expected benefit or limit its relevance. FP64 closeness is not a product tolerance, and matching dense is not automatically the more accurate outcome.
