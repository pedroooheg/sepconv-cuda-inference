| Resolucao | Comparacao | A x B | Latencia | t | p | VRAM de A / B | Saturada |
|---|---|---|---|---|---|---|---|
| 720p | Forma de execucao, FP32 | cpp x python | -24.1% | 222.9 | 1.9e-59 | 137.0% | nao |
| 720p | Forma de execucao, FP16 | cpp_fp16 x python_fp16 | -39.4% | 461.3 | 1.1e-78 | 84.7% | nao |
| 720p | Precisao, Python | python_fp16 x python | -24.8% | 220.2 | 3e-65 | 85.6% | nao |
| 720p | Precisao, C++ | cpp_fp16 x cpp | -40.0% | 545.6 | 1.7e-90 | 52.9% | nao |
| 720p | Combinado | cpp_fp16 x python | -54.4% | 528.4 | 5.9e-64 | 72.4% | nao |
| 1080p | Forma de execucao, FP32 | cpp x python | -24.6% | 200.4 | 9.5e-77 | 106.6% | nao |
| 1080p | Forma de execucao, FP16 | cpp_fp16 x python_fp16 | -39.8% | 225.1 | 2.4e-46 | 80.4% | nao |
| 1080p | Precisao, Python | python_fp16 x python | -25.1% | 161.4 | 4e-63 | 82.9% | nao |
| 1080p | Precisao, C++ | cpp_fp16 x cpp | -40.2% | 329.7 | 4.8e-53 | 62.5% | nao |
| 1080p | Combinado | cpp_fp16 x python | -54.9% | 639.1 | 5.2e-62 | 66.7% | nao |
| 1440p | Forma de execucao, FP32 | cpp x python | +96.1% | -1193.0 | 1 | 107.6% | sim |
| 1440p | Forma de execucao, FP16 | cpp_fp16 x python_fp16 | -39.5% | 701.3 | 4.9e-76 | 72.5% | nao |
| 1440p | Precisao, Python | python_fp16 x python | -59.5% | 858.5 | 6.8e-70 | 97.9% | sim |
| 1440p | Precisao, C++ | cpp_fp16 x cpp | -87.5% | 3659.2 | 1.3e-81 | 66.0% | sim |
| 1440p | Combinado | cpp_fp16 x python | -75.5% | 1130.4 | 5.6e-65 | 71.0% | sim |
| 2160p | Forma de execucao, FP32 | cpp x python | sem dados | - | - | - | - |
| 2160p | Forma de execucao, FP16 | cpp_fp16 x python_fp16 | -82.9% | 49.3 | 0.00021 | 99.5% | sim |
| 2160p | Precisao, Python | python_fp16 x python | -12.4% | 2.1 | 0.079 | 100.4% | sim |
| 2160p | Precisao, C++ | cpp_fp16 x cpp | sem dados | - | - | - | - |
| 2160p | Combinado | cpp_fp16 x python | -85.0% | 14.8 | 0.0023 | 100.0% | sim |
