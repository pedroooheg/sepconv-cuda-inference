| Conjunto | Implementacao | Pares | PSNR (dB) | SSIM |
|---|---|---|---|---|
| middlebury | cpp | 12 | 37.22 +/- 4.30 | 0.9680 +/- 0.0194 |
| middlebury | cpp_fp16 | 12 | 37.22 +/- 4.29 | 0.9679 +/- 0.0194 |
| middlebury | python | 12 | 37.22 +/- 4.29 | 0.9680 +/- 0.0194 |
| vimeo90k | cpp | 200 | 34.46 +/- 4.55 | 0.9590 +/- 0.0492 |
| vimeo90k | cpp_fp16 | 200 | 34.46 +/- 4.55 | 0.9590 +/- 0.0492 |
| vimeo90k | python | 200 | 34.46 +/- 4.55 | 0.9590 +/- 0.0492 |

Referencia: original/benchmark.py (Niklaus), que arredonda a saida para 8 bits,
no mesmo Middlebury: PSNR 37.28 dB, SSIM 0.9681. As implementacoes acima
truncam a saida (como o motor C++ e o run.py), o que explica os ~0,06 dB a menos.
