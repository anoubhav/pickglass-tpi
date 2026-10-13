# Frozen-counter invariance on the augmented crops

relative difference |pred - ref| / ref pooled over both axes; ref = the counter's prediction on the original (h/v swapped for k = 90/270); flip_% = crops whose larger per-axis difference exceeds 10 % (octave / stripe-vs-yarn jumps of a bistable estimate)

| subset | n_crops | n_sources | median_% | p90_% | within_5% | within_10% | median_h_% | median_v_% | median_of_source_medians_% | flip_% | weave_same_% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: no-op re-encode (150 tool crops) | 150 | 150 | 0.01 | 0.1 | 96.7 | 97.0 | 0.01 | 0.01 | 0.01 | 5.3 | 96.7 |
| baseline: pure rot90 + re-encode (150 x 3) | 450 | 150 | 0.04 | 0.68 | 97.2 | 97.4 | 0.04 | 0.04 | 0.05 | 4.4 | 94.4 |
| baseline: matched (source re-rectified at aug S, rotated by k) | 843 | 150 | 0.06 | 0.79 | 95.4 | 96.4 | 0.06 | 0.06 | 0.09 | 6.6 | 95.6 |
| AUG: all usable | 843 | 150 | 0.12 | 1.98 | 92.9 | 93.4 | 0.11 | 0.13 | 0.13 | 10.7 | 90.4 |
| AUG: all 853 incl. flagged | 853 | 150 | 0.12 | 2.25 | 92.3 | 92.9 | 0.12 | 0.13 | 0.13 | 11.3 | 90.3 |
| AUG: flagged only | 10 | 9 | 8.54 | 78.42 | 45.0 | 55.0 | 6.79 | 9.31 | 10.79 | 60.0 | 80.0 |
| AUG usable, source confidence >= median (0.83) | 426 | 75 | 0.11 | 0.71 | 98.6 | 98.7 | 0.1 | 0.13 | 0.12 | 1.6 | 96.5 |
| AUG usable, source confidence < median | 417 | 75 | 0.13 | 25.69 | 87.1 | 87.9 | 0.13 | 0.12 | 0.16 | 19.9 | 84.2 |
| AUG usable, k = 0 | 202 | 119 | 0.09 | 3.77 | 90.6 | 91.3 | 0.08 | 0.09 | 0.11 | 14.4 | 91.6 |
| AUG usable, k = 90 | 207 | 114 | 0.13 | 2.07 | 92.8 | 93.5 | 0.09 | 0.2 | 0.2 | 10.1 | 90.3 |
| AUG usable, k = 180 | 230 | 125 | 0.16 | 1.45 | 94.3 | 94.6 | 0.15 | 0.19 | 0.22 | 8.3 | 91.7 |
| AUG usable, k = 270 | 204 | 118 | 0.11 | 1.61 | 93.6 | 93.9 | 0.15 | 0.08 | 0.14 | 10.3 | 87.7 |
| AUG usable, Perspective yes | 582 | 150 | 0.12 | 2.66 | 91.9 | 92.4 | 0.12 | 0.13 | 0.14 | 11.9 | 90.9 |
| AUG usable, Perspective no | 261 | 129 | 0.12 | 1.22 | 95.0 | 95.4 | 0.1 | 0.13 | 0.18 | 8.0 | 89.3 |
| AUG usable, ShiftScale (no persp.) yes | 172 | 106 | 0.14 | 1.26 | 94.8 | 95.1 | 0.11 | 0.17 | 0.21 | 8.1 | 85.5 |
| AUG usable, ShiftScale (no persp.) no | 89 | 72 | 0.07 | 1.06 | 95.5 | 96.1 | 0.08 | 0.06 | 0.12 | 7.9 | 96.6 |
| AUG usable, BrightnessContrast yes | 591 | 150 | 0.14 | 2.32 | 91.8 | 92.4 | 0.14 | 0.15 | 0.18 | 11.5 | 89.0 |
| AUG usable, BrightnessContrast no | 245 | 129 | 0.07 | 0.87 | 95.5 | 95.7 | 0.07 | 0.07 | 0.11 | 8.6 | 93.9 |
| AUG usable, GaussNoise yes | 76 | 59 | 0.36 | 53.43 | 78.3 | 78.9 | 0.3 | 0.38 | 0.38 | 30.3 | 75.0 |
| AUG usable, GaussNoise no | 767 | 150 | 0.11 | 1.4 | 94.3 | 94.8 | 0.1 | 0.12 | 0.12 | 8.7 | 91.9 |
| AUG usable, JPEG q70-80 yes | 548 | 150 | 0.11 | 1.49 | 94.0 | 94.5 | 0.1 | 0.12 | 0.15 | 8.9 | 91.4 |
| AUG usable, JPEG q70-80 no | 219 | 121 | 0.1 | 1.13 | 95.2 | 95.4 | 0.1 | 0.11 | 0.17 | 8.2 | 93.2 |
| AUG usable, rect_size/source in [0,0.97) | 12 | 11 | 0.09 | 1.55 | 91.7 | 91.7 | 0.06 | 0.13 | 0.14 | 8.3 | 83.3 |
| AUG usable, rect_size/source in [0.97,1.03) | 234 | 125 | 0.12 | 1.19 | 95.5 | 95.7 | 0.11 | 0.13 | 0.17 | 7.7 | 90.2 |
| AUG usable, rect_size/source in [1.03,1.15) | 323 | 139 | 0.13 | 2.26 | 92.1 | 92.7 | 0.12 | 0.15 | 0.18 | 11.5 | 91.0 |
| AUG usable, rect_size/source in [1.15,9) | 274 | 133 | 0.11 | 3.13 | 91.6 | 92.2 | 0.11 | 0.11 | 0.18 | 12.4 | 90.1 |
| AUG usable, no noise / no BC / no perspective | 79 | 63 | 0.08 | 1.35 | 93.7 | 93.7 | 0.08 | 0.09 | 0.14 | 12.7 | 91.1 |
| AUG usable, excl. 14 counter-unstable sources | 768 | 136 | 0.11 | 1.02 | 96.4 | 96.5 | 0.11 | 0.12 | 0.12 | 5.7 | 92.2 |
| AUG usable, excl. unstable sources, no GaussNoise | 699 | 136 | 0.1 | 0.86 | 97.6 | 97.8 | 0.1 | 0.11 | 0.12 | 3.9 | 93.8 |
| baseline: matched, excl. unstable sources | 768 | 136 | 0.06 | 0.6 | 97.7 | 98.0 | 0.05 | 0.06 | 0.08 | 3.5 | 96.6 |
