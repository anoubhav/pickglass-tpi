### Legacy-13 (axis-matched, 26 counts; reported once per final variant)

| method | MAE | MAPE [95% CI] | n_w5 | n_w10 | bias | MAPE sorted | vs CNN-synth: dMAPE [CI]; Wilcoxon | vs ours: dMAPE [CI]; Wilcoxon |
|---|---|---|---|---|---|---|---|---|
| CNN, synthetic only (zero-shot) | 10.32 | 16.0 [10.8, 21.8] | 4 | 10 | +5.10 | 14.4 | -- | -- |
| Ours (PG-Lattice: lattice + fusion) | 6.99 | 11.6 [3.1, 23.1] | 19 | 21 | +2.35 | 10.9 | -- | -- |
| V1 synthetic, existing aug (GPU re-run), ResNet-18 | 10.34 | 16.9 [11.0, 23.6] | 4 | 11 | +3.59 | 15.6 | +0.9 [-1.9, +3.7]; p_W=0.784 (Holm 1.00) | +5.3 [-5.0, +14.3]; p_W=0.003 (Holm 0.03) |
| V2 synthetic + whitelist aug, ResNet-18 | 8.40 | 14.0 [8.2, 22.4] | 6 | 16 | +1.80 | 13.1 | -2.0 [-5.7, +2.2]; p_W=0.084 (Holm 0.67) | +2.4 [-8.5, +12.5]; p_W=0.009 (Holm 0.07) |
| V3 pseudo-label self-training (real only), ResNet-18 | 14.35 | 20.7 [13.6, 28.5] | 7 | 10 | +8.65 | 19.6 | +4.7 [-2.7, +12.0]; p_W=0.165 (Holm 0.83) | +9.1 [-1.5, +19.5]; p_W=0.005 (Holm 0.04) |
| V4 synthetic -> pseudo-real fine-tune, ResNet-18 | 8.43 | 12.1 [8.2, 16.3] | 8 | 14 | +2.14 | 11.0 | -3.9 [-9.1, +1.1]; p_W=0.094 (Holm 0.67) | +0.5 [-8.5, +7.6]; p_W=0.019 (Holm 0.13) |
| V5 = V4 + TTA, ResNet-18 | 8.41 | 11.9 [8.1, 16.2] | 10 | 15 | +2.57 | 11.2 | -4.1 [-8.7, +0.6]; p_W=0.105 (Holm 0.67) | +0.3 [-8.6, +7.3]; p_W=0.071 (Holm 0.21) |
| V1 synthetic, existing aug (GPU re-run), EfficientNet-B0 | 11.70 | 21.7 [7.4, 45.3] | 8 | 16 | +6.69 | 20.0 | +5.6 [-6.3, +24.8]; p_W=0.291 (Holm 1.00) | +10.0 [-7.0, +34.7]; p_W=0.049 (Holm 0.20) |
| V2 synthetic + whitelist aug, EfficientNet-B0 | 9.53 | 16.5 [8.5, 28.5] | 8 | 14 | +4.98 | 15.2 | +0.5 [-5.0, +7.7]; p_W=0.499 (Holm 1.00) | +4.9 [-7.1, +17.7]; p_W=0.027 (Holm 0.16) |
| V3 pseudo-label self-training (real only), EfficientNet-B0 | 11.34 | 17.1 [12.7, 22.4] | 2 | 8 | +5.42 | 16.1 | +1.1 [-6.7, +9.2]; p_W=1.000 (Holm 1.00) | +5.5 [-4.1, +14.3]; p_W=0.029 (Holm 0.16) |
| V4 synthetic -> pseudo-real fine-tune, EfficientNet-B0 | 7.17 | 10.7 [7.0, 14.9] | 11 | 16 | +0.66 | 9.1 | -5.3 [-9.3, -1.7]; p_W=0.049 (Holm 0.44) | -0.9 [-10.2, +6.3]; p_W=0.084 (Holm 0.21) |
| V5 = V4 + TTA, EfficientNet-B0 | 7.11 | 10.4 [7.0, 14.5] | 10 | 16 | +0.69 | 9.0 | -5.6 [-9.6, -2.0]; p_W=0.043 (Holm 0.43) | -1.2 [-10.3, +5.8]; p_W=0.143 (Holm 0.21) |

### Selection metrics (synth_val + held-out pseudo-real; the only basis for every choice)

| variant | run | step | synth_MAPE | synth_acc10 | synth_logratio_r | pseudo_MAPE | pseudo_acc10 | pseudo_bias | score |
|---|---|---|---|---|---|---|---|---|---|
| V1_r18 | V1_r18 | 4750 | 6.96 | 78.83 | 0.90 | 62.14 | 30.00 | 37.96 | 34.55 |
| V2_r18 | V2O_r18 | 4750 | 7.92 | 75.50 | 0.88 | 63.01 | 20.00 | 35.95 | 35.46 |
| V3_r18 | V3O_r18 | 1500 | 24.76 | 31.33 | 0.49 | 49.50 | 35.00 | 17.50 | 37.13 |
| V4_r18 | V4_r18_rho1 | 1750 | 15.80 | 43.67 | 0.76 | 40.71 | 35.00 | 14.12 | 28.26 |
| V5_r18 | V4_r18_rho1 | 1750 | 15.80 | 43.67 | 0.76 | 40.71 | 35.00 | 14.12 | 28.26 |
| V1_b0 | V1_b0 | 4750 | 5.17 | 88.00 | 0.93 | 81.84 | 25.00 | 44.44 | 43.51 |
| V2_b0 | V2E_b0 | 4750 | 5.01 | 88.33 | 0.94 | 57.92 | 30.00 | 35.23 | 31.46 |
| V3_b0 | V3O_b0 | 750 | 25.17 | 28.83 | 0.41 | 40.72 | 35.00 | 12.29 | 32.94 |
| V4_b0 | V4_b0_rho05 | 750 | 6.13 | 84.83 | 0.90 | 41.22 | 45.00 | 18.88 | 23.68 |
| V5_b0 | V4_b0_rho05 | 750 | 6.13 | 84.83 | 0.90 | 41.22 | 45.00 | 18.88 | 23.68 |

Candidates compared by the pre-declared score:

| variant | candidate | step | synth_MAPE | pseudo_MAPE | score | chosen |
|---|---|---|---|---|---|---|
| V2_r18 | V2A_r18 | 4750 | 7.26 | 71.78 | 39.52 | False |
| V2_r18 | V2E_r18 | 4750 | 6.84 | 69.33 | 38.09 | False |
| V2_r18 | V2O_r18 | 4750 | 7.91 | 63.01 | 35.46 | True |
| V3_r18 | V3E_r18 | 1500 | 27.52 | 52.83 | 40.17 | False |
| V3_r18 | V3O_r18 | 1500 | 24.76 | 49.50 | 37.13 | True |
| V4_r18 | V4_r18_rho1 | 1750 | 15.80 | 40.71 | 28.26 | True |
| V4_r18 | V4_r18_rho05 | 1750 | 8.38 | 49.96 | 29.17 | False |
| V2_b0 | V2A_b0 | 4750 | 5.20 | 67.50 | 36.35 | False |
| V2_b0 | V2E_b0 | 4750 | 5.01 | 57.92 | 31.46 | True |
| V2_b0 | V2O_b0 | 4750 | 5.23 | 74.97 | 40.10 | False |
| V3_b0 | V3E_b0 | 2750 | 27.35 | 42.09 | 34.72 | False |
| V3_b0 | V3O_b0 | 750 | 25.17 | 40.72 | 32.94 | True |
| V4_b0 | V4_b0_rho1 | 250 | 12.11 | 36.00 | 24.06 | False |
| V4_b0 | V4_b0_rho05 | 750 | 6.13 | 41.22 | 23.68 | True |

### synth_val_v2 (300, exact labels, not used for selection)

| variant | MAPE | acc10 | MAPE_single_scale |
|---|---|---|---|
| V1_r18 | 6.11 | 84.8 | 6.66 |
| V2_r18 | 6.48 | 83.5 | 6.99 |
| V3_r18 | 23.45 | 35.7 | 23.85 |
| V4_r18 | 15.33 | 52.7 | 15.69 |
| V5_r18 | 15.33 | 52.7 | 15.69 |
| V1_b0 | 4.28 | 90.3 | 4.56 |
| V2_b0 | 4.27 | 89.7 | 4.50 |
| V3_b0 | 24.41 | 32.5 | 24.35 |
| V4_b0 | 5.13 | 86.5 | 5.48 |
| V5_b0 | 5.13 | 86.5 | 5.48 |
| REF_mps_r18 | 6.43 | 82.3 | 6.97 |

### Confirmation bias, synthetic (student vs teacher where the teacher is unreliable)

| variant | subset | n_images | MAPE_student | MAPE_ours | dMAPE | dMAPE_lo | dMAPE_hi | p_wilcoxon | student_gross_share |
|---|---|---|---|---|---|---|---|---|---|
| V1_r18 | ours accepted (rule) | 172 | 4.98 | 2.23 | 2.74 | 1.79 | 3.60 | 2.05e-19 | 0.01 |
| V1_r18 | ours rejected (rule) | 128 | 7.64 | 26.94 | -19.30 | -25.97 | -13.08 | 4.49e-05 | 0.05 |
| V1_r18 | ours gross (APE>25%) | 45 | 11.47 | 71.02 | -59.55 | -71.10 | -49.03 | 1.71e-13 | 0.11 |
| V2_r18 | ours accepted (rule) | 172 | 5.62 | 2.23 | 3.39 | 2.42 | 4.42 | 5.34e-19 | 0.03 |
| V2_r18 | ours rejected (rule) | 128 | 7.64 | 26.94 | -19.30 | -25.91 | -13.15 | 9.3e-06 | 0.05 |
| V2_r18 | ours gross (APE>25%) | 45 | 12.47 | 71.02 | -58.56 | -70.32 | -48.04 | 1.14e-13 | 0.13 |
| V3_r18 | ours accepted (rule) | 172 | 18.06 | 2.23 | 15.83 | 13.87 | 17.84 | 1.75e-28 | 0.26 |
| V3_r18 | ours rejected (rule) | 128 | 30.71 | 26.94 | 3.77 | -2.73 | 9.73 | 0.000772 | 0.55 |
| V3_r18 | ours gross (APE>25%) | 45 | 41.94 | 71.02 | -29.08 | -41.81 | -17.84 | 2.53e-07 | 0.73 |
| V4_r18 | ours accepted (rule) | 172 | 10.88 | 2.23 | 8.65 | 7.40 | 9.98 | 3.19e-27 | 0.08 |
| V4_r18 | ours rejected (rule) | 128 | 21.32 | 26.94 | -5.63 | -12.36 | 0.66 | 0.504 | 0.30 |
| V4_r18 | ours gross (APE>25%) | 45 | 31.10 | 71.02 | -39.92 | -53.10 | -28.25 | 2.03e-09 | 0.51 |
| V5_r18 | ours accepted (rule) | 172 | 10.88 | 2.23 | 8.65 | 7.40 | 9.98 | 3.19e-27 | 0.08 |
| V5_r18 | ours rejected (rule) | 128 | 21.32 | 26.94 | -5.63 | -12.36 | 0.66 | 0.504 | 0.30 |
| V5_r18 | ours gross (APE>25%) | 45 | 31.10 | 71.02 | -39.92 | -53.10 | -28.25 | 2.03e-09 | 0.51 |
| V1_b0 | ours accepted (rule) | 172 | 3.79 | 2.23 | 1.56 | 0.54 | 2.47 | 2.71e-14 | 0.01 |
| V1_b0 | ours rejected (rule) | 128 | 4.95 | 26.94 | -21.99 | -28.61 | -15.84 | 9.11e-08 | 0.00 |
| V1_b0 | ours gross (APE>25%) | 45 | 7.03 | 71.02 | -63.99 | -74.97 | -54.32 | 5.68e-14 | 0.00 |
| V2_b0 | ours accepted (rule) | 172 | 3.39 | 2.23 | 1.16 | 0.13 | 2.08 | 2.68e-10 | 0.00 |
| V2_b0 | ours rejected (rule) | 128 | 5.45 | 26.94 | -21.49 | -28.02 | -15.42 | 1.69e-08 | 0.02 |
| V2_b0 | ours gross (APE>25%) | 45 | 8.26 | 71.02 | -62.77 | -73.85 | -52.84 | 5.68e-14 | 0.04 |
| V3_b0 | ours accepted (rule) | 172 | 17.85 | 2.23 | 15.62 | 13.78 | 17.47 | 5.19e-29 | 0.23 |
| V3_b0 | ours rejected (rule) | 128 | 33.23 | 26.94 | 6.28 | -0.30 | 12.33 | 4.09e-05 | 0.61 |
| V3_b0 | ours gross (APE>25%) | 45 | 44.02 | 71.02 | -27.01 | -39.69 | -15.92 | 4.04e-06 | 0.73 |
| V4_b0 | ours accepted (rule) | 172 | 4.22 | 2.23 | 1.99 | 1.04 | 2.80 | 4.85e-19 | 0.00 |
| V4_b0 | ours rejected (rule) | 128 | 6.35 | 26.94 | -20.59 | -27.17 | -14.48 | 3.57e-06 | 0.01 |
| V4_b0 | ours gross (APE>25%) | 45 | 8.84 | 71.02 | -62.19 | -73.31 | -52.18 | 5.68e-14 | 0.02 |
| V5_b0 | ours accepted (rule) | 172 | 4.22 | 2.23 | 1.99 | 1.04 | 2.80 | 4.85e-19 | 0.00 |
| V5_b0 | ours rejected (rule) | 128 | 6.35 | 26.94 | -20.59 | -27.17 | -14.48 | 3.57e-06 | 0.01 |
| V5_b0 | ours gross (APE>25%) | 45 | 8.84 | 71.02 | -62.19 | -73.31 | -52.18 | 5.68e-14 | 0.02 |
| REF_mps_r18 | ours accepted (rule) | 172 | 5.49 | 2.23 | 3.26 | 2.20 | 4.39 | 6.49e-20 | 0.03 |
| REF_mps_r18 | ours rejected (rule) | 128 | 7.70 | 26.94 | -19.24 | -25.93 | -13.03 | 2.82e-05 | 0.06 |
| REF_mps_r18 | ours gross (APE>25%) | 45 | 11.41 | 71.02 | -59.61 | -71.04 | -49.33 | 1.14e-13 | 0.13 |

### Real 150, scene representatives (76): disagreement with ours (|log ratio|, %, mean of the two axes) by scene split (train = pseudo-train scenes, val = held-out accepted scenes, lowconf = rejected by the rule)

| variant | median_dis_train | median_dis_val | median_dis_lowconf | gross_share_val | gross_share_lowconf | spearman_conf_vs_disagreement | p_MWU_lowconf_gt_heldout |
|---|---|---|---|---|---|---|---|
| V1_r18 | 10.76 | 46.17 | 32.01 | 0.80 | 0.76 | -0.28 | 0.42 |
| V2_r18 | 9.48 | 35.00 | 29.42 | 0.80 | 0.76 | -0.29 | 0.54 |
| V3_r18 | 2.90 | 33.75 | 34.16 | 0.60 | 0.79 | -0.55 | 0.33 |
| V4_r18 | 2.06 | 19.35 | 35.89 | 0.50 | 0.74 | -0.63 | 0.19 |
| V5_r18 | 2.02 | 21.16 | 35.70 | 0.50 | 0.74 | -0.64 | 0.2 |
| V1_b0 | 10.29 | 34.85 | 32.51 | 0.80 | 0.76 | -0.25 | 0.62 |
| V2_b0 | 10.16 | 37.34 | 31.19 | 0.80 | 0.71 | -0.28 | 0.39 |
| V3_b0 | 4.88 | 41.63 | 32.14 | 0.70 | 0.71 | -0.61 | 0.23 |
| V4_b0 | 2.76 | 25.44 | 29.49 | 0.50 | 0.61 | -0.56 | 0.16 |
| V5_b0 | 2.11 | 26.02 | 35.98 | 0.50 | 0.66 | -0.60 | 0.19 |
| REF_mps_r18 | 13.29 | 39.74 | 31.28 | 0.80 | 0.79 | -0.18 | 0.61 |

### POST HOC: real 150 representatives by ours' weave label ('unknown' = ours' weave classifier abstains); median log(pred / ours) over both axes, and the pseudo-val MAPE split the same way

| variant | n_weave_unknown | median_logratio_weave_unknown | median_logratio_weave_known | pseudoval_n_weave_unknown | pseudoval_MAPE_weave_unknown | pseudoval_MAPE_weave_known |
|---|---|---|---|---|---|---|
| V1_r18 | 13 | 0.72 | 0.10 | 5 | 107.05 | 17.23 |
| V2_r18 | 13 | 0.78 | 0.05 | 5 | 110.48 | 15.54 |
| V3_r18 | 13 | 0.07 | -0.02 | 5 | 82.93 | 16.06 |
| V4_r18 | 13 | 0.07 | -0.02 | 5 | 65.90 | 15.52 |
| V5_r18 | 13 | 0.09 | -0.03 | 5 | 66.11 | 15.48 |
| V1_b0 | 13 | 0.74 | 0.09 | 5 | 146.80 | 16.89 |
| V2_b0 | 13 | 0.73 | 0.10 | 5 | 100.73 | 15.11 |
| V3_b0 | 13 | 0.17 | -0.04 | 5 | 63.15 | 18.28 |
| V4_b0 | 13 | 0.11 | -0.00 | 5 | 71.61 | 10.84 |
| V5_b0 | 13 | 0.17 | -0.00 | 5 | 70.83 | 10.49 |
| REF_mps_r18 | 13 | 0.83 | 0.10 | 5 | 118.37 | 20.63 |

### Legacy-13: student errors on the counts where ours fails (ours APE > 25%)

| variant | n_ours_gross | student_APE_on_ours_gross | ours_APE_on_ours_gross | student_also_gross | student_APE_elsewhere | ours_APE_elsewhere | spearman_student_vs_ours_error | share_closer_to_ours_than_to_GT |
|---|---|---|---|---|---|---|---|---|
| V1_r18 | 4 | 27.16 | 57.11 | 3 | 15.08 | 3.34 | 0.47 | 0.69 |
| V2_r18 | 4 | 22.00 | 57.11 | 2 | 12.60 | 3.34 | 0.37 | 0.54 |
| V3_r18 | 4 | 28.51 | 57.11 | 3 | 19.25 | 3.34 | 0.19 | 0.65 |
| V4_r18 | 4 | 25.85 | 57.11 | 2 | 9.59 | 3.34 | 0.33 | 0.54 |
| V5_r18 | 4 | 25.93 | 57.11 | 2 | 9.32 | 3.34 | 0.34 | 0.50 |
| V1_b0 | 4 | 27.09 | 57.11 | 2 | 20.67 | 3.34 | 0.20 | 0.50 |
| V2_b0 | 4 | 26.21 | 57.11 | 2 | 14.72 | 3.34 | 0.20 | 0.58 |
| V3_b0 | 4 | 26.93 | 57.11 | 2 | 15.34 | 3.34 | 0.30 | 0.65 |
| V4_b0 | 4 | 24.43 | 57.11 | 2 | 8.16 | 3.34 | 0.27 | 0.62 |
| V5_b0 | 4 | 25.28 | 57.11 | 3 | 7.70 | 3.34 | 0.32 | 0.62 |

### Ablation, stage 2: one family dropped in the V4-R18 fine-tuning (init V2O_r18, policy whitelist_optics, rho = rho1); selection metrics only; seed range [27.99, 29.71]

| dropped_family | best_step | synth_MAPE | pseudo_MAPE | score | d_score_vs_seed_mean | outside_seed_range |
|---|---|---|---|---|---|---|
| none (seed 0) | 750 | 15.89 | 40.52 | 28.21 | -0.43 | False |
| none (seed 1) | 750 | 15.27 | 40.71 | 27.99 | -0.64 | False |
| none (seed 2) | 500 | 15.67 | 43.76 | 29.71 | 1.08 | False |
| compression_blur | 750 | 16.26 | 41.91 | 29.09 | 0.45 | False |
| geometry | 750 | 17.05 | 39.79 | 28.42 | -0.21 | False |
| illumination | 500 | 15.39 | 43.98 | 29.68 | 1.05 | False |
| optics | 250 | 15.05 | 43.42 | 29.24 | 0.60 | False |
| phone | 750 | 16.58 | 41.79 | 29.19 | 0.55 | False |
| photometric | 750 | 16.04 | 41.70 | 28.87 | 0.23 | False |
| tiling | 250 | 16.04 | 39.29 | 27.67 | -0.97 | True |
| zoom | 500 | 17.62 | 39.92 | 28.77 | 0.14 | False |

### Ablation, stage 1: one family dropped in synthetic-only pre-training ('whitelist_ext', 4750 steps, final step; adopted from the superseded round); seed range [32.33, 38.09]

| dropped_family | synth_MAPE | pseudo_MAPE | score | d_score_vs_seed_mean | outside_seed_range |
|---|---|---|---|---|---|
| none (seed 0) | 6.84 | 69.33 | 38.09 | 2.59 | False |
| none (seed 1) | 6.90 | 57.75 | 32.33 | -3.17 | False |
| none (seed 2) | 7.07 | 65.10 | 36.09 | 0.59 | False |
| geometry | 7.36 | 68.89 | 38.13 | 2.63 | True |
| photometric | 7.08 | 82.50 | 44.79 | 9.29 | True |
| compression_blur | 6.75 | 71.56 | 39.16 | 3.65 | True |
| illumination | 6.89 | 68.92 | 37.91 | 2.41 | False |
| zoom | 7.18 | 69.80 | 38.49 | 2.99 | True |
| tiling | 6.95 | 63.01 | 34.98 | -0.52 | False |

GPU hours (RTX 4090; queue ledgers + final predictions; benches excluded): {"superseded_round_train_h": 2.28, "scene_round_train_h": 3.0, "superseded_round_final_h": 1.01, "scene_round_final_h": 0.76, "total_h": 7.04}


GPU-vs-MPS parity of the existing checkpoint on real150: {"n": 150, "max_abs_log_ratio": 0.004561990433833719, "median_abs_log_ratio": 0.0003278273169922024}

