# Counting-specialist results

The supplement (Sec. S8) cites this file for how the counting specialists fail on the 13 legacy crops. `score_gpu_baselines.py` scores the raw answers (`raw/`, `countgd/outputs/`) and writes `scored/` and `tables/legacy13_gpu_rows.csv` (rows with `assign` = `axis` are the axis-matched scores of the paper):

- Molmo2-8B in bf16 on an RTX 4090: MAPE 79.9% with prompt P1 and 24.9% with P4q (rows `molmo2_8b_bf16 P1` and `molmo2_8b_bf16 P4q`); its point lists: `scored/molmo2_pointing_summary.csv` and `scored/molmo2_pointing_diagnostics.csv`.
- CountGD: 341.8% (row `countgd CG-B-yarn-ex3`); the post-hoc variant chosen on the legacy windows, 18.3% (row `countgd CG-W-ex3-lat`); threshold sensitivity: `scored/countgd_threshold_sensitivity_recomputed.csv`.
- Every run, with confidence intervals: `scored/legacy13_stageA_gpu_baselines.csv`.

On PickGlass the predictions are `scored/molmo2_rect150_predictions_wide.csv` and `countgd/outputs/results/rect150_predictions_wide.csv`, scored with `paper/tools/score150.py` (README, *Scoring human counts*).
