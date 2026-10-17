# VLM results

The supplement (Sec. S3) cites this file for the repeat spread of the VLMs on the legacy crops. The numbers are in `results/`, written by `evaluate_vlm.py` from the raw answers in `raw/`:

- `results/stability_P1_legacy.csv`: three repeats of prompt P1 at temperature 0 on the legacy crops. `mean_range_tpi` is the mean per-image spread: 4.875 TPI for GPT-6.1-sol (20 crops parsed in all three repeats) and 14.87 TPI for Gemini 3.8 Flash (19 crops), the 4.9 and 14.9 of the supplement.
- `results/stageA_legacy_metrics.csv` and `results/legacy_predictions.csv`: every model and prompt on the legacy crops. The paper's legacy rows under one crop rule come from `paper/tools/legacy_unified.py` (supplement Tab. S3).
- `results/rect150_P1_predictions_wide.csv` (with `_long` and `_summary`): prompt P1 on the 150 PickGlass crops, scored by `paper/tools/score150.py` (README, *Scoring human counts*).
- `results/legacy_deg_predictions.csv`: the degraded legacy crops (`run_degrade_vlm.py`), read by `paper/tools/degradation_table.py`.

Models, prompts and settings: [MODEL_ROSTER.md](MODEL_ROSTER.md) and `prompts.py`.
