# 2019 baseline harness (verbatim copy)

Copied unmodified on 2026-10-07 from the session scratchpad
`.../scratchpad/pipeline/` (run_pipeline.py sha1 ccc56a28…). The originals were not modified.

- `run_pipeline.py`: faithful re-implementation of the 2019 IIT-M counting step (TQA/doc3.py dot map + TQA/count.py single-row transition count). It also has the GT parser `load_gt()`, which `../ablate.py` reuses.
- `seg_final.py`: the 2024 classical segmentation (mean IoU 0.88), extracted verbatim from the notebook.
- `repro_*.py`, `kmeans_variability.py`, `smoke_test.py`, `build_seg_final.py`, `diff_2019_vs_2024_checks.py`, `dump_nb.py`: helper and reproduction scripts from the same harness.
- `run1/results.csv`: per-image 2019 predictions (variant `precropped` = images fed straight to doc3, seed 0). The column `xlsx_rgb_weft_pred` is the 2019 "predicted weft by RGB irregularity" (FR-final attempt/test.py, weft only).
- `run1/metrics.json`, `run_log.txt`, `seed_sweep_precropped.csv`: the original metrics. On the 13 usable images, warp MAE is 114.9 and weft MAE is 95.0. For test.py the weft MAE is 19.0, with 5 of 13 within 10 %.

Re-run (needs the 2019 venv with numpy<2, cv2 and scipy): `python run_pipeline.py --out run1`. `../summarise_legacy.py` reads `run1/results.csv` and reproduces these numbers exactly.
- `count_wrap_edges.py` + `run1/dotmaps_precropped/` (the 20 seed-0 dot maps of this run; not in this repository): shows the uint8-wrap diagnosis on this run. The supplement reports the same diagnosis on the paper's own run (149 of 167 edges; warp MAE 99 to 142 TPI over five seeds), so the numbers below differ from it. Of the edges that count.py counts, 110 of 124 on the warp rows of all 20 crops, and 136 of 157 (87 %) on both axes of the 13 hand-counted crops, are -1 grey-level steps (JPEG ringing), not 0->255 rises. Run: `<venv2>/bin/python count_wrap_edges.py`.
- Seed sweep (`run1/seed_sweep_precropped.csv` + seed 0 in `results.csv`): warp MAE on the 13 images is 114.9 / 76.0 / 95.1 / 73.4 / 112.1 TPI for seeds 0-4 (weft 95.0 / 55.2 / 69.5 / 52.8 / 93.7).
