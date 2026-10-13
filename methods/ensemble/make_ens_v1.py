#!/usr/bin/env python3
"""ens-v1 (Oct 12, frozen before the second annotator's counts): per file and axis, the geometric mean of
(a) the ImageNet-initialised ResNet-18 fine-tuned on the FIRST annotator's (FL_SA) counts with 5-fold cross-validation
    grouped by the 44 fabric groups (out-of-fold predictions: every file is predicted by a model that never saw its fabric), and
(b) GPT-6.1-sol, prompt P1 (frozen predictions, vlm/results/rect150_P1_predictions_wide.csv; h = <key>__v, v = <key>__h,
    the convention of score150.py / eval_octave_v1.py).
If GPT has no count for an axis, the CNN's count is used. Output: predictions_ens_v1_150.csv (id, h_count, v_count) and
cnn_fabric_oof_FLSA_150.csv (the CNN component alone, same format). Rule chosen post hoc on FL_SA (25.5% vs GPT 28.8%)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
cnn_p = sys.argv[1]   # train_cv.py predictions.csv of the fabric-grouped FL_SA run
c = pd.read_csv(cnn_p); c = c[c.experiment == "imagenet_ft"].set_index("image")
w = pd.read_csv(os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")).set_index("image_id")
gh = pd.to_numeric(w["gpt-6.1-sol__v"], errors="coerce"); gv = pd.to_numeric(w["gpt-6.1-sol__h"], errors="coerce")
rows, cnn = [], []
for i, r in c.iterrows():
    h_g, v_g = gh.get(i, np.nan), gv.get(i, np.nan)
    eh = np.sqrt(r.pred_h * h_g) if np.isfinite(h_g) and h_g > 0 else r.pred_h
    ev = np.sqrt(r.pred_v * v_g) if np.isfinite(v_g) and v_g > 0 else r.pred_v
    rows.append(dict(id=int(i), h_count=round(float(eh), 4), v_count=round(float(ev), 4)))
    cnn.append(dict(id=int(i), h_count=round(float(r.pred_h), 4), v_count=round(float(r.pred_v), 4)))
pd.DataFrame(rows).sort_values("id").to_csv(os.path.join(HERE, "predictions_ens_v1_150.csv"), index=False)
pd.DataFrame(cnn).sort_values("id").to_csv(os.path.join(HERE, "cnn_fabric_oof_FLSA_150.csv"), index=False)
print(len(rows), "files")
