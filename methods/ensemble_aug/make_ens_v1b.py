#!/usr/bin/env python3
"""ens-v1b (Oct 12, frozen before the second annotator's counts): same rule as ens-v1 (methods/ensemble/make_ens_v1.py),
but the CNN is label_day's "aug1016" config (train_cv.py --extra-aug1016: extra augmented crops), ImageNet init,
5-fold CV grouped by the 44 fabric groups, trained on the FIRST annotator's (FL_SA) counts, seed 0.
Per file and axis: geometric mean of the CNN's out-of-fold prediction and GPT-6.1-sol P1 (h = <key>__v, v = <key>__h);
GPT missing -> CNN. Chosen post hoc on FL_SA as the best of 5 CNN variants (23.8% vs ens-v1 25.5%, paired -1.7 [-4.2, +0.3]).
Usage: make_ens_v1b.py cnn_aug1016_fabric_oof_FLSA_raw_predictions.csv"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
c = pd.read_csv(sys.argv[1]); c = c[c.experiment == "imagenet_ft"].set_index("image")
w = pd.read_csv(os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")).set_index("image_id")
gh = pd.to_numeric(w["gpt-6.1-sol__v"], errors="coerce"); gv = pd.to_numeric(w["gpt-6.1-sol__h"], errors="coerce")
rows, cnn = [], []
for i, r in c.iterrows():
    h_g, v_g = gh.get(i, np.nan), gv.get(i, np.nan)
    eh = np.sqrt(r.pred_h * h_g) if np.isfinite(h_g) and h_g > 0 else r.pred_h
    ev = np.sqrt(r.pred_v * v_g) if np.isfinite(v_g) and v_g > 0 else r.pred_v
    rows.append(dict(id=int(i), h_count=round(float(eh), 4), v_count=round(float(ev), 4)))
    cnn.append(dict(id=int(i), h_count=round(float(r.pred_h), 4), v_count=round(float(r.pred_v), 4)))
pd.DataFrame(rows).sort_values("id").to_csv(os.path.join(HERE, "predictions_ens_v1b_150.csv"), index=False)
pd.DataFrame(cnn).sort_values("id").to_csv(os.path.join(HERE, "cnn_aug1016_fabric_oof_FLSA_150.csv"), index=False)
print(len(rows), "files")
