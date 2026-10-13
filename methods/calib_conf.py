"""calib_conf.py -- fit the no-reference confidence of the 'ours' counter on SYNTHETIC data only.

    python calib_conf.py results/dev/synth_ours.csv      (CSV from eval_synth.py --methods ours)

Target: both counts within 10 % of the synthetic ground truth.  Features (all no-reference,
computed by counting.estimate(method='ours')):
  g_min        weaker of the two yarn-peak strengths (whitened z minus noise baseline)
  margin       score gap to the best hypothesis with a different count (ambiguity)
  weave_g      evidence of the selected weave sub-lattice
  fine         strength of unexplained finer-than-yarn peaks (should be ~0)
  tile_support fraction of 3x3 half-size tiles where both yarn peaks are present
  agree_prof   fraction of axes where the profile/YIN estimate agrees with the lattice (6 %)
  agree_tiles  agreement of the fft tile votes (min over axes)
  fused_lattice fraction of axes where the fusion kept the lattice estimate
  sharpness    Laplacian-variance ratio of the working image
  glare_frac   area fraction masked as specular glare
Standardised logistic regression (L2, C=1).  Writes conf_model.json next to counting.py and
prints 5-fold cross-validated AUC / Brier score.  The 13 legacy labels are NOT used.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["g_min", "margin", "weave_g", "fine", "tile_support", "agree_prof", "agree_tiles",
         "fused_lattice", "sharpness", "glare_frac"]


def main(path, m="ours"):
    df = pd.read_csv(path)
    ok = ((np.abs(df[f"{m}_h"] / df.h_count - 1) <= .1) & (np.abs(df[f"{m}_v"] / df.v_count - 1) <= .1)).astype(int)
    X = df[[f"{m}_f_{k}" for k in FEATS]].astype(float)
    X = X.fillna(X.median())
    mu, sd = X.mean().values, X.std().values + 1e-9
    Z = (X.values - mu) / sd
    y = ok.values
    pcv = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        clf = LogisticRegression(C=1.0).fit(Z[tr], y[tr])
        pcv[te] = clf.predict_proba(Z[te])[:, 1]
    clf = LogisticRegression(C=1.0).fit(Z, y)
    model = {"intercept": float(clf.intercept_[0]),
             "coef": {k: [float(a), float(b), float(w)] for k, a, b, w in zip(FEATS, mu, sd, clf.coef_[0])},
             "fitted_on": os.path.basename(path), "n": int(len(y)), "pos_rate": float(y.mean()),
             "cv5_auc": float(roc_auc_score(y, pcv)), "cv5_brier": float(brier_score_loss(y, pcv)),
             "target": "both counts within 10 % (synthetic GT)"}
    json.dump(model, open(os.path.join(HERE, "conf_model.json"), "w"), indent=1)
    print(json.dumps(model, indent=1))
    # single-feature AUCs for the paper
    for k in FEATS:
        try:
            print(f"  AUC({k}) = {roc_auc_score(y, X[f'{m}_f_{k}']):.3f}")
        except ValueError:
            pass


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
