"""Collect every legacy-13 number used in Table 3 (tab:legacy) into tables/legacy13_all.csv.

Sources (read-only):
  methods/results/legacy13_metrics.csv          2019 baselines + all spectral estimators (crop=gt
                                                 i.e. protocol P1, preproc lcn, glare mask, clean)
  methods/results/legacy13_protocol_P1_P2.csv   P2 = frame-free window from methods/legacy.detect_window
  vlm/results/legacy_predictions.csv            VLM answers (window trimmed by vlm/data_vlm.py, 1536 px)
  cnn/runs/synth_r18_v2mix_aniso/eval_final/legacy_predictions_detect.csv   CNN zero-shot

GT for the VLM/CNN rows is the gt_* column those jobs stored (both reuse run_pipeline.load_gt).
Pooled over 26 counts (13 images x 2 axes); 'axis' = h<->warp, 'sorted' = best assignment.

    <venv2>/bin/python tools/legacy_table.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)


def pooled(ph, pv, gw, gf, assign):
    P = np.c_[ph, pv].astype(float)
    G = np.c_[gw, gf].astype(float)
    if assign == "sorted":
        P, G = np.sort(P, 1), np.sort(G, 1)
    e = np.abs(P - G)
    pe = 100 * e / G
    return dict(n=e.size, MAE=e.mean(), MAPE=pe.mean(), n_w5=int((pe <= 5).sum()), n_w10=int((pe <= 10).sum()),
                medAE=np.median(e))


def main():
    rows = []
    m = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_metrics.csv"))
    base = m[(m.subset == "all13") & m.estimator.isna()]
    for _, r in base.iterrows():
        rows.append(dict(group="2019", method=r["method"], crop="2019 crop", assign=r["assign"], axis=r["axis"],
                         n=r["n"], MAE=r["MAE"], MAPE=r["MAPE"], n_w5=r["n_w5"], n_w10=r["n_w10"], medAE=r["medAE"]))
    est = m[(m.subset == "all13") & (m.crop == "gt") & (m.preproc == "lcn") & (m.glare == "mask") &
            (m.kind == "clean") & (m.axis == "both")]
    for _, r in est.iterrows():
        rows.append(dict(group="spectral", method=r["estimator"], crop="P1 whole crop", assign=r["assign"], axis="both",
                         n=r["n"], MAE=r["MAE"], MAPE=r["MAPE"], n_w5=r["n_w5"], n_w10=r["n_w10"], medAE=r["medAE"]))
    p = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_protocol_P1_P2.csv"))
    for _, r in p[(p.protocol == "P2") & (p.axis == "both")].iterrows():
        rows.append(dict(group="spectral", method=r["estimator"], crop="P2 detected window", assign=r["assign"],
                         axis="both", n=r["n"], MAE=r["MAE"], MAPE=r["MAPE"], n_w5=r["n_w5"], n_w10=r["n_w10"],
                         medAE=r["medAE"]))
    v = pd.read_csv(os.path.join(ROOT, "vlm", "results", "legacy_predictions.csv"))
    v = v[(v.set == "legacy") & (v.repeat == 0) & v.gt_warp.notna() & (v.ok == True)]  # noqa: E712
    for (model, prompt), g in v.groupby(["model", "method"]):
        for assign in ("axis", "sorted"):
            crop = ("VLM trimmed window, 644 px (local MLX)" if model.startswith("molmo2")
                    else "VLM trimmed window, 1536 px")
            rows.append(dict(group="vlm", method=f"{model} {prompt}", crop=crop,
                             assign=assign, axis="both", n_images=len(g),
                             **pooled(g.vertical, g.horizontal, g.gt_warp, g.gt_weft, assign)))
    c = pd.read_csv(os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso", "eval_final",
                                 "legacy_predictions_detect.csv"))
    for assign in ("axis", "sorted"):
        rows.append(dict(group="cnn", method="CNN synthetic zero-shot", crop="detected window", assign=assign,
                         axis="both", n_images=len(c), **pooled(c.pred_h, c.pred_v, c.warp, c.weft, assign)))
    # leave-one-out mean predictor on the same GT
    gw, gf = c.warp.to_numpy(float), c.weft.to_numpy(float)
    n = len(gw)
    lw, lf = (gw.sum() - gw) / (n - 1), (gf.sum() - gf) / (n - 1)
    for assign in ("axis", "sorted"):
        rows.append(dict(group="reference", method="leave-one-out mean", crop="-", assign=assign, axis="both",
                         n_images=n, **pooled(lw, lf, gw, gf, assign)))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    out.round(2).to_csv(os.path.join(PAPER, "tables", "legacy13_all.csv"), index=False)
    show = out[(out.axis == "both") | (out.group == "2019")]
    print(show[["group", "method", "crop", "assign", "n", "MAE", "MAPE", "n_w5", "n_w10"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
