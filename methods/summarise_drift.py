"""summarise_drift.py -- self-consistency of the counters under controlled degradations (no GT).

    python summarise_drift.py results/all150_degrade.csv [--out results/all150_drift_summary.csv]

For every (estimator, kind, level): drift = pred_degraded / pred_clean - 1 per axis (same image,
same crop/preprocessing); |drift| = worse of the two axes.  Reports n, median |drift|, share
within 5 % / 10 % of the clean prediction, share of non-finite predictions, mean confidence, and
the Spearman correlation between the degraded image's confidence and its |drift| (is the
no-reference confidence a usable capture-quality score?).  Also writes the crop-source table
(seg_final / centre vs annotated corners) when the CSV contains several crop sources.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def drift_table(df):
    df = df[df.error.isna()] if "error" in df else df
    keys = ["image", "crop", "preproc", "glare", "estimator"]
    clean = df[df.kind == "clean"][keys + ["h", "v", "conf"]].rename(columns={"h": "h0", "v": "v0", "conf": "conf0"})
    d = df[df.kind != "clean"].merge(clean, on=keys)
    d["drift_h"] = d.h / d.h0 - 1
    d["drift_v"] = d.v / d.v0 - 1
    d["absdrift"] = np.fmax(np.abs(d.drift_h), np.abs(d.drift_v)).fillna(np.inf)
    return d


def summary(d):
    rows = []
    for (est, kind, level), g in d.groupby(["estimator", "kind", "level"]):
        ok = np.isfinite(g.absdrift)
        rho = stats.spearmanr(g.conf, np.where(ok, g.absdrift, 1e3))[0] if len(g) > 5 else np.nan  # failures rank last
        rows.append(dict(estimator=est, kind=kind, level=level, n=len(g),
                         median_absdrift_pct=100 * float(np.median(g.absdrift)),
                         within5_pct=100 * float((g.absdrift <= .05).mean()),
                         within10_pct=100 * float((g.absdrift <= .10).mean()),
                         nonfinite_pct=100 * float((~ok).mean()),
                         mean_conf=float(g.conf.mean()), mean_conf_clean=float(g.conf0.mean()),
                         spearman_conf_vs_absdrift=float(rho)))
    return pd.DataFrame(rows)


def crop_table(df):
    """Agreement of predictions from automatically segmented / centre crops with the
    annotated-corner crop (clean images)."""
    df = df[(df.kind == "clean")]
    if "error" in df:
        df = df[df.error.isna()]
    ref = df[df.crop == "gt"][["image", "estimator", "h", "v"]].rename(columns={"h": "h0", "v": "v0"})
    d = df[df.crop != "gt"].merge(ref, on=["image", "estimator"])
    d["absdrift"] = np.fmax(np.abs(d.h / d.h0 - 1), np.abs(d.v / d.v0 - 1)).fillna(np.inf)
    rows = []
    for (est, crop), g in d.groupby(["estimator", "crop"]):
        rows.append(dict(estimator=est, crop=crop, n=len(g),
                         median_absdrift_pct=100 * float(np.median(g.absdrift)),
                         within5_pct=100 * float((g.absdrift <= .05).mean()),
                         within10_pct=100 * float((g.absdrift <= .10).mean()),
                         seg_fallback_n=int(g.seg_fallback.fillna(False).astype(bool).sum())
                         if "seg_fallback" in g else 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = a.out or a.csv.replace(".csv", "_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if df.kind.nunique() > 1:
        s = summary(drift_table(df))
        s.to_csv(out, index=False)
        print(s.round(2).to_string(index=False))
    if df.crop.nunique() > 1:
        c = crop_table(df)
        c.to_csv(out.replace(".csv", "_crops.csv"), index=False)
        print(c.round(2).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
