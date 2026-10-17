"""summarise_legacy.py -- accuracy on the 13 hand-counted legacy images (held-out test set).

Inputs : one or more tidy CSVs written by ``ablate.py --set legacy13`` + the legacy baselines
         (legacy_2019/run1/results.csv: doc3+count faithful, precropped, seed 0; and the
         'predicted weft by RGB irregularity' column of the legacy xlsx = FR-final test.py).
Output : results/legacy13_metrics.csv (tidy) and a printed table.

Assignments (GT axis vs image axis is uncertain for these photos):
  axis      h_count (vertical yarns, counted along a row) <-> GT 'warp',  v_count <-> 'weft'.
            This is the legacy convention: count.py counted warp along a row, test.py counted
            weft down a column.
  sorted    best assignment without looking at errors: larger prediction <-> larger GT.
  nearest   (single-axis baselines only) prediction vs the nearer of the two GT values (oracle).
Metrics: n, MAE [threads/inch], MAPE [%], within 5 % / 10 % (fraction and count), median AE.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE2019 = os.path.join(HERE, "legacy_2019", "run1", "results.csv")


def metr(p, t):
    p, t = np.asarray(p, float), np.asarray(t, float)
    ok = np.isfinite(t)
    p, t = p[ok], t[ok]
    p = np.where(np.isfinite(p), p, 0.0)  # a missing prediction counts as 0 (100 % error)
    e = np.abs(p - t)
    r = e / t
    n = len(t)
    return dict(n=n, MAE=float(e.mean()) if n else np.nan, MAPE=float(100 * r.mean()) if n else np.nan,
                w5=float((r <= .05).mean()) if n else np.nan, w10=float((r <= .10).mean()) if n else np.nan,
                n_w5=int((r <= .05).sum()), n_w10=int((r <= .10).sum()),
                medAE=float(np.median(e)) if n else np.nan)


def pair_metrics(df, hcol, vcol):
    """df with gt_warp, gt_weft and prediction columns -> list of metric rows."""
    out = []
    h, v = df[hcol].values.astype(float), df[vcol].values.astype(float)
    wa, we = df.gt_warp.values.astype(float), df.gt_weft.values.astype(float)
    out.append(dict(assign="axis", axis="warp(h)", **metr(h, wa)))
    out.append(dict(assign="axis", axis="weft(v)", **metr(v, we)))
    out.append(dict(assign="axis", axis="both", **metr(np.r_[h, v], np.r_[wa, we])))
    ph, pl = np.fmax(h, v), np.fmin(h, v)
    th, tl = np.fmax(wa, we), np.fmin(wa, we)
    out.append(dict(assign="sorted", axis="both", **metr(np.r_[ph, pl], np.r_[th, tl])))
    return out


def baselines():
    b = pd.read_csv(BASE2019)
    b = b[(b.variant == "precropped") & (b.gt_usable == True)].copy()  # noqa: E712
    b["image"] = b["key"]
    b["gt_unclear"] = b["gt_unclear"].astype(bool)
    rows = []
    for subset, d in (("all13", b), ("clear9", b[~b.gt_unclear])):
        for r in pair_metrics(d, "pred_warp_per_inch", "pred_weft_per_inch"):
            rows.append(dict(method="2019 doc3+count (faithful)", subset=subset, **r))
        rows.append(dict(method="2019 test.py RGB irregularity (weft only)", subset=subset, assign="axis",
                         axis="weft(v)", **metr(d.xlsx_rgb_weft_pred, d.gt_weft)))
        near = np.where(np.abs(d.xlsx_rgb_weft_pred - d.gt_warp) < np.abs(d.xlsx_rgb_weft_pred - d.gt_weft),
                        d.gt_warp, d.gt_weft)
        rows.append(dict(method="2019 test.py RGB irregularity (weft only)", subset=subset, assign="nearest",
                         axis="either", **metr(d.xlsx_rgb_weft_pred, near)))
    return pd.DataFrame(rows), b


def summarise(csvs, protocol_scale=None):
    rows = []
    allp = []
    for f in csvs:
        d = pd.read_csv(f)
        if "error" in d:
            d = d[d.error.isna()]
        d = d[d.gt_usable == True]  # noqa: E712
        allp.append(d)
    d = pd.concat(allp, ignore_index=True)
    keys = ["crop", "preproc", "glare", "estimator", "kind", "level"]
    d = d.drop_duplicates(keys + ["image"], keep="last")
    for k, g in d.groupby(keys, dropna=False):
        meta = dict(zip(keys, k))
        for subset, gg in (("all13", g), ("clear9", g[~g.gt_unclear.astype(bool)])):
            for r in pair_metrics(gg, "h", "v"):
                rows.append(dict(method=meta["estimator"], subset=subset, **meta, **r))
    return pd.DataFrame(rows), d


def weave_accuracy(d):
    """Coarse weave accuracy (plain / twill) for rows with a GT weave."""
    m = {"1/1": "plain", "2/1": "twill", "1/2": "twill"}
    out = []
    for k, g in d.groupby(["crop", "preproc", "glare", "estimator", "kind", "level"], dropna=False):
        if g.estimator.iloc[0] not in ("ours", "lattice"):
            continue
        gt = g.gt_weave.map(m)
        pr = g.weave.map(lambda w: "plain" if w == "plain" else ("twill" if str(w).startswith("twill")
                                                                   else ("satin" if w == "satin" else "unknown")))
        ok = gt.notna()
        out.append(dict(estimator=k[3], crop=k[0], preproc=k[1], glare=k[2], kind=k[4], level=k[5],
                        n=int(ok.sum()), acc=float((gt[ok] == pr[ok]).mean()),
                        n_unknown=int((pr[ok] == "unknown").sum()),
                        confusion=pd.crosstab(gt[ok], pr[ok]).to_dict()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "legacy13_metrics.csv"))
    a = ap.parse_args()
    bl, _ = baselines()
    m, d = summarise(a.csvs)
    res = pd.concat([bl, m], ignore_index=True)
    res.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = res[(res.subset == "all13") & (res.axis.isin(["both", "weft(v)", "warp(h)", "either"]))]
    cols = ["method", "crop", "preproc", "glare", "kind", "level", "assign", "axis", "n", "MAE", "MAPE", "n_w5",
            "n_w10", "medAE"]
    print(show[[c for c in cols if c in show]].round(1).to_string(index=False))
    wa = weave_accuracy(d)
    if len(wa):
        print(wa.to_string(index=False))
        wa.to_csv(a.out.replace(".csv", "_weave.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
