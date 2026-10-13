"""Capture-quality results on the 150 photos AGAINST THE HUMAN COUNTS (WVAQ section).

score150.py scores the clean predictions. This script adds the label-dependent parts of the
capture-quality study that already have all their predictions on disk:

  A. confidence as a task-referenced quality score on real labels
     -> conf150.csv: per estimator, Spearman(conf, per-image error), AUC of conf for
        "both axes within 10% of the human count", accuracy/coverage at conf >= 0.5 / 0.9.
        Fills the TODO in the "Confidence as a task-referenced quality score" paragraph.
  B. accuracy under the eight physically scaled degradations
     -> degrade150_vs_gt.csv: per estimator x degradation x level, % of the 2n counts within
        5% / 10% of the human count and MAPE (clean row included). This is the accuracy
        counterpart of the self-consistency curves in fig_degradation_drift / tab:quality.
  C. automatic window vs annotated corners, scored against the human counts
     -> crops150_vs_gt.csv (crop = gt | seg_final | centre).

Inputs (read-only): the annotation-tool CSV exports (same rules as score150.py: done=1,
nonwoven=0, newest record per image x annotator, mean over annotators),
methods/predictions_ours_150.csv, methods/results/all150_degrade.csv,
methods/results/all150_crops.csv. Axes: h = vertical yarns crossing the horizontal guide
line on the same rectified crop for annotators and methods, so no axis ambiguity.

Run (venv2):
  <venv2>/bin/python tools/quality150.py --labels <A.csv>,<B.csv> [--blind-only] [--unit scene|image] [--out DIR]
Use --out <scratch dir> for plumbing checks on partial or fake labels.

--unit scene (default since Oct 8, Job P): one row per scene, i.e. the 76 representatives, with labels
propagated by annotation_tool/propagate_scene_labels.py exactly as in score150.py (the 150 files are 76
scenes; copies and notebook-synthetic files are not scored). --unit image: one row per labelled file
(the pre-Oct-8 behaviour, byte-identical outputs).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score150 as S  # noqa: E402  (reuse load_labels / gt_table so both scripts use one GT rule)

ROOT = S.ROOT
P_DEG = os.path.join(ROOT, "methods", "results", "all150_degrade.csv")
P_CROPS = os.path.join(ROOT, "methods", "results", "all150_crops.csv")


def ape(pred, gt):
    return 100 * np.abs(pred - gt) / gt


def auc(score, pos):
    """P(score of a positive > score of a negative), ties count 1/2 (Mann-Whitney)."""
    score, pos = np.asarray(score, float), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return np.nan
    r = stats.rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def count_metrics(d, gt):
    """d: DataFrame indexed by image with h, v. Returns % within 5/10 and MAPE over 2n counts."""
    j = gt[["h", "v"]].join(d[["h", "v"]], rsuffix="_p", how="inner")
    ok = np.isfinite(j.h_p) & np.isfinite(j.v_p)
    n_img = len(j)
    pe = np.r_[ape(j.h_p[ok], j.h[ok]), ape(j.v_p[ok], j.v[ok])]
    # a missing estimate counts as a failure (outside both tolerances)
    n_counts = 2 * n_img
    return dict(n_images=n_img, n_missing=int((~ok).sum()),
                w5=100 * np.sum(pe <= 5) / n_counts if n_counts else np.nan,
                w10=100 * np.sum(pe <= 10) / n_counts if n_counts else np.nan,
                MAPE_answered=float(np.mean(pe)) if len(pe) else np.nan)


def conf_table(gt):
    rows = []
    cands = {}
    o = pd.read_csv(S.P_OURS).set_index("id")
    cands["ours (predictions_ours_150.csv)"] = pd.DataFrame({"h": o.h_count, "v": o.v_count, "conf": o.confidence})
    if os.path.exists(P_DEG):
        d = pd.read_csv(P_DEG)
        d = d[(d.kind == "clean") & (d.get("crop", "gt") == "gt")]
        for est, g in d.groupby("estimator"):
            if g.conf.notna().any():
                cands[f"{est} (clean rows of all150_degrade.csv)"] = g.set_index("image")[["h", "v", "conf"]]
    for name, p in cands.items():
        j = gt[["h", "v"]].join(p, rsuffix="_p", how="inner").dropna()
        if len(j) < 5:
            continue
        err = np.maximum(ape(j.h_p, j.h), ape(j.v_p, j.v))   # per-image worst axis
        ok10 = err <= 10
        row = dict(estimator=name, n_images=len(j),
                   spearman_conf_vs_err=stats.spearmanr(j.conf, err)[0],
                   auc_conf_both_within10=auc(j.conf, ok10),
                   both_within10_all=100 * ok10.mean())
        for t in (0.5, 0.9):
            k = j.conf >= t
            row[f"coverage_conf_ge_{t}"] = 100 * k.mean()
            row[f"both_within10_conf_ge_{t}"] = 100 * ok10[k].mean() if k.any() else np.nan
            row[f"both_within10_conf_lt_{t}"] = 100 * ok10[~k].mean() if (~k).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def degrade_table(gt):
    d = pd.read_csv(P_DEG)
    rows = []
    for (est, kind, level), g in d.groupby(["estimator", "kind", "level"]):
        m = count_metrics(g.set_index("image")[["h", "v"]], gt)
        rows.append(dict(estimator=est, kind=kind, level=level, **m))
    return pd.DataFrame(rows)


def crops_table(gt):
    c = pd.read_csv(P_CROPS)
    rows = []
    for (est, crop), g in c.groupby(["estimator", "crop"]):
        m = count_metrics(g.set_index("image")[["h", "v"]], gt)
        rows.append(dict(estimator=est, crop=crop, **m))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="append", required=True)
    ap.add_argument("--blind-only", action="store_true")
    ap.add_argument("--annotator", default=None)
    ap.add_argument("--out", default=os.path.join(S.PAPER, "tables"))
    ap.add_argument("--unit", default="scene", choices=("scene", "image"),
                    help="scene (default): one row per scene representative; image: one row per labelled file")
    ap.add_argument("--manifest", default=S.MANIFEST)
    a = ap.parse_args()
    paths = [p for arg in a.labels for s in arg.split(",") for p in sorted(glob.glob(s.strip()))]
    if not paths:
        raise SystemExit("no label files found")
    os.makedirs(a.out, exist_ok=True)
    rec = S.load_labels(paths, a.blind_only, a.annotator)
    gt = S.gt_table(rec)
    if a.unit == "scene":   # Job P: same propagation as score150.py --unit scene
        import json
        gt.index = gt.index.astype(int)
        PS = S.scene_module(a.manifest)
        _, gt_scene = PS.propagate(gt, PS.load_scenes(json.load(open(a.manifest))))
        n_own = len(gt)
        gt = gt_scene.drop(columns=["label_src_kind"], errors="ignore")
        gt.index = gt.index.astype(int)
        print(f"--unit scene: {n_own} labelled files -> {len(gt)} scenes (rep photo)")
    gt = gt[(gt.h > 0) & (gt.v > 0)]
    print(f"labels: {len(rec)} records, {len(gt)} {'scenes' if a.unit == 'scene' else 'images'} with human counts")
    pd.set_option("display.width", 200)

    ct = conf_table(gt)
    ct.round(3).to_csv(os.path.join(a.out, "conf150.csv"), index=False)
    print("\nA. confidence vs error against human counts")
    print(ct[["estimator", "n_images", "spearman_conf_vs_err", "auc_conf_both_within10", "both_within10_all",
              "coverage_conf_ge_0.9", "both_within10_conf_ge_0.9"]].round(2).to_string(index=False))

    if os.path.exists(P_DEG):
        dt = degrade_table(gt)
        dt.round(2).to_csv(os.path.join(a.out, "degrade150_vs_gt.csv"), index=False)
        print("\nB. % of counts within 10% of the human count under degradation (ours | fft)")
        piv = dt[dt.estimator.isin(["ours", "fft"])].pivot_table(index=["kind", "level"], columns="estimator",
                                                                   values="w10")
        print(piv.round(0).to_string())
    if os.path.exists(P_CROPS):
        cr = crops_table(gt)
        cr.round(2).to_csv(os.path.join(a.out, "crops150_vs_gt.csv"), index=False)
        print("\nC. crop source vs human count")
        print(cr.round(1).to_string(index=False))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
