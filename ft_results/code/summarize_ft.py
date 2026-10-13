"""Turn raw Molmo2-4B records (ft/raw/*.jsonl) into per-image counts for every readout, score the
synthetic validation set, and write prediction CSVs for legacy / rect150 (scored on the Mac with the
paper's unified scorer).

Readouts (per image; v = vertical yarns <-> warp, h = horizontal yarns <-> weft):
  P1      JSON answer on the whole view
  P4      n_points on the whole-view guide lines                      (project convention)
  P4q     4 x n_points on the central quarter-inch patch               (project convention)
  P4s     sum of n_points over the 4 diagonal quarter patches          (whole inch, no extrapolation)
  secondary (spacing, not used for selection):
  P4q_sp  (n-1) / (span of the points in inches) on the central quarter
  P4s_sp  mean over the 4 diagonal patches of the spacing estimate
PRIMARY-READOUT RULE (fixed before any legacy-13 scoring of the fine-tuned model): the readout among
{P1, P4, P4q, P4s} with the lowest MAPE on the synthetic validation crops (against the exact crossing
counts) -> selection.json.

    python summarize_ft.py --raw ../raw --out ../results [--select-from molmo2_4b_synthft]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY_CANDIDATES = ["P1", "P4", "P4q", "P4s"]


def load_raw(raw_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            recs.append(r)
    # de-duplicate (resumed runs / shards): keep the last record per key
    d = {}
    for r in recs:
        d[(r["model"], r["set"], r["image_id"], r["view"])] = r
    return list(d.values())


def _coords(r, axis):
    pts = r["parsed"].get("points", []) or []
    return sorted(p[0] for p in pts) if axis == "h" else sorted(p[1] for p in pts)


def _spacing_tpi(c, width_in):
    if len(c) < 3:
        return np.nan
    span = (c[-1] - c[0]) / 1000.0 * width_in
    return (len(c) - 1) / span if span > 0 else np.nan


def readouts(recs):
    """-> DataFrame model,set,image,method,vertical,horizontal,weave,ok,n_trunc,seconds"""
    by = defaultdict(dict)
    for r in recs:
        by[(r["model"], r["set"], r["image_id"])][r["view"]] = r
    rows = []
    for (model, set_name, iid), V in by.items():
        def add(method, v, h, weave=None, secs=np.nan, trunc=0):
            ok = v is not None and h is not None and np.isfinite(v) and np.isfinite(h)
            rows.append(dict(model=model, set=set_name, image=iid, method=method,
                             vertical=v if ok else np.nan, horizontal=h if ok else np.nan, weave=weave,
                             ok=bool(ok), n_truncated=trunc, seconds=secs))
        if "crop" in V:
            p = V["crop"]["parsed"]
            add("P1", p.get("vertical"), p.get("horizontal"), p.get("weave"), V["crop"]["seconds"])

        def n(view):
            return V[view]["parsed"].get("n_points", 0)

        def tr(views):
            return int(sum(V[x]["finish_reason"] == "length" for x in views))

        def secs(views):
            return float(sum(V[x]["seconds"] for x in views))
        g = ["guide_h", "guide_v"]
        if all(x in V for x in g):
            add("P4", n("guide_h"), n("guide_v"), secs=secs(g), trunc=tr(g))
        q = ["qc_guide_h", "qc_guide_v"]
        if all(x in V for x in q):
            add("P4q", 4 * n("qc_guide_h"), 4 * n("qc_guide_v"), secs=secs(q), trunc=tr(q))
            add("P4q_sp", _spacing_tpi(_coords(V["qc_guide_h"], "h"), 0.25),
                _spacing_tpi(_coords(V["qc_guide_v"], "v"), 0.25), secs=secs(q), trunc=tr(q))
        s = [f"qd{k}_guide_{a}" for k in range(4) for a in "hv"]
        if all(x in V for x in s):
            add("P4s", sum(n(f"qd{k}_guide_h") for k in range(4)), sum(n(f"qd{k}_guide_v") for k in range(4)),
                secs=secs(s), trunc=tr(s))
            add("P4s_sp", float(np.nanmean([_spacing_tpi(_coords(V[f"qd{k}_guide_h"], "h"), 0.25) for k in range(4)])),
                float(np.nanmean([_spacing_tpi(_coords(V[f"qd{k}_guide_v"], "v"), 0.25) for k in range(4)])),
                secs=secs(s), trunc=tr(s))
    return pd.DataFrame(rows)


def synval_scores(R, val_labels, B=10000, seed=7):
    lab = {str(r["seed"]): r for r in (json.loads(l) for l in open(val_labels))}
    d = R[R.set == "synval"].copy()
    d["gt_v"] = d.image.map(lambda i: lab[i]["n_h_full"])      # vertical yarns crossing the h line
    d["gt_h"] = d.image.map(lambda i: lab[i]["n_v_full"])
    d["epi"] = d.image.map(lambda i: lab[i]["epi"])
    d["ppi"] = d.image.map(lambda i: lab[i]["ppi"])
    d["weave_gt"] = d.image.map(lambda i: lab[i]["weave_coarse"])
    rows, per = [], []
    for (model, meth), g in d.groupby(["model", "method"]):
        n_img = len(g)
        g = g[g.ok]
        ape = np.concatenate([100 * np.abs(g.vertical - g.gt_v) / g.gt_v, 100 * np.abs(g.horizontal - g.gt_h) / g.gt_h])
        gt = np.concatenate([g.gt_v, g.gt_h])
        img_ape = (100 * np.abs(g.vertical - g.gt_v) / g.gt_v + 100 * np.abs(g.horizontal - g.gt_h) / g.gt_h).to_numpy() / 2
        rng = np.random.default_rng(seed)
        bs = img_ape[rng.integers(0, len(img_ape), (B, len(img_ape)))].mean(1) if len(img_ape) else np.array([np.nan])
        r = dict(model=model, method=meth, n_images=n_img, n_ok=len(g), MAPE=ape.mean() if len(ape) else np.nan,
                 MAPE_lo=np.percentile(bs, 2.5), MAPE_hi=np.percentile(bs, 97.5), medAPE=np.median(ape) if len(ape) else np.nan,
                 w5=float((ape <= 5).mean()) if len(ape) else np.nan, w10=float((ape <= 10).mean()) if len(ape) else np.nan,
                 gross25=float((ape > 25).mean()) if len(ape) else np.nan,
                 n_truncated_calls=int(g.n_truncated.sum()))
        for lo, hi in ((30, 60), (60, 100), (100, 150), (150, 205)):
            m = (gt >= lo) & (gt < hi)
            r[f"MAPE_{lo}_{hi}"] = ape[m].mean() if m.any() else np.nan
            r[f"n_{lo}_{hi}"] = int(m.sum())
        if meth == "P1":
            w = g[g.weave.notna()]
            r["weave_acc"] = float((w.weave == w.weave_gt).mean()) if len(w) else np.nan
        rows.append(r)
        per.append(g.assign(ape_v=100 * np.abs(g.vertical - g.gt_v) / g.gt_v, ape_h=100 * np.abs(g.horizontal - g.gt_h) / g.gt_h))
    return pd.DataFrame(rows), (pd.concat(per) if per else pd.DataFrame())


def point_matching(recs, val_labels):
    """Localisation on synthetic val: one-to-one matching (monotone, greedy by distance) of predicted
    coordinates along the guide line to the exact crossings, tolerance = half the local GT spacing.
    -> per call: n_pred, n_gt, TP, precision, recall, F1, median |offset| in units of GT spacing,
    spacing CV of the predictions (a regular hallucinated grid has CV ~ 0)."""
    lab = {str(r["seed"]): r for r in (json.loads(l) for l in open(val_labels))}
    rows = []
    for r in recs:
        if r["set"] != "synval" or not r["prompt_id"].startswith("P4"):
            continue
        axis = r["view"][-1]
        vname = "full" if r["view"] in ("guide_h", "guide_v") else r["view"].split("_guide_")[0]
        gt = np.array(lab[r["image_id"]]["views"][vname][axis]) * 1000.0
        pr = np.array(_coords(r, axis), float)
        sp = np.median(np.diff(gt)) if len(gt) > 1 else np.nan
        tol = 0.5 * sp
        pairs = sorted(((abs(p - g), i, j) for i, p in enumerate(pr) for j, g in enumerate(gt) if abs(p - g) <= tol))
        used_p, used_g, offs = set(), set(), []
        for dist, i, j in pairs:
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
                offs.append(dist / sp)
        tp = len(offs)
        prec = tp / len(pr) if len(pr) else np.nan
        rec = tp / len(gt) if len(gt) else np.nan
        cv = float(np.std(np.diff(pr)) / np.mean(np.diff(pr))) if len(pr) > 2 else np.nan
        rows.append(dict(model=r["model"], image=r["image_id"], view=r["view"], scale="full" if vname == "full" else "quarter",
                         n_pred=len(pr), n_gt=len(gt), tp=tp, precision=prec, recall=rec,
                         f1=2 * prec * rec / (prec + rec) if prec and rec else 0.0,
                         med_offset_spacing=float(np.median(offs)) if offs else np.nan, pred_spacing_cv=cv,
                         gt_spacing_cv=float(np.std(np.diff(gt)) / np.mean(np.diff(gt))) if len(gt) > 2 else np.nan,
                         truncated=r["finish_reason"] == "length"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--val-labels", default=os.path.join(HERE, "..", "data", "val", "labels.jsonl"))
    ap.add_argument("--select-from", default=None, help="model tag whose synval scores fix the primary readout")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = load_raw(a.raw)
    R = readouts(recs)
    R.to_csv(os.path.join(a.out, "readouts_all.csv"), index=False)
    for s in ("legacy", "rect150"):
        x = R[R.set == s]
        if len(x):
            x.to_csv(os.path.join(a.out, f"{s}_predictions.csv"), index=False)
    # wide file for paper/tools/score150.py --extra-wide PATH::key (key__v = VERTICAL yarns = our h)
    x = R[R.set == "rect150"]
    if len(x):
        W = pd.DataFrame({"image_id": sorted(x.image.astype(int).unique())}).set_index("image_id")
        p1 = x[x.method == "P1"].assign(i=lambda d: d.image.astype(int)).set_index("i")
        for (model, meth), g in x.groupby(["model", "method"]):
            g = g.assign(i=g.image.astype(int)).set_index("i")
            k = f"{model}@{meth}"
            W[f"{k}__v"] = g.vertical
            W[f"{k}__h"] = g.horizontal
            W[f"{k}__weave"] = p1[p1.model == model].weave if meth == "P1" else None
        W.reset_index().to_csv(os.path.join(a.out, "rect150_predictions_wide.csv"), index=False)
    if os.path.exists(a.val_labels) and (R.set == "synval").any():
        S, per = synval_scores(R, a.val_labels)
        S.round(3).to_csv(os.path.join(a.out, "synval_scores.csv"), index=False)
        per.round(3).to_csv(os.path.join(a.out, "synval_per_image.csv"), index=False)
        print(S[["model", "method", "n_images", "n_ok", "MAPE", "MAPE_lo", "MAPE_hi", "w10", "gross25",
                 "n_truncated_calls"]].round(2).to_string(index=False))
        M = point_matching(recs, a.val_labels)
        if len(M):
            M.round(4).to_csv(os.path.join(a.out, "synval_point_matching.csv"), index=False)
            agg = M.groupby(["model", "scale"]).agg(n_calls=("f1", "size"), precision=("precision", "mean"),
                                                     recall=("recall", "mean"), f1=("f1", "mean"),
                                                     med_offset_spacing=("med_offset_spacing", "median"),
                                                     pred_spacing_cv=("pred_spacing_cv", "median"),
                                                     gt_spacing_cv=("gt_spacing_cv", "median"),
                                                     truncated=("truncated", "sum")).reset_index()
            agg.round(4).to_csv(os.path.join(a.out, "synval_point_matching_summary.csv"), index=False)
            print(agg.round(3).to_string(index=False))
        if a.select_from:
            sel_path = os.path.join(a.out, "selection.json")
            c = S[(S.model == a.select_from) & S.method.isin(PRIMARY_CANDIDATES)].sort_values("MAPE")
            choice = dict(model=a.select_from, primary=c.iloc[0].method, rule="lowest synthetic-val MAPE among "
                          + ",".join(PRIMARY_CANDIDATES), synval=c[["method", "MAPE", "n_ok"]].round(3).to_dict("records"))
            if os.path.exists(sel_path):
                old = json.load(open(sel_path))
                print("selection already fixed:", old["primary"], "(not overwritten)")
            else:
                json.dump(choice, open(sel_path, "w"), indent=1)
                print("PRIMARY READOUT FIXED:", choice)


if __name__ == "__main__":
    main()
