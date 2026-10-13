"""eval_octave_v1.py -- the PRE-SPECIFIED evaluation of "octave-v1" (frozen with the method,
SHA-256 in methods/octave/PREREG_octave_v1.json).  Written before any count of the second
annotator (FL_NG) existed; it reads no label file except the ones given on the command line.

What it scores (one row per SCENE = the scene representative's rectified crop; scene_id = rep id)
------------------------------------------------------------------------------------------------
  reference : per-scene human counts h_tpi / v_tpi from 1 or 2 per-annotator scene CSVs with the
              header  scene_id,scene_members,scene_resolution,iaa,rect_size,h_tpi,v_tpi,h_count,
              v_count,weave,warp_dir,h_angle_deg,v_angle_deg,uncertain,confidence,n_annotations,
              annotators,label_src_id,label_src_kind,n_labelled_files,labelled
              --reference each : every file is scored separately (one block per file)
              --reference mean : per scene and axis, mean of the files that have a positive,
                                 finite count for that axis (n of single-annotator scenes reported)
              A scene enters a block when both reference axes are finite and > 0 (and `labelled`
              is truthy when that column exists) and its id is a manifest scene representative.
  methods   : octave-v1   methods/octave/predictions_octave_150.csv   (h_count, v_count by id)
              ours-frozen methods/predictions_ours_150.csv            (h_count, v_count by id)
              gpt-6.1-sol P1  vlm/results/rect150_P1_predictions_wide.csv
                          axis convention of paper/tools/score150.py: VLM "<m>__v" = VERTICAL
                          yarns = our h, "<m>__h" = horizontal yarns = our v.
              Axis convention (same as score150.py): reference h_tpi <-> prediction h_count
              (vertical yarns crossing the horizontal guide line), v_tpi <-> v_count.
  per scene : APE_ax = 100 |pred - ref| / ref ;  scene MAPE = (APE_h + APE_v) / 2 ;
              both5 = (APE_h <= 5) and (APE_v <= 5).  A missing / non-positive prediction counts
              as APE = 100 on that axis (coverage reported), so every method is scored on the
              same scenes.
  summary   : mean scene MAPE, both5 share (%), coverage; PAIRED octave-v1 minus other:
              mean difference of scene MAPE with a scene bootstrap 95% CI (10,000 resamples of
              scenes with replacement, numpy default_rng(20261012), percentile interval) and a
              two-sided Wilcoxon signed-rank p (scipy, zero_method='wilcox'); both5 share
              difference with the same bootstrap CI and an exact two-sided McNemar p.

Pre-registered analyses (see PREREG_octave_v1.json): PRIMARY = --labels <FL_NG scene csv>
--reference each; SECONDARY = --labels <FL_NG> <FL_SA> --reference mean, and FL_SA alone
(post hoc: FL_SA's counts revealed the failure mode).

    python eval_octave_v1.py --labels FL_NG_scenes.csv [FL_SA_scenes.csv] --reference each \
        --out-json out.json --out-txt out.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
P_OCT = os.path.join(HERE, "predictions_octave_150.csv")
P_OURS = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
P_VLM = os.path.join(ROOT, "vlm", "results", "rect150_P1_predictions_wide.csv")
MANIFEST = os.path.join(ROOT, "annotation_tool", "data", "manifest.json")
GPT_KEY = "gpt-6.1-sol"
N_BOOT = 10000
SEED = 20261012


def load_preds(p_oct=P_OCT, p_ours=P_OURS, p_vlm=P_VLM):
    o = pd.read_csv(p_oct).set_index("id")
    f = pd.read_csv(p_ours).set_index("id")
    w = pd.read_csv(p_vlm).set_index("image_id")
    return {
        "octave-v1": pd.DataFrame({"h": o["h_count"], "v": o["v_count"]}),
        "ours-frozen": pd.DataFrame({"h": f["h_count"], "v": f["v_count"]}),
        f"{GPT_KEY} P1": pd.DataFrame({"h": pd.to_numeric(w[f"{GPT_KEY}__v"], errors="coerce"),
                                       "v": pd.to_numeric(w[f"{GPT_KEY}__h"], errors="coerce")}),
    }


def _truthy(x):
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    try:
        return bool(int(x))
    except (TypeError, ValueError):
        return bool(x)


def load_reference(path, reps):
    d = pd.read_csv(path)
    for c in ("scene_id", "h_tpi", "v_tpi"):
        if c not in d:
            raise SystemExit(f"{path}: missing column {c}")
    d["scene_id"] = d["scene_id"].astype(int)
    for ax in ("h", "v"):
        d[ax] = pd.to_numeric(d[f"{ax}_tpi"], errors="coerce")
        d.loc[~(d[ax] > 0), ax] = np.nan
    if "labelled" in d:
        d.loc[~d["labelled"].map(_truthy), ["h", "v"]] = np.nan
    d = d[d["scene_id"].isin(reps)].drop_duplicates("scene_id", keep="last").set_index("scene_id")
    return d[["h", "v"]]


def scene_scores(ref, pred):
    j = ref.join(pred, rsuffix="_p", how="left")
    out = pd.DataFrame(index=ref.index)
    for ax in ("h", "v"):
        p = j[f"{ax}_p"]
        ape = 100 * (p - j[ax]).abs() / j[ax]
        ape[~(p > 0)] = 100.0  # missing / non-positive prediction
        out[f"ape_{ax}"] = ape
        out[f"have_{ax}"] = (p > 0).astype(bool)
    out["mape"] = 0.5 * (out.ape_h + out.ape_v)
    out["both5"] = (out.ape_h <= 5) & (out.ape_v <= 5)
    return out


def mcnemar_exact(a, b):
    """two-sided exact McNemar p for paired booleans a, b"""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def paired(sa, sb, n_boot=N_BOOT, seed=SEED):
    from scipy.stats import wilcoxon
    d = (sa.mape - sb.mape).to_numpy()
    e = (sa.both5.astype(float) - sb.both5.astype(float)).to_numpy()
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bd = d[idx].mean(axis=1)
    be = e[idx].mean(axis=1)
    try:
        wp = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        wp = 1.0
    return {"n": n, "mape_diff": float(d.mean()), "mape_diff_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "wilcoxon_p": wp, "n_better": int(np.sum(d < 0)), "n_worse": int(np.sum(d > 0)), "n_tied": int(np.sum(d == 0)),
            "both5_diff_pts": float(100 * e.mean()),
            "both5_diff_ci95_pts": [float(100 * np.percentile(be, 2.5)), float(100 * np.percentile(be, 97.5))],
            "both5_mcnemar_p": mcnemar_exact(sa.both5.to_numpy(), sb.both5.to_numpy())}


def evaluate(ref, preds, label, n_boot=N_BOOT, seed=SEED):
    ref = ref.dropna()
    sc = {m: scene_scores(ref, p) for m, p in preds.items()}
    block = {"reference": label, "n_scenes": int(len(ref)), "methods": {}, "paired_octave_minus": {}}
    for m, s in sc.items():
        block["methods"][m] = {"MAPE": float(s.mape.mean()), "both5_pct": float(100 * s.both5.mean()),
                               "coverage_pct": float(100 * (s.have_h & s.have_v).mean()),
                               "MdAPE_axis": float(np.median(np.r_[s.ape_h, s.ape_v]))}
    for m in preds:
        if m != "octave-v1":
            block["paired_octave_minus"][m] = paired(sc["octave-v1"], sc[m], n_boot, seed)
    block["per_scene"] = {m: s[["ape_h", "ape_v", "mape", "both5"]].round(3).reset_index().to_dict("records")
                          for m, s in sc.items()}
    return block


def text_table(blocks):
    lines = []
    for b in blocks:
        lines.append(f"== reference: {b['reference']}  (n = {b['n_scenes']} scenes)")
        lines.append(f"{'method':18s} {'MAPE %':>8s} {'both5 %':>8s} {'cover %':>8s}")
        for m, r in b["methods"].items():
            lines.append(f"{m:18s} {r['MAPE']:8.2f} {r['both5_pct']:8.1f} {r['coverage_pct']:8.1f}")
        for m, r in b["paired_octave_minus"].items():
            lines.append(f"  octave-v1 - {m}: dMAPE {r['mape_diff']:+.2f} pts [95% CI {r['mape_diff_ci95'][0]:+.2f}, "
                         f"{r['mape_diff_ci95'][1]:+.2f}], Wilcoxon p = {r['wilcoxon_p']:.4g} "
                         f"(better/worse/tied {r['n_better']}/{r['n_worse']}/{r['n_tied']}); "
                         f"d both5 {r['both5_diff_pts']:+.1f} pts [{r['both5_diff_ci95_pts'][0]:+.1f}, "
                         f"{r['both5_diff_ci95_pts'][1]:+.1f}], McNemar p = {r['both5_mcnemar_p']:.4g}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="1 or 2 per-annotator scene CSVs")
    ap.add_argument("--names", nargs="*", default=None, help="display names (default: file basenames)")
    ap.add_argument("--reference", choices=("each", "mean"), default="each")
    ap.add_argument("--pred", default=P_OCT)
    ap.add_argument("--frozen", default=P_OURS)
    ap.add_argument("--gpt", default=P_VLM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)
    a = ap.parse_args(argv)
    if not 1 <= len(a.labels) <= 2:
        raise SystemExit("--labels takes 1 or 2 files")
    names = a.names or [os.path.splitext(os.path.basename(p))[0] for p in a.labels]
    reps = set(json.load(open(MANIFEST))["scenes"]["reps"])
    refs = [load_reference(p, reps) for p in a.labels]
    preds = load_preds(a.pred, a.frozen, a.gpt)
    blocks = []
    if a.reference == "each":
        for nm, r in zip(names, refs):
            blocks.append(evaluate(r, preds, nm, a.n_boot, a.seed))
    else:
        allidx = sorted(set().union(*[r.index for r in refs]))
        st = {ax: pd.concat([r[ax].reindex(allidx) for r in refs], axis=1) for ax in ("h", "v")}
        ref = pd.DataFrame({ax: st[ax].mean(axis=1, skipna=True) for ax in ("h", "v")}, index=allidx)
        nsingle = int(((st["h"].notna().sum(axis=1) < len(refs)) | (st["v"].notna().sum(axis=1) < len(refs))).sum())
        b = evaluate(ref, preds, "mean(" + ", ".join(names) + ")", a.n_boot, a.seed)
        b["n_scenes_single_annotator_axis"] = nsingle
        blocks.append(b)
    if len(reps) != 76 or any(b["n_scenes"] != 76 for b in blocks):
        print(f"NOTE: scored scenes per block = {[b['n_scenes'] for b in blocks]} (76 scene representatives in the manifest)",
              file=sys.stderr)
    out = {"script": os.path.basename(__file__), "labels": [os.path.abspath(p) for p in a.labels],
           "reference_mode": a.reference, "n_boot": a.n_boot, "seed": a.seed,
           "predictions": {"octave-v1": os.path.abspath(a.pred), "ours-frozen": os.path.abspath(a.frozen),
                           f"{GPT_KEY} P1": os.path.abspath(a.gpt)},
           "blocks": blocks}
    txt = text_table(blocks)
    print(txt)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
    if a.out_txt:
        with open(a.out_txt, "w") as f:
            f.write(txt + "\n")
    return out


if __name__ == "__main__":
    main()
