"""branchB_numbers.py -- agreement numbers of the two annotators used in the paper (main Secs. 3 and 5.4; supplement
Secs. S1, S4 and S9). Post hoc unless noted.

  same_photo_median_absdiff_pct_rel_to_pair_mean   median over the 152 lines of |A2 - A1| / ((A2 + A1) / 2) (Sec. 3: 32.0%)
  N2_lines_diff_gt10, N2_of_which_about_2x          lines with A2/A1 off by more than 10%, and of them those in
                                                    0.4-0.62x or 1.6-2.4x
  prereg_2.1_disputed_lines / _scenes               |log2(A2/A1)| > 0.5 (item 1 of section 2 of the plan for the reference
                                                    annotator, not in this repository, SHA-256 9ae1d9ca..., see
                                                    CHECKSUMS.md; Tab. S17)
  reshoot_by_resolution_A1 / _A2                    re-shoot median difference (reshoot_consistency.py definition, without
                                                    the near-2x tally) for pairs of two full-resolution, two quarter-resolution
                                                    or mixed photos (Sec. S4)
  N3_iqa_vs_disagreement_post_hoc                   AUROC of each generic quality score (paper/tables/iqa150_scores.csv) for
                                                    the scenes on which A1 and A2 differ by more than 10% on an axis; scene
                                                    bootstrap 2,000, numpy default_rng(20261012) (Sec. S9)

    python paper/tools/reference/branchB_numbers.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

The per-scene count files need the columns scene_id, h_tpi, v_tpi and scene_resolution ('full' / 'quarter_only').
Output: <out>/branchB_numbers.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi, scene_resolution")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi, scene_resolution")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
args = ap.parse_args()

A = pd.read_csv(args.a1).set_index("scene_id"); B = pd.read_csv(args.a2).set_index("scene_id")
res = {}
r = pd.concat([np.log2(B.h_tpi / A.h_tpi), np.log2(B.v_tpi / A.v_tpi)])
q = 2 ** r
big = (np.abs(q - 1) > 0.10); oct2 = q.between(0.4, 0.62) | q.between(1.6, 2.4)
res["N2_lines_diff_gt10"] = int(big.sum()); res["N2_of_which_about_2x"] = int((big & oct2).sum())
res["prereg_2.1_disputed_lines_abs_log2_gt_0.5"] = int((r.abs() > 0.5).sum())
res["prereg_2.1_disputed_scenes"] = int(((np.log2(B.h_tpi / A.h_tpi).abs() > 0.5) | (np.log2(B.v_tpi / A.v_tpi).abs() > 0.5)).sum())
d = (B[["h_tpi", "v_tpi"]] - A[["h_tpi", "v_tpi"]]).abs() / ((B[["h_tpi", "v_tpi"]] + A[["h_tpi", "v_tpi"]]) / 2)
res["same_photo_median_absdiff_pct_rel_to_pair_mean"] = round(100 * float(np.median(d.values)), 1)
# re-shoot by resolution (A1 and A2)
pairs = [(p["scene_a"], p["scene_b"]) for p in json.load(open(os.path.join(ROOT, "paper/tables/fabric_groups_v2.json")))["accepted_pairs"]]
def rd(L, a, b):
    rel = lambda x, y: abs(x - y) / ((x + y) / 2)
    return min(max(rel(L.at[a, "h_tpi"], L.at[b, "h_tpi"]), rel(L.at[a, "v_tpi"], L.at[b, "v_tpi"])),
               max(rel(L.at[a, "h_tpi"], L.at[b, "v_tpi"]), rel(L.at[a, "v_tpi"], L.at[b, "h_tpi"])))
for nm, L in (("A1", A), ("A2", B)):
    by = {}
    for a, b in pairs:
        k = "both_full" if (L.at[a, "scene_resolution"] == "full" and L.at[b, "scene_resolution"] == "full") else ("both_quarter" if L.at[a, "scene_resolution"] != "full" and L.at[b, "scene_resolution"] != "full" else "mixed")
        by.setdefault(k, []).append(rd(L, a, b))
    res[f"reshoot_by_resolution_{nm}"] = {k: dict(n=len(v), median_pct=round(100 * float(np.median(v)), 1)) for k, v in by.items()}
# N3: generic IQA vs 'annotators differ > 10% on an axis' (post hoc), scene bootstrap
iq = pd.read_csv(os.path.join(ROOT, "paper/tables/iqa150_scores.csv"))
idc = "image" if "image" in iq.columns else iq.columns[0]
iq = iq.set_index(idc)
y = ((np.abs(B.h_tpi / A.h_tpi - 1) > 0.10) | (np.abs(B.v_tpi / A.v_tpi - 1) > 0.10)).astype(int)
from scipy.stats import mannwhitneyu
rng = np.random.default_rng(20261012); out = {}
cols = [c for c in iq.columns if any(k in c.lower() for k in ("brisque", "niqe", "musiq", "topiq", "clip", "eta", "rho"))]
for c in cols:
    x = pd.to_numeric(iq[c].reindex(y.index), errors="coerce"); m = x.notna(); xx, yy = x[m].values, y[m].values
    def auc(xv, yv):
        p, n = xv[yv == 1], xv[yv == 0]
        return mannwhitneyu(p, n).statistic / (len(p) * len(n)) if len(p) and len(n) else np.nan
    a0 = auc(xx, yy); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(xx), len(xx)); bs.append(auc(xx[i], yy[i]))
    out[c] = dict(auroc=round(float(a0), 2), ci=[round(float(np.nanpercentile(bs, 2.5)), 2), round(float(np.nanpercentile(bs, 97.5)), 2)], n_pos=int(yy.sum()), n=int(len(yy)))
res["N3_iqa_vs_disagreement_post_hoc"] = out
os.makedirs(args.out, exist_ok=True)
json.dump(res, open(os.path.join(args.out, "branchB_numbers.json"), "w"), indent=1); print(json.dumps(res, indent=1))
