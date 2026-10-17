"""logratio_endpoint.py -- the log-ratio endpoint: mean |log2(prediction / reference)| over the scene axes and the share
of axes with |log2| > 0.5, against A2 alone, A1 alone and their mean (main Sec. 5.1; supplement Sec. S1, Tab. S15).

Pre-specified as item 5 of the plan for the reference annotator (not in this repository; SHA-256 9ae1d9ca..., see
CHECKSUMS.md), fixed before A2's counts were opened: the metric against A2 and against the mean, and the octave-v1 minus
PG-Lattice paired change in mean |log2| against A2 (scene bootstrap 10,000, seed 20261012), a secondary endpoint. Each
annotator as a prediction of the other and the paired comparisons with A1's counts (method minus A1, against A2) are
post hoc.
Bootstrap: one numpy default_rng(20261012) shared by the four paired comparisons, in the order written below.

    python paper/tools/reference/logratio_endpoint.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

Output: <out>/logratio_endpoint.json; keys "<method> vs <reference>" (reference A2, A1 or mean; an annotator is not
scored against itself) and the four paired rows "... (paired mean |log2|)".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
args = ap.parse_args()

A = pd.read_csv(args.a1).set_index("scene_id"); B = pd.read_csv(args.a2).set_index("scene_id")
R = {"A2": B, "A1": A, "mean": pd.DataFrame({"h_tpi": (A.h_tpi + B.h_tpi) / 2, "v_tpi": (A.v_tpi + B.v_tpi) / 2})}
w = pd.read_csv(os.path.join(ROOT, "vlm/results/rect150_P1_predictions_wide.csv")).set_index("image_id")
rd = lambda p: pd.read_csv(os.path.join(ROOT, p)).set_index("id").rename(columns={"h_count": "h", "v_count": "v"})
M = {"PG-Lattice": rd("methods/predictions_ours_150.csv"), "octave-v1": rd("methods/octave/predictions_octave_150.csv"),
     "GPT-6.1-sol P1": pd.DataFrame({"h": pd.to_numeric(w["gpt-6.1-sol__v"], errors="coerce"), "v": pd.to_numeric(w["gpt-6.1-sol__h"], errors="coerce")}),
     "CNN fabric-grouped": rd("methods/ensemble/cnn_fabric_oof_FLSA_150.csv"), "ens-v1": rd("methods/ensemble/predictions_ens_v1_150.csv"),
     "A1 (as prediction)": A.rename(columns={"h_tpi": "h", "v_tpi": "v"}), "A2 (as prediction)": B.rename(columns={"h_tpi": "h", "v_tpi": "v"})}
def per_scene(p, ref):
    l = [np.abs(np.log2(p[ax].reindex(ref.index) / ref[f"{ax}_tpi"])) for ax in "hv"]
    l = [x.fillna(np.log2(100)) for x in l]   # missing prediction: large penalty (not used by these methods)
    return (l[0] + l[1]) / 2, pd.concat(l)
out = {}
for rn, ref in R.items():
    for mn, p in M.items():
        if mn.startswith(rn): continue
        s, axes = per_scene(p, ref)
        out[f"{mn} vs {rn}"] = dict(mean_abs_log2=round(float(axes.mean()), 3), share_gt_0_5=round(float((axes > 0.5).mean()), 3))
rng = np.random.default_rng(20261012)
d = (per_scene(M["octave-v1"], B)[0] - per_scene(M["PG-Lattice"], B)[0]).values
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
out["octave-v1 minus PG-Lattice vs A2 (paired mean |log2|)"] = dict(diff=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
for k2, (m1, m2, rr) in {"GPT minus A1, vs A2": ("GPT-6.1-sol P1", "A1 (as prediction)", B), "ens-v1 minus A1, vs A2": ("ens-v1", "A1 (as prediction)", B),
                        "GPT minus A2, vs A1": ("GPT-6.1-sol P1", "A2 (as prediction)", A)}.items():
    d = (per_scene(M[m1], rr)[0] - per_scene(M[m2], rr)[0]).values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    out[k2 + " (paired mean |log2|)"] = dict(diff=round(float(d.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "logratio_endpoint.json"), "w"), indent=1)
for k, v in out.items(): print(f"{k:45s} {v}")
