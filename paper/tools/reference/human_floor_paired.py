"""human_floor_paired.py -- methods against the human noise floor: each method's error against one annotator minus the
other annotator's error against that annotator (main Sec. 5.1, Tab. 1 first row, Tab. 2 MAPE column; supplement S4).

Comparison pre-specified (rule 4 of addendum 1 to the plan for the reference annotator, not in this repository,
SHA-256 586b606c..., see CHECKSUMS.md: method-vs-A2 error against A1-vs-A2 error); the paired test and its scheme
are post hoc. Per scene: mean of the h and v APE (a missing prediction counts 100%);
method APE vs annotator X minus the other annotator's APE vs X; scene bootstrap 10,000 resamples, one numpy
default_rng(20261012) shared by all comparisons in the order written below (X = A2 first).

    python paper/tools/reference/human_floor_paired.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

Output: <out>/human_floor_paired.json; keys "<method> vs <X> minus <Y> vs <X>" with X, Y in {A1, A2}:
method = the method's MAPE, human = Y's MAPE, diff = method minus human, ci = 95% percentile CI of diff.
"PG-Lattice octave scenes vs <X> (band 0.14 ...)": scenes with an axis | |log2(PG-Lattice / X)| - 1 | <= 0.14 (the
stricter band of supplement Sec. S17, narrower than the paper's Sec. 5.2 half-or-double band of 0.3, which
make_branchB_macros.py uses for \\BNoursOct*).
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

L = {"A1": pd.read_csv(args.a1).set_index("scene_id"), "A2": pd.read_csv(args.a2).set_index("scene_id")}
w = pd.read_csv(os.path.join(ROOT, "vlm/results/rect150_P1_predictions_wide.csv")).set_index("image_id")
M = {"GPT-6.1-sol": pd.DataFrame({"h": pd.to_numeric(w["gpt-6.1-sol__v"], errors="coerce"), "v": pd.to_numeric(w["gpt-6.1-sol__h"], errors="coerce")}),
     "ens-v1": pd.read_csv(os.path.join(ROOT, "methods/ensemble/predictions_ens_v1_150.csv")).set_index("id").rename(columns={"h_count": "h", "v_count": "v"}),
     "PG-Lattice": pd.read_csv(os.path.join(ROOT, "methods/predictions_ours_150.csv")).set_index("id").rename(columns={"h_count": "h", "v_count": "v"})}
def ape(p, R):
    e = [100 * (p[ax].reindex(R.index) - R[f"{ax}_tpi"]).abs() / R[f"{ax}_tpi"] for ax in "hv"]
    return ((e[0].fillna(100) + e[1].fillna(100)) / 2)
out = {}; rng = np.random.default_rng(20261012)
for X, Y in (("A2", "A1"), ("A1", "A2")):
    R = L[X]; hum = ape(L[Y].rename(columns={"h_tpi": "h", "v_tpi": "v"}), R)
    for nm, p in M.items():
        d = (ape(p, R) - hum).values
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        out[f"{nm} vs {X} minus {Y} vs {X}"] = dict(method=round(float(ape(p, R).mean()), 1), human=round(float(hum.mean()), 1), diff=round(float(d.mean()), 1), ci=[round(float(np.percentile(b, 2.5)), 1), round(float(np.percentile(b, 97.5)), 1)])
    oct_ = 0
    o = M["PG-Lattice"]
    for s in R.index:
        if any(abs(abs(np.log2(o.at[s, ax] / R.at[s, f"{ax}_tpi"])) - 1) <= 0.14 for ax in "hv"): oct_ += 1
    out[f"PG-Lattice octave scenes vs {X} (band 0.14 as in Sec. S17, not the 0.3 band of Sec. 5.2)"] = oct_
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "human_floor_paired.json"), "w"), indent=1)
for k, v in out.items(): print(k, v)
