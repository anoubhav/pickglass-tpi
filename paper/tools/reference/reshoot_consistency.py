"""reshoot_consistency.py -- re-shoot self-consistency of each annotator and of the machine counters (main Tab. 2,
Sec. 5.3; supplement Sec. S4).

Definitions (binding: rule 1 of addendum 1 to the plan for the reference annotator, fixed before the reference
annotator's counts were opened; the addendum is not in this repository, SHA-256 586b606c..., see CHECKSUMS.md):
per accepted same-patch pair (paper/tables/fabric_groups_v2.json, 41 pairs) the larger relative axis difference
|a - b| / ((a + b) / 2) under the better of the direct and the swapped h-v assignment; the median over the pairs, the
pairs within 3 / 5 / 10%, and the pairs near 2x (ratio 1.6-2.4 on an axis of the chosen assignment).
Rows: A1 (non-expert), A2 (textile-trained reference), their mean, PG-Lattice, the CNN-VLM ensemble (ens-v1) and
GPT-6.1-sol P1. The machine rows need no labels (--machines-only).

    python paper/tools/reference/reshoot_consistency.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR] [--machines-only]

Output: <out>/reshoot_consistency.json (keys A1, A2, mean, PG-Lattice, ens-v1, GPT-6.1-sol).
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
ap.add_argument("--machines-only", action="store_true", help="only the label-free machine rows")
args = ap.parse_args()

pairs = [(p["scene_a"], p["scene_b"]) for p in json.load(open(os.path.join(ROOT, "paper/tables/fabric_groups_v2.json")))["accepted_pairs"]]
src = {}
if not args.machines_only:
    A = pd.read_csv(args.a1).set_index("scene_id"); B = pd.read_csv(args.a2).set_index("scene_id")
    src["A1"] = {s: (A.at[s, "h_tpi"], A.at[s, "v_tpi"]) for s in A.index}
    src["A2"] = {s: (B.at[s, "h_tpi"], B.at[s, "v_tpi"]) for s in B.index}
    src["mean"] = {s: ((src["A1"][s][0] + src["A2"][s][0]) / 2, (src["A1"][s][1] + src["A2"][s][1]) / 2) for s in A.index}
for nm, path in (("PG-Lattice", "methods/predictions_ours_150.csv"), ("ens-v1", "methods/ensemble/predictions_ens_v1_150.csv")):
    p = pd.read_csv(os.path.join(ROOT, path)).set_index("id"); src[nm] = {s: (p.at[s, "h_count"], p.at[s, "v_count"]) for s in p.index}
w = pd.read_csv(os.path.join(ROOT, "vlm/results/rect150_P1_predictions_wide.csv")).set_index("image_id")
# VLM axis convention (paper/tools/score150.py): <model>__v = vertical yarns = our h
src["GPT-6.1-sol"] = {s: (pd.to_numeric(w.at[s, "gpt-6.1-sol__v"], errors="coerce"), pd.to_numeric(w.at[s, "gpt-6.1-sol__h"], errors="coerce")) for s in w.index}
def d(x, y):
    rel = lambda a, b: abs(a - b) / ((a + b) / 2)
    direct = max(rel(x[0], y[0]), rel(x[1], y[1])); swapped = max(rel(x[0], y[1]), rel(x[1], y[0]))
    near2 = any(1.6 <= max(a, b) / min(a, b) <= 2.4 for a, b in (((x[0], y[0]), (x[1], y[1])) if direct <= swapped else ((x[0], y[1]), (x[1], y[0]))))
    return min(direct, swapped), near2
out = {}
for nm, S in src.items():
    v = [d(S[a], S[b]) for a, b in pairs if a in S and b in S and all(np.isfinite(S[a])) and all(np.isfinite(S[b]))]
    dd = np.array([x[0] for x in v])
    out[nm] = dict(n=len(v), median_pct=round(100 * float(np.median(dd)), 1), within3=int((dd <= .03).sum()), within5=int((dd <= .05).sum()),
                   within10=int((dd <= .10).sum()), near2x=int(sum(x[1] for x in v)))
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "reshoot_consistency.json"), "w"), indent=1)
for k, v in out.items(): print(f"{k:12s} {v}")
