"""rule1_histogram.py -- rule 1 of the plan for the reference annotator (section 2, item 1; fixed before A2's counts
were opened; the plan is not in this repository, SHA-256 9ae1d9ca..., see CHECKSUMS.md): the histogram of
r = log2(A2 / A1) over the 152 lines, the octave-disputed lines (|r| > 0.5), and the disputed lines among the lines on
which PG-Lattice reads about half or double A1 (the paper's band | |log2(PG-Lattice / A1)| - 1 | <= 0.3). Supplement Sec. S1 (half-octave histogram
1/29/12/62/24/16/8; 44 of the 72 disputed lines, 33 scenes, among the 64 band lines) and Tab. S17 (72 lines, 50 scenes).

PG-Lattice's counts are methods/predictions_ours_150.csv rounded to 3 decimals, as paper/tools/label_day.py stores
them in its per-scene table (main150_per_scene.csv), which the paper's numbers were computed from.

    python paper/tools/reference/rule1_histogram.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

Output: <out>/rule1_histogram.json (the numbers and the checks against the paper's values). One check is not against a
printed value: the 102 lines more than 10% apart are the value of \\BNlinesBig, a macro of the paper's sources that the
paper does not print (the check's label still says 'main text').
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
CHK = []


def chk(name, ok):
    CHK.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name)


def r1(x):
    return float(np.round(float(x) + 0.0, 1))


# PG-Lattice as in label_day.py's per-scene table (3 decimals)
p = pd.read_csv(os.path.join(ROOT, "methods/predictions_ours_150.csv")).set_index("id")
o = pd.DataFrame({"h_p": p.h_count, "v_p": p.v_count}).round(3)
o.index = o.index.astype(int)

a = pd.read_csv(args.a1).set_index("scene_id")
b = pd.read_csv(args.a2).set_index("scene_id")
rh, rv = np.log2(b.h_tpi / a.h_tpi), np.log2(b.v_tpi / a.v_tpi)
r = pd.concat([rh, rv]).to_numpy()
chk("152 lines", len(r) == 152 and np.isfinite(r).all())
edges = [-np.inf, -1.3, -0.7, -0.5, -0.14, 0.14, 0.5, 0.7, 1.3, 1.7, np.inf]
hist = np.histogram(r, bins=edges)[0]
E = dict(edges=[str(x) for x in edges], counts=[int(x) for x in hist],
         n_disputed=int((np.abs(r) > 0.5).sum()), n_disputed_scenes=int(((rh.abs() > 0.5) | (rv.abs() > 0.5)).sum()),
         n_within_10pct=int((np.abs(2 ** r - 1) <= 0.10).sum()), median_r=r1(np.median(r) * 100) / 100,
         n_neg_disputed=int((r < -0.5).sum()), n_pos_disputed=int((r > 0.5).sum()),
         n_about_half=int(((r >= -1.3) & (r <= -0.7)).sum()), n_about_double=int(((r >= 0.7) & (r <= 1.3)).sum()),
         n_about_triple=int((r > 1.3).sum()), n_below_third=int((r < -1.3).sum()),
         n_between_10pct_and_141=int(((np.abs(2 ** r - 1) > 0.10) & (np.abs(r) <= 0.5)).sum()))
hedges = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]          # half-octave bins centred on -1.5 .. +1.5
E["half_octave_edges"] = hedges
E["half_octave_counts"] = [int(x) for x in np.histogram(r, bins=hedges)[0]]
E["r_min_max"] = [float(np.round(r.min(), 3)), float(np.round(r.max(), 3))]
chk("every r inside the half-octave bins", sum(E["half_octave_counts"]) == 152)
chk("disputed lines / scenes = Tab. S17 caption (72 / 50)", E["n_disputed"] == 72 and E["n_disputed_scenes"] == 50)
# 102 = \BNlinesBig, a macro the paper does not print; the label keeps its original wording
chk("lines > 10% apart = main text (102)", 152 - E["n_within_10pct"] == 102)
# PG-Lattice's half-or-double lines against A1 (paper band | |log2(PG-Lattice/A1)| - 1 | <= 0.3; 64 lines, 45 scenes)
oh = np.log2(o.h_p.reindex(a.index) / a.h_tpi)
ov = np.log2(o.v_p.reindex(a.index) / a.v_tpi)
octh, octv = (np.abs(np.abs(oh) - 1) <= 0.3), (np.abs(np.abs(ov) - 1) <= 0.3)
n_oct = int(octh.sum() + octv.sum())
n_oct_sc = int((octh | octv).sum())
chk("PG-Lattice half-or-double lines against A1 = rule 4 band (64 lines, 45 scenes)", n_oct == 64 and n_oct_sc == 45)
disp_h, disp_v = rh.abs() > 0.5, rv.abs() > 0.5
E["octave_lines"] = n_oct
E["octave_scenes"] = n_oct_sc
E["disputed_among_octave_lines"] = int((octh & disp_h).sum() + (octv & disp_v).sum())
E["disputed_among_octave_lines_scenes"] = int(((octh & disp_h) | (octv & disp_v)).sum())
E["disputed_outside_octave_lines"] = E["n_disputed"] - E["disputed_among_octave_lines"]
E["checks"] = dict(n=len(CHK), passed=sum(ok for _, ok in CHK), failed=[nm for nm, ok in CHK if not ok])
os.makedirs(args.out, exist_ok=True)
json.dump(E, open(os.path.join(args.out, "rule1_histogram.json"), "w"), indent=1)
print(json.dumps(E, indent=1))
