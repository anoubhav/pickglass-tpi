"""rule4_band.py -- rule 4 of the plan for the reference annotator (section 2, item 4; fixed before A2's counts were
opened; the plan is not in this repository, SHA-256 9ae1d9ca..., see CHECKSUMS.md), evaluated on the lines where
PG-Lattice reads about half or double A1 (main Sec. 5.2, supplement Sec. S17; macros \\BNrfourN, \\BNrfourScenes,
\\BNrfourNGSA, \\BNrfourNGours).

Rule 4: on those lines, if A2 is within 10% of A1 on at least 80% of them, case 1 (the machines misread); else if A2
is within 10% of PG-Lattice on at least half, case 2 (the human reading is ambiguous); otherwise case 3 (both readings
stand). The plan names "47 scenes; 59 axis counts", which no symmetric band reproduces, so the rule is applied to the
paper's band | |log2(PG-Lattice / A1)| - 1 | <= 0.3 (64 lines, 45 scenes) and, as a sweep, to every band
| |log2(PG-Lattice / A1)| - 1 | <= tol for tol = 0.01, 0.0125, ... 0.7975 (one row per distinct (lines, scenes) pair;
"the same outcome on every band"). PG-Lattice's counts: methods/predictions_ours_150.csv (unrounded).

    python paper/tools/reference/rule4_band.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

Output: <out>/rule4_band.json: band, lines, scenes, a2_within10_of_a1, a2_within10_of_counter (line counts), case,
and the sweep.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OCT_TOL = 0.3   # the paper's half-or-double band (Sec. 5.2)
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
args = ap.parse_args()

SA = pd.read_csv(args.a1).set_index("scene_id")
NG = pd.read_csv(args.a2).set_index("scene_id")
P = pd.read_csv(os.path.join(ROOT, "methods/predictions_ours_150.csv")).set_index("id")
D = pd.DataFrame([(s, ax, SA.at[s, f"{ax}_tpi"], NG.at[s, f"{ax}_tpi"], P.at[s, f"{ax}_count"]) for s in SA.index for ax in "hv"],
                 columns=["scene", "ax", "sa", "ng", "p"])
D["d"] = (np.log2(D.p / D.sa).abs() - 1).abs()


def case(S):
    a = ((S.ng - S.sa).abs() / S.sa <= 0.10).mean(); b = ((S.ng - S.p).abs() / S.p <= 0.10).mean()
    return 1 if a >= 0.8 else (2 if b >= 0.5 else 3)


S = D[D.d <= OCT_TOL]
out = dict(band="| |log2(counter/A1)| - 1 | <= 0.3 (paper Sec. 5.2 band)", lines=int(len(S)), scenes=int(S.scene.nunique()),
           a2_within10_of_a1=int(((S.ng - S.sa).abs() / S.sa <= 0.10).sum()),
           a2_within10_of_counter=int(((S.ng - S.p).abs() / S.p <= 0.10).sum()), case=f"case {case(S)}")
# sweep over bands: (lines, scenes) -> first tol giving it, rule-4 case
seen = {}
for tol in np.arange(0.01, 0.80, 0.0025):
    T = D[D.d <= tol]
    key = (len(T), T.scene.nunique())
    if key in seen: continue
    seen[key] = (round(tol, 4), case(T))
out["sweep"] = [dict(lines=int(k[0]), scenes=int(k[1]), first_tol=float(v[0]), case=int(v[1])) for k, v in seen.items()]
out["sweep_cases"] = sorted({v[1] for v in seen.values()})
out["sweep_59_lines_47_scenes_reachable"] = (59, 47) in seen
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "rule4_band.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "sweep"})
print("(lines, scenes): (first tol, rule-4 case)")
for k, v in seen.items(): print(k, v)
print("59 lines & 47 scenes reachable:", (59, 47) in seen, "| any 59-line band:", [k for k in seen if k[0] == 59],
      "| any 47-scene band:", [k for k in seen if k[1] == 47])
print("cases over all bands:", sorted({v[1] for v in seen.values()}))
