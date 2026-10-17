"""molmo_reshoot.py -- re-shoot self-consistency of Molmo2-4B + synthetic LoRA (P1 readout) for main Tab. 2 and
Sec. 5.3 (post hoc: the row was added after labeling), with the exact definitions of reshoot_consistency.py (rule 1 of
addendum 1 to the plan for the reference annotator; not in this repository, SHA-256 586b606c..., see CHECKSUMS.md).
Label-free. The PG-Lattice and GPT-6.1-sol rows are recomputed and, when reshoot_consistency.json exists, checked
against it.

    python paper/tools/reference/molmo_reshoot.py [--out DIR] [--reshoot-json <out>/reshoot_consistency.json]

Output: <out>/molmo_reshoot.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
ap.add_argument("--reshoot-json", default=None,
                help="reshoot_consistency.py output for the check (default <out>/reshoot_consistency.json)")
args = ap.parse_args()

pairs = [(p["scene_a"], p["scene_b"]) for p in json.load(open(os.path.join(ROOT, "paper/tables/fabric_groups_v2.json")))["accepted_pairs"]]
def d(x, y):
    rel = lambda a, b: abs(a - b) / ((a + b) / 2)
    direct = max(rel(x[0], y[0]), rel(x[1], y[1])); swapped = max(rel(x[0], y[1]), rel(x[1], y[0]))
    near2 = any(1.6 <= max(a, b) / min(a, b) <= 2.4 for a, b in (((x[0], y[0]), (x[1], y[1])) if direct <= swapped else ((x[0], y[1]), (x[1], y[0]))))
    return min(direct, swapped), near2
src = {}
p = pd.read_csv(os.path.join(ROOT, "methods/predictions_ours_150.csv")).set_index("id"); src["PG-Lattice"] = {s: (p.at[s, "h_count"], p.at[s, "v_count"]) for s in p.index}
w = pd.read_csv(os.path.join(ROOT, "vlm/results/rect150_P1_predictions_wide.csv")).set_index("image_id")
src["GPT-6.1-sol"] = {s: (pd.to_numeric(w.at[s, "gpt-6.1-sol__v"], errors="coerce"), pd.to_numeric(w.at[s, "gpt-6.1-sol__h"], errors="coerce")) for s in w.index}
m = pd.read_csv(os.path.join(ROOT, "ft_results/results/rect150_predictions_wide.csv")).set_index("image_id")
k = "molmo2_4b_synthft@P1"
src["Molmo2-4B + LoRA P1"] = {s: (pd.to_numeric(m.at[s, f"{k}__v"], errors="coerce"), pd.to_numeric(m.at[s, f"{k}__h"], errors="coerce")) for s in m.index}
out = {}
for nm, S in src.items():
    v = [d(S[a], S[b]) for a, b in pairs if a in S and b in S and all(np.isfinite(S[a])) and all(np.isfinite(S[b]))]
    dd = np.array([x[0] for x in v])
    out[nm] = dict(n=len(v), median_pct=round(100 * float(np.median(dd)), 1), within3=int((dd <= .03).sum()), within5=int((dd <= .05).sum()),
                   within10=int((dd <= .10).sum()), near2x=int(sum(x[1] for x in v)))
rj = args.reshoot_json or os.path.join(args.out, "reshoot_consistency.json")
if os.path.exists(rj):
    ref = json.load(open(rj))
    out["check_vs_reshoot_consistency_json"] = {nm: out[nm] == ref[nm] for nm in ("PG-Lattice", "GPT-6.1-sol")}
else:
    out["check_vs_reshoot_consistency_json"] = f"not checked ({rj} not found; run reshoot_consistency.py first)"
out["status"] = ("post hoc: Molmo2-4B row added to Tab. 2 after labeling; definitions fixed in advance in rule 1 of addendum 1 "
                 "to the plan for the reference annotator (not in this repository; SHA-256 586b606c..., see CHECKSUMS.md)")
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, "molmo_reshoot.json"), "w"), indent=1)
for k2, v2 in out.items(): print(k2, v2)
