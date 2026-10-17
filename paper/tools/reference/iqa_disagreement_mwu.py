"""iqa_disagreement_mwu.py -- supplement Sec. S9 / main Sec. 5.4 (post hoc): two-sided Mann-Whitney p of each generic quality
score for the scenes on which the two annotators differ by more than 10% on an axis, Holm-adjusted over the ten
score x input pairs, and the AUROCs in both orientations (MUSIQ and TOPIQ-NR at the 1024-px input: 0.73 / 0.72,
p = 0.014 / 0.015, Holm 0.14).

Same data and target as branchB_numbers.py (N3):
  x = the ten *_work / *_fix1024 columns of paper/tables/iqa150_scores.csv (rows = scene representatives, id = scene_id)
  y = 1 if |A2/A1 - 1| > 0.10 on h or on v (per-scene counts)
Prints the numbers; with --iqa-summary (an iqa150_summary.json written by paper/tools/iqa150.py) it also prints that
run's Spearman CIs for PG-Lattice at both inputs.

    python paper/tools/reference/iqa_disagreement_mwu.py [--a1 A1.csv] [--a2 A2.csv] [--iqa-summary iqa150_summary.json]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--iqa-summary", default=None, help="optional iqa150_summary.json of a paper/tools/iqa150.py run")
args = ap.parse_args()

A = pd.read_csv(args.a1).set_index("scene_id")
B = pd.read_csv(args.a2).set_index("scene_id")
y = ((np.abs(B.h_tpi / A.h_tpi - 1) > 0.10) | (np.abs(B.v_tpi / A.v_tpi - 1) > 0.10)).astype(int)
iq = pd.read_csv(os.path.join(ROOT, "paper/tables/iqa150_scores.csv")).set_index("id")
reps = iq[iq.is_scene_rep]
assert (reps.index == reps.scene_id).all() and set(y.index) == set(reps.index), "scene mapping"
cols = [f"{s}_{p}" for p in ("work", "fix1024") for s in ("brisque", "niqe", "musiq", "clipiqa", "topiq_nr")]
print(f"N3 target: {int(y.sum())} positives of {len(y)} scenes")
rows = []
for c in cols:
    x = pd.to_numeric(iq[c].reindex(y.index), errors="raise")
    pos, neg = x[y == 1].values, x[y == 0].values
    r = mannwhitneyu(pos, neg, alternative="two-sided")       # scipy default: asymptotic, continuity correction
    auc = r.statistic / (len(pos) * len(neg))                 # stored orientation: P(score_pos > score_neg)
    rows.append(dict(col=c, auroc_stored=auc, p=float(r.pvalue)))
# Holm over the ten pairs
m = len(rows)
order = sorted(range(m), key=lambda i: rows[i]["p"])
run = 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * rows[i]["p"]))
    rows[i]["p_holm"] = run
for r in rows:
    print(f"  {r['col']:18s} AUROC(stored) {r['auroc_stored']:.3f}  flipped {1 - r['auroc_stored']:.3f}  "
          f"MWU two-sided p {r['p']:.4f}  Holm(10) {r['p_holm']:.4f}")
others = [r["p"] for r in rows if r["col"] not in ("musiq_fix1024", "topiq_nr_fix1024")]
print(f"  min p of the other eight pairs: {min(others):.4f}; min Holm p overall: {min(r['p_holm'] for r in rows):.4f}")

# Spearman CIs of an iqa150.py run (target: PG-Lattice = 'ours'), both inputs
if args.iqa_summary:
    S = json.load(open(args.iqa_summary))["results"]["ours"]
    for k, v in S.items():
        if "[" in k:
            print(f"  Spearman {k:20s} {v['spearman']:+.3f} [{v['spearman_lo']:+.3f}, {v['spearman_hi']:+.3f}] "
                  f"spans0={v['spearman_lo'] <= 0 <= v['spearman_hi']}")
