"""ruler_vs_ours_ref.py -- main Sec. 5.5 (post hoc): PG-Lattice on the window found from the glass's own ruler minus
PG-Lattice on the hand-marked window, against A2 (the textile-trained reference), paired over the scenes the ruler
calibrates (75 of 76), label_day.py paired() semantics (per-scene mean of h/v APE; 10,000 scene resamples,
default_rng(0), percentile CI).

The per-scene APEs are those of paper/tools/label_day.py's per-scene table (main150_per_scene.csv, 3 decimals) on A2's
counts: rebuilt here from --a2 (counts to 2 decimals, as the tool export) with label_day.per_scene_ape() and round(3),
or read with --per-scene from a label_day.py single-annotator run on A2's export. Ruler-crop predictions:
paper/tables/src/extra150_wide.csv (pg-lattice@ruler).

    python paper/tools/reference/ruler_vs_ours_ref.py [--a2 A2.csv] [--per-scene main150_per_scene.csv]

Prints the result (n, MAPE of each, difference and its CI).
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--per-scene", default=None, help="optional: label_day.py per-scene table of a run on A2's export")
args = ap.parse_args()

OURS, RULER = "Ours (lattice + fusion)", "pg-lattice@ruler"
if args.per_scene:
    d = pd.read_csv(args.per_scene)
else:
    spec = importlib.util.spec_from_file_location("label_day_ro", os.path.join(ROOT, "paper", "tools", "label_day.py"))
    LD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LD)
    S = LD.load_score150()
    preds = S.load_predictions()
    preds.update(S.load_extra_wide([os.path.join(ROOT, "paper", "tables", "src", "extra150_wide.csv") + "::" + RULER]))
    L = pd.read_csv(args.a2)
    L["scene_id"] = L.scene_id.astype(int)
    L = L.set_index("scene_id").sort_index()
    gt = pd.DataFrame({ax: pd.to_numeric(L[f"{ax}_tpi"], errors="coerce").round(2) for ax in ("h", "v")})
    gt = gt.where(gt > 0)   # score150.load_labels: a count <= 0 is missing
    per = []
    for name in (OURS, RULER):   # label_day.py step_score: per_scene_ape(gt, pr), written with round(3)
        pr = preds[name].copy()
        pr.index = pr.index.astype(int)
        e = LD.per_scene_ape(gt, pr)
        e.insert(0, "method", name)
        per.append(e.reset_index().rename(columns={"index": "scene_id"}))
    d = pd.concat(per, ignore_index=True).round(3)
print(sorted(d.method.unique()))
d["s"] = (d.ape_h + d.ape_v) / 2
a = d[d.method == OURS].set_index("scene_id").s
rul = [m for m in d.method.unique() if "ruler" in m.lower()]
print("ruler methods:", rul)
b = d[d.method == rul[0]].set_index("scene_id").s.dropna()
idx = b.index.intersection(a.dropna().index); x = (b[idx] - a[idx]).values
rng = np.random.default_rng(0); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(10000)]
print(f"n={len(x)} ruler {b[idx].mean():.2f} ours {a[idx].mean():.2f} diff {x.mean():+.2f} CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
