"""scaffold_insets_ref.py -- GPT-6.1-sol given four magnified 0.1-inch insets of the window (scaffold S3) against A2,
the textile-trained reference annotator: the paired tests behind main Sec. 5.1 ("level on both errors; the latter
beats the mean-density guess on MAPE only") and the scaffold paragraph of supplement Sec. S6. Its Tab. 1 row
(24.9% [20, 31]) comes from make_tab1_ref.py.

Read-only on the repository. Writes only scaffold_insets_ref.json in --out.

    python paper/tools/reference/scaffold_insets_ref.py [--a1 A1.csv] [--a2 A2.csv] [--out DIR]

The scaffold study pre-declared the insets as its primary input (vlm/scaffold/FROZEN.json), and their counts of the 150
photos were exported before any human count (vlm/scaffold/results/rect150_S3_predictions_wide.csv). Every comparison
below is POST HOC.

Scheme (the same as paired_ref.py):
  MAPE: frozen paper/tools/label_day.py paired() (imported, never edited): mean per-scene APE difference, 10,000 scene
        resamples, numpy default_rng(0), percentile 95% CI, Wilcoxon p;
  symmetric error: per scene, mean over the two counts of |log2(prediction / reference)|; same paired scheme;
  patches: fabric-clustered bootstrap over the 44 groups of paper/tables/fabric_groups_v2.csv (10,000,
        default_rng(20261012)).
Reference: A2's per-scene counts (--a2), unrounded, in ascending scene order (label_day's order; it fixes the bootstrap).
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--a1", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A1.csv"),
                help="A1's (non-expert's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--a2", default=os.path.join(ROOT, "data", "human_counts", "labels_per_scene_A2.csv"),
                help="A2's (reference annotator's) per-scene counts: scene_id, h_tpi, v_tpi")
ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tools", "reference", "out"), help="output directory")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.dont_write_bytecode = True
LD = load_module("label_day_frozen", f"{ROOT}/paper/tools/label_day.py")
S = LD.load_score150()
paired, per_scene_ape = LD.paired, LD.per_scene_ape


def counts(path):
    L = pd.read_csv(path).set_index("scene_id")
    L.index = L.index.astype(int)
    return pd.DataFrame({"h": L.h_tpi.astype(float), "v": L.v_tpi.astype(float)})


ORDER = sorted(counts(args.a2).index)
A2 = counts(args.a2).loc[ORDER]
A1 = counts(args.a1).loc[ORDER]


def wide(path, key):   # score150 axis convention: <key>__v = vertical yarns = our h
    w = pd.read_csv(f"{ROOT}/{path}").set_index("image_id")
    return pd.DataFrame({"h": pd.to_numeric(w[f"{key}__v"], errors="coerce"),
                         "v": pd.to_numeric(w[f"{key}__h"], errors="coerce")}).reindex(ORDER)


INSETS = "GPT-6.1-sol, insets (S3)"
P = {INSETS: wide("vlm/scaffold/results/rect150_S3_predictions_wide.csv", "gpt-6.1-sol@S3"),
     "Molmo2-4B + LoRA P1": wide("ft_results/results/rect150_predictions_wide.csv", "molmo2_4b_synthft@P1"),
     "GPT-6.1-sol P1": wide("vlm/results/rect150_P1_predictions_wide.csv", "gpt-6.1-sol"),
     "Leave-one-out mean": S.loo_mean(A2),
     "A1 counts": A1}
FG = pd.read_csv(f"{ROOT}/paper/tables/fabric_groups_v2.csv").set_index("scene_id")["fabric_group"]


def boot_clustered(cl, stat, n_boot=10000, seed=20261012):   # fabric-clustered bootstrap
    cl = np.asarray(cl)
    ug = np.unique(cl)
    members = [np.where(cl == u)[0] for u in ug]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = stat(np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def scene_log(gt, pr):
    j = gt[["h", "v"]].join(pr[["h", "v"]], rsuffix="_p", how="inner").dropna()
    return 0.5 * (np.abs(np.log2(j.h_p / j.h)) + np.abs(np.log2(j.v_p / j.v)))


def num(v):
    return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v


def mape_paired(b):
    r = {k: num(v) for k, v in paired(A2, P[INSETS], P[b]).items()}
    ea, eb = per_scene_ape(A2, P[INSETS]).ape, per_scene_ape(A2, P[b]).ape
    c = ea.index.intersection(eb.index)
    d = (ea[c] - eb[c]).to_numpy()
    r["fabric_ci"] = boot_clustered(FG.reindex(c).to_numpy(), lambda ix: d[ix].mean())
    return r


def log_paired(b, n_boot=10000, seed=0):
    ea, eb = scene_log(A2, P[INSETS]), scene_log(A2, P[b])
    c = ea.index.intersection(eb.index)
    d = (ea[c] - eb[c]).to_numpy()
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return dict(n=int(len(c)), log_a=float(ea[c].mean()), log_b=float(eb[c].mean()), delta=float(d.mean()),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                p_wilcoxon=float(stats.wilcoxon(d).pvalue),
                fabric_ci=boot_clustered(FG.reindex(c).to_numpy(), lambda ix: d[ix].mean()))


out = {"reference": "A2 (textile-trained), 76 scenes", "status": "POST HOC (the insets' counts were exported before "
       "any human count; the scaffold study pre-declared the insets as primary)", "mape_paired": {}, "log2_paired": {}}
for b in ("Molmo2-4B + LoRA P1", "GPT-6.1-sol P1", "Leave-one-out mean", "A1 counts"):
    out["mape_paired"][f"{INSETS} minus {b}"] = mape_paired(b)
    out["log2_paired"][f"{INSETS} minus {b}"] = log_paired(b)
json.dump(out, open(os.path.join(args.out, "scaffold_insets_ref.json"), "w"), indent=1)
for sec in ("mape_paired", "log2_paired"):
    for k, v in out[sec].items():
        print(f"{sec:11s} | {k:52s} | {v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]  patches "
              f"[{v['fabric_ci'][0]:+.3f}, {v['fabric_ci'][1]:+.3f}]")
