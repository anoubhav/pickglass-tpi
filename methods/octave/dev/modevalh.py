"""Evaluate methods/octave/octave_v1.py (the module itself) on synthetic (fresh seeds) and legacy13."""
import os, sys, json, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/methods")
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
import octave_v1h as O

def job(a):
    src, key, th, tv = a
    cv2.setNumThreads(1)
    if src == "synth":
        crop = cv2.imread(os.path.join(HERE, "synthC" if key.startswith("c") else "synth", key))
    else:
        import legacy as L
        crop = L.rectify_legacy(key)[0]
    r = O.estimate(crop); r.pop("frozen_debug")
    r.update(src=src, key=key, true_h=th, true_v=tv, S=crop.shape[0])
    return r

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "legacy":
        L13 = pd.read_csv(f"{ROOT}/methods/results/legacy13_per_image.csv"); L13 = L13[L13.gt_usable & L13.gt_warp.notna()]
        jobs = [("legacy", r.image, r.gt_warp, r.gt_weft) for r in L13.itertuples()]
    elif which == "synthC":
        lab = pd.read_csv(os.path.join(HERE, "synthC", "labels.csv"))
        jobs = [("synth", r.file, r.h_count, r.v_count) for r in lab.itertuples()]
    else:
        lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
        jobs = [("synth", r.file, r.h_count, r.v_count) for r in lab.itertuples()]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "9"))) as p: rows = p.map(job, jobs, chunksize=2)
    df = pd.DataFrame(rows); df.to_pickle(os.path.join(HERE, f"modh_{which}_" + os.environ.get("HMODE","gsub") + ".pkl")); print("done", len(df), time.time() - t0)
