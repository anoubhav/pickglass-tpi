"""eval_stress_classical.py -- frozen classical counters on PickGlass-Stress.

Runs methods/results/counting_frozen_final.py (conf_model.json from methods/, exactly as
methods/run_legacy_vlmcrops.py does) with estimate_many(img, ["ours", "gabor"], preproc="lcn",
glare="mask") on every render of camera_aug/stress/{set}/manifest.csv and on the clean source crops
(condition 'clean').  Resumable; one row per (render, estimator).

    <venv2>/bin/python eval_stress_classical.py --set legacy13 --workers 5
    -> results/stress_{set}_classical.csv
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import hashlib
import importlib.util
import json
import os
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FROZEN = os.path.join(ROOT, "methods", "results", "counting_frozen_final.py")
CONF_JSON = os.path.join(ROOT, "methods", "conf_model.json")
ESTS = ["ours", "gabor"]
_C = None


def frozen():
    global _C
    if _C is None:
        spec = importlib.util.spec_from_file_location("counting_frozen_final", FROZEN)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.CONF_MODEL = json.load(open(CONF_JSON))
        _C = m
    return _C


def _job(a):
    key, path, md5, ests = a
    cv2.setNumThreads(1)
    b = open(path, "rb").read()
    if md5 and hashlib.md5(b).hexdigest() != md5:
        raise RuntimeError(f"file changed: {path}")
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    t0 = time.time()
    res = frozen().estimate_many(img, ests, preproc="lcn", glare="mask")
    rows = []
    for est in ests:
        e = res[est]
        lat = e["debug"].get("lattice") or {}
        rows.append(dict(source_id=key[0], condition=key[1], rep=key[2], estimator=est, S_in=img.shape[0],
                         h=e["h_count"], v=e["v_count"], conf=e["confidence"], angle_x=lat.get("angle_x"),
                         angle_y=lat.get("angle_y"), wall_s=round(time.time() - t0, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="legacy13")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--conds", default="")
    ap.add_argument("--ests", default="ours,gabor")
    a = ap.parse_args()
    ests = a.ests.split(",")
    man = pd.read_csv(os.path.join(HERE, "stress", a.set, "manifest.csv"), dtype={"source_id": str})
    if a.conds:
        man = man[man.condition.isin(a.conds.split(","))]
    out = os.path.join(HERE, "results", f"stress_{a.set}_classical.csv")
    jobs = []
    for sid, g in man.groupby("source_id"):
        jobs.append(((sid, "clean", 0), os.path.join(ROOT, g.source_path.iloc[0]), g.source_md5.iloc[0], ests))
    jobs += [((r.source_id, r.condition, int(r.rep)), os.path.join(HERE, r.path), r.md5, ests)
             for r in man.itertuples()]
    old = pd.read_csv(out, dtype={"source_id": str}) if os.path.exists(out) else None
    if old is not None:
        done = set(zip(old.source_id, old.condition, old.rep))
        jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} images to count", flush=True)
    rows, t0 = [], time.time()
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))       # big files first (better packing)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows += r
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                d = pd.DataFrame(rows)
                if old is not None:
                    d = pd.concat([old, d], ignore_index=True)
                d.to_csv(out, index=False)
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
