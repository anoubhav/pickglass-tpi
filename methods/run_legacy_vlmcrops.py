"""run_legacy_vlmcrops.py -- classical counters on the CANONICAL legacy crops (one crop rule).

Canonical legacy input = vlm/crops/legacy/{name}.png, i.e. exactly the pixels every VLM,
Molmo2 (Mac MLX and GPU bf16) and CountGD saw: vlm/data_vlm.detect_legacy_window trims the
dark pick-glass frame (absolute dark level 40, axis-aligned), the remainder is resized to a
square of side round((h+w)/2) with INTER_AREA and taken as the 1-inch window.

What runs here (nothing is re-tuned; the counter is the FROZEN file):
  * ours + every estimator of the frozen counter (methods/results/counting_frozen_final.py,
    loaded by path; conf_model.json loaded from methods/), estimate_many(..., preproc='lcn',
    glare='mask') exactly as ablate.py does for the legacy13 runs.
  * legacy 'predicted weft by RGB irregularity' (FR-final attempt/test.py, weft only), re-run on
    the crops with the variant that reproduces the legacy spreadsheet column (no Gaussian blur:
    14/20 exact, others within 3; see legacy_2019/repro_testpy_rgb.py).
  * latency: ours alone, sequential, default OpenCV threading, per image.
  * --check-p1: re-run ours on the ORIGINAL images (whole image = 1 inch, protocol P1) and
    compare with results/legacy13_estimators.csv (reproduction check of the frozen file).

    <venv2>/bin/python run_legacy_vlmcrops.py --out results/legacy13_vlmcrops_estimators.csv --check-p1
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FROZEN = os.path.join(HERE, "results", "counting_frozen_final.py")
CONF_JSON = os.path.join(HERE, "conf_model.json")
CROPS = os.path.join(ROOT, "vlm", "crops", "legacy")
ORIG = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "FR-final attempt", "images")
NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
ESTS = ["ours", "lattice", "fft", "fft_harm", "profile", "peaks", "gabor", "tiles", "fusion3"]


def frozen():
    spec = importlib.util.spec_from_file_location("counting_frozen_final", FROZEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.CONF_MODEL = json.load(open(CONF_JSON))  # same model counting.py loads (fitted on synthetic only)
    return m


def orig_path(n):
    p = os.path.join(ORIG, f"{n}.jpg")
    return p if os.path.exists(p) else os.path.join(ORIG, f"{n} .jpg")


def load(n, source):
    return cv2.imread(os.path.join(CROPS, f"{n}.png") if source == "vlmcrop" else orig_path(n), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- legacy test.py (weft only)
def _crossings(ch):
    avg = sum(ch) / len(ch)
    c = 0
    for i in range(len(ch) - 1):
        if ch[i] < avg and ch[i + 1] > avg:
            c += 1
        elif ch[i] > avg and ch[i + 1] < avg:
            c += 1
    return c


def testpy_weft(img, blur=False):
    """FR-final attempt/test.py L17-L120 headless; blur=False reproduces the legacy xlsx column."""
    img = cv2.resize(img, (600, 400))
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    temp = img[:, 100:105, :].astype(np.int64)
    b = [sum(temp[i, :, 0]) / 5 for i in range(img.shape[0])]
    g = [sum(temp[i, :, 1]) / 5 for i in range(img.shape[0])]
    r = [sum(temp[i, :, 2]) / 5 for i in range(img.shape[0])]
    return sum([_crossings(b), _crossings(r), _crossings(g)]) // 3


# ------------------------------------------------------------------ worker
def _job(a):
    n, source = a
    cv2.setNumThreads(1)
    C = frozen()
    img = load(n, source)
    t0 = time.time()
    res = C.estimate_many(img, ESTS, preproc="lcn", glare="mask")
    wall = time.time() - t0
    rows = []
    for est in ESTS:
        e = res[est]
        d = e["debug"]
        cf = d.get("conf_features") or {}
        rows.append(dict(image=n, source=source, S_in_w=img.shape[1], S_in_h=img.shape[0], estimator=est,
                         h=e["h_count"], v=e["v_count"], conf=e["confidence"], weave=e["weave"],
                         fusion="|".join(d["fusion"]) if d.get("fusion") else None,
                         g_min=cf.get("g_min"), tile_support=cf.get("tile_support"),
                         runtime_s_shared=round(d.get("runtime_s", np.nan), 3), wall_s_all_estimators=round(wall, 3)))
    if source == "vlmcrop":
        rows.append(dict(image=n, source=source, S_in_w=img.shape[1], S_in_h=img.shape[0],
                         estimator="testpy_rgb_2019", h=np.nan, v=float(testpy_weft(img, blur=False))))
        rows.append(dict(image=n, source=source, S_in_w=img.shape[1], S_in_h=img.shape[0],
                         estimator="testpy_rgb_2019_blur", h=np.nan, v=float(testpy_weft(img, blur=True))))
    return rows


def latency(C, names, source="vlmcrop", methods=("ours", "gabor")):
    cv2.setNumThreads(0)  # OpenCV default threading, as a user would run it
    out = []
    for n in names:
        img = load(n, source)
        for m in methods:
            t0 = time.perf_counter()
            r = C.estimate(img, method=m)
            out.append(dict(image=n, estimator=m, latency_s=time.perf_counter() - t0, h=r["h_count"], v=r["v_count"]))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "legacy13_vlmcrops_estimators.csv"))
    ap.add_argument("--check-p1", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = [(n, "vlmcrop") for n in NAMES] + ([(n, "orig_P1") for n in NAMES] if a.check_p1 else [])
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            rows += r
    df = pd.DataFrame(rows).sort_values(["source", "image", "estimator"])
    print(f"estimators: {len(jobs)} jobs in {time.time() - t0:.0f}s")
    C = frozen()
    lat = latency(C, NAMES)
    lat.to_csv(a.out.replace(".csv", "_latency.csv"), index=False)
    # determinism: the sequential single-estimator call must give the same counts as estimate_many
    m = df[(df.source == "vlmcrop")].merge(lat, on=["image", "estimator"], suffixes=("", "_seq"))
    print("estimate() vs estimate_many() max |diff|:", float(np.nanmax(np.abs(np.r_[m.h - m.h_seq, m.v - m.v_seq]))))
    print("latency per image (s), median [min, max]:")
    print(lat.groupby("estimator").latency_s.describe()[["50%", "min", "max", "mean"]].round(2))
    df.to_csv(a.out, index=False)
    print("wrote", a.out)
    if a.check_p1:
        ref = pd.read_csv(os.path.join(HERE, "results", "legacy13_estimators.csv"))
        ref = ref[(ref.crop == "gt") & (ref.preproc == "lcn") & (ref.glare == "mask") & (ref.kind == "clean")]
        p1 = df[df.source == "orig_P1"].merge(ref[["image", "estimator", "h", "v"]], on=["image", "estimator"],
                                              suffixes=("", "_ref"))
        d = np.abs(np.r_[p1.h - p1.h_ref, p1.v - p1.v_ref])
        print(f"P1 reproduction vs legacy13_estimators.csv: {len(p1)} (image, estimator) pairs, "
              f"max |diff| = {np.nanmax(d):.2e} TPI, n exact(<1e-6) = {(d < 1e-6).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
