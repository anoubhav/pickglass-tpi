"""ablate.py -- capture-quality / pipeline ablations of the thread counters -> tidy CSV.

One row per (image, crop source, preprocessing, glare handling, estimator, degradation, level).

Sets
  legacy13  the 13 legacy (2019) images with usable hand counts (+ the 7 without counts, flagged
            usable=False).  They are already hand-cropped to ~ the 1-inch window, so the whole
            image is treated as 1 inch (protocol P1).  GT from the 2019 harness's load_gt().
  all150    the 150 pick-glass photos; crops via common/rectify.py (GT window corners).  No
            thread-count labels yet -> self-consistency (drift vs. the clean prediction).

Crop source (--crop, comma list)
  gt         legacy: full image; all150: common/rectify.rectify(i) (annotated window corners)
  seg_final  the 2024 classical segmentation (pipeline/seg_final.py, imported read-only,
             cv2 RNG seeded 0); its box is ASSUMED to be the 1-inch window
  centre     centred 0.5W x 0.5H box of the photo (seg_final's fallback), ASSUMED 1 inch

Estimators (--est): ours, lattice, fft, fft_harm, profile, peaks, gabor, tiles, fusion3
Degradations (--degrade): clean and/or kinds from degrade.LEVELS (all levels), e.g.
  --degrade clean,gblur,down,jpeg  or  --degrade all

Examples
  python ablate.py --set legacy13 --est ours,lattice,fft,profile,peaks,gabor,tiles,fusion3 --out results/legacy13_estimators.csv
  python ablate.py --set legacy13 --est ours --preproc lcn,clahe,hp,none --glare mask,inpaint,off --out results/legacy13_preproc.csv
  python ablate.py --set all150 --est ours --degrade all --out results/all150_degrade.csv
  python ablate.py --set all150 --est ours --crop gt,seg_final,centre --out results/all150_crops.csv
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")  # one BLAS thread per worker (the machine is shared)

import argparse
import itertools
import os
import sys
import time
import zlib
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import counting as C  # noqa: E402
import degrade as D  # noqa: E402
import rectify as R  # noqa: E402

PIPELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_2019")
LEGACY_2019 = os.path.join(HERE, "legacy_2019")  # verbatim copy of the 2019 harness (fallback)
LEGACY_DIR = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "FR-final attempt", "images")


def _pipeline_module(name):
    """Import run_pipeline / seg_final read-only from the original harness (or its copy)."""
    for d in (PIPELINE, LEGACY_2019):
        if os.path.exists(os.path.join(d, name + ".py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return __import__(name)
    raise ImportError(name)


def legacy_gt():
    """{key: gt dict} via run_pipeline.load_gt (the 2019 harness parser, reused as-is)."""
    rp = _pipeline_module("run_pipeline")
    gt, _ = rp.load_gt(rp.DEF_GT, rp.DEF_GT2)
    return gt


def legacy_files():
    rp = _pipeline_module("run_pipeline")
    return {k: os.path.join(LEGACY_DIR, rp._fname_for(k, LEGACY_DIR)) for k in rp.ORDER}


# ----------------------------------------------------------------------------- crops
def get_crop(set_name, key, source):
    """-> (crop_bgr, meta).  meta has photo/corners for corner perturbation when available."""
    if set_name == "all150":
        path = R.image_path(key)
        if source == "gt":
            photo = R.load_image(key)
            corners = R.load_corners(key)
            return R.rectify(key, img=photo), {"photo": photo, "corners": corners, "path": path}
    else:
        path = legacy_files()[key]
        if source == "gt":
            img = cv2.imread(path)
            return img, {"path": path}
    if source == "seg_final":
        sf = _pipeline_module("seg_final")
        crop, box, fb = sf.segment(path, cv_seed=0)
        return crop, {"path": path, "seg_fallback": bool(fb)}
    if source == "centre":
        img = cv2.imread(path)
        H, W = img.shape[:2]
        return img[H // 4:H // 4 + H // 2, W // 4:W // 4 + W // 2].copy(), {"path": path}
    raise ValueError(source)


# ----------------------------------------------------------------------------- worker
def _job(a):
    set_name, key, crops, preprocs, glares, ests, conds, extra = a
    rows = []
    for src in crops:
        try:
            crop, meta = get_crop(set_name, key, src)
        except Exception as e:  # noqa: BLE001
            rows.append({"set": set_name, "image": key, "crop": src, "error": f"crop: {e}"})
            continue
        for (kind, level) in conds:
            if kind == "corners" and src != "gt":
                continue
            seed = extra["seed_base"] * 100003 + zlib.crc32(str(key).encode()) % 100003  # per image
            img = D.apply(crop, kind, level, seed=seed, photo=meta.get("photo"), corners=meta.get("corners"))
            for pp, gl in itertools.product(preprocs, glares):
                t0 = time.time()
                try:
                    res = C.estimate_many(img, ests, preproc=pp, glare=gl)
                except Exception as ex:  # noqa: BLE001
                    rows.append({"set": set_name, "image": key, "crop": src, "preproc": pp, "glare": gl,
                                 "kind": kind, "level": level, "error": repr(ex)})
                    continue
                for est in ests:
                    e = res[est]
                    d = e["debug"]
                    cf = d.get("conf_features") or {}
                    comp = d.get("components") or {}
                    rows.append({"set": set_name, "image": key, "crop": src, "preproc": pp, "glare": gl,
                                 "estimator": est, "kind": kind, "level": level,
                                 "S_in_w": crop.shape[1], "S_in_h": crop.shape[0],
                                 "S_deg": img.shape[1], "h": e["h_count"], "v": e["v_count"],
                                 "conf": e["confidence"], "weave": e["weave"],
                                 "glare_frac": d.get("glare_frac"), "sharpness": d.get("sharpness"),
                                 "tile_support": cf.get("tile_support"), "g_min": cf.get("g_min"),
                                 "margin": cf.get("margin"), "agree_prof": cf.get("agree_prof"),
                                 "fusion": "|".join(d["fusion"]) if d.get("fusion") else None,
                                 "lat_h": (comp.get("lattice") or (None, None))[0],
                                 "lat_v": (comp.get("lattice") or (None, None))[1],
                                 "seg_fallback": meta.get("seg_fallback"),
                                 "runtime_s": round(d.get("runtime_s", np.nan), 3),
                                 "wall_s_all_estimators": round(time.time() - t0, 3)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["legacy13", "all150"], required=True)
    ap.add_argument("--est", default="ours")
    ap.add_argument("--crop", default="gt")
    ap.add_argument("--preproc", default="lcn")
    ap.add_argument("--glare", default="mask")
    ap.add_argument("--degrade", default="clean", help="clean,<kind>,... or 'all'")
    ap.add_argument("--ids", default="", help="comma list of image keys (default: all)")
    ap.add_argument("--all-legacy", action="store_true", help="also run the 7 legacy images without GT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    kinds = list(D.LEVELS) if a.degrade == "all" else [k for k in a.degrade.split(",") if k != "clean"]
    conds = [("clean", 0)] + [(k, v) for k in kinds for v in D.LEVELS[k]]
    if a.set == "legacy13":
        gt = legacy_gt()
        keys = [k for k in gt if gt[k]["usable"] or a.all_legacy]
    else:
        keys = list(range(R.N_IMAGES))
    if a.ids:
        keys = [type(keys[0])(k) for k in a.ids.split(",")]
    jobs = [(a.set, k, a.crop.split(","), a.preproc.split(","), a.glare.split(","), a.est.split(","), conds,
             {"seed_base": a.seed}) for k in keys]
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows += r
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {i + 1}/{len(jobs)} images  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if a.set == "legacy13":
        g = pd.DataFrame([{"image": k, "gt_warp": v["warp"], "gt_weft": v["weft"], "gt_weave": v["weave"],
                           "gt_unclear": v["unclear"], "gt_usable": v["usable"],
                           "gt_warp_raw": v["warp_raw"], "gt_weft_raw": v["weft_raw"]} for k, v in gt.items()])
        df = df.merge(g, on="image", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, {len(jobs)} images, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
