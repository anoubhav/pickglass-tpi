"""degrade_vlmcrops.py -- capture-quality study for VLMs: degraded versions of the
13 hand-counted legacy crops, built ONCE from the canonical crops that the VLMs saw
(vlm/crops/legacy/{name}.png), then scored with the frozen counter ("ours"), log-Gabor and FFT.

Degradations come from degrade.py unchanged (degrade.apply, same severities, same per-image seed
rule as ablate.py: seed = crc32(name) % 100003).  Legacy crops have no original photo with
annotated corners, so 'corners' uses degrade.corners_on_crop (the crop's own corners are
perturbed and the crop is re-warped), exactly as ablate.py does for legacy13.

Every degraded crop is written as a lossless PNG (+ md5 manifest) so that the CNN
(cnn/degrade_vlmcrops.py) and the VLM driver (vlm/run_degrade_vlm.py) read byte-identical
pixels; each method then applies only its own usual input handling (ours: INTER_AREA to
<=1024 px working image; CNN: cap 1024 + 3-scale TTA; VLMs: 1536-px square JPEG q92).

Families / levels (all levels of the five requested families; the VLMs are run on the subset
VLM_CONDS, chosen a priori to match the levels reported in paper Table 6):
    gblur 1,2,3,4,6 u | down x2,x4,x8 | jpeg 90,50,20 | glare 5,10,20,30 % | corners 2,5,10 %

    <venv2>/bin/python degrade_vlmcrops.py --make      # write degraded PNGs + manifest
    <venv2>/bin/python degrade_vlmcrops.py --score     # ours / gabor / fft -> results CSV
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import hashlib
import json
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
import counting as C  # noqa: E402
import degrade as D  # noqa: E402

CROP_DIR = os.path.join(ROOT, "vlm", "crops", "legacy")
DEG_DIR = os.environ.get(
    "DEG_LEGACY13_DIR",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "scratch")), "deg_legacy13"))
OUT_CSV = os.path.join(HERE, "results", "legacy13_vlmcrops_degrade.csv")
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
FAMILIES = ["gblur", "down", "jpeg", "glare", "corners"]
CONDS = [("clean", 0)] + [(k, v) for k in FAMILIES for v in D.LEVELS[k]]
VLM_CONDS = [("gblur", 3), ("gblur", 6), ("down", 4), ("down", 8), ("jpeg", 20),
             ("glare", 0.05), ("glare", 0.30), ("corners", 0.05)]
ESTS = ["ours", "gabor", "fft"]


def cond_tag(kind, level):
    return "clean" if kind == "clean" else f"{kind}{level:g}"


def seed_for(name):
    return 0 * 100003 + zlib.crc32(str(name).encode()) % 100003  # == ablate.py with --seed 0


def deg_path(name, kind, level):
    return os.path.join(DEG_DIR, cond_tag(kind, level), f"{name}.png")


def make():
    man = []
    for name in NAMES13:
        crop = cv2.imread(os.path.join(CROP_DIR, f"{name}.png"), cv2.IMREAD_COLOR)
        for kind, level in CONDS:
            img = D.apply(crop, kind, level, seed=seed_for(name))
            p = deg_path(name, kind, level)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            man.append(dict(image=name, kind=kind, level=level, tag=cond_tag(kind, level), path=p,
                            w=img.shape[1], h=img.shape[0], seed=seed_for(name),
                            md5=hashlib.md5(open(p, "rb").read()).hexdigest()))
        print(f"  {name}: {len(CONDS)} conditions", flush=True)
    pd.DataFrame(man).to_csv(os.path.join(DEG_DIR, "manifest.csv"), index=False)
    meta = dict(source_crops=CROP_DIR, degrade_py_md5=hashlib.md5(open(D.__file__, "rb").read()).hexdigest(),
                cv2=cv2.__version__, numpy=np.__version__, conds=CONDS, vlm_conds=VLM_CONDS,
                seed_rule="crc32(name) % 100003 (ablate.py, --seed 0)", created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(meta, open(os.path.join(DEG_DIR, "meta.json"), "w"), indent=1)
    print(f"wrote {len(man)} degraded crops to {DEG_DIR}")


def _job(a):
    name, kind, level = a
    img = cv2.imread(deg_path(name, kind, level), cv2.IMREAD_COLOR)
    t0 = time.time()
    res = C.estimate_many(img, ESTS, preproc="lcn", glare="mask")
    rows = []
    for est in ESTS:
        e = res[est]
        d = e["debug"]
        cf = d.get("conf_features") or {}
        comp = d.get("components") or {}
        rows.append(dict(image=name, kind=kind, level=level, tag=cond_tag(kind, level), estimator=est,
                         S_deg=img.shape[1], h=e["h_count"], v=e["v_count"], conf=e["confidence"],
                         weave=e["weave"], glare_frac=d.get("glare_frac"), sharpness=d.get("sharpness"),
                         g_min=cf.get("g_min"), margin=cf.get("margin"), tile_support=cf.get("tile_support"),
                         agree_prof=cf.get("agree_prof"),
                         fusion="|".join(d["fusion"]) if d.get("fusion") else None,
                         lat_h=(comp.get("lattice") or (None, None))[0],
                         lat_v=(comp.get("lattice") or (None, None))[1],
                         runtime_s=round(d.get("runtime_s", np.nan), 3), wall_s=round(time.time() - t0, 3)))
    return rows


def score(workers):

    import ablate  # GT via the legacy harness parser (same as every other legacy row)
    gt = ablate.legacy_gt()
    jobs = [(n, k, v) for n in NAMES13 for k, v in CONDS]
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows += r
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["gt_warp"] = df.image.map(lambda k: gt[k]["warp"])
    df["gt_weft"] = df.image.map(lambda k: gt[k]["weft"])
    df["gt_weave"] = df.image.map(lambda k: gt[k]["weave"])
    man = pd.read_csv(os.path.join(DEG_DIR, "manifest.csv"))[["image", "tag", "md5"]]
    df = df.merge(man, on=["image", "tag"], how="left")
    df.to_csv(OUT_CSV, index=False)
    meta = dict(counting_py_md5=hashlib.md5(open(C.__file__, "rb").read()).hexdigest(),
                conf_model_md5=hashlib.md5(open(os.path.join(HERE, "conf_model.json"), "rb").read()).hexdigest(),
                note="counting.py differs from results/counting_frozen_final.py only in guide_positions() "
                     "(display helper); estimators and conf model are the frozen ones.",
                preproc="lcn", glare="mask", max_work=1024, estimators=ESTS, wall_s=round(time.time() - t0, 1))
    json.dump(meta, open(OUT_CSV.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {OUT_CSV}: {len(df)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.make:
        make()
    if a.score:
        score(a.workers)
