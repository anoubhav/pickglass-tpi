"""cascade_synth.py -- SYNTHETIC calibration set for the quality-aware cascade (ours -> VLM).

The escalation thresholds of the cascade (cascade_eval.py) are chosen on this set only; the 13
hand-counted legacy images are not used to choose anything.

Set "S-cal": fresh v2 synthetic crops (seeds 930000+, never used for training, for the confidence
fit (synth_val, 900000+) or for development (synth_val_v2, 910000+)), generator defaults
(cnn/synth.py sample_params) except that the sensor grid is drawn like the legacy crops' pixel size
(r_sensor ~ U[1630, 1940] px per inch, the range of the 20 canonical legacy crops; image SIZES only,
no labels) and the crop is not capped at 1024 px.  This keeps the pixel-relative degradation
(downscale x8 -> ~225 px/inch) comparable with the legacy study.  Every crop is degraded with
methods/degrade.py at the 19 legacy-study conditions (clean + gblur 1,2,3,4,6 | down 2,4,8 |
jpeg 90,50,20 | glare 5,10,20,30 % | corners 2,5,10 %), same seed rule as degrade_vlmcrops.py
(crc32(name) % 100003), corners via corners_on_crop as for the legacy crops.  Each degraded crop is
scored with the FROZEN counter (counting.py, preproc lcn, glare mask) for ours, log-Gabor and FFT.

    <venv2>/bin/python cascade_synth.py --make --n 200          # crops -> scratchpad
    <venv2>/bin/python cascade_synth.py --score --workers 7     # -> results/cascade_synth_cal.csv
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("SYNTH_PROFILE", "v2")

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
sys.path.insert(0, os.path.join(ROOT, "cnn"))
import counting as C  # noqa: E402
import degrade as D  # noqa: E402

SCRATCH = os.environ.get(
    "CASCADE_SCRATCH",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "scratch")), "cascade"))
SEED0 = 930000
FAMILIES = ["gblur", "down", "jpeg", "glare", "corners"]
CONDS = [("clean", 0)] + [(k, v) for k in FAMILIES for v in D.LEVELS[k]]
ESTS = ["ours", "gabor", "fft"]
GRID = (1630.0, 1940.0)  # legacy canonical crop sides (min, max) in px


def crop_dir(variant):
    return os.path.join(SCRATCH, f"synth_cal_{variant}")


def _gen(a):
    seed, variant = a
    import synth  # noqa: E402  (imported in the worker: SYNTH_PROFILE is read at import)
    rng = np.random.default_rng(seed)
    p = synth.sample_params(rng, seed)
    if variant == "legacygrid":
        r2 = np.random.default_rng(seed + 7_000_000)
        p.r_sensor = float(max(p.r_eff, r2.uniform(*GRID)))
        _, crop, lab = synth.generate(seed, params=p, max_crop_side=2048, frame_margin=0.1)
    else:  # "default": exactly the dataset-mode generator (crop capped at 1024 px)
        _, crop, lab = synth.generate(seed, params=p, max_crop_side=1024, frame_margin=0.1)
    name = f"c{seed:07d}"
    cv2.imwrite(os.path.join(crop_dir(variant), name + ".png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    lab["name"] = name
    lab["S"] = int(crop.shape[1])
    lab.pop("corners", None)
    return lab


def make(n, variant, workers):
    os.makedirs(crop_dir(variant), exist_ok=True)
    t0 = time.time()
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        labs = pool.map(_gen, [(SEED0 + k, variant) for k in range(n)], chunksize=2)
    df = pd.DataFrame(labs).sort_values("seed")
    df.to_csv(os.path.join(crop_dir(variant), "labels.csv"), index=False)
    print(f"wrote {len(df)} crops to {crop_dir(variant)} in {time.time() - t0:.0f}s; "
          f"S median {df.S.median():.0f} [{df.S.min()}, {df.S.max()}]")


def seed_for(name):
    return zlib.crc32(str(name).encode()) % 100003


def _job(a):
    name, kind, level, variant = a
    crop = cv2.imread(os.path.join(crop_dir(variant), name + ".png"), cv2.IMREAD_COLOR)
    img = D.apply(crop, kind, level, seed=seed_for(name))
    t0 = time.time()
    try:
        res = C.estimate_many(img, ESTS, preproc="lcn", glare="mask")
    except Exception:  # e.g. log-Gabor on a tiny (x8-downscaled small) crop: score estimators one by one
        res = {}
        for est in ESTS:
            try:
                res[est] = C.estimate(img, est, preproc="lcn", glare="mask", with_weave=False)
            except Exception as ex:  # noqa: BLE001
                res[est] = {"h_count": np.nan, "v_count": np.nan, "confidence": 0.0, "weave": "unknown",
                            "debug": {"error": repr(ex)[:200]}}
    row = dict(name=name, kind=kind, level=level, S_deg=int(img.shape[1]),
               N_work=int(min(img.shape[1], 1024)))
    for est in ESTS:
        e = res[est]
        d = e["debug"]
        row[f"{est}_h"], row[f"{est}_v"], row[f"{est}_conf"] = e["h_count"], e["v_count"], e["confidence"]
        if est == "ours":
            row["ours_weave"] = e["weave"]
            row["sharpness"] = d.get("sharpness")
            row["glare_frac"] = d.get("glare_frac")
            row["ours_runtime_s"] = round(d.get("runtime_s", np.nan), 3)
            row["fusion"] = "|".join(d["fusion"]) if d.get("fusion") else None
            comp = d.get("components") or {}
            for cname in ("lattice", "profile", "tiles"):
                c = comp.get(cname) or (np.nan, np.nan)
                row[f"{cname}_h"], row[f"{cname}_v"] = c[0], c[1]
            for k, v in (d.get("conf_features") or {}).items():
                row[f"f_{k}"] = v
    row["wall_s"] = round(time.time() - t0, 3)
    return row


VLM_CONDS = [("clean", 0), ("gblur", 3), ("gblur", 6), ("down", 4), ("down", 8), ("jpeg", 20),
             ("glare", 0.05), ("glare", 0.30), ("corners", 0.05)]


def score(variant, workers, out, conds="all19"):
    lab = pd.read_csv(os.path.join(crop_dir(variant), "labels.csv"))
    jobs = [(n, k, v, variant) for n in lab.name for k, v in (CONDS if conds == "all19" else VLM_CONDS)]
    t0, rows = time.time(), []
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    keep = ["name", "seed", "weave", "weave_coarse", "h_count", "v_count", "r_eff", "r_sensor",
            "blur_sigma_px", "glare", "shadow", "S"]
    df = df.merge(lab[keep], on="name", how="left").sort_values(["name", "kind", "level"])
    df.to_csv(out, index=False)
    import platform
    meta = dict(variant=variant, n_images=int(lab.shape[0]), conds=CONDS if conds == "all19" else VLM_CONDS,
                estimators=ESTS, host=platform.node(), numpy=np.__version__, cv2=cv2.__version__,
                seeds=f"{SEED0}..{SEED0 + lab.shape[0] - 1}", synth_profile=os.environ.get("SYNTH_PROFILE"),
                counting_py_md5=hashlib.md5(open(C.__file__, "rb").read()).hexdigest(),
                conf_model_md5=hashlib.md5(open(os.path.join(HERE, "conf_model.json"), "rb").read()).hexdigest(),
                degrade_py_md5=hashlib.md5(open(D.__file__, "rb").read()).hexdigest(),
                wall_s=round(time.time() - t0, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(meta, open(out.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {out}: {len(df)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--variant", default="legacygrid", choices=["legacygrid", "default"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conds", default="all19", choices=["all19", "vlm9"])
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "results", f"cascade_synth_cal_{a.variant}.csv")
    if a.make:
        make(a.n, a.variant, a.workers)
    if a.score:
        score(a.variant, a.workers, out, a.conds)
