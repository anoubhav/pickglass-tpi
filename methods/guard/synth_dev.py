"""synth_dev.py -- SYNTHETIC development set for the native-resolution guard (native_res_guard.py).

Every guard parameter is chosen on this set only.  No real crop (legacy-13, R-HR, the 150) is
read by this script or by tune_synth.py.

Crops: fresh cnn/synth.py v2 crops (SYNTH_PROFILE=v2), seeds never used elsewhere
(training 0..102399, synth_val 900000+, synth_val_v2 910000+, cascade S-cal 930000+):

  set "hr"  seeds 960000+   sensor grid r_sensor ~ U[1500, 1950] px/inch (the R-HR / legacy crop
                            sides), crop not capped (max_crop_side 2048) -- the regime of the
                            genai_sr references and of the cascade's degraded legacy crops.
  set "lr"  seeds 965000+   r_sensor ~ U[450, 620] px/inch (the 85 low-resolution captures of the
                            150), r_eff <= r_sensor -- native low-resolution crops, on which the
                            guard must NOT change the count.

Conditions (degraded crop, plus the true native side n_true when it is known):
  clean                         n_true = S
  none_x{2,4,8}                 methods/degrade.downscale (INTER_AREA); the LR crop itself
  {bicubic,bilinear,lanczos,sharp}_x{2,4,8}
                                LR crop brought back to S x S with cv2 INTER_CUBIC / INTER_LINEAR /
                                INTER_LANCZOS4, or 'sharp' = INTER_CUBIC + unsharp mask at the LR
                                scale (sigma 0.6 k HR px, amount 1.0): an SR-like edge enhancer
                                that boosts the band just below and above the LR Nyquist
  bicubicJ_x4                   jpeg(downscale(x4), 50) then INTER_CUBIC (the genai_sr D4J input
                                to the bicubic restorer)
  bicubic_x4_jpeg85             bicubic x4 up-sampling then JPEG q85 (an up-scaled upload)
  gblur{1,2,3,4,6}              methods/degrade.gaussian_blur (sigma in milli-inch), as in the
                                cascade / legacy degradation study
  jpeg50 (set lr only)          methods/degrade.jpeg q50 of the native low-res crop

    <venv2>/bin/python synth_dev.py --make --n-hr 120 --n-lr 60 --workers 10
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("SYNTH_PROFILE", "v2")

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.dirname(HERE)
ROOT = os.path.dirname(METHODS)
sys.path.insert(0, METHODS)
sys.path.insert(0, os.path.join(ROOT, "cnn"))
import degrade as D  # noqa: E402

SCRATCH = os.environ.get(
    "GUARD_SCRATCH",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "scratch")), "guard_synth"))
SEED_HR, SEED_LR = 960000, 965000
GRID_HR = (1500.0, 1950.0)
GRID_LR = (450.0, 620.0)
FACTORS = (2, 4, 8)
UPS = ("bicubic", "bilinear", "lanczos", "sharp")
INTERP = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4,
          "sharp": cv2.INTER_CUBIC}
GBLUR = (1, 2, 3, 4, 6)


def conditions(set_name):
    if set_name == "lr":
        return ["clean", "jpeg50"]
    c = ["clean"] + [f"none_x{k}" for k in FACTORS]
    c += [f"{u}_x{k}" for u in UPS for k in FACTORS]
    c += ["bicubicJ_x4", "bicubic_x4_jpeg85"] + [f"gblur{g}" for g in GBLUR]
    return c


def upsample(lr, S, how, k):
    up = cv2.resize(lr, (S, S), interpolation=INTERP[how])
    if how == "sharp":
        f = up.astype(np.float32)
        bl = cv2.GaussianBlur(f, (0, 0), 0.6 * k)
        up = np.clip(f + 1.0 * (f - bl), 0, 255).astype(np.uint8)
    return up


def apply_condition(crop, cond):
    """-> (degraded crop, n_true): n_true = side of the native sampling grid of the content
    (the LR side for up-sampled / LR inputs; S for clean and blur)."""
    S = crop.shape[1]
    if cond == "clean":
        return crop, S
    if cond.startswith("gblur"):
        return D.gaussian_blur(crop, float(cond[5:])), S
    if cond == "jpeg50":
        return D.jpeg(crop, 50), S
    how, rest = cond.split("_x", 1)
    k = int(rest.split("_")[0])
    lr = D.downscale(crop, k)
    n = lr.shape[1]
    if how == "none":
        return lr, n
    if how == "bicubicJ":
        return upsample(D.jpeg(lr, 50), S, "bicubic", k), n
    up = upsample(lr, S, how, k)
    if cond.endswith("_jpeg85"):
        up = D.jpeg(up, 85)
    return up, n


def crop_dir():
    return os.path.join(SCRATCH, "crops")


def _gen(a):
    seed, set_name = a
    import synth  # noqa: E402  (SYNTH_PROFILE read at import)
    rng = np.random.default_rng(seed)
    p = synth.sample_params(rng, seed)
    r2 = np.random.default_rng(seed + 7_000_000)
    grid = GRID_HR if set_name == "hr" else GRID_LR
    p.r_sensor = float(r2.uniform(*grid))
    if set_name == "hr":
        p.r_sensor = max(p.r_eff, p.r_sensor)
    else:
        p.r_eff = min(p.r_eff, p.r_sensor)
    _, crop, lab = synth.generate(seed, params=p, max_crop_side=2048, frame_margin=0.1)
    name = f"g{seed:07d}"
    cv2.imwrite(os.path.join(crop_dir(), name + ".png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    lab.pop("corners", None)
    lab.update(name=name, set=set_name, S=int(crop.shape[1]))
    return lab


def make(n_hr, n_lr, workers):
    os.makedirs(crop_dir(), exist_ok=True)
    t0 = time.time()
    jobs = [(SEED_HR + i, "hr") for i in range(n_hr)] + [(SEED_LR + i, "lr") for i in range(n_lr)]
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        labs = pool.map(_gen, jobs, chunksize=1)
    df = pd.DataFrame(labs).sort_values("seed")
    df.to_csv(os.path.join(SCRATCH, "labels.csv"), index=False)
    json.dump(dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), synth_profile=os.environ.get("SYNTH_PROFILE"),
                   n_hr=n_hr, n_lr=n_lr, seeds_hr=f"{SEED_HR}..{SEED_HR + n_hr - 1}",
                   seeds_lr=f"{SEED_LR}..{SEED_LR + n_lr - 1}", grid_hr=GRID_HR, grid_lr=GRID_LR,
                   conditions_hr=conditions("hr"), conditions_lr=conditions("lr")),
              open(os.path.join(SCRATCH, "meta.json"), "w"), indent=1)
    print(f"wrote {len(df)} crops in {time.time() - t0:.0f}s; S by set:\n"
          f"{df.groupby('set').S.describe()[['min', '50%', 'max']]}")


def labels():
    return pd.read_csv(os.path.join(SCRATCH, "labels.csv"))


def load_crop(name):
    return cv2.imread(os.path.join(crop_dir(), name + ".png"), cv2.IMREAD_COLOR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--n-hr", type=int, default=120)
    ap.add_argument("--n-lr", type=int, default=60)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.make:
        make(a.n_hr, a.n_lr, a.workers)
