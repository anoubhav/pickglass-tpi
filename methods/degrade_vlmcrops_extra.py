"""degrade_vlmcrops_extra.py -- the three Table-6 degradation families that degrade_vlmcrops.py
did not run on the canonical legacy crops: motion blur, noise and hard shadow.

Same code path as degrade_vlmcrops.py (imported unchanged: same canonical crops, degrade.apply,
per-image seed rule, estimate_many(lcn, glare mask) for ours / log-Gabor / FFT); only the
families differ, and the outputs go to separate files so degrade_vlmcrops.py's CSV is untouched.

    <venv2>/bin/python methods/degrade_vlmcrops_extra.py      # make + score, ~2-5 min CPU
Out: methods/results/legacy13_vlmcrops_degrade_extra.csv (+ _summary.csv: ours/gabor/fft
     counts within 10 % of the hand count, out of 26, axis-matched, per condition)
"""
import hashlib
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import degrade as D  # noqa: E402
import degrade_vlmcrops as J  # noqa: E402

J.DEG_DIR = os.environ.get("DEG_LEGACY13_EXTRA_DIR", J.DEG_DIR + "_extra")
FAMILIES = ["motion", "noise", "shadow"]
CONDS = [(k, v) for k in FAMILIES for v in D.LEVELS[k]]
OUT = os.path.join(HERE, "results", "legacy13_vlmcrops_degrade_extra.csv")


def make():
    man = []
    for name in J.NAMES13:
        crop = cv2.imread(os.path.join(J.CROP_DIR, f"{name}.png"), cv2.IMREAD_COLOR)
        for kind, level in CONDS:
            img = D.apply(crop, kind, level, seed=J.seed_for(name))
            p = J.deg_path(name, kind, level)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            man.append(dict(image=name, kind=kind, level=level, tag=J.cond_tag(kind, level),
                            md5=hashlib.md5(open(p, "rb").read()).hexdigest()))
    pd.DataFrame(man).to_csv(os.path.join(J.DEG_DIR, "manifest.csv"), index=False)


def main():
    t0 = time.time()
    make()
    import ablate
    gt = ablate.legacy_gt()
    jobs = [(n, k, v) for n in J.NAMES13 for k, v in CONDS]
    rows = []
    with Pool(6) as pool:
        for r in pool.imap_unordered(J._job, jobs):
            rows += r
    df = pd.DataFrame(rows)
    df["gt_warp"] = df.image.map(lambda k: gt[k]["warp"])
    df["gt_weft"] = df.image.map(lambda k: gt[k]["weft"])
    df = df.merge(pd.read_csv(os.path.join(J.DEG_DIR, "manifest.csv"))[["image", "tag", "md5"]],
                  on=["image", "tag"], how="left")
    df.to_csv(OUT, index=False)
    S = []
    for (est, kind, level), g in df.groupby(["estimator", "kind", "level"]):
        ape = np.r_[100 * abs(g.h - g.gt_warp) / g.gt_warp, 100 * abs(g.v - g.gt_weft) / g.gt_weft]
        S.append(dict(estimator=est, kind=kind, level=level, n_images=len(g), n_w10=int((ape <= 10).sum()),
                      n_w5=int((ape <= 5).sum()), MAPE=round(float(np.nanmean(ape)), 2),
                      conf_mean=round(float(g.conf.mean()), 3)))
    S = pd.DataFrame(S)
    S.to_csv(OUT.replace(".csv", "_summary.csv"), index=False)
    print(S.to_string(index=False))
    print(f"wrote {OUT} ({len(df)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
