#!/usr/bin/env python
"""Quantitative realism check: real rectified crops vs synthetic crops.

All crops are resampled to the canonical 768 px/inch, a 3.5% border is
dropped, and simple capture statistics are computed per crop:

    lum            mean grey level (0-255)
    lum_p02/p98    2nd / 98th percentile of the heavily smoothed (sigma 40 px)
                   grey level  -> large-scale illumination range
    global_cv      std / mean of the smoothed grey level (shadows, vignetting)
    local_contrast median |high-pass| / local mean  (yarn-scale contrast)
    sat            mean HSV saturation (0-255)
    lap            log10 variance of the Laplacian (sharpness / noise)
    dark_frac      fraction of the window darker than 35% of the median
                   (deep hand / phone shadows)
    clip_frac      fraction of pixels > 250 (blown-out glare)

The table reports the 10th / 50th / 90th percentile of every statistic for
each source.

    python realism_stats.py --synth data/synth_val,data/synth_val_v2 --out figs/realism_stats.csv
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cnn_common as C  # noqa: E402

TOOL_CROPS = os.path.join(os.path.dirname(HERE), "annotation_tool", "data", "crops")


def stats(crop):
    c = C.resize_to(C.cap_side(crop, 1024), 768)
    b = int(0.035 * 768)
    c = c[b:-b, b:-b]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lo = cv2.GaussianBlur(g, (0, 0), 6)
    big = cv2.GaussianBlur(g, (0, 0), 40)
    return dict(lum=g.mean(), lum_p02=np.percentile(big, 2), lum_p98=np.percentile(big, 98),
                global_cv=big.std() / (big.mean() + 1e-3),
                local_contrast=np.median(np.abs(g - lo) / (lo + 5)),
                sat=hsv[..., 1].mean(), lap=np.log10(cv2.Laplacian(g, cv2.CV_32F).var() + 1),
                dark_frac=(big < 0.35 * np.median(big)).mean(), clip_frac=(g > 250).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", default="data/synth_val,data/synth_val_v2")
    ap.add_argument("--n-synth", type=int, default=300)
    ap.add_argument("--legacy", action="store_true", help="also the 13 legacy crops (shared frame detector)")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "realism_stats.csv"))
    a = ap.parse_args()
    rows = []
    for i in range(150):
        p = os.path.join(TOOL_CROPS, f"{i}.jpg")
        rows.append(dict(src="real150", **stats(cv2.imread(p))))
    if a.legacy:
        import evaluate as E
        for g in E.load_legacy_gt():
            rows.append(dict(src="legacy13", **stats(E.legacy_crop(g, "detect"))))
    for d in a.synth.split(","):
        d = d.strip()
        name = os.path.basename(os.path.normpath(d))
        for f in sorted(glob.glob(os.path.join(d if os.path.isabs(d) else os.path.join(HERE, d), "s*.jpg")))[:a.n_synth]:
            rows.append(dict(src=name, **stats(cv2.imread(f))))
    df = pd.DataFrame(rows)
    q = df.groupby("src").quantile([0.1, 0.5, 0.9])
    q.index.names = ["src", "q"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    q.round(4).to_csv(a.out)
    pd.set_option("display.width", 220)
    print(q.round(3).to_string())
    print("n per source:", df.src.value_counts().to_dict())


if __name__ == "__main__":
    main()
