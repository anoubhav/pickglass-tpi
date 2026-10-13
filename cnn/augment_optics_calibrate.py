#!/usr/bin/env python
"""augment_optics_calibrate.py -- how strong are the optics effects in the REAL crops?

For every real crop (annotation_tool/data/crops, 150) and legacy-13 crop:
  * lateral chromatic aberration: affine ECC registration of the red and blue channels onto green
    (high-passed, at <= 1024 px).  The relative magnification sqrt(det A) - 1 is the CA strength
    eps of augment_optics.chromatic_aberration (the affine translation absorbs the off-centre optical
    axis).  Registered at the crop resolution; the per-channel scale is invariant to the crop's
    rectifying homography to first order.
  * vignetting / illumination fall-off: ratio of the mean of the 4 corner patches to the central
    patch of the heavily blurred luminance (includes LED fall-off and shadows: an upper bound).
  * clipped (specular) area: fraction of pixels with min channel >= 245 (cf. led_hotspot clipped_frac).
Output: aug_optics/realism_calibration.csv (+ printed quantiles).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
cv2.setNumThreads(1)


def hp(x):
    x = x.astype(np.float32)
    return x - cv2.GaussianBlur(x, (0, 0), 4.0)


def ca_scale(img):
    S = img.shape[0]
    if S > 1024:
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    g = hp(img[..., 1])
    out = {}
    for name, ch in (("R", 2), ("B", 0)):
        c = hp(img[..., ch])
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(g, c, W, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7), None, 5)
            # W maps template (G) coords -> channel coords; a channel magnified by (1 + eps) about the axis
            # gives sqrt(det W) = 1 + eps (validated on injected CA: eps 0.003 -> 0.0027-0.0028, 0.01 -> 0.0096-0.0099)
            out[name] = float(np.sqrt(abs(np.linalg.det(W[:, :2]))) - 1.0)
        except cv2.error:
            out[name] = np.nan
    return out, img.shape[0]


def falloff(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    S = L.shape[0]
    B = cv2.GaussianBlur(L, (0, 0), S / 30)
    p = max(4, S // 10)
    cen = B[S // 2 - p:S // 2 + p, S // 2 - p:S // 2 + p].mean()
    cor = np.mean([B[:p, :p].mean(), B[:p, -p:].mean(), B[-p:, :p].mean(), B[-p:, -p:].mean()])
    return float(cor / max(1e-3, cen))


def job(a):
    sname, name, path = a
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    ca, S = ca_scale(im)
    return dict(set=sname, image=name, side_native=im.shape[0], side_used=S, ca_R=ca["R"], ca_B=ca["B"],
                ca_absmax=np.nanmax([abs(ca["R"]), abs(ca["B"])]),
                fringe_px_native_corner=np.nanmax([abs(ca["R"]), abs(ca["B"])]) * im.shape[0] / np.sqrt(2),
                corner_centre_ratio=falloff(im),
                clipped_frac=float((im.min(axis=2) >= 245).mean()))


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [("real150", str(i), os.path.join(ROOT, "annotation_tool", "data", "crops", f"{i}.jpg")) for i in range(150)]
    jobs += [("legacy13", n, os.path.join(ROOT, "vlm", "crops", "legacy", f"{n}.png"))
             for n in ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]]
    with Pool(int(os.environ.get("WORKERS", "4"))) as p:
        rows = p.map(job, jobs, chunksize=4)
    d = pd.DataFrame(rows)
    import json
    M = {str(m["id"]): m["rect_size"][0] for m in json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))["images"]}
    d["subset"] = np.where(d.set == "legacy13", "legacy13",
                           np.where(d.image.map(M).fillna(0) > 1000, "real150_highres", "real150_lowres"))
    os.makedirs(os.path.join(HERE, "aug_optics"), exist_ok=True)
    d.to_csv(os.path.join(HERE, "aug_optics", "realism_calibration.csv"), index=False, float_format="%.5g")
    q = [50, 75, 90, 95, 99]
    for sub, g in d.groupby("subset"):
        print(sub, len(g))
        for c in ("ca_R", "ca_B", "ca_absmax", "fringe_px_native_corner", "corner_centre_ratio", "clipped_frac"):
            print(f"  {c:26s}", np.round(np.nanpercentile(g[c], q), 5))
    print("photo 77:", d[(d.set == "real150") & (d.image == "77")].to_dict("records"))
