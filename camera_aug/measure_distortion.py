"""measure_distortion.py -- how much radial distortion does the real phone + pick glass show?

Read-only measurement on the full-resolution originals (fabric_notebooks/final_image_set, via
common/rectify.py).  The window frame edges are physically straight, and a homography maps straight
lines to straight lines, so after rectifying with the annotated corners any remaining CURVATURE of a
frame edge comes from lens/loupe distortion.  Under the corner-rectified radial model of
camera_sim.py (g = 1 + k1 r^2, r normalised by the half side R, corners fixed) the edge midpoint is
displaced outward by ~ k1 R from the chord between the corners (k1 > 0 barrel / convex edges,
k1 < 0 pincushion / concave).

Per side: the rectified image is padded by 12 % of the side; at 41 stations along the side (central
80 %) the dark-frame -> fabric transition is located on the smoothed intensity profile across the edge
(max gradient, sub-pixel), a parabola e(t) = a + b t + c t^2 (t in [-1, 1]) is fitted robustly
(2 rounds of 2.5-MAD rejection) and the sagitta c (px, positive = bulging outward) gives
k1_side = c / R.  Images whose sides fail (dark fabric, frame not visible) are skipped.

    <venv2>/bin/python measure_distortion.py   -> results/distortion_measured.csv (+ summary print)
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as RF  # noqa: E402


def side_edges(img, corners, pad_frac=0.12, n_st=41):
    c = np.asarray(corners, np.float32)
    S = int(round(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])))
    p = int(round(pad_frac * S))
    dst = np.array([[p, p], [p + S, p], [p + S, p + S], [p, p + S]], np.float32)
    H = cv2.getPerspectiveTransform(c, dst)
    W = S + 2 * p
    rect = cv2.warpPerspective(img, H, (W, W), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    g = cv2.GaussianBlur(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2.0)
    out = {}
    half = int(0.10 * S)       # search +-10 % of the side around the nominal edge
    band = max(4, S // 120)    # average the profile over a short stretch along the edge
    ts = np.linspace(-0.8, 0.8, n_st)
    for side in ("top", "bottom", "left", "right"):
        pos = []
        for t in ts:
            s = int(round(p + S / 2 + t * S / 2))
            if side in ("top", "bottom"):
                e0 = p if side == "top" else p + S
                prof = g[e0 - half:e0 + half, s - band:s + band].mean(1)
            else:
                e0 = p if side == "left" else p + S
                prof = g[s - band:s + band, e0 - half:e0 + half].mean(0)
            if side in ("bottom", "right"):
                prof = prof[::-1]                      # walk from outside (frame) to inside (fabric)
            d = np.gradient(cv2.GaussianBlur(prof[None, :], (0, 0), 3.0)[0])
            i = int(np.argmax(d))
            if 1 <= i < len(d) - 1:
                a, b_, cc = d[i - 1], d[i], d[i + 1]
                den = a - 2 * b_ + cc
                i = i + (0.5 * (a - cc) / den if den != 0 else 0)
            contrast = float(np.percentile(prof, 90) - np.percentile(prof, 10))
            # outward displacement of the edge relative to the nominal edge (px, + = outside)
            pos.append((t, half - i, contrast, float(d.max())))
        out[side] = np.array(pos)
    return out, S


def fit_side(arr, S):
    t, e, con, gmax = arr.T
    ok = (con > 25) & (gmax > 1.0)
    if ok.sum() < 20:
        return None
    t, e = t[ok], e[ok]
    keep = np.ones_like(t, bool)
    for _ in range(2):
        cf = np.polyfit(t[keep], e[keep], 2)
        res = e - np.polyval(cf, t)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 0.3
        keep = np.abs(res) < 2.5 * mad
        if keep.sum() < 15:
            return None
    cf = np.polyfit(t[keep], e[keep], 2)
    # sagitta of the edge relative to the chord between its ends at t = +-1: -c (parabola a + b t + c t^2
    # bulges outward at t = 0 by -c relative to the chord when c < 0)
    sag = -cf[0]
    return dict(sagitta_px=float(sag), k1=float(sag / (S / 2)), n=int(keep.sum()),
                rms_px=float(np.sqrt(np.mean((e[keep] - np.polyval(cf, t[keep])) ** 2))))


def main():
    man = json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))
    prov = json.load(open(os.path.join(HERE, "originals_provenance.json")))
    synth = {131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144}
    rows = []
    for e in man["images"]:
        i = e["id"]
        if max(e["orig_w"], e["orig_h"]) < 4000 or i in synth:
            continue
        img = RF.load_image(i)
        corners = RF.load_corners(i)
        ed, S = side_edges(img, corners)
        for side, arr in ed.items():
            f = fit_side(arr, S)
            if f:
                rows.append(dict(id=i, side=side, S=S, **f))
    d = pd.DataFrame(rows)
    grp = {i: gi for gi, g in enumerate(prov["groups"]) for i in g}
    d["scene"] = d.id.map(grp)
    d.to_csv(os.path.join(HERE, "results", "distortion_measured.csv"), index=False)
    per_img = d.groupby("id").agg(k1=("k1", "median"), n_sides=("k1", "size"), scene=("scene", "first"))
    per_scene = per_img[per_img.n_sides >= 3].groupby("scene").k1.median()
    print(f"sides fitted: {len(d)} on {d.id.nunique()} images; images with >=3 sides: "
          f"{(per_img.n_sides >= 3).sum()} ({per_scene.size} scenes)")
    print("k1 per scene: median %.4f, IQR [%.4f, %.4f], min %.4f, max %.4f" % (
        per_scene.median(), per_scene.quantile(.25), per_scene.quantile(.75), per_scene.min(), per_scene.max()))
    print("per side sagitta px: median %.1f, IQR [%.1f, %.1f]; fit rms px median %.2f" % (
        d.sagitta_px.median(), d.sagitta_px.quantile(.25), d.sagitta_px.quantile(.75), d.rms_px.median()))


if __name__ == "__main__":
    main()
