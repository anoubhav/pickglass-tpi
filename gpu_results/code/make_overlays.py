"""Qualitative P4 pointing overlays for the paper (Molmo2-8B bf16).

For a legacy image, the points Molmo2 returned for the two P4 calls are drawn on the 644-px
model input (the crop with the red guide line):
    cyan   = "every vertical yarn where it crosses the red horizontal line" (guide_h, P4h)
    yellow = "every horizontal yarn where it crosses the red vertical line" (guide_v, P4v)
Molmo2 coordinates are on a 0-1000 scale of the image width/height.
Each figure: full window (left) and a 2x zoom on the central half (right) where the yarns are
visible, with the counts, the GT and the spacing regularity (CV of the gaps between
consecutive points along the line) in the title.

    python make_overlays.py                   # auto-select 3 GT images (best / median / worst P4)
    python make_overlays.py --ids r7,a5,r2
    python make_overlays.py --method P4q     # same for the central quarter-inch patch (x 4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as C          # noqa: E402

ROOT = C.ROOT
RAW = os.path.join(ROOT, "raw", "molmo2_8b_bf16__legacy.jsonl")
FIGS = os.path.join(ROOT, "figs")


def load():
    recs = {}
    for line in open(RAW):
        r = json.loads(line)
        if r.get("error"):
            continue
        recs[(r["image_id"], r["view"])] = r
    return recs


def cv_gaps(pts, axis):
    xs = np.sort(np.array([p[axis] for p in pts], float))
    g = np.diff(xs)
    return float(g.std() / g.mean()) if len(g) > 1 and g.mean() > 0 else float("nan")


VIEWS = {  # method -> (H-line view, V-line view, background view without the line, count multiplier, caption)
    "P4": ("guide_h", "guide_v", "crop", 1, "full 1-inch window"),
    "P4q": ("qc_guide_h", "qc_guide_v", "qc", 4, "central 1/4-inch patch (count x 4)"),
}


def draw(name, recs, gt, out_png, method="P4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vh, vv, vbg, mult, cap = VIEWS[method]
    rh, rv = recs[(name, vh)], recs[(name, vv)]
    img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, recs[(name, vbg)]["image_path"])), cv2.COLOR_BGR2RGB)
    S = img.shape[0]
    ph = [(x / 1000 * S, y / 1000 * S) for x, y in rh["parsed"]["points"]]
    pv = [(x / 1000 * S, y / 1000 * S) for x, y in rv["parsed"]["points"]]
    nh, nv = rh["parsed"]["n_points"], rv["parsed"]["n_points"]
    g = gt.get(name)
    if g:
        pred = sorted([mult * nh, mult * nv]); tru = sorted([g["warp"], g["weft"]])
        err = np.mean(np.abs(np.array(pred) - np.array(tru)))
        gts = f"GT warp {g['warp']:g} x weft {g['weft']:g} ({g['weave_name']}); best-assignment MAE {err:.1f}"
    else:
        gts = "no GT"
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150)
    for k, ax in enumerate(axs):
        ax.imshow(img, interpolation="nearest" if k else "lanczos")
        ax.axhline(S / 2, color="red", lw=0.8, alpha=0.6)
        ax.axvline(S / 2, color="red", lw=0.8, alpha=0.6)
        ms = 10 if k == 0 else 26
        if ph:
            ax.scatter(*zip(*ph), s=ms, c="cyan", edgecolors="black", linewidths=0.4, zorder=3,
                       label=f"vertical yarns on H line: {nh}")
        if pv:
            ax.scatter(*zip(*pv), s=ms, c="yellow", edgecolors="black", linewidths=0.4, zorder=3,
                       label=f"horizontal yarns on V line: {nv}")
        ax.set_xticks([]); ax.set_yticks([])
        if k == 1:
            ax.set_xlim(S / 4, 3 * S / 4); ax.set_ylim(3 * S / 4, S / 4)
            ax.set_title("central half (2x zoom)", fontsize=9)
        else:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
            ax.set_title(f"{name}: {cap}, {S}x{S} px model input", fontsize=9)
    trunc = [t for t, r in (("H", rh), ("V", rv)) if r["parsed"].get("truncated")]
    est = f" -> {mult * nh} x {mult * nv} TPI" if mult != 1 else ""
    fig.suptitle(f"Molmo2-8B bf16, {method} pointing: {nh} x {nv} points{est} | {gts}\n"
                 f"gap CV H {cv_gaps(rh['parsed']['points'], 0):.2f}, V {cv_gaps(rv['parsed']['points'], 1):.2f}"
                 + (f" | truncated at token cap: {','.join(trunc)}" if trunc else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_png[:-4] + ".pdf")
    plt.close(fig)
    with open(out_png[:-4] + ".txt", "w") as f:
        f.write(f"{name}\nP4h raw ({rh['generation_tokens']} tokens, {rh['finish_reason']}):\n{rh['raw']}\n\n"
                f"P4v raw ({rv['generation_tokens']} tokens, {rv['finish_reason']}):\n{rv['raw']}\n")
    return nh, nv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None)
    ap.add_argument("--method", default="P4", choices=list(VIEWS))
    a = ap.parse_args()
    vh, vv, _, mult, _ = VIEWS[a.method]
    os.makedirs(FIGS, exist_ok=True)
    recs = load()
    gt = C.legacy_gt()
    if a.ids:
        ids = a.ids.split(",")
    else:   # best, median and worst P4 best-assignment error among the GT images
        errs = []
        for n, g in gt.items():
            if (n, vh) in recs and (n, vv) in recs:
                p = sorted([mult * recs[(n, vh)]["parsed"]["n_points"], mult * recs[(n, vv)]["parsed"]["n_points"]])
                errs.append((np.mean(np.abs(np.array(p) - np.array(sorted([g["warp"], g["weft"]])))), n))
        errs.sort()
        ids = [errs[0][1], errs[len(errs) // 2][1], errs[-1][1]]
    for k, n in enumerate(ids, 1):
        tag = "" if a.method == "P4" else a.method + "_"
        out = os.path.join(FIGS, f"molmo2_bf16_pointing_{tag}{k}_{n}.png")
        print(n, draw(n, recs, gt, out, a.method), "->", out)
