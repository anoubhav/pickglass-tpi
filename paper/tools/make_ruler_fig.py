"""make_ruler_fig.py -- Fig. (ruler self-calibration): detected ticks on two photos.

    <venv2>/bin/python tools/make_ruler_fig.py [--ids 8 55] [--out figs/fig_ruler.pdf]

Each panel: the photo around the glass frame, foot points of the detected ticks (circles: fine
1-division scale, squares: coarse 10-division scale), the window from the ruler fit (solid) and
the human-annotated polygon (dashed), plus a zoom on one corner.  Also writes
figs/supp_ruler_constants.pdf (per-photo window side W and pitch ratio r, from
methods/results/ruler/calib150.csv).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
sys.path.insert(0, os.path.join(ROOT, "methods"))
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402
import ruler_calib as RC  # noqa: E402

# validated with the dataviz validator (dark photo background): fine / coarse / ruler window
C_FINE, C_COARSE, C_WIN, C_HUMAN = "#3987e5", "#d95926", "#199e70", "#ffffff"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42})


def _gamma(img_rgb, g=0.8):
    x = img_rgb.astype(np.float32) / 255.0
    return np.clip(x ** g, 0, 1)


def panel(ax, i, label, corner_idx=3, zoom_mm=8.0):
    img = R.load_image(i)
    gt = R.load_corners(i)
    cal = RC.calibrate(img)
    assert cal["ok"], i
    C = cal["corners"]
    side = np.linalg.norm(C - np.roll(C, -1, 0), axis=1).mean()
    pad = 0.2 * side
    x0, y0 = np.floor(C.min(0) - pad).astype(int)
    x1, y1 = np.ceil(C.max(0) + pad).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    rgb = _gamma(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ext = (x0, x1, y1, y0)

    def draw(a, ms, lw):
        a.imshow(rgb, extent=ext, interpolation="lanczos")
        for sd, o in cal["obs"].items():
            f = o["feet"]
            if o["scale"] == "mm":
                a.plot(f[:, 0], f[:, 1], "o", ms=ms, mfc=C_FINE, mec="black", mew=0.3, ls="none", zorder=5)
            else:
                a.plot(f[:, 0], f[:, 1], "s", ms=ms * 1.05, mfc=C_COARSE, mec="black", mew=0.3, ls="none", zorder=5)
        P = np.vstack([gt, gt[:1]])
        a.plot(P[:, 0], P[:, 1], "--", color=C_HUMAN, lw=lw * 0.8, dashes=(3, 2))
        P = np.vstack([C, C[:1]])
        a.plot(P[:, 0], P[:, 1], "-", color=C_WIN, lw=lw)
        a.set_xticks([])
        a.set_yticks([])

    draw(ax, 2.0, 0.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    mm = side / cal["W_mm"]
    err = np.linalg.norm(C - gt, axis=1) * 25.4 / np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
    H, W = img.shape[:2]
    ax.set_title(f"{label} #{i}, {W}$\\times${H} px", fontsize=7, pad=2)
    # zoom on one corner (centred slightly outside it, so both scales' ticks show), inset placed
    # over the fabric on the opposite side of the window
    cx, cy = C[corner_idx]
    half = zoom_mm * mm / 2
    m = C.mean(0)
    out = np.array([cx, cy]) - m
    out /= np.linalg.norm(out)
    zx, zy = np.array([cx, cy]) + out * 0.2 * half
    # inset position in axes fraction, at the window centre shifted away from the zoomed corner
    wx = (m[0] - x0) / (x1 - x0) - out[0] * 0.04
    wy = 1 - (m[1] - y0) / (y1 - y0) + out[1] * 0.04
    sz = 0.42
    loc = [wx - sz / 2, wy - sz / 2, sz, sz]
    axin = ax.inset_axes(loc)
    draw(axin, 4.0, 1.0)
    axin.set_xlim(zx - half, zx + half)
    axin.set_ylim(zy + half, zy - half)
    for s_ in axin.spines.values():
        s_.set_edgecolor("white")
        s_.set_linewidth(0.6)
    ax.add_patch(plt.Rectangle((zx - half, zy - half), 2 * half, 2 * half, fill=False, ec="white", lw=0.5, zorder=6))
    return cal, err


def constants_fig(out):
    d = pd.read_csv(os.path.join(ROOT, "methods", "results", "ruler", "calib150.csv"))
    d = d[d.ok]
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.25))
    for a, col, ref, xl in ((axs[0], "W_mm", None, "window side $W$ [fine divisions]"),
                            (axs[1], "r10_free", (2.5, 2.54), "coarse / fine pitch ratio $r$")):
        x = d[col].dropna().values
        lo, hi = np.percentile(x, [1, 99])
        a.hist(np.clip(x, lo, hi), bins=30, color="#2a78d6", edgecolor="white", linewidth=0.3)
        a.axvline(np.median(x), color="#0b0b0b", lw=0.8)
        if ref:
            for v, ls in zip(ref, ("-", ":")):
                a.axvline(v, color="#52514e", lw=0.6, ls=ls)
            a.text(2.543, a.get_ylim()[1] * 0.80, "2.54", fontsize=5.5, ha="left", color="#52514e")
            a.text(2.497, a.get_ylim()[1] * 0.95, "2.5", fontsize=5.5, ha="right", color="#52514e")
        a.set_xlabel(xl, fontsize=6.5)
        a.tick_params(labelsize=5.5, length=2, width=0.4)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_yticks([])
    axs[0].set_ylabel("photos", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs=2, type=int, default=[8, 55])
    ap.add_argument("--out", default=os.path.join(PAPER, "figs", "fig_ruler.pdf"))
    a = ap.parse_args()
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 1.75))
    for ax, i, lab, ci in zip(axs, a.ids, ("(a)", "(b)"), (3, 1)):
        cal, err = panel(ax, i, lab, corner_idx=ci)
        print(i, "corner err mm", err.round(3), "W", round(cal["W_mm"], 3))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.03)
    fig.savefig(a.out, dpi=400)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=250)
    plt.close(fig)
    constants_fig(os.path.join(os.path.dirname(a.out), "supp_ruler_constants.pdf"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
