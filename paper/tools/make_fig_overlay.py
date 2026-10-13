"""make_fig_overlay.py -- Fig. 3 (Job W4, Oct 12): human dots vs the counter's crossings on a zoomed strip.

    <python with numpy, matplotlib, pillow> tools/make_fig_overlay.py <labels.csv> [--ids 39,29] [--outdir figs]

Same strip as handover/results/labelday_diag_FLSA_20261012/overlay.py (the middle 18% of the horizontal guide line
of the contrast-enhanced crop), drawn with matplotlib so that the ticks are thick and every panel carries a
px-per-yarn scale bar.  <labels.csv> is a label_day merged export with the columns id, rect_size, h_line_y, h_dots
(e.g. <run>/work/merged/scoring_headline/annotations_<annotator>.csv); the counter's crossings come from the
annotation tool's manifest (annotation_tool/data/manifest.json, proposal.h_positions; never shown to the annotators).

PROVISIONAL until label day: the Oct 12 version uses the first annotator's dots only.
Writes <outdir>/fig_overlay.pdf/.png (3.25 in wide). Deterministic. Reads only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
TOOL = os.path.join(os.path.dirname(PAPER), "annotation_tool")
FS = 6.5
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS, "pdf.fonttype": 42})
C_HUM, C_OURS = "#D55E00", "#0072B2"


def strip(i, rows, imgs):
    it, r = imgs[i], rows[i]
    im = ImageOps.autocontrast(Image.open(os.path.join(TOOL, it["crop_enh"])).convert("L"), cutoff=1)
    W, H = im.size
    rs = float(r["rect_size"]) if r["rect_size"] else W
    s = W / rs
    y = float(r["h_line_y"]) * s
    hd = np.array([float(v) * s for v in r["h_dots"].split()])
    pr = it.get("proposal") or {}
    ps = W / pr["rect_size"][0] if pr else 1.0
    pd_ = np.array([float(v) * ps for v in pr.get("h_positions", [])])
    x0, x1 = int(W * 0.40), int(W * 0.58)
    half = max(12, int(W * 0.07))
    crop = np.asarray(im.crop((x0, int(y - half), x1, int(y + half))), dtype=float)
    hum = hd[(hd >= x0) & (hd < x1)] - x0
    ours = pd_[(pd_ >= x0) & (pd_ < x1)] - x0
    hc = float(r["h_count"])
    return crop, hum, ours, W / hc, rs / hc   # bar length in strip px; native px per yarn (label)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels")
    ap.add_argument("--ids", default="39,29")
    ap.add_argument("--titles", default="scene 39: counter reads half|scene 29: within 5%")
    ap.add_argument("--outdir", default=os.path.join(PAPER, "figs"))
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    titles = a.titles.split("|")
    m = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
    imgs = {it["id"]: it for it in m["images"]}
    rows = {int(r["id"]): r for r in csv.DictReader(open(a.labels))}
    W, H = 3.25, 1.42
    fig, axs = plt.subplots(1, len(ids), figsize=(W, H))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.03, top=0.88, wspace=0.04)
    for ax, i, t in zip(np.atleast_1d(axs), ids, titles):
        crop, hum, ours, ppy, ppy_nat = strip(i, rows, imgs)
        h, w = crop.shape
        pad = 0.16 * h
        ax.imshow(crop, cmap="gray", extent=(0, w, h, 0), interpolation="nearest", vmin=0, vmax=255)
        for x in hum:
            ax.plot([x, x], [-pad, 0], color=C_HUM, lw=1.3, solid_capstyle="butt", clip_on=False)
        for x in ours:
            ax.plot([x, x], [h, h + pad], color=C_OURS, lw=1.3, solid_capstyle="butt", clip_on=False)
        # scale bar: 5 yarns at the human count's px per yarn (Job W7: label with one decimal, as the caption)
        L = 5 * ppy
        xb, yb = w - L - 0.04 * w, h - 0.09 * h
        ax.plot([xb, xb + L], [yb, yb], color="white", lw=2.2, solid_capstyle="butt")
        ax.plot([xb, xb + L], [yb, yb], color="black", lw=1.0, solid_capstyle="butt")
        ax.text(xb + L / 2, yb - 0.05 * h, f"5 yarns ({ppy_nat:.1f} px/yarn)", ha="center", va="bottom", fontsize=FS - 1,
                color="white", bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.55))
        ax.set_xlim(0, w)
        ax.set_ylim(h + pad, -pad)
        ax.set_title(t, fontsize=FS, pad=1.5)
        ax.axis("off")
        print(f"#{i}: strip {w}x{h}px, human ticks {len(hum)}, counter ticks {len(ours)}, {ppy_nat:.1f} px/yarn (native)")
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "fig_overlay")
    fig.savefig(out + ".pdf", dpi=600, metadata={"CreationDate": None, "Creator": "tools/make_fig_overlay.py"})
    fig.savefig(out + ".png", dpi=300)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
