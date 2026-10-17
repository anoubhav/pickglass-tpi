"""make_fig_octave.py -- an octave (half-or-double) figure for Sec. 5.2: predicted vs human thread counts.

    <python with numpy, matplotlib> tools/make_fig_octave.py <main150_per_scene.csv> [--outdir figs]
        [--methods "Ours (lattice + fusion)" "GPT-6.1-sol, P1"] [--labels PG-Lattice GPT-6.1-sol]

Reads a per-scene file written by tools/label_day.py (work/stage/main150_per_scene.csv: one row per scene and
method with the human counts h, v and the predictions h_p, v_p) and plots, for each method, every count
(h and v of every scene) as prediction against human count on log axes, with the lines y = x, y = 2x and
y = x/2 and the +-5% tolerance band.  Points near y = x/2 (y = 2x) are counts that read half (double) the
human number of yarns: octave errors.

The human count is the one in the per-scene file: e.g.
a run on one annotator's export scores against that annotator's counts
only, and the per-scene file of a --real run uses the mean of the two annotators. The paper's Fig. 3
(figs/fig_octave_hist_H2.pdf) is a histogram against A2, the reference annotator; this script plots whichever
human count its per-scene file holds.

Bottom panel: histogram of log2(predicted / human) for both methods.
Writes <outdir>/fig_octave.pdf/.png at the printed size (3.25 in wide; include with width=\\linewidth).
Deterministic (no creation date in the PDF). Reads only.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OCT_TOL = 0.3   # the paper's octave band (Sec. 5.2)

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
FS = 6.5
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                     "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.3,
                     "ytick.minor.width": 0.3, "pdf.fonttype": 42})
C_H, C_V = "#0072B2", "#D55E00"   # Okabe-Ito blue / vermillion


def load(path, method):
    pts = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != method:
                continue
            for ax in ("h", "v"):
                try:
                    hu, pr = float(r[ax]), float(r[ax + "_p"])
                except (TypeError, ValueError):
                    continue
                if hu > 0 and pr > 0:
                    pts.append((ax, hu, pr, int(r["scene_id"])))
    return pts


def panel(ax, pts, title, lo, hi):
    x = np.geomspace(lo, hi, 50)
    ax.fill_between(x, 0.95 * x, 1.05 * x, color="0.55", lw=0, zorder=1)  # darker band
    for k, ls, lab in ((1.0, "-", "$y=x$"), (2.0, "--", "$2x$"), (0.5, ":", "$x/2$")):
        ax.plot(x, k * x, ls, color="0.35", lw=0.6, zorder=2)
    for axis, col, mk in (("h", C_H, "o"), ("v", C_V, "^")):
        p = [(hu, pr) for a, hu, pr, _ in pts if a == axis]
        if p:
            p = np.array(p)
            ax.scatter(p[:, 0], p[:, 1], s=5, marker=mk, facecolor="none", edgecolor=col, lw=0.5, zorder=3,
                       label=f"${axis}$")
    r = np.array([pr / hu for _, hu, pr, _ in pts])
    n = len(r)
    # the paper's one octave band (Sec. 5.2): | |log2 r| - 1 | <= 0.3,
    # i.e. x0.41-0.62 (half) and x1.62-2.46 (double); the older 0.4-0.6x / 1.7-2.4x tallies are not used
    lr = np.log2(r)
    half = int(np.sum(np.abs(lr + 1) <= OCT_TOL))
    dbl = int(np.sum(np.abs(lr - 1) <= OCT_TOL))
    ok = int(np.sum(np.abs(r - 1) <= 0.05))
    ax.text(0.04, 0.96, f"{title}\n{ok}/{n} within $\\pm$5%\n{half} near $x/2$, {dbl} near $2x$",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS - 0.5, linespacing=1.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [25, 50, 100, 200]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_aspect("equal")
    ax.tick_params(length=2, pad=1.5)
    return half, dbl, ok, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("per_scene", help="per-scene CSV written by tools/label_day.py (main150_per_scene.csv)")
    ap.add_argument("--outdir", default=os.path.join(PAPER, "figs"))
    ap.add_argument("--methods", nargs="+", default=["Ours (lattice + fusion)", "GPT-6.1-sol, P1"])
    ap.add_argument("--labels", nargs="+", default=["PG-Lattice", "GPT-6.1-sol P1"])
    a = ap.parse_args()
    W, H = 3.25, 2.55
    fig, axs = plt.subplots(1, len(a.methods), figsize=(W, H))
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.47, top=0.99, wspace=0.10)
    rat = []
    lo, hi = 14, 330
    for i, (ax, m, lab) in enumerate(zip(np.atleast_1d(axs), a.methods, a.labels)):
        pts = load(a.per_scene, m)
        assert pts, f"no rows for method {m!r}"
        half, dbl, ok, n = panel(ax, pts, lab, lo, hi)
        print(f"{lab}: {n} counts, {ok} within 5%, {half} near x/2, {dbl} near 2x (| |log2 r| - 1 | <= {OCT_TOL})")
        if i == 0:
            ax.set_ylabel("predicted count", labelpad=1)
        else:
            ax.set_yticklabels([])
        if i == len(a.methods) - 1:
            ax.legend(loc="lower right", frameon=False, handletextpad=0.1, borderaxespad=0.2, markerscale=1.4)
        # label the x/2 and 2x lines in both panels
        ax.text(hi * 0.62, hi * 0.62 * 0.5 * 0.72, "$x/2$", fontsize=FS - 1, color="0.25")
        ax.text(lo * 1.15, lo * 1.15 * 2 * 1.25, "$2x$", fontsize=FS - 1, color="0.25")
        rat.append((lab, np.array([pr / hu for _, hu, pr, _ in pts])))
    # histogram of log2(predicted / human) for both methods
    axh = fig.add_axes([0.085, 0.085, 0.91, 0.20])
    bins = np.arange(-2.0, 2.0001, 0.125)
    for (lab, r), col, ls in zip(rat, ("#000000", "#CC79A7"), ("-", "--")):
        axh.hist(np.clip(np.log2(r), -1.99, 1.99), bins=bins, histtype="step", color=col, ls=ls, lw=0.9, label=lab)
    for xv, t in ((-1, "$\\times\\frac{1}{2}$"), (0, "$\\times 1$"), (1, "$\\times 2$")):
        axh.axvline(xv, color="0.45", lw=0.5, ls=":" if xv else "-", zorder=0)
        axh.text(xv, 1.02, t, transform=axh.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 0.5)
    axh.set_xlim(-2, 2)
    axh.set_xticks([-2, -1, 0, 1, 2])
    axh.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    axh.set_xlabel("predicted / human count (log scale)", labelpad=1)
    axh.set_ylabel("counts", labelpad=1)
    axh.tick_params(length=2, pad=1.5)
    axh.spines[["top", "right"]].set_visible(False)
    axh.legend(loc="upper left", frameon=False, fontsize=FS - 0.5, handlelength=1.6, borderaxespad=0.1)
    fig.text(0.54, 0.385, "human count (threads per window side)", ha="center", va="bottom", fontsize=FS)
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "fig_octave")
    fig.savefig(out + ".pdf", metadata={"CreationDate": None, "Creator": "tools/make_fig_octave.py"})
    fig.savefig(out + ".png", dpi=300)
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
