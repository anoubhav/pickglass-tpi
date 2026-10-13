"""Paper-ready qualitative overlays for the two remote-GPU baselines (column width, 3.3 in).

    <venv2>/bin/python gpu_results/make_paper_figs.py     # -> gpu_results/figs_paper/*.pdf / *.png

fig_gpu_molmo2_pointing   Molmo2-8B bf16 points on the exact 644-px model inputs (whole inch = P4,
                          central quarter inch = P4q) for a coarse (a5) and a dense (r6) plain weave.
fig_gpu_countgd_strips    CountGD's pre-registered rule (CG-B-yarn-ex3): counted boxes in the
                          1-inch horizontal strip, zoom on 8 yarn spacings, same two fabrics.
fig_gpu_countgd_cells     CountGD whole-crop cell detections -> lattice TPI (CG-W-ex3-lat, post hoc):
                          a success (a5) and a period-doubling failure (a10), with the automatic exemplars.
Points are re-parsed from the raw answers with vlm/parse_output.py; boxes are the stored detections
(score > 0.23, owned by their tile).  Colours: the dataviz reference categorical slots 1-3
(validated all-pairs, light mode); every mark has a white ring and identity is also carried by shape
and the legend.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe          # noqa: E402
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Rectangle     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VLM = os.path.join(ROOT, "vlm")
sys.path.insert(0, VLM)
from gt_legacy import legacy_gt              # noqa: E402
from parse_output import parse_points        # noqa: E402

OUT = os.path.join(HERE, "figs_paper")
INK, INK2 = "#0b0b0b", "#52514e"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"          # blue, orange, aqua (validated, light)
COLW = 3.3
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.5,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
RING = [pe.withStroke(linewidth=1.6, foreground="white")]
TXT_BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88)
GT = legacy_gt()


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK2)
        s.set_linewidth(0.4)


# ============================================================================ Molmo2
def molmo_points():
    P = {}
    for line in open(os.path.join(HERE, "raw", "molmo2_8b_bf16__legacy.jsonl")):
        r = json.loads(line)
        if r["prompt_id"].startswith("P4") and r.get("repeat", 0) == 0:
            P[(r["image_id"], r["view"])] = parse_points(r["raw"])
    return P


def fig_molmo(ids=("a5", "r6")):
    P = molmo_points()
    fig, axes = plt.subplots(len(ids), 2, figsize=(COLW, COLW * len(ids) / 2 + 0.6))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.95, bottom=0.115, wspace=0.03, hspace=0.05)
    for row, img in enumerate(ids):
        g = GT[img]
        for col, (view, k, title) in enumerate((("", 1, "whole inch (P4)"), ("qc_", 4, "central ¼ inch, ×4 (P4q)"))):
            ax = axes[row, col]
            im = rgb(os.path.join(VLM, "inputs", "molmo2_8b", "legacy", f"{img}__{'qc' if view else 'crop'}.png"))
            S = im.shape[0]
            ax.imshow(im, interpolation="none")
            for xs, ys in (([0, S], [S // 2, S // 2]), ([S // 2, S // 2], [0, S])):
                ax.plot(xs, ys, color="white", lw=0.6, alpha=0.9, zorder=2)
            ph, pv = P[(img, view + "guide_h")], P[(img, view + "guide_v")]
            for pts, col_, mk in ((ph["points"], C1, "o"), (pv["points"], C2, "D")):
                if pts:
                    a = np.array(pts, float) / 1000.0 * S
                    ax.scatter(a[:, 0], a[:, 1], s=7 if mk == "o" else 5.5, marker=mk, c=col_, edgecolors="white",
                               linewidths=0.45, zorder=3)
            if not view:
                q = S / 4.0
                a0 = round(1.5 * q)
                ax.add_patch(Rectangle((a0, a0), q, q, fill=False, ec="white", lw=0.6, ls=(0, (2, 1.5)), zorder=2))
            nv, nh = ph["n_points"], pv["n_points"]
            # top-left quadrant holds no points (the guide lines run through the centre)
            ax.text(0.035, 0.965, f"{nv * k} × {nh * k}\nGT {g['warp']:g} × {g['weft']:g}", transform=ax.transAxes,
                    fontsize=6.3, color=INK, va="top", ha="left", bbox=TXT_BOX, zorder=4, linespacing=1.15)
            ax.set_xlim(0, S)
            ax.set_ylim(S, 0)
            clean(ax)
            if row == 0:
                ax.set_title(title, fontsize=7, color=INK, pad=2.5)
            if col == 0:
                ax.set_ylabel(f"{img} ({g['weave_name']})", fontsize=7, color=INK, labelpad=2)
    h = [Line2D([], [], marker="o", ls="", ms=3.6, mfc=C1, mec="white", mew=0.45,
                label="point on horizontal line → vertical yarns (warp)"),
         Line2D([], [], marker="D", ls="", ms=3.0, mfc=C2, mec="white", mew=0.45,
                label="point on vertical line → horizontal yarns (weft)")]
    fig.legend(handles=h, loc="lower center", ncol=1, frameon=False, fontsize=6.3, handletextpad=0.3,
               bbox_to_anchor=(0.53, 0.0), labelcolor=INK)
    save(fig, "fig_gpu_molmo2_pointing")


# ============================================================================ CountGD
def cg_records():
    R = {}
    for line in open(os.path.join(HERE, "countgd", "outputs", "raw", "countgd__legacy__shard0of1.jsonl")):
        r = json.loads(line)
        R[(r["image_id"], r["method"], r["part"])] = r
    return R


def fig_cg_strips(ids=("a5", "r6"), method="CG-B-yarn-ex3", n_sp=12, margin=0.45):
    """Zoom on n_sp vertical-yarn spacings of the horizontal 1-inch strip (rows c +- p_y, i.e. two pick
    spacings tall, run_countgd.make_strip) that CountGD counts to give the warp (ends) count."""
    R = cg_records()
    fig, axes = plt.subplots(len(ids), 1, figsize=(COLW, 2.8))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.165, hspace=0.5)
    for ax, img in zip(axes, ids):
        r = R[(img, method, "h")]
        per = R[(img, "PER", "both")]
        crop = rgb(os.path.join(VLM, "crops", "legacy", f"{img}.png"))
        S, px, py = r["side_px"], r["p_x"], r["p_y"]
        c = S / 2.0
        half = max(py, 4.0)
        y0s, y1s = round(c - half), round(c + half)
        w = n_sp * px
        x0, x1 = c - w / 2, c + w / 2
        ax.imshow(crop, interpolation="none")
        ax.add_patch(Rectangle((x0 - 5, y0s), w + 10, y1s - y0s, fill=False, ec="white", lw=0.8, zorder=2))
        for yb in (c - py / 2, c + py / 2):
            ax.plot([x0, x1], [yb, yb], color="white", lw=0.6, ls=(0, (2, 1.5)), zorder=2)
        n_view = 0
        for d in r["detections"]:
            if not (d[4] > 0.23 and d[5]):
                continue
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            if not (x0 <= cx <= x1):
                continue
            n_view += 1
            ax.add_patch(Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, ec=C1, lw=0.6, zorder=3))
            ax.scatter([cx], [cy], s=6, marker="o", c=C1, edgecolors="white", linewidths=0.4, zorder=4)
        # exemplars are stored 3 per tile, in tile order: draw the 3 of the tile nearest the view centre
        ex = r.get("exemplars_crop_xyxy", [])
        tiles = [ex[i:i + 3] for i in range(0, len(ex) - 2, 3)]
        if tiles:
            t = min(tiles, key=lambda T: abs(np.mean([(e[0] + e[2]) / 2 for e in T]) - c))
            for e in t:
                ax.add_patch(Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], fill=False, ec=C3, lw=1.0,
                                       ls=(0, (3, 1.2)), zorder=3))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1s + margin * py, y0s - margin * py)
        clean(ax)
        g = GT[img]
        band = sum(1 for d in r["detections"] if d[4] > 0.23 and d[5] and abs((d[1] + d[3]) / 2 - c) <= py / 2)
        ax.set_title(f"{img}, GT {g['warp']:g} ends: CountGD {r['count']}, centre band {band}, "
                     f"periodicity {S / per['p_x']:.0f}", fontsize=6.5, color=INK, pad=2.5, loc="left")
        ax.text(0.0, -0.03, f"shown: {n_sp} yarn spacings of the 1-inch strip, {n_view} boxes counted in view",
                transform=ax.transAxes, fontsize=5.9, color=INK2, ha="left", va="top")
    h = [Line2D([], [], marker="s", ls="", ms=4.2, mfc="none", mec=C1, mew=0.7, label="counted box (score > 0.23)"),
         Line2D([], [], marker="s", ls="", ms=4.2, mfc="none", mec=C3, mew=0.9, label="automatic exemplars (one tile)"),
         Line2D([], [], color=INK2, lw=0.8, label="strip (2 pick spacings)"),
         Line2D([], [], color=INK2, lw=0.6, ls=(0, (2, 1.5)), label="centre band (1 pick spacing)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.0, handletextpad=0.35,
               columnspacing=0.8, bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    save(fig, "fig_gpu_countgd_strips")


def fig_cg_cells(ids=("a5", "a10"), method="CG-W-ex3", n_cells=9):
    R = cg_records()
    fig, axes = plt.subplots(1, len(ids), figsize=(COLW, COLW / 2 + 0.62))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.115, wspace=0.04)
    for ax, img in zip(axes, ids):
        r = R[(img, method, "whole")]
        crop = rgb(os.path.join(VLM, "crops", "legacy", f"{img}.png"))
        S = r["side_px"]
        g = GT[img]
        pitch = S / np.mean([g["warp"], g["weft"]])                  # GT mean yarn pitch, sets the zoom only
        ex = r.get("exemplars_crop_xyxy", [])
        # zoom centred on the stored exemplar closest to the crop centre (stored for the first tiles only)
        e0 = min(ex, key=lambda e: ((e[0] + e[2]) / 2 - S / 2) ** 2 + ((e[1] + e[3]) / 2 - S / 2) ** 2)
        cx, cy = (e0[0] + e0[2]) / 2, (e0[1] + e0[3]) / 2
        half = n_cells * pitch / 2
        x0, x1 = max(0, cx - half), min(S, cx + half)
        y0, y1 = max(0, cy - half), min(S, cy + half)
        ax.imshow(crop, interpolation="none")
        pts = np.array([((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in r["detections"] if d[4] > 0.23 and d[5]])
        m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        ax.scatter(pts[m, 0], pts[m, 1], s=7, marker="o", c=C1, edgecolors="white", linewidths=0.45, zorder=3)
        for e in ex:
            if x0 <= (e[0] + e[2]) / 2 <= x1 and y0 <= (e[1] + e[3]) / 2 <= y1:
                ax.add_patch(Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], fill=False, ec=C3, lw=1.0,
                                       ls=(0, (3, 1.2)), zorder=4))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        clean(ax)
        ax.set_title(f"{img}: lattice {r['lat_ends']:.0f} × {r['lat_picks']:.0f} TPI\n"
                     f"GT {g['warp']:g} × {g['weft']:g}", fontsize=6.6, color=INK, pad=2.5, linespacing=1.2)
    h = [Line2D([], [], marker="o", ls="", ms=3.6, mfc=C1, mec="white", mew=0.45, label="detected cell centre"),
         Line2D([], [], marker="s", ls="", ms=4.2, mfc="none", mec=C3, mew=0.9, label="automatic exemplar (one cell)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.2, handletextpad=0.35,
               columnspacing=0.9, bbox_to_anchor=(0.5, 0.0), labelcolor=INK)
    save(fig, "fig_gpu_countgd_cells")


if __name__ == "__main__":
    fig_molmo()
    fig_cg_strips()
    fig_cg_cells()
    print("written:", sorted(os.listdir(OUT)))
