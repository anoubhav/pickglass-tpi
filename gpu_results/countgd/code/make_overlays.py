"""Qualitative overlays of CountGD detections on the counting strips (for the paper).

Selection rule (no cherry-picking): among the 13 legacy GT images, rank by the primary
method's best-assignment mean absolute % error and take the best, the median and the worst
image.  Each figure: (a) the 1-inch crop with the H and V strips outlined, (b) a zoom on the
H strip around its centre (12 spacings), (c) the same for the V strip.  Green boxes = counted
detections (score > 0.23, owned by their tile), red dots = their centres, blue = automatic
exemplars, dashed yellow = the +-p_perp/2 centre band used by the post-hoc band rule.
Also writes one whole-crop cell-detection overlay (CG-W-ex3) for the best image.

    python make_overlays.py            -> outputs/figs/countgd_qual_{1,2,3}_<img>.{png,pdf}
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.patches import Rectangle       # noqa: E402
import numpy as np                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cg_lib as L                              # noqa: E402
import evaluate_countgd as E                    # noqa: E402

FIG = os.path.join(L.ROOT, "outputs", "figs")
METHOD = E.PRIMARY
THR = E.THR


def draw_dets(ax, r, x0, y0, x1, y1, lw=0.8):
    for d in r["detections"]:
        if d[4] > THR and d[5]:
            bx = d[:4]
            cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                ax.add_patch(Rectangle((bx[0] - x0, bx[1] - y0), bx[2] - bx[0], bx[3] - bx[1],
                                       fill=False, ec="lime", lw=lw))
                ax.plot(cx - x0, cy - y0, "o", ms=2.2, color="red")
    ex = r["exemplars_crop_xyxy"]

    def inside(e):
        cx, cy = (e[0] + e[2]) / 2, (e[1] + e[3]) / 2
        return x0 <= cx <= x1 and y0 <= cy <= y1

    # only exemplar triples (one per tile) that actually fall inside the zoom window are drawn
    trip = [t for t in (ex[i:i + 3] for i in range(0, len(ex), 3)) if any(inside(e) for e in t)]
    if trip:
        zc = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        best = min(trip, key=lambda t: np.linalg.norm(np.mean([[(e[0] + e[2]) / 2, (e[1] + e[3]) / 2]
                                                                for e in t], 0) - zc))
        for e in [e for e in best if inside(e)]:
            ax.add_patch(Rectangle((e[0] - x0, e[1] - y0), e[2] - e[0], e[3] - e[1],
                                   fill=False, ec="deepskyblue", lw=1.6))


def overlay(rank, img_id, leg, gt, est):
    img = L.load_image("legacy", img_id)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    S = img.shape[0]
    c = S / 2
    rh, rv = leg[(img_id, METHOD, "h")], leg[(img_id, METHOD, "v")]
    px, py = rh["p_x"], rh["p_y"]
    g = gt[img_id]
    e = est[(METHOD, img_id)]
    eb = est[(METHOD + "@band", img_id)]
    fig = plt.figure(figsize=(11, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.9, 0.62], wspace=0.06)
    a0 = fig.add_subplot(gs[0, 0])
    a0.imshow(rgb)
    a0.add_patch(Rectangle((0, c - py), S, 2 * py, fill=False, ec="orange", lw=1.2))
    a0.add_patch(Rectangle((c - px, 0), 2 * px, S, fill=False, ec="magenta", lw=1.2))
    a0.set_title(f"({'abc'[rank - 1]}) {img_id}: 1-inch crop\nGT warp {g['warp']:.0f} / weft {g['weft']:.0f} "
                 f"({g['weave_name']})", fontsize=8)
    a0.axis("off")
    # H strip zoom: 8 spacings around the centre
    half = 4 * px
    x0, x1 = int(max(0, c - half)), int(min(S, c + half))
    y0, y1 = int(max(0, c - 1.5 * py)), int(min(S, c + 1.5 * py))
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(rgb[y0:y1, x0:x1])
    a1.add_patch(Rectangle((0, c - py - y0), x1 - x0, 2 * py, fill=False, ec="orange", lw=1.2))
    for s_ in (-0.5, 0.5):
        a1.plot([0, x1 - x0], [c + s_ * py - y0] * 2, color="yellow", ls="--", lw=0.9)
    draw_dets(a1, rh, x0, y0, x1, y1)
    a1.set_title(f"H strip, zoom on 8 of ~{S / px:.0f} spacings (counts vertical yarns): CountGD {e['v']:.0f}, "
                 f"band rule {eb['v']:.0f}, periodicity {S / px:.0f}, GT {g['warp']:.0f} or {g['weft']:.0f}",
                 fontsize=8)
    a1.axis("off")
    # V strip zoom
    half = 4 * py
    yy0, yy1 = int(max(0, c - half)), int(min(S, c + half))
    xx0, xx1 = int(max(0, c - 1.5 * px)), int(min(S, c + 1.5 * px))
    a2 = fig.add_subplot(gs[0, 2])
    a2.imshow(rgb[yy0:yy1, xx0:xx1])
    a2.add_patch(Rectangle((c - px - xx0, 0), 2 * px, yy1 - yy0, fill=False, ec="magenta", lw=1.2))
    for s_ in (-0.5, 0.5):
        a2.plot([c + s_ * px - xx0] * 2, [0, yy1 - yy0], color="yellow", ls="--", lw=0.9)
    draw_dets(a2, rv, xx0, yy0, xx1, yy1)
    a2.set_title(f"V strip (zoom)\nCountGD {e['h']:.0f}, band {eb['h']:.0f}\nperiodicity {S / py:.0f}", fontsize=8)
    a2.axis("off")
    fig.text(0.01, 0.01,
             f"{METHOD}: CountGD tiled strip (official crop rule; {rh['n_tiles']}+{rv['n_tiles']} tiles), text 'yarn' + 3 "
             f"automatic exemplars per tile, score > {THR}.\nGreen: counted detections (red: centres); blue: the "
             "3 exemplars of the tile nearest the zoom centre; yellow dashed: centre band (one perpendicular spacing).",
             fontsize=7, va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.11)
    os.makedirs(FIG, exist_ok=True)
    base = os.path.join(FIG, f"countgd_qual_{rank}_{img_id}")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base


def whole_overlay(img_id, leg, gt):
    r = leg[(img_id, "CG-W-ex3", "whole")]
    img = L.load_image("legacy", img_id)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    S = img.shape[0]
    q = int(S * 0.3)
    x0 = y0 = int(S / 2 - q / 2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].add_patch(Rectangle((x0, y0), q, q, fill=False, ec="orange", lw=1.2))
    ax[0].set_title(f"{img_id}: whole 1-inch crop", fontsize=9)
    ax[1].imshow(rgb[y0:y0 + q, x0:x0 + q])
    draw_dets(ax[1], r, x0, y0, x0 + q, y0 + q, lw=0.6)
    g = gt[img_id]
    ax[1].set_title(f"CG-W-ex3 cells (zoom): N = {r['count']} vs GT ends x picks = {g['warp'] * g['weft']:.0f}; "
                    f"lattice TPI = ({r['lat_ends']:.1f}, {r['lat_picks']:.1f})", fontsize=8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    base = os.path.join(FIG, f"countgd_whole_cells_{img_id}")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    return base


def main():
    gt = L.legacy_gt()
    leg = E.load("legacy")
    est = E.estimates(leg)
    errs = []
    for img, g in gt.items():
        e = est.get((METHOD, img))
        if not e or not e["ok"]:
            continue
        p, t = sorted([e["v"], e["h"]]), sorted([g["warp"], g["weft"]])
        errs.append((np.mean([abs(a - b) / b for a, b in zip(p, t)]), img))
    errs.sort()
    pick = [errs[0][1], errs[len(errs) // 2][1], errs[-1][1]]
    sel = []
    for k, img in enumerate(pick, 1):
        sel.append(dict(rank=k, image=img, mean_abs_pct_err_best=float(dict((i, v) for v, i in errs)[img] * 100),
                        file=os.path.basename(overlay(k, img, leg, gt, est))))
    sel.append(dict(rank="whole", image=pick[0], file=os.path.basename(whole_overlay(pick[0], leg, gt))))
    json.dump(sel, open(os.path.join(FIG, "countgd_qual_selection.json"), "w"), indent=1)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
