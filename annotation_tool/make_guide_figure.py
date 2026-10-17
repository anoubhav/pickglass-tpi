"""Annotation guide / paper figure: how thread counts are marked.

For a few example photos (different weaves, resolutions, capture conditions) this
draws, left to right:
  (a) the original phone photo through the pick glass, with the original hand-marked
      1-inch window (red polygon),
  (b) the rectified 1-inch crop with the two guide lines and one dot per yarn,
  (c) a 1/10-inch zoom along the horizontal guide line (dots on vertical yarns),
  (d) a 1/10-inch zoom along the vertical guide line (dots on horizontal yarns).

Dots come from data/saved annotations when a human has finished that image
(--source human), otherwise from the algorithm proposals in manifest.json
(--source proposals, the default), which the figure labels as unverified.

Usage (venv with matplotlib, numpy, pillow):
  python make_guide_figure.py --ids 8 1 20 77 2 --out guide/annotation_guide
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "photos")), "final_image_set")

WEAVE_NAMES = {
    "plain": "plain 1/1", "twill_2_1": "twill 2/1", "twill_3_1": "twill 3/1",
    "twill_2_2": "twill 2/2", "satin": "satin", "unsure": "unsure",
}


def load_manifest():
    with open(os.path.join(HERE, "data", "manifest.json")) as f:
        m = json.load(f)
    m = m.get("images", m) if isinstance(m, dict) else m
    return {str(e["id"]): e for e in m}


def load_human():
    """Finished human records from data/saved/*.json, keyed by image id."""
    out = {}
    d = os.path.join(HERE, "data", "saved")
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        recs = data.get("records", data) if isinstance(data, dict) else data
        items = recs.items() if isinstance(recs, dict) else [(str(r.get("id")), r) for r in recs]
        for k, r in items:
            if isinstance(r, dict) and r.get("done"):
                out[str(k)] = r
    return out


def dots_for(entry, human, source):
    """Return (h_line_y, h_xs, v_line_x, v_ys, label)."""
    if source == "human" and str(entry["id"]) in human:
        r = human[str(entry["id"])]
        h = r.get("h_line", {}) or {}
        v = r.get("v_line", {}) or {}
        hy = h.get("y", r.get("h_line_y"))
        vx = v.get("x", r.get("v_line_x"))
        hx = [p[0] if isinstance(p, (list, tuple)) else p.get("x") for p in (h.get("dots") or r.get("h_dots") or [])]
        vy = [p[1] if isinstance(p, (list, tuple)) else p.get("y") for p in (v.get("dots") or r.get("v_dots") or [])]
        if hy is not None and vx is not None and hx and vy:
            return hy, sorted(hx), vx, sorted(vy), "human-verified"
    p = entry["proposal"]
    return p["h_line_y"], sorted(p["h_positions"]), p["v_line_x"], sorted(p["v_positions"]), "algorithm proposal (unverified)"


def zoom_window(center, size, span):
    lo = int(round(center - span / 2))
    lo = max(0, min(lo, size - int(span)))
    return lo, lo + int(span)


def draw_row(axes, entry, human, source):
    a0, a1, a2, a3 = axes
    photo = Image.open(os.path.join(PHOTOS, entry["file"])).convert("RGB")
    crop = np.asarray(Image.open(os.path.join(HERE, entry["crop"])).convert("RGB"))
    W = crop.shape[1]
    hy, hx, vx, vy, label = dots_for(entry, human, source)
    weave = WEAVE_NAMES.get(entry["proposal"].get("weave_guess", ""), "?")

    # (a) original photo + hand-marked window
    a0.imshow(photo)
    a0.add_patch(Polygon(entry["polygon"], closed=True, fill=False, ec="#ff3b30", lw=1.6))
    a0.set_title(f"#{entry['id']}  {entry['orig_w']}x{entry['orig_h']} px", fontsize=8)

    # (b) 1-inch crop, guide lines, dots, zoom boxes
    a1.imshow(crop)
    a1.axhline(hy, color="#ff2d92", lw=0.8)
    a1.axvline(vx, color="#ff2d92", lw=0.8)
    a1.scatter(hx, [hy] * len(hx), s=3, c="#00e5ff", edgecolors="k", linewidths=0.3, zorder=3)
    a1.scatter([vx] * len(vy), vy, s=3, c="#ffd60a", edgecolors="k", linewidths=0.3, zorder=3)
    span = W / 10.0  # 1/10 inch
    zx0, zx1 = zoom_window(W * 0.35, W, span)
    zy0, zy1 = zoom_window(W * 0.35, W, span)
    a1.add_patch(Rectangle((zx0, hy - span / 2), span, span, fill=False, ec="#00e5ff", lw=1.0))
    a1.add_patch(Rectangle((vx - span / 2, zy0), span, span, fill=False, ec="#ffd60a", lw=1.0))
    a1.set_title(f"1 inch: {len(hx)} x {len(vy)} threads/inch ({weave})", fontsize=8)

    # (c) zoom on the horizontal line: dots on vertical yarns
    y0, y1 = zoom_window(hy, W, span)
    a2.imshow(crop[y0:y1, zx0:zx1], extent=(zx0, zx1, y1, y0), interpolation="lanczos")
    xs = [x for x in hx if zx0 <= x <= zx1]
    a2.axhline(hy, color="#ff2d92", lw=1.0)
    a2.scatter(xs, [hy] * len(xs), s=60, c="#00e5ff", edgecolors="k", linewidths=0.6, zorder=3)
    a2.set_title(f"1/10 inch along H line: {len(xs)} vertical yarns", fontsize=8)

    # (d) zoom on the vertical line: dots on horizontal yarns
    x0, x1 = zoom_window(vx, W, span)
    a3.imshow(crop[zy0:zy1, x0:x1], extent=(x0, x1, zy1, zy0), interpolation="lanczos")
    ys = [y for y in vy if zy0 <= y <= zy1]
    a3.axvline(vx, color="#ff2d92", lw=1.0)
    a3.scatter([vx] * len(ys), ys, s=60, c="#ffd60a", edgecolors="k", linewidths=0.6, zorder=3)
    a3.set_title(f"1/10 inch along V line: {len(ys)} horizontal yarns", fontsize=8)

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["8", "1", "20", "77", "2"])
    ap.add_argument("--source", choices=["proposals", "human"], default="proposals")
    ap.add_argument("--out", default=os.path.join(HERE, "guide", "annotation_guide"))
    a = ap.parse_args()

    man = load_manifest()
    human = load_human() if a.source == "human" else {}
    ids = [i for i in a.ids if i in man]
    fig, axes = plt.subplots(len(ids), 4, figsize=(11, 2.75 * len(ids)), squeeze=False)
    labels = set()
    for row, i in zip(axes, ids):
        labels.add(draw_row(row, man[i], human, a.source))
    fig.suptitle(
        "How threads are marked: (a) photo with the hand-marked 1-inch window (red), (b) rectified inch with the two guide lines,\n"
        "(c, d) 1/10-inch zooms: one dot per yarn crossing the line.  Dots shown: " + ", ".join(sorted(labels)),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=170)
    fig.savefig(a.out + ".pdf")
    print("wrote", a.out + ".png", "and .pdf;", "rows:", ids)


if __name__ == "__main__":
    main()
