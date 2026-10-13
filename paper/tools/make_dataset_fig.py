"""Dataset teaser figure for the paper (Fig. 1).

Top row: raw smartphone photos taken through the pick glass with the annotated
1-inch window polygon. Bottom row: the rectified 1-inch crop used by every counter
(annotation_tool/data/crops, produced by common/rectify.py) and a 1/4-inch zoom.

Run with the shared venv2 python (numpy<2, cv2, matplotlib):
    <venv2>/bin/python tools/make_dataset_fig.py [--ids 0,86,55,15] [--rows raw,crop,quarter]
The paper uses --rows raw,quarter (compact two-row version, Oct 7 page cut).
Writes figs/fig_dataset.pdf and figs/fig_dataset.png. Reads only; never modifies data.
"""
import argparse
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TOOL = os.path.join(ROOT, "annotation_tool")
RAW = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "final_image_set")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0,86,55,15")
    ap.add_argument("--out", default=os.path.join(PAPER, "figs", "fig_dataset"))
    ap.add_argument("--rows", default="raw,crop,quarter", help="subset/order of raw,crop,quarter")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    man = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
    by_id = {im["id"]: im for im in man["images"]}

    rows = a.rows.split(",")
    hgt = {"raw": 1.55, "crop": 2.0, "quarter": 2.0}
    fig, axes = plt.subplots(len(rows), len(ids), figsize=(1.75 * len(ids), sum(hgt[r] for r in rows) - 0.4),
                             gridspec_kw={"height_ratios": [hgt[r] for r in rows]}, squeeze=False)
    for j, i in enumerate(ids):
        im = by_id[i]
        raw = cv2.cvtColor(cv2.imread(os.path.join(RAW, im["file"])), cv2.COLOR_BGR2RGB)
        crop = cv2.cvtColor(cv2.imread(os.path.join(TOOL, im["crop"])), cv2.COLOR_BGR2RGB)
        h, w = raw.shape[:2]
        scale = 900.0 / max(h, w)
        small = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        poly = [(x * scale, y * scale) for x, y in im["polygon"]]
        ax = axes[rows.index("raw"), j] if "raw" in rows else None
        S = crop.shape[0]
        if ax is not None:
            ax.imshow(small)
            xs, ys = zip(*(poly + [poly[0]]))
            ax.plot(xs, ys, "-", color="#f2c200", lw=1.4)
            ax.set_title(f"#{i}: {w}x{h}", fontsize=7)
        ax2 = axes[rows.index("crop"), j] if "crop" in rows else None
        if ax2 is not None:
            ax2.imshow(crop)
            ax2.set_title(f"1-inch crop, S={S} px", fontsize=7)
        q = S // 4
        c0 = (S - q) // 2
        ax3 = axes[rows.index("quarter"), j] if "quarter" in rows else None
        if ax3 is not None:
            ax3.imshow(crop[c0:c0 + q, c0:c0 + q], interpolation="nearest")
            ax3.set_title("central 1/4 inch" if ax2 is not None else f"central 1/4 in., S={S}", fontsize=7)
        for x in (ax, ax2, ax3):
            if x is not None:
                x.set_xticks([])
                x.set_yticks([])
    plt.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=220)
    fig.savefig(a.out + ".png", dpi=160)
    print("wrote", a.out + ".pdf/.png")


if __name__ == "__main__":
    main()
