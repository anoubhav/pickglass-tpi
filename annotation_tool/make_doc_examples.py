"""Example images for the annotation guide doc.

howto_count.png  - where the dots go (plain weave zoom) and why twill ribs are not yarns.
bad_photos.png   - real examples of the common photo problems, picked automatically
                   from the counter's per-image quality measures.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from make_guide_figure import load_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "guide")
PRED = os.path.join(HERE, "..", "methods", "predictions_ours_150.csv")


def crop_img(entry):
    return np.asarray(Image.open(os.path.join(HERE, entry["crop"])).convert("RGB"))


def zoom(img, cx, cy, span):
    h, w = img.shape[:2]
    x0 = int(max(0, min(w - span, cx - span / 2)))
    y0 = int(max(0, min(h - span, cy - span / 2)))
    return img[y0:y0 + int(span), x0:x0 + int(span)], x0, y0


def howto(man):
    e = man["8"]  # plain weave, high resolution
    t = man["20"]  # twill 3/1, high resolution
    img, timg = crop_img(e), crop_img(t)
    p = e["proposal"]
    W = img.shape[1]
    span = W / 10
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))

    ax[0].imshow(img)
    ax[0].axhline(p["h_line_y"], color="#ff2d92", lw=1.2)
    ax[0].axvline(p["v_line_x"], color="#ff2d92", lw=1.2)
    ax[0].set_title("1-inch window + two guide lines", fontsize=10)

    z, x0, y0 = zoom(img, W * 0.35, p["h_line_y"], span)
    ax[1].imshow(z, extent=(x0, x0 + z.shape[1], y0 + z.shape[0], y0), interpolation="lanczos")
    xs = [x for x in p["h_positions"] if x0 <= x <= x0 + z.shape[1]]
    ax[1].axhline(p["h_line_y"], color="#ff2d92", lw=1.5)
    ax[1].scatter(xs, [p["h_line_y"]] * len(xs), s=90, c="#00e5ff", edgecolors="k", linewidths=0.8, zorder=3)
    ax[1].set_title(f"1/10 inch: one dot per yarn ({len(xs)} here)", fontsize=10)

    tz, _, _ = zoom(timg, timg.shape[1] * 0.4, timg.shape[0] * 0.45, timg.shape[1] / 4)
    ax[2].imshow(tz, interpolation="lanczos")
    ax[2].set_title("Twill (1/4 inch): diagonal stripes are not yarns", fontsize=10)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "howto_count.png"), dpi=150)
    plt.close(fig)


def bad_photos(man):
    df = pd.read_csv(PRED)
    df["id"] = df["id"].astype(str)
    df["tilt"] = df[["angle_x", "angle_y"]].abs().max(axis=1)
    df["S"] = df["S"].astype(float)
    df["px_per_yarn"] = df["S"] / df[["h_count", "v_count"]].max(axis=1)
    hi = df[df["S"] > 1000]
    picks = []

    def take(frame, col, asc, label, advice):
        for _, r in frame.sort_values(col, ascending=asc).iterrows():
            if r["id"] not in [q[0] for q in picks]:
                picks.append((r["id"], label, advice))
                return

    take(df, "glare_frac", False, "Glare", "Move the line to a clear row.\nTick Glare.")
    take(hi, "sharpness", True, "Blur", "If yarns are visible when zoomed,\ncount; tick Blur, confidence 2.")
    take(df, "dark_frac", False, "Dark / shadow", "Press E for the enhanced view.\nTick Shadow.")
    take(df, "tilt", False, "Tilted yarns", "Press A (auto-align), check the\npreview, then count.")
    take(df, "px_per_yarn", True, "Low resolution", "Zoom in; if yarns blur together,\ncount anyway, confidence 1.")
    take(df[df["weave"].astype(str).str.contains("satin|unsure")], "confidence", True,
         "Hard pattern", "Count yarns, not colour blocks.\nWeave: Unsure is fine.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 9.6))
    for a, (i, label, advice) in zip(axes.ravel(), picks):
        a.imshow(Image.open(os.path.join(HERE, man[i]["crop"])).convert("RGB"))
        a.set_title(f"{label}  (photo #{i})", fontsize=11, fontweight="bold")
        a.set_xlabel(advice, fontsize=9.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout(h_pad=3.0)
    fig.savefig(os.path.join(OUT, "bad_photos.png"), dpi=130)
    plt.close(fig)
    print("bad photo picks:", picks)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    man = load_manifest()
    howto(man)
    bad_photos(man)
    print("wrote", os.path.join(OUT, "howto_count.png"), "and bad_photos.png")
