#!/usr/bin/env python3
"""make_readme_figure.py -- the README's overview figure, on a SYNTHETIC fabric only (no mill photo is used).

    python3 tools/make_readme_figure.py [--seed 900070] [--out scratch/overview_synthetic.png]
    (the committed copy is docs/img/overview_synthetic.png, made with the defaults)

Panels: (1) a synthetic photo through the pick glass (cnn/synth.py: rendered fabric, frame and engraved-looking
ticks, perspective, blur, noise) with the true window; (2) the rectified 1-inch window with the two guide lines and
one dot per yarn as found by the FROZEN counter (methods/results/counting_frozen_final.py, confidence model
methods/conf_model.json); (3) a zoom on the horizontal line; (4) the counter's output against the generator's
exact labels. Needs numpy, scipy, opencv-python-headless, matplotlib.
"""
import argparse
import importlib.util
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "cnn"))
import synth  # noqa: E402


def frozen_counter():
    p = os.path.join(REPO, "methods", "results", "counting_frozen_final.py")
    spec = importlib.util.spec_from_file_location("counting_frozen_final", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.CONF_MODEL = json.load(open(os.path.join(REPO, "methods", "conf_model.json")))  # as methods/run_*.py do
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=900070)
    ap.add_argument("--out", default=os.path.join(REPO, "scratch", "overview_synthetic.png"))
    a = ap.parse_args()
    scene, crop, lab = synth.generate(a.seed, max_crop_side=1200, frame_margin=0.18)
    C = frozen_counter()
    est = C.estimate(crop, method="ours")
    gp = C.guide_positions(crop, est["h_count"], est["v_count"])
    S = crop.shape[0]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.82]))
    ax[0].imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    w = np.array(lab["corners"] + [lab["corners"][0]])
    ax[0].plot(w[:, 0], w[:, 1], "-", color="#e8590c", lw=1.6)
    ax[0].set_title("1. Photo through the pick glass\n(synthetic; orange = 1-inch window)", fontsize=9)
    ax[1].imshow(rgb)
    hy, vx = gp["h_line_y"], gp["v_line_x"]
    ax[1].axhline(hy, color="#f2c200", lw=0.8)
    ax[1].axvline(vx, color="#f2c200", lw=0.8)
    ax[1].plot(gp["h_positions"], [hy] * len(gp["h_positions"]), "o", ms=2.2, color="#22d3ee", mec="k", mew=0.3)
    ax[1].plot([vx] * len(gp["v_positions"]), gp["v_positions"], "o", ms=2.2, color="#f472b6", mec="k", mew=0.3)
    ax[1].set_title("2. Rectified window: one dot per yarn\non each guide line (frozen counter)", fontsize=9)
    z = S // 6
    x0 = max(0, int(S / 2 - z))
    ax[2].imshow(rgb[hy - z // 2: hy + z // 2, x0: x0 + 2 * z], extent=(x0, x0 + 2 * z, hy + z // 2, hy - z // 2))
    xs = [x for x in gp["h_positions"] if x0 <= x < x0 + 2 * z]
    ax[2].axhline(hy, color="#f2c200", lw=1.0)
    ax[2].plot(xs, [hy] * len(xs), "o", ms=7, color="#22d3ee", mec="k", mew=0.6)
    ax[2].set_title("3. Zoom on the H line\n(each dot = one vertical yarn)", fontsize=9)
    for k in range(3):
        ax[k].set_xticks([])
        ax[k].set_yticks([])
    ax[3].axis("off")
    txt = (f"4. Output (threads per window side)\n\n"
           f"H line: {est['h_count']:.1f}   (true {lab['h_count']:.1f})\n"
           f"V line: {est['v_count']:.1f}   (true {lab['v_count']:.1f})\n"
           f"weave: {est['weave']}   (true {lab['weave']})\n"
           f"confidence: {est['confidence']:.2f}\n\n"
           f"synthetic seed {a.seed}; no factory\nphoto is shown anywhere in this repo")
    ax[3].text(0.0, 0.95, txt, va="top", ha="left", fontsize=9, family="monospace", transform=ax[3].transAxes)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}: H {est['h_count']:.2f} vs {lab['h_count']:.2f}, V {est['v_count']:.2f} vs "
          f"{lab['v_count']:.2f}, weave {est['weave']} vs {lab['weave']}, conf {est['confidence']:.2f}")


if __name__ == "__main__":
    main()
