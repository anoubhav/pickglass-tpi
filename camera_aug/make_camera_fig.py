"""make_camera_fig.py -- paper figure: one fabric through every PickGlass-Stress camera condition, with
1:1-pixel zooms (denoise smearing, sharpening halos, JPEG / demosaic artefacts), a CFA-moire example and
the label-safety curve (ours' drift vs px per yarn, label_audit.py sweep).

    <venv2>/bin/python make_camera_fig.py [--src r8] [--moire-src <id>]
    -> ../paper/figs/camera_profiles.pdf / .png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import camera_sim as CS  # noqa: E402

FIGS = os.path.join(ROOT, "paper", "figs")
PANELS = [("clean", "Clean crop\n(source)"), ("budget_binned_12mp", "Budget\n12.5 MP"),
          ("budget_fullres_50mp", "Budget\n50 MP mode"), ("midrange_binned", "Mid-range\n(OIS)"),
          ("flagship", "Flagship\n(OIS, merge)"), ("legacy_16mp_2019", "Legacy 16 MP\n(our 2019 phone)"),
          ("budget_lowlight", "Budget,\nlow light"), ("budget_whatsapp", "Budget +\nmessaging app"),
          ("legacy_quarter", "Legacy + q75\nx0.25 resave")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROUTE_STYLE = {"area": ("#2a78d6", "area resample only"), "sensor": ("#eb6834", "sensor at that pitch (Bayer CFA)"),
               "phone": ("#1baf7a", "full budget phone at that pitch"),
               "messaging": ("#eda100", "budget phone + messaging resize")}


def load(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]


def zoom(img, c=(0.5, 0.5), half=0.06, out=220):
    S = img.shape[0]
    x0, x1 = int(round((c[0] - half) * S)), int(round((c[0] + half) * S))
    y0, y1 = int(round((c[1] - half) * S)), int(round((c[1] + half) * S))
    return cv2.resize(img[y0:y1, x0:x1], (out, out), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="r8")
    ap.add_argument("--moire-src", default="")
    ap.add_argument("--moire-ppt", type=float, default=2.5)
    ap.add_argument("--zc", default="0.5,0.5")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(HERE, "stress", "legacy13", "manifest.csv"), dtype={"source_id": str})
    lab = pd.read_csv(os.path.join(HERE, "stress", "legacy13", "labels.csv"), dtype={"source_id": str})
    m = man[(man.source_id == a.src) & (man.rep == 0)].set_index("condition")
    zc = tuple(float(v) for v in a.zc.split(","))
    plt.rcParams.update({"font.size": 6.5, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(7.0, 3.85))
    gs = fig.add_gridspec(3, 9, height_ratios=[1, 1, 1.3], hspace=0.42, wspace=0.05,
                          left=0.01, right=0.99, top=0.93, bottom=0.085)
    for j, (cond, title) in enumerate(PANELS):
        if cond == "clean":
            img = load(os.path.join(ROOT, m.source_path.iloc[0]))
            sub = f"{img.shape[0]} px/in\n(as captured)"
        else:
            r = m.loc[cond]
            img = load(os.path.join(HERE, r.path))
            P = json.loads(r.params_json)
            sub = f"{img.shape[0]} px/in\nISO {P['iso']:.0f}, q{P['jpeg_q']}"
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA))
        S = img.shape[0]
        ax.add_patch(plt.Rectangle(((zc[0] - 0.06) * 360, (zc[1] - 0.06) * 360), 0.12 * 360, 0.12 * 360, fill=False,
                                   ec="#ffffff", lw=0.6))
        ax.set_title(title, fontsize=5.6, color=INK, pad=1.5, linespacing=1.0)
        ax.text(0.5, -0.03, sub, transform=ax.transAxes, ha="center", va="top", fontsize=4.9, color=INK2,
                linespacing=1.0)
        ax.axis("off")
        az = fig.add_subplot(gs[1, j])
        az.imshow(zoom(img, zc))
        az.axis("off")
        if j == 0:
            az.text(0.03, 0.97, "0.12 in, pixels", transform=az.transAxes, ha="left", va="top", fontsize=4.8,
                    color="#ffffff", bbox=dict(fc="#0b0b0b", ec="none", alpha=0.55, pad=0.8))
    # ---- moire example: sensor route (Bayer + bilinear demosaic) vs area at the same px per yarn ----------
    ms = a.moire_src or a.src
    src_path = os.path.join(ROOT, m.source_path.iloc[0]) if ms == a.src else os.path.join(
        ROOT, "annotation_tool", "data", "crops", f"{ms}.jpg")
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)
    g = lab[lab.source_id == ms]
    fine = float(max(g.h_src.iloc[0], g.v_src.iloc[0])) if len(g) else None
    if fine is None:
        o = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
        fine = float(max(o.loc[int(ms), "h_count"], o.loc[int(ms), "v_count"]))
    side = int(round(a.moire_ppt * fine))
    area = cv2.resize(src, (side, side), interpolation=cv2.INTER_AREA)
    P = CS.sample_params("budget_binned_12mp", CS.stable_seed("fig-moire", ms))
    P.update(ppi_profile=float(side), theta_deg=2.0, demosaic="bilinear", lens_sigma_px=0.4)
    sens, _ = CS.render(src, P, enabled={"sample", "cfa"})
    P2 = dict(P, demosaic="edge_aware")
    sens2, _ = CS.render(src, P2, enabled={"sample", "cfa"})
    for j, (im, t) in enumerate(((area, f"ideal resampling\n{a.moire_ppt:g} px/yarn"),
                                 (sens, f"Bayer CFA, bilinear\n{a.moire_ppt:g} px/yarn"),
                                 (sens2, f"Bayer CFA, edge-aware\n{a.moire_ppt:g} px/yarn"))):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        ax.imshow(zoom(im[..., ::-1], (0.5, 0.5), half=0.2, out=400))
        ax.set_anchor("W")
        ax.set_title(t, fontsize=5.6, color=INK, pad=1.5, linespacing=1.0, loc="left")
        ax.axis("off")
    # ---- label-safety curve -------------------------------------------------------------------------------
    sw_p = os.path.join(HERE, "results", "label_audit_sweep.csv")
    ax = fig.add_subplot(gs[2, 6:9])
    if os.path.exists(sw_p):
        sw = pd.read_csv(sw_p)
        for route, (col, lab_) in ROUTE_STYLE.items():
            s = sw[(sw.route == route) & (sw.n >= 10) & (sw.ppt_med <= 8)].sort_values("ppt_med")
            ax.plot(s.ppt_med, s.med_abs, color=col, lw=1.6, marker="o", ms=3.0, mec="#fcfcfb", mew=0.6,
                    label=lab_)
        ax.axhline(3.0, color=INK2, lw=0.8, ls=(0, (3, 2)))
        ax.text(7.9, 3.4, "3 % (collapse criterion)", color=INK2, fontsize=4.8, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_xlim(1.6, 8.1)
        ax.set_ylim(0.1, 100)
        ax.set_xlabel("px per yarn (output crop, counted axis)", fontsize=5.6, labelpad=1)
        ax.set_ylabel("ours: median |count change| (%)", fontsize=5.6, labelpad=1)
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=5.2, length=2, pad=1)
        ax.legend(fontsize=4.6, frameon=False, loc="upper right", handlelength=1.6, borderaxespad=0.2)
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"camera_profiles.{ext}"), dpi=300)
    print("wrote", os.path.join(FIGS, "camera_profiles.pdf"), "and .png")


if __name__ == "__main__":
    main()
