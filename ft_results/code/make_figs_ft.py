"""Figures for the synthetic-pointing fine-tune (run on the Mac after syncing ft_results/).

  figs/ft_points_legacy.jpg     guide-line bands of real legacy crops: base Molmo2-4B points vs the
                                LoRA model's points (P4 whole view and P4s diagonal quarters)
  figs/ft_points_synval.jpg     synthetic validation bands: exact crossings (green) vs LoRA points (yellow)
  figs/ft_synval_density.pdf    synthetic-val MAPE by density bin, base vs LoRA readouts
  figs/ft_training_curve.pdf    training / validation loss
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FTR = os.path.dirname(HERE)
ROOT = os.path.dirname(FTR)
sys.path.insert(0, HERE)
from views import guide_line_view, quarter_patch, resize_square  # noqa: E402

FIGS = os.path.join(FTR, "figs")
LEG = os.path.join(ROOT, "vlm", "crops", "legacy")


def load_raw(pattern):
    d = {}
    for p in sorted(glob.glob(os.path.join(FTR, "raw", pattern))):
        for l in open(p):
            r = json.loads(l)
            d[(r["model"], r["set"], r["image_id"], r["view"])] = r
    return d


def band(img, axis, pts, colour, half=56, gt=None, line=None):
    """Horizontal band around the guide line (vertical guide: transposed) with point markers."""
    im = img.copy()
    if gt is not None:
        for q in gt:
            x, y = (q * 644 - .5, line * 644 - .5) if axis == "h" else (line * 644 - .5, q * 644 - .5)
            cv2.drawMarker(im, (int(round(x)), int(round(y))), (80, 220, 80), cv2.MARKER_TILTED_CROSS, 9, 2)
    for p in pts:
        cv2.drawMarker(im, (int(round(p[0] / 1000 * 644)), int(round(p[1] / 1000 * 644))), colour,
                       cv2.MARKER_CROSS, 9, 2)
    c = 322
    b = im[c - half:c + half] if axis == "h" else np.ascontiguousarray(im[:, c - half:c + half].transpose(1, 0, 2))
    return b


def label(b, text):
    b = b.copy()
    cv2.rectangle(b, (0, 0), (b.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(b, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return b


def fig_legacy(items=(("a2", "h"), ("r2", "h"), ("a1", "v"))):
    """Per item: whole-inch band, base vs LoRA (left half of the line, x1.5), then the LoRA quarter view qd1."""
    R = load_raw("*__legacy*.jsonl")
    gt = pd.read_csv(os.path.join(ROOT, "paper", "tables", "legacy13_unified_per_count.csv"))
    gt = gt[(gt["key"] == "est:ours") & (gt["assign"] == "axis")].set_index(["image", "axis"])["true"]
    rows = []
    for nm, axis in items:
        crop = cv2.imread(os.path.join(LEG, f"{nm}.png"))
        g = gt.get((nm, "warp" if axis == "h" else "weft"), np.nan)
        what = "vertical" if axis == "h" else "horizontal"
        full = guide_line_view(resize_square(crop, 644), axis)
        q = guide_line_view(resize_square(quarter_patch(crop, "d1"), 644), axis)
        for model, col, short, img, view, mult in (
                ("molmo2_4b_base", (255, 0, 255), "base, 1 in", full, f"guide_{axis}", 1),
                ("molmo2_4b_synthft", (0, 255, 255), "LoRA, 1 in", full, f"guide_{axis}", 1),
                ("molmo2_4b_synthft", (0, 255, 255), "LoRA, 1/4 in", q, f"qd1_guide_{axis}", 4)):
            r = R.get((model, "legacy", nm, view))
            if r is None:
                continue
            pts = r["parsed"].get("points", [])
            b = band(img, axis, pts, col, half=34)[:, :322]
            b = cv2.resize(b, (644, int(b.shape[0] * 2)), interpolation=cv2.INTER_NEAREST)
            rows.append(label(b, f"{nm} {what} yarns, hand count {g:.0f}/in | {short}: {len(pts)} pts"
                                 + (f" (x4 = {4 * len(pts)})" if mult == 4 else "") + " (left half shown)"))
        rows.append(np.full((8, 644, 3), 255, np.uint8))
    if rows:
        cv2.imwrite(os.path.join(FIGS, "ft_points_legacy.jpg"), np.vstack(rows[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def fig_synval(n=4):
    R = load_raw("*__synval*.jsonl")
    labs = {str(r["seed"]): r for r in (json.loads(l) for l in open(os.path.join(FTR, "data_val_labels.jsonl")))}
    ids = sorted({k[2] for k in R if k[0] == "molmo2_4b_synthft"})[:n]
    rows = []
    for iid in ids:
        lab = labs[iid]
        crop = cv2.imread(os.path.join(FTR, "synval_examples", f"s{int(iid):07d}.jpg"))
        if crop is None:
            continue
        for vname, view in (("full", "guide_h"), ("qd1", "qd1_guide_v")):
            axis = view[-1]
            V = lab["views"][vname]
            src = resize_square(crop, 644) if vname == "full" else resize_square(quarter_patch(crop, "d1"), 644)
            img = guide_line_view(src, axis)
            r = R.get(("molmo2_4b_synthft", "synval", iid, view))
            pts = r["parsed"].get("points", []) if r else []
            rows.append(label(band(img, axis, pts, (0, 255, 255), gt=V[axis], line=V["line"]),
                              f"s{iid} {vname} {lab['weave']} | exact {len(V[axis])} (green) | LoRA {len(pts)} (yellow)"))
    if rows:
        cv2.imwrite(os.path.join(FIGS, "ft_points_synval.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def fig_density():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = os.path.join(FTR, "results", "synval_per_image.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    long = pd.concat([d.assign(ape=d.ape_v, gtc=d.gt_v), d.assign(ape=d.ape_h, gtc=d.gt_h)])
    bins = [30, 50, 75, 100, 130, 165, 205]
    long["bin"] = pd.cut(long["gtc"], bins)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = {("molmo2_4b_base", "P4"): ("#9a9a9a", "--", "base 4B, P4"),
              ("molmo2_4b_base", "P4s"): ("#9a9a9a", ":", "base 4B, P4s"),
              ("molmo2_4b_synthft", "P1"): ("#2a7fb8", "-.", "LoRA, P1 (count)"),
              ("molmo2_4b_synthft", "P4"): ("#d95f02", "--", "LoRA, P4 (whole inch)"),
              ("molmo2_4b_synthft", "P4q"): ("#7570b3", ":", "LoRA, P4q (x4)"),
              ("molmo2_4b_synthft", "P4s"): ("#1b9e77", "-", "LoRA, P4s (4 quarters)")}
    for (m, meth), (c, ls, lab) in styles.items():
        g = long[(long.model == m) & (long.method == meth)]
        if not len(g):
            continue
        s = g.groupby("bin", observed=False).ape.median()
        ax.plot([b.mid for b in s.index], s.values, ls, color=c, marker="o", ms=3, label=lab)
    ax.set_xlabel("exact crossings per inch (synthetic validation)")
    ax.set_ylabel("median abs. % error")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ft_synval_density.pdf"))
    fig.savefig(os.path.join(FIGS, "ft_synval_density.png"), dpi=160)


def fig_training():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = os.path.join(FTR, "logs", "train_rank0.log")
    if not os.path.exists(p):
        return
    tr, va = [], []
    for l in open(p):
        m = re.search(r"step (\d+) lr \S+ gn \S+ P1=([\d.]+)\(n\d+\) P4full=([\d.]+)\(n\d+\) P4zoom=([\d.]+)", l)
        if m:
            tr.append([float(x) for x in m.groups()])
        m = re.search(r"VAL step (\d+) loss_P1 ([\d.]+) loss_P4 ([\d.]+)", l)
        if m:
            va.append([float(x) for x in m.groups()])
    tr, va = np.array(tr), np.array(va)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    if len(tr):
        k = 5
        sm = lambda x: np.convolve(x, np.ones(k) / k, mode="valid")  # noqa: E731
        for j, (lab, c) in enumerate((("train P1", "#2a7fb8"), ("train P4 whole", "#d95f02"),
                                      ("train P4 zoom", "#1b9e77")), 1):
            ax.plot(tr[k - 1:, 0], sm(tr[:, j]), color=c, lw=1, label=lab)
    if len(va):
        ax.plot(va[:, 0], va[:, 2], "k--o", ms=3, lw=1, label="val P4 (synthetic)")
        ax.plot(va[:, 0], va[:, 1], "k:s", ms=3, lw=1, label="val P1 (synthetic)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (24 examples)")
    ax.set_ylabel("answer-token CE")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ft_training_curve.pdf"))
    fig.savefig(os.path.join(FIGS, "ft_training_curve.png"), dpi=160)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_legacy()
    fig_synval()
    fig_density()
    fig_training()
    print("figs ->", FIGS)
