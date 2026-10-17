"""make_fig_dataset_pipeline.py -- the dataset teaser and the pipeline in ONE column figure.

    <python with numpy<2, cv2, matplotlib> tools/make_fig_dataset_pipeline.py [--ids 0,86,55,118] [--pipe 118]
        [--name PG-Lattice] [--outdir figs]

Same footprint as the old fig_dataset (which sat at 0.85 of the column width with two rows):
  top row    : four photos through the pick glass with the annotated window (yellow), as in
               tools/make_dataset_fig.py (its old bottom row, the central quarter inch, is dropped);
  bottom row : the task on the last of them (default #118, chosen in tools/make_fig_pipeline.py):
               (a) a corner of the window: tick feet of the engraved scales (ruler method, run here),
                   the corner fitted to them, the human polygon and the legacy segmentation's box;
               (b) the rectified window with the guide lines H, V and the zoom box;
               (c) 1/8 of the window side with one dot per crossing yarn (the counter's output from
                   methods/proposals.json; never shown to the annotators, not human labels);
               (d) the counter's output: counts per window side, the number of dots on each guide line,
                   weave, confidence and the gate decision (spelled out as "(high)" / "(low)", not "t_c").
Panels (b)-(d) use the crop rectified from the HUMAN corners (as every reported number); with the
ruler's corners the counter's confidence on #118 is 0.63 (methods/results/ruler/count150.csv).
All drawing helpers and every number come from tools/make_fig_pipeline.py (imported, not modified):
no counter is re-run; the ruler calibration and the legacy segmentation (cv_seed=0) are re-run read-only.
Writes <outdir>/fig_dataset_pipeline.pdf/.png at the final printed size (3.25 in wide; include it with
width=\\linewidth). Deterministic (no creation date in the PDF). Reads only; never modifies data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fig_pipeline as MP  # noqa: E402  (sets the matplotlib backend, paths and colours)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

PAPER = MP.PAPER
TOOL = os.path.join(MP.ROOT, "annotation_tool")
RAW = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "final_image_set")  # as in tools/make_dataset_fig.py
FS = 6.0          # every text in the figure (pt)
C_WIN = "#f2c200"  # yellow window polygon, as in tools/make_dataset_fig.py


def photo_panel(ax, im, long_px=520):
    """Whole photo with the annotated window, as in make_dataset_fig.py (raw row)."""
    raw = cv2.imread(os.path.join(RAW, im["file"]))
    h, w = raw.shape[:2]
    ax.imshow(MP._resize_to(MP._rgb(raw, gamma=1.0), long_px), extent=(0, w, h, 0), interpolation="none")
    P = np.array(im["polygon"] + [im["polygon"][0]], float)
    ax.plot(P[:, 0], P[:, 1], "-", color=C_WIN, lw=0.8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    MP._clean(ax)
    return w, h


def corner_panel(ax, d, corner_idx=3, zoom_mm=8.0, zoom_out=0.25, ms=2.4, lw=0.9):
    """Zoom on one window corner, where both engraved scales meet (the inset of make_fig_pipeline's panel 2)."""
    img, cal, gt, box = d["img"], d["cal"], d["gt"], d["box"]
    H, W = img.shape[:2]
    C = cal["corners"]
    side = np.linalg.norm(C - np.roll(C, -1, 0), axis=1).mean()
    hz = zoom_mm * side / cal["W_mm"] / 2
    m = C.mean(0)
    out = (C[corner_idx] - m) / np.linalg.norm(C[corner_idx] - m)
    zc = C[corner_idx] + out * zoom_out * hz
    zx0, zy0 = int(max(0, zc[0] - hz)), int(max(0, zc[1] - hz))
    zx1, zy1 = int(min(W, zc[0] + hz)), int(min(H, zc[1] + hz))
    ax.imshow(MP._resize_to(MP._rgb(img[zy0:zy1, zx0:zx1]), 300), extent=(zx0, zx1, zy1, zy0), interpolation="none")
    P = np.vstack([box, box[:1]])
    ax.plot(P[:, 0], P[:, 1], ls=MP.SEG_LS, color=MP.C_SEG, lw=lw, zorder=4)
    P = np.vstack([C, C[:1]])
    ax.plot(P[:, 0], P[:, 1], "-", color=MP.C_RULER, lw=lw, zorder=5)
    P = np.vstack([gt, gt[:1]])
    ax.plot(P[:, 0], P[:, 1], ls=MP.HUMAN_LS, color=MP.C_HUMAN, lw=lw * 0.8, zorder=6, path_effects=MP.HUMAN_PE)
    for o in cal["obs"].values():
        f = o["feet"]
        ax.plot(f[:, 0], f[:, 1], "o", ms=ms, mfc=MP.C_TICK, mec="black", mew=0.3, ls="none", zorder=8)
    ax.plot(C[:, 0], C[:, 1], "s", ms=ms * 2.0, mfc="none", mec=MP.C_RULER, mew=0.9, ls="none", zorder=7)
    ax.set_xlim(zc[0] - hz, zc[0] + hz)
    ax.set_ylim(zc[1] + hz, zc[1] - hz)
    MP._clean(ax)


def output_panel(ax, d):
    p = d["prop"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#f3f2ee", ec="#9a9893", lw=0.5, transform=ax.transAxes))
    keep = "kept offline (no VLM)" if not d["gate_escalate"] else "escalated to the VLM"
    # no "t_c" here (it is not defined in the figure); the gate's verdict is spelled out instead,
    # and the integer dot counts of the guide lines (panel c) are printed next to the spectral counts.
    level = "high" if not d["gate_escalate"] else "low"
    lines = [
        (f"$h\\times v$ = {p['h_count']:.1f}$\\times${p['v_count']:.1f}", FS + 1.0, "bold", "black"),
        ("threads per window side", FS, "normal", MP.C_SUB),
        (f"({len(p['h_positions'])}, {len(p['v_positions'])} dots)", FS, "normal", MP.C_SUB),
        (f"weave: {MP.WEAVE.get(p['weave_guess'], p['weave_guess'])}", FS, "normal", "black"),
        (f"confidence {p['confidence']:.2f} ({level})", FS, "normal", "black"),
        (f"$\\rightarrow$ {keep}", FS, "normal", MP.C_SUB),
    ]
    if d.get("human") is not None:
        lines[2] = (f"human: {d['human'][0]:g}$\\times${d['human'][1]:g}", FS, "normal", "black")
    ys = np.linspace(0.875, 0.125, len(lines))
    for (t, fs, fw, col), y in zip(lines, ys):
        ax.text(0.5, y, t, ha="center", va="center", fontsize=fs, fontweight=fw, color=col, transform=ax.transAxes)


def octave_panel(ax, scene, labels_csv, seg=(0.36, 0.66), half_frac=0.06):
    """Panel (d), in place of the earlier text box: a strip of one scene along its h guide line, with
    the annotator's dots above (orange) and the spectral counter's crossings below (blue), as in the overlay figure
    (tools/make_fig_overlay.py, same colours). Reads the label CSV (merged format of tools/label_day.py: id, h_dots,
    rect_size, h_line_y, h_count) and the counter's line proposals from the manifest. Reads only."""
    import csv
    from PIL import Image, ImageOps
    C_HUM, C_OURS = "#D55E00", "#0072B2"
    man = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
    it = {im["id"]: im for im in man["images"]}[scene]
    r = {int(x["id"]): x for x in csv.DictReader(open(labels_csv))}[scene]
    im = ImageOps.autocontrast(Image.open(os.path.join(TOOL, it["crop_enh"])).convert("L"), cutoff=1)
    W, H = im.size
    s = W / (float(r["rect_size"]) if r["rect_size"] else W)
    y = float(r["h_line_y"]) * s
    hd = np.array([float(v) * s for v in r["h_dots"].split()])
    pr = it.get("proposal") or {}
    pd_ = np.array([float(v) * W / pr["rect_size"][0] for v in pr.get("h_positions", [])]) if pr else np.array([])
    x0, x1 = int(W * seg[0]), int(W * seg[1])
    half = max(8, int(W * half_frac))
    crop = np.asarray(im.crop((x0, int(y - half), x1, int(y + half))), dtype=float)
    hum, ours = hd[(hd >= x0) & (hd < x1)] - x0, pd_[(pd_ >= x0) & (pd_ < x1)] - x0
    h, w = crop.shape
    pad = 0.22 * h
    ax.imshow(crop, cmap="gray", extent=(0, w, h, 0), interpolation="nearest", vmin=0, vmax=255)
    for x in hum:
        ax.plot([x, x], [-pad, 0], color=C_HUM, lw=0.9, solid_capstyle="butt", clip_on=False)
    for x in ours:
        ax.plot([x, x], [h, h + pad], color=C_OURS, lw=0.9, solid_capstyle="butt", clip_on=False)
    lab = dict(fontsize=FS - 0.5, fontweight="bold", path_effects=MP.HALO, transform=ax.transAxes)
    ax.text(0.02, 0.80, f"annotator {int(round(float(r['h_count'])))}", ha="left", va="top", color=C_HUM, **lab)
    ax.text(0.02, 0.20, f"counter {len(pr.get('h_positions', []))}", ha="left", va="bottom", color=C_OURS, **lab)
    ax.set_xlim(0, w)
    ax.set_ylim(h + pad, -pad)
    ax.axis("off")
    return len(hum), len(ours)


def title(ax, text, gap_pt=1.0, bold=False):
    ax.annotate(text, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, gap_pt), textcoords="offset points",
                ha="center", va="bottom", fontsize=FS, fontweight="bold" if bold else "normal")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", default="0,86,55,118", help="photos of the top row (tools/make_dataset_fig.py uses 0,86,55,15)")
    ap.add_argument("--pipe", type=int, default=118, help="scene of the bottom (pipeline) row")
    ap.add_argument("--name", default="PG-Lattice", help="counter name (\\ours)")
    ap.add_argument("--zoom", type=float, default=0.125, help="zoom side as a fraction of the window side")
    ap.add_argument("--human", default=None, help="optional 'h,v' human counts of the pipeline scene")
    ap.add_argument("--row", type=float, default=0.52, help="height of each row (in)")
    ap.add_argument("--top-titles", action="store_true", help="titles above the photos (default: labels inside them)")
    ap.add_argument("--outdir", default=os.path.join(PAPER, "figs"))
    ap.add_argument("--panel-d", choices=("octave", "output"), default="octave",
                    help="(d): 'octave' = annotator vs counter dots on --oct-scene (default); 'output' = old text box")
    ap.add_argument("--oct-scene", type=int, default=45)
    ap.add_argument("--oct-labels", default=None,
                    help="label CSV with h_dots (merged export of tools/label_day.py, e.g. A2's; required for --panel-d octave)")
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    man = json.load(open(os.path.join(TOOL, "data", "manifest.json")))
    by_id = {im["id"]: im for im in man["images"]}
    d = MP.load_scene(a.pipe, a.zoom)
    d["human"] = tuple(float(x) for x in a.human.split(",")) if a.human else None

    Wf = 3.25
    xl, xr = 0.02, 3.23
    tz = 0.115                 # one 6-pt title line above each row
    hr = a.row
    Hf = round((2 if a.top_titles else 1) * tz + 2 * hr + (0.03 if a.top_titles else 0.04), 3)
    fig = plt.figure(figsize=(Wf, Hf))
    # top row: four photos (4:3), centred, equal gaps
    ar = [by_id[i]["orig_w"] / by_id[i]["orig_h"] for i in ids]
    ws = [hr * r for r in ar]
    gap1 = (xr - xl - sum(ws)) / (len(ids) - 1)
    y1 = Hf - (tz if a.top_titles else 0.01) - hr
    x = xl
    for i, w in zip(ids, ws):
        ax = MP._ax(fig, Wf, Hf, x, y1, w, hr)
        pw, ph = photo_panel(ax, by_id[i])
        if a.top_titles:
            title(ax, f"#{i}: {pw}$\\times${ph}")
        else:
            ax.text(0.03, 0.96, f"#{i}", transform=ax.transAxes, ha="left", va="top", fontsize=FS, color="white",
                    fontweight="bold", path_effects=MP.HALO)
            ax.text(0.97, 0.04, f"{pw}$\\times${ph}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS - 0.5,
                    color="white", path_effects=MP.HALO)
        x += w + gap1
    # bottom row: (a) corner, (b) rectified, (c) zoom, (d) output, with arrows
    y2 = 0.01
    gap = 0.14
    a1 = MP._ax(fig, Wf, Hf, xl, y2, hr, hr)
    a2 = MP._ax(fig, Wf, Hf, xl + hr + gap, y2, hr, hr)
    a3 = MP._ax(fig, Wf, Hf, xl + 2 * (hr + gap), y2, hr, hr)
    x4 = xl + 3 * (hr + gap)
    a4 = MP._ax(fig, Wf, Hf, x4, y2, xr - x4, hr)
    corner_panel(a1, d)
    title(a1, "(a) ticks, corner")
    MP.panel_rectified(a2, d, long_px=300, lw=0.6)
    title(a2, f"(b) #{a.pipe}, rectified")
    nh, nv = MP.panel_zoom(a3, d, long_px=300, ms=2.2, lw=0.6)
    title(a3, f"(c) dots, 1/{round(1 / a.zoom)} side")
    if a.panel_d == "octave":
        if a.oct_labels is None:
            sys.exit("--panel-d octave needs --oct-labels <merged export with h_dots, e.g. A2's>; the human counts with dot "
                     "positions are released after review")
        oh, oo = octave_panel(a4, a.oct_scene, a.oct_labels)
        title(a4, f"(d) #{a.oct_scene}: a factor of two")
        print(f"(d) #{a.oct_scene}: {oh} annotator / {oo} counter ticks in the strip")
    else:
        output_panel(a4, d)
        title(a4, f"(d) {a.name} output")
    for p_, q_ in ((a1, a2), (a2, a3), (a3, a4)):
        MP._arrow(fig, p_, q_)
    out = os.path.join(a.outdir, "fig_dataset_pipeline")
    os.makedirs(a.outdir, exist_ok=True)
    fig.savefig(out + ".pdf", dpi=600, metadata={"CreationDate": None, "Creator": "tools/make_fig_dataset_pipeline.py"})
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png ({Wf} x {Hf} in); #{a.pipe}: {nh} H / {nv} V dots in the zoom, "
          f"ours {d['prop']['h_count']:.2f} x {d['prop']['v_count']:.2f}, corners {np.round(d['corner_mm'], 2)} mm")


if __name__ == "__main__":
    main()
