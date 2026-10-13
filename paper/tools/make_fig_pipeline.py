"""make_fig_pipeline.py -- pipeline overview figure: the whole task on ONE real scene.

    <python with numpy<2, cv2, matplotlib> tools/make_fig_pipeline.py [--id 118] [--variant strip|col|both]

Panels (left to right):
  (1) the phone photo taken through the pick glass (no overlay);
  (2) window localisation: tick feet of the glass's engraved scales found by the ruler method
      (methods/ruler_calib.py, run here), the window and its four corners fitted to them, the
      human polygon (Labelbox XML via common/rectify.py) and the box of the 2019 segmentation
      (methods/legacy_2019/seg_final.py, run here with cv_seed=0 exactly as methods/ablate.py does);
  (3) the rectified window (common/rectify.py, from the human polygon as in every reported
      number) with the H and V guide lines and the zoom box;
  (4) zoom: one dot per yarn crossing each line = the counter's output
      (methods/proposals.json; machine output, never shown to the annotators, NOT human labels;
      Job Z, Oct 12: the annotators count from scratch, so the old word "proposals" is gone);
  (5) the counter's output for this photo: h x v per window side, weave guess, confidence and
      the frozen cascade gate decision (tables/cascade_150_decisions.csv).
Numbers quoted in the panels are read from stored results (no counter is re-run):
methods/proposals.json (h, v, weave, confidence), methods/results/ruler/count150.csv (count
with the ruler's corners), methods/results/all150_crops.csv (count on the 2019-segmentation
crop), tables/cascade_150_decisions.csv (gate).

Writes figs/fig_pipeline_strip.pdf/.png (full text width, 6.875 in) and/or
figs/fig_pipeline_col.pdf/.png (one column, 3.25 in).  Figure sizes are the final printed
sizes, so include them with width=\\textwidth / \\linewidth (font sizes are then exact).
Deterministic: no randomness except cv2.kmeans inside the 2019 segmentation, seeded with
cv2.setRNGSeed(0) (the seed used for every 2019-segmentation number in the paper); the PDF
carries no creation date, so reruns are byte-identical.  Reads only; never modifies data.
Text sizes: 7 pt titles, >= 6 pt everywhere else; Okabe-Ito colours plus distinct markers/dashes.

Default scene #118 (--id): a full-resolution scene representative (4608x3456, S = 1849 px),
not in the 20-scene blind IAA subset and not one of the annotation-guide examples (27, 65, 93,
110) or the scenes of Fig. dataset / fig_ruler (0/9, 86, 55, 15, 8); all four tick rows are
sharp; the ruler fit is among the best (quality 0.355, corners 0.09-0.22 mm from the human
polygon, count with ruler corners within 0.1 %); the counter is confident (0.93, 2/1 twill,
72.5 x 51.5, about 25 px per yarn, so the zoom dots are well separated) and the frozen gate keeps
it offline; and the 2019 segmentation shows its typical failure (a non-fallback box that takes in
the frame, IoU 0.58, counts 30-33 % high; the 2019 segmentation misses 5 % on 58 % of scenes).
The lower-right shadow of the phone is left in on purpose (a real capture condition).
Optional --human h,v adds the human counts to panel 5 after label day.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
METHODS = os.path.join(ROOT, "methods")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, METHODS)
sys.path.insert(0, os.path.join(METHODS, "legacy_2019"))
import rectify as R  # noqa: E402
import ruler_calib as RC  # noqa: E402
import seg_final as SF  # noqa: E402  (verbatim 2019-segmentation harness; imported read-only)

# Okabe-Ito colours (colour-blind safe); every overlay also differs in marker or line style.
C_TICK = "#E69F00"    # orange: tick feet of the engraved scales
C_RULER = "#009E73"   # bluish green: window + corners fitted to the ticks
C_HUMAN = "#FFFFFF"   # white dashed: human polygon
C_SEG = "#CC79A7"     # reddish purple dash-dot: 2019 segmentation box
C_H = "#56B4E9"       # sky blue: H guide line and its dots (vertical yarns)
C_V = "#F0E442"       # yellow: V guide line and its dots (horizontal yarns)
C_ARROW = "#52514e"
C_SUB = "#3a3a3a"
FS_T, FS_S = 7.0, 6.0  # title / everything else (pt); nothing below 6 pt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS_S, "axes.linewidth": 0.4,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
HALO = [pe.withStroke(linewidth=1.3, foreground="black")]
HUMAN_LS = (0, (2.5, 2))
HUMAN_PE = [pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()]  # white dashes, black outline
SEG_LS = (0, (4, 1.5, 1, 1.5))
WEAVE = {"plain": "plain", "twill_2_1": "2/1 twill", "twill_3_1": "3/1 twill", "twill_2_2": "2/2 twill",
         "satin": "satin", "unsure": "unsure"}


# ----------------------------------------------------------------------------- data
def _csv_row(path, key, val):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if all(r[k] == v for k, v in zip(key, val)):
                return r
    raise KeyError((path, val))


def load_scene(i, zoom_in=0.125):
    img = R.load_image(i)
    gt = R.load_corners(i)
    cal = RC.calibrate(img)
    assert cal["ok"], f"ruler calibration failed on #{i}"
    _, box, fb = SF.segment(R.image_path(i), cv_seed=0)
    box = R.order_corners(np.asarray(box, np.float64))
    crop = R.rectify(i, img=img)
    prop = json.load(open(os.path.join(METHODS, "proposals.json")))[str(i)]
    S = crop.shape[0]
    assert prop["rect_size"] == [S, S], (prop["rect_size"], S)

    side_gt = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
    corner_mm = np.linalg.norm(cal["corners"] - gt, axis=1) * 25.4 / side_gt
    # IoU of the 2019 box with the human polygon (raster, full resolution)
    H_, W_ = img.shape[:2]
    m1 = np.zeros((H_, W_), np.uint8)
    m2 = np.zeros((H_, W_), np.uint8)
    cv2.fillPoly(m1, [np.int32(np.round(gt))], 1)
    cv2.fillPoly(m2, [np.int32(np.round(box))], 1)
    iou = float((m1 & m2).sum()) / float((m1 | m2).sum())

    rr = _csv_row(os.path.join(METHODS, "results", "ruler", "count150.csv"), ["id"], [str(i)])
    rows = {}
    with open(os.path.join(METHODS, "results", "all150_crops.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["image"] == str(i) and r["estimator"] == "ours" and r["kind"] == "clean":
                rows[r["crop"]] = r
    seg_h, seg_v = float(rows["seg_final"]["h"]), float(rows["seg_final"]["v"])
    gate = _csv_row(os.path.join(PAPER, "tables", "cascade_150_decisions.csv"), ["image"], [str(i)])

    z = int(round(zoom_in * S))
    cx, cy = prop["v_line_x"], prop["h_line_y"]
    zb = (cx - z // 2, cy - z // 2, z)  # x0, y0, side (rectified px)
    return dict(i=i, img=img, gt=gt, cal=cal, box=box, seg_fallback=bool(fb), crop=crop, prop=prop, S=S,
                corner_mm=corner_mm, iou=iou, seg_h=seg_h, seg_v=seg_v,
                ruler_h=float(rr["h_ruler"]), ruler_v=float(rr["v_ruler"]),
                gate_escalate=gate["escalate_gate"] == "True", t_c=float(gate["t_c"]), zb=zb,
                zoom_in=zoom_in)


# ----------------------------------------------------------------------------- drawing helpers
def _rgb(bgr, gamma=0.85):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(x ** gamma, 0, 1)


def _resize_to(arr, long_px):
    h, w = arr.shape[:2]
    s = long_px / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    out = cv2.resize(arr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)
    return np.clip(out, 0, 1) if out.dtype.kind == "f" else out


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def _title(ax, num, title, sub=None, gap_pt=1.2):
    """Two-line panel heading in figure points, above the axes."""
    ax.annotate(f"({num}) {title}", xy=(0.5, 1), xycoords="axes fraction", xytext=(0, gap_pt + (FS_S + 1.2 if sub else 0)),
                textcoords="offset points", ha="center", va="bottom", fontsize=FS_T, fontweight="bold")
    if sub:
        ax.annotate(sub, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, gap_pt), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS_S, color=C_SUB)


def _arrow(fig, a, b, y=None):
    """Horizontal arrow from the right edge of axes a to the left edge of axes b (figure coords)."""
    pa, pb = a.get_position(), b.get_position()
    yy = y if y is not None else (pb.y0 + pb.y1) / 2
    fig.add_artist(FancyArrowPatch((pa.x1 + 0.006, yy), (pb.x0 - 0.006, yy), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=7, lw=0.9, color=C_ARROW))


def panel_photo(ax, d, long_px=560):
    img = d["img"]
    H, W = img.shape[:2]
    ax.imshow(_resize_to(_rgb(img), long_px), extent=(0, W, H, 0), interpolation="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    _clean(ax)


def panel_window(ax, d, long_px=460, ms=1.3, lw=0.8, inset=True, corner_idx=3, zoom_mm=8.0, inset_sz=0.46,
                 zoom_out=0.25):
    """Photo around the frame: tick feet, ruler window + corners, human polygon, 2019 box.
    With inset=True a zoom on one window corner (both scales meet there) is drawn over the
    fabric on the opposite side, so that the ticks and the corner agreement are visible."""
    img, cal, gt, box = d["img"], d["cal"], d["gt"], d["box"]
    H, W = img.shape[:2]
    allp = np.vstack([box, cal["corners"], gt])
    span = (allp.max(0) - allp.min(0)).max()
    c = (allp.max(0) + allp.min(0)) / 2
    half = 0.54 * span
    x0, x1 = max(0, c[0] - half), min(W, c[0] + half)
    y0, y1 = max(0, c[1] - half), min(H, c[1] + half)
    xi0, yi0, xi1, yi1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
    rgb = _rgb(img[yi0:yi1, xi0:xi1])
    C = cal["corners"]

    def draw(a, im, ext, ms_, lw_, msc):
        a.imshow(im, extent=ext, interpolation="none")
        P = np.vstack([box, box[:1]])
        a.plot(P[:, 0], P[:, 1], ls=SEG_LS, color=C_SEG, lw=lw_, zorder=4)
        P = np.vstack([C, C[:1]])
        a.plot(P[:, 0], P[:, 1], "-", color=C_RULER, lw=lw_, zorder=5)
        P = np.vstack([gt, gt[:1]])
        a.plot(P[:, 0], P[:, 1], ls=HUMAN_LS, color=C_HUMAN, lw=lw_ * 0.8, zorder=6, path_effects=HUMAN_PE)
        for o in cal["obs"].values():
            f = o["feet"]
            a.plot(f[:, 0], f[:, 1], "o", ms=ms_, mfc=C_TICK, mec="black", mew=0.25, ls="none", zorder=8)
        a.plot(C[:, 0], C[:, 1], "s", ms=msc, mfc="none", mec=C_RULER, mew=0.9, ls="none", zorder=7)
        _clean(a)

    draw(ax, _resize_to(rgb, long_px), (xi0, xi1, yi1, yi0), ms, lw, ms * 2.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    if not inset:
        return
    side = np.linalg.norm(C - np.roll(C, -1, 0), axis=1).mean()
    px_mm = side / cal["W_mm"]
    hz = zoom_mm * px_mm / 2
    m = C.mean(0)
    out = C[corner_idx] - m
    out /= np.linalg.norm(out)
    zc = C[corner_idx] + out * zoom_out * hz
    zx0, zy0 = int(max(0, zc[0] - hz)), int(max(0, zc[1] - hz))
    zx1, zy1 = int(min(W, zc[0] + hz)), int(min(H, zc[1] + hz))
    # inset centred in the window, shifted away from the zoomed corner, in axes fraction
    wx = (m[0] - x0) / (x1 - x0) - out[0] * 0.10
    wy = 1 - (m[1] - y0) / (y1 - y0) + out[1] * 0.10
    axin = ax.inset_axes([wx - inset_sz / 2, wy - inset_sz / 2, inset_sz, inset_sz])
    draw(axin, _resize_to(_rgb(img[zy0:zy1, zx0:zx1]), 240), (zx0, zx1, zy1, zy0), ms * 2.4, lw * 1.3, ms * 5)
    axin.set_xlim(zc[0] - hz, zc[0] + hz)
    axin.set_ylim(zc[1] + hz, zc[1] - hz)
    for s_ in axin.spines.values():
        s_.set_edgecolor("white")
        s_.set_linewidth(0.6)
    ax.add_patch(Rectangle((zc[0] - hz, zc[1] - hz), 2 * hz, 2 * hz, fill=False, ec="white", lw=0.5, zorder=9))


def panel_rectified(ax, d, long_px=460, lw=0.7, label_lines=True):
    crop, S, p = d["crop"], d["S"], d["prop"]
    ax.imshow(_resize_to(_rgb(crop), long_px), extent=(0, S, S, 0), interpolation="none")
    ax.axhline(p["h_line_y"], color=C_H, lw=lw)
    ax.axvline(p["v_line_x"], color=C_V, lw=lw)
    x0, y0, z = d["zb"]
    ax.add_patch(Rectangle((x0, y0), z, z, fill=False, ec="white", lw=0.7, zorder=6))
    ax.add_patch(Rectangle((x0, y0), z, z, fill=False, ec="black", lw=0.25, zorder=7))
    if label_lines:
        ax.text(0.035 * S, p["h_line_y"] - 0.03 * S, "H", color=C_H, fontsize=FS_S, fontweight="bold",
                ha="left", va="bottom", path_effects=HALO)
        ax.text(p["v_line_x"] + 0.03 * S, 0.035 * S, "V", color=C_V, fontsize=FS_S, fontweight="bold",
                ha="left", va="top", path_effects=HALO)
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    _clean(ax)


def panel_zoom(ax, d, long_px=360, ms=3.0, lw=0.7):
    crop, p = d["crop"], d["prop"]
    x0, y0, z = d["zb"]
    sub = crop[y0:y0 + z, x0:x0 + z]
    ax.imshow(_resize_to(_rgb(sub), long_px), extent=(x0, x0 + z, y0 + z, y0), interpolation="none")
    ax.axhline(p["h_line_y"], color=C_H, lw=lw)
    ax.axvline(p["v_line_x"], color=C_V, lw=lw)
    hx = np.array([x for x in p["h_positions"] if x0 <= x <= x0 + z], float)
    vy = np.array([y for y in p["v_positions"] if y0 <= y <= y0 + z], float)
    ax.plot(hx, np.full_like(hx, p["h_line_y"]), "o", ms=ms, mfc=C_H, mec="black", mew=0.4, ls="none", zorder=5)
    ax.plot(np.full_like(vy, p["v_line_x"]), vy, "o", ms=ms, mfc=C_V, mec="black", mew=0.4, ls="none", zorder=5)
    ax.set_xlim(x0, x0 + z)
    ax.set_ylim(y0 + z, y0)
    for s in ax.spines.values():
        s.set_edgecolor("black")
    _clean(ax)
    return len(hx), len(vy)


def panel_output(ax, d, fs_big=8.0, compact=False):
    """Text panel: the counter's estimate for this photo (machine output, no human count yet)."""
    p = d["prop"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#f3f2ee", ec="#9a9893", lw=0.5, transform=ax.transAxes))
    keep = "answered offline (no VLM)" if not d["gate_escalate"] else "escalated to the VLM"
    conf = f"confidence {p['confidence']:.2f}"
    lines = [
        (f"$h \\times v$ = {p['h_count']:.1f} $\\times$ {p['v_count']:.1f}", fs_big, "bold", "black"),
        ("threads per window side", FS_S, "normal", C_SUB),
        (f"({len(p['h_positions'])} and {len(p['v_positions'])} dots)", FS_S, "normal", C_SUB)
        if d.get("human") is None else
        (f"human: {d['human'][0]:g} $\\times$ {d['human'][1]:g}", FS_S + 0.5, "normal", "black"),
        (f"weave: {WEAVE.get(p['weave_guess'], p['weave_guess'])}", FS_S + 0.5, "normal", "black"),
        (conf + (" $\\geq t_c$" if not d["gate_escalate"] else " $< t_c$"), FS_S + 0.5, "normal", "black"),
        (f"$\\rightarrow$ {keep}", FS_S, "normal", C_SUB),
    ]
    if not compact:
        lines.insert(3, (f"ruler corners: {d['ruler_h']:.1f} $\\times$ {d['ruler_v']:.1f}", FS_S, "normal", C_SUB))
    ys = np.linspace(0.87, 0.13, len(lines))
    for (t, fs, fw, col), y in zip(lines, ys):
        ax.text(0.5, y, t, ha="center", va="center", fontsize=fs, fontweight=fw, color=col, transform=ax.transAxes)


def legend_handles():
    return [Line2D([], [], ls="none", marker="o", ms=2.8, mfc=C_TICK, mec="black", mew=0.3, label="glass ticks"),
            Line2D([], [], color=C_RULER, lw=0.9, marker="s", ms=3.2, mfc="none", mec=C_RULER, mew=0.8,
                   label="ruler fit, corners"),
            Line2D([], [], color=C_HUMAN, lw=0.8, ls=HUMAN_LS, path_effects=HUMAN_PE, label="human polygon"),
            Line2D([], [], color=C_SEG, lw=0.9, ls=SEG_LS, label="2019 segmentation")]


# ----------------------------------------------------------------------------- layouts
def _ax(fig, W, H, x, y, w, h):
    """Axes at (x, y) inches from the bottom-left, size w x h inches."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def strip(d, out, name):
    """Full text width (6.875 in), 1.42 in tall: five panels left to right with arrows."""
    W, H = 6.875, 1.42
    hI = 1.02            # image height
    yI = 0.17            # bottom of the image row (room for the legend line under it)
    gap = 0.19           # arrow gap between panels
    img = d["img"]
    w1 = hI * img.shape[1] / img.shape[0]
    w5 = W - 0.01 - (w1 + 3 * hI + 4 * gap) - 0.01
    fig = plt.figure(figsize=(W, H))
    x = 0.01
    a1 = _ax(fig, W, H, x, yI, w1, hI); x += w1 + gap
    a2 = _ax(fig, W, H, x, yI, hI, hI); x += hI + gap
    a3 = _ax(fig, W, H, x, yI, hI, hI); x += hI + gap
    a4 = _ax(fig, W, H, x, yI, hI, hI); x += hI + gap
    a5 = _ax(fig, W, H, x, yI, w5, hI)
    Hh, Ww = img.shape[:2]
    panel_photo(a1, d)
    _title(a1, 1, "Phone photo", f"through the pick glass, {Ww}$\\times${Hh}")
    panel_window(a2, d)
    _title(a2, 2, "Window", "glass ticks $\\rightarrow$ 4 corners")
    panel_rectified(a3, d)
    _title(a3, 3, "Rectified window", "from the human corners; lines H, V")  # Job Z (Oct 12)
    nh, nv = panel_zoom(a4, d)
    _title(a4, 4, "Count crossings", f"one dot per yarn, 1/{round(1 / d['zoom_in'])} of the side")
    panel_output(a5, d)
    _title(a5, 5, "Output", f"{name}, machine estimate")
    for a, b in ((a1, a2), (a2, a3), (a3, a4), (a4, a5)):
        _arrow(fig, a, b)
    yl = 0.075 / H
    fig.legend(handles=legend_handles(), loc="center left", ncol=4, fontsize=FS_S, frameon=False,
               bbox_to_anchor=(0.004, yl), bbox_transform=fig.transFigure,
               handlelength=1.8, handletextpad=0.35, columnspacing=0.9, borderaxespad=0, borderpad=0)
    p4, p5 = a4.get_position(), a5.get_position()
    fig.text((p4.x0 + p5.x1) / 2, yl, "dots in (4): the counter's output, never shown to the annotators", ha="center",
             va="center", fontsize=FS_S, color=C_SUB, style="italic")
    _save(fig, out)
    return nh, nv


def column(d, out, name):
    """One column (3.25 in), 2.0 in tall: (1)-(3) on the first row, (4)-(5) + legend on the second."""
    W, H = 3.25, 2.0
    tz = 0.235           # title zone (two lines) above each row
    xl, xr = 0.03, 3.22  # both rows span the same width
    img = d["img"]
    h1 = 0.82
    w1 = h1 * img.shape[1] / img.shape[0]
    gap = (xr - xl - w1 - 2 * h1) / 2
    y1 = H - tz - h1
    fig = plt.figure(figsize=(W, H))
    x = xl
    a1 = _ax(fig, W, H, x, y1, w1, h1); x += w1 + gap
    a2 = _ax(fig, W, H, x, y1, h1, h1); x += h1 + gap
    a3 = _ax(fig, W, H, x, y1, h1, h1)
    h2 = 0.68
    y2 = 0.015
    assert y2 + h2 + tz <= y1 - 0.015, "rows overlap"
    x = xl
    a4 = _ax(fig, W, H, x, y2, h2, h2); x += h2 + gap
    w5 = 1.16
    a5 = _ax(fig, W, H, x, y2, w5, h2); x += w5 + 0.07
    Hh, Ww = img.shape[:2]
    panel_photo(a1, d)
    _title(a1, 1, "Photo", f"{Ww}$\\times${Hh}")
    panel_window(a2, d, ms=1.0, lw=0.6, inset_sz=0.48)
    _title(a2, 2, "Window", "ticks $\\rightarrow$ corners")
    panel_rectified(a3, d, lw=0.6)
    _title(a3, 3, "Rectified", f"$S$ = {d['S']} px")
    panel_zoom(a4, d, ms=2.5)
    _title(a4, 4, "Count", "machine dots")
    panel_output(a5, d, fs_big=7.0, compact=True)
    _title(a5, 5, "Output", f"{name} (machine)")
    for a, b in ((a1, a2), (a2, a3), (a4, a5)):
        _arrow(fig, a, b)
    leg = fig.legend(handles=legend_handles(), loc="upper left", ncol=1, fontsize=FS_S, frameon=False,
                     bbox_to_anchor=(x / W, (y2 + h2 + 0.03) / H), bbox_transform=fig.transFigure,
                     handlelength=1.7, handletextpad=0.35, labelspacing=0.45, borderaxespad=0, borderpad=0,
                     title="in (2):", title_fontsize=FS_S)
    leg._legend_box.align = "left"
    _save(fig, out)


def _save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", dpi=600, metadata={"CreationDate": None, "Creator": "tools/make_fig_pipeline.py"})
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", type=int, default=118, help="scene representative to show (default 118)")
    ap.add_argument("--variant", choices=["strip", "col", "both"], default="both")
    ap.add_argument("--name", default="PG-Lattice", help="counter name printed in panel 5 (\\ours)")
    ap.add_argument("--zoom", type=float, default=0.125, help="zoom side as a fraction of the window side")
    ap.add_argument("--human", default=None, help="optional 'h,v' human counts of this scene (after label day); "
                    "replaces the dot-count line of panel 5")
    ap.add_argument("--outdir", default=os.path.join(PAPER, "figs"))
    a = ap.parse_args()
    d = load_scene(a.id, a.zoom)
    d["human"] = tuple(float(x) for x in a.human.split(",")) if a.human else None
    p = d["prop"]
    print(f"#{a.id}: {d['img'].shape[1]}x{d['img'].shape[0]}, S={d['S']}, ours h x v = {p['h_count']:.2f} x "
          f"{p['v_count']:.2f} ({len(p['h_positions'])}/{len(p['v_positions'])} dots), weave {p['weave_guess']}, "
          f"conf {p['confidence']:.3f}, gate escalate={d['gate_escalate']} (t_c={d['t_c']:.3f})")
    print(f"  ruler: W={d['cal']['W_mm']:.2f} fine div., corner error vs human (mm) {np.round(d['corner_mm'], 2)}"
          f" (mean {d['corner_mm'].mean():.2f}); count with ruler corners {d['ruler_h']:.2f} x {d['ruler_v']:.2f}")
    print(f"  2019 segmentation: box {d['box'].round(0).tolist()}, fallback={d['seg_fallback']}, IoU vs human "
          f"{d['iou']:.3f}; count on its crop {d['seg_h']:.1f} x {d['seg_v']:.1f} "
          f"({100 * (d['seg_h'] / p['h_count'] - 1):+.0f}% / {100 * (d['seg_v'] / p['v_count'] - 1):+.0f}%)")
    if a.variant in ("strip", "both"):
        nh, nv = strip(d, os.path.join(a.outdir, "fig_pipeline_strip"), a.name)
        print(f"  zoom shows {nh} H dots and {nv} V dots")
    if a.variant in ("col", "both"):
        column(d, os.path.join(a.outdir, "fig_pipeline_col"), a.name)


if __name__ == "__main__":
    main()
