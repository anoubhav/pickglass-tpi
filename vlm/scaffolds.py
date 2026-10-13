"""Scaffolded visual prompting for VLM thread counting (Job 2, WVAQ 2027).

Hypothesis
----------
VLMs fail at dense periodic counting partly because they lack a METRIC reference: they do not
know how much fabric a given image distance is, and they lose their place when enumerating
~50-120 near-identical yarns.  Drawing a measurement scaffold on the image (and telling the
model how to use it) should help.  Each scaffold is ONE composite image (SIDE x SIDE, the same
1536-px input size as the P1-P3 API runs) plus a prompt; the answer is a FLAT JSON object whose
`vertical_yarns` / `horizontal_yarns` are the model's per-inch counts (parsed with the unchanged
parse_output.parse_counts) and whose list fields hold the per-cell counts (for a harness-side
re-computation, reported as a secondary diagnostic).

Scaffolds (designed on 3 synthetic images + common sense; frozen before any legacy run)
-------------------------------------------------------------------------------------
S1   ruler      : white margin rulers on the top and left edges with a tick every 0.1 inch
                  (labelled .1 ... .9, 0 and 1 in); the crop fills the rest (1440 px).  Prompt:
                  count the yarns inside one 0.1-inch interval at 3 places, average, x10.
S1t  text-only  : CONTROL for S1 - the plain crop (same pixels as the P1 input) with the S1
                  instruction re-worded without the ruler ("one tenth of the image width").
                  S1 vs S1t isolates the effect of the drawn metric reference.
S2   guide lines: a horizontal and a vertical guide line through the centre (black core,
                  white edges, visible on any fabric) cut by tick marks into 10 numbered
                  0.1-inch segments.  Prompt: count the crossings per segment, sum (full
                  enumeration with bookkeeping, no extrapolation).
S3   insets     : four 0.1 x 0.1-inch patches (centred at the four quadrant centres of the
                  window) magnified x~4 into a 2 x 2 tiling, each labelled with its physical
                  size.  Prompt: count per patch (fractions allowed), yarns/inch = 10 x mean.
S4   profile    : crop (1152 px) with two marginal plots: below it the brightness profile of
                  a horizontal band (0.05 inch tall, centred) -> one peak per vertical yarn;
                  right of it the profile of the vertical band -> one peak per horizontal yarn.
                  Grid lines every 0.1 inch.  Prompt: count the peaks per 0.1-inch cell, sum.
                  `profile_oracle_counts` counts the peaks that are actually in the plotted
                  curves (find_peaks; prominence fixed on the synthetic images), which separates
                  "the plot does not encode the count" from "the model cannot read the plot".

Axis convention as everywhere else: vertical_yarns <-> warp (ends), horizontal_yarns <-> weft.

    python scaffolds.py --synth            # write the 3 synthetic design images + labels
    python scaffolds.py --render a1 S1     # write one composite to scaffold/preview/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SC_DIR = os.path.join(HERE, "scaffold")
SYNTH_DIR = os.path.join(SC_DIR, "synth")
LEGACY_CROP_DIR = os.path.join(HERE, "crops", "legacy")
SIDE = 1536
SCAFFOLD_SET_VERSION = "sc1-2026-10-07"
SCAFFOLDS = ["S1", "S1t", "S2", "S3", "S4"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK, WHITE, GREY = (0, 0, 0), (255, 255, 255), (200, 200, 200)
CYAN = (255, 255, 0)  # BGR


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def resize_square(img, side):
    interp = cv2.INTER_AREA if img.shape[0] > side else cv2.INTER_CUBIC
    return cv2.resize(img, (side, side), interpolation=interp)


def put_text(img, text, org, scale=1.0, color=BLACK, thick=2, anchor="lt", bg=None, pad=4):
    """cv2.putText with an anchor (l/c/r x t/m/b) and an optional filled background box."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    if anchor[0] == "c":
        x -= tw // 2
    elif anchor[0] == "r":
        x -= tw
    if anchor[1] == "t":
        y += th
    elif anchor[1] == "m":
        y += th // 2
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def two_tone_line(img, p0, p1, core=3, edge=7):
    cv2.line(img, p0, p1, WHITE, edge, cv2.LINE_AA)
    cv2.line(img, p0, p1, BLACK, core, cv2.LINE_AA)


TENTHS = ["0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9", "1 in"]


# ----------------------------------------------------------------------------
# S1 ruler
# ----------------------------------------------------------------------------
def render_s1(crop, side=SIDE, margin=96):
    W = side - margin
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[margin:, margin:] = resize_square(crop, W)
    # rulers: baseline + a tick every 0.1 inch (extending 18 px into the fabric as a two-tone mark)
    cv2.line(canvas, (margin, margin - 1), (side - 1, margin - 1), BLACK, 2)
    cv2.line(canvas, (margin - 1, margin), (margin - 1, side - 1), BLACK, 2)
    for k in range(11):
        t = margin + int(round(k * W / 10)) - (1 if k == 10 else 0)
        # top ruler
        cv2.line(canvas, (t, margin - 38), (t, margin - 1), BLACK, 3)
        two_tone_line(canvas, (t, margin), (t, margin + 18), core=3, edge=7)
        # left ruler
        cv2.line(canvas, (margin - 38, t), (margin - 1, t), BLACK, 3)
        two_tone_line(canvas, (margin, t), (margin + 18, t), core=3, edge=7)
        lab = TENTHS[k]
        if k == 0:
            continue
        ax = "rb" if k == 10 else "cb"
        put_text(canvas, lab, (t - (4 if k == 10 else 0), margin - 44), 0.9, BLACK, 2, ax)
        put_text(canvas, lab, (margin - 44, t - (0 if k < 10 else 18)), 0.9, BLACK, 2, "rm")
    put_text(canvas, "inch", (6, 8), 0.8, BLACK, 2, "lt")
    return canvas


# ----------------------------------------------------------------------------
# S2 guide lines with 10 numbered 0.1-inch segments
# ----------------------------------------------------------------------------
def render_s2(crop, side=SIDE):
    canvas = resize_square(crop, side).copy()
    c = side // 2
    two_tone_line(canvas, (0, c), (side - 1, c))
    two_tone_line(canvas, (c, 0), (c, side - 1))
    for k in range(11):
        t = min(side - 1, int(round(k * side / 10)))
        two_tone_line(canvas, (t, c - 22), (t, c + 22))        # ticks on the horizontal line
        two_tone_line(canvas, (c - 22, t), (c + 22, t))        # ticks on the vertical line
    for k in range(10):
        m = int(round((k + 0.5) * side / 10))
        put_text(canvas, f"H{k + 1}", (m, c - 40), 0.8, WHITE, 2, "cb", bg=BLACK)      # above h-line
        put_text(canvas, f"V{k + 1}", (c + 40, m), 0.8, WHITE, 2, "lm", bg=BLACK)       # right of v-line
    return canvas


# ----------------------------------------------------------------------------
# S3 four magnified 0.1-inch insets
# ----------------------------------------------------------------------------
S3_CENTRES = {"A": (0.25, 0.25), "B": (0.75, 0.25), "C": (0.25, 0.75), "D": (0.75, 0.75)}  # (x, y)


def s3_patch(crop, cx, cy, size_in=0.1):
    S = crop.shape[0]
    w = S * size_in
    x0, y0 = int(round(cx * S - w / 2)), int(round(cy * S - w / 2))
    n = int(round(w))
    return crop[y0:y0 + n, x0:x0 + n]


def render_s3(crop, side=SIDE, label_h=34):
    tile = (side - 3 * 20) // 2                     # 738 px
    tile = tile - label_h // 2                      # leave room for the label band
    canvas = np.full((side, side, 3), 255, np.uint8)
    gap_x = (side - 2 * tile) // 3
    gap_y = (side - 2 * (tile + label_h)) // 3
    for i, (name, (cx, cy)) in enumerate(S3_CENTRES.items()):
        r, col = divmod(i, 2)
        x0 = gap_x + col * (tile + gap_x)
        y0 = gap_y + r * (tile + label_h + gap_y)
        put_text(canvas, f"{name}: 0.1 inch x 0.1 inch", (x0, y0 + 4), 0.9, BLACK, 2, "lt")
        p = cv2.resize(s3_patch(crop, cx, cy), (tile, tile), interpolation=cv2.INTER_CUBIC)
        canvas[y0 + label_h:y0 + label_h + tile, x0:x0 + tile] = p
        cv2.rectangle(canvas, (x0 - 1, y0 + label_h - 1), (x0 + tile, y0 + label_h + tile), BLACK, 2)
    return canvas


# ----------------------------------------------------------------------------
# S4 intensity profiles
# ----------------------------------------------------------------------------
BAND_IN = 0.05          # band height (inch) averaged across the profile direction


def band_profile(crop, axis):
    """axis='h': mean over the rows of a horizontal band through the centre -> profile along x
    (one peak per VERTICAL yarn).  axis='v': same with a vertical band -> profile along y.
    Detrended by a 0.1-inch moving average, normalised by a 0.1-inch moving std, lightly smoothed."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    S = g.shape[0]
    h = max(2, int(round(BAND_IN * S / 2)))
    c = S // 2
    prof = g[c - h:c + h, :].mean(0) if axis == "h" else g[:, c - h:c + h].mean(1)
    win = max(3, int(round(0.1 * S)) | 1)
    k = np.ones(win, np.float32) / win
    pad = np.pad(prof, win, mode="reflect")
    mu = np.convolve(pad, k, "same")[win:-win]
    sd = np.sqrt(np.maximum(np.convolve((pad - np.convolve(pad, k, "same")) ** 2, k, "same")[win:-win], 1e-6))
    z = (prof - mu) / sd
    sig = max(0.6, S / 1500.0)
    r = int(3 * sig) + 1
    gk = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    gk /= gk.sum()
    return np.convolve(np.pad(z, r, mode="reflect"), gk, "same")[r:-r]


def resample(prof, n):
    x_old = (np.arange(len(prof)) + 0.5) / len(prof)
    x_new = (np.arange(n) + 0.5) / n
    return np.interp(x_new, x_old, prof)


def _plot_profile(prof, length, depth, vertical=False, pad=34):
    """Draw a profile as an image.  Horizontal plot: (depth rows x length cols), position along x,
    brightness upwards.  Vertical plot: (length rows x depth cols), position along y (top -> bottom,
    aligned with the photo rows), brightness to the right; tick labels written horizontally."""
    p = resample(prof, length)
    lo, hi = np.percentile(p, 1), np.percentile(p, 99)
    a = np.clip((p - lo) / max(hi - lo, 1e-6), 0, 1)
    if not vertical:
        img = np.full((depth, length, 3), 255, np.uint8)
        top, bot = 10, depth - pad
        for k in range(11):                                   # 0.1-inch grid
            t = min(length - 1, int(round(k * length / 10)))
            cv2.line(img, (t, top), (t, bot), GREY, 2)
        y = (bot - a * (bot - top)).astype(np.int32)
        pts = np.stack([np.arange(length), y], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, BLACK, 2, cv2.LINE_AA)
        for k in range(1, 10):
            put_text(img, TENTHS[k], (int(round(k * length / 10)), depth - 4), 0.7, BLACK, 2, "cb")
        return img
    padv = 44
    img = np.full((length, depth, 3), 255, np.uint8)
    left, right = 10, depth - padv
    for k in range(11):
        t = min(length - 1, int(round(k * length / 10)))
        cv2.line(img, (left, t), (right, t), GREY, 2)
    x = (left + a * (right - left)).astype(np.int32)
    pts = np.stack([x, np.arange(length)], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, BLACK, 2, cv2.LINE_AA)
    for k in range(1, 10):
        put_text(img, TENTHS[k], (depth - 4, int(round(k * length / 10))), 0.7, BLACK, 2, "rm")
    return img


def render_s4(crop, side=SIDE, plot=384):
    W = side - plot
    canvas = np.full((side, side, 3), 255, np.uint8)
    canvas[:W, :W] = resize_square(crop, W)
    ph, pv = band_profile(crop, "h"), band_profile(crop, "v")
    canvas[W:, :W] = _plot_profile(ph, W, plot)
    vimg = _plot_profile(pv, W, plot, vertical=True)
    canvas[:W, W:] = vimg
    # band markers on the photo edges (short cyan/black marks at the band limits)
    b = int(round(BAND_IN * W / 2))
    c = W // 2
    for s in (-b, b):
        cv2.line(canvas, (0, c + s), (40, c + s), CYAN, 3)
        cv2.line(canvas, (W - 41, c + s), (W - 1, c + s), CYAN, 3)
        cv2.line(canvas, (c + s, 0), (c + s, 40), CYAN, 3)
        cv2.line(canvas, (c + s, W - 41), (c + s, W - 1), CYAN, 3)
    cv2.line(canvas, (0, W), (side, W), BLACK, 2)
    cv2.line(canvas, (W, 0), (W, side), BLACK, 2)
    # legend in the free corner
    x0, y0 = W + 14, W + 20
    for i, s in enumerate(["Below: brightness", "along the horizontal", "band (1 peak per", "vertical yarn).",
                           "Right: brightness", "along the vertical", "band (1 peak per", "horizontal yarn).",
                           "Grid: 0.1 inch."]):
        put_text(canvas, s, (x0, y0 + i * 38), 0.75, BLACK, 2, "lt")
    return canvas


def profile_oracle_counts(crop, prominence=None):
    """Number of peaks actually present in the two S4 curves (as plotted, i.e. resampled to the
    plot width), with a fixed minimum prominence (chosen on the 3 synthetic design images)."""
    from scipy.signal import find_peaks
    prom = PROFILE_ORACLE_PROMINENCE if prominence is None else prominence
    out = []
    for ax in ("h", "v"):
        p = resample(band_profile(crop, ax), SIDE - 384)
        out.append(int(len(find_peaks(p, prominence=prom)[0])))
    return tuple(out)          # (vertical_yarns, horizontal_yarns)


PROFILE_ORACLE_PROMINENCE = 0.5    # set on the synthetic images (see scaffold/synth/design_notes.json)


def render(crop, sid, side=SIDE):
    if sid == "S1":
        return render_s1(crop, side)
    if sid in ("S1t", "P1"):
        return resize_square(crop, side)
    if sid == "S2":
        return render_s2(crop, side)
    if sid == "S3":
        return render_s3(crop, side)
    if sid == "S4":
        return render_s4(crop, side)
    raise ValueError(sid)


# ----------------------------------------------------------------------------
# prompts (versioned; never edit in place)
# ----------------------------------------------------------------------------
HEAD = ("This is a magnified photo of woven fabric seen through a 1-inch pick glass; "
        "the photo shows exactly 1 inch x 1 inch of the fabric. ")
WEAVE = '"weave": "plain|twill|satin|other"'

PROMPTS = {
    "S1": dict(version="v1", text=(
        HEAD +
        "Rulers along the top and left edges of the photo mark every 0.1 inch (labels .1 to .9). "
        "Use them as a measuring scale:\n"
        "1. Take one 0.1-inch interval of the top ruler and count the vertical yarns that lie inside it "
        "(count a yarn cut by the interval boundary as 0.5). Do this at 3 different places across the photo.\n"
        "2. Take one 0.1-inch interval of the left ruler and count the horizontal yarns inside it, "
        "at 3 different places.\n"
        "3. Yarns per inch = 10 x the average count per 0.1-inch interval.\n"
        "Answer with only one JSON object on one line: "
        '{"vertical_per_tenth": [c1, c2, c3], "horizontal_per_tenth": [c1, c2, c3], '
        '"vertical_yarns": <vertical yarns per inch>, "horizontal_yarns": <horizontal yarns per inch>, ' + WEAVE + "}")),
    "S1t": dict(version="v1", text=(
        HEAD +
        "One tenth of the photo width (or height) is 0.1 inch. Use this as a measuring scale:\n"
        "1. Take one interval of 0.1 inch (one tenth of the photo width) and count the vertical yarns that "
        "lie inside it (count a yarn cut by the interval boundary as 0.5). Do this at 3 different places "
        "across the photo.\n"
        "2. Take one interval of 0.1 inch (one tenth of the photo height) and count the horizontal yarns "
        "inside it, at 3 different places.\n"
        "3. Yarns per inch = 10 x the average count per 0.1-inch interval.\n"
        "Answer with only one JSON object on one line: "
        '{"vertical_per_tenth": [c1, c2, c3], "horizontal_per_tenth": [c1, c2, c3], '
        '"vertical_yarns": <vertical yarns per inch>, "horizontal_yarns": <horizontal yarns per inch>, ' + WEAVE + "}")),
    "S2": dict(version="v1", text=(
        HEAD +
        "A horizontal and a vertical guide line (black with white edges) cross the photo. Tick marks divide "
        "each guide line into 10 numbered segments of 0.1 inch: H1 to H10 along the horizontal line (left to "
        "right) and V1 to V10 along the vertical line (top to bottom).\n"
        "1. Along the horizontal guide line, count the vertical yarns that cross it in each segment H1 ... H10.\n"
        "2. Along the vertical guide line, count the horizontal yarns that cross it in each segment V1 ... V10.\n"
        "3. The sum over the 10 segments is the number of yarns per inch.\n"
        "Answer with only one JSON object on one line: "
        '{"vertical_per_segment": [10 counts], "horizontal_per_segment": [10 counts], '
        '"vertical_yarns": <sum>, "horizontal_yarns": <sum>, ' + WEAVE + "}")),
    "S3": dict(version="v1", text=(
        "This image shows four strongly magnified patches (A, B, C, D) of one woven fabric, taken from four "
        "different places of a photo through a 1-inch pick glass. Each patch shows exactly 0.1 inch x 0.1 inch "
        "of fabric (one tenth of an inch on each side).\n"
        "1. In each patch, count the vertical yarns crossing a horizontal line across the patch and the "
        "horizontal yarns crossing a vertical line across the patch (a yarn cut by the patch border counts as 0.5).\n"
        "2. Yarns per inch = 10 x the average count per patch.\n"
        "Answer with only one JSON object on one line: "
        '{"vertical_per_patch": [A, B, C, D], "horizontal_per_patch": [A, B, C, D], '
        '"vertical_yarns": <vertical yarns per inch>, "horizontal_yarns": <horizontal yarns per inch>, ' + WEAVE + "}")),
    "S4": dict(version="v1", text=(
        "Top left: a magnified photo of woven fabric seen through a 1-inch pick glass; the photo shows exactly "
        "1 inch x 1 inch of the fabric. Short cyan marks at the photo edges delimit a narrow horizontal band and a "
        "narrow vertical band through the centre.\n"
        "Below the photo: the brightness profile along the horizontal band; every vertical yarn crossing the band "
        "makes one peak. Right of the photo: the brightness profile along the vertical band (top to bottom); every "
        "horizontal yarn makes one peak. Grid lines in both plots mark every 0.1 inch.\n"
        "1. In the lower plot, count the peaks in each of the 10 grid cells; their sum is the number of vertical "
        "yarns per inch.\n"
        "2. In the right plot, count the peaks in each of the 10 grid cells; their sum is the number of horizontal "
        "yarns per inch.\n"
        "Use the photo to check that the peak spacing matches the yarn spacing.\n"
        "Answer with only one JSON object on one line: "
        '{"vertical_per_cell": [10 counts], "horizontal_per_cell": [10 counts], '
        '"vertical_yarns": <sum>, "horizontal_yarns": <sum>, ' + WEAVE + "}")),
}
LIST_KEYS = {"S1": ("vertical_per_tenth", "horizontal_per_tenth", "mean10"),
             "S1t": ("vertical_per_tenth", "horizontal_per_tenth", "mean10"),
             "S2": ("vertical_per_segment", "horizontal_per_segment", "sum"),
             "S3": ("vertical_per_patch", "horizontal_per_patch", "mean10"),
             "S4": ("vertical_per_cell", "horizontal_per_cell", "sum")}


def get_prompt(sid):
    if sid == "P1":                       # reference: the unchanged P1 prompt (prompts.py) on the plain crop
        import prompts as PR
        p = PR.get("P1")
        p["set_version"] = p["set_version"] + "|" + SCAFFOLD_SET_VERSION
        return p
    p = dict(PROMPTS[sid])
    p["id"] = sid
    p["set_version"] = SCAFFOLD_SET_VERSION
    return p


def cell_estimate(sid, text):
    """Harness-side re-computation from the per-cell lists (secondary diagnostic):
    S1/S1t/S3 -> 10 x mean, S2/S4 -> sum.  Returns (v, h) or (None, None)."""
    import re
    if sid not in LIST_KEYS:
        return None, None
    kv, kh, rule = LIST_KEYS[sid]

    def grab(k):
        m = re.search(r'"?' + k + r'"?\s*:\s*\[([^\]]*)\]', text or "")
        if not m:
            return None
        xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if not xs:
            return None
        return 10 * float(np.mean(xs)) if rule == "mean10" else float(np.sum(xs))
    return grab(kv), grab(kh)


def code_md5():
    return hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()


# ----------------------------------------------------------------------------
# synthetic design images (exact labels from cnn/synth.py v2)
# ----------------------------------------------------------------------------
SYNTH_SPECS = {
    # name: (weave, epi=vertical yarns/in, ppi=horizontal yarns/in, extra params)
    "syn1": ("plain", 52.0, 46.0, dict(rotation_deg=1.5, colour_style="solid", shadow=0.0, glare=0.0, seed=101)),
    "syn2": ("twill21", 82.0, 64.0, dict(rotation_deg=-2.0, colour_style="two_tone", shadow=0.4, glare=0.0, seed=202)),
    "syn3": ("plain", 108.0, 90.0, dict(rotation_deg=0.5, colour_style="melange", shadow=0.0, glare=0.2, seed=303)),
}


def make_synth():
    sys.path.insert(0, os.path.join(ROOT, "cnn"))
    os.environ.setdefault("SYNTH_PROFILE", "v2")
    import synth as SY
    os.makedirs(SYNTH_DIR, exist_ok=True)
    labels = {}
    for name, (weave, epi, ppi, ex) in SYNTH_SPECS.items():
        p = SY.SynthParams(weave=weave, epi=epi, ppi=ppi, r_eff=1100.0, r_sensor=1760.0, blur_sigma_px=0.8,
                           noise=0.012, jpeg_q=90, exposure=1.0, persp=0.03, seg_jitter=0.004,
                           skew_deg=0.5, waviness=0.25, density_drift=0.015, **ex)
        _, crop, lab = SY.generate(ex["seed"], params=p, max_crop_side=4000)
        cv2.imwrite(os.path.join(SYNTH_DIR, f"{name}.png"), crop)
        labels[name] = dict(warp=epi, weft=ppi, weave=SY.COARSE[weave], weave_fine=weave, side=int(crop.shape[0]),
                            rotation_deg=ex["rotation_deg"], colour_style=ex["colour_style"])
        print(name, crop.shape, labels[name])
    json.dump(labels, open(os.path.join(SYNTH_DIR, "labels.json"), "w"), indent=1)
    return labels


def load_crop(set_name, image_id):
    if set_name == "synth":
        return cv2.imread(os.path.join(SYNTH_DIR, f"{image_id}.png"))
    if set_name == "legacy":
        return cv2.imread(os.path.join(LEGACY_CROP_DIR, f"{image_id}.png"))
    if set_name == "rect150":       # the annotation-tool crops (= P1 Stage-B inputs); added after the freeze
        return cv2.imread(os.path.join(ROOT, "annotation_tool", "data", "crops", f"{image_id}.jpg"))
    raise ValueError(set_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--render", nargs=2, metavar=("IMAGE", "SCAFFOLD"))
    ap.add_argument("--set", default="legacy")
    ap.add_argument("--prompts", action="store_true")
    a = ap.parse_args()
    if a.synth:
        make_synth()
    if a.render:
        img, sid = a.render
        out = render(load_crop(a.set, img), sid)
        d = os.path.join(SC_DIR, "preview")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{img}__{sid}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if a.prompts:
        for k in SCAFFOLDS:
            print(f"[{k} {PROMPTS[k]['version']}]\n{PROMPTS[k]['text']}\n")


# ----------------------------------------------------------------------------
# Molmo2 (GPU) inputs: 644 px PNG (its native maximum for a square image)
# ----------------------------------------------------------------------------
MOLMO_SIDE = 644
MOLMO_DIR = os.path.join(SC_DIR, "molmo_inputs")


def molmo_input(crop, sid):
    """P1/S1t: the crop resized straight to 644 px (identical to the existing Molmo2 P1 input);
    composites: rendered at 1536 px like the API inputs, then INTER_AREA down to 644 px."""
    if sid in ("P1", "S1t"):
        return resize_square(crop, MOLMO_SIDE)
    return cv2.resize(render(crop, sid, SIDE), (MOLMO_SIDE, MOLMO_SIDE), interpolation=cv2.INTER_AREA)


def export_molmo(set_name, names, sids):
    d = os.path.join(MOLMO_DIR, set_name)
    os.makedirs(d, exist_ok=True)
    for n in names:
        crop = load_crop(set_name, n)
        for sid in sids:
            cv2.imwrite(os.path.join(d, f"{n}__{sid}.png"), molmo_input(crop, sid))
    pj = dict(set_version=SCAFFOLD_SET_VERSION, scaffolds_py_md5=code_md5(),
              prompts={sid: {k: v for k, v in get_prompt(sid).items() if k in ("version", "text", "id")}
                       for sid in ["P1"] + SCAFFOLDS})
    json.dump(pj, open(os.path.join(MOLMO_DIR, "prompts.json"), "w"), indent=1)
    return d
