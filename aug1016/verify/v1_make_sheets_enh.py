"""Verification check 1b: enhanced blind sheets for the samples that are ambiguous at thumbnail
size.  Per sample, two rows:
   row A: low-pass, per-image standardised grey (illumination / glare / vignette layout;
          insensitive to brightness-contrast jitter and noise)
   row B: sharp centre zoom (central 30 %; twill direction / weave chirality)
columns: AUG | src rot0 | rot90 | rot180 | rot270  (np.rot90, CCW)."""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
TOOL = os.path.join(ROOT, "annotation_tool", "data", "crops")
CROPS = os.path.join(AUG, "crops")
OUT = os.path.join(HERE, "sheets")
T = 190

ids = sys.argv[1:]


def lowpass(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 2.5)
    m = 8
    inner = g[m:-m, m:-m]
    g = (g - inner.mean()) / (inner.std() + 1e-6)
    g = np.clip(128 + 50 * g, 0, 255).astype(np.uint8)
    g = cv2.resize(g, (T, T), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def zoom(im):
    S = im.shape[0]
    a = int(S * 0.35)
    b = int(S * 0.65)
    c = im[a:b, a:b]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (4, 4)).apply(g)
    return cv2.cvtColor(cv2.resize(g, (T, T), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return im


def pad(c):
    return np.pad(c, ((3, 3), (3, 3), (0, 0)), constant_values=255)


for s in range(0, len(ids), 5):
    rows = []
    for aid in ids[s:s + 5]:
        src = cv2.imread(os.path.join(TOOL, f"{aid.split('_aug')[0]}.jpg"))
        aug = cv2.imread(os.path.join(CROPS, f"{aid}.jpg"))
        cands = [np.ascontiguousarray(np.rot90(src, j)) for j in range(4)]
        ra = [label(lowpass(aug), f"{aid} AUG lp")] + [label(lowpass(c), f"rot{90 * j}") for j, c in enumerate(cands)]
        rb = [label(zoom(aug), "AUG zoom")] + [label(zoom(c), f"rot{90 * j}") for j, c in enumerate(cands)]
        rows.append(np.hstack([pad(c) for c in ra]))
        rows.append(np.hstack([pad(c) for c in rb]))
        rows.append(np.full((8, rows[-1].shape[1], 3), 0, np.uint8))
    cv2.imwrite(os.path.join(OUT, f"enh{s // 5}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("ok", len(ids))
