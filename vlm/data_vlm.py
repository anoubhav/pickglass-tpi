"""Data preparation for the VLM thread-counting baseline (WVAQ 2027 fabric project).

What this module does (all source data are READ-ONLY; outputs go to ./crops):

1. 150-image set  -> rectified 1-inch window crops using the shared contract in
   ../common/rectify.py (perspective warp from the hand-annotated XML polygon,
   S = mean polygon side length, bilinear).  Saved as crops/rect150/{i}.png.
2. 20 legacy images (FR-final attempt/images/{a1..a10,r1..r10}.jpg).  These were
   hand-cropped in the earlier project around the pick-glass opening, so no
   polygon exists.  We trim the dark pick-glass frame that is visible on some
   sides with a simple border-profile detector (`detect_legacy_window`) and
   resize the remainder to a square (the opening is square; residual perspective
   is ignored).  Saved as crops/legacy/{name}.png together with an overlay
   contact sheet for checking.
3. Ground truth for the legacy set parsed from "Warp-Weft count.xlsx".
4. Helper views used by the prompts: quarter-inch patches (P3) and a guide-line
   overlay (P4).

Axis convention (shared with the CNN code):
    vertical_yarns   = number of yarns crossed walking HORIZONTALLY across the
                       window  (= yarns that run top-to-bottom in the image)
    horizontal_yarns = number of yarns crossed walking VERTICALLY.
Legacy GT is warp/weft; photo orientation is unknown, so evaluation reports an
"axis-matched" mapping (warp -> vertical_yarns, weft -> horizontal_yarns) and a
"best-assignment" mapping (the better of the two permutations, per image).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # <repo root>
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as shared_rectify                  # noqa: E402  (shared contract)

LEGACY_DIR = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "FR-final attempt", "images")
LEGACY_XLSX = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "Warp-Weft count.xlsx")
IMG150_DIR = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "photos")), "final_image_set")
LEGACY_NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]

CROPS = os.path.join(HERE, "crops")
RECT150_DIR = os.path.join(CROPS, "rect150")
LEGACY_CROP_DIR = os.path.join(CROPS, "legacy")


# ----------------------------------------------------------------------------
# ground truth
# ----------------------------------------------------------------------------
@dataclass
class LegacyGT:
    name: str
    weft: float               # picks per inch (midpoint if a range)
    warp: float               # ends per inch
    weave: Optional[str]      # "plain" (1/1) | "twill" (2/1) | None
    unclear: bool             # '*' after the name: "image not clear enough"
    is_range: bool            # a count was given as a range, e.g. "110-120"
    weft_raw: str
    warp_raw: str


def _parse_count(v):
    """'26 + 25*2 = 76' -> 76 ; '110-120' -> 115 (flag) ; 64.0 -> 64."""
    if v is None:
        return None, False
    if isinstance(v, (int, float)):
        return float(v), False
    s = str(v).strip()
    if "=" in s:
        return float(s.split("=")[-1].strip()), False
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0, True
    return float(s), False


def _parse_weave(v):
    """Excel mangled '1/1' -> date YYYY-01-01 and '2/1' -> YYYY-02-01 (month, day read back)."""
    if v is None:
        return None
    a, b = (v.month, v.day) if hasattr(v, "month") else map(int, re.findall(r"\d+", str(v))[:2])
    return {(1, 1): "plain", (2, 1): "twill", (1, 2): "twill"}.get((a, b), f"{a}/{b}")


def load_legacy_gt(xlsx: str = LEGACY_XLSX) -> list[LegacyGT]:
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    out = []
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or row[0] is None or not re.fullmatch(r"[ar]\d+\*?", str(row[0]).strip()):
            continue
        raw = str(row[0]).strip()
        weft, r1 = _parse_count(row[1])
        warp, r2 = _parse_count(row[2])
        if weft is None or warp is None:
            continue
        out.append(LegacyGT(name=raw.rstrip("*"), weft=weft, warp=warp, weave=_parse_weave(row[3]),
                            unclear=raw.endswith("*"), is_range=r1 or r2,
                            weft_raw=str(row[1]), warp_raw=str(row[2])))
    return out


def legacy_path(name: str) -> str:
    p = os.path.join(LEGACY_DIR, f"{name}.jpg")
    if not os.path.exists(p) and os.path.exists(os.path.join(LEGACY_DIR, f"{name} .jpg")):
        p = os.path.join(LEGACY_DIR, f"{name} .jpg")      # 'r6 .jpg' has a space
    return p


# ----------------------------------------------------------------------------
# legacy window detection (trim dark frame)
# ----------------------------------------------------------------------------
def detect_legacy_window(img: np.ndarray, band: float = 0.08, dark_frac: float = 0.5,
                         dark_level: float = 40.0, safety: float = 0.004):
    """Return (x0, y0, x1, y1) of the fabric window inside a hand-cropped legacy image.

    The pick-glass frame is black metal with thin bright tick marks.  For every side
    we scan rows/columns inward from the image edge (at most `band` of the size) and
    keep cutting while the fraction of "dark" pixels in that row/column exceeds
    `dark_frac`.  Dark = blurred gray < `dark_level` (absolute; the black frame sits at
    ~12-25 while even navy/purple yarns are mostly above 40).  The run must start at
    the edge, so dark yarn stripes inside the fabric are not cut.  Known limitation:
    where glare lifts the frame above `dark_level` (r8 bottom) the frame is kept.
    `safety` adds a small extra trim to remove the frame's anti-aliased edge.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    h, w = g.shape
    dark = g < dark_level
    # use the central 70% of the orthogonal extent (corners may contain frame from both sides)
    rows = dark[:, int(.15 * w): int(.85 * w)].mean(1)
    cols = dark[int(.15 * h): int(.85 * h), :].mean(0)

    def run(profile):
        n = int(band * len(profile))
        k, gap = 0, 0
        for t in range(n):
            if profile[t] > dark_frac:
                k, gap = t + 1, 0
            else:
                gap += 1
                if gap > 6:                     # allow tick-mark gaps of a few pixels
                    break
        return k

    top, bottom = run(rows), run(rows[::-1])
    left, right = run(cols), run(cols[::-1])
    sy, sx = int(safety * h), int(safety * w)
    x0 = left + (sx if left else 0)
    y0 = top + (sy if top else 0)
    x1 = w - right - (sx if right else 0)
    y1 = h - bottom - (sy if bottom else 0)
    return x0, y0, x1, y1


def legacy_window_crop(name: str, out_size: Optional[int] = None):
    img = cv2.imread(legacy_path(name), cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = detect_legacy_window(img)
    crop = img[y0:y1, x0:x1]
    S = out_size or int(round((crop.shape[0] + crop.shape[1]) / 2))
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, x1, y1), img.shape[:2]


# ----------------------------------------------------------------------------
# views used by prompts
# ----------------------------------------------------------------------------
def quarter_patch(img: np.ndarray, which: str):
    """Quarter-inch square patch of a rectified 1-inch crop.

    which = 'c'  -> the central patch  [3/8, 5/8] x [3/8, 5/8]
          = 'd0'..'d3' -> k-th patch on the main diagonal of a 4x4 grid.  The four
            diagonal patches cover all four x-quarters and all four y-quarters once,
            so summing their counts gives a full-inch count without extrapolation.
    """
    S = img.shape[0]
    q = S / 4.0
    if which == "c":
        a = int(round(1.5 * q))
        return img[a:a + int(round(q)), a:a + int(round(q))]
    k = int(which[1])
    a, b = int(round(k * q)), int(round((k + 1) * q))
    return img[a:b, a:b]


def guide_line_view(img: np.ndarray, axis: str, color=(0, 0, 255), frac: float = 0.004):
    """Draw a thin straight guide line through the centre.

    axis='h' -> horizontal line (vertical yarns cross it); axis='v' -> vertical line.
    Color is BGR (default red).
    """
    out = img.copy()
    S = out.shape[0]
    t = max(2, int(round(frac * S)))
    c = S // 2
    if axis == "h":
        cv2.line(out, (0, c), (S - 1, c), color, t)
    else:
        cv2.line(out, (c, 0), (c, S - 1), color, t)
    return out


def resize_square(img: np.ndarray, side: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > side else cv2.INTER_CUBIC
    return cv2.resize(img, (side, side), interpolation=interp)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# build all crops
# ----------------------------------------------------------------------------
def build_all(verbose: bool = True):
    os.makedirs(RECT150_DIR, exist_ok=True)
    os.makedirs(LEGACY_CROP_DIR, exist_ok=True)
    rows = []
    for i in range(shared_rectify.N_IMAGES):
        p = os.path.join(RECT150_DIR, f"{i}.png")
        if not os.path.exists(p):
            crop = shared_rectify.rectify(i)
            cv2.imwrite(p, crop)
        rows.append((i, cv2.imread(p).shape[0]))
    if verbose:
        s = np.array([r[1] for r in rows])
        print(f"rect150: {len(rows)} crops, side px min/median/max = {s.min()}/{int(np.median(s))}/{s.max()}")

    # legacy
    thumbs, info = [], []
    for name in LEGACY_NAMES:
        crop, box, (h, w) = legacy_window_crop(name)
        cv2.imwrite(os.path.join(LEGACY_CROP_DIR, f"{name}.png"), crop)
        img = cv2.imread(legacy_path(name))
        vis = img.copy()
        cv2.rectangle(vis, box[:2], box[2:], (0, 255, 0), max(3, w // 200))
        vis = cv2.resize(vis, (360, int(360 * h / w)))
        cv2.putText(vis, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        thumbs.append(cv2.resize(vis, (360, 360)))
        trims = (box[0], box[1], w - box[2], h - box[3])
        info.append((name, (w, h), box, trims, crop.shape[0]))
        if verbose:
            print(f"legacy {name}: {w}x{h} -> window {box} trims(l,t,r,b)={trims} side={crop.shape[0]}")
    grid = np.vstack([np.hstack(thumbs[r * 5:(r + 1) * 5]) for r in range(4)])
    cv2.imwrite(os.path.join(CROPS, "legacy_window_check.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return info


if __name__ == "__main__":
    for g in load_legacy_gt():
        print(asdict(g))
    build_all()
