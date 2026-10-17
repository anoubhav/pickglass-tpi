"""Data utilities for the pick-glass thread-density project.

* read the Labelbox/VOC-like polygon XMLs (4 window corners per image)
* order the corners robustly (TL, TR, BR, BL) and rectify the 1-inch window
  with a perspective warp to an S x S square (S = native side length, so the
  rectified crop keeps the photo's real pixels-per-inch)
* parse the legacy hand counts (Warp-Weft count.xlsx)
* locate the inner 1-inch window of the 20 legacy (already roughly cropped)
  images

All source data are treated READ-ONLY.  Nothing in this module writes outside
the caller-supplied output paths.

Axis convention used throughout the CNN code
--------------------------------------------
h_count : number of threads crossed when walking HORIZONTALLY across the
          1-inch window  (= number of vertical yarns per inch)
v_count : number of threads crossed when walking VERTICALLY  (= number of
          horizontal yarns per inch)
In the legacy spreadsheet we do not know the photo orientation for certain,
so evaluation reports both an "axis-matched" mapping (warp -> vertical yarns
-> h_count, weft -> v_count) and a "best-assignment" mapping (min over the
two permutations).
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

NOTEBOOKS = (os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "photos"))
IMG_DIR = os.path.join(NOTEBOOKS, "final_image_set")
XML_DIR = os.path.join(NOTEBOOKS, "final_bbox_annotation_set")
LEGACY_DIR = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "FR-final attempt", "images")
LEGACY_XLSX = os.path.join((os.environ.get("PICKGLASS_LEGACY_ROOT") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data", "legacy2019")), "Warp-Weft count.xlsx")


# --------------------------------------------------------------------------
# corners / rectification
# --------------------------------------------------------------------------
def read_corners(xml_path: str) -> np.ndarray:
    """Return the 4 polygon corners (float32, shape 4x2) in original pixels."""
    root = ET.parse(xml_path).getroot()
    poly = root.find("object").find("polygon")
    pts = []
    for k in range(1, 5):
        pts.append([float(poly.find(f"x{k}").text), float(poly.find(f"y{k}").text)])
    return np.array(pts, dtype=np.float32)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (robust to the annotation order)."""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # counter-clockwise in image coords = clockwise visually
    # start from the point with the smallest x+y (top-left)
    s = pts.sum(1)
    i0 = int(np.argmin(s))
    pts = np.roll(pts, -i0, axis=0)
    # ensure order TL -> TR (TR has larger x than BL)
    if pts[1, 0] < pts[3, 0]:
        pts = pts[[0, 3, 2, 1]]
    return pts


def rectify(img: np.ndarray, corners: np.ndarray, side: Optional[int] = None,
            margin: float = 0.0) -> np.ndarray:
    """Perspective-warp the quadrilateral to a square of `side` pixels.

    side=None -> mean edge length of the quadrilateral (keeps native px/inch).
    margin    -> fraction of the side to shave off on each edge (avoids the
                 frame edge creeping in when the polygon is slightly loose).
    """
    q = order_corners(corners)
    if side is None:
        e = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
        side = int(round(float(np.mean(e))))
    m = margin * side
    dst = np.array([[-m, -m], [side - 1 + m, -m], [side - 1 + m, side - 1 + m],
                    [-m, side - 1 + m]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, H, (side, side), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REFLECT)


def load_rectified(idx: int, margin: float = 0.0, max_side: Optional[int] = None):
    """Rectified 1-inch crop of final_image_set/{idx}.jpeg (BGR uint8)."""
    img = cv2.imread(os.path.join(IMG_DIR, f"{idx}.jpeg"))
    if img is None:
        raise FileNotFoundError(idx)
    q = read_corners(os.path.join(XML_DIR, f"{idx}.xml"))
    crop = rectify(img, q, margin=margin)
    if max_side is not None and crop.shape[0] > max_side:
        crop = cv2.resize(crop, (max_side, max_side), interpolation=cv2.INTER_AREA)
    return crop


# --------------------------------------------------------------------------
# legacy GT spreadsheet
# --------------------------------------------------------------------------
@dataclass
class LegacyGT:
    name: str              # e.g. "a1"
    file: str              # full path of the image
    weft: float            # picks / inch (midpoint if a range)
    warp: float            # ends / inch
    weave: Optional[str]   # "plain" (1/1) | "twill21" (2/1) | None
    unclear: bool          # '*' in the sheet: image not clear enough
    is_range: bool         # one of the counts was a range like "110-120"
    raw: tuple


def _parse_count(v):
    """Return (value, is_range) from a messy spreadsheet cell."""
    if v is None:
        return None, False
    if isinstance(v, (int, float)):
        return float(v), False
    s = str(v).strip()
    if "=" in s:                      # "26 + 25*2 = 76" -> 76
        return float(s.split("=")[-1].strip()), False
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", s)
    if m:                             # "110-120" -> 115 (flagged)
        return (float(m.group(1)) + float(m.group(2))) / 2.0, True
    return float(s), False


def _parse_weave(v):
    """Excel turned '1/1' and '2/1' into dates (month 1 day 1, month 2 day 1): read month / day back."""
    if v is None:
        return None
    if hasattr(v, "month"):
        a, b = v.month, v.day
    else:
        m = re.match(r"(\d+)\s*/\s*(\d+)", str(v))
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
    if (a, b) == (1, 1):
        return "plain"
    if (a, b) in ((2, 1), (1, 2)):
        return "twill21"
    return f"{a}/{b}"


def legacy_file(name: str) -> str:
    p = os.path.join(LEGACY_DIR, f"{name}.jpg")
    if not os.path.exists(p):
        p2 = os.path.join(LEGACY_DIR, f"{name} .jpg")  # 'r6 .jpg' has a space
        if os.path.exists(p2):
            return p2
    return p


def load_legacy_gt(xlsx: str = LEGACY_XLSX):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb.worksheets[0]
    out = []
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or row[0] is None:
            continue
        name_raw = str(row[0]).strip()
        if not re.fullmatch(r"[ar]\d+\*?", name_raw):
            continue
        weft, r1 = _parse_count(row[1])
        warp, r2 = _parse_count(row[2])
        if weft is None or warp is None:
            continue
        name = name_raw.rstrip("*")
        out.append(LegacyGT(name=name, file=legacy_file(name), weft=weft, warp=warp,
                            weave=_parse_weave(row[3]), unclear=name_raw.endswith("*"),
                            is_range=r1 or r2, raw=tuple(row)))
    return out


# --------------------------------------------------------------------------
# legacy images: find the inner window (they are hand-cropped around the
# pick-glass opening with a thin dark frame visible on some sides)
# --------------------------------------------------------------------------
def legacy_inner_window(img: np.ndarray, max_trim: float = 0.08) -> np.ndarray:
    """Trim the dark pick-glass frame from the borders of a legacy crop.

    For each side we look at the mean brightness of successive rows/columns
    (within `max_trim` of the edge) and cut where it rises above the midpoint
    between frame-dark and fabric level.  Returns the trimmed crop.
    Conservative: never trims more than max_trim per side.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    core = g[int(.25 * h):int(.75 * h), int(.25 * w):int(.75 * w)]
    fab = float(np.median(core))
    cut = {}
    for side in ("top", "bottom", "left", "right"):
        n = int(max_trim * (h if side in ("top", "bottom") else w))
        if side == "top":
            prof = np.median(g[:n, int(.15 * w):int(.85 * w)], axis=1)
        elif side == "bottom":
            prof = np.median(g[::-1][:n, int(.15 * w):int(.85 * w)], axis=1)
        elif side == "left":
            prof = np.median(g[int(.15 * h):int(.85 * h), :n], axis=0)
        else:
            prof = np.median(g[int(.15 * h):int(.85 * h), ::-1][:, :n], axis=0)
        dark = float(np.min(prof[: max(3, n // 4)]))
        if fab - dark < 25:           # no frame visible on this side
            cut[side] = 0
            continue
        thr = dark + 0.6 * (fab - dark)
        above = np.where(prof > thr)[0]
        k = int(above[0]) if len(above) else 0
        # only accept if it stays above afterwards (avoid dark yarn dips)
        cut[side] = k
    y0, y1 = cut["top"], h - cut["bottom"]
    x0, x1 = cut["left"], w - cut["right"]
    return img[y0:y1, x0:x1]


if __name__ == "__main__":
    for g in load_legacy_gt():
        print(g.name, g.weft, g.warp, g.weave, "unclear" if g.unclear else "",
              "range" if g.is_range else "", os.path.exists(g.file))
