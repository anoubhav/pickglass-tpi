"""Shared rectification of the 1-inch pick-glass window (WVAQ 2027 fabric project).

CONTRACT (shared by the annotation tool and the counting methods -- do not change
behaviour without updating every consumer):

    load_corners(i)            -> float64 array (4, 2): TL, TR, BR, BL corners of the
                                  pick-glass window in ORIGINAL image pixels (x, y).
    rectify(i, out_size=None)  -> uint8 BGR array (S, S, 3): perspective-warped square
                                  crop of the window. Default S = round(mean side length
                                  of the annotated polygon in original pixels), i.e. the
                                  crop is roughly at native resolution (no upsampling
                                  beyond native).

Extra helpers (optional to use):
    homography(i, out_size=None) -> (H, S): 3x3 matrix mapping original px -> rectified px.
    default_out_size(corners)    -> int S as defined above.
    order_corners(pts)           -> canonical TL, TR, BR, BL ordering of any 4 points.
    load_image(i)                -> original BGR image.

Notes
-----
* Annotations are Labelbox VOC-like XML files
  (<object><polygon><x1>..<y4></polygon>). 149/150 are stored TL,TR,BR,BL; image 22 is
  stored TR,TL,BL,BR, so the corners are always re-ordered canonically here
  (clockwise on screen, starting from the corner with the smallest x+y).
* All 150 JPEGs have EXIF orientation 1, so raw pixel coords == displayed coords.
* Rectified pixel (u, v) in [0, S) maps to the window: u = 0 at the left edge, u = S at
  the right edge, so 1 inch == S px in both directions. A count along a full guide line
  is therefore directly threads per inch.
* Interpolation is bilinear (cv2.INTER_LINEAR).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# Data locations (common/paths.py): $PICKGLASS_NOTEBOOKS holds final_image_set/ and final_bbox_annotation_set/.
_NB = os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "photos")
IMAGE_DIR = os.path.join(_NB, "final_image_set")
ANN_DIR = os.path.join(_NB, "final_bbox_annotation_set")
# Release: without the XML files, load_corners() reads the same corners from data/window_polygons.csv.
POLYGONS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "window_polygons.csv")
N_IMAGES = 150


def image_path(i) -> str:
    return os.path.join(IMAGE_DIR, f"{int(i)}.jpeg")


def annotation_path(i) -> str:
    return os.path.join(ANN_DIR, f"{int(i)}.xml")


def order_corners(pts) -> np.ndarray:
    """Return the 4 points ordered TL, TR, BR, BL (clockwise on screen, y down).

    Points are sorted by angle around their centroid (increasing atan2 with y pointing
    down == clockwise on screen) and the sequence is rotated so it starts at the corner
    with the smallest x + y (top-left).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _corners_from_csv(i) -> np.ndarray:
    """Release fallback: the XML polygon of image i as exported verbatim to data/window_polygons.csv."""
    import csv
    with open(POLYGONS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["id"]) == int(i):
                return order_corners([[float(r[f"x{k}"]), float(r[f"y{k}"])] for k in range(1, 5)])
    raise ValueError(f"no polygon for image {i} in {POLYGONS_CSV}")


def load_corners(i) -> np.ndarray:
    """Window polygon of image i as a (4, 2) float array, ordered TL, TR, BR, BL."""
    if not os.path.exists(annotation_path(i)) and os.path.exists(POLYGONS_CSV):
        return _corners_from_csv(i)
    root = ET.parse(annotation_path(i)).getroot()
    poly = root.find("object/polygon")
    if poly is None:
        raise ValueError(f"no polygon in {annotation_path(i)}")
    pts = [[float(poly.find(f"x{k}").text), float(poly.find(f"y{k}").text)] for k in range(1, 5)]
    return order_corners(pts)


def default_out_size(corners) -> int:
    """round(mean of the 4 polygon side lengths), in original pixels."""
    c = np.asarray(corners, dtype=np.float64)
    sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    return int(round(float(sides.mean())))


def homography(i, out_size=None, corners=None):
    """3x3 perspective matrix (original px -> rectified px) and the output size S."""
    if corners is None:
        corners = load_corners(i)
    corners = np.asarray(corners, dtype=np.float32)
    S = int(out_size) if out_size is not None else default_out_size(corners)
    dst = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)
    return H, S


def load_image(i) -> np.ndarray:
    img = cv2.imread(image_path(i), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path(i))
    return img


def rectify(i, out_size=None, img=None) -> np.ndarray:
    """Square BGR crop (S x S) of the 1-inch window of image i via perspective warp."""
    if img is None:
        img = load_image(i)
    H, S = homography(i, out_size)
    return cv2.warpPerspective(img, H, (S, S), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


if __name__ == "__main__":  # quick self-test: python rectify.py [ids...]
    import sys

    ids = [int(a) for a in sys.argv[1:]] or [0, 22]
    for i in ids:
        c = load_corners(i)
        crop = rectify(i)
        print(i, load_image(i).shape[:2], "corners", c.round(1).tolist(), "->", crop.shape)
