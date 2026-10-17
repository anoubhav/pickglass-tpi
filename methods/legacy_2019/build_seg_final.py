"""Generates seg_final.py by copying notebook cell 26 source byte-for-byte (no edits),
adding only an import header, a crop/warp wrapper and a CLI."""
import json, os
NB = os.path.join((os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos")), "2_Segmentation_Result_classical_CV.ipynb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_final.py")
nb = json.load(open(NB))
cell = nb["cells"][26]
assert cell["cell_type"] == "code" and cell["execution_count"] == 269
src = "".join(cell["source"])
assert src.startswith("# Preprocessing function") and "def final_algorithm(" in src

HEADER = '''"""
seg_final.py -- standalone extraction of the FINAL fabric-segmentation algorithm
behind the reported mean IoU 0.88 on the 150-image set.

Provenance
----------
Source notebook : $PICKGLASS_NOTEBOOKS/2_Segmentation_Result_classical_CV.ipynb
Source cell     : cells[26] ("# Functions" > "## Image Segmentation related"), execution_count 269
0.88 result     : cells[39] (execution_count 275) ran
                  evaluate_algorithm(dataset[:150], annotations[:150], final_algorithm)
                  -> 'Mean IoU': 0.8843114933906611, 'Fallback Count': 14 (cells[40] raw summary).
                  No cell with execution_count in 270..275 redefines any of these functions.

Everything between the BEGIN/END VERBATIM markers is cells[26] source copied
byte-for-byte by build_seg_final.py (no algorithmic changes). Only the import
header above it and the segment()/warp/CLI code below it were added.

Entry point
-----------
    segment(image_path) -> (crop_bgr, box_points, used_fallback)
      crop_bgr      : fabric region perspective-warped to an upright rectangle,
                      cut from the ORIGINAL full-resolution image (added helper,
                      not part of the notebook code, which only returns a box)
      box_points    : (4,2) int array, box corners in original-image pixel coords
                      (exactly the `bounding_box` returned by final_algorithm)
      used_fallback : True when extract_contour returned status "fallback"
                      (centred 0.5W x 0.5H box)
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

# ======================= BEGIN VERBATIM: notebook cells[26] =======================
'''

FOOTER = '''
# ======================== END VERBATIM: notebook cells[26] ========================


# ---------------------------------------------------------------------------
# Added for standalone use (NOT part of the notebook code).
# ---------------------------------------------------------------------------
def _order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def warp_box(image_bgr, box_points):
    """Perspective-warp the quadrilateral `box_points` of `image_bgr` to an upright
    rectangle (the report's 'perspective-warp the fabric region' step; no implementation
    of this step was found in the original code or the notebook, so this is a minimal helper)."""
    quad = _order_quad(box_points)
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


def segment(image_path, fallback_area_factor=1/7, cv_seed=None):
    """Run the final notebook algorithm (defaults identical to the 0.88 run) and crop.

    cv_seed: optional cv2.setRNGSeed value. The notebook only called
    np.random.seed(42), which does NOT seed cv2.kmeans(KMEANS_RANDOM_CENTERS);
    default None keeps the notebook behaviour.
    Returns (crop_bgr, box_points, used_fallback).
    """
    if cv_seed is not None:
        cv2.setRNGSeed(int(cv_seed))
    _final_img, box, status = final_algorithm(
        image_path, fallback_area_factor=fallback_area_factor,
        return_null_box_if_no_contour=False)
    original = cv2.imread(image_path)
    crop = warp_box(original, box)
    return crop, np.asarray(box), status == "fallback"


def _visualise(image_path, box, used_fallback, crop, out_path, gt_box=None):
    """Side-by-side: original with predicted box (green; red if fallback) [+ GT in blue], and crop."""
    img = cv2.imread(image_path)
    vis = img.copy()
    t = max(3, int(round(max(img.shape[:2]) / 400)))
    if gt_box is not None:
        cv2.polylines(vis, [np.asarray(gt_box, dtype=np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), t)
    color = (0, 0, 255) if used_fallback else (0, 255, 0)
    cv2.polylines(vis, [np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)], True, color, t)
    H = 900
    left = cv2.resize(vis, (int(vis.shape[1] * H / vis.shape[0]), H))
    if crop is not None and crop.size:
        right = cv2.resize(crop, (max(1, int(crop.shape[1] * H / crop.shape[0])), H))
    else:
        right = np.zeros((H, H // 2, 3), np.uint8)
    canvas = np.concatenate([left, np.full((H, 10, 3), 255, np.uint8), right], axis=1)
    cv2.imwrite(out_path, canvas)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Final fabric segmentation (notebook cells[26]).")
    ap.add_argument("images", nargs="+", help="image paths")
    ap.add_argument("--out", default=None, help="directory for <stem>_crop.png and <stem>_vis.jpg")
    ap.add_argument("--fallback-area-factor", type=float, default=1/7)
    ap.add_argument("--cv-seed", type=int, default=None)
    args = ap.parse_args(argv)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    for p in args.images:
        crop, box, fb = segment(p, args.fallback_area_factor, args.cv_seed)
        print(f"{p}\\tfallback={fb}\\tbox={box.tolist()}\\tcrop={None if crop is None else crop.shape}")
        if args.out:
            stem = os.path.splitext(os.path.basename(p))[0].strip().replace(" ", "_")
            if crop is not None:
                cv2.imwrite(os.path.join(args.out, stem + "_crop.png"), crop)
            _visualise(p, box, fb, crop, os.path.join(args.out, stem + "_vis.jpg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

with open(OUT, "w") as f:
    f.write(HEADER + src + FOOTER)
# self-check: verbatim block equals cell source
txt = open(OUT).read()
b = txt.index("# ======================= BEGIN VERBATIM: notebook cells[26] =======================\n") + len("# ======================= BEGIN VERBATIM: notebook cells[26] =======================\n")
e = txt.index("\n# ======================== END VERBATIM")
assert txt[b:e] == src, "verbatim mismatch"
print("wrote", OUT, "verbatim block bytes:", len(src))
