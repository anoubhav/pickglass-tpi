"""Verification check 1d: independent rotation recovery (does NOT use aug_common.py).

For every augmentation on disk:
  * re-rectify the raw aug JPEG with its XML polygon through common/rectify.py's own contract
    (order_corners -> homography -> warpPerspective, bilinear, BORDER_REPLICATE) and compare with
    the builder's crops/{id}.jpg (mean |diff|, size)
  * k by my own matcher, hypotheses np.rot90(src_tool_crop, j) (CCW), two features:
      A  low-pass grey 64x64, inner 80 %, standardised -> NCC
      B  doubled-angle structure-tensor field (Jxx-Jyy, 2Jxy) at 32x32, NOT mean-subtracted, so
         the global twill / weave direction counts; a 90-degree turn moves the field AND negates
         it (2*theta shifts by 180 degrees) -> cosine similarity
    score = (A + B) / 2, k = argmax, margin = best - second
Writes verify/ncc_mine.csv.
"""
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402

NB = (os.environ.get("PICKGLASS_NOTEBOOKS") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "data", "photos"))
TOOL = os.path.join(ROOT, "annotation_tool", "data", "crops")
CROPS = os.path.join(AUG, "crops")


def read_poly(p):
    poly = ET.parse(p).getroot().find("object/polygon")
    return np.array([[float(poly.find(f"x{k}").text), float(poly.find(f"y{k}").text)] for k in range(1, 5)])


def my_rectify(img, pts):
    c = R.order_corners(pts)
    H, S = R.homography(None, corners=c)
    return cv2.warpPerspective(img, H, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def feat_lp(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def feat_orient(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA if g.shape[0] > 256 else cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    a = cv2.GaussianBlur(gx * gx - gy * gy, (0, 0), 4.0)
    b = cv2.GaussianBlur(2 * gx * gy, (0, 0), 4.0)
    return np.stack([cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA),
                     cv2.resize(b, (32, 32), interpolation=cv2.INTER_AREA)], -1)


def ncc(a, b, m):
    a = a[m:-m, m:-m].astype(np.float64)
    b = b[m:-m, m:-m].astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def cos(a, b, m):
    a = a[m:-m, m:-m].ravel().astype(np.float64)
    b = b[m:-m, m:-m].ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def scores(src, aug):
    la, oa = feat_lp(aug), feat_orient(aug)
    ls, os_ = feat_lp(src), feat_orient(src)
    out = []
    for j in range(4):
        A = ncc(np.rot90(ls, j), la, 6)
        B = cos(np.rot90(os_, j, axes=(0, 1)) * (-1) ** j, oa, 3)
        out.append((A, B, (A + B) / 2))
    return np.array(out)


def main():
    ids = sorted(f[:-4] for f in os.listdir(CROPS) if f.endswith(".jpg"))
    src_cache = {}
    rows = []
    for n, aid in enumerate(ids):
        i = aid.split("_aug")[0]
        if i not in src_cache:
            src_cache = {i: cv2.imread(os.path.join(TOOL, f"{i}.jpg"))}
        src = src_cache[i]
        raw = cv2.imread(os.path.join(NB, "augmented_images", f"{aid}.jpeg"))
        pts = read_poly(os.path.join(NB, "augmented_xml", f"{aid}.xml"))
        mine = my_rectify(raw, pts)
        built = cv2.imread(os.path.join(CROPS, f"{aid}.jpg"))
        same_size = mine.shape == built.shape
        mad = float(np.abs(mine.astype(np.int16) - built.astype(np.int16)).mean()) if same_size else np.nan
        sc = scores(src, mine)
        order = np.argsort(-sc[:, 2])
        rows.append(dict(aug_id=aid, source_id=int(i), S_mine=mine.shape[0], S_built=built.shape[0],
                         crop_mad=round(mad, 3), k_mine=int(order[0]) * 90,
                         margin_mine=round(float(sc[order[0], 2] - sc[order[1], 2]), 4),
                         best_mine=round(float(sc[order[0], 2]), 4),
                         k_lp_only=int(np.argmax(sc[:, 0])) * 90, k_orient_only=int(np.argmax(sc[:, 1])) * 90,
                         **{f"s{90 * j}": round(float(sc[j, 2]), 4) for j in range(4)}))
        if n % 100 == 0:
            print(n, aid, rows[-1]["k_mine"], rows[-1]["margin_mine"], rows[-1]["crop_mad"], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "ncc_mine.csv"), index=False)


if __name__ == "__main__":
    main()
