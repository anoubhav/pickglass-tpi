"""Verification check 1a (independent of aug_common.py): blind contact sheets.

Samples 40 augmentations (fixed seed) from the files ON DISK (not from manifest.csv) and draws,
per row:   AUG crop (aug1016/crops) | source tool crop rotated by np.rot90 j = 0,1,2,3 (CCW)
so the matching column can be picked by eye without seeing the manifest's k.
Also writes sample.csv (aug_id only).
"""
import os
import random

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUG = os.path.dirname(HERE)
ROOT = os.path.dirname(AUG)
TOOL = os.path.join(ROOT, "annotation_tool", "data", "crops")
CROPS = os.path.join(AUG, "crops")
OUT = os.path.join(HERE, "sheets")
T = 190

ids = sorted(f[:-4] for f in os.listdir(CROPS) if f.endswith(".jpg"))
rng = random.Random(20261007)
sample = rng.sample(ids, 40)
with open(os.path.join(HERE, "sample.csv"), "w") as f:
    f.write("aug_id\n" + "\n".join(sample) + "\n")


def thumb(im):
    return cv2.resize(im, (T, T), interpolation=cv2.INTER_AREA)


def label(im, s):
    im = im.copy()
    cv2.putText(im, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return im


for s in range(4):
    rows = []
    for aid in sample[s * 10:(s + 1) * 10]:
        src_id = aid.split("_aug")[0]
        src = cv2.imread(os.path.join(TOOL, f"{src_id}.jpg"))
        aug = cv2.imread(os.path.join(CROPS, f"{aid}.jpg"))
        cells = [label(thumb(aug), f"{aid} AUG")]
        for j in range(4):
            cells.append(label(thumb(np.ascontiguousarray(np.rot90(src, j))), f"src rot{j * 90}"))
        row = np.hstack([np.pad(c, ((3, 3), (3, 3), (0, 0)), constant_values=255) for c in cells])
        rows.append(row)
    sheet = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT, f"sheet{s}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("sample:", sample)
