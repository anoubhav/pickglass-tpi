#!/usr/bin/env python
"""Side-by-side visual calibration figure: real rectified crops vs synthetic.

Top two rows   : 8 real crops (rectified with the annotated corners), shown
                 (a) whole and (b) as the 224-px tile the CNN sees at the
                 canonical 768 px/inch.
Bottom two rows: the same for 8 synthetic crops.

    python compare_real_synth.py --out figs/real_vs_synth.jpg
"""
import argparse
import cv2
import numpy as np

import fabric_data as fd
import synth

P = 768  # canonical px / inch used by the tile models
T = 224


def tile_view(crop, tile=T, P=P):
    c = cv2.resize(crop, (P, P), interpolation=cv2.INTER_AREA if crop.shape[0] > P else cv2.INTER_CUBIC)
    o = (P - tile) // 2
    return c[o:o + tile, o:o + tile]


def label(im, txt):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(im, txt, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/real_vs_synth.jpg")
    ap.add_argument("--real", default="3,20,40,57,60,85,110,139")
    ap.add_argument("--seed", type=int, default=500)
    a = ap.parse_args()
    S = 224
    rows = [[], [], [], []]
    for idx in [int(x) for x in a.real.split(",")]:
        crop = fd.load_rectified(idx)
        rows[0].append(label(cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA), f"real #{idx} ({crop.shape[0]}px/in)"))
        rows[1].append(label(tile_view(crop), "real tile @768px/in"))
    for k in range(8):
        _, crop, lab = synth.generate(a.seed + k)
        rows[2].append(label(cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA),
                             f"syn {lab['weave']} {lab['epi']:.0f}x{lab['ppi']:.0f}"))
        rows[3].append(label(tile_view(crop), f"syn tile r_eff={lab['r_eff']:.0f}"))
    sheet = np.vstack([np.hstack(r) for r in rows])
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", a.out, sheet.shape)


if __name__ == "__main__":
    main()
