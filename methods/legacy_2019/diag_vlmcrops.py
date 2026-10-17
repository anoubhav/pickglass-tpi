"""Diagnostics of the legacy doc3 + count.py counter on the canonical legacy crops (vlm/crops/legacy).

Same analyses as count_wrap_edges.py / the seed sweep / the in-memory (no JPEG round trip)
variant, but on run_vlmcrops/ (the one-crop-rule run used by paper/tables/legacy13_unified.csv).
Run: <venv2>/bin/python methods/legacy_2019/diag_vlmcrops.py
Paper: supplement Sec. S3 (full legacy results).
"""
import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_vlmcrops")
USABLE = "a1 a2 a4 a5 a6 a8 a10 r2 r3 r4 r6 r7 r8".split()


def counted(seq):
    out = []
    for x in range(len(seq)):
        d = (int(seq[x]) - int(seq[x - 1])) % 256
        if d == 255:
            out.append(int(seq[x]) - int(seq[x - 1]))
    return out


def ape(pred, gt):
    return 100 * np.abs(pred - gt) / gt


def main():
    r = pd.read_csv(os.path.join(RUN, "results.csv"))
    r = r[r.key.isin(USABLE)].set_index("key").loc[USABLE]
    gw, gf = r.gt_warp.astype(float), r.gt_weft.astype(float)
    pw, pf = r.pred_warp_per_inch.astype(float), r.pred_weft_per_inch.astype(float)
    print(f"seed 0 (JPEG dot map): warp MAE {np.mean(abs(pw - gw)):.1f} ({ape(pw, gw).mean():.1f}%, "
          f"{int((ape(pw, gw) <= 10).sum())}/13 within 10%), weft MAE {np.mean(abs(pf - gf)):.1f} "
          f"({ape(pf, gf).mean():.1f}%, {int((ape(pf, gf) <= 10).sum())}/13)")
    iw, iff = r.pred_warp_per_inch_inmem.astype(float), r.pred_weft_per_inch_inmem.astype(float)
    print(f"seed 0, in memory (no JPEG round trip): warp {ape(iw, gw).mean():.1f}% MAPE, "
          f"weft {ape(iff, gf).mean():.1f}%")
    # uint8 wrap-around: counted edges that are -1 grey-level steps
    tw = tm = 0
    for k in USABLE:
        img = cv2.imread(os.path.join(RUN, "precropped", "dotmaps", f"{k}.png.jpg"), 0)
        for seq in (img[0, :], img[:, 0]):
            c = counted(seq)
            tw += len(c)
            tm += sum(d == -1 for d in c)
    print(f"counted edges that are -1 steps (13 crops, both axes): {tm}/{tw} ({100 * tm / max(tw, 1):.0f}%)")
    # seed sweep
    s = pd.read_csv(os.path.join(RUN, "seed_sweep_precropped.csv"))
    s = s[s.key.isin(USABLE)]
    rows = [(0, np.mean(abs(pw - gw)), np.mean(abs(pf - gf)))]
    for seed, g in s.groupby("seed"):
        g = g.set_index("key").loc[USABLE]
        rows.append((seed, np.mean(abs(g.pred_warp_per_inch - gw)), np.mean(abs(g.pred_weft_per_inch - gf))))
    for seed, a, b in rows:
        print(f"seed {seed}: warp MAE {a:.1f}, weft MAE {b:.1f}")
    print(f"warp MAE range over seeds: {min(x[1] for x in rows):.1f}-{max(x[1] for x in rows):.1f} TPI")


if __name__ == "__main__":
    main()
