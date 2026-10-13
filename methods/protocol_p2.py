"""protocol_p2.py -- sensitivity of the legacy-13 numbers to the '1 inch' convention.

P1 (primary): the whole hand-cropped legacy image is the 1-inch window (2019 convention).
P2: the window is the frame-free region found by legacy.detect_window (dark-frame slivers
    removed); counts per inch = cycles per pixel x detected window side, i.e.
    count_P2 = count_P1 * (window side / image side) per axis.
Writes results/legacy13_window_scale.csv and prints metrics under P1 and P2 for every estimator.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import legacy  # noqa: E402
import summarise_legacy as SL  # noqa: E402


def window_scale():
    rows = []
    for n in legacy.NAMES:
        img = legacy.load_legacy(n)
        H, W = img.shape[:2]
        c, info = legacy.detect_window(img)
        win_w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
        win_h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
        rows.append(dict(image=n, W=W, H=H, win_w=win_w, win_h=win_h, sx=win_w / W, sy=win_h / H))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ws = window_scale()
    ws.to_csv(os.path.join(HERE, "results", "legacy13_window_scale.csv"), index=False)
    print(ws.round(3).to_string(index=False))
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "legacy13_estimators.csv"))
    d = d[(d.crop == "gt") & (d.gt_usable == True)].merge(ws[["image", "sx", "sy"]], on="image")  # noqa: E712
    out = []
    for est, g in d.groupby("estimator"):
        for proto, hs, vs in (("P1", 1.0, 1.0), ("P2", g.sx.values, g.sy.values)):
            gg = g.assign(hP=g.h * hs, vP=g.v * vs)
            for r in SL.pair_metrics(gg, "hP", "vP"):
                out.append(dict(estimator=est, protocol=proto, **r))
    o = pd.DataFrame(out)
    o.to_csv(os.path.join(HERE, "results", "legacy13_protocol_P1_P2.csv"), index=False)
    print(o[(o.axis == "both")].round(1).to_string(index=False))
