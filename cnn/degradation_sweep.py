#!/usr/bin/env python
"""Capture-quality sweep of a CNN checkpoint on the 13 legacy hand-counted crops.

Each legacy 1-inch crop (shared frame detector, ~1800 px/inch native) is
degraded synthetically and re-scored (best-assignment and axis-matched MAPE):

    resolution : resample to R px/inch (R in 600, 400, 300, 200, 150) and back
                 -> the effective pixels per thread falls to R / density
    blur       : Gaussian defocus, sigma in px at 600 px/inch
    jpeg       : JPEG quality at 600 px/inch
    glare      : additive specular blob (Gaussian, sigma 0.25 in) of peak
                 brightness g in {0.3, 0.6, 1.0} (fraction of 255) + clipping

    python degradation_sweep.py --ckpt runs/synth_r18_v2mix_aniso/model.pt --out runs/.../degradation
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

import cnn_common as C
import evaluate as E

BASE = 600  # px / inch at which blur / JPEG / glare are applied


def at_res(crop, R):
    lo = cv2.resize(crop, (R, R), interpolation=cv2.INTER_AREA)
    return lo


def degrade(crop, kind, val):
    c = cv2.resize(crop, (BASE, BASE), interpolation=cv2.INTER_AREA)
    if kind == "clean":
        return c
    if kind == "resolution":
        return at_res(crop, int(val))
    if kind == "blur":
        return cv2.GaussianBlur(c, (0, 0), float(val))
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, int(val)])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if kind == "glare":
        yy, xx = np.mgrid[0:BASE, 0:BASE].astype(np.float32) / BASE
        blob = np.exp(-((xx - 0.4) ** 2 + (yy - 0.45) ** 2) / (2 * 0.25 ** 2))
        return np.clip(c.astype(np.float32) + 255 * float(val) * blob[..., None], 0, 255).astype(np.uint8)
    raise ValueError(kind)


SWEEP = [("clean", 0), ("resolution", 400), ("resolution", 300), ("resolution", 200), ("resolution", 150),
         ("blur", 1.0), ("blur", 2.0), ("blur", 3.0), ("jpeg", 50), ("jpeg", 20), ("jpeg", 10),
         ("glare", 0.3), ("glare", 0.6), ("glare", 1.0)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = C.get_device(a.device)
    model = C.load_ckpt(a.ckpt, dev)
    data = E.legacy_crops("detect")
    rows, summ = [], []
    for kind, val in SWEEP:
        ph, pv = [], []
        for g, crop in data:
            d = degrade(crop, kind, val)
            # the crop is still exactly 1 inch: predict_crop resamples to the test px/inch
            o = C.predict_crop(model, d, dev, Ps=(768,), flips=False)
            ph.append(o["h"]), pv.append(o["v"])
            rows.append(dict(kind=kind, val=val, name=g.name, warp=g.warp, weft=g.weft, pred_h=o["h"], pred_v=o["v"]))
        wa = np.array([g.warp for g, _ in data]); we = np.array([g.weft for g, _ in data])
        am = C.count_metrics(ph, pv, wa, we)
        pa, pb, _ = C.best_assignment(ph, pv, wa, we)
        ba = C.count_metrics(pa, pb, wa, we)
        gm_t, gm_p = np.sqrt(wa * we), np.sqrt(np.array(ph) * np.array(pv))
        summ.append(dict(kind=kind, val=val, n=len(wa), MAPE_axis=am["MAPE"], MAE_axis=am["MAE"],
                         n10_axis=am["n10_h"] + am["n10_v"], MAPE_best=ba["MAPE"], MAE_best=ba["MAE"],
                         MAPE_geomean=float(100 * np.mean(np.abs(gm_p - gm_t) / gm_t))))
        print(summ[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "degradation_predictions.csv"), index=False)
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(a.out, "degradation_summary.csv"), index=False)
    print(s.round(1).to_string())
    with open(os.path.join(a.out, "degradation_meta.json"), "w") as f:
        json.dump({"ckpt": a.ckpt, "base_ppi": BASE, "test_P": 768, "flips": False,
                   "legacy_window": "detect"}, f, indent=1)


if __name__ == "__main__":
    main()
