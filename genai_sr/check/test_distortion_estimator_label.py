"""check/test_distortion_estimator_label.py -- which label does the frozen counter follow under radial
distortion?  (independent check of the 'Hann-weighted local scale' claim of the optics study)

A synthetic two-family grating (h = 41, v = 33 TPI, untilted, 1024 px = 1 inch, mild yarn-like nonlinearity) is
distorted with cnn/augment_optics.radial_distortion (read-only import) at several k1. The frozen counter
(methods/counting.estimate, method='ours', defaults) is run on each output and its measured factor
h_meas / h_clean is compared with the label factors of the engine: uniform (area-mean = guide-line mean),
Hann-weighted and Hann^2-weighted local scale.

    <venv2>/bin/python check/test_distortion_estimator_label.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

WVAQ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WVAQ, "cnn"))
sys.path.insert(0, os.path.join(WVAQ, "methods"))
import augment_optics as AO  # noqa: E402
import counting as C  # noqa: E402

N, H_TPI, V_TPI = 1024, 41.0, 33.0


def grating():
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    a = np.cos(2 * np.pi * H_TPI * xx / N + 0.3)
    b = np.cos(2 * np.pi * V_TPI * yy / N + 1.1)
    # plain-weave-like interlacing: product term creates the (h, v) diagonal peaks of a real weave
    I = 0.5 + 0.18 * a + 0.18 * b + 0.12 * a * b
    rng = np.random.default_rng(0)
    I = I + rng.normal(0, 0.02, I.shape)
    g = np.clip(I * 255, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], 2)


def main():
    img = grating()
    r0 = C.estimate(img, method="ours")
    h0, v0 = float(r0["h_count"]), float(r0["v_count"])
    print(f"clean: ours h={h0:.3f} v={v0:.3f} (truth {H_TPI}, {V_TPI})")
    rows = []
    for k1 in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10):
        out, lt = AO.radial_distortion(img, k1=k1, center=(0.5, 0.5))
        r = C.estimate(out, method="ours")
        mh, mv = float(r["h_count"]) / h0, float(r["v_count"]) / v0
        fu = lt.weighted_scale("uniform")
        fh = lt.weighted_scale("hann")
        f2 = lt.weighted_scale("hann2")
        fc = lt.crossing((1.0, 1.0))[:2]
        rows.append((k1, mh, mv, fc[0], fu[0], fh[0], f2[0]))
        print(f"k1={k1:+.2f}: ours factor h {mh:.4f} v {mv:.4f} | label factor: guide-line mean {fc[0]:.4f}  "
              f"area-mean {fu[0]:.4f}  Hann {fh[0]:.4f}  Hann^2 {f2[0]:.4f}")
    R = np.array(rows)
    meas = np.log(np.r_[R[:, 1], R[:, 2]])
    for j, name in ((3, "guide-line mean"), (5, "Hann"), (6, "Hann^2")):
        pred = np.log(np.r_[R[:, j], R[:, j]])
        slope = float((pred * meas).sum() / (pred * pred).sum())
        resid = np.abs(np.exp(meas - pred) - 1)
        print(f"  vs {name:16s}: through-origin slope of log factors {slope:.3f}, median |residual| {100 * np.median(resid):.2f}%, "
              f"max {100 * resid.max():.2f}%")


if __name__ == "__main__":
    main()
