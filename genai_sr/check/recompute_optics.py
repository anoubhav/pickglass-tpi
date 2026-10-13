"""check/recompute_optics.py -- independent re-computation of two optics-job headline numbers from its saved
long-format data (cnn/aug_optics/*.csv[.gz]); does not import augment_optics_verify.py.

  (1) label tracking, real-150, zoom/stretch/shear/bow: slope 0.986 [0.980, 0.989], median |resid| 0.17 %
  (2) minimum resolution, high-res real (n = 65), phone pipeline at 4 px/yarn: median change 0.49 %, within-10 % 89 %;
      plus the same numbers restricted to a FIXED image subset (selection check across px/yarn levels).

    <venv2>/bin/python check/recompute_optics.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

WVAQ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(WVAQ, "cnn", "aug_optics")
GEO = ("zoom_", "strx_", "stry_", "sheary_", "shearx_", "bowy_")


def main():
    e = pd.read_csv(os.path.join(AO, "drift_long.csv.gz"), dtype={"image": str})
    g = e[(e.set == "real150") & e.cond.str.startswith(GEO) & ~e.cond.str.contains(r"\+") & e.in_band & ~e.failed].copy()
    g = g[np.isfinite(g.fc) & (g.obs > 0) & (g.base > 0)]
    g["x"], g["y"] = np.log(g.fc), np.log(g.obs / g.base)
    g["k"] = g.cond + "|" + g.axis
    print("conditions:", g.cond.nunique(), "points:", g.k.nunique(), "values:", len(g), "images:", g.image.nunique())
    X = g.pivot_table(index="image", columns="k", values="x")
    Y = g.pivot_table(index="image", columns="k", values="y")

    def fit(idx):
        xm, ym = np.nanmedian(X.values[idx], 0), np.nanmedian(Y.values[idx], 0)
        ok = np.isfinite(xm) & np.isfinite(ym)
        return np.polyfit(xm[ok], ym[ok], 1)[0], xm[ok], ym[ok]

    b, xm, ym = fit(np.arange(len(X)))
    rng = np.random.default_rng(7)
    bs = [fit(rng.integers(0, len(X), len(X)))[0] for _ in range(1000)]
    print(f"(1) slope {b:.4f} [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]; median |resid| "
          f"{100 * np.median(np.abs(ym - xm)):.3f}%  max {100 * np.max(np.abs(ym - xm)):.2f}%")
    # tail: share of individual values more than 10 % off the predicted change, per condition
    g["flip"] = np.abs(g.y - g.x) > np.log(1.1)
    t = g.groupby("cond").flip.mean().mul(100).round(1)
    print("    share of values >10% off prediction, by condition:", t.to_dict())

    r = pd.read_csv(os.path.join(AO, "resolution_long.csv"), dtype={"image": str})
    hr = r[(r.subset == "real150_highres") & (r.pipeline == "phone") & r.applicable]
    print("\n(2) high-res real, phone pipeline, vs clean prediction (all applicable images per level)")
    for p, q in hr.groupby("ppt_target"):
        a = np.abs(q.err_ref.astype(float)) * 100
        print(f"   ppt {p:>4}: n_img {q.image.nunique():2d}  median |change| {np.median(a):6.2f}%  within10 {100 * np.mean(a <= 10):5.1f}%  "
              f">10% {100 * np.mean(a > 10):5.1f}%")
    s = pd.read_csv(os.path.join(AO, "resolution_long.csv"), dtype={"image": str})
    sy = s[(s.subset == "synth100") & (s.pipeline == "phone") & s.applicable & s.truth.notna()]
    common = set.intersection(*[set(q.image) for p, q in sy.groupby("ppt_target") if p <= 5])
    print(f"\n    synth100 vs exact label, phone pipeline, on the {len(common)} images applicable at every level 2-5 px/yarn:")
    for p, q in sy[sy.image.isin(common)].groupby("ppt_target"):
        if p > 5:
            continue
        a = np.abs(q.err_gt.astype(float)) * 100
        print(f"   ppt {p:>4}: median |err| {np.median(a):5.2f}%  within10 {100 * np.mean(a <= 10):5.1f}%")
    cl = pd.read_csv(os.path.join(AO, "raw_ours_synth.csv"), dtype={"image": str}) if os.path.exists(os.path.join(AO, "raw_ours_synth.csv")) else None
    if cl is not None:
        print("    (raw_ours_synth.csv columns:", list(cl.columns)[:12], ")")


if __name__ == "__main__":
    main()
