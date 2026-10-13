"""check/test_label_transforms_grating.py -- independent synthetic-grating check of the label-transform
derivations in cnn/augment_optics.py (read-only import; that file is not modified).

Method (independent of the module's label engine and of its own Hilbert-based tests):
  * Render ONE yarn family as a PHASE-ENCODED image: channel 0 = cos(phi), channel 1 = sin(phi),
    phi = 2 pi k.p with the reciprocal vector k (cycles / input px) defined from the label convention
    h = crossings per inch along a row  ->  k_h = (h, h tan ax) / ppi,
    v = crossings per inch along a column -> k_v = (v tan ay, v) / ppi.
  * Push the image through the module's own transform function (the pixels the CNN would see).
  * Recover the wrapped phase of the OUTPUT pixels with atan2(sin, cos), unwrap along every row (h family)
    or column (v family), extend by half a pixel at each end, and divide the end-to-end phase by 2 pi and by
    the DECLARED output extent -> the measured guide-line crossing count per inch.
  * Compare the measured count with (a) the closed-form derivation in the module docstring and
    (b) the module's label engine OLT.crossing().
The measurement is exact up to interpolation error of the cos / sin channels (~1e-4 relative here).

    <venv2>/bin/python check/test_label_transforms_grating.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

WVAQ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WVAQ, "cnn"))
import augment_optics as AO  # noqa: E402

N = 1024
H_TPI, V_TPI = 41.0, 33.0          # cycles per inch along the guide lines (25-31 px per cycle)
TOL_CLOSED = 2e-3                  # 0.2 % (integer windows / bands make the closed form differ by < 0.1 %)
TOL_ENGINE = 5e-4                  # 0.05 %


def render(family, ax_deg, ay_deg, n=N, ppi=N):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    if family == "h":
        k = np.array([H_TPI, H_TPI * math.tan(math.radians(ax_deg))]) / ppi
    else:
        k = np.array([V_TPI * math.tan(math.radians(ay_deg)), V_TPI]) / ppi
    phi = 2 * np.pi * (k[0] * xx + k[1] * yy) + 0.37
    img = np.stack([np.cos(phi), np.sin(phi), np.full_like(phi, 0.5)], 2).astype(np.float32)
    return img


def measure(out, family, extent_out):
    """mean over guide lines of the end-to-end unwrapped phase / 2 pi, per declared inch."""
    psi = np.arctan2(out[..., 1].astype(np.float64), out[..., 0].astype(np.float64))
    if family == "h":
        u = np.unwrap(psi, axis=1)
        tot = (u[:, -1] - u[:, 0]) + 0.5 * (u[:, 1] - u[:, 0]) + 0.5 * (u[:, -1] - u[:, -2])
        ext = extent_out[0]
    else:
        u = np.unwrap(psi, axis=0)
        tot = (u[-1, :] - u[0, :]) + 0.5 * (u[1, :] - u[0, :]) + 0.5 * (u[-1, :] - u[-2, :])
        ext = extent_out[1]
    per_line = np.abs(tot) / (2 * np.pi) / ext
    return float(per_line.mean()), float(per_line.min()), float(per_line.max())


def run_case(name, fn, kwargs, ax, ay, closed_h, closed_v, results):
    meas = {}
    lt = None
    for fam in ("h", "v"):
        out, lt = fn(render(fam, ax, ay), extent=(1.0, 1.0), **kwargs)
        meas[fam] = measure(out, fam, lt.extent_out)
    eng_h, eng_v, _ = lt.crossing((H_TPI, V_TPI), angles=(ax, ay))
    kh, kv = lt._k_in((H_TPI, V_TPI), (ax, ay), "crossing")
    dense_h = float(lt._row_col(kh, n=lt.out_size[1])[0].mean())
    dense_v = float(lt._row_col(kv, n=lt.out_size[0])[1].mean())
    ch, cv = closed_h(lt), closed_v(lt)
    row = dict(case=name, ax=ax, ay=ay, meas_h=meas["h"][0], meas_v=meas["v"][0], closed_h=ch, closed_v=cv,
               engine_h=eng_h, engine_v=eng_v, engine_dense_h=dense_h, engine_dense_v=dense_v, line_spread_h=(meas["h"][2] - meas["h"][1]) / meas["h"][0],
               line_spread_v=(meas["v"][2] - meas["v"][1]) / meas["v"][0])
    row["err_closed"] = max(abs(row["meas_h"] / ch - 1), abs(row["meas_v"] / cv - 1)) if ch and cv else float("nan")
    row["err_engine"] = max(abs(row["meas_h"] / eng_h - 1), abs(row["meas_v"] / eng_v - 1))
    row["err_engine_dense"] = max(abs(row["meas_h"] / dense_h - 1), abs(row["meas_v"] / dense_v - 1))
    row["pass"] = (min(row["err_engine"], row["err_engine_dense"]) <= TOL_ENGINE) and (not (ch and cv) or row["err_closed"] <= TOL_CLOSED)
    results.append(row)
    print(f"{'PASS' if row['pass'] else 'FAIL'} {name:34s} ax={ax:+5.1f} ay={ay:+5.1f} | measured h={row['meas_h']:8.4f} "
          f"v={row['meas_v']:8.4f} | closed h={ch:8.4f} v={cv:8.4f} | engine h={eng_h:8.4f} v={eng_v:8.4f} | "
          f"err closed {100 * row['err_closed']:.3f}% engine {100 * row['err_engine']:.3f}% (dense {100 * row['err_engine_dense']:.3f}%)")


def main():
    R = []
    t = math.tan
    rad = math.radians
    for ax, ay in ((0.0, 0.0), (7.0, -5.0), (-9.0, 6.0)):
        # (S) stretch: h' = h / sx, v' = v / sy at any tilt (closed form uses the integer window actually cropped)
        run_case("stretch sx=1.3 sy=1", AO.stretch, dict(sx=1.3, sy=1.0), ax, ay,
                 lambda lt: H_TPI / lt.params["sx_eff"], lambda lt: V_TPI / lt.params["sy_eff"], R)
        run_case("stretch sx=1 sy=1.25", AO.stretch, dict(sx=1.0, sy=1.25), ax, ay,
                 lambda lt: H_TPI / lt.params["sx_eff"], lambda lt: V_TPI / lt.params["sy_eff"], R)
        run_case("stretch nominal check sx=1.3", AO.stretch, dict(sx=1.3, sy=1.0), ax, ay,
                 lambda lt: H_TPI / 1.3, lambda lt: V_TPI, R)
        # (Z) zoom
        run_case("zoom s=1.25", AO.zoom, dict(s=1.25), ax, ay,
                 lambda lt: H_TPI / 1.25, lambda lt: V_TPI / 1.25, R)
        # (H) weft skew y_out = y_in + g (x_in - cx): v' = v, h' = h (1 - g tan ax)
        for g in (0.1, -0.1, 0.05):
            run_case(f"shear y g={g:+.2f} inner", AO.shear, dict(gamma=g, axis="y", mode="inner"), ax, ay,
                     lambda lt, g=g: H_TPI * (1 - g * t(rad(ax))), lambda lt: V_TPI, R)
        run_case("shear y g=+0.10 frame", AO.shear, dict(gamma=0.1, axis="y", mode="frame"), ax, ay,
                 lambda lt: H_TPI * (1 - 0.1 * t(rad(ax))), lambda lt: V_TPI * lt.params["Ho"] / N, R)
        # warp skew (axis x), derived here: x_out = x_in + g (y_in - cy) -> h' = h, v' = v (1 - g tan ay)
        run_case("shear x g=+0.08 inner", AO.shear, dict(gamma=0.08, axis="x", mode="inner"), ax, ay,
                 lambda lt: H_TPI, lambda lt: V_TPI * (1 - 0.08 * t(rad(ay))), R)
        # (B) bow: both counts preserved exactly (end points of every row have zero displacement)
        for d in (0.02, 0.04):
            run_case(f"bow y depth={d}", AO.bow, dict(depth=d, axis="y"), ax, ay,
                     lambda lt: H_TPI, lambda lt: V_TPI, R)
        run_case("bow x depth=0.03", AO.bow, dict(depth=0.03, axis="x"), ax, ay,
                 lambda lt: H_TPI, lambda lt: V_TPI, R)
        run_case("bow y depth=0.03 sign=-1", AO.bow, dict(depth=0.03, axis="y", sign=-1), ax, ay,
                 lambda lt: H_TPI, lambda lt: V_TPI, R)
        # (D) radial distortion: no closed form; engine (mean guide-line count) vs measurement, and the
        # docstring identity 'mean row count == area-mean local scale' for untilted yarns
        for k1 in (0.02, 0.1, -0.1):
            run_case(f"radial k1={k1:+.2f} (engine only)", AO.radial_distortion, dict(k1=k1, center=(0.5, 0.5)), ax, ay,
                     (lambda lt: H_TPI * lt.params["area_mean_scale"][0]) if (ax == 0 and ay == 0) else (lambda lt: 0.0),
                     (lambda lt: V_TPI * lt.params["area_mean_scale"][1]) if (ax == 0 and ay == 0) else (lambda lt: 0.0), R)
    n_pass = sum(r["pass"] for r in R)
    worst_c = max(r["err_closed"] for r in R if np.isfinite(r["err_closed"]) and r["closed_h"])
    worst_e = max(r["err_engine"] for r in R)
    worst_d = max(r["err_engine_dense"] for r in R)
    print(f"\n{n_pass}/{len(R)} cases pass; worst |measured/closed-1| = {100 * worst_c:.4f}%, "
          f"worst |measured/engine-1| = {100 * worst_e:.4f}% (33-line engine), {100 * worst_d:.4f}% (engine on every line)")
    # what the 'exact' label claims mean physically for radial distortion: spread across guide lines
    for r in R:
        if r["case"].startswith("radial") and r["ax"] == 0:
            print(f"  {r['case']}: per-line count spread (max-min)/mean h {100 * r['line_spread_h']:.2f}%  "
                  f"v {100 * r['line_spread_v']:.2f}%  label factor h' / h = {r['engine_h'] / H_TPI:.4f}")
    import csv
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_label_transforms_grating.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(R[0].keys()))
        w.writeheader()
        w.writerows(R)
    return 0 if n_pass == len(R) else 1


if __name__ == "__main__":
    sys.exit(main())
