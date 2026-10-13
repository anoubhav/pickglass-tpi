"""octave_v1.py -- "octave-v1": a label-free octave (factor-of-two) decision on top of the frozen
"ours" counter (methods/results/counting_frozen_final.py, imported unchanged).

Why.  The frozen counter returns the dominant periodicity of the fabric image along each axis.
When the visible repeat along an axis contains TWO yarns (paired / basket / oxford ends, a colour
alternation such as white and blue ends, floats that cover two yarns) the dominant period is two
yarns and the counter reads half the yarns; when it locks onto a harmonic it reads double.

Rule (per axis; the frozen count c, the frozen lattice angle, the frozen pre-processing):
  1. PERIOD SHAPE.  Deskew the frozen pre-processed (PCA-grey, local-contrast-normalised) image so
     that the yarns of that family are axis-parallel (shear by the frozen lattice angle; of +-angle
     the one with more projection power at c), project along the yarns in 8 bands, cut every band
     profile into windows of 6 periods (hop 3) and fit harmonics k = 1..3 of the period N/c.  Align
     every window on the phase of k = 1 and average the coefficients (each window normalised by the
     sum of its |c_k|): the mean period shape G(phi) = Re sum_k C_k exp(i k phi), phi in [0, 2 pi).
  2. SECONDARY-RIDGE TEST.  r2 = prominence of the second-largest local maximum of G divided by
     that of the largest, counted only when it lies 0.35..0.5 periods from the main maximum (i.e.
     between two counted ridges).  A yarn-like ridge between every two counted ridges means that
     the counted period holds two yarns: if r2 >= RHO = 0.15 and the doubled frequency is still
     inside the analysed band (2c <= 0.42 N, the frozen Nyquist guard), the count is doubled and
     the test is repeated at the new count (at most MAX_DOUBLE = 2 doublings).  Otherwise the
     frozen count is kept.  Weave label and confidence are the frozen ones.
  The rule only ever DOUBLES a count, on positive evidence (a resolved second ridge).  A symmetric
  halving test (no second ridge at twice the counted period) was tried and rejected: on synthetic
  renders it broke as many correct counts as it fixed, and absence of a second ridge is also what
  alternately coloured yarns look like (ENABLE_HALVING = False; kept only as a diagnostic).
Parameters (RHO, MAX_DOUBLE, K, bands, window, 0.35) were chosen on fresh synthetic renders
(seeds 990000-990299 default generator, 991000-991299 with paired-end / paired-pick weaves,
992000-992149 unbalanced poplin-like plain / twill), the 13 legacy crops and the label-free
consistency of the 150 real crops (same-patch re-shoots, full/quarter-resolution copies, rotated
copies); no human count of the real scenes was read.  See PREREG_octave_v1.json.

    python octave_v1.py run150 [--workers 8]     -> predictions_octave_150.csv, octave_v1_150_wide.csv
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "methods", "results"))
sys.path.insert(0, os.path.join(ROOT, "common"))
import counting_frozen_final as C  # noqa: E402  (frozen, md5 3d0520d6bec8a016543ce7e5b353d784)

# the frozen module looks for conf_model.json next to itself (methods/results/, absent) and would fall
# back to its heuristic; methods/predictions_ours_150.csv was made with methods/conf_model.json, so load
# that file (read-only) to reproduce the frozen confidence exactly
_CONF = os.path.join(ROOT, "methods", "conf_model.json")
if os.path.exists(_CONF):
    with open(_CONF) as _f:
        C.CONF_MODEL = json.load(_f)

METHOD = "octave-v1"
RHO = 0.15          # doubling: secondary-ridge prominence ratio
RHO_H = 0.05        # (disabled) halving threshold of the rejected candidate
ENABLE_HALVING = False
MAX_DOUBLE = 2      # at most two doublings per axis (x4)
K = 3               # harmonics of the counted period
N_BANDS = 8
WIN_PERIODS = 6.0
D2_MIN = 0.35       # secondary ridge must lie 0.35..0.5 periods from the main ridge
R_MIN = C.R_MIN     # 18 threads per side, frozen search floor
NYQ = C.NYQ_FRAC    # 0.42 cycles / working pixel
REPS = ("pca",)


# ------------------------------------------------------------------------------ geometry
def _band_profiles(g, w, axis, angle_deg, n_bands=N_BANDS):
    N = g.shape[0]
    gi, wi = C._shear(g, w, math.tan(math.radians(angle_deg)), axis)
    edges = np.linspace(0, N, n_bands + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if axis == "x":
            num, den = gi[a:b].sum(0), wi[a:b].sum(0)
        else:
            num, den = gi[:, a:b].sum(1), wi[:, a:b].sum(1)
        out.append((num / np.maximum(den, 1e-3), den >= 0.5 * (b - a)))
    return out


def _proj_power(g, w, axis, angle_deg, f):
    N = g.shape[0]
    p = C._project(g, w, math.tan(math.radians(angle_deg)), axis)
    q = (p - p.mean()) * np.hanning(N)
    return float(np.abs((q * np.exp(-2j * np.pi * f * np.arange(N) / N)).sum()) ** 2)


def _shear_angle(g, w, axis, angle_deg, f):
    cands = sorted({round(angle_deg, 4), round(-angle_deg, 4)})
    return max(cands, key=lambda a: _proj_power(g, w, axis, a, f))


def _folded(profiles, f, k_max, win_periods, align):
    """Mean phase-aligned harmonic coefficients (1..k_max) of period N/f over all windows."""
    N = len(profiles[0][0])
    T = N / f
    L = int(round(win_periods * T))
    if L < 8 or L > N:
        return None
    hop = max(1, L // 2)
    x = np.arange(L)
    win = np.hanning(L)
    ks = np.arange(1, k_max + 1)
    E = np.exp(-2j * np.pi * np.outer(ks, x) / T)
    acc = np.zeros(k_max, complex)
    n = 0
    for p, ok in profiles:
        for s in range(0, N - L + 1, hop):
            if ok[s:s + L].mean() < 0.9:
                continue
            seg = p[s:s + L] - p[s:s + L].mean()
            c = (E * (seg * win)).sum(1)
            ref = np.angle(c[align - 1]) / align
            if align == 2 and (c[0] * np.exp(-1j * ref)).real < 0:
                ref += np.pi  # pi ambiguity of the k=2 phase: put the larger ridge at phi = 0
            acc += c * np.exp(-1j * ks * ref) / (np.abs(c).sum() + 1e-12)
            n += 1
    return acc / n if n else None


def _second_ridge(cm, n_phi=360):
    """Prominence ratio (2nd / 1st largest local maximum) of G and its distance (periods)."""
    if cm is None:
        return np.nan, np.nan
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    G = np.real(sum(cm[k] * np.exp(1j * (k + 1) * ph) for k in range(len(cm))))
    pk, pr = signal.find_peaks(np.r_[G, G, G], prominence=0)
    sel = (pk >= n_phi) & (pk < 2 * n_phi)
    pk, prom = pk[sel] - n_phi, pr["prominences"][sel]
    if len(pk) == 0:
        return np.nan, np.nan
    if len(pk) < 2:
        return 0.0, np.nan
    o = np.argsort(-prom)
    d = abs(pk[o[1]] - pk[o[0]]) / n_phi
    return float(prom[o[1]] / prom[o[0]]), float(min(d, 1 - d))


def ridge_ratio(g, w, axis, count, angle_deg):
    """(r2 at the counted period, r2 at the doubled period) for one image representation;
    a ratio counts only when the second ridge lies D2_MIN..0.5 periods from the main ridge."""
    a = _shear_angle(g, w, axis, angle_deg, count)
    prof = _band_profiles(g, w, axis, a)
    r2, d2 = _second_ridge(_folded(prof, count, K, WIN_PERIODS, 1))
    r2h, d2h = _second_ridge(_folded(prof, count / 2.0, 2 * K, WIN_PERIODS / 2.0, 2))
    r2 = r2 if (np.isfinite(d2) and d2 >= D2_MIN) else (0.0 if np.isfinite(r2) else np.nan)
    r2h = r2h if (np.isfinite(d2h) and d2h >= D2_MIN) else (0.0 if np.isfinite(r2h) else np.nan)
    return r2, r2h


# ------------------------------------------------------------------------------ main entry
def representations(crop_bgr, g=None, w=None, reps=REPS):
    """Image representations tested: the frozen PCA-grey pre-processed image ("pca"); optionally
    ("b", "g", "r") each colour channel pre-processed identically (candidate C3, NOT used by
    octave-v1: it created re-shoot disagreements, see PREREG)."""
    if g is None:
        g, w, _, _ = C.preprocess(crop_bgr)
    out = {"pca": (g, w)}
    for name, k in (("b", 0), ("g", 1), ("r", 2)):
        if name in reps:
            ch = crop_bgr[:, :, k]
            gg, ww, _, _ = C.preprocess(np.dstack([ch, ch, ch]))
            out[name] = (gg, ww)
    return out


def octave_axis(reps, axis, count, angle_deg, N, trace):
    """Return the octave factor for one axis."""
    if not (np.isfinite(count) and count > 0):
        return 1.0
    hi = NYQ * N
    fac = 1.0
    c = count
    first = None
    for it in range(MAX_DOUBLE):
        if 2 * c > hi:
            trace.append(f"{axis}:stop(2c>{hi:.0f})")
            break
        stats = {nm: ridge_ratio(g, w, axis, c, angle_deg) for nm, (g, w) in reps.items()}
        if it == 0:
            first = stats
        r2 = np.nanmax([s[0] for s in stats.values()]) if any(np.isfinite(s[0]) for s in stats.values()) else np.nan
        trace.append(f"{axis}:c={c:.1f} r2max={r2:.3f}")
        if np.isfinite(r2) and r2 >= RHO:
            c *= 2.0
            fac *= 2.0
        else:
            break
    if ENABLE_HALVING and fac == 1.0:
        if first is None:
            first = {nm: ridge_ratio(g, w, axis, c, angle_deg) for nm, (g, w) in reps.items()}
        r2h = [s[1] for s in first.values()]
        if all(np.isfinite(r2h)):
            m = max(r2h)
            trace.append(f"{axis}:r2h_max={m:.3f}")
            if m < RHO_H and count / 2.0 >= R_MIN:
                fac = 0.5
    return fac


def estimate(crop_bgr):
    """octave-v1 on one rectified crop: frozen 'ours' + octave factors per axis."""
    t0 = time.time()
    g, w, N, info = C.preprocess(crop_bgr)
    e = C._estimate_pre(g, w, N, info, "ours", "lcn", "mask", with_weave=False, cache={})
    d = e["debug"]
    lat = d.get("lattice") or {}
    ax = float(lat.get("angle_x") or 0.0)
    ay = float(lat.get("angle_y") or 0.0)
    reps = representations(crop_bgr, g, w)
    trace = []
    fx = octave_axis(reps, "x", e["h_count"], ax, N, trace)
    fy = octave_axis(reps, "y", e["v_count"], ay, N, trace)
    return {"h_count": e["h_count"] * fx, "v_count": e["v_count"] * fy, "weave": e["weave"],
            "confidence": e["confidence"], "frozen_h": e["h_count"], "frozen_v": e["v_count"],
            "octave_fx": fx, "octave_fy": fy, "angle_x": ax, "angle_y": ay, "trace": " ".join(trace),
            "fusion": "|".join(d.get("fusion") or []), "runtime_s": time.time() - t0, "frozen_debug": d}


# ------------------------------------------------------------------------------ 150 runner
def _job150(i):
    cv2.setNumThreads(1)
    import rectify as R
    crop = R.rectify(int(i))
    r = estimate(crop)
    r.pop("frozen_debug")
    r["id"] = int(i)
    r["S"] = int(crop.shape[0])
    return r


def run150(workers=8):
    import pandas as pd
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(workers) as p:
        rows = p.map(_job150, range(150), chunksize=1)
    df = pd.DataFrame(rows).sort_values("id")
    # same leading columns as methods/predictions_ours_150.csv (id, S, h_count, v_count, weave, confidence, ...)
    ref = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    assert np.allclose(df.set_index("id")["frozen_h"], ref["h_count"], rtol=1e-9, equal_nan=True), "frozen h mismatch"
    assert np.allclose(df.set_index("id")["frozen_v"], ref["v_count"], rtol=1e-9, equal_nan=True), "frozen v mismatch"
    assert np.allclose(df.set_index("id")["confidence"], ref["confidence"], rtol=1e-9, equal_nan=True), "confidence mismatch"
    assert (df.set_index("id")["weave"] == ref["weave"]).all(), "weave mismatch"
    # SAME columns (and order) as methods/predictions_ours_150.csv: every column is the frozen one except
    # h_count / v_count (octave-corrected); octave columns are appended at the end
    o = df.set_index("id")
    out = ref.copy()
    out["h_count"] = o["h_count"].reindex(out.index)
    out["v_count"] = o["v_count"].reindex(out.index)
    out["frozen_h_count"] = o["frozen_h"].reindex(out.index)
    out["frozen_v_count"] = o["frozen_v"].reindex(out.index)
    out["octave_fx"] = o["octave_fx"].reindex(out.index)
    out["octave_fy"] = o["octave_fy"].reindex(out.index)
    out["octave_method"] = METHOD
    out["octave_trace"] = o["trace"].reindex(out.index)
    out["octave_runtime_s"] = o["runtime_s"].reindex(out.index)
    out.reset_index().to_csv(os.path.join(HERE, "predictions_octave_150.csv"), index=False)
    # wide file for label_day EXTRA_SOURCES / score150 --extra-wide: <key>__v = VERTICAL yarns = our h
    wide = pd.DataFrame({"image_id": df["id"], f"{METHOD}__v": df["h_count"], f"{METHOD}__h": df["v_count"],
                         f"{METHOD}__source": np.where((df.octave_fx != 1) | (df.octave_fy != 1), "octave", "frozen"),
                         f"{METHOD}__weave": df["weave"]})
    wide.to_csv(os.path.join(HERE, "octave_v1_150_wide.csv"), index=False)
    print(f"wrote predictions_octave_150.csv + octave_v1_150_wide.csv in {time.time() - t0:.0f}s; "
          f"changed axes: x {int((df.octave_fx != 1).sum())}, y {int((df.octave_fy != 1).sum())}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run150"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "run150":
        run150(a.workers)
