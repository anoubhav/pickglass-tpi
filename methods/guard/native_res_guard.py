"""native_res_guard.py -- native-resolution guard around the FROZEN thread counter.

Problem (genai_sr README, deviation D1; cascade 'resolvability' router):
  The frozen counter ("ours", methods/results/counting_frozen_final.py) whitens the power spectrum
  by its radial median.  When a crop has been up-sampled (bicubic / bilinear / Lanczos / SR, digital
  zoom) or low-passed after quantisation (defocus applied to a stored image), the band above the
  true signal bandwidth is EMPTY: 8-bit re-quantisation noise (1/12 grey level^2) plus the
  kernel-attenuated imaging replicas f_s - f0 of the yarn peaks (or quantisation products of the
  blurred lattice).  Whitening turns those weak structures into huge z-scores and the lattice prior
  ("the finest periodicity is the yarn") prefers them: x8 + bicubic gives MAPE 301.7 % (legacy-13)
  and 239.7 % (R-HR) instead of 7.6 % / 15.8 % when the low-resolution crop is counted as is.

Guard (the frozen counter is imported, never modified):
  1. Effective bandwidth (estimate_bandwidth).  For each image axis, the row (column) power spectrum
     is averaged over all rows (columns), Hann-windowed and normalised so that white noise of
     variance s^2 has level s^2 (8-bit re-quantisation: log10(1/12) = -1.08).  Its FLOOR curve L(f)
     is the 30th percentile of log10 P in 1/30-decade log-spaced bands (ignores the narrow yarn
     lines).  A floor below L_dead = log10(1/12) + dead_delta is "empty".  Per axis:
       f_d   lower edge of the empty region: the lowest band from which at most live_eps of the
             (log-f weighted) floor curve is above L_dead (replica bumps tolerated);
       depth L_ref - L_dead, L_ref = median floor in [ref_win] * f_d (the live floor below it);
       f_c   highest frequency below f_d where the floor still exceeds L_dead + mid_frac * depth
             (the fall of the interpolation kernel / defocus MTF into the empty band).
     The crop has a dead band when depth >= depth_min on both axes; then n_eff = 2 f_c (the side,
     in px per crop side, of the sampling grid the content supports; the axis with the larger
     relative bandwidth decides).  Otherwise n_eff = S (no evidence of up-sampling).
  2. Native-resolution re-count (option b), with a consistency gate.  The unguarded frozen estimate
     is computed first.  The guard FIRES only if
       (i)  there is a dead band and the native working side t = kappa * n_eff is materially below
            the counter's own working side, t < (1 - fire_tol) * min(S, 1024), and
       (ii) the unguarded count lies OUTSIDE the band the counter would search at side t,
            max(h, v) > min(320, 0.42 t) cycles per crop (the frozen counter's own band rule):
            the claimed periodicity is finer than the content supports.
     When it fires, the result is the frozen estimate with max_work = t (the counter then works on
     an INTER_AREA down-sampled image whose candidate band ends at the content's bandwidth, so the
     empty band and its replicas are never searched).  When it does not fire, the result is the
     unguarded estimate, bit-identical.
  3. Flags: 'upsampled' (dead band and f_c < (1 - fire_tol) of the grid's Nyquist frequency: the pixel
     grid is finer than the content), 'insufficient_resolution' (px_per_yarn = n_eff / max(h, v) <
     ppy_min, the optics-study deployment rule of ~4-5 effective px per yarn), 'unsupported_claim'
     (gate (ii) was true: the unguarded count was beyond the content's bandwidth), and the margin
     px_per_yarn / ppy_min.

All parameters were chosen on SYNTHETIC data only (synth_dev.py + tune_synth.py -> guard_params.json;
see RESULTS.md); the real crops were scored once afterwards (eval_real.py).

API
    count_guarded(crop_bgr, params=None, **estimate_kw) -> dict
        the frozen estimate() dict ('h_count', 'v_count', 'weave', 'confidence', 'debug') plus
        'guard': {fired, flag, upsampled, insufficient_resolution, unsupported_claim, n_eff, f_c, S,
                  N_work_unguarded, N_work_guarded, target, px_per_yarn, px_per_yarn_margin, ppy_min,
                  depth, dead_band, unguarded_h, unguarded_v, ...}
        and top-level 'guard_fired', 'guard_flag', 'px_per_yarn', 'px_per_yarn_margin'.
    estimate_bandwidth(crop_bgr, params=None) -> dict   (the analysis alone, ~0.1-0.3 s)
    frozen_counter() -> the imported frozen module
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import time
import warnings

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.dirname(HERE)
FROZEN_PATH = os.path.join(METHODS, "results", "counting_frozen_final.py")
CONF_MODEL_PATH = os.path.join(METHODS, "conf_model.json")
PARAMS_PATH = os.path.join(HERE, "guard_params.json")

Q_LOG = math.log10(1.0 / 12.0)  # 8-bit re-quantisation noise level (variance 1/12 grey level^2)

# Defaults = the values frozen by tune_synth.py; guard_params.json (same values) overrides them.
DEFAULTS = dict(
    n_bands_per_decade=30,   # log-spaced bands of the floor curve            (fixed a priori)
    floor_pct=30.0,          # percentile of log10 P per band                  (fixed a priori)
    f_lo=8.0,                # lowest analysed frequency, cycles / crop side   (fixed a priori)
    dead_delta=0.8,          # L_dead = log10(1/12) + dead_delta   [decades]   (locator, synthetic)
    live_eps=0.5,            # tolerated live fraction of the empty region     (locator, synthetic)
    ref_win=(0.3, 0.5),      # live reference floor: bands in [0.3, 0.5] * f_d (locator, synthetic)
    mid_frac=0.3,            # f_c where the floor falls to L_dead + mid_frac * depth (locator, synthetic)
    depth_min=1.1,           # dead band needs depth >= depth_min [decades]    (counting, synthetic)
    kappa=0.9,               # native working side t = kappa * n_eff           (counting, synthetic)
    fire_tol=0.3,            # eligible iff t < (1 - fire_tol) * N_work        (counting, synthetic)
    n_min=64,                # never work below 64 px                          (fixed a priori)
    ppy_min=5.0,             # flag below this many effective px per yarn      (rule 4-5; synthetic)
    max_work=1024,           # the frozen counter's default working side
)

_PARAMS_CACHE: dict = {}


def load_params(path: str | None = None) -> dict:
    path = PARAMS_PATH if path is None else path
    key = (path, os.path.getmtime(path) if path and os.path.exists(path) else None)
    if key in _PARAMS_CACHE:
        return dict(_PARAMS_CACHE[key])
    p = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        p.update(d.get("params", d))
    p["ref_win"] = tuple(p["ref_win"])
    _PARAMS_CACHE[key] = dict(p)
    return p


_C = None


def frozen_counter():
    """Import methods/results/counting_frozen_final.py read-only, as module 'counting_frozen_final'.

    The frozen file looks for conf_model.json next to itself (methods/results/), where it does not
    exist; every study (genai_sr, cascade, legacy) used methods/conf_model.json, so that model is
    injected into the imported module's CONF_MODEL global (the file itself is not touched)."""
    global _C
    if _C is None:
        spec = importlib.util.spec_from_file_location("counting_frozen_final", FROZEN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with open(CONF_MODEL_PATH) as f:
            mod.CONF_MODEL = json.load(f)
        _C = mod
    return _C


def counter_band_top(n_work: int) -> float:
    """Upper end of the frozen counter's candidate band at working side n_work (its _band rule)."""
    C = frozen_counter()
    return float(C._band(int(n_work))[1])


# ------------------------------------------------------------------------------------ analysis
def _gray(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr.ndim == 2:
        return crop_bgr.astype(np.float64)
    if crop_bgr.shape[2] == 4:
        crop_bgr = crop_bgr[..., :3]
    if crop_bgr.dtype == np.uint8:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return crop_bgr[..., :3].astype(np.float64) @ np.array([0.114, 0.587, 0.299])


def axis_spectrum(g: np.ndarray) -> np.ndarray:
    """Mean power spectrum of the rows of g (FFT along axis 1), Hann window, normalised so that
    white noise of variance s^2 has expected level s^2.  Index = cycles per row length."""
    n = g.shape[1]
    w = np.hanning(n)
    X = (g - g.mean(axis=1, keepdims=True)) * w
    return (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(axis=0) / float((w ** 2).sum())


def floor_curve(P: np.ndarray, f_lo: float, per_decade: int, pct: float):
    """Low-percentile log10 level of P in log-spaced bands -> (centres, lo edges, hi edges, L, weights)."""
    n = len(P)
    f_hi = n - 1
    if f_hi <= f_lo * 1.5:
        return (np.zeros(0),) * 5
    nb = max(8, int(round(per_decade * math.log10(f_hi / f_lo))))
    edges = np.unique(np.round(np.geomspace(f_lo, f_hi, nb + 1)).astype(int))
    lp = np.log10(P + 1e-12)
    cen, lo, hi, L = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        cen.append(math.sqrt(a * b))
        lo.append(a)
        hi.append(b)
        L.append(float(np.percentile(lp[a:b], pct)))
    cen, lo, hi, L = (np.asarray(x, float) for x in (cen, lo, hi, L))
    wts = np.log(hi / lo)
    return cen, lo, hi, L, wts


def _cutoff(cen, lo, hi, L, wts, p, n_axis):
    """Effective cut-off (cycles per axis length) of one axis; n_axis/2 if there is no dead band.

    1. empty region: the lowest band b such that at most live_eps of the (log-f weighted) floor
       curve from b to the end is above L_dead (replica bumps of strong yarn peaks are tolerated),
       moved up to the first band that is itself below L_dead -> its lower edge f_d;
    2. live reference L_ref = median floor in [ref_win[0], ref_win[1]] * f_d; depth = L_ref - L_dead;
    3. cut-off f_c = the highest frequency below f_d where the floor is still >= L_dead +
       mid_frac * depth (log-f interpolated).
    The axis is 'dead' (eligible for the guard) when depth >= depth_min."""
    out = dict(f_c=n_axis / 2.0, dead=False, depth=float("nan"), f_d=float("nan"), live_above=float("nan"),
               L_ref=float("nan"))
    if len(L) < 4:
        return out
    L_dead = Q_LOG + p["dead_delta"]
    live = (L > L_dead).astype(float)
    frac = np.cumsum((live * wts)[::-1])[::-1] / np.cumsum(wts[::-1])[::-1]
    ok = np.nonzero(frac <= p["live_eps"])[0]
    if len(ok) == 0:
        return out
    b = int(ok[0])
    while b < len(L) and live[b] > 0:
        b += 1
    if b >= len(L):
        return out
    f_d = float(lo[b])
    r0, r1 = p["ref_win"]
    sel = (cen >= r0 * f_d) & (cen <= r1 * f_d)
    if not sel.any():
        return out
    L_ref = float(np.median(L[sel]))
    depth = L_ref - L_dead
    out.update(depth=depth, f_d=f_d, live_above=float(frac[b]), L_ref=L_ref)
    L_mid = L_dead + p["mid_frac"] * depth
    idx = np.nonzero((cen < f_d) & (L >= L_mid))[0]
    if depth <= 0 or len(idx) == 0:
        return out
    i = int(idx[-1])
    if i + 1 < len(L) and L[i + 1] < L_mid:
        t = (L[i] - L_mid) / (L[i] - L[i + 1])
        f_c = float(math.exp(math.log(cen[i]) + t * (math.log(cen[i + 1]) - math.log(cen[i]))))
    else:
        f_c = float(cen[i])
    out["f_c_cand"] = f_c
    if depth >= p["depth_min"]:
        out.update(f_c=min(f_c, n_axis / 2.0), dead=True)
    return out


def combine_axes(axes: dict, S: int) -> dict:
    fr = max(axes["x"]["frac_nyq"], axes["y"]["frac_nyq"])
    dep = [axes[k]["depth"] for k in ("x", "y")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        depth = float(np.nanmin(dep)) if np.isfinite(dep).any() else float("nan")
    return dict(frac_nyq=float(fr), f_c=float(fr * S / 2.0), n_eff=float(fr * S),
                dead_band=bool(axes["x"]["dead"] and axes["y"]["dead"]), depth=depth)


def estimate_bandwidth(crop_bgr: np.ndarray, params: dict | None = None, keep_curves: bool = False) -> dict:
    """Effective bandwidth of a crop; frequencies in cycles per crop side, n_eff in px per side."""
    p = load_params() if params is None else {**DEFAULTS, **params}
    t0 = time.time()
    g = _gray(crop_bgr)
    H, W = g.shape
    S = int(max(H, W))
    axes = {}
    for ax, gg, n in (("x", g, W), ("y", g.T, H)):
        P = axis_spectrum(gg)
        cen, lo, hi, L, wts = floor_curve(P, p["f_lo"], p["n_bands_per_decade"], p["floor_pct"])
        c = _cutoff(cen, lo, hi, L, wts, p, n)
        c["n_axis"] = int(n)
        c["frac_nyq"] = float(c["f_c"] / (n / 2.0))
        if keep_curves:
            c["curve"] = (cen.tolist(), L.tolist())
        axes[ax] = c
    res = {"S": S, "H": int(H), "W": int(W), "axes": axes, **combine_axes(axes, S),
           "seconds": round(time.time() - t0, 3)}
    return res


# ------------------------------------------------------------------------------------- wrapper
def guard_decision(bw: dict, p: dict, max_work: int | None = None) -> dict:
    """Eligibility (step 2 (i)) and flags that do not depend on the counts."""
    S = bw["S"]
    mw = int(p["max_work"] if max_work is None else max_work)
    N_work = int(min(S, mw))
    n_eff = float(bw["n_eff"]) if bw["dead_band"] else float(S)
    target = int(max(p["n_min"], round(p["kappa"] * n_eff)))
    eligible = bool(bw["dead_band"] and target < (1.0 - p["fire_tol"]) * N_work)
    upsampled = bool(bw["dead_band"] and bw["frac_nyq"] < (1.0 - p["fire_tol"]))
    return dict(eligible=eligible, upsampled=upsampled, n_eff=n_eff, f_c=n_eff / 2.0, S=int(S),
                N_work_unguarded=N_work, target=target, frac_nyq=float(bw["frac_nyq"]), depth=float(bw["depth"]),
                dead_band=bool(bw["dead_band"]))


def claim_outside_band(h: float, v: float, target: int) -> bool:
    """Gate (ii): the unguarded count is outside the band the counter searches at side target
    (a missing count also counts as unsupported)."""
    c = np.nanmax([h, v]) if np.isfinite([h, v]).any() else np.nan
    return bool(not (np.isfinite(c) and c <= counter_band_top(target)))


def count_guarded(crop_bgr: np.ndarray, params: dict | None = None, method: str = "ours",
                  _unguarded: dict | None = None, _bandwidth: dict | None = None, **estimate_kw) -> dict:
    """Frozen-counter estimate with the native-resolution guard (see module docstring).

    _unguarded / _bandwidth: optional precomputed unguarded estimate / bandwidth analysis of the SAME
    crop with the same estimate_kw (evaluation speed-up)."""
    p = load_params() if params is None else {**DEFAULTS, **params}
    C = frozen_counter()
    t0 = time.time()
    bw = estimate_bandwidth(crop_bgr, p) if _bandwidth is None else _bandwidth
    dec = guard_decision(bw, p, estimate_kw.get("max_work"))
    ung = _unguarded if _unguarded is not None else C.estimate(crop_bgr, method=method, **estimate_kw)
    outside = claim_outside_band(ung["h_count"], ung["v_count"], dec["target"])
    fired = bool(dec["eligible"] and outside)
    if fired:
        kw = dict(estimate_kw)
        kw["max_work"] = dec["target"]
        out = C.estimate(crop_bgr, method=method, **kw)
    else:
        out = ung
    h, v = out["h_count"], out["v_count"]
    dens = np.nanmax([h, v]) if np.isfinite([h, v]).any() else np.nan
    ppy = float(dec["n_eff"] / dens) if np.isfinite(dens) and dens > 0 else float("nan")
    insufficient = bool(np.isfinite(ppy) and ppy < p["ppy_min"])
    unsupported = bool(dec["dead_band"] and outside)
    flags = [f for f, on in (("upsampled", dec["upsampled"]), ("unsupported_claim", unsupported),
                             ("insufficient_resolution", insufficient)) if on]
    g = dict(dec, fired=fired, unsupported_claim=unsupported, N_work_guarded=int(dec["target"] if fired else
                                                                                 dec["N_work_unguarded"]),
             unguarded_h=float(ung["h_count"]), unguarded_v=float(ung["v_count"]),
             unguarded_confidence=float(ung["confidence"]),
             px_per_yarn=ppy, ppy_min=float(p["ppy_min"]),
             px_per_yarn_margin=float(ppy / p["ppy_min"]) if np.isfinite(ppy) else float("nan"),
             insufficient_resolution=insufficient, flag="+".join(flags) if flags else "ok",
             bandwidth_axes={k: {kk: vv for kk, vv in a.items() if kk != "curve"} for k, a in bw["axes"].items()},
             analysis_s=bw.get("seconds"), total_s=round(time.time() - t0, 3))
    res = dict(out)
    res["guard"] = g
    res["guard_fired"] = g["fired"]
    res["guard_flag"] = g["flag"]
    res["px_per_yarn"] = g["px_per_yarn"]
    res["px_per_yarn_margin"] = g["px_per_yarn_margin"]
    return res


if __name__ == "__main__":
    import sys
    for pth in sys.argv[1:]:
        img = cv2.imread(pth, cv2.IMREAD_COLOR)
        r = count_guarded(img)
        gd = r["guard"]
        print(f"{pth}: h={r['h_count']:.1f} v={r['v_count']:.1f} conf={r['confidence']:.2f} "
              f"fired={gd['fired']} flag={gd['flag']} n_eff={gd['n_eff']:.0f}/{gd['S']} "
              f"N_work={gd['N_work_guarded']} px/yarn={gd['px_per_yarn']:.2f}")
