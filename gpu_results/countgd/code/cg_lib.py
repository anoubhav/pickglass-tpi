"""Shared helpers for the CountGD thread-counting baseline (WVAQ 2027 fabric project).

Everything here is deterministic and label-free.  The ground-truth parser, the legacy
window-trimming function and the metric definitions are NOT re-implemented: they are
extracted verbatim with `ast` from the Mac-side harness files that were copied to
data/code_ref/ (run_pipeline.py, vlm/data_vlm.py, vlm/evaluate_vlm.py), exactly like
vlm/gt_legacy.py does, so CountGD numbers are computed with the same code as the VLM rows.

Axis convention (shared with vlm/data_vlm.py and the CNN code):
    vertical_yarns   = yarns crossed walking HORIZONTALLY across the 1-inch window
                       (= yarns running top-to-bottom; counted in the HORIZONTAL strip)
    horizontal_yarns = yarns crossed walking VERTICALLY (counted in the VERTICAL strip)
"""
from __future__ import annotations

import ast
import datetime as _dt
import os
import re

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # gpu_results/countgd (or its GPU-server copy)
DATA = os.path.join(ROOT, "data")
CODE_REF = os.path.join(DATA, "code_ref")
LEGACY_RAW = os.path.join(DATA, "legacy_raw")
CROPS150 = os.path.join(DATA, "crops150")
XLSX = os.path.join(DATA, "Warp-Weft count.xlsx")
LEGACY_NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
WEAVE_NAME = {"1/1": "plain", "2/1": "twill", "1/2": "twill"}


# ----------------------------------------------------------------------------
# verbatim function extraction
# ----------------------------------------------------------------------------
def _extract(path, names, ns):
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    found = {n.name for n in keep}
    assert found == set(names), f"{path}: missing {set(names) - found}"
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


_GT_NS = None


def legacy_gt(usable_only=True):
    """Same output as vlm/gt_legacy.legacy_gt(): GT from the legacy spreadsheet, parser
    taken verbatim from pipeline/run_pipeline.py (load_gt, _parse_count, _parse_weave)."""
    global _GT_NS
    if _GT_NS is None:
        import openpyxl
        _GT_NS = _extract(os.path.join(CODE_REF, "run_pipeline.py"),
                          ("_parse_count", "_parse_weave", "load_gt"),
                          {"re": re, "os": os, "openpyxl": openpyxl, "_dt": _dt})
    gt, _ = _GT_NS["load_gt"](XLSX)
    out = {}
    for k, g in gt.items():
        if usable_only and not g["usable"]:
            continue
        g = dict(g)
        g["weave_name"] = WEAVE_NAME.get(g["weave"], g["weave"] or None)
        g["is_range"] = (g["weft_lo"] != g["weft_hi"]) or (g["warp_lo"] != g["warp_hi"])
        out[k] = g
    return out


_EV_NS = None


def eval_fns():
    """errors / metrics / loo_mean_baseline / weave_acc / write_csv from vlm/evaluate_vlm.py."""
    global _EV_NS
    if _EV_NS is None:
        _EV_NS = _extract(os.path.join(CODE_REF, "evaluate_vlm.py"),
                          ("errors", "metrics", "loo_mean_baseline", "weave_acc", "write_csv"),
                          {"np": np, "os": os, "math": __import__("math")})
    return _EV_NS


_DV_NS = None


def legacy_window_crop(name):
    """Identical to vlm/data_vlm.legacy_window_crop (detect_legacy_window copied verbatim):
    trim the dark pick-glass frame, resize the remainder to a square of side
    round((h+w)/2) with INTER_AREA.  Returns (crop_bgr, box, (h, w))."""
    global _DV_NS
    if _DV_NS is None:
        _DV_NS = _extract(os.path.join(CODE_REF, "data_vlm.py"), ("detect_legacy_window",),
                          {"cv2": cv2, "np": np})
    p = os.path.join(LEGACY_RAW, f"{name}.jpg")
    if not os.path.exists(p):
        p = os.path.join(LEGACY_RAW, f"{name} .jpg")       # 'r6 .jpg' has a space
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    x0, y0, x1, y1 = _DV_NS["detect_legacy_window"](img)
    crop = img[y0:y1, x0:x1]
    S = int(round((crop.shape[0] + crop.shape[1]) / 2))
    crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, x1, y1), img.shape[:2]


def load_image(set_name, image_id):
    """-> square BGR crop that spans exactly the 1-inch window."""
    if set_name == "legacy":
        return legacy_window_crop(image_id)[0]
    if set_name == "rect150":
        return cv2.imread(os.path.join(CROPS150, f"{int(image_id)}.jpg"), cv2.IMREAD_COLOR)
    raise ValueError(set_name)


# ----------------------------------------------------------------------------
# label-free periodicity (used for strip geometry and automatic exemplars)
# ----------------------------------------------------------------------------
TPI_MIN, TPI_MAX = 8.0, 250.0


def acf_period(profile, lag_lo, lag_hi):
    """Dominant period (px) of a 1-D profile from its windowed, bias-corrected autocorrelation.

    The highest ACF local maximum in [lag_lo, lag_hi] is taken, then replaced by a
    sub-multiple L/2 or L/3 if that is also a local maximum with ACF >= 0.6 x ACF(L)
    (guards against locking onto a harmonic of the yarn spacing).  Parabolic refinement.
    Returns (period_px, acf_value_at_period)."""
    x = np.asarray(profile, float)
    n = len(x)
    x = x - x.mean()
    w = np.hanning(n)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x * w, m)
    ac = np.fft.irfft(f * np.conj(f), m)[:n]
    fw = np.fft.rfft(w, m)
    acw = np.fft.irfft(fw * np.conj(fw), m)[:n]
    ac = ac / np.maximum(acw, 1e-9)
    ac = ac / max(ac[0], 1e-12)
    lo = max(2, int(np.floor(lag_lo)))
    hi = min(int(np.ceil(lag_hi)), n // 3)
    if hi <= lo + 2:
        return float("nan"), float("nan")
    idx = np.arange(lo, hi + 1)
    seg = ac[lo:hi + 1]
    pk = [int(idx[i]) for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not pk:
        L = int(idx[np.argmax(seg)])
    else:
        L = max(pk, key=lambda q: ac[q])
        for k in (3, 2):
            t = L / k
            if t < lo:
                continue
            cand = [q for q in pk if abs(q - t) <= max(1.5, 0.12 * t)]
            if cand:
                c = max(cand, key=lambda q: ac[q])
                if ac[c] >= 0.6 * ac[L]:
                    L = c
                    break
    if 1 <= L < n - 1:
        a, b, c = ac[L - 1], ac[L], ac[L + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den < 0 else 0.0
        off = float(np.clip(off, -0.5, 0.5))
    else:
        off = 0.0
    return float(L + off), float(ac[L])


def estimate_periods(img_bgr):
    """Spacing (px) of the vertical yarns along x (p_x) and of the horizontal yarns along y
    (p_y), each from the mean profile of the central 1/4-inch band (perpendicular to the
    profile), detrended with a Gaussian of sigma S/16, lag range = TPI in [8, 250]."""
    S = img_bgr.shape[0]
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    a, b = int(round(S * 3 / 8)), int(round(S * 5 / 8))
    col = g[a:b, :].mean(0)
    row = g[:, a:b].mean(1)
    out = {}
    for key, prof in (("x", col), ("y", row)):
        hp = prof - gaussian_filter1d(prof, S / 16.0, mode="reflect")
        p, v = acf_period(hp, S / TPI_MAX, S / TPI_MIN)
        out[f"p_{key}"], out[f"acf_{key}"] = p, v
    return out


def profile_maxima(gray_strip_along, period):
    """Positions (px along the strip) of yarn centres = local maxima of the across-averaged,
    detrended and lightly smoothed intensity profile, at least 0.6 period apart."""
    prof = gray_strip_along.mean(0) if gray_strip_along.ndim == 2 else gray_strip_along
    prof = prof - gaussian_filter1d(prof, 2.0 * period, mode="reflect")
    prof = gaussian_filter1d(prof, max(period / 6.0, 0.5), mode="reflect")
    cand = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1]]
    cand.sort(key=lambda i: -prof[i])
    keep = []
    for i in cand:
        if all(abs(i - j) >= 0.6 * period for j in keep):
            keep.append(i)
    return sorted(keep), prof


def pick_exemplar_centres(maxima, length, period, fracs=(0.25, 0.5, 0.75), margin=0.6):
    """3 distinct yarn-centre maxima closest to the given fractions of the strip length,
    kept at least `margin` x period from the ends.  Falls back to the fractions themselves."""
    ok = [m for m in maxima if margin * period <= m <= length - margin * period]
    chosen = []
    for f in fracs:
        t = f * length
        pool = [m for m in ok if m not in chosen]
        chosen.append(min(pool, key=lambda m: abs(m - t)) if pool else t)
    return [float(c) for c in chosen]


def tile_starts(length, win, overlap):
    """Official CountGD crop() placement along one axis: windows of size `win` advancing by
    win - overlap, the last one shifted back to end exactly at `length`."""
    win = int(min(win, length))
    starts, s = [], 0
    step = max(1, int(win - overlap))
    while True:
        if s + win >= length:
            starts.append(max(0, length - win))
            break
        starts.append(s)
        s += step
    return sorted(set(starts)), win


def owner_index(coord, starts, win):
    """Voronoi ownership: the window whose centre is nearest to `coord` owns it."""
    centres = np.array(starts, float) + win / 2.0
    return int(np.argmin(np.abs(centres - coord)))
