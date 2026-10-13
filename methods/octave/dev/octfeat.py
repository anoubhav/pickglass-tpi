"""Octave features (label-free) for a hypothesised yarn lattice.  Development version."""
import os  # added by the release export (path configuration)
import math, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, f"{ROOT}/methods/results")
import numpy as np, cv2
from scipy import ndimage
import counting_frozen_final as C


class Ctx:
    """Pre-processed image + spectrum + background-normalised power."""

    def __init__(self, crop, max_work=1024):
        g, w, N, info = C.preprocess(crop, max_work=max_work)
        self.g, self.w, self.N, self.info = g, w, N, info
        self.sp = C.Spectrum(g)
        sp = self.sp
        c = sp.c
        nr = c
        pol = cv2.warpPolar(sp.logP.astype(np.float32), (nr, 360), (float(c), float(c)), float(c),
                            cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR)
        med = np.median(pol, axis=0).astype(np.float64)
        med = ndimage.median_filter(med, size=5, mode="nearest")
        bg = np.interp(sp.r, np.arange(nr) * (c / nr), med)
        self.Q = np.exp(sp.logP - bg)            # power / radial-median background
        self.P = np.exp(sp.logP)
        self.Pbg = np.exp(bg)


def basis_from(h, v, ax_deg=0.0, ay_deg=0.0):
    return np.array([[h, v * math.tan(math.radians(ay_deg))],
                     [h * math.tan(math.radians(ax_deg)), v]], float)


def snap(ctx, B, rel=0.06, ang=15.0):
    """Move each basis vector to the strongest whitened peak within +-rel radius / +-ang deg
    of the axis-aligned target (truth-hypothesis construction for dev evaluation)."""
    sp = ctx.sp
    out = B.copy()
    for k in (0, 1):
        t = B[:, k]
        rr = float(np.hypot(*t))
        th0 = 0.0 if k == 0 else 90.0
        sel = (np.abs(sp.r / rr - 1) <= rel)
        ang_map = np.degrees(np.arctan2(sp.V, sp.U))
        d = np.abs(((ang_map - th0 + 90) % 180) - 90)
        sel &= d <= ang
        if k == 0:
            sel &= sp.U > 0
        else:
            sel &= sp.V > 0
        if not sel.any():
            continue
        Zs = np.where(sel, sp.Z, -np.inf)
        y, x = np.unravel_index(np.argmax(Zs), Zs.shape)
        if sp.Z[y, x] < 3.0:
            continue
        u, v = sp.centroid(sp.U[y, x], sp.V[y, x])
        out[:, k] = (u, v)
    return out


def rfeat(ctx, B, axis, weight="snr", thr=3.0, c1min=0.25, c1max=4.5, c2max=3.5):
    """Self-similarity of the x- (axis 0) or y- (axis 1) yarn structure under fractional
    lattice translations, from the background-normalised power spectrum.
    Returns dict of R values in [-1, 1]."""
    sp = ctx.sp
    lo, hi = sp.band
    Binv = np.linalg.inv(B)
    K = np.stack([sp.U.ravel(), sp.V.ravel()], 0)
    Cc = Binv @ K
    c1, c2 = (Cc[0], Cc[1]) if axis == 0 else (Cc[1], Cc[0])
    r = sp.r.ravel()
    m = (np.abs(c1) >= c1min) & (np.abs(c1) <= c1max) & (np.abs(c2) <= c2max) & (r >= 5) & (r <= hi)
    if weight == "snr":
        wv = np.clip(ctx.Q.ravel() - thr, 0, None)
    elif weight == "pow":
        wv = np.clip(ctx.P.ravel() - thr * ctx.Pbg.ravel(), 0, None)
    else:
        wv = ctx.P.ravel()
    wv = wv * m
    tot = wv.sum() + 1e-12
    f = lambda ph: float((wv * np.cos(ph)).sum() / tot)
    return {"d0": f(np.pi * c1), "dc1": f(np.pi * c1 + 2 * np.pi * c2), "dc2": f(np.pi * (c1 + c2)),
            "h0": f(2 * np.pi * c1), "hp": f(2 * np.pi * (c1 + c2)), "hm": f(2 * np.pi * (c1 - c2)),
            "q0": f(4 * np.pi * c1), "wtot": float(tot)}


def gvals(ctx, B, axis):
    """Whitened z (minus baseline) at characteristic points, for axis 0 (a) or 1 (b)."""
    a, b = (B[:, 0], B[:, 1]) if axis == 0 else (B[:, 1], B[:, 0])
    pts = {"a": a, "2a": 2 * a, "a/2": a / 2, "a+b": a + b, "a-b": a - b, "2a+b": 2 * a + b, "2a-b": 2 * a - b,
           "a+b/2": a + b / 2, "a-b/2": a - b / 2, "a/2+b/2": a / 2 + b / 2, "a/2-b/2": a / 2 - b / 2}
    U = np.array([p[0] for p in pts.values()]); V = np.array([p[1] for p in pts.values()])
    g, r = ctx.sp.lookup(U, V)
    lo, hi = ctx.sp.band
    g = np.where((r > hi) | (r < 5), np.nan, g)
    return dict(zip(pts.keys(), g.tolist()))


def Rsym(ctx, B, s, need=(0,), weight="snr", thr=3.0, cmin=0.25, cmax=4.5, triv=0.125):
    """Symmetry score of a real-space shift s = s1*t1 + s2*t2 (lattice coords of the hypothesis
    B): R = sum_k E_k cos(2 pi k.s) / sum_k E_k over the informative spectral components:
    |c_i| >= cmin for every axis i in `need` (the yarn family under test), all |c_i| <= cmax,
    and |k.s| >= triv (components perpendicular to s are trivially invariant)."""
    sp = ctx.sp
    lo, hi = sp.band
    Binv = np.linalg.inv(B)
    Cc = Binv @ np.stack([sp.U.ravel(), sp.V.ravel()], 0)
    r = sp.r.ravel()
    ks = Cc[0] * s[0] + Cc[1] * s[1]
    m = (r >= 5) & (r <= hi) & (np.abs(Cc[0]) <= cmax) & (np.abs(Cc[1]) <= cmax) & (np.abs(ks) >= triv)
    for i in need:
        m &= np.abs(Cc[i]) >= cmin
    if weight == "snr":
        wv = np.clip(ctx.Q.ravel() - thr, 0, None)
    else:
        wv = np.clip(ctx.P.ravel() - thr * ctx.Pbg.ravel(), 0, None)
    wv = wv * m
    tot = wv.sum()
    if tot <= 0:
        return np.nan
    return float((wv * np.cos(2 * np.pi * ks)).sum() / tot)


SHIFTS = {
    # doubling tests (is a half-period translation a symmetry?)
    "Dx0": ((0.5, 0.0), (0,)), "Dxp": ((0.5, 1.0), (0,)), "Dxm": ((0.5, -1.0), (0,)),
    "Dy0": ((0.0, 0.5), (1,)), "Dyp": ((1.0, 0.5), (1,)), "Dym": ((-1.0, 0.5), (1,)),
    "Cp": ((0.5, 0.5), ()), "Cm": ((0.5, -0.5), ()),
    "Cpx": ((0.5, 0.5), (0,)), "Cmx": ((0.5, -0.5), (0,)), "Cpy": ((0.5, 0.5), (1,)), "Cmy": ((0.5, -0.5), (1,)),
    # halving tests (is the hypothesised period a symmetry at all?)
    "Hx0": ((1.0, 0.0), (0,)), "Hxp": ((1.0, 1.0), (0,)), "Hxm": ((1.0, -1.0), (0,)),
    "Hy0": ((0.0, 1.0), (1,)), "Hyp": ((1.0, 1.0), (1,)), "Hym": ((-1.0, 1.0), (1,)),
}


def symfeats(ctx, B, weight="snr"):
    return {k: Rsym(ctx, B, s, need, weight=weight) for k, (s, need) in SHIFTS.items()}


# ----------------------------------------------------------------------- folded-ridge test
from scipy import signal as _sig


def band_profiles(g, w, axis, angle_deg, n_bands=8):
    """Deskewed per-band projections along the yarn direction (frozen _shear convention)."""
    N = g.shape[0]
    gi, wi = C._shear(g, w, math.tan(math.radians(angle_deg)), axis)
    edges = np.linspace(0, N, n_bands + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if axis == "x":
            num, den = gi[a:b].sum(0), wi[a:b].sum(0)
        else:
            num, den = gi[:, a:b].sum(1), wi[:, a:b].sum(1)
        p = num / np.maximum(den, 1e-3)
        ok = den >= 0.5 * (b - a)
        out.append((p.astype(np.float64), ok))
    return out


def folded_shape(profiles, f_cyc, K=3, win_periods=6, align=1):
    """Mean period shape (harmonics 1..K of frequency f_cyc cycles per N px), phase-aligned per
    window on harmonic `align`.  Returns complex mean coefficients (K,) and #windows."""
    N = len(profiles[0][0])
    T = N / f_cyc
    L = int(round(win_periods * T))
    if L < 8 or L > N:
        return None, 0
    hop = max(1, L // 2)
    x = np.arange(L)
    win = np.hanning(L)
    E = np.exp(-2j * np.pi * np.outer(np.arange(1, K + 1), x) / T)  # (K, L)
    acc = np.zeros(K, complex)
    n = 0
    for p, ok in profiles:
        for s in range(0, N - L + 1, hop):
            if ok[s:s + L].mean() < 0.9:
                continue
            seg = p[s:s + L] - p[s:s + L].mean()
            c = (E * (seg * win)).sum(1)
            ref = np.angle(c[align - 1]) / align
            if align == 2:  # resolve the pi ambiguity: hump at 0 is the larger one
                c1r = c[0] * np.exp(-1j * ref)
                if c1r.real < 0:
                    ref += np.pi
            ck = c * np.exp(-1j * np.arange(1, K + 1) * ref)
            acc += ck / (np.abs(c).sum() + 1e-12)
            n += 1
    if n == 0:
        return None, 0
    return acc / n, n


def ridge_stats(cm, n_phi=360):
    """Local maxima of the synthesised period shape: (n_sig, r2, pos2) where r2 = prominence of
    the 2nd-largest maximum / prominence of the largest, pos2 = its distance from the main
    maximum in periods (0..0.5)."""
    if cm is None:
        return np.nan, np.nan
    K = len(cm)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    G = np.real(sum(cm[k] * np.exp(1j * (k + 1) * ph) for k in range(K)))
    G3 = np.r_[G, G, G]
    pk, pr = _sig.find_peaks(G3, prominence=0)
    sel = (pk >= n_phi) & (pk < 2 * n_phi)
    pk, prom = pk[sel] - n_phi, pr["prominences"][sel]
    if len(pk) == 0:
        return np.nan, np.nan
    o = np.argsort(-prom)
    if len(pk) < 2:
        return 0.0, np.nan
    d = abs(pk[o[1]] - pk[o[0]]) / n_phi
    d = min(d, 1 - d)
    return float(prom[o[1]] / prom[o[0]]), float(d)


def proj_power(g, w, axis, angle_deg, f_cyc):
    N = g.shape[0]
    p = C._project(g, w, math.tan(math.radians(angle_deg)), axis)
    x = np.arange(N)
    q = (p - p.mean()) * np.hanning(N)
    return float(np.abs((q * np.exp(-2j * np.pi * f_cyc * x / N)).sum()) ** 2)


def best_shear(g, w, axis, angle_deg, f_cyc):
    cands = sorted({round(angle_deg, 3), round(-angle_deg, 3)})
    return max(cands, key=lambda a: proj_power(g, w, axis, a, f_cyc))


def ridge_test(ctx, count, axis, angle_deg, K=3, n_bands=8, win_periods=6):
    """Secondary-ridge statistics for yarn period N/count: (r2_at_f, d_at_f, r2_at_f/2, d_at_f/2)."""
    g, w = ctx.g, ctx.w
    a = best_shear(g, w, axis, angle_deg, count)
    prof = band_profiles(g, w, axis, a, n_bands)
    cm, n = folded_shape(prof, count, K=K, win_periods=win_periods, align=1)
    r2, d2 = ridge_stats(cm)
    cmh, nh = folded_shape(prof, count / 2, K=2 * K, win_periods=win_periods / 2, align=2)
    r2h, d2h = ridge_stats(cmh)
    return {"r2": r2, "d2": d2, "r2h": r2h, "d2h": d2h, "n": n, "shear": a,
            "c": None if cm is None else np.abs(cm).round(3).tolist(),
            "ch": None if cmh is None else np.abs(cmh).round(3).tolist()}
