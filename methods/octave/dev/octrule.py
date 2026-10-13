"""Octave decision by lattice self-similarity (development version)."""
import math
import numpy as np
import octfeat as F

SH = {
    "Dx0": ((0.5, 0.0), 0), "Dxp": ((0.5, 1.0), 0), "Dxm": ((0.5, -1.0), 0),
    "Dy0": ((0.0, 0.5), 1), "Dyp": ((1.0, 0.5), 1), "Dym": ((-1.0, 0.5), 1),
    "Cp": ((0.5, 0.5), None), "Cm": ((0.5, -0.5), None),
    "Hx0": ((1.0, 0.0), 0), "Hxp": ((1.0, 1.0), 0), "Hxm": ((1.0, -1.0), 0),
    "Hy0": ((0.0, 1.0), 1), "Hyp": ((1.0, 1.0), 1), "Hym": ((-1.0, 1.0), 1),
}


class Feat:
    def __init__(self, ctx, thr=3.0, cmin=0.25, cmax=4.5, triv=0.125):
        self.ctx = ctx
        sp = ctx.sp
        lo, hi = sp.band
        self.K = np.stack([sp.U.ravel(), sp.V.ravel()], 0)
        r = sp.r.ravel()
        self.inband = (r >= 5) & (r <= hi)
        self.K = self.K[:, self.inband]
        self.W = {"snr": np.clip(ctx.Q.ravel()[self.inband] - thr, 0, None),
                  "pow": np.clip(ctx.P.ravel()[self.inband] - thr * ctx.Pbg.ravel()[self.inband], 0, None)}
        nz = (self.W["snr"] > 0) | (self.W["pow"] > 0)
        self.K = self.K[:, nz]
        self.W = {k: v[nz] for k, v in self.W.items()}
        self.cmin, self.cmax, self.triv = cmin, cmax, triv
        self.cache = {}

    def __call__(self, B, weight="snr"):
        key = (tuple(np.round(np.asarray(B).ravel(), 4)), weight)
        if key in self.cache:
            return self.cache[key]
        Cc = np.linalg.solve(B, self.K)
        base = (np.abs(Cc[0]) <= self.cmax) & (np.abs(Cc[1]) <= self.cmax)
        ax_ok = [np.abs(Cc[0]) >= self.cmin, np.abs(Cc[1]) >= self.cmin]
        wv = self.W[weight]
        out = {}
        for k, (s, need) in SH.items():
            ks = Cc[0] * s[0] + Cc[1] * s[1]
            m = base & (np.abs(ks) >= self.triv)
            if need is not None:
                m &= ax_ok[need]
            w = wv * m
            tot = w.sum()
            out[k] = float((w * np.cos(2 * np.pi * ks)).sum() / tot) if tot > 0 else np.nan
        out["Dx"] = np.nanmax([out["Dx0"], out["Dxp"], out["Dxm"]]) if np.isfinite([out["Dx0"], out["Dxp"], out["Dxm"]]).any() else np.nan
        out["Dy"] = np.nanmax([out["Dy0"], out["Dyp"], out["Dym"]]) if np.isfinite([out["Dy0"], out["Dyp"], out["Dym"]]).any() else np.nan
        out["C"] = np.nanmax([out["Cp"], out["Cm"]]) if np.isfinite([out["Cp"], out["Cm"]]).any() else np.nan
        out["Hx"] = np.nanmax([out["Hx0"], out["Hxp"], out["Hxm"]]) if np.isfinite([out["Hx0"], out["Hxp"], out["Hxm"]]).any() else np.nan
        out["Hy"] = np.nanmax([out["Hy0"], out["Hyp"], out["Hym"]]) if np.isfinite([out["Hy0"], out["Hyp"], out["Hym"]]).any() else np.nan
        self.cache[key] = out
        return out


def decide(feat, B0, tau_d=0.15, tau_h=None, weight="snr", max_iter=3, hi=None, use_C=True):
    """Greedy octave correction.  Returns (B, factors (fx, fy), trace)."""
    B = np.array(B0, float)
    fx = fy = 1.0
    trace = []
    for _ in range(max_iter):
        f = feat(B, weight)
        Sx = f["Dx"]; Sy = f["Dy"]
        Sxy = np.nanmax([f["C"] if use_C else np.nan, np.nanmin([Sx, Sy])]) if np.isfinite([Sx, Sy]).any() else np.nan
        cands = []
        if fx >= 1 and np.isfinite(Sx):
            cands.append((Sx, "x"))
        if fy >= 1 and np.isfinite(Sy):
            cands.append((Sy, "y"))
        if fx >= 1 and fy >= 1 and np.isfinite(Sxy):
            cands.append((Sxy, "xy"))
        # feasibility: doubled frequency must stay inside the analysed band
        ok = []
        for s, c in cands:
            M = np.diag([2.0 if "x" in c else 1.0, 2.0 if "y" in c else 1.0])
            Bn = B @ M
            if hi is not None and max(np.hypot(*Bn[:, 0]), np.hypot(*Bn[:, 1])) > hi:
                continue
            ok.append((s, c))
        if ok:
            s, c = max(ok)
            if s > tau_d:
                M = np.diag([2.0 if "x" in c else 1.0, 2.0 if "y" in c else 1.0])
                B = B @ M
                fx *= M[0, 0]; fy *= M[1, 1]
                trace.append(("D" + c, round(s, 3)))
                continue
        # halving
        if tau_h is not None:
            hx = f["Hx"] if fx == 1 else np.inf
            hy = f["Hy"] if fy == 1 else np.inf
            lo_ = 18.0
            cand = []
            if np.isfinite(hx) and hx < tau_h and np.hypot(*B[:, 0]) / 2 >= lo_:
                cand.append((hx, "x"))
            if np.isfinite(hy) and hy < tau_h and np.hypot(*B[:, 1]) / 2 >= lo_:
                cand.append((hy, "y"))
            if cand:
                s, c = min(cand)
                M = np.diag([0.5 if c == "x" else 1.0, 0.5 if c == "y" else 1.0])
                B = B @ M
                fx *= M[0, 0]; fy *= M[1, 1]
                trace.append(("H" + c, round(s, 3)))
                continue
        break
    return B, (fx, fy), trace
