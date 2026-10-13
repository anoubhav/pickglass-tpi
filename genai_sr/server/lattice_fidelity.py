"""lattice_fidelity.py -- spectral lattice fidelity of every degraded / restored crop (README section 6).

Uses the counter's own pre-processing (lcn + glare mask, working side N = min(S, 1024)) and whitened spectrum
(counting.Spectrum, z-scores).  Reference lattice B_ref (centroid basis a, b) and weave sub-lattice W_ref from
est_lattice on the CLEAN reference crop.

  (a) dominant-peak shift: per family, the strongest local maximum within +-10 deg of the axis in [18, 0.42 N]
      cycles/inch; relative frequency shift and angle shift vs the same quantity on the reference.
  (b) HPEF: significant maxima (z >= 6, 5x5) in [18, 0.42 N]; explained if within max(1.5 bins, 3 % |f|) of a
      reference lattice / W_ref point (order <= 6, counting._explain) or of a significant reference peak (z >= 4);
      HPEF = sum g+(unexplained) / sum g+(all), g+ = max(z - z0, 0) (z0 = noise baseline of the 3x3 max map).
  (c) lattice erasure: reference lattice points (a, b, a+b, a-b, 2a, 2b, W_ref points) with reference g >= 3;
      erased if restored g < 3; split by |f| below / above the LR Nyquist limit (LR side / 2 cycles per inch).

    WVAQ_ROOT=... venv_eval/bin/python code/lattice_fidelity.py --restorers none,bicubic,... --workers 16
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import cv2
cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402
sys.path.insert(0, os.path.join(G.WVAQ, "methods"))
import counting as C  # noqa: E402

Z_SIG, Z_REF, G_SIG, ORDER, ANG = 6.0, 4.0, 3.0, 6, 10.0
CORE = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)]


def spectrum(img):
    g, w, N, info = C.preprocess(img)
    sp = C.Spectrum(g)
    sp._maxmap(1)
    return sp, N


def sig_peaks(sp, zmin):
    pk = sp.peaks(zmin=zmin, size=5)
    if not pk:
        return np.zeros((0, 2)), np.zeros(0)
    P = np.array([(u, v) for u, v, _ in pk])
    z = np.array([z for _, _, z in pk])
    return P, z


def dominant(sp, P, z):
    out = {}
    for fam in ("x", "y"):
        best = None
        for (u, v), zz in zip(P, z):
            uu, vv = (u, v) if (fam == "x" or v > 0) else (-u, -v)
            ang = math.degrees(math.atan2(vv, uu)) if fam == "x" else math.degrees(math.atan2(uu, vv))
            if abs(ang) <= ANG and (best is None or zz > best[2]):
                best = (math.hypot(uu, vv), ang, zz)
        out[fam] = best
    return out


def dominant_raw(sp):
    """Deviation D1: strongest near-axis local maximum of the RAW log power (not whitened) in [18, 0.42 N]."""
    from scipy import ndimage
    L = sp.logP
    mx = ndimage.maximum_filter(L, size=5, mode="nearest")
    lo, hi = sp.band
    m = (L == mx) & (sp.r >= lo) & (sp.r <= hi) & ((sp.U > 0) | ((sp.U == 0) & (sp.V > 0)))
    ys, xs = np.nonzero(m)
    out = {}
    for fam in ("x", "y"):
        best = None
        for y, x in zip(ys, xs):
            u, v = sp.U[y, x], sp.V[y, x]
            uu, vv = (u, v) if (fam == "x" or v > 0) else (-u, -v)
            ang = math.degrees(math.atan2(vv, uu)) if fam == "x" else math.degrees(math.atan2(uu, vv))
            if abs(ang) <= ANG and (best is None or L[y, x] > best[2]):
                best = (math.hypot(uu, vv), ang, L[y, x])
        out[fam] = best
    return out


def ref_info(key):
    img = cv2.imread(G.ref_path(key), cv2.IMREAD_COLOR)
    sp, N = spectrum(img)
    lat = C.est_lattice(sp)
    B = np.array(lat["B"])          # columns a, b (cycles / crop)
    W = lat["weave_W"]
    P, z = sig_peaks(sp, Z_REF)
    Pd, zd = sig_peaks(sp, 4.0)
    dom = dominant(sp, Pd, zd)
    pts = [np.array(c, float) for c in CORE] + [np.array(p, float) for p in C.WEAVE_PTS.get(W, [])]
    Q = np.array([B @ c for c in pts])
    g, r = sp.lookup(Q[:, 0], Q[:, 1])
    lo, hi = sp.band
    keep = (g >= G_SIG) & (r >= 5) & (r <= hi)
    # anchors for the sampling-replica test (deviation D1): reference peaks + all lattice points, both signs
    A = np.concatenate([P, Q, -P, -Q], 0) if len(P) else np.concatenate([Q, -Q], 0)
    domP = dominant_raw(sp)
    return dict(B=B, W=W, P_ref=P, dom=dom, domP=domP, Q=Q[keep], r=r[keep], N=N, A=A)


def job(args):
    key, set_name, cond, restorer, path, S, lr_side, ref = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    sp, N = spectrum(img)
    lo, hi = sp.band
    P, z = sig_peaks(sp, Z_SIG)
    gplus = np.clip(z - sp._z0[1], 0, None)
    row = dict(key=key, set=set_name, cond=cond, restorer=restorer, N=N, n_sig=len(P))
    # (b) hallucinated-peak energy fraction
    if len(P):
        B = ref["B"]
        Cq = P @ np.linalg.inv(B).T
        tol = np.maximum(1.5, 0.03 * np.hypot(P[:, 0], P[:, 1]))
        ok, _ = C._explain(Cq, P, B, ref["W"], ORDER, tol)
        if len(ref["P_ref"]):
            d = np.hypot(P[:, None, 0] - ref["P_ref"][None, :, 0], P[:, None, 1] - ref["P_ref"][None, :, 1])
            ok |= (d <= tol[:, None]).any(axis=1)
        row["n_unexpl"] = int((~ok).sum())
        row["hpef"] = float(gplus[~ok].sum() / gplus.sum()) if gplus.sum() > 0 else 0.0
        row["g_unexpl"] = float(gplus[~ok].sum())
        # deviation D1: unexplained peaks that sit on a sampling replica q + f_s * k (alias / imaging)
        fs = float(lr_side)
        ks = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)], float) * fs
        R = (ref["A"][:, None, :] + ks[None]).reshape(-1, 2)
        drep = np.hypot(P[:, None, 0] - R[None, :, 0], P[:, None, 1] - R[None, :, 1]).min(axis=1)
        rep = (~ok) & (drep <= tol)
        nov = (~ok) & ~rep
        row["n_replica"], row["n_novel"] = int(rep.sum()), int(nov.sum())
        row["hpef_replica"] = float(gplus[rep].sum() / gplus.sum()) if gplus.sum() > 0 else 0.0
        row["hpef_novel"] = float(gplus[nov].sum() / gplus.sum()) if gplus.sum() > 0 else 0.0
    else:
        row.update(n_unexpl=0, hpef=0.0, g_unexpl=0.0, n_replica=0, n_novel=0, hpef_replica=0.0, hpef_novel=0.0)
    # (c) lattice erasure, below / above the LR Nyquist limit
    nyq = lr_side / 2.0
    Q, rq = ref["Q"], ref["r"]
    inb = rq <= hi
    if len(Q):
        g, _ = sp.lookup(Q[:, 0], Q[:, 1])
        er = (g < G_SIG) | ~np.isfinite(g)
        below = (rq <= nyq) & inb
        above = (rq > nyq) & inb
        row["n_lat_below"], row["n_lat_above"] = int(below.sum()), int(above.sum())
        row["erase_below"] = float(er[below].mean()) if below.any() else np.nan
        row["resynth_above"] = float((~er[above]).mean()) if above.any() else np.nan
    # (a) dominant-peak shift
    Pd, zd = sig_peaks(sp, 4.0)
    dom = dominant(sp, Pd, zd)
    for fam in ("x", "y"):
        rd, md = ref["dom"][fam], dom[fam]
        if rd is None or md is None:
            row[f"dom_{fam}_rel"], row[f"dom_{fam}_dang"] = np.nan, np.nan
        else:
            row[f"dom_{fam}_rel"] = md[0] / rd[0] - 1
            row[f"dom_{fam}_dang"] = md[1] - rd[1]
            row[f"dom_{fam}_f"] = md[0]
    domP = dominant_raw(sp)
    for fam in ("x", "y"):
        rd, md = ref["domP"][fam], domP[fam]
        if rd is None or md is None:
            row[f"domP_{fam}_rel"], row[f"domP_{fam}_dang"] = np.nan, np.nan
        else:
            row[f"domP_{fam}_rel"] = md[0] / rd[0] - 1
            row[f"domP_{fam}_dang"] = md[1] - rd[1]
            row[f"domP_{fam}_f"] = md[0]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    man = pd.read_csv(G.MANIFEST)
    lr = {(r.key, r.cond): r.lr_w for r in man.itertuples()}
    keys = list(man.key.unique())
    t0 = time.time()
    with Pool(a.workers) as p:
        refs = dict(zip(keys, p.map(ref_info, keys)))
    print(f"refs {time.time() - t0:.0f}s", flush=True)
    # reference self-check rows (clean vs itself = 0 by construction for HPEF; erasure 0)
    for rest in a.restorers.split(","):
        outp = os.path.join(G.GSR, "results", f"lattice_{rest}.csv")
        its = G.items(rest, man)
        if os.path.exists(outp) and len(pd.read_csv(outp)) >= len(its) and len(its) > 0:
            print(rest, "done", flush=True)
            continue
        jobs = [(k, s, c, rr, p, S, lr.get((k, c), S), refs[k]) for k, s, c, rr, p, S in its]
        t1 = time.time()
        with Pool(a.workers) as p:
            rows = p.map(job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"{rest}: {len(rows)} in {time.time() - t1:.0f}s", flush=True)
    pd.DataFrame([dict(key=k, W_ref=v["W"], a_u=v["B"][0, 0], a_v=v["B"][1, 0], b_u=v["B"][0, 1], b_v=v["B"][1, 1],
                       n_lat_sig=len(v["Q"]), n_ref_peaks=len(v["P_ref"])) for k, v in refs.items()]).to_csv(
        os.path.join(G.GSR, "results", "lattice_refs.csv"), index=False)


if __name__ == "__main__":
    main()
