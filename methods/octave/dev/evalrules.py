import os, sys, json, time, itertools
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import devrun as DR, octfeat as F, octrule as O
import counting_frozen_final as C

GRID = [dict(weight=w, tau_d=td, tau_h=th, use_C=uc)
        for w in ("snr", "pow") for td in (0.10, 0.15, 0.20, 0.25, 0.30)
        for th in (None, -0.15, -0.10, -0.05, 0.0) for uc in (True, False)]
VN = [f"{g['weight']}_d{g['tau_d']}_h{g['tau_h']}_C{int(g['use_C'])}" for g in GRID]


def job(a):
    src, key, th, tv = a
    try:
        crop = DR.load(src, key)
        ctx = F.Ctx(crop)
        cache = {"sp": ctx.sp}
        e = C._estimate_pre(ctx.g, ctx.w, ctx.N, ctx.info, "ours", with_weave=False, cache=cache)
        lat = e["debug"].get("lattice") or {}
        ax, ay = lat.get("angle_x") or 0.0, lat.get("angle_y") or 0.0
        h, v = e["h_count"], e["v_count"]
        row = {"src": src, "key": key, "true_h": th, "true_v": tv, "ours_h": h, "ours_v": v, "W": lat.get("weave_W"),
               "fusion": "|".join(e["debug"].get("fusion") or [])}
        if not (np.isfinite(h) and np.isfinite(v)):
            return row
        B0 = F.snap(ctx, F.basis_from(h, v, ax, ay), rel=0.04)
        feat = O.Feat(ctx)
        hi = ctx.sp.band[1]
        f0 = feat(B0, "snr")
        for k in ("Dx", "Dy", "C", "Hx", "Hy"):
            row["f0_" + k] = f0[k]
        for nm, g in zip(VN, GRID):
            B, (fx, fy), tr = O.decide(feat, B0, hi=hi, **g)
            row[nm + "_fx"], row[nm + "_fy"] = fx, fy
            if nm == "snr_d0.15_hNone_C1":
                row["trace"] = str(tr)
        return row
    except Exception as ex:
        import traceback
        return {"src": src, "key": key, "err": traceback.format_exc()}


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which in ("synth",):
        lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
        jobs += [("synth", r.file, r.h_count, r.v_count) for r in lab.itertuples()]
    if which in ("legacy",):
        L13 = pd.read_csv(f"{DR.ROOT}/methods/results/legacy13_per_image.csv")
        L13 = L13[L13.gt_usable & L13.gt_warp.notna()]
        jobs += [("legacy", r.image, r.gt_warp, r.gt_weft) for r in L13.itertuples()]
    if which in ("real",):
        jobs += [("real", str(i), np.nan, np.nan) for i in range(150)]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "11"))) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(HERE, f"rules_{which}.pkl"))
    print("done", len(df), time.time() - t0, df["err"].notna().sum() if "err" in df else 0)
