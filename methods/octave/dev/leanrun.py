"""Lean dev run: frozen ours + symmetry features (Feat) + ridge stats at several hypotheses."""
import os, sys, json, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import devrun as DR, octfeat as F, octrule as O
import counting_frozen_final as C

HY = {"t": (1, 1), "hx": (0.5, 1), "hy": (1, 0.5), "dx": (2, 1), "dy": (1, 2), "hh": (0.5, 0.5)}

def job(a):
    src, key, th, tv = a
    try:
        crop = DR.load(src, key); ctx = F.Ctx(crop)
        e = C._estimate_pre(ctx.g, ctx.w, ctx.N, ctx.info, "ours", with_weave=False, cache={"sp": ctx.sp})
        d = e["debug"]; lat = d.get("lattice") or {}
        ax, ay = lat.get("angle_x") or 0.0, lat.get("angle_y") or 0.0
        row = {"src": src, "key": key, "true_h": th, "true_v": tv, "ours_h": e["h_count"], "ours_v": e["v_count"],
               "lat_h": lat.get("h_count", np.nan), "lat_v": lat.get("v_count", np.nan), "W": lat.get("weave_W"),
               "N": ctx.N, "S": crop.shape[0], "ang_x": ax, "ang_y": ay, "fusion": "|".join(d.get("fusion") or []),
               "hi": ctx.sp.band[1]}
        for k, (h, v) in (d.get("components") or {}).items():
            row[f"{k}_h"], row[f"{k}_v"] = h, v
        feat = O.Feat(ctx)
        h, v = e["h_count"], e["v_count"]
        if np.isfinite(h) and np.isfinite(v):
            Bo = F.snap(ctx, F.basis_from(h, v, ax, ay), rel=0.04)
            row["o_B"] = json.dumps(Bo.tolist())
            for wt in ("snr", "pow"):
                for k, val in feat(Bo, wt).items():
                    row[f"o_{wt}_{k}"] = val
            for axis, cnt, ang in (("x", h, ax), ("y", v, ay)):
                r = F.ridge_test(ctx, cnt, axis, ang)
                for k in ("r2", "d2", "r2h", "d2h"):
                    row[f"o_{axis}_{k}"] = r[k]
                row[f"o_{axis}_c"] = json.dumps(r["c"])
        if np.isfinite(th):
            Bt = F.snap(ctx, F.basis_from(th, tv, ax, ay))
            row["t_B"] = json.dumps(Bt.tolist())
            for tag, (mx, my) in HY.items():
                B = Bt @ np.diag([mx, my])
                for wt in ("snr",):
                    for k, val in feat(B, wt).items():
                        row[f"{tag}_{wt}_{k}"] = val
            for tag, m in (("t", 1.0), ("hx", 0.5), ("dx", 2.0)):
                for axis, cnt, ang in (("x", th * m, ax), ("y", tv * m, ay)):
                    r = F.ridge_test(ctx, cnt, axis, ang)
                    for k in ("r2", "d2", "r2h", "d2h"):
                        row[f"{tag}_{axis}_{k}"] = r[k]
        return row
    except Exception:
        import traceback; return {"src": src, "key": key, "err": traceback.format_exc()}

if __name__ == "__main__":
    which = sys.argv[1]
    lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
    jobs = [("synth", r.file, r.h_count, r.v_count) for r in lab.itertuples()]
    with Pool(int(os.environ.get("NW", "9"))) as p: rows = p.map(job, jobs, chunksize=2)
    df = pd.DataFrame(rows); df.to_pickle(os.path.join(HERE, "lean_synth.pkl"))
    print("done", len(df), df["err"].notna().sum() if "err" in df else 0)
