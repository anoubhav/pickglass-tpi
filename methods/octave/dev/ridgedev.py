import os, sys, json, time
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import devrun as DR, octfeat as F
import counting_frozen_final as C

def job(a):
    src, key, th, tv = a
    try:
        crop = DR.load(src, key); ctx = F.Ctx(crop)
        e = C._estimate_pre(ctx.g, ctx.w, ctx.N, ctx.info, "ours", with_weave=False, cache={"sp": ctx.sp})
        lat = e["debug"].get("lattice") or {}
        ax, ay = lat.get("angle_x") or 0.0, lat.get("angle_y") or 0.0
        row = {"src": src, "key": key, "true_h": th, "true_v": tv, "ours_h": e["h_count"], "ours_v": e["v_count"],
               "lat_h": lat.get("h_count", np.nan), "lat_v": lat.get("v_count", np.nan), "W": lat.get("weave_W"), "N": ctx.N}
        hyps = {"o": (e["h_count"], e["v_count"])}
        if np.isfinite(th):
            hyps.update({"t": (th, tv), "h": (th / 2, tv / 2), "d": (th * 2, tv * 2)})
        for tag, (h, v) in hyps.items():
            for axis, cnt, ang in (("x", h, ax), ("y", v, ay)):
                if not np.isfinite(cnt) or cnt <= 0: continue
                r = F.ridge_test(ctx, cnt, axis, ang)
                for k in ("r2", "d2", "r2h", "d2h", "n"):
                    row[f"{tag}_{axis}_{k}"] = r[k]
                row[f"{tag}_{axis}_c"] = json.dumps(r["c"]); row[f"{tag}_{axis}_ch"] = json.dumps(r["ch"])
        return row
    except Exception:
        import traceback; return {"src": src, "key": key, "err": traceback.format_exc()}

if __name__ == "__main__":
    which = sys.argv[1]; jobs = []
    if which == "legacy":
        L13 = pd.read_csv(f"{DR.ROOT}/methods/results/legacy13_per_image.csv"); L13 = L13[L13.gt_usable & L13.gt_warp.notna()]
        jobs = [("legacy", r.image, r.gt_warp, r.gt_weft) for r in L13.itertuples()]
    elif which == "real":
        jobs = [("real", str(i), np.nan, np.nan) for i in range(150)]
    elif which == "synth":
        lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
        n = int(sys.argv[2]) if len(sys.argv) > 2 else len(lab)
        jobs = [("synth", r.file, r.h_count, r.v_count) for r in lab.head(n).itertuples()]
    with Pool(int(os.environ.get("NW", "8"))) as p: rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_pickle(os.path.join(HERE, f"ridge_{which}.pkl"))
    print("done", len(df), df["err"].notna().sum() if "err" in df else 0)
