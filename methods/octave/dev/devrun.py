"""Run frozen 'ours' + octave features on synthetic / legacy / real150 (real: no labels)."""
import os, sys, json, time, pickle
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, HERE); sys.path.insert(0, f"{ROOT}/common"); sys.path.insert(0, f"{ROOT}/methods")
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
import octfeat as F
import counting_frozen_final as C


def load(src, key):
    if src == "synth":
        return cv2.imread(os.path.join(HERE, "synth", key))
    if src == "legacy":
        import legacy as L
        return L.rectify_legacy(key)[0]
    if src == "real":
        import rectify as R
        return R.rectify(int(key))
    raise ValueError(src)


def feats_for(ctx, B, tag):
    out = {}
    for wt in ("snr", "pow"):
        for k, v in F.symfeats(ctx, B, weight=wt).items():
            out[f"{tag}_{wt}_{k}"] = v
    for ax, nm in ((0, "x"), (1, "y")):
        for wt in ("snr", "pow"):
            r = F.rfeat(ctx, B, ax, weight=wt)
            for k, v in r.items():
                out[f"{tag}_{nm}_{wt}_{k}"] = v
        g = F.gvals(ctx, B, ax)
        for k, v in g.items():
            out[f"{tag}_{nm}_g_{k}"] = v
    return out


def job(a):
    src, key, th, tv = a
    t0 = time.time()
    try:
        crop = load(src, key)
        ctx = F.Ctx(crop)
        cache = {"sp": ctx.sp}
        e = C._estimate_pre(ctx.g, ctx.w, ctx.N, ctx.info, "ours", with_weave=False, cache=cache)
        d = e["debug"]
        lat = d.get("lattice") or {}
        row = {"src": src, "key": key, "S": crop.shape[0], "N": ctx.N, "true_h": th, "true_v": tv,
               "ours_h": e["h_count"], "ours_v": e["v_count"], "weave": e["weave"], "conf": e["confidence"],
               "fusion": "|".join(d.get("fusion") or []),
               "lat_h": lat.get("h_count", np.nan), "lat_v": lat.get("v_count", np.nan),
               "ang_x": lat.get("angle_x", 0.0), "ang_y": lat.get("angle_y", 0.0), "W": lat.get("weave_W")}
        comp = d.get("components", {})
        for k, (h, v) in comp.items():
            row[f"{k}_h"], row[f"{k}_v"] = h, v
        Bo = F.basis_from(e["h_count"], e["v_count"], row["ang_x"] or 0.0, row["ang_y"] or 0.0)
        Bo = F.snap(ctx, Bo, rel=0.04)
        row.update(feats_for(ctx, Bo, "o"))
        row["o_B"] = json.dumps(Bo.tolist())
        if np.isfinite(th):
            Bt = F.snap(ctx, F.basis_from(th, tv, row["ang_x"] or 0.0, row["ang_y"] or 0.0))
            row["t_B"] = json.dumps(Bt.tolist())
            row.update(feats_for(ctx, Bt, "t"))
            for tag, M in (("hx", np.diag([0.5, 1])), ("hy", np.diag([1, 0.5])), ("dx", np.diag([2, 1])),
                           ("dy", np.diag([1, 2])), ("hh", np.diag([0.5, 0.5])), ("dd", np.diag([2, 2]))):
                row.update(feats_for(ctx, Bt @ M, tag))
        row["rt"] = time.time() - t0
        return row
    except Exception as ex:  # noqa
        return {"src": src, "key": key, "err": repr(ex)}


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which in ("synth", "all"):
        lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
        n = int(sys.argv[2]) if len(sys.argv) > 2 else len(lab)
        jobs += [("synth", r.file, r.h_count, r.v_count) for r in lab.head(n).itertuples()]
    if which in ("legacy", "all"):
        L13 = pd.read_csv(f"{ROOT}/methods/results/legacy13_per_image.csv")
        L13 = L13[L13.gt_usable & L13.gt_warp.notna()]
        jobs += [("legacy", r.image, r.gt_warp, r.gt_weft) for r in L13.itertuples()]
    if which in ("real", "all"):
        jobs += [("real", str(i), np.nan, np.nan) for i in range(150)]
    t0 = time.time()
    with Pool(int(os.environ.get("NW", "10"))) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(HERE, f"dev2_{which}.pkl"))
    print("done", len(df), time.time() - t0, df.get("err", pd.Series(dtype=str)).notna().sum())
