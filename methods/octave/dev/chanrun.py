"""Per-colour-channel folded-ridge statistics at the frozen output (and truth for labelled sets)."""
import os, sys, json
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
import devrun as DR, octfeat as F
import counting_frozen_final as C

class G:  # minimal ctx for ridge_test
    def __init__(self, g, w): self.g, self.w = g, w

def chan_ctx(crop, k):
    ch = crop[:, :, k]
    g, w, N, info = C.preprocess(np.dstack([ch, ch, ch]))
    return G(g, w)

def job(a):
    src, key, th, tv = a
    try:
        crop = DR.load(src, key)
        g, w, N, info = C.preprocess(crop)
        sp = C.Spectrum(g)
        e = C._estimate_pre(g, w, N, info, "ours", with_weave=False, cache={"sp": sp})
        lat = e["debug"].get("lattice") or {}
        ax, ay = lat.get("angle_x") or 0.0, lat.get("angle_y") or 0.0
        row = {"src": src, "key": key, "true_h": th, "true_v": tv, "ours_h": e["h_count"], "ours_v": e["v_count"], "N": N}
        ctxs = {"pca": G(g, w), "b": chan_ctx(crop, 0), "g": chan_ctx(crop, 1), "r": chan_ctx(crop, 2)}
        hyps = {"o": (e["h_count"], e["v_count"])}
        if np.isfinite(th): hyps.update({"t": (th, tv), "hx": (th / 2, tv / 2)})
        for tag, (h, v) in hyps.items():
            for axis, cnt, ang in (("x", h, ax), ("y", v, ay)):
                if not (np.isfinite(cnt) and cnt > 0): continue
                for cn, cx in ctxs.items():
                    r = F.ridge_test(cx, cnt, axis, ang)
                    row[f"{tag}_{axis}_{cn}_r2"] = r["r2"]; row[f"{tag}_{axis}_{cn}_d2"] = r["d2"]; row[f"{tag}_{axis}_{cn}_r2h"] = r["r2h"]
                    row[f"{tag}_{axis}_{cn}_c"] = json.dumps(r["c"])
        return row
    except Exception:
        import traceback; return {"src": src, "key": key, "err": traceback.format_exc()}

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "legacy":
        L13 = pd.read_csv(f"{DR.ROOT}/methods/results/legacy13_per_image.csv"); L13 = L13[L13.gt_usable & L13.gt_warp.notna()]
        jobs = [("legacy", r.image, r.gt_warp, r.gt_weft) for r in L13.itertuples()]
    elif which == "real":
        jobs = [("real", str(i), np.nan, np.nan) for i in range(150)]
    else:
        lab = pd.read_csv(os.path.join(HERE, "synth", "labels.csv"))
        jobs = [("synth", r.file, r.h_count, r.v_count) for r in lab.itertuples()]
    with Pool(int(os.environ.get("NW", "9"))) as p: rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows); df.to_pickle(os.path.join(HERE, f"chan_{which}.pkl"))
    print("done", len(df), df["err"].notna().sum() if "err" in df else 0)
