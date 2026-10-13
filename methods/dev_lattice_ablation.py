import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
"""dev_lattice_ablation.py -- synthetic-only ablation of the lattice score terms (dev set).
Spectrum is computed once per image; est_lattice is re-run per option set."""
import os, sys, time, json
from multiprocessing import Pool
import cv2
cv2.setNumThreads(1)
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import counting as C
SYN = os.path.join(os.path.dirname(HERE), "cnn", "data", "synth_val")
VARIANTS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

def job(f):
    im = cv2.imread(os.path.join(SYN, f))
    g, w, N, info = C.preprocess(im)
    sp = C.Spectrum(g)
    out = {"file": f}
    fx, fy, _ = C.est_fft(g, sp=sp)
    out["fft_h"], out["fft_v"] = (fx["count"] if fx else np.nan), (fy["count"] if fy else np.nan)
    px, py = C.est_profile(g, w, "x"), C.est_profile(g, w, "y")
    tl = C.est_tiles(g)
    out["prof_h"], out["prof_v"] = px["count"], py["count"]
    out["tile_h"], out["tile_v"] = tl["x"]["count"], tl["y"]["count"]
    for name, opts in VARIANTS.items():
        lat = C.est_lattice(sp, opts)
        out[f"{name}_h"], out[f"{name}_v"] = (lat["h_count"], lat["v_count"]) if lat else (np.nan, np.nan)
        out[f"{name}+fus_h"] = C._fuse(out[f"{name}_h"], px["count"], tl["x"]["count"])[0]
        out[f"{name}+fus_v"] = C._fuse(out[f"{name}_v"], py["count"], tl["y"]["count"])[0]
    out["fft+fus_h"] = C._fuse(out["fft_h"], px["count"], tl["x"]["count"])[0]
    out["fft+fus_v"] = C._fuse(out["fft_v"], py["count"], tl["y"]["count"])[0]
    return out

if __name__ == "__main__":
    lab = pd.read_csv(os.path.join(SYN, "labels.csv"))
    t0 = time.time()
    with Pool(6) as p:
        res = p.map(job, list(lab.file), chunksize=4)
    df = lab.merge(pd.DataFrame(res), on="file")
    df.to_csv(sys.argv[1], index=False)
    rows = []
    for m in ["fft", "fft+fus", "prof", "tile"] + [x for v in VARIANTS for x in (v, v + "+fus")]:
        rh, rv = df[f"{m}_h"] / df.h_count, df[f"{m}_v"] / df.v_count
        w10 = ((np.abs(rh - 1) <= .1).mean() + (np.abs(rv - 1) <= .1).mean()) / 2
        w5 = ((np.abs(rh - 1) <= .05).mean() + (np.abs(rv - 1) <= .05).mean()) / 2
        mape = 100 * (np.abs(rh - 1).mean() + np.abs(rv - 1).mean()) / 2
        rows.append((m, round(w5, 3), round(w10, 3), round(mape, 1)))
    print(pd.DataFrame(rows, columns=["variant", "w5", "w10", "MAPE"]).to_string(index=False))
    print(f"{time.time()-t0:.0f}s")
