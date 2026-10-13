"""eval_synth.py -- development evaluation of the counting estimators on the labelled
procedural synthetic validation set (cnn/data/synth_val, 300 images, exact labels).

This is the ONLY set used to debug / choose design options of methods/counting.py.
The 13 hand-counted legacy images are held out (see eval_legacy.py).

    python eval_synth.py --methods fft,profile,tiles,ours --n 300 --out results/dev/synth.csv
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")  # one BLAS thread per worker (the machine is shared)

import argparse
import os
import sys
import time
from multiprocessing import Pool

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import counting as C  # noqa: E402

SYN = os.path.join(os.path.dirname(HERE), "cnn", "data", "synth_val")


def _job(a):
    f, methods, opts = a
    im = cv2.imread(os.path.join(SYN, f))
    out = {"file": f}
    res = C.estimate_many(im, methods, **opts)
    for m in methods:
        e = res[m]
        out[f"{m}_h"], out[f"{m}_v"], out[f"{m}_conf"] = e["h_count"], e["v_count"], e["confidence"]
        if m in ("ours", "lattice"):
            out[f"{m}_weave"] = e["weave"]
            lat = e["debug"].get("lattice")
            if lat:
                for k in ("B_peak", "B_lsq", "B_centroid"):
                    out[f"{m}_{k}_h"], out[f"{m}_{k}_v"] = abs(lat[k][0][0]), abs(lat[k][1][1])
                out[f"{m}_W"] = lat["weave_W"]
            for k, v in e["debug"].get("conf_features", {}).items():
                out[f"{m}_f_{k}"] = v
            if e["debug"].get("fusion"):
                out[f"{m}_fusion"] = "|".join(e["debug"]["fusion"])
    return out


def summarise(df, methods):
    rows = []
    extra = [c[:-2] for c in df.columns if c.endswith("_h") and "_B_" in c and c.startswith("lattice")]
    for m in list(methods) + extra:
        for ax, g in (("h", "h_count"), ("v", "v_count")):
            p, t = df[f"{m}_{ax}"].values, df[g].values
            r = p / t
            e = np.abs(p - t)
            rows.append(dict(method=m, axis=ax, n=len(t), MAE=np.nanmean(e), MAPE=100 * np.nanmean(e / t),
                             w5=np.nanmean(np.abs(r - 1) <= .05), w10=np.nanmean(np.abs(r - 1) <= .10),
                             half=np.nanmean(np.abs(r - .5) < .06), double=np.nanmean(np.abs(r - 2) < .12),
                             nan=np.mean(~np.isfinite(p))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="fft,profile,tiles,ours")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "dev", "synth.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preproc", default="lcn")
    ap.add_argument("--glare", default="mask")
    args = ap.parse_args()
    lab = pd.read_csv(os.path.join(SYN, "labels.csv")).head(args.n)
    methods = args.methods.split(",")
    t0 = time.time()
    with Pool(args.workers) as pool:
        res = pool.map(_job, [(f, methods, {"preproc": args.preproc, "glare": args.glare}) for f in lab.file])
    df = lab.merge(pd.DataFrame(res), on="file")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    s = summarise(df, methods)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(f"{len(df)} images, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
