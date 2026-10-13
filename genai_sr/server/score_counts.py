"""score_counts.py -- ours (frozen) + log-Gabor (+ lattice/fft components) on clean refs, LR ('none') and every
restored image (README section 5).  CPU, one BLAS/cv2 thread per worker.

    WVAQ_ROOT=$G/wvaq_copy venv_eval/bin/python code/score_counts.py --restorers clean,none,bicubic,... --workers 16
Writes results/counts_classical_<restorer>.csv (skips restorers whose CSV already covers every image).
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
import argparse
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

ESTS = ["ours", "gabor", "lattice", "fft"]


def job(it):
    key, set_name, cond, restorer, path, S = it
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    t0 = time.time()
    res = C.estimate_many(img, ESTS)
    e = res["ours"]
    d = e["debug"]
    lat = d.get("lattice") or {}
    cf = d.get("conf_features") or {}
    row = dict(key=key, set=set_name, cond=cond, restorer=restorer, S_in=img.shape[1], N_work=d.get("N_work"),
               ours_h=e["h_count"], ours_v=e["v_count"], ours_conf=e["confidence"], ours_weave=e["weave"],
               lat_angle_x=lat.get("angle_x"), lat_angle_y=lat.get("angle_y"), weave_W=lat.get("weave_W"),
               fusion="|".join(d["fusion"]) if d.get("fusion") else None,
               glare_frac=d.get("glare_frac"), sharpness=d.get("sharpness"), contrast_rel=d.get("contrast_rel"),
               **{f"f_{k}": v for k, v in cf.items()})
    for m in ESTS[1:]:
        row[f"{m}_h"], row[f"{m}_v"], row[f"{m}_conf"] = res[m]["h_count"], res[m]["v_count"], res[m]["confidence"]
    row["md5"] = G.md5(path)
    row["seconds"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.join(G.GSR, "results"), exist_ok=True)
    for rest in a.restorers.split(","):
        its = G.items(rest)
        outp = os.path.join(G.GSR, "results", f"counts_classical_{rest}.csv")
        if os.path.exists(outp) and len(pd.read_csv(outp)) >= len(its) and len(its) > 0:
            print(rest, "already scored", flush=True)
            continue
        t0 = time.time()
        with Pool(a.workers) as p:
            rows = p.map(job, its, chunksize=1)
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"{rest}: {len(rows)} images, {time.time() - t0:.0f}s", flush=True)
    G.write_json(os.path.join(G.GSR, "logs", "score_counts_env.json"), dict(
        counting_md5=G.md5(C.__file__), conf_model_md5=G.md5(os.path.join(G.WVAQ, "methods", "conf_model.json")),
        cv2=cv2.__version__, numpy=np.__version__, estimators=ESTS, preproc="lcn", glare="mask", max_work=1024))


if __name__ == "__main__":
    main()
