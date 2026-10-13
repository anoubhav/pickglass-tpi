"""posthoc_ungated.py -- POST-HOC sensitivity analysis (not part of the pre-specified evaluation).

The frozen guard re-counts only when the unguarded count lies outside the native band (consistency
gate, added on synthetic data before the real run; see RESULTS.md, design log).  This script asks what
the UNGATED variant (the first synthetic choice: re-count whenever eligible) would have given on the
same real crops: for every real row that was eligible but not fired, the frozen counter is run at
max_work = target (the same native side the guard would have used).  Nothing in guard_params.json
or native_res_guard.py is changed; results go to results/real_posthoc_ungated.csv.

    <venv2>/bin/python posthoc_ungated.py --workers 10
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_real as E  # noqa: E402
import native_res_guard as G  # noqa: E402

OUT = os.path.join(HERE, "results", "real_posthoc_ungated.csv")


def _job(j):
    b = E.fetch_server(j["path"]) if j["src"] == "server" else open(j["path"], "rb").read()
    if j["md5"] is not None and E.md5b(b) != j["md5"]:
        raise RuntimeError("md5 mismatch " + j["path"])
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    e = G.frozen_counter().estimate(img, method="ours", max_work=int(j["target"]))
    return dict(group=j["group"], set=j["set"], key=j["key"], cond=j["cond"], restorer=j["restorer"],
                target=int(j["target"]), ung_h=e["h_count"], ung_v=e["v_count"], ung_conf=e["confidence"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    R = pd.read_csv(os.path.join(HERE, "results", "real_counts.csv"))
    todo = R[R.eligible & ~R.fired]
    J = {(j["group"], j["set"], j["key"], j["cond"], j["restorer"]): j for j in E.jobs()}
    jobs = []
    for r in todo.itertuples():
        j = dict(J[(r.group, r.set, str(r.key), r.cond, r.restorer)])
        j["target"] = int(r.target)
        jobs.append(j)
    print("eligible but not fired:", len(jobs))
    t0 = time.time()
    with Pool(a.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(rows)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
