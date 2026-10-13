"""run_ours_150.py -- run the 'ours' counter on all 150 rectified 1-inch crops.

Writes
  predictions_ours_150.csv   one row per image: counts, weave, confidence + features, angles,
                             capture-quality features, and the alternative estimators
                             (fft, profile, tiles, gabor) for comparison
  proposals.json             annotation-tool contract (coordinates in the rectified crop of
                             common/rectify.py):
      {"<id>": {"rect_size":[S,S], "h_line_y":int, "h_positions":[x...], "v_line_x":int,
                "v_positions":[y...], "weave_guess":str, "method":str}}
      h_positions = x of the vertical yarns crossing the horizontal guide line y = h_line_y,
      v_positions = y of the horizontal yarns crossing the vertical guide line x = v_line_x.
Then (optionally) runs ../annotation_tool/prepare.py --refresh-proposals.

    python run_ours_150.py [--workers 6] [--refresh]
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")  # one BLAS thread per worker (the machine is shared)

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import counting as C  # noqa: E402
import rectify as R  # noqa: E402

TOOL_WEAVES = {"plain", "twill_2_1", "twill_3_1", "twill_2_2", "satin", "other", "unsure"}
METHOD_TAG = "ours-2026-lattice-v1"


def _job(i):
    t0 = time.time()
    crop = R.rectify(i)
    S = crop.shape[0]
    alts = ("fft", "profile", "tiles", "gabor", "lattice")
    res = C.estimate_many(crop, ("ours",) + alts)
    e = res["ours"]
    d = e["debug"]
    lat = d.get("lattice") or {}
    row = {"id": i, "S": S, "h_count": e["h_count"], "v_count": e["v_count"], "weave": e["weave"],
           "confidence": e["confidence"], "angle_x": lat.get("angle_x"), "angle_y": lat.get("angle_y"),
           "weave_W": lat.get("weave_W"), "score": lat.get("score"),
           "alt_h": (lat.get("alt_counts") or (np.nan, np.nan))[0],
           "alt_v": (lat.get("alt_counts") or (np.nan, np.nan))[1],
           "h_peak": abs(lat["B_peak"][0][0]) if lat else np.nan, "v_peak": abs(lat["B_peak"][1][1]) if lat else np.nan,
           "glare_frac": d.get("glare_frac"), "sat_frac": d.get("sat_frac"), "dark_frac": d.get("dark_frac"),
           "shadow_frac": d.get("shadow_frac"), "contrast_rel": d.get("contrast_rel"),
           "sharpness": d.get("sharpness")}
    for k, v in (d.get("conf_features") or {}).items():
        row[f"f_{k}"] = v
    for m in alts:
        row[f"h_{m}"], row[f"v_{m}"] = res[m]["h_count"], res[m]["v_count"]
    row["fusion"] = "|".join(d["fusion"]) if d.get("fusion") else None
    # guide-line crossing positions (shear angle = -lattice angle, verified on a synthetic grating)
    gp = C.guide_positions(crop, e["h_count"], e["v_count"],
                           angle_x=-(lat.get("angle_x") or 0.0), angle_y=-(lat.get("angle_y") or 0.0))
    wv = e["weave"] if e["weave"] in TOOL_WEAVES else "unsure"
    prop = {"rect_size": [S, S], "h_line_y": gp["h_line_y"], "h_positions": gp["h_positions"],
            "v_line_x": gp["v_line_x"], "v_positions": gp["v_positions"], "weave_guess": wv,
            "method": METHOD_TAG, "h_count": round(e["h_count"], 2), "v_count": round(e["v_count"], 2),
            "confidence": round(e["confidence"], 3)}
    row["n_h_positions"], row["n_v_positions"] = len(gp["h_positions"]), len(gp["v_positions"])
    row["runtime_s"] = time.time() - t0
    return row, prop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="run ../annotation_tool/prepare.py --refresh-proposals")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(_job, range(R.N_IMAGES), chunksize=2)
    rows = [r for r, _ in res]
    props = {str(r["id"]): pp for r, pp in res}
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(os.path.join(HERE, "predictions_ours_150.csv"), index=False)
    tmp = os.path.join(HERE, "proposals.json.tmp")
    with open(tmp, "w") as f:
        json.dump(props, f)
    os.replace(tmp, os.path.join(HERE, "proposals.json"))
    print(f"wrote predictions_ours_150.csv + proposals.json ({len(df)} images, {time.time() - t0:.0f}s)")
    print(df[["h_count", "v_count", "confidence", "runtime_s"]].describe().round(2).to_string())
    print(df.weave.value_counts().to_string())
    if a.refresh:
        prep = os.path.join(ROOT, "annotation_tool", "prepare.py")
        r = subprocess.run([a.python, prep, "--refresh-proposals"], capture_output=True, text=True,
                           cwd=os.path.dirname(prep))
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    main()
