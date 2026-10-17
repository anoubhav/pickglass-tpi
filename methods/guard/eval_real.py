"""eval_real.py -- ONE-SHOT evaluation of the native-resolution guard on the saved real crops.

Run once, after guard_params.json was frozen on synthetic data (tune_synth.py).  Refuses to run
again if results/real_counts.csv exists (use --force only to recover from a crash; the meta file
records every start).

Inputs (all read-only; every file is md5-checked against the manifest of the study that made it):
  genai   the genai_sr study (genai_sr/README.md): 66 references (R-HR 53, R-LEG 13) x
          D4 / D8 / D4J for
            clean       genai_sr/inputs/ref/<key>.{jpg,png}                  (manifest ref_md5)
            none        genai_sr/inputs/lr/<cond>/<key>.png                  (manifest lr_md5)
            bicubic, swinir, realesrgan, seedvr2, invsr, flux2klein
                        full-resolution PNGs, which stay on gpu-server (genai_sr README section 8): streamed
                        read-only with `ssh gpu-server cat ...` into memory (nothing written to disk),
                        md5 = genai_sr/results/outputs_manifest.csv png_md5
            nbpro       genai_sr/outputs/nbpro/<cond>/<key>.png (R-LEG only, supplementary)
  clean150  annotation_tool/data/crops/{0..149}.jpg (the 150 clean crops; no labels used)
  cascade   the 13 legacy crops x 20 degradations of degrade_vlmcrops.py (the cascade's inputs,
            paper/tables/cascade_legacy13_inputs.csv lineage): $PICKGLASS_SCRATCH/deg_legacy13/<tag>/<name>.png
            (manifest md5)

Per image: estimate_bandwidth -> unguarded frozen estimate (ours, lcn, glare mask, max_work 1024) ->
count_guarded with those two precomputed (it re-counts at the native side only when it fires).  Output:
results/real_counts.csv (+ real_counts_meta.json).  Statistics: analyze_real.py.

    <venv2>/bin/python eval_real.py --workers 10
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import hashlib
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
METHODS = os.path.dirname(HERE)
ROOT = os.path.dirname(METHODS)
sys.path.insert(0, HERE)
import native_res_guard as G  # noqa: E402

GSR = os.path.join(ROOT, "genai_sr")
SERVER = os.environ.get("PICKGLASS_GPU_HOST", "gpu-server")
SERVER_GSR = ("~/" + os.environ.get("PICKGLASS_REMOTE_ROOT", "pickglass") + "/genai_sr")
DEG_DIR = (os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), "scratch")), "deg_legacy13"))
RES = os.path.join(HERE, "results")
OUT_CSV = os.path.join(RES, "real_counts.csv")
META = os.path.join(RES, "real_counts_meta.json")
SERVER_RESTORERS = ["bicubic", "swinir", "realesrgan", "seedvr2", "invsr", "flux2klein"]
CONDS = ["D4", "D8", "D4J"]


def md5b(b):
    return hashlib.md5(b).hexdigest()


def fetch_server(rel):
    last = None
    for attempt in range(5):
        try:
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SERVER,
                                f"cat {SERVER_GSR}/{rel}"], capture_output=True, timeout=300)
            if p.returncode == 0 and p.stdout:
                return p.stdout
            last = p.stderr.decode(errors="replace")[:200]
        except subprocess.TimeoutExpired as ex:  # noqa: PERF203
            last = repr(ex)
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {rel}: {last}")


def jobs():
    out = []
    man = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv"))
    om = pd.read_csv(os.path.join(GSR, "results", "outputs_manifest.csv"))
    om_md5 = {(r.restorer, r.cond, r.key): r.png_md5 for r in om.itertuples()}
    for r in man.drop_duplicates("key").itertuples():
        out.append(dict(group="genai", set=r.set, key=r.key, cond="clean", restorer="clean",
                        src="local", path=os.path.join(GSR, r.ref_path), md5=r.ref_md5))
    for r in man.itertuples():
        out.append(dict(group="genai", set=r.set, key=r.key, cond=r.cond, restorer="none",
                        src="local", path=os.path.join(GSR, r.lr_path), md5=r.lr_md5))
        for rest in SERVER_RESTORERS:
            out.append(dict(group="genai", set=r.set, key=r.key, cond=r.cond, restorer=rest, src="server",
                            path=f"outputs/{rest}/{r.cond}/{r.key}.png", md5=om_md5[(rest, r.cond, r.key)]))
        if (("nbpro", r.cond, r.key) in om_md5 and
                os.path.exists(os.path.join(GSR, "outputs", "nbpro", r.cond, r.key + ".png"))):
            out.append(dict(group="genai", set=r.set, key=r.key, cond=r.cond, restorer="nbpro", src="local",
                            path=os.path.join(GSR, "outputs", "nbpro", r.cond, r.key + ".png"),
                            md5=om_md5[("nbpro", r.cond, r.key)]))
    for i in range(150):
        out.append(dict(group="clean150", set="all150", key=str(i), cond="clean", restorer="clean", src="local",
                        path=os.path.join(ROOT, "annotation_tool", "data", "crops", f"{i}.jpg"), md5=None))
    dm = pd.read_csv(os.path.join(DEG_DIR, "manifest.csv"))
    for r in dm.itertuples():
        out.append(dict(group="cascade", set="legacy13", key=r.image, cond=r.tag, restorer=r.kind, src="local",
                        path=os.path.join(DEG_DIR, r.tag, f"{r.image}.png"), md5=r.md5, level=float(r.level)))
    return out


def _job(j):
    t0 = time.time()
    b = fetch_server(j["path"]) if j["src"] == "server" else open(j["path"], "rb").read()
    m = md5b(b)
    if j["md5"] is not None and m != j["md5"]:
        raise RuntimeError(f"md5 mismatch {j['path']}: {m} != {j['md5']}")
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    t_load = time.time() - t0
    C = G.frozen_counter()
    t1 = time.time()
    bw = G.estimate_bandwidth(img)
    ung = C.estimate(img, method="ours")
    t_ung = time.time() - t1
    gd = G.count_guarded(img, _unguarded=ung, _bandwidth=bw)
    g = gd["guard"]
    du, dg = ung["debug"], gd["debug"]
    ax = g["bandwidth_axes"]
    row = dict({k: v for k, v in j.items() if k not in ("path",)}, md5_read=m, S=int(img.shape[1]), H=int(img.shape[0]),
               ug_h=ung["h_count"], ug_v=ung["v_count"], ug_conf=ung["confidence"], ug_weave=ung["weave"],
               ug_N_work=du.get("N_work"), ug_fusion="|".join(du["fusion"]) if du.get("fusion") else None,
               ug_sharpness=du.get("sharpness"),
               g_h=gd["h_count"], g_v=gd["v_count"], g_conf=gd["confidence"], g_weave=gd["weave"],
               g_N_work=dg.get("N_work"), g_fusion="|".join(dg["fusion"]) if dg.get("fusion") else None,
               fired=g["fired"], eligible=g["eligible"], unsupported=g["unsupported_claim"], flag=g["flag"],
               upsampled=g["upsampled"],
               insufficient=g["insufficient_resolution"], n_eff=g["n_eff"], frac_nyq=g["frac_nyq"],
               depth=g["depth"], dead_band=g["dead_band"], target=g["target"], ppy=g["px_per_yarn"],
               ppy_margin=g["px_per_yarn_margin"],
               fc_x=ax["x"]["f_c"], fc_y=ax["y"]["f_c"], depth_x=ax["x"]["depth"], depth_y=ax["y"]["depth"],
               analysis_s=bw["seconds"], load_s=round(t_load, 2), unguarded_s=round(t_ung, 2),
               total_s=round(time.time() - t0, 2))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N jobs only, written to a _debug CSV")
    a = ap.parse_args()
    out_csv = OUT_CSV if not a.limit else OUT_CSV.replace(".csv", "_debug.csv")
    if os.path.exists(out_csv) and not a.force and not a.limit:
        raise SystemExit(f"{out_csv} exists: the real evaluation is run once")
    params_md5 = hashlib.md5(open(G.PARAMS_PATH, "rb").read()).hexdigest()
    J = jobs()
    if a.limit:
        J = J[: a.limit]
    meta = json.load(open(META)) if os.path.exists(META) else {"starts": []}
    meta["starts"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), n_jobs=len(J), limit=a.limit, force=a.force,
                               guard_params_md5=params_md5))
    meta.update(guard_params=G.load_params(), guard_module_md5=hashlib.md5(open(G.__file__, "rb").read()).hexdigest(),
                frozen_counter_md5=hashlib.md5(open(G.FROZEN_PATH, "rb").read()).hexdigest(),
                conf_model_md5=hashlib.md5(open(G.CONF_MODEL_PATH, "rb").read()).hexdigest(),
                cv2=cv2.__version__, numpy=np.__version__)
    if not a.limit:
        json.dump(meta, open(META, "w"), indent=1, default=str)
    # big server images first (load balance); local small ones last
    J.sort(key=lambda j: (j["src"] != "server", j["restorer"] == "none"))
    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, J, chunksize=1)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(J)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["group", "set", "key", "restorer", "cond"])
    df.to_csv(out_csv, index=False)
    if not a.limit:
        meta["finished"] = dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), n_rows=len(df), wall_s=round(time.time() - t0))
        json.dump(meta, open(META, "w"), indent=1, default=str)
    print(f"wrote {out_csv}: {len(df)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
