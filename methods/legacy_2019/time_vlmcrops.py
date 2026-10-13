"""Per-image wall clock of the 2019 doc3.py + count.py pipeline on the canonical legacy crops
(sequential, one process, seed 0) -> run_vlmcrops/latency.json.  Same functions as run_pipeline.py."""
import json, os, sys, time
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pipeline as rp
CROPS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "vlm", "crops", "legacy")
ts = []
for k in ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]:
    t0 = time.perf_counter()
    img = cv2.imread(os.path.join(CROPS, f"{k}.png"))
    cv2.setRNGSeed(0)
    r = rp.doc3_dotmap(img)
    rp.count_py_warp(r["final"]); rp.count_py_weft_mirror(r["final"])
    ts.append(time.perf_counter() - t0)
res = dict(hw="M2 Max CPU, 1 process, median", median_s=float(np.median(ts)), min_s=float(min(ts)),
           max_s=float(max(ts)), n=len(ts))
print(res)
json.dump(res, open(os.path.join(HERE, "run_vlmcrops", "latency.json"), "w"), indent=1)
