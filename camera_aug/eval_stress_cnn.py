#!/usr/bin/env python
"""eval_stress_cnn.py -- the synthetic-only CNN (cnn/runs/synth_r18_v2mix_aniso/model.pt) on
PickGlass-Stress, with exactly the input handling of the reported zero-shot row
(cnn/degrade_vlmcrops.py, evaluate.eval_legacy): cap_side(crop, 1024), then predict_crop with the
3-scale x 3-flip TTA (Ps = C.P_TEST).  Files are md5-checked against the stress manifest.

    VC=<scratchpad>/venv_cnn/bin/python
    $VC eval_stress_cnn.py --set legacy13      -> results/stress_{set}_cnn.csv
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "cnn"))
import cnn_common as C  # noqa: E402

CKPT = os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso", "model.pt")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="legacy13")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--conds", default="")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(HERE, "stress", a.set, "manifest.csv"), dtype={"source_id": str})
    if a.conds:
        man = man[man.condition.isin(a.conds.split(","))]
    out = os.path.join(HERE, "results", f"stress_{a.set}_cnn.csv")
    jobs = []
    for sid, g in man.groupby("source_id"):
        jobs.append(((sid, "clean", 0), os.path.join(ROOT, g.source_path.iloc[0]), g.source_md5.iloc[0]))
    jobs += [((r.source_id, r.condition, int(r.rep)), os.path.join(HERE, r.path), r.md5) for r in man.itertuples()]
    old = pd.read_csv(out, dtype={"source_id": str}) if os.path.exists(out) else None
    if old is not None:
        done = set(zip(old.source_id, old.condition, old.rep))
        jobs = [j for j in jobs if j[0] not in done]
    dev = C.get_device(a.device)
    model = C.load_ckpt(CKPT, dev)
    print(f"{len(jobs)} images on {dev}", flush=True)
    rows, t0 = [], time.time()

    def save():
        d = pd.DataFrame(rows)
        if old is not None:
            d = pd.concat([old, d], ignore_index=True)
        d.to_csv(out, index=False)

    for i, (key, path, md5) in enumerate(jobs):
        b = open(path, "rb").read()
        assert hashlib.md5(b).hexdigest() == md5, path
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        t1 = time.time()
        o = C.predict_crop(model, C.cap_side(img, 1024), dev, Ps=C.P_TEST, flips=True)
        rows.append(dict(source_id=key[0], condition=key[1], rep=key[2], estimator="cnn", S_in=img.shape[0],
                         h=o["h"], v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                         weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))], wall_s=round(time.time() - t1, 2)))
        if (i + 1) % 20 == 0 or i + 1 == len(jobs):
            save()
            print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    save()
    json.dump({"ckpt": CKPT, "ckpt_md5": hashlib.md5(open(CKPT, "rb").read()).hexdigest(), "Ps": list(C.P_TEST),
               "flips": True, "cap_side": 1024, "device": str(dev)},
              open(out.replace(".csv", "_meta.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
