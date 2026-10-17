#!/usr/bin/env python
"""Capture-quality study (WVAQ 2027): the synthetic-only CNN on the degraded legacy-13 crops.

Reads the byte-identical degraded crops written by methods/degrade_vlmcrops.py --make (degrade.py
operators on the canonical VLM crops vlm/crops/legacy/{name}.png; md5 checked against that
manifest) and predicts exactly as evaluate.eval_legacy does for the reported zero-shot row:
cap_side(crop, 1024), then predict_crop with the 3-scale x 3-flip TTA (Ps = P_TEST).

    VC=<venv_cnn>/bin/python
    $VC degrade_vlmcrops.py   # -> runs/synth_r18_v2mix_aniso/degrade_vlmcrops/predictions.csv
"""
import argparse
import csv
import hashlib
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

import cnn_common as C
import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
DEG_DIR = os.environ.get(
    "DEG_LEGACY13_DIR",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "scratch")), "deg_legacy13"))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "model.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "synth_r18_v2mix_aniso", "degrade_vlmcrops"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = C.get_device(a.device)
    model = C.load_ckpt(a.ckpt, dev)
    gt = {g.name: g for g in E.load_legacy_gt()}
    with open(os.path.join(DEG_DIR, "manifest.csv")) as f:
        man = list(csv.DictReader(f))
    rows, t0 = [], time.time()
    for i, m in enumerate(man):
        b = open(m["path"], "rb").read()
        assert hashlib.md5(b).hexdigest() == m["md5"], m["path"]
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        crop = C.cap_side(img, 1024)
        o = C.predict_crop(model, crop, dev, Ps=C.P_TEST, flips=True)
        g = gt[m["image"]]
        rows.append(dict(image=m["image"], kind=m["kind"], level=float(m["level"]), tag=m["tag"],
                         S_deg=img.shape[1], pred_h=o["h"], pred_v=o["v"], h_iqr=o["h_iqr"], v_iqr=o["v_iqr"],
                         pred_weave=C.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))],
                         warp=g.warp, weft=g.weft, md5=m["md5"]))
        if (i + 1) % 19 == 0:
            print(f"  {i + 1}/{len(man)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    json.dump({"ckpt": a.ckpt, "ckpt_md5": hashlib.md5(open(a.ckpt, "rb").read()).hexdigest(),
               "Ps": list(C.P_TEST), "flips": True, "cap_side": 1024, "device": str(dev),
               "deg_dir": DEG_DIR, "n": len(rows), "seconds": round(time.time() - t0, 1)},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"wrote {a.out}/predictions.csv: {len(rows)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
