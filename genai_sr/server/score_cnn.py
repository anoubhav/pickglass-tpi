"""score_cnn.py -- synthetic-only CNN (synth_r18_v2mix_aniso) on clean refs, LR and restored images, exactly as
cnn/degrade_vlmcrops.py: cap_side(img, 1024), predict_crop with 3-scale x 3-flip TTA.

    WVAQ_ROOT=$G/wvaq_copy [CUDA_VISIBLE_DEVICES=<idle>] venv_eval/bin/python code/score_cnn.py --restorers ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402
sys.path.insert(0, os.path.join(G.WVAQ, "cnn"))
import cnn_common as CC  # noqa: E402

CKPT = os.path.join(G.WVAQ, "cnn", "runs", "synth_r18_v2mix_aniso", "model.pt")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", required=True)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = torch.device("cuda") if (a.device == "auto" and torch.cuda.is_available()) else torch.device(
        "cpu" if a.device == "auto" else a.device)
    model = CC.load_ckpt(CKPT, dev)
    for rest in a.restorers.split(","):
        its = G.items(rest)
        outp = os.path.join(G.GSR, "results", f"counts_cnn_{rest}.csv")
        if os.path.exists(outp) and len(pd.read_csv(outp)) >= len(its) and len(its) > 0:
            print(rest, "already scored", flush=True)
            continue
        rows, t0 = [], time.time()
        for key, set_name, cond, restorer, path, S in its:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            o = CC.predict_crop(model, CC.cap_side(img, 1024), dev, Ps=CC.P_TEST, flips=True)
            rows.append(dict(key=key, set=set_name, cond=cond, restorer=restorer, cnn_h=o["h"], cnn_v=o["v"],
                             cnn_h_iqr=o["h_iqr"], cnn_v_iqr=o["v_iqr"],
                             cnn_weave=CC.WEAVE_CLASSES[int(np.argmax(o["weave_probs"]))]))
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"{rest}: {len(rows)} images, {time.time() - t0:.0f}s on {dev}", flush=True)
    G.write_json(os.path.join(G.GSR, "logs", "score_cnn_env.json"), dict(
        ckpt_md5=G.md5(CKPT), torch=torch.__version__, device=str(dev), Ps=list(CC.P_TEST), flips=True, cap_side=1024,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")))


if __name__ == "__main__":
    main()
