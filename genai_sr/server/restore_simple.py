"""restore_simple.py -- bicubic (control), SwinIR-M classical x4, Real-ESRGAN x4plus (README section 4).

    CUDA_VISIBLE_DEVICES=<idle> venv_gen/bin/python code/restore_simple.py --restorers bicubic,swinir,realesrgan

x4 models are applied ONCE to the LR PNG; their output is resized to S x S (INTER_CUBIC if up, INTER_AREA if down),
so D8 = x4 model then bicubic x2.  Weights (official GitHub releases) are loaded with spandrel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402

WEIGHTS = {
    "swinir": ("001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
               "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"),
    "realesrgan": ("RealESRGAN_x4plus.pth",
                   "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}


def to_size(img, S):
    if img.shape[0] == S and img.shape[1] == S:
        return img
    interp = cv2.INTER_CUBIC if img.shape[0] < S else cv2.INTER_AREA
    return cv2.resize(img, (S, S), interpolation=interp)


def load_model(name, device):
    import torch
    import spandrel
    fn, url = WEIGHTS[name]
    p = os.path.join(G.GSR, "models", fn)
    if not os.path.exists(p):
        torch.hub.download_url_to_file(url, p, progress=False)
    m = spandrel.ModelLoader(device=device).load_from_file(p)
    m.eval()
    return m, p


def run_model(m, lr_bgr, device):
    import torch
    x = torch.from_numpy(cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)).float().div(255).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        y = m(x)
    y = (y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(y, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", default="bicubic,swinir,realesrgan")
    a = ap.parse_args()
    man = pd.read_csv(G.MANIFEST)
    meta = {}
    for name in a.restorers.split(","):
        t0 = time.time()
        if name == "bicubic":
            m, wpath, device = None, None, "cpu"
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m, wpath = load_model(name, device)
        times = []
        for r in man.itertuples():
            op = G.out_path(name, r.cond, r.key)
            if os.path.exists(op):
                continue
            lr = cv2.imread(os.path.join(G.GSR, r.lr_path), cv2.IMREAD_COLOR)
            assert G.md5(os.path.join(G.GSR, r.lr_path)) == r.lr_md5
            t1 = time.time()
            if name == "bicubic":
                out = cv2.resize(lr, (r.S, r.S), interpolation=cv2.INTER_CUBIC)
            else:
                out = to_size(run_model(m, lr, device), r.S)
            times.append(time.time() - t1)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            cv2.imwrite(op, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        meta[name] = dict(weights=wpath, weights_md5=G.md5(wpath) if wpath else None,
                          url=WEIGHTS.get(name, (None, None))[1], device=device, n=len(times),
                          s_per_img=float(np.median(times)) if times else None, wall_s=round(time.time() - t0, 1),
                          cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(name, meta[name], flush=True)
    G.write_json(os.path.join(G.GSR, "logs", f"restore_simple_{'_'.join(a.restorers.split(','))}.json"), meta)


if __name__ == "__main__":
    main()
