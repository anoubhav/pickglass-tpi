"""iq_metrics.py -- image-quality metrics of every restored crop (README section 6).

Full-reference against the clean reference crop: PSNR / SSIM on luminance (BT.601 Y) at S x S (skimage, Gaussian
window sigma 1.5, data_range 255); LPIPS (AlexNet, v0.1) on RGB after INTER_AREA to 1024^2.
No-reference (pyiqa) on the 1024^2 image: MUSIQ (KonIQ) and CLIP-IQA.  Clean references get the NR metrics only.

    WVAQ_ROOT=... CUDA_VISIBLE_DEVICES=<idle> venv_eval/bin/python code/iq_metrics.py --restorers clean,bicubic,...
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

W = 1024


def y_channel(bgr):
    f = bgr.astype(np.float64)
    return 16.0 + (65.481 * f[..., 2] + 128.553 * f[..., 1] + 24.966 * f[..., 0]) / 255.0


def to_t(bgr, dev):
    x = cv2.cvtColor(cv2.resize(bgr, (W, W), interpolation=cv2.INTER_AREA) if bgr.shape[0] != W else bgr,
                     cv2.COLOR_BGR2RGB)
    return torch.from_numpy(x).float().div(255).permute(2, 0, 1)[None].to(dev)


@torch.no_grad()
def main():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips
    import pyiqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(dev).eval()
    musiq = pyiqa.create_metric("musiq", device=dev)
    clipiqa = pyiqa.create_metric("clipiqa", device=dev)
    man = pd.read_csv(G.MANIFEST)
    refs = {}
    for rest in a.restorers.split(","):
        outp = os.path.join(G.GSR, "results", f"iq_{rest}.csv")
        its = G.items(rest, man)
        if os.path.exists(outp) and len(pd.read_csv(outp)) >= len(its) and len(its) > 0:
            print(rest, "done", flush=True)
            continue
        rows, t0 = [], time.time()
        for key, set_name, cond, restorer, path, S in its:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            t = to_t(img, dev)
            row = dict(key=key, set=set_name, cond=cond, restorer=restorer,
                       musiq=float(musiq(t).item()), clipiqa=float(clipiqa(t).item()))
            if rest != "clean" and rest != "none":
                if key not in refs:
                    rp = G.ref_path(key)
                    refs[key] = cv2.imread(rp, cv2.IMREAD_COLOR)
                ref = refs[key]
                assert ref.shape == img.shape, (key, ref.shape, img.shape)
                yr, yi = y_channel(ref), y_channel(img)
                row["psnr_y"] = float(peak_signal_noise_ratio(yr, yi, data_range=255))
                row["ssim_y"] = float(structural_similarity(yr, yi, data_range=255, gaussian_weights=True, sigma=1.5,
                                                            use_sample_covariance=False))
                row["lpips_alex"] = float(lp(to_t(ref, dev) * 2 - 1, t * 2 - 1).item())
            rows.append(row)
        pd.DataFrame(rows).to_csv(outp, index=False)
        print(f"{rest}: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    G.write_json(os.path.join(G.GSR, "logs", "iq_env.json"), dict(
        pyiqa=pyiqa.__version__, lpips="alex v0.1", torch=torch.__version__, side_nr=W, device=str(dev),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")))


if __name__ == "__main__":
    main()
