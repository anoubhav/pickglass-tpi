"""run_seedvr2.py -- SeedVR2-3B (fp16) one-step diffusion restoration of every LR input (README section 4).

Drives the numz/ComfyUI-SeedVR2_VideoUpscaler standalone CLI in-process so that each image gets its own target
resolution (= reference side S) while the DiT/VAE stay cached.  CUDA_VISIBLE_DEVICES must be set by the caller
(an idle GPU); the CLI then sees it as device "0" and does not touch the variable.

    CUDA_VISIBLE_DEVICES=<idle> venv_gen/bin/python code/run_seedvr2.py
"""
from __future__ import annotations

import json
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES to an idle GPU"
GSR_CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(GSR_CODE))
import gsr_common as G  # noqa: E402

REPO = os.path.join(G.GSR, "repos", "seedvr2_numz")
MODEL_DIR = os.path.join(G.GSR, "models", "SEEDVR2")
DIT = "seedvr2_ema_3b_fp16.safetensors"
sys.argv = ["inference_cli.py", os.path.join(G.LR_DIR, "D4"), "--dit_model", DIT, "--model_dir", MODEL_DIR,
            "--batch_size", "1", "--seed", str(G.SEED), "--resolution", "1024", "--output_format", "png",
            "--cache_dit", "--cache_vae", "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
            "--vae_encode_tiled", "--vae_decode_tiled"]
sys.path.insert(0, REPO)
import inference_cli as IC  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    args = IC.parse_arguments()
    IC.debug.enabled = False
    assert IC.download_weight(dit_model=args.dit_model, vae_model=IC.DEFAULT_VAE, model_dir=args.model_dir,
                              debug=IC.debug)
    man = pd.read_csv(G.MANIFEST)
    runner_cache = {}
    tmp = os.path.join(G.GSR, "outputs", "_seedvr2_raw")
    times, rows = [], []
    t0 = time.time()
    for r in man.itertuples():
        op = G.out_path("seedvr2", r.cond, r.key)
        if os.path.exists(op):
            continue
        lp = os.path.join(G.GSR, r.lr_path)
        assert G.md5(lp) == r.lr_md5
        raw = os.path.join(tmp, r.cond, r.key + ".png")
        args.resolution = int(r.S)
        t1 = time.time()
        IC.process_single_file(lp, args, ["0"], raw, format_auto_detected=False, runner_cache=runner_cache)
        dt = time.time() - t1
        out = cv2.imread(raw, cv2.IMREAD_COLOR)
        h, w = out.shape[:2]
        if (h, w) != (r.S, r.S):
            out = cv2.resize(out, (r.S, r.S), interpolation=cv2.INTER_CUBIC if h < r.S else cv2.INTER_AREA)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        times.append(dt)
        rows.append(dict(key=r.key, cond=r.cond, raw_w=w, raw_h=h, S=r.S, seconds=round(dt, 2)))
        print(f"{len(times)} {r.cond} {r.key} raw {w}x{h} -> {r.S}  {dt:.1f}s", flush=True)
    import torch
    pd.DataFrame(rows).to_csv(os.path.join(G.GSR, "logs", "seedvr2_per_image.csv"), index=False)
    G.write_json(os.path.join(G.GSR, "logs", "restore_seedvr2.json"), dict(
        repo="numz/ComfyUI-SeedVR2_VideoUpscaler", repo_commit=open(os.path.join(G.GSR, "logs", "seedvr2_commit.txt")).read().strip()
        if os.path.exists(os.path.join(G.GSR, "logs", "seedvr2_commit.txt")) else None,
        dit=DIT, vae=IC.DEFAULT_VAE, weights_repo="numz/SeedVR2_comfyUI (Apache-2.0; ByteDance-Seed/SeedVR2-3B)",
        dit_md5=G.md5(os.path.join(MODEL_DIR, DIT)), vae_md5=G.md5(os.path.join(MODEL_DIR, IC.DEFAULT_VAE)),
        args={k: v for k, v in vars(args).items() if k not in ("input",)}, n=len(times),
        s_per_img=float(np.median(times)) if times else None, wall_s=round(time.time() - t0, 1),
        gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        note="resolution set per image to the reference side S; raw output resized to S x S if needed"))


if __name__ == "__main__":
    main()
