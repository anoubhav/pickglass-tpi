"""run_seedvar.py -- EXPLORATORY (not pre-registered): seed sensitivity of the open generative restorers.

Same settings as the main runs (run_flux2klein.py / run_seedvr2.py), only the seed changes; outputs go to
outputs/<model>_s<seed>/<cond>/<key>.png so every scorer treats them as an extra 'restorer'.

    CUDA_VISIBLE_DEVICES=<idle> venv_gen/bin/python code/run_seedvar.py --model flux2klein --seeds 1,2 --conds D4,D8
    CUDA_VISIBLE_DEVICES=<idle> venv_gen/bin/python code/run_seedvar.py --model seedvr2 --seeds 1,2 --conds D4,D8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES to an idle GPU"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["flux2klein", "seedvr2"])
ap.add_argument("--seeds", default="1,2")
ap.add_argument("--conds", default="D4,D8")
A = ap.parse_args()

if A.model == "seedvr2":
    REPO = os.path.join(G.GSR, "repos", "seedvr2_numz")
    sys.argv = ["inference_cli.py", os.path.join(G.LR_DIR, "D4"), "--dit_model", "seedvr2_ema_3b_fp16.safetensors",
                "--model_dir", os.path.join(G.GSR, "models", "SEEDVR2"), "--batch_size", "1", "--seed", "0",
                "--resolution", "1024", "--output_format", "png", "--cache_dit", "--cache_vae",
                "--dit_offload_device", "cpu", "--vae_offload_device", "cpu", "--vae_encode_tiled", "--vae_decode_tiled"]
    sys.path.insert(0, REPO)
    import inference_cli as IC  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


def to_S(out, S):
    h = out.shape[0]
    return out if out.shape[:2] == (S, S) else cv2.resize(out, (S, S), interpolation=cv2.INTER_CUBIC if h < S else cv2.INTER_AREA)


def main():
    man = pd.read_csv(G.MANIFEST)
    man = man[man.cond.isin(A.conds.split(","))]
    if A.model == "flux2klein":
        from diffusers import Flux2KleinPipeline
        from PIL import Image
        pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16,
                                                  revision="e7b7dc27f91deacad38e78976d1f2b499d76a294").to("cuda")
        pipe.set_progress_bar_config(disable=True)
    else:
        args = IC.parse_arguments()
        IC.debug.enabled = False
        cache = {}
    t0, n = time.time(), 0
    for seed in [int(s) for s in A.seeds.split(",")]:
        name = f"{A.model}_s{seed}"
        for r in man.itertuples():
            op = G.out_path(name, r.cond, r.key)
            if os.path.exists(op):
                continue
            lp = os.path.join(G.GSR, r.lr_path)
            if A.model == "flux2klein":
                lr = cv2.imread(lp, cv2.IMREAD_COLOR)
                pil = Image.fromarray(cv2.cvtColor(cv2.resize(lr, (1024, 1024), interpolation=cv2.INTER_CUBIC),
                                                   cv2.COLOR_BGR2RGB))
                img = pipe(prompt=G.PROMPT_ENHANCE, image=[pil], height=1024, width=1024, guidance_scale=1.0,
                           num_inference_steps=4, generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
                out = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                raw = os.path.join(G.OUT_DIR, f"_{name}_raw", r.cond, r.key + ".png")
                args.resolution, args.seed = int(r.S), seed
                IC.process_single_file(lp, args, ["0"], raw, format_auto_detected=False, runner_cache=cache)
                out = cv2.imread(raw, cv2.IMREAD_COLOR)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            cv2.imwrite(op, to_S(out, r.S), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n += 1
            if n % 20 == 0:
                print(f"{name} {n} {time.time() - t0:.0f}s", flush=True)
    G.write_json(os.path.join(G.GSR, "logs", f"seedvar_{A.model}.json"),
                 dict(model=A.model, seeds=A.seeds, conds=A.conds, n=n, wall_s=round(time.time() - t0, 1),
                      gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")))


if __name__ == "__main__":
    main()
