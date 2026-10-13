"""run_flux2klein.py -- FLUX.2 [klein] 4B prompted enhancement ("open AI-enhance"), README section 4.

Input: the LR PNG bicubic-upsampled to 1024 x 1024; reference-image editing with the same prompt as the commercial
model; 4 steps, guidance 1.0 (model-card defaults); seed 20261008; output 1024 x 1024 -> resized to S x S.

    CUDA_VISIBLE_DEVICES=<idle> venv_gen/bin/python code/run_flux2klein.py
"""
from __future__ import annotations

import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "set CUDA_VISIBLE_DEVICES to an idle GPU"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gsr_common as G  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
SIDE = 1024


def main():
    import diffusers
    from diffusers import Flux2KleinPipeline
    from huggingface_hub import HfApi
    sha = HfApi().model_info(REPO_ID).sha
    pipe = Flux2KleinPipeline.from_pretrained(REPO_ID, torch_dtype=torch.bfloat16, revision=sha)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    man = pd.read_csv(G.MANIFEST)
    times, rows = [], []
    t0 = time.time()
    for r in man.itertuples():
        op = G.out_path("flux2klein", r.cond, r.key)
        if os.path.exists(op):
            continue
        lp = os.path.join(G.GSR, r.lr_path)
        assert G.md5(lp) == r.lr_md5
        lr = cv2.imread(lp, cv2.IMREAD_COLOR)
        up = cv2.resize(lr, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        img = pipe(prompt=G.PROMPT_ENHANCE, image=[pil], height=SIDE, width=SIDE, guidance_scale=1.0,
                   num_inference_steps=4, generator=torch.Generator(device="cuda").manual_seed(G.SEED)).images[0]
        torch.cuda.synchronize()
        dt = time.time() - t1
        out = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = out.shape[:2]
        out = cv2.resize(out, (r.S, r.S), interpolation=cv2.INTER_CUBIC if h < r.S else cv2.INTER_AREA)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        cv2.imwrite(op, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        times.append(dt)
        rows.append(dict(key=r.key, cond=r.cond, raw_w=w, raw_h=h, S=r.S, seconds=round(dt, 2)))
        print(f"{len(times)} {r.cond} {r.key} {dt:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(G.GSR, "logs", "flux2klein_per_image.csv"), index=False)
    G.write_json(os.path.join(G.GSR, "logs", "restore_flux2klein.json"), dict(
        repo_id=REPO_ID, revision=sha, licence="Apache-2.0", diffusers=diffusers.__version__, torch=torch.__version__,
        prompt=G.PROMPT_ENHANCE, steps=4, guidance_scale=1.0, side=SIDE, seed=G.SEED, dtype="bfloat16",
        input="LR bicubic-upsampled to 1024^2 as the single reference image", n=len(times),
        s_per_img=float(np.median(times)) if times else None, wall_s=round(time.time() - t0, 1),
        gpu=torch.cuda.get_device_name(0), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES")))


if __name__ == "__main__":
    main()
