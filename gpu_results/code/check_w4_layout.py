"""CPU check that the MLX 4-bit checkpoint dequantises onto the HF weights (nibble order, names)."""
import glob
import json
import os

import numpy as np
from safetensors import safe_open

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = glob.glob(f"{R}/hf_home/hub/models--allenai--Molmo2-8B/snapshots/*")[0]
M = glob.glob(f"{R}/hf_home/hub/models--mlx-community--Molmo2-8B-4bit/snapshots/*")[0]
idx = json.load(open(f"{H}/model.safetensors.index.json"))["weight_map"]
midx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
for base, hf in [("language_model.model.blocks.0.self_attn.att_proj", "model.transformer.blocks.0.self_attn.att_proj.weight"),
                 ("language_model.model.blocks.17.mlp.ff_proj", "model.transformer.blocks.17.mlp.ff_proj.weight"),
                 ("language_model.model.blocks.35.mlp.ff_out", "model.transformer.blocks.35.mlp.ff_out.weight"),
                 ("language_model.lm_head", "lm_head.weight")]:
    with safe_open(f"{H}/{idx[hf]}", framework="np") as f:
        W = f.get_tensor(hf)[:512]
    with safe_open(f"{M}/{midx[base + '.weight']}", framework="np") as f:
        P = f.get_tensor(base + ".weight")[:512]
        S = f.get_tensor(base + ".scales")[:512].astype(np.float32)
        B = f.get_tensor(base + ".biases")[:512].astype(np.float32)
    for order in ("lsb_first", "msb_first"):
        sh = np.arange(0, 32, 4, dtype=np.uint32) if order == "lsb_first" else np.arange(28, -4, -4).astype(np.uint32)
        q = ((P[..., None] >> sh) & 0xF).reshape(P.shape[0], -1).astype(np.float32)
        gs = q.shape[1] // S.shape[1]
        Wq = (q.reshape(q.shape[0], -1, gs) * S[..., None] + B[..., None]).reshape(q.shape)
        rel = np.linalg.norm(Wq - W) / np.linalg.norm(W)
        print(base, order, W.dtype, W.shape, "gs", gs, "rel err %.4f" % rel)
