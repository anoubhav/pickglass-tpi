#!/usr/bin/env python
"""Train the tile CNN on synthetic pick-glass crops only.

Used (a) as the synthetic-only zero-shot baseline and (b) as the
initialisation for sim-to-real fine-tuning in train_cv.py.

    python train_synth.py --synth-dir data/synth_train --backbone resnet18 \
        --minutes 6 --out runs/synth_resnet18
    # then evaluate (synthetic held-out + 13 legacy images)
    python evaluate.py --ckpt runs/synth_resnet18/model.pt --synth-val data/synth_val \
        --legacy --out runs/synth_resnet18/eval
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import cnn_common as C
from evaluate import synth_items

HERE = os.path.dirname(os.path.abspath(__file__))


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    np.random.seed(torch.initial_seed() % 2 ** 32)


def train_synth(synth_dir, backbone="resnet18", minutes=6.0, steps=None, batch=32, tiles_per_image=4,
                lr=3e-4, workers=4, device="auto", pretrained=True, rot90=True, out=None, seed=0,
                log=print, aniso=0.5, lam_ratio=1.0, augment="existing", augment_drop=()):
    """synth_dir: one folder or a comma-separated list of folders (labels.csv each).
    augment: 'existing' (original recipe) or 'whitelist' / 'whitelist_ext' (aug_pipeline.AugTileDataset,
    synthetic labels treated as intrinsic density; see train_aug.py / RESULTS_AUG.md)."""
    C.seed_all(seed)
    dev = C.get_device(device)
    items = []
    for d in str(synth_dir).split(","):
        items += synth_items(d.strip())[0]
    if augment == "existing":
        ds = C.TileDataset(items, tiles_per_image=tiles_per_image, train=True, rot90=rot90, seed=seed,
                           aniso=aniso)
    else:
        import aug_pipeline as AP
        ds = AP.AugTileDataset(AP.to_aug_items(items, "density", "synth"), tiles_per_image=tiles_per_image,
                               policy=augment, drop=augment_drop, aniso=aniso, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                                     drop_last=True, persistent_workers=workers > 0,
                                     worker_init_fn=worker_init, prefetch_factor=4 if workers else None)
    model = C.build_model(backbone, pretrained=pretrained)
    log(f"[synth] {backbone} on {dev}: {len(items)} synthetic crops, {len(ds)} tiles/epoch, "
        f"budget {minutes} min / {steps} steps; aniso {aniso}, lam_ratio {lam_ratio}")
    t0 = time.time()
    hist = C.train_loop(model, dl, dev, epochs=100, max_steps=steps or 10 ** 9, max_minutes=minutes,
                        lr=lr, log=log, lam_ratio=lam_ratio)
    meta = {"backbone": backbone, "synth_dir": synth_dir, "n_synth": len(items), "augment": augment,
            "augment_drop": list(augment_drop), "steps": hist[-1][0] if hist else 0,
            "train_minutes": (time.time() - t0) / 60, "batch": batch, "lr": lr, "rot90": rot90,
            "pretrained_imagenet": pretrained, "aniso": aniso, "lam_ratio": lam_ratio, "hist": hist}
    if out:
        C.save_ckpt(model, os.path.join(out, "model.pt"), meta)
        with open(os.path.join(out, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    return model, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", default=os.path.join(HERE, "data", "synth_train"),
                    help="folder or comma-separated folders, e.g. data/synth_train,data/synth_train_v2")
    ap.add_argument("--backbone", default="resnet18", help="any timm name, e.g. resnet18, efficientnet_b0")
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tiles-per-image", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-rot90", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aniso", type=float, default=0.5,
                    help="log-range of independent x/y px-per-inch jitter (0 = isotropic, the v1 recipe)")
    ap.add_argument("--lam-ratio", type=float, default=1.0, help="weight of the log(h/v) loss (0 = v1 recipe)")
    ap.add_argument("--augment", default="existing", choices=("existing", "whitelist", "whitelist_ext"),
                    help="'existing' (original recipe) or the count-preserving whitelist pipeline (aug_pipeline.py)")
    ap.add_argument("--augment-drop", default="", help="comma list of families to drop (whitelist* only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    train_synth(a.synth_dir, a.backbone, a.minutes, a.steps, a.batch, a.tiles_per_image, a.lr, a.workers,
                a.device, not a.no_pretrained, not a.no_rot90, a.out, a.seed, log=log, aniso=a.aniso,
                lam_ratio=a.lam_ratio, augment=a.augment,
                augment_drop=tuple(x for x in a.augment_drop.split(",") if x))
    print("saved", os.path.join(a.out, "model.pt"))


if __name__ == "__main__":
    main()
