"""gsr_common.py -- shared definitions for the genai_sr study (see README.md, protocol sections 1-8).

Runs on the Mac (paths under wvaq2027/) and on gpu-server (paths under ~/pickglass/genai_sr/),
selected by the GSR_ROOT environment variable (default: the directory of this file).
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GSR = os.environ.get("GSR_ROOT", HERE)               # genai_sr root (Mac: wvaq2027/genai_sr)
WVAQ = os.environ.get("WVAQ_ROOT", os.path.dirname(GSR))  # Mac: wvaq2027 ; server: copy of the needed files

SEED = 20261008
T_C = 0.5740764461064712        # cascade conf-only gate (paper/tables/cascade_thresholds.json fits.vlm9_a10.conf.t_c)
HR_IDS = [1, 5, 9, 15, 17, 18, 25, 27, 28, 30, 33, 38, 39, 45, 47, 48, 49, 50, 52, 53, 54, 60, 62, 64, 65, 67,
          70, 73, 77, 80, 84, 86, 88, 90, 91, 95, 98, 99, 110, 118, 123, 129, 130, 131, 133, 137, 138, 140, 141,
          143, 144, 145, 146]
LEG_NAMES = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
CONDS = ["D4", "D8", "D4J"]
RESTORERS = ["none", "bicubic", "swinir", "realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
PROMPT_ENHANCE = "Enhance and upscale this fabric photo, keep it realistic"

INPUTS = os.path.join(GSR, "inputs")
REF_DIR = os.path.join(INPUTS, "ref")
LR_DIR = os.path.join(INPUTS, "lr")
OUT_DIR = os.path.join(GSR, "outputs")          # restored, S x S, lossless PNG
MANIFEST = os.path.join(INPUTS, "manifest.csv")


def key_of(set_name, ident):
    return f"hr{int(ident)}" if set_name == "R-HR" else f"leg_{ident}"


def ref_path(key):
    ext = ".jpg" if key.startswith("hr") else ".png"
    return os.path.join(REF_DIR, key + ext)


def lr_path(cond, key):
    return os.path.join(LR_DIR, cond, key + ".png")


def out_path(restorer, cond, key):
    return os.path.join(OUT_DIR, restorer, cond, key + ".png")


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def items(restorer, man=None):
    """[(key, set, cond, restorer, path, S)] for restorer in {'clean', 'none', <restorer>} (existing files only)."""
    import pandas as pd
    man = pd.read_csv(MANIFEST) if man is None else man
    out = []
    if restorer == "clean":
        for r in man.drop_duplicates("key").itertuples():
            out.append((r.key, r.set, "clean", "clean", os.path.join(GSR, r.ref_path), int(r.S)))
        return out
    for r in man.itertuples():
        p = os.path.join(GSR, r.lr_path) if restorer == "none" else out_path(restorer, r.cond, r.key)
        if os.path.exists(p):
            out.append((r.key, r.set, r.cond, restorer, p, int(r.S)))
    return out
