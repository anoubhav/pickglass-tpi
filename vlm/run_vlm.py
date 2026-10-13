"""Run local VLMs (MLX, Apple silicon) on the fabric thread-counting task.

ROSTER: only `molmo2_8b` (Molmo2-8B, MLX 4-bit) is a reported local model
(MODEL_ROSTER.md).  The other entries of models.json (qwen35_9b, gemma4_12b,
internvl3_2b) were harness smoke tests during development and are NOT reported;
`--model all` therefore runs the roster only.  Qwen3.8-27B and Gemma-4-26B-A4B are
run through OpenRouter instead (openrouter_client.py) to save disk.

Usage (from this folder, with the venv_vlm python):
    python run_vlm.py --model molmo2_8b --set legacy            # P1,P2,P3,P4,P4q on the 20 legacy crops
    python run_vlm.py --model molmo2_8b --set legacy_x4         # P1 on the x4-downscaled legacy crops
    python run_vlm.py --model molmo2_8b --set rect150           # P1 on the 150 rectified crops
Options: --prompts P1,P2  --ids a1,a2  --limit N  --force (ignore cached records)

Views: P1/P2 whole crop; P3 central quarter-inch patch (count x 4); P4 red guide line
through the whole crop (n points = count); P4q guide line through the central
quarter-inch patch (n points x 4).

Every generation is appended as one JSON line to raw/{model}__{set}.jsonl with the
raw text, the parsed result, prompt id+version, model repo+revision, input size,
token counts, wall-clock seconds and peak memory.  Re-running resumes: jobs whose
(image, view, prompt) key is already in the file are skipped.

Decoding is greedy (temperature 0) for reproducibility; Qwen3.5 "thinking" is off
(the chat template default in mlx-vlm), so P2 reasoning is visible in the answer.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_vlm as D            # noqa: E402
import prompts as PR            # noqa: E402
from parse_output import parse_counts, parse_points   # noqa: E402

MODELS = {k: v for k, v in json.load(open(os.path.join(HERE, "models.json"))).items() if not k.startswith("_")}
RAW_DIR = os.path.join(HERE, "raw")
INPUT_DIR = os.path.join(HERE, "inputs")

# Per-model input side (px) for square crops = the model's native maximum for a
# square image (see README):
#   Qwen3.5      : dynamic res, 32 px per visual token -> 1344 px = 42x42 = 1764 tokens (practical cap)
#   Gemma 4      : max_soft_tokens raised from the default 280 to 1120 (the largest documented
#                  budget, mm_posemb_size=1120); 48 px per soft token -> 33x33 grid = 1584 px
#   Molmo2       : max_crops=8 overlapping 378-px crops -> a square gets a 2x2 tiling = 644 px
#   InternVL3    : 448-px tiles, max 12 -> 3x3 tiles + thumbnail = 1344 px
MODEL_CFG = {
    "qwen35_9b":    dict(side=1344, p4=("P4h_json", "P4v_json")),
    "gemma4_12b":   dict(side=1584, p4=None, max_soft_tokens=1120),
    "molmo2_8b":    dict(side=644,  p4=("P4h", "P4v")),
    "internvl3_2b": dict(side=1344, p4=None),
}

MAX_FAILS = 3        # give up on a job after this many logged failures
METAL_EXIT = 3       # exit code: Metal error -> the process' GPU queue is poisoned, driver restarts us
PREFILL_STEP = 512   # smaller prefill chunks -> shorter Metal command buffers (avoids GPU timeouts)

ROSTER_LOCAL = ["molmo2_8b"]
ANN_CROPS = os.path.join(os.path.dirname(HERE), "annotation_tool", "data", "crops")

DEFAULT_PROMPTS = {
    "legacy": ["P1", "P2", "P3", "P4", "P4q"],
    "legacy_x4": ["P1"],
    "legacy_raw": ["P1"],
    "rect150": ["P1"],
    "full150": ["P1"],
}


# ----------------------------------------------------------------------------
# job construction
# ----------------------------------------------------------------------------
def _write(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        cv2.imwrite(path, img)
    return path


def build_jobs(model_key: str, set_name: str, prompt_ids: list[str], limit: int | None, ids=None):
    """Return a list of dict(image_id, view, prompt_id, image_path, input_wh)."""
    cfg = MODEL_CFG[model_key]
    side = cfg["side"]
    jobs = []

    def add(image_id, view, pid, img):
        p = _write(img, os.path.join(INPUT_DIR, model_key, set_name, f"{image_id}__{view}.png"))
        jobs.append(dict(image_id=str(image_id), view=view, prompt_id=pid, image_path=p,
                         input_wh=[int(img.shape[1]), int(img.shape[0])]))

    if set_name in ("legacy", "legacy_x4"):
        names = ids or D.LEGACY_NAMES
        names = names[:limit] if limit else names
        for name in names:
            crop = cv2.imread(os.path.join(D.LEGACY_CROP_DIR, f"{name}.png"))
            full = D.resize_square(crop, side)
            for pid in prompt_ids:
                if set_name == "legacy_x4":
                    assert pid == "P1"
                    h, w = crop.shape[:2]
                    small = cv2.resize(crop, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
                    add(name, "crop_x4", pid, D.resize_square(small, side))
                elif pid in ("P1", "P2"):
                    add(name, "crop", pid, full)
                elif pid == "P3":
                    add(name, "qc", "P3", D.resize_square(D.quarter_patch(crop, "c"), side))
                elif pid in ("P4", "P4q"):
                    if cfg["p4"] is None:
                        continue
                    ph, pv = cfg["p4"]
                    if pid == "P4":
                        add(name, "guide_h", ph, D.guide_line_view(full, "h"))
                        add(name, "guide_v", pv, D.guide_line_view(full, "v"))
                    else:
                        q = D.resize_square(D.quarter_patch(crop, "c"), side)
                        add(name, "qc_guide_h", ph, D.guide_line_view(q, "h"))
                        add(name, "qc_guide_v", pv, D.guide_line_view(q, "v"))
    elif set_name == "legacy_raw":
        ids = D.LEGACY_NAMES[:limit] if limit else D.LEGACY_NAMES
        for name in ids:
            img = cv2.imread(D.legacy_path(name))
            for pid in prompt_ids:
                add(name, "raw", pid, D.resize_max(img, side))
    elif set_name == "rect150":
        idx = [int(i) for i in ids] if ids else list(range(150))
        idx = idx[:limit] if limit else idx
        for i in idx:
            # the annotation tool's crops (common/rectify.py contract), i.e. exactly what the
            # human annotators count on
            crop = cv2.imread(os.path.join(ANN_CROPS, f"{i}.jpg"))
            for pid in prompt_ids:
                add(i, "crop", pid, D.resize_square(crop, side))
    elif set_name == "full150":
        ids = list(range(150))[:limit] if limit else range(150)
        for i in ids:
            img = cv2.imread(os.path.join(D.IMG150_DIR, f"{i}.jpeg"))
            for pid in prompt_ids:
                add(i, "full", pid, D.resize_max(img, side))
    else:
        raise ValueError(set_name)
    return jobs


# ----------------------------------------------------------------------------
# model wrapper
# ----------------------------------------------------------------------------
class MLXVLM:
    def __init__(self, key: str):
        from huggingface_hub import snapshot_download
        from mlx_vlm import load
        spec = MODELS[key]
        self.key, self.spec = key, spec
        t = time.time()
        path = snapshot_download(spec["repo"], revision=spec["revision"])
        self.model, self.processor = load(path)
        cfg = MODEL_CFG[key]
        if "max_soft_tokens" in cfg:       # Gemma 4 visual token budget
            ip = self.processor.image_processor
            ip.max_soft_tokens = cfg["max_soft_tokens"]
            if hasattr(self.processor, "image_seq_length"):
                self.processor.image_seq_length = cfg["max_soft_tokens"]
        self.load_seconds = time.time() - t

    def __call__(self, text: str, image_path: str, max_tokens: int):
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(self.processor, self.model.config, text, num_images=1)
        last_err = None
        for attempt in range(3):            # the GPU is shared; retry Metal timeouts
            try:
                t = time.time()
                r = generate(self.model, self.processor, prompt, image=[image_path],
                             max_tokens=max_tokens, temperature=0.0, verbose=False,
                             prefill_step_size=PREFILL_STEP)
                dt = time.time() - t
                break
            except RuntimeError as e:
                if "METAL" in str(e).upper():
                    raise               # GPU queue of this process is unusable after a Metal error
                last_err = e
                mx.clear_cache()
                time.sleep(10 * (attempt + 1))
        else:
            raise last_err
        out = dict(text=r.text, seconds=dt, prompt_tokens=int(getattr(r, "prompt_tokens", 0) or 0),
                   generation_tokens=int(getattr(r, "generation_tokens", 0) or 0),
                   peak_memory_gb=float(getattr(r, "peak_memory", 0.0) or 0.0),
                   finish_reason=getattr(r, "finish_reason", None))
        mx.clear_cache()
        return out


# ----------------------------------------------------------------------------
def run(model_key: str, set_name: str, prompt_ids: list[str], limit=None, force=False, ids=None):
    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, f"{model_key}__{set_name}.jsonl")
    done, fails = set(), {}
    if os.path.exists(out_path) and not force:
        for line in open(out_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            key = (r["image_id"], r["view"], r["prompt_id"])
            if r.get("error"):
                fails[key] = fails.get(key, 0) + 1
            else:
                done.add(key)
    # a job that already failed MAX_FAILS times is given up (its error records stay in the log)
    jobs = [j for j in build_jobs(model_key, set_name, prompt_ids, limit, ids)
            if (j["image_id"], j["view"], j["prompt_id"]) not in done
            and fails.get((j["image_id"], j["view"], j["prompt_id"]), 0) < MAX_FAILS]
    print(f"[{model_key}/{set_name}] {len(jobs)} jobs to run ({len(done)} cached) -> {out_path}", flush=True)
    if not jobs:
        return out_path
    vlm = MLXVLM(model_key)
    print(f"  loaded {vlm.spec['repo']} in {vlm.load_seconds:.1f}s", flush=True)
    # warm-up (compilation / kernel caching) so timings are steady-state
    try:
        vlm("Describe the image in one word.", jobs[0]["image_path"], 4)
    except RuntimeError as e:
        print("  warm-up failed:", repr(e)[:200], flush=True)
        os._exit(METAL_EXIT)

    with open(out_path, "a") as f:
        for k, j in enumerate(jobs):
            p = PR.get(j["prompt_id"])
            try:
                g = vlm(p["text"], j["image_path"], p["max_tokens"])
                err = None
            except Exception as e:      # log the failure; Metal errors poison the process -> restart
                g = dict(text="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0,
                         peak_memory_gb=0.0, finish_reason="error")
                err = repr(e)
            parsed = parse_points(g["text"]) if j["prompt_id"].startswith("P4") else parse_counts(g["text"])
            if "points" in parsed:
                parsed = {k2: v for k2, v in parsed.items() if k2 != "points"} | {"points": parsed["points"][:400]}
            rec = dict(model=model_key, repo=vlm.spec["repo"], revision=vlm.spec["revision"],
                       set=set_name, image_id=j["image_id"], view=j["view"],
                       prompt_id=j["prompt_id"], prompt_version=p["version"],
                       prompt_set_version=p["set_version"], max_tokens=p["max_tokens"],
                       input_wh=j["input_wh"], image_path=os.path.relpath(j["image_path"], HERE),
                       raw=g["text"], parsed=parsed, error=err,
                       seconds=g["seconds"], prompt_tokens=g["prompt_tokens"],
                       generation_tokens=g["generation_tokens"], peak_memory_gb=g["peak_memory_gb"],
                       finish_reason=g["finish_reason"], host=platform.node(),
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if err and "METAL" in err.upper():
                print(f"  Metal error on {j['image_id']}/{j['view']}/{j['prompt_id']} -> exiting for restart", flush=True)
                f.close()
                os._exit(METAL_EXIT)
            short = g["text"].replace("\n", " ")[:110]
            print(f"  [{k+1}/{len(jobs)}] {j['image_id']}/{j['view']}/{j['prompt_id']} "
                  f"{g['seconds']:.1f}s gen={g['generation_tokens']} :: {short}", flush=True)
    del vlm
    gc.collect()
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in models.json or 'all'")
    ap.add_argument("--set", required=True, choices=list(DEFAULT_PROMPTS))
    ap.add_argument("--prompts", default=None, help="comma list, e.g. P1,P2,P3,P4")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    keys = ROSTER_LOCAL if a.model == "all" else a.model.split(",")
    pids = a.prompts.split(",") if a.prompts else DEFAULT_PROMPTS[a.set]
    for key in keys:
        run(key, a.set, pids, a.limit, a.force, a.ids.split(",") if a.ids else None)
