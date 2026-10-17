"""Molmo2-8B at reference precision (bf16, Hugging Face transformers, trust_remote_code) on one
CUDA GPU, with the SAME prompts, views, parsers and raw-record schema as the Mac MLX harness
(<repo root>/vlm/run_vlm.py).  Records are appended to raw/molmo2_8b_bf16__{set}[__{tag}].jsonl.

Sets / views (identical construction to run_vlm.build_jobs; side = 644 px, Molmo2's native
maximum for a square image: max_crops = 8 -> 2 x 2 tiling of 378-px crops with 4-patch overlap
margins = 2*266 + 112 = 644 px; larger inputs are bilinearly downsampled by the processor
WITHOUT antialiasing, so we pre-resize with INTER_AREA exactly like the Mac):
  legacy   20 frame-trimmed legacy crops (Mac vlm/crops/legacy/*.png)
           crop/P1, crop/P2, qc/P3, qd0..qd3/P3 (-> P3s), guide_h/P4h, guide_v/P4v,
           qc_guide_h/P4h, qc_guide_v/P4v (-> P4q)
  rect150  150 annotation-tool crops: crop/P1, qc/P3, qd0..qd3/P3, guide_h/P4h, guide_v/P4v
  Stability (tag 'stability', repeat 1/2): crop/P1 + guide_h/P4h + guide_v/P4v with the crop
           pre-resized to 588 px (repeat 1) and 700 px (repeat 2) instead of 644 (repeat 0 =
           main run); the processor then maps every square input to its 644-px 2x2 tiling.
           Images: the 20 legacy crops and 20 rect150 crops (ids 0, 8, ..., 144, 149).

Decoding: greedy (do_sample=False), max_new_tokens = the prompt's max_tokens (prompts.py).

    CUDA_VISIBLE_DEVICES=0 python run_molmo2_hf.py --plan main      # legacy + stability + rect150
    python run_molmo2_hf.py --set legacy --prompts P1 --ids a1,a2   # ad hoc
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlm_common as C                                      # noqa: E402
from vlm_common import PR, parse_counts, parse_points       # noqa: E402

ROOT = C.ROOT
RAW_DIR = os.path.join(ROOT, "raw")
INPUT_DIR = os.path.join(ROOT, "inputs")
MAC_LEGACY_INPUTS = os.path.join(ROOT, "data", "mac_inputs_legacy")   # Mac vlm/inputs/molmo2_8b/legacy
MODEL_KEY = "molmo2_8b_bf16"
REPO = "allenai/Molmo2-8B"
REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
SIDE = 644
STAB_SIDES = {1: 588, 2: 700}
STAB_RECT_IDS = [str(i) for i in range(0, 150, 8)] + ["149"]   # 0, 8, ..., 144, 149 = 20 crops
P4 = ("P4h", "P4v")


# ----------------------------------------------------------------------------
def _write(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        cv2.imwrite(path, img)
    return path


def _load(set_name, image_id):
    if set_name == "legacy":
        return cv2.imread(os.path.join(C.LEGACY_CROP_DIR, f"{image_id}.png"))
    return cv2.imread(os.path.join(C.CROPS150_DIR, f"{image_id}.jpg"))


def build_jobs(set_name, prompt_ids, ids=None, repeat=0):
    side = SIDE if repeat == 0 else STAB_SIDES[repeat]
    if ids is None:
        ids = C.LEGACY_NAMES if set_name == "legacy" else [str(i) for i in range(150)]
    sub = set_name + ("" if repeat == 0 else f"_rs{side}")
    jobs = []

    def add(image_id, view, pid, img):
        src = "generated"
        mac = os.path.join(MAC_LEGACY_INPUTS, f"{image_id}__{view}.png")
        if set_name == "legacy" and repeat == 0 and os.path.exists(mac):
            # use the exact pixels the Mac MLX-4bit run saw (our cv2 differs by <=1 grey level on
            # a few % of pixels), so bf16-vs-4bit differences are due to the model only
            img, src = cv2.imread(mac), "mac_mlx_input"
        p = _write(img, os.path.join(INPUT_DIR, "molmo2_8b_bf16", sub, f"{image_id}__{view}.png"))
        jobs.append(dict(set=set_name, repeat=repeat, image_id=str(image_id), view=view, prompt_id=pid,
                         image_path=p, input_wh=[int(img.shape[1]), int(img.shape[0])], resize_side=side,
                         input_source=src))

    for name in ids:
        crop = _load(set_name, name)
        assert crop is not None, (set_name, name)
        full = C.resize_square(crop, side)
        for pid in prompt_ids:
            if pid in ("P1", "P2"):
                add(name, "crop", pid, full)
            elif pid == "P3":
                add(name, "qc", "P3", C.resize_square(C.quarter_patch(crop, "c"), side))
            elif pid == "P3s":
                for k in range(4):
                    add(name, f"qd{k}", "P3", C.resize_square(C.quarter_patch(crop, f"d{k}"), side))
            elif pid == "P4":
                add(name, "guide_h", P4[0], C.guide_line_view(full, "h"))
                add(name, "guide_v", P4[1], C.guide_line_view(full, "v"))
            elif pid == "P4q":
                q = C.resize_square(C.quarter_patch(crop, "c"), side)
                add(name, "qc_guide_h", P4[0], C.guide_line_view(q, "h"))
                add(name, "qc_guide_v", P4[1], C.guide_line_view(q, "v"))
            else:
                raise ValueError(pid)
    return jobs


# ----------------------------------------------------------------------------
MLX_REPO, MLX_REVISION = "mlx-community/Molmo2-8B-4bit", "4fcbe926577641af12aa5086cf5c292db9dd9730"


def patch_mlx_w4(model, log_path):
    """Replace every language-model Linear weight (att_proj, attn_out, ff_proj, ff_out, lm_head)
    by the DEQUANTISED weight of the Mac's MLX checkpoint (affine 4-bit, group 64, fp16
    scales/biases; MLX packs 8 nibbles per uint32, lowest bits first).  Everything else (vision
    tower, connector, embeddings, norms, preprocessing, prompt, kernels) stays bf16/HF, so the
    difference to the bf16 run isolates the effect of the 4-bit weights."""
    import glob as _glob
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path = snapshot_download(MLX_REPO, revision=MLX_REVISION, token=False,
                             allow_patterns=["*.safetensors", "*.json"])
    params = dict(model.named_parameters())
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int32)
    stats, n = [], 0
    for f in sorted(_glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="np") as fh:
            keys = set(fh.keys())
            for k in sorted(keys):
                if not k.endswith(".scales"):
                    continue
                base = k[:-len(".scales")]
                hf = ("lm_head" if base == "language_model.lm_head" else
                      base.replace("language_model.model.", "model.transformer.")) + ".weight"
                W = params[hf]
                P = fh.get_tensor(base + ".weight").view(np.int32)
                S_ = fh.get_tensor(base + ".scales")
                B_ = fh.get_tensor(base + ".biases")
                assert P.shape[0] == W.shape[0] and P.shape[1] * 8 == W.shape[1], (hf, P.shape, W.shape)
                gs = W.shape[1] // S_.shape[1]
                se = sa = mx = rn = 0.0
                for r0 in range(0, W.shape[0], 4096):          # row chunks keep GPU memory small
                    r1 = min(r0 + 4096, W.shape[0])
                    packed = torch.from_numpy(np.ascontiguousarray(P[r0:r1])).cuda()
                    sc = torch.from_numpy(np.ascontiguousarray(S_[r0:r1])).cuda().float()
                    bi = torch.from_numpy(np.ascontiguousarray(B_[r0:r1])).cuda().float()
                    q = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(r1 - r0, -1).float()
                    Wq = (q.view(r1 - r0, -1, gs) * sc.unsqueeze(-1) + bi.unsqueeze(-1)).view(r1 - r0, -1)
                    ref = W[r0:r1].detach().float()
                    d = (Wq - ref)
                    se += float((d * d).sum()); rn += float((ref * ref).sum()); sa += float(d.abs().sum())
                    mx = max(mx, float((d.abs().view(r1 - r0, -1, gs) / (sc.unsqueeze(-1) / 2 + 1e-12)).max()))
                    with torch.no_grad():
                        W[r0:r1].copy_(Wq.to(W.dtype))
                    del packed, sc, bi, q, Wq, ref, d
                rel = (se / rn) ** 0.5
                assert rel < 0.25, f"{hf}: rel err {rel:.3f} -> layout/mapping wrong"
                stats.append(dict(param=hf, mlx=base, shape=list(W.shape), group_size=gs, rel_fro_err=rel,
                                  mean_abs_err=sa / W.numel(), max_abs_err_over_half_step=mx))
                n += 1
    torch.cuda.empty_cache()
    json.dump(dict(n_patched=n, layers=stats), open(log_path, "w"), indent=1)
    rels = [x["rel_fro_err"] for x in stats]
    print(f"patched {n} Linear weights with MLX 4-bit dequantised values; rel. Frobenius error "
          f"median {np.median(rels):.4f} max {np.max(rels):.4f}", flush=True)
    return dict(w4_source=f"{MLX_REPO}@{MLX_REVISION[:7]}", w4_n_patched=n,
                w4_rel_fro_err_median=float(np.median(rels)))


class HFMolmo2:
    def __init__(self, variant="bf16"):
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        t = time.time()
        path = snapshot_download(REPO, revision=REVISION, token=False)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0}).eval()
        self.load_seconds = time.time() - t
        extra = {}
        if variant == "w4mlx":
            extra = patch_mlx_w4(self.model, os.path.join(ROOT, "logs", "w4mlx_dequant_check.json"))
        self.meta = dict(backend="transformers", dtype="bfloat16", torch=torch.__version__,
                         transformers=transformers.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0),
                         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                         attn_implementation=getattr(self.model.config, "_attn_implementation", None),
                         max_crops=getattr(self.processor.image_processor, "max_crops", None),
                         variant=variant, **extra)
        self.eos = self.model.generation_config.eos_token_id

    def __call__(self, text, image_path, max_tokens):
        from PIL import Image
        torch = self.torch
        img = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [dict(type="text", text=text), dict(type="image", image=img)]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        n_in = int(inputs["input_ids"].shape[1])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t
        gen = out[0, n_in:]
        eos = set(self.eos if isinstance(self.eos, (list, tuple)) else [self.eos])
        hit_eos = len(gen) > 0 and int(gen[-1]) in eos
        n_gen = int(len(gen)) - (1 if hit_eos else 0)
        txt = self.processor.tokenizer.decode(gen, skip_special_tokens=True)
        return dict(text=txt, seconds=dt, prompt_tokens=n_in, generation_tokens=n_gen,
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                    finish_reason="stop" if hit_eos or len(gen) < max_tokens else "length")


# ----------------------------------------------------------------------------
def out_path_for(set_name, tag):
    return os.path.join(RAW_DIR, f"{MODEL_KEY}__{set_name}" + (f"__{tag}" if tag else "") + ".jsonl")


def done_keys(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("error"):
                done.add((r["image_id"], r["view"], r["prompt_id"], r.get("repeat", 0)))
    return done


def run_jobs(vlm, jobs, path):
    os.makedirs(RAW_DIR, exist_ok=True)
    done = done_keys(path)
    jobs = [j for j in jobs if (j["image_id"], j["view"], j["prompt_id"], j["repeat"]) not in done]
    print(f"[{os.path.basename(path)}] {len(jobs)} jobs to run ({len(done)} cached)", flush=True)
    with open(path, "a") as f:
        for k, j in enumerate(jobs):
            p = PR.get(j["prompt_id"])
            try:
                g = vlm(p["text"], j["image_path"], p["max_tokens"])
                err = None
            except Exception as e:                      # log and continue (OOM etc.)
                g = dict(text="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0,
                         peak_memory_gb=0.0, finish_reason="error")
                err = repr(e)
                vlm.torch.cuda.empty_cache()
            parsed = parse_points(g["text"]) if j["prompt_id"].startswith("P4") else parse_counts(g["text"])
            if "points" in parsed:
                parsed = {k2: v for k2, v in parsed.items() if k2 != "points"} | {"points": parsed["points"][:400]}
            rec = dict(model=MODEL_KEY, repo=REPO, revision=REVISION,
                       set=j["set"], image_id=j["image_id"], view=j["view"],
                       prompt_id=j["prompt_id"], prompt_version=p["version"],
                       prompt_set_version=p["set_version"], max_tokens=p["max_tokens"],
                       input_wh=j["input_wh"], image_path=os.path.relpath(j["image_path"], ROOT),
                       raw=g["text"], parsed=parsed, error=err,
                       seconds=g["seconds"], prompt_tokens=g["prompt_tokens"],
                       generation_tokens=g["generation_tokens"], peak_memory_gb=g["peak_memory_gb"],
                       finish_reason=g["finish_reason"], host=platform.node(),
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       repeat=j["repeat"], resize_side=j["resize_side"], input_source=j["input_source"],
                       **vlm.meta)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            short = g["text"].replace("\n", " ")[:100]
            print(f"  [{k+1}/{len(jobs)}] r{j['repeat']} {j['image_id']}/{j['view']}/{j['prompt_id']} "
                  f"{g['seconds']:.1f}s in={g['prompt_tokens']} gen={g['generation_tokens']} "
                  f"{g['finish_reason']} :: {short}", flush=True)


PLANS = {
    # (set, prompts, ids, repeat, tag) in execution order
    "main": [
        ("legacy", ["P1", "P4", "P3", "P4q", "P2", "P3s"], None, 0, None),
        ("legacy", ["P1", "P4"], None, 1, "stability"),
        ("legacy", ["P1", "P4"], None, 2, "stability"),
        ("rect150", ["P1", "P4"], None, 0, None),
        ("rect150", ["P3"], None, 0, None),
        ("rect150", ["P1", "P4"], STAB_RECT_IDS, 1, "stability"),
        ("rect150", ["P1", "P4"], STAB_RECT_IDS, 2, "stability"),
        ("rect150", ["P3s"], None, 0, None),
    ],
    "smoke": [("legacy", ["P1", "P4"], ["a1", "r7"], 0, "smoke")],
    # quantisation isolation: bf16 model with the Mac's 4-bit LM weights (dequantised)
    "w4mlx": [("legacy", ["P1", "P4", "P3", "P4q", "P2"], None, 0, None),
              ("rect150", ["P1"], None, 0, None),
              ("rect150", ["P4"], STAB_RECT_IDS, 0, None)],
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, choices=list(PLANS))
    ap.add_argument("--set", default=None, choices=["legacy", "rect150"])
    ap.add_argument("--prompts", default="P1")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--variant", default="bf16", choices=["bf16", "w4mlx"])
    a = ap.parse_args()
    plan = PLANS[a.plan] if a.plan else [(a.set, a.prompts.split(","), a.ids.split(",") if a.ids else None,
                                          a.repeat, a.tag)]
    if a.variant == "w4mlx":
        MODEL_KEY = "molmo2_8b_bf16_w4mlx"
    vlm = HFMolmo2(a.variant)
    print(f"loaded {REPO}@{REVISION[:7]} in {vlm.load_seconds:.1f}s :: {vlm.meta}", flush=True)
    for set_name, pids, ids, rep, tag in plan:
        jobs = build_jobs(set_name, pids, ids, rep)
        run_jobs(vlm, jobs, out_path_for(set_name, tag))
    print("ALL DONE", flush=True)
