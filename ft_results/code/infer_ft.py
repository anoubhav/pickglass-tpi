"""Greedy inference for Molmo2-4B (base, or + the synthetic-pointing LoRA adapter merged) on the
project's views, batched (all views are 644 x 644 and a batch shares one prompt, so every sequence
has the same length: no padding).  Raw records -> ft/raw/{tag}__{set}.jsonl (resumable).

Views / prompts (same construction as gpu_results/code/run_molmo2_hf.build_jobs):
  P1   crop                      (P1 text)
  P4   guide_h / guide_v         (P4h / P4v on the whole 1-inch view)
  P4q  qc_guide_h / qc_guide_v   (central quarter-inch patch; count x 4)
  P4s  qd{k}_guide_h / _v, k=0..3 (the 4 diagonal quarter patches; counts summed = whole inch)
Sets: synval (ft/data/val, first --n-syn crops), legacy (20 canonical crops = Mac vlm/crops/legacy),
      rect150 (150 annotation-tool crops).
max_new_tokens: P1 200; P4* --max-new (default 2600 ~ 236 points; the project's P4 v1.2 cap of 1536
allows ~135 points, too few for 200-TPI synthetic fabrics) -> prompt_version 'v1.2+cap2600'.
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
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_ft import FW, PR, model_path, parse_counts, parse_points  # noqa: E402
from views import guide_line_view, quarter_patch, resize_square  # noqa: E402

SIDE = 644
RAW = os.path.join(HERE, "..", "raw")


def load_crop(set_name, image_id):
    if set_name == "legacy":
        return cv2.imread(os.path.join(FW, "data", "legacy_crops", f"{image_id}.png"))
    if set_name == "rect150":
        return cv2.imread(os.path.join(FW, "data", "crops150", f"{image_id}.jpg"))
    return cv2.imread(os.path.join(HERE, "..", "data", "val", "img", f"s{int(image_id):07d}.jpg"))


def set_ids(set_name, n_syn):
    if set_name == "legacy":
        return [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
    if set_name == "rect150":
        return [str(i) for i in range(150)]
    labs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data", "val", "labels.jsonl"))]
    return [str(s) for s in sorted(r["seed"] for r in labs)[:n_syn]]


def build_views(crop, methods):
    full = resize_square(crop, SIDE)
    out = []
    for m in methods:
        if m == "P1":
            out.append(("crop", "P1", full))
        elif m == "P4":
            out += [("guide_h", "P4h", guide_line_view(full, "h")), ("guide_v", "P4v", guide_line_view(full, "v"))]
        elif m == "P4q":
            q = resize_square(quarter_patch(crop, "c"), SIDE)
            out += [("qc_guide_h", "P4h", guide_line_view(q, "h")), ("qc_guide_v", "P4v", guide_line_view(q, "v"))]
        elif m == "P4s":
            for k in range(4):
                q = resize_square(quarter_patch(crop, f"d{k}"), SIDE)
                out += [(f"qd{k}_guide_h", "P4h", guide_line_view(q, "h")),
                        (f"qd{k}_guide_v", "P4v", guide_line_view(q, "v"))]
        else:
            raise ValueError(m)
    return out


class Runner:
    def __init__(self, adapter):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        t = time.time()
        p = model_path()
        self.proc = AutoProcessor.from_pretrained(p, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(p, trust_remote_code=True, dtype=torch.bfloat16,
                                                            device_map={"": 0})
        if adapter and adapter != "none":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.eos = self.proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.load_s = time.time() - t

    def generate(self, text, images, max_new):
        convs = [[{"role": "user", "content": [dict(type="text", text=text),
                                               dict(type="image", image=Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))]}]
                 for im in images]
        b = self.proc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, padding=True)
        assert bool(b["attention_mask"].all()), "unequal prompt lengths in a batch"
        b = {k: v.to(0) for k, v in b.items()}
        n_in = b["input_ids"].shape[1]
        torch.cuda.synchronize()
        t = time.time()
        with torch.inference_mode():
            out = self.model.generate(**b, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t
        res = []
        for i in range(len(images)):
            g = out[i, n_in:].tolist()
            n_gen = g.index(self.eos) if self.eos in g else len(g)
            txt = self.proc.tokenizer.decode(out[i, n_in:n_in + n_gen], skip_special_tokens=True)
            res.append(dict(text=txt, generation_tokens=n_gen, prompt_tokens=n_in,
                            finish_reason="stop" if n_gen < len(g) or self.eos in g else "length"))
        return res, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="none")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="legacy")
    ap.add_argument("--methods", default="P1,P4,P4q,P4s")
    ap.add_argument("--n-syn", type=int, default=150)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=2600)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job group slice i (parallel GPUs)")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    os.makedirs(RAW, exist_ok=True)
    R = Runner(a.adapter)
    print(f"loaded in {R.load_s:.0f}s adapter={a.adapter}", flush=True)
    meta = dict(model=a.tag, base=f"allenai/Molmo2-4B@{os.path.basename(model_path())[:7]}", adapter=a.adapter,
                gpu=torch.cuda.get_device_name(0), host=platform.node(), torch=torch.__version__)
    for set_name in a.sets.split(","):
        path = os.path.join(RAW, f"{a.tag}__{set_name}" + (f"__shard{si}of{sn}" if sn > 1 else "") + ".jsonl")
        done = set()
        if os.path.exists(path):
            for l in open(path):
                r = json.loads(l)
                done.add((r["image_id"], r["view"]))
        ids = a.ids.split(",") if a.ids else set_ids(set_name, a.n_syn)
        ids = ids[si::sn]
        jobs = {}
        for iid in ids:
            crop = load_crop(set_name, iid)
            assert crop is not None, (set_name, iid)
            for view, pid, img in build_views(crop, a.methods.split(",")):
                if (iid, view) not in done:
                    jobs.setdefault(pid, []).append((iid, view, img))
        print(f"[{set_name}] {sum(len(v) for v in jobs.values())} jobs ({len(done)} cached) -> {path}", flush=True)
        with open(path, "a") as f:
            for pid, js in jobs.items():
                p = PR.get(pid)
                max_new = p["max_tokens"] if pid == "P1" else a.max_new
                for s in range(0, len(js), a.batch):
                    chunk = js[s:s + a.batch]
                    res, dt = R.generate(p["text"], [j[2] for j in chunk], max_new)
                    for (iid, view, _), g in zip(chunk, res):
                        parsed = parse_points(g["text"]) if pid.startswith("P4") else parse_counts(g["text"])
                        if "points" in parsed:
                            parsed["points"] = parsed["points"][:400]
                        rec = dict(set=set_name, image_id=iid, view=view, prompt_id=pid, prompt_version=p["version"]
                                   + ("" if pid == "P1" else f"+cap{max_new}"), max_tokens=max_new, raw=g["text"],
                                   parsed=parsed, generation_tokens=g["generation_tokens"],
                                   prompt_tokens=g["prompt_tokens"], finish_reason=g["finish_reason"],
                                   batch_seconds=dt, batch_size=len(chunk), seconds=dt / len(chunk),
                                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), **meta)
                        f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"  {set_name} {pid} {s + len(chunk)}/{len(js)} batch {dt:.1f}s "
                          f"maxgen {max(g['generation_tokens'] for g in res)}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
