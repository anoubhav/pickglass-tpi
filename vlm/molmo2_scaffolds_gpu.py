"""Molmo2-8B bf16 (HF transformers, RTX 4090 on gpu-server) on the scaffolded inputs (Job 2).

Runs on the GPU server from ~/pickglass/code/ next to run_molmo2_hf.py, whose
HFMolmo2 class (same repo revision, bf16, greedy decoding, same chat template) it reuses, so the
only differences to the existing Molmo2 bf16 P1-P4 rows are the input image and the prompt.

Inputs are rendered ON THE MAC by scaffolds.py (1536-px composite, INTER_AREA down to 644 px =
Molmo2's native maximum for a square image, PNG) and copied to scaffold/inputs/{set}/; the prompt
texts are copied as scaffold/prompts.json (exported from scaffolds.py, md5 recorded).  Records go
to raw/molmo2_8b_bf16__scaffold_{set}.jsonl; the Mac re-parses them with parse_output.parse_counts.

    CUDA_VISIBLE_DEVICES=<idle> venv/bin/python code/molmo2_scaffolds_gpu.py --set synth
    CUDA_VISIBLE_DEVICES=<idle> venv/bin/python code/molmo2_scaffolds_gpu.py --set legacy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_molmo2_hf as RM                       # noqa: E402  (HFMolmo2, REPO, REVISION)
from vlm_common import parse_counts              # noqa: E402

ROOT = os.path.dirname(HERE)
SC = os.path.join(ROOT, "scaffold")
MAX_NEW = 1200          # = P2's budget (the longest-output prompt of the existing Molmo2 runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["synth", "legacy", "rect150"])
    ap.add_argument("--scaffolds", default="P1,S1,S1t,S2,S3,S4")
    a = ap.parse_args()
    prompts = json.load(open(os.path.join(SC, "prompts.json")))
    out = os.path.join(ROOT, "raw", f"molmo2_8b_bf16__scaffold_{a.set}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["image_id"], r["scaffold"]))
    sids = a.scaffolds.split(",")
    jobs = []
    for p in sorted(glob.glob(os.path.join(SC, "inputs", a.set, "*.png"))):
        img, sid = os.path.basename(p)[:-4].split("__")
        if sid in sids and (img, sid) not in done:
            jobs.append((img, sid, p))
    print(f"{len(jobs)} jobs ({len(done)} cached) -> {out}", flush=True)
    if not jobs:
        return
    vlm = RM.HFMolmo2("bf16")
    print(f"loaded {RM.REPO}@{RM.REVISION[:7]} in {vlm.load_seconds:.1f}s :: {vlm.meta}", flush=True)
    with open(out, "a") as f:
        for k, (img, sid, p) in enumerate(jobs):
            pr = prompts["prompts"][sid]
            try:
                g, err = vlm(pr["text"], p, MAX_NEW), None
            except Exception as e:
                g = dict(text="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0, peak_memory_gb=0.0,
                         finish_reason="error")
                err = repr(e)
                vlm.torch.cuda.empty_cache()
            rec = dict(model="molmo2_8b_bf16", repo=RM.REPO, revision=RM.REVISION, set=a.set, image_id=img,
                       scaffold=sid, view=sid, prompt_id=sid, prompt_version=pr["version"],
                       scaffold_set_version=prompts["set_version"], scaffolds_py_md5=prompts["scaffolds_py_md5"],
                       max_tokens=MAX_NEW, image_path=os.path.relpath(p, ROOT), raw=g["text"],
                       parsed=parse_counts(g["text"]), error=err, seconds=g["seconds"],
                       prompt_tokens=g["prompt_tokens"], generation_tokens=g["generation_tokens"],
                       peak_memory_gb=g["peak_memory_gb"], finish_reason=g["finish_reason"], host=platform.node(),
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), repeat=0, **vlm.meta)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  [{k + 1}/{len(jobs)}] {img}/{sid} {g['seconds']:.1f}s in={g['prompt_tokens']} "
                  f"gen={g['generation_tokens']} {g['finish_reason']} :: {g['text'][-140:]!r}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
