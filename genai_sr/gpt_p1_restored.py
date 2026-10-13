"""gpt_p1_restored.py -- GPT-6.1-sol, prompt P1, on the R-LEG D8 crops restored by one generative model and by the
bicubic control (README section 5).  Same call path as the paper's VLM rows: vlm/openrouter_client.call() with the
'min' reasoning protocol, temperature 0, max_tokens 1500; input = data_vlm.resize_square(crop, 1536) as JPEG q92.

    <venv_vlm>/bin/python gpt_p1_restored.py --restorers seedvr2,bicubic [--cond D8]

Costs go to this job's ledger (logs/openrouter_costs.csv; hard stop US$5.9 for the whole job); raw records to
raw/gpt-6.1-sol__<restorer>_<cond>.jsonl.  The key is never printed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import pandas as pd

import gsr_common as G

sys.path.insert(0, os.path.join(G.WVAQ, "vlm"))
import data_vlm as D  # noqa: E402
import openrouter_client as OR  # noqa: E402
import prompts as PR  # noqa: E402
from parse_output import parse_counts  # noqa: E402

SHORT = "gpt-6.1-sol"
LEDGER = os.path.join(G.GSR, "logs", "openrouter_costs.csv")
JOB_CAP = 5.9
FIELDS = ["timestamp", "model", "purpose", "key", "cond", "prompt_tokens", "completion_tokens", "reasoning_tokens",
          "cost_usd", "seconds", "status", "error"]


def spent():
    return float(pd.read_csv(LEDGER).cost_usd.fillna(0).sum()) if os.path.exists(LEDGER) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorers", default="seedvr2,bicubic")
    ap.add_argument("--cond", default="D8")
    a = ap.parse_args()
    assert OR.load_key(), "no key"
    p1 = PR.get("P1")
    man = pd.read_csv(G.MANIFEST)
    man = man[(man.set == "R-LEG") & (man.cond == a.cond)]
    for rest in a.restorers.split(","):
        rawp = os.path.join(G.GSR, "raw", f"{SHORT}__{rest}_{a.cond}.jsonl")
        os.makedirs(os.path.dirname(rawp), exist_ok=True)
        done = set()
        if os.path.exists(rawp):
            for line in open(rawp):
                r = json.loads(line)
                if not r.get("error"):
                    done.add(r["key"])
        for r in man.itertuples():
            if r.key in done:
                continue
            if spent() + 0.02 > JOB_CAP:
                print("budget stop", spent())
                return
            src = G.out_path(rest, a.cond, r.key)
            if not os.path.exists(src):          # e.g. Nano Banana Pro content-filter refusals (no output)
                print(f"{rest} {r.key}: no restored image, skipped", flush=True)
                continue
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            jp = os.path.join(G.GSR, "inputs", "vlm1536", rest, a.cond, r.key + ".jpg")
            os.makedirs(os.path.dirname(jp), exist_ok=True)
            cv2.imwrite(jp, D.resize_square(img, OR.SIDE), [cv2.IMWRITE_JPEG_QUALITY, 92])
            err = None
            try:
                g = OR.call(SHORT, p1["text"], jp, OR.max_tokens_for(SHORT, "P1"), "min")
            except Exception as e:  # noqa: BLE001
                g, err = dict(text="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0, reasoning_tokens=0,
                              cost=None, finish_reason="error", provider=None, gen_id=None), OR.scrub(repr(e))[:300]
            parsed = parse_counts(g["text"])
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            rec = dict(timestamp=ts, model=OR.ROSTER[SHORT], short=SHORT, restorer=rest, cond=a.cond, key=r.key,
                       prompt_id="P1", prompt_version=p1["version"], reasoning=OR.REASONING_MIN[SHORT], temperature=0,
                       input=os.path.relpath(jp, G.GSR), src_md5=G.md5(src), raw=g["text"], parsed=parsed, error=err,
                       seconds=g["seconds"], prompt_tokens=g["prompt_tokens"], generation_tokens=g["generation_tokens"],
                       reasoning_tokens=g["reasoning_tokens"], cost_usd=g["cost"], provider=g["provider"],
                       gen_id=g["gen_id"], finish_reason=g["finish_reason"])
            with open(rawp, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            new = not os.path.exists(LEDGER)
            with open(LEDGER, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(dict(timestamp=ts, model=OR.ROSTER[SHORT], purpose=f"gpt_p1_{rest}", key=r.key, cond=a.cond,
                                prompt_tokens=g["prompt_tokens"], completion_tokens=g["generation_tokens"],
                                reasoning_tokens=g["reasoning_tokens"], cost_usd=float(g["cost"] or 0),
                                seconds=round(g["seconds"], 1) if g["seconds"] == g["seconds"] else None,
                                status="ok" if not err else "fail", error=(err or "")[:200]))
            print(f"{rest} {r.key} {parsed} ${float(g['cost'] or 0):.4f} job=${spent():.3f} {err or ''}", flush=True)


if __name__ == "__main__":
    main()
