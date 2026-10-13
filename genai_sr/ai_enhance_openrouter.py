"""ai_enhance_openrouter.py -- commercial "AI enhance/upscale" (Nano Banana Pro = google/gemini-3-pro-image) on the
R-LEG LR inputs (README section 4/8): 13 images x {D8, D4, D4J} = 39 calls max, own US$ ledger, hard stops.

    <venv_vlm>/bin/python ai_enhance_openrouter.py --pilot          # 1 image (leg_a1, D8), prints cost + output size
    <venv_vlm>/bin/python ai_enhance_openrouter.py                  # the rest (D8 first, then D4, then D4J)

The API key is read by vlm/openrouter_client.load_key() from $OPENROUTER_API_KEY only and never printed or logged.
Raw outputs: outputs/_nbpro_raw/<cond>/<key>.png (as returned); final: outputs/nbpro/<cond>/<key>.png (S x S).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import requests

import gsr_common as G

sys.path.insert(0, os.path.join(G.WVAQ, "vlm"))
import openrouter_client as OR  # noqa: E402  (load_key / scrub only; its global ledger is not touched)

MODEL = "google/gemini-3-pro-image"
LEDGER = os.path.join(G.GSR, "logs", "openrouter_costs.csv")
RAW_JSONL = os.path.join(G.GSR, "logs", "nbpro_calls.jsonl")
JOB_CAP = 5.9          # whole job (image edit + GPT)
EDIT_CAP = 5.3         # projected image-edit total must stay below this
FIELDS = ["timestamp", "model", "purpose", "key", "cond", "prompt_tokens", "completion_tokens", "reasoning_tokens",
          "cost_usd", "seconds", "status", "error"]


def spent():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(pd.read_csv(LEDGER).cost_usd.fillna(0).sum())


def spent_edit():
    if not os.path.exists(LEDGER):
        return 0.0, 0
    d = pd.read_csv(LEDGER)
    d = d[d.purpose == "nbpro"]
    return float(d.cost_usd.fillna(0).sum()), int((d.status == "ok").sum())


def log(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def call(png_path, image_size="2K", timeout=300):
    key = OR.load_key()
    assert key, "no OpenRouter key"
    b64 = base64.b64encode(open(png_path, "rb").read()).decode()
    body = {"model": MODEL, "modalities": ["image", "text"], "seed": G.SEED, "usage": {"include": True},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": G.PROMPT_ENHANCE},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    if image_size:
        body["image_config"] = {"aspect_ratio": "1:1", "image_size": image_size}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "fabric-thread-count-genai-sr"}
    t = time.time()
    r = requests.post(OR.API_URL, headers=headers, json=body, timeout=timeout)
    dt = time.time() - t
    if r.status_code != 200:
        return None, dict(error=f"HTTP {r.status_code}: {OR.scrub(r.text)[:300]}", seconds=dt)
    j = r.json()
    ch = (j.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    u = j.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    imgs = msg.get("images") or []
    meta = dict(seconds=dt, cost=u.get("cost"), prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"), reasoning_tokens=det.get("reasoning_tokens"),
                image_tokens=det.get("image_tokens"), finish_reason=ch.get("finish_reason"),
                provider=j.get("provider"), gen_id=j.get("id"), text=(msg.get("content") or "")[:500],
                n_images=len(imgs), model=j.get("model"))
    if not imgs:
        meta["error"] = "no image returned"
        return None, meta
    url = imgs[0].get("image_url", {}).get("url", "")
    data = base64.b64decode(url.split(",", 1)[1])
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    meta["mime"] = url.split(";", 1)[0][5:]
    return (arr, data), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--image_size", default="2K")
    a = ap.parse_args()
    man = pd.read_csv(G.MANIFEST)
    man = man[man.set == "R-LEG"].copy()
    man["co"] = man.cond.map({"D8": 0, "D4": 1, "D4J": 2})
    man = man.sort_values(["co", "key"])
    if a.pilot:
        man = man[(man.key == "leg_a1") & (man.cond == "D8")]
    for r in man.itertuples():
        op = G.out_path("nbpro", r.cond, r.key)
        if os.path.exists(op):
            continue
        cost_e, n_ok = spent_edit()
        per = cost_e / n_ok if n_ok else 0.14
        if spent() + per > JOB_CAP or cost_e + per > EDIT_CAP:
            print(f"budget stop: job ${spent():.3f}, edit ${cost_e:.3f} ({n_ok} images)")
            break
        lp = os.path.join(G.GSR, r.lr_path)
        assert G.md5(lp) == r.lr_md5
        out, meta = call(lp, a.image_size)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec = dict(timestamp=ts, key=r.key, cond=r.cond, model_requested=MODEL, prompt=G.PROMPT_ENHANCE,
                   image_size=a.image_size, seed=G.SEED, lr_md5=r.lr_md5, **meta)
        status = "ok" if out is not None else "fail"
        if out is not None:
            arr, data = out
            ext = ".png" if rec.get("mime", "").endswith("png") else ".jpg"
            raw = os.path.join(G.OUT_DIR, "_nbpro_raw", r.cond, r.key + ext)
            os.makedirs(os.path.dirname(raw), exist_ok=True)
            open(raw, "wb").write(data)
            h, w = arr.shape[:2]
            rec.update(raw_path=os.path.relpath(raw, G.GSR), raw_w=w, raw_h=h, raw_md5=G.md5(raw))
            fin = arr if (h, w) == (r.S, r.S) else cv2.resize(arr, (r.S, r.S), interpolation=cv2.INTER_CUBIC if h < r.S else cv2.INTER_AREA)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            cv2.imwrite(op, fin, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        with open(RAW_JSONL, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log(dict(timestamp=ts, model=meta.get("model") or MODEL, purpose="nbpro", key=r.key, cond=r.cond,
                 prompt_tokens=meta.get("prompt_tokens"), completion_tokens=meta.get("completion_tokens"),
                 reasoning_tokens=meta.get("reasoning_tokens"), cost_usd=float(meta.get("cost") or 0),
                 seconds=round(meta.get("seconds", 0), 1), status=status, error=(meta.get("error") or "")[:200]))
        print(f"{r.cond} {r.key} {status} {rec.get('raw_w')}x{rec.get('raw_h')} ${float(meta.get('cost') or 0):.4f} "
              f"tok={meta.get('prompt_tokens')}+{meta.get('completion_tokens')} (img {meta.get('image_tokens')}, "
              f"r {meta.get('reasoning_tokens')}) {meta.get('seconds', 0):.0f}s job=${spent():.3f} "
              f"{(meta.get('error') or '')[:150]}", flush=True)


if __name__ == "__main__":
    main()
