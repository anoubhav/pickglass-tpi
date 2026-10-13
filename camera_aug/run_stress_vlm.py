"""run_stress_vlm.py -- API VLMs (prompt P1) on the two harshest PickGlass-Stress profiles, legacy-13.

Inputs are the md5-checked stress renders (camera_aug/stress/legacy13/manifest.csv, rep 0); each goes
through exactly the clean-run input path of vlm/openrouter_client.py:
    P1 : data_vlm.resize_square(crop, 1536) -> JPEG q92   (view 'crop@<condition>')
API calls, prompt, reasoning settings, max_tokens, parser, raw-record format and cost log are those
of openrouter_client.py (its run() is reused with the job builder swapped, as vlm/run_degrade_vlm.py
does).  The two conditions are chosen BEFORE any VLM call from the classical + CNN results
(score_stress.py writes results/stress_harshest.json).  Clean results are the existing main-run
records (repeat 0; repeats 1-2 = run-to-run noise reference).

    <venv_vlm>/bin/python run_stress_vlm.py --models gpt-6.1-sol,gemini-3.8-flash --budget 1.5 [--dry-run]
    <venv_vlm>/bin/python run_stress_vlm.py --collect     # -> results/stress_legacy13_vlm.csv
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VLM = os.path.join(ROOT, "vlm")
sys.path.insert(0, VLM)
import data_vlm as D            # noqa: E402
import openrouter_client as OC  # noqa: E402

SET = "stress_cam"
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
MAN = os.path.join(HERE, "stress", "legacy13", "manifest.csv")
HARSH = os.path.join(HERE, "results", "stress_harshest.json")
OUT_CSV = os.path.join(HERE, "results", "stress_legacy13_vlm.csv")
JOB_BUDGET_CAP = 5.0          # this job's API budget (user: <= US$5)
_CONDS: list[str] = []
BENCH_VERSION = "v2"          # view suffix: v1 (capped simulator) records stay in the raw log, marked superseded


def load_render(cond, name, rep=0):
    man = pd.read_csv(MAN, dtype={"source_id": str})
    r = man[(man.condition == cond) & (man.source_id == name) & (man.rep == rep)].iloc[0]
    b = open(os.path.join(HERE, r.path), "rb").read()
    assert hashlib.md5(b).hexdigest() == r.md5, f"stress render changed: {r.path}"
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def build_jobs_stress(set_name, prompt_ids, limit=None, ids=None, p3_views=("c",)):
    assert set_name == SET and prompt_ids == ["P1"], (set_name, prompt_ids)
    names = ids or NAMES13
    names = names[:limit] if limit else names
    jobs = []
    for cond in _CONDS:
        for name in names:
            view = f"crop@{cond}@{BENCH_VERSION}"
            p = os.path.join(OC.INPUT_DIR, SET, f"{name}__{view}.jpg")
            if not os.path.exists(p):
                OC._write_jpg(p, D.resize_square(load_render(cond, name), OC.SIDE))
            jobs.append(dict(image_id=name, view=view, prompt_id="P1", image_path=p))
    return jobs


def stress_spend():
    """US$ logged for this set (the job's own budget)."""
    if not os.path.exists(OC.COST_CSV):
        return 0.0
    c = pd.read_csv(OC.COST_CSV)
    return float(c[c["set"] == SET].cost_usd.astype(float).sum())


def collect():
    rows = []
    files = sorted(glob.glob(os.path.join(OC.RAW_DIR, f"or-*__{SET}__main.jsonl")))
    for f in files:
        best = {}
        for line in open(f):
            r = json.loads(line)
            k = (r["image_id"], r["view"], r["prompt_id"], r.get("repeat", 0))
            if k not in best or (best[k].get("error") and not r.get("error")):
                best[k] = r
        for (img, view, pid, rep), r in best.items():
            p = r["parsed"]
            ok = bool(p.get("ok")) and not r.get("error")
            parts = view.split("@")
            rows.append(dict(source_id=img, condition=parts[1], bench_version=parts[2] if len(parts) > 2 else "v1",
                             rep=0, vlm_repeat=rep,
                             estimator=r["model"].split(":", 1)[1], h=p.get("vertical") if ok else np.nan,
                             v=p.get("horizontal") if ok else np.nan, ok=ok, cost_usd=r.get("cost_usd"),
                             seconds=r.get("seconds"), finish_reason=r.get("finish_reason"),
                             raw_tail=(r.get("raw") or "")[-160:].replace("\n", " ")))
    # clean = the main legacy run (P1, repeat 0) and its repeats 1-2 (noise reference)
    v = pd.read_csv(os.path.join(VLM, "results", "legacy_predictions.csv"), dtype={"image": str})
    v = v[(v.set == "legacy") & (v.method == "P1") & v.image.isin(NAMES13)
          & v.model.isin(["gpt-6.1-sol", "gemini-3.8-flash"])]
    for r in v.itertuples():
        ok = str(r.ok) == "True"
        rows.append(dict(source_id=r.image, condition="clean" if r.repeat == 0 else f"clean_repeat{r.repeat}",
                         bench_version="clean", rep=0,
                         vlm_repeat=int(r.repeat), estimator=r.model, h=r.vertical if ok else np.nan,
                         v=r.horizontal if ok else np.nan, ok=ok, cost_usd=r.usd, seconds=r.seconds))
    d = pd.DataFrame(rows)
    d.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}: {len(d)} rows; stress-set spend ${stress_spend():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-6.1-sol,gemini-3.8-flash")
    ap.add_argument("--conds", default="", help="default: the two of results/stress_harshest.json")
    ap.add_argument("--budget", type=float, default=0.0, help="max US$ this invocation may add")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
        sys.exit(0)
    _CONDS[:] = a.conds.split(",") if a.conds else json.load(open(HARSH))["harshest"]
    shorts = a.models.split(",")
    for s in shorts:
        assert s in OC.ROSTER, s
    OC.build_jobs = build_jobs_stress
    spent_all, spent_job = OC.spent_usd(), stress_spend()
    room = min(a.budget, JOB_BUDGET_CAP - spent_job)
    stop_at = spent_all + room
    print(f"conditions {_CONDS}; logged spend (all VLM jobs) ${spent_all:.3f}, this set ${spent_job:.3f}; "
          f"this run may add ${room:.2f} -> stop at ${stop_at:.3f}", flush=True)
    if not a.dry_run:
        assert room > 0, "no budget left"
    OC.run(shorts, SET, ["P1"], "main", "min", repeat=0, workers=a.workers, stop_at=stop_at, dry_run=a.dry_run)
