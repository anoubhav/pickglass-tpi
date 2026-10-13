"""run_degrade_vlm.py -- Job C (WVAQ capture-quality study): API VLMs on degraded legacy-13 crops.

Inputs are the byte-identical degraded crops written by methods/degrade_vlmcrops.py --make
(degrade.py operators applied to the canonical crops vlm/crops/legacy/{name}.png; md5 checked
against that manifest here).  Each degraded crop then goes through exactly the clean-run input
path of openrouter_client.py:
    P1 : D.resize_square(crop, 1536)                         -> JPEG q92  (view 'crop@<cond>')
    P3 : D.resize_square(D.quarter_patch(crop, 'c'), 1536)  -> JPEG q92  (view 'qc@<cond>'; x4)
The API calls, prompts, reasoning settings, max_tokens, parser, raw-record format, cost log and
budget stop are those of openrouter_client.py (its run() is reused; only the job builder is
swapped for the 'legacy_deg' set).  Clean results are NOT re-run: the clean inputs are identical
to inputs/openrouter/legacy/{name}__crop.jpg / __qc.jpg (checked with --verify-clean) and the
existing main-run records (repeat 0) are used.

    <venv_vlm>/bin/python run_degrade_vlm.py --verify-clean
    <venv_vlm>/bin/python run_degrade_vlm.py --models gpt-6.1-sol --prompts P1 --budget 1.6 [--dry-run]
    <venv_vlm>/bin/python run_degrade_vlm.py --collect          # -> results/legacy_deg_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_vlm as D            # noqa: E402
import openrouter_client as OC  # noqa: E402

SET = "legacy_deg"
DEG_DIR = os.environ.get(
    "DEG_LEGACY13_DIR",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "scratch")), "deg_legacy13"))
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
# fixed a priori (= methods/degrade_vlmcrops.VLM_CONDS): the moderate/severe levels of paper Table 6
VLM_CONDS = [("gblur", 3), ("gblur", 6), ("down", 4), ("down", 8), ("jpeg", 20),
             ("glare", 0.05), ("glare", 0.30), ("corners", 0.05)]
HARD_JOB_CAP = 30.0       # the API budget of the whole VLM job (openrouter_client.HARD_CAP_USD)
OUT_CSV = os.path.join(HERE, "results", "legacy_deg_predictions.csv")


def cond_tag(kind, level):
    return "clean" if kind == "clean" else f"{kind}{level:g}"


_MAN = None


def manifest():
    global _MAN
    if _MAN is None:
        with open(os.path.join(DEG_DIR, "manifest.csv")) as f:
            _MAN = {(r["image"], r["tag"]): r["md5"] for r in csv.DictReader(f)}
    return _MAN


def load_deg(name, tag):
    p = os.path.join(DEG_DIR, tag, f"{name}.png")
    b = open(p, "rb").read()
    assert hashlib.md5(b).hexdigest() == manifest()[(name, tag)], f"degraded crop changed: {p}"
    return cv2.imread(p, cv2.IMREAD_COLOR)


def make_view(img, pid):
    if pid == "P1":
        return D.resize_square(img, OC.SIDE)
    if pid == "P3":
        return D.resize_square(D.quarter_patch(img, "c"), OC.SIDE)
    raise ValueError(pid)


def view_name(pid, tag):
    return {"P1": "crop", "P3": "qc"}[pid] + "@" + tag


_CONDS = list(VLM_CONDS)


def build_jobs_deg(set_name, prompt_ids, limit=None, ids=None, p3_views=("c",)):
    assert set_name == SET, set_name
    names = ids or NAMES13
    names = names[:limit] if limit else names
    jobs = []
    for kind, level in _CONDS:
        tag = cond_tag(kind, level)
        for name in names:
            img = None
            for pid in prompt_ids:
                view = view_name(pid, tag)
                p = os.path.join(OC.INPUT_DIR, SET, f"{name}__{view}.jpg")
                if not os.path.exists(p):
                    img = load_deg(name, tag) if img is None else img
                    OC._write_jpg(p, make_view(img, pid))
                jobs.append(dict(image_id=name, view=view, prompt_id=pid, image_path=p))
    return jobs


def verify_clean():
    """Rebuild the clean inputs through this path and compare with the files the main run sent."""
    tmp = os.path.join(DEG_DIR, "_verify_clean")
    os.makedirs(tmp, exist_ok=True)
    same = 0
    for name in NAMES13:
        img = load_deg(name, "clean")
        for pid, old in (("P1", "crop"), ("P3", "qc")):
            p = os.path.join(tmp, f"{name}__{pid}.jpg")
            cv2.imwrite(p, make_view(img, pid), [cv2.IMWRITE_JPEG_QUALITY, 92])
            a = hashlib.md5(open(p, "rb").read()).hexdigest()
            b = hashlib.md5(open(os.path.join(OC.INPUT_DIR, "legacy", f"{name}__{old}.jpg"), "rb").read()).hexdigest()
            same += a == b
            if a != b:
                print(f"DIFFERENT: {name} {pid}")
    print(f"clean inputs byte-identical to the main run: {same}/{2 * len(NAMES13)}")


def collect():
    """Long table of parsed answers: degraded runs + the clean main run (repeat 0, and the P1
    repeats 1-2 as a run-to-run noise reference)."""
    rows = []
    gt = {}
    for f in sorted(glob.glob(os.path.join(OC.RAW_DIR, "or-*__legacy__main.jsonl"))
                    + sorted(glob.glob(os.path.join(OC.RAW_DIR, f"or-*__{SET}__main.jsonl")))):
        best = {}
        for line in open(f):
            r = json.loads(line)
            k = (r["image_id"], r["view"], r["prompt_id"], r.get("repeat", 0))
            if k not in best or (best[k].get("error") and not r.get("error")):
                best[k] = r
        for (img, view, pid, rep), r in best.items():
            if img not in NAMES13:
                continue
            if r["set"] == "legacy":
                if (pid, view) not in (("P1", "crop"), ("P3", "qc")):
                    continue
                kind, level, tag = "clean", 0, "clean"
            else:
                tag = view.split("@", 1)[1]
                kind, level = next((k, lv) for k, lv in VLM_CONDS if cond_tag(k, lv) == tag)
            p = r["parsed"]
            k4 = 4 if pid == "P3" else 1
            v, h = p.get("vertical"), p.get("horizontal")
            ok = bool(p.get("ok")) and not r.get("error")
            raw = r.get("raw") or ""
            # explicit abstention: the model answered, but with null counts and/or a stated reason
            # ("too blurred", "almost entirely white") instead of numbers
            abstained = (not ok) and bool(raw) and bool(re.search(
                r'"(?:vertical_yarns|horizontal_yarns)"\s*:\s*null|"error"\s*:|cannot|unable|not possible', raw, re.I))
            rows.append(dict(model=r["model"].split(":", 1)[1], prompt=pid, repeat=rep, image=img, kind=kind,
                             level=level, tag=tag, vertical=None if v is None else k4 * v,
                             horizontal=None if h is None else k4 * h, weave=p.get("weave"),
                             ok=ok, abstained=abstained, parse_method=p.get("method"),
                             finish_reason=r.get("finish_reason"), error=r.get("error"),
                             cost_usd=r.get("cost_usd"), seconds=r.get("seconds"),
                             reasoning_tokens=r.get("reasoning_tokens"), image_path=r.get("image_path"),
                             raw_tail=(r.get("raw") or "")[-160:].replace("\n", " ")))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="gpt-6.1-sol")
    ap.add_argument("--prompts", default="P1")
    ap.add_argument("--conds", default="", help="comma list of condition tags (default: all VLM_CONDS)")
    ap.add_argument("--budget", type=float, default=0.0, help="max USD this invocation may add to the cost log")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-clean", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.verify_clean:
        verify_clean()
        sys.exit(0)
    if a.collect:
        collect()
        sys.exit(0)
    if a.conds:
        want = a.conds.split(",")
        _CONDS[:] = [c for c in VLM_CONDS if cond_tag(*c) in want]
    shorts = a.models.split(",")
    for s in shorts:
        assert s in OC.ROSTER, s
    OC.build_jobs = build_jobs_deg            # swap in the degraded-input job builder
    spent = OC.spent_usd()
    stop_at = min(spent + a.budget, HARD_JOB_CAP - 0.25)
    print(f"logged spend so far ${spent:.3f}; this run may add ${a.budget:.2f} -> stop at ${stop_at:.3f} "
          f"(job cap ${HARD_JOB_CAP:.2f})", flush=True)
    if not a.dry_run:
        assert a.budget > 0 and stop_at > spent, "no budget left for this run"
    OC.run(shorts, SET, a.prompts.split(","), "main", "min", repeat=0, workers=a.workers, stop_at=stop_at,
           dry_run=a.dry_run)
