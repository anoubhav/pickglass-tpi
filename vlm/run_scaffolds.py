"""Run the scaffolded-visual-prompting conditions (scaffolds.py) through OpenRouter.

Same client (openrouter_client.call: key handling, retries, reasoning protocol 'min', provider
pinning for open-weight models), same input size (1536 x 1536 JPEG q92) and same parser
(parse_output.parse_counts) as the P1-P3 baselines, so the scaffold runs are paired with each
model's own P1 run on the same images.

Budget: Job 2 sub-cap US$8 of the round's US$20.  Every call is logged to
scaffold/logs/openrouter_costs_job2.csv (this job only; NOT the shared logs/openrouter_costs.csv,
so other jobs' stop rules are unaffected).  Before each call the job total plus a reservation for
the calls in flight is compared against --stop-at (default 7.5) and the run stops when reached.

    python run_scaffolds.py --set synth  --models gpt-6.1-sol,gemini-3.8-flash,qwen3.8-27b --tag pilot
    python run_scaffolds.py --set legacy --models ... --tag main [--repeat 1]
    python run_scaffolds.py --spent
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import openrouter_client as O            # noqa: E402
import scaffolds as SC                   # noqa: E402
from parse_output import parse_counts    # noqa: E402

RAW = os.path.join(SC.SC_DIR, "raw")
LOGS = os.path.join(SC.SC_DIR, "logs")
INPUTS = os.path.join(SC.SC_DIR, "inputs")
COST_CSV = os.path.join(LOGS, "openrouter_costs_job2.csv")
SUB_CAP = 8.0
MODELS = ["gpt-6.1-sol", "gemini-3.8-flash", "qwen3.8-27b"]
MAX_TOKENS = {"gpt-6.1-sol": 2500, "gemini-3.8-flash": 3000, "qwen3.8-27b": 2000}
RESERVE = {"gpt-6.1-sol": 0.04, "gemini-3.8-flash": 0.02, "qwen3.8-27b": 0.01}   # per in-flight call
LEGACY_NAMES = [f"a{i}" for i in range(1, 11)] + [f"r{i}" for i in range(1, 11)]
SYNTH_NAMES = ["syn1", "syn2", "syn3"]
FIELDS = ["timestamp", "model", "set", "tag", "image_id", "scaffold", "repeat", "prompt_tokens",
          "completion_tokens", "reasoning_tokens", "cost_usd", "cost_source", "seconds", "finish_reason",
          "provider", "error"]
_lock = threading.Lock()


def spent():
    if not os.path.exists(COST_CSV):
        return 0.0
    return sum(float(r["cost_usd"] or 0) for r in csv.DictReader(open(COST_CSV)))


def log_cost(row):
    os.makedirs(LOGS, exist_ok=True)
    new = not os.path.exists(COST_CSV)
    with open(COST_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def input_path(set_name, image_id, sid):
    p = os.path.join(INPUTS, set_name, f"{image_id}__{sid}.jpg")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        img = SC.render(SC.load_crop(set_name, image_id), sid, SC.SIDE)
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return p


def raw_path(short, set_name, tag):
    return os.path.join(RAW, f"or-{short}__{set_name}__{tag}.jsonl")


def done_keys(path):
    done, fails = set(), {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            k = (r["image_id"], r["scaffold"], r["repeat"])
            if r.get("error"):
                fails[k] = fails.get(k, 0) + 1
            else:
                done.add(k)
    return done, fails


def run(shorts, set_name, sids, tag, repeat=0, ids=None, workers=4, stop_at=7.5):
    assert stop_at <= SUB_CAP
    names = ids or {"legacy": LEGACY_NAMES, "synth": SYNTH_NAMES, "rect150": [str(i) for i in range(150)]}[set_name]
    jobs = [(n, s) for n in names for s in sids]
    for n, s in jobs:
        input_path(set_name, n, s)
    os.makedirs(RAW, exist_ok=True)
    state = dict(spent=spent(), inflight=0.0, stop=False, reason="")
    print(f"start: job-2 spend so far ${state['spent']:.3f} (stop at ${stop_at:.2f}, sub-cap ${SUB_CAP:.2f})",
          flush=True)

    def one(short, n, sid, fh):
        res = RESERVE[short]
        with _lock:
            if state["stop"]:
                return
            if state["spent"] + state["inflight"] + res >= stop_at:
                state["stop"], state["reason"] = True, f"budget stop at ${state['spent']:.2f}"
                return
            state["inflight"] += res
        p = SC.get_prompt(sid)
        mt = MAX_TOKENS[short]
        ip = input_path(set_name, n, sid)
        try:
            g, err = O.call(short, p["text"], ip, mt, "min"), None
        except O.AccessBlocked as e:
            with _lock:
                state["stop"], state["reason"] = True, f"access blocked: {e}"
                state["inflight"] -= res
            return
        except Exception as e:
            g, err = dict(text="", reasoning="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0,
                          reasoning_tokens=0, cost=None, finish_reason="error", provider=None,
                          gen_id=None), O.scrub(repr(e))[:400]
        cost, src = g["cost"], "usage"
        if cost is None and not err:
            pr = O.pricing().get(O.ROSTER[short], {}).get("pricing", {})
            cost = (g["prompt_tokens"] * float(pr.get("prompt", 0) or 0)
                    + g["generation_tokens"] * float(pr.get("completion", 0) or 0))
            src = "estimate"
        parsed = parse_counts(g["text"])
        cv_, ch_ = SC.cell_estimate(sid, g["text"])
        rec = dict(model=f"or:{short}", repo=O.ROSTER[short], revision="openrouter-" + time.strftime("%Y-%m-%d"),
                   set=set_name, tag=tag, scaffold=sid, prompt_id=sid, prompt_version=p["version"],
                   scaffold_set_version=p["set_version"], scaffolds_py_md5=SC.code_md5(),
                   reasoning_effort="min", reasoning_param=O.reasoning_param(short, "min"),
                   provider_pref=O.PROVIDER.get(short), repeat=repeat, image_id=n, view=sid,
                   max_tokens=mt, temperature=0, input_wh=[SC.SIDE, SC.SIDE],
                   image_path=os.path.relpath(ip, HERE), raw=g["text"], reasoning=g["reasoning"],
                   parsed=parsed, cell_estimate=dict(vertical=cv_, horizontal=ch_), error=err,
                   seconds=g["seconds"], prompt_tokens=g["prompt_tokens"],
                   generation_tokens=g["generation_tokens"], reasoning_tokens=g["reasoning_tokens"],
                   cost_usd=cost, cost_source=src, provider=g["provider"], gen_id=g["gen_id"],
                   finish_reason=g["finish_reason"], timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with _lock:
            state["inflight"] -= res
            state["spent"] += float(cost or 0)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log_cost(dict(timestamp=rec["timestamp"], model=O.ROSTER[short], set=set_name, tag=tag, image_id=n,
                          scaffold=sid, repeat=repeat, prompt_tokens=g["prompt_tokens"],
                          completion_tokens=g["generation_tokens"], reasoning_tokens=g["reasoning_tokens"],
                          cost_usd=f"{float(cost or 0):.6f}", cost_source=src, seconds=f"{g['seconds']:.2f}",
                          finish_reason=g["finish_reason"], provider=g["provider"], error=(err or "")[:200]))
            tail = (g["text"] or "").replace("\n", " ")[-110:]
            print(f"  {short} {n}/{sid}/r{repeat} {g['seconds']:.1f}s tok={g['prompt_tokens']}+"
                  f"{g['generation_tokens']}(r{g['reasoning_tokens']}) ${float(cost or 0):.4f} "
                  f"tot=${state['spent']:.3f} ok={parsed['ok']} {('ERR ' + err[:80]) if err else ''}:: {tail!r}",
                  flush=True)

    files, futs = {}, []
    with ThreadPoolExecutor(max_workers=workers * len(shorts)) as ex:
        for short in shorts:
            path = raw_path(short, set_name, tag)
            done, fails = done_keys(path)
            todo = [(n, s) for n, s in jobs if (n, s, repeat) not in done and fails.get((n, s, repeat), 0) < 3]
            print(f"[{short}/{set_name}/{tag}/r{repeat}] {len(todo)} calls ({len(done)} cached)", flush=True)
            files[short] = open(path, "a")
            futs += [ex.submit(one, short, n, s, files[short]) for n, s in todo]
        for f in as_completed(futs):
            f.result()
    for fh in files.values():
        fh.close()
    print(f"end: job-2 spend ${state['spent']:.3f}" + (f"  STOPPED: {state['reason']}" if state["stop"] else ""),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="synth", choices=["synth", "legacy", "rect150"])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--scaffolds", default=",".join(SC.SCAFFOLDS))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stop-at", type=float, default=7.5)
    ap.add_argument("--inputs-only", action="store_true")
    ap.add_argument("--spent", action="store_true")
    a = ap.parse_args()
    if a.spent:
        print(f"job-2 logged spend: ${spent():.4f}")
        sys.exit(0)
    shorts = a.models.split(",")
    for s in shorts:
        assert s in MODELS, s
    ids = a.ids.split(",") if a.ids else None
    if a.inputs_only:
        names = ids or {"legacy": LEGACY_NAMES, "synth": SYNTH_NAMES, "rect150": [str(i) for i in range(150)]}[a.set]
        for n in names:
            for s in a.scaffolds.split(","):
                print(input_path(a.set, n, s))
        sys.exit(0)
    run(shorts, a.set, a.scaffolds.split(","), a.tag, a.repeat, ids, a.workers, a.stop_at)
