"""OpenRouter client for the API VLM baselines (fixed roster, see MODEL_ROSTER.md).

Same prompts (prompts.py), parser (parse_output.py) and record format as the local
MLX runs, so evaluate_vlm.py reads both.

API key
-------
Read ONLY from the environment variable OPENROUTER_API_KEY (export it in your shell; never
commit it).  The key is never printed, logged or written anywhere; error strings are scrubbed
before logging.

Cost control
------------
Every request asks for usage accounting (``usage: {include: true}``); each call is
appended to logs/openrouter_costs.csv.  Before every call the running total of
that CSV is compared against --stop-at (default US$28 of a hard US$30 cap) and the
run stops when it is reached.

Sets / views
------------
legacy     : 20 legacy window crops; P1, P2 on the whole crop ('crop'); P3 on the central
             quarter-inch patch ('qc', count x 4 = per inch).  The 4 diagonal patches
             ('qd0'..'qd3', counts summed = full inch, no extrapolation) were run for
             GPT-6.1-sol and Gemini 3.8 Flash only (budget).
legacy_x4  : resolution ablation, P1 on the legacy crop downscaled x4 (INTER_AREA)
             and resized back to the same input side (so token count is unchanged,
             only the information content drops).  legacy_x8 / legacy_x16: same with
             x8 / x16 (run for the two most stable models only).
rect150    : the 150 rectified 1-inch crops of the annotation tool
             (annotation_tool/data/crops/{i}.jpg, common/rectify.py contract).
Repeats (stability) are the same jobs with --repeat r (r=0 is the main run).

Usage
-----
    python openrouter_client.py --check                      # verify roster ids + key status
    python openrouter_client.py --set legacy --prompts P1,P2,P3
    python openrouter_client.py --set rect150 --prompts P1 --models gemini-3.8-flash
    python openrouter_client.py --set legacy --prompts P1 --repeat 1 --models ...
    python openrouter_client.py --dry-run --set legacy
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data_vlm as D                     # noqa: E402
import prompts as PR                     # noqa: E402
from parse_output import parse_counts, parse_points   # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
KEY_URL = "https://openrouter.ai/api/v1/key"
RAW_DIR = os.path.join(HERE, "raw")
LOG_DIR = os.path.join(HERE, "logs")
COST_CSV = os.path.join(LOG_DIR, "openrouter_costs.csv")
INPUT_DIR = os.path.join(HERE, "inputs", "openrouter")
ANN_CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
SIDE = 1536            # every input is a SIDE x SIDE square JPEG (q=92)
HARD_CAP_USD = 30.0

# Fixed roster (MODEL_ROSTER.md).  short name -> OpenRouter id
ROSTER = {
    "gpt-6.1-sol":          "openai/gpt-6.1-sol",
    "claude-fable-5.1":     "anthropic/claude-fable-5.1",
    "gemini-3.1-pro":       "google/gemini-3.1-pro-preview",
    "gemini-3.8-flash":     "google/gemini-3.8-flash",
    "qwen3.8-max-prime":    "qwen/qwen3.8-max-prime",
    "qwen3.8-27b":          "qwen/qwen3.8-27b",
    "gemma-4-26b-a4b":      "google/gemma-4-26b-a4b-it",
}

# Reasoning protocol "min" = the lowest hidden-reasoning setting each model accepts.
# Pilot (a1, r2; logged under tags pilot-*): with provider defaults GPT-6.1-sol,
# Claude Fable 5.1 and both Qwen models spent the whole 1000-token budget thinking and
# returned no answer.  'enabled: false' is rejected ("Reasoning is mandatory") by
# GPT-6.1-sol, Claude Fable 5.1 and Qwen3.8-max-prime, so they get their lowest effort
# (GPT 'minimal' ~800 hidden tokens; Claude 'low' -> 0 thinking tokens in the pilot;
# Qwen-max ~950 hidden tokens whatever budget is requested).  Gemini 3.x cannot switch
# thinking off and ignores a thinking-token budget; its thinking grows with max_tokens.
REASONING_MIN = {
    "gpt-6.1-sol":       {"effort": "minimal"},
    "claude-fable-5.1":  {"effort": "low"},
    "gemini-3.1-pro":    {"effort": "low"},
    "gemini-3.8-flash":  {"effort": "minimal"},
    "qwen3.8-max-prime": {"effort": "low"},
    "qwen3.8-27b":       {"enabled": False},
    "gemma-4-26b-a4b":   {"enabled": False},
}
# max output tokens (visible answer + any hidden reasoning).  Visible output is bounded
# by the prompt (JSON for P1/P3; <=~1500 for P2); models whose hidden reasoning cannot be
# disabled get extra head-room so that the answer is not truncated.
MAX_TOKENS = {  # short: (P1/P3, P2)
    "gpt-6.1-sol":       (1500, 2500),
    "claude-fable-5.1":  (1000, 1500),
    "gemini-3.1-pro":    (2000, 3000),
    "gemini-3.8-flash":  (2000, 3000),
    "qwen3.8-max-prime": (1500, 2500),
    "qwen3.8-27b":       (1000, 1500),
    "gemma-4-26b-a4b":   (1000, 1500),
}


def max_tokens_for(short, pid):
    return MAX_TOKENS[short][1 if pid == "P2" else 0]


# open-weight models are served by many third-party hosts (fp4..bf16); pin full-precision
# endpoints so that results are reproducible (closed models have a single first-party host)
PROVIDER = {
    "qwen3.8-27b":     {"order": ["deepinfra/bf16", "cerebras/fp16"], "allow_fallbacks": False},
    "gemma-4-26b-a4b": {"order": ["coreweave/bf16", "nextbit/bf16", "io-net/bf16"], "allow_fallbacks": False},
}

_KEY: str | None = None


# ----------------------------------------------------------------------------
# key handling (never print / log the key)
# ----------------------------------------------------------------------------
def load_key() -> str | None:
    global _KEY
    if _KEY:
        return _KEY
    k = os.environ.get("OPENROUTER_API_KEY")   # environment only (no .env file is read)
    _KEY = k or None
    return _KEY


def scrub(s: str) -> str:
    k = _KEY or os.environ.get("OPENROUTER_API_KEY")
    s = str(s)
    if k:
        s = s.replace(k, "***")
    return re.sub(r"sk-or-[A-Za-z0-9_-]+", "***", s)


class AccessBlocked(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
def _write_jpg(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def legacy_crop(name):
    return cv2.imread(os.path.join(D.LEGACY_CROP_DIR, f"{name}.png"))


def downscale(img, k=4):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)


def build_jobs(set_name: str, prompt_ids: list[str], limit=None, ids=None, p3_views=("c",)):
    jobs = []

    def add(image_id, view, pid, make):
        p = os.path.join(INPUT_DIR, set_name, f"{image_id}__{view}.jpg")
        if not os.path.exists(p):
            _write_jpg(p, make())
        jobs.append(dict(image_id=str(image_id), view=view, prompt_id=pid, image_path=p))

    if set_name in ("legacy", "legacy_x4", "legacy_x8", "legacy_x16"):
        names = ids or D.LEGACY_NAMES
        names = names[:limit] if limit else names
        for name in names:
            for pid in prompt_ids:
                if set_name.startswith("legacy_x"):
                    assert pid == "P1"
                    k = int(set_name.split("_x")[1])
                    add(name, f"crop_x{k}", pid, lambda n=name, k=k: D.resize_square(downscale(legacy_crop(n), k), SIDE))
                elif pid in ("P1", "P2"):
                    add(name, "crop", pid, lambda n=name: D.resize_square(legacy_crop(n), SIDE))
                elif pid == "P3":
                    for w in p3_views:
                        add(name, f"q{w}", "P3",
                            lambda n=name, w=w: D.resize_square(D.quarter_patch(legacy_crop(n), w), SIDE))
                else:
                    raise ValueError(pid)
    elif set_name == "rect150":
        idx = [int(i) for i in ids] if ids else list(range(150))
        idx = idx[:limit] if limit else idx
        for i in idx:
            for pid in prompt_ids:
                assert pid == "P1"
                add(i, "crop", pid, lambda i=i: D.resize_square(cv2.imread(os.path.join(ANN_CROPS, f"{i}.jpg")), SIDE))
    else:
        raise ValueError(set_name)
    return jobs


# ----------------------------------------------------------------------------
# cost bookkeeping
# ----------------------------------------------------------------------------
COST_FIELDS = ["timestamp", "model", "set", "tag", "image_id", "view", "prompt_id", "repeat",
               "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd", "cost_source",
               "seconds", "finish_reason", "provider", "error"]
_lock = threading.Lock()
_pricing = None


def pricing():
    """Per-token prices of the roster from a snapshot of the models API (for estimates only)."""
    global _pricing
    if _pricing is None:
        p = os.path.join(LOG_DIR, "openrouter_roster_snapshot.json")
        _pricing = json.load(open(p)) if os.path.exists(p) else {}
    return _pricing


def spent_usd() -> float:
    if not os.path.exists(COST_CSV):
        return 0.0
    tot = 0.0
    with open(COST_CSV) as f:
        for r in csv.DictReader(f):
            try:
                tot += float(r["cost_usd"] or 0)
            except ValueError:
                pass
    return tot


def log_cost(row: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(COST_CSV)
    with open(COST_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in COST_FIELDS})


# ----------------------------------------------------------------------------
# one API call
# ----------------------------------------------------------------------------
def reasoning_param(short: str, effort: str):
    """effort: 'min' -> REASONING_MIN[short]; 'default' -> send nothing;
    'off' -> {'enabled': False}; '{...}' -> that JSON; else {'effort': effort}."""
    if effort == "min":
        return REASONING_MIN[short]
    if effort == "default":
        return None
    if effort == "off":
        return {"enabled": False}
    if effort.startswith("{"):
        return json.loads(effort)
    return {"effort": effort}


def call(short: str, text: str, image_path: str, max_tokens: int, effort: str, timeout=300):
    key = load_key()
    model_id = ROSTER[short]
    b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    body = {
        "model": model_id,
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    rp = reasoning_param(short, effort)
    if rp is not None:
        body["reasoning"] = rp
    if short in PROVIDER:
        body["provider"] = PROVIDER[short]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "fabric-thread-count-baseline"}
    last = None
    for attempt in range(5):
        t = time.time()
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            last = f"network: {scrub(repr(e))[:200]}"
            time.sleep(5 * 2 ** attempt)
            continue
        dt = time.time() - t
        if r.status_code in (401, 402, 403):
            raise AccessBlocked(f"HTTP {r.status_code}: {scrub(r.text)[:300]}")
        if r.status_code in (408, 429, 500, 502, 503, 504, 524, 529):
            last = f"HTTP {r.status_code}: {scrub(r.text)[:200]}"
            time.sleep(5 * 2 ** attempt)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {scrub(r.text)[:400]}")
        j = r.json()
        if "error" in j and not j.get("choices"):
            last = f"body error: {scrub(json.dumps(j['error']))[:300]}"
            time.sleep(5 * 2 ** attempt)
            continue
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = j.get("usage") or {}
        det = u.get("completion_tokens_details") or {}
        content = msg.get("content") or ""
        if isinstance(content, list):          # some providers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return dict(text=content, reasoning=(msg.get("reasoning") or "")[:6000], seconds=dt,
                    prompt_tokens=u.get("prompt_tokens", 0) or 0,
                    generation_tokens=u.get("completion_tokens", 0) or 0,
                    reasoning_tokens=det.get("reasoning_tokens", 0) or 0,
                    cost=u.get("cost"), finish_reason=ch.get("finish_reason") or ch.get("native_finish_reason"),
                    provider=j.get("provider"), gen_id=j.get("id"))
    raise RuntimeError(f"OpenRouter kept failing: {last}")


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
def raw_path(short, set_name, tag):
    return os.path.join(RAW_DIR, f"or-{short}__{set_name}__{tag}.jsonl")


def load_done(path):
    done, fails = set(), {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r["image_id"], r["view"], r["prompt_id"], r.get("repeat", 0))
            if r.get("error"):
                fails[k] = fails.get(k, 0) + 1
            else:
                done.add(k)
    return done, fails


def run(shorts, set_name, prompt_ids, tag, effort, repeat=0, limit=None, ids=None, workers=4,
        stop_at=28.0, dry_run=False, p3_views=("c",), max_tokens_override=None):
    jobs = build_jobs(set_name, prompt_ids, limit, ids, p3_views)
    if dry_run:
        print(f"{len(shorts)} models x {len(jobs)} jobs = {len(shorts) * len(jobs)} calls "
              f"(set={set_name}, tag={tag}, repeat={repeat})")
        return
    if not load_key():
        print("No OPENROUTER_API_KEY available -> skipping the API baselines.")
        return
    os.makedirs(RAW_DIR, exist_ok=True)
    state = dict(spent=spent_usd(), stop=False, reason="")
    print(f"start: spent so far ${state['spent']:.3f} (stop at ${stop_at:.2f}, hard cap ${HARD_CAP_USD:.2f})",
          flush=True)

    def one(short, j, fh):
        if state["stop"]:
            return
        with _lock:
            if state["spent"] >= stop_at:
                state["stop"], state["reason"] = True, f"budget stop at ${state['spent']:.2f}"
                return
        model_id = ROSTER[short]
        p = PR.get(j["prompt_id"])
        mt = max_tokens_override or max_tokens_for(short, j["prompt_id"])
        try:
            g, err = call(short, p["text"], j["image_path"], mt, effort), None
        except AccessBlocked as e:
            state["stop"], state["reason"] = True, f"access blocked: {e}"
            return
        except Exception as e:
            g, err = dict(text="", reasoning="", seconds=float("nan"), prompt_tokens=0, generation_tokens=0,
                          reasoning_tokens=0, cost=None, finish_reason="error", provider=None,
                          gen_id=None), scrub(repr(e))[:400]
        cost, src = g["cost"], "usage"
        if cost is None and not err:
            pr = pricing().get(model_id, {}).get("pricing", {})
            cost = (g["prompt_tokens"] * float(pr.get("prompt", 0) or 0)
                    + g["generation_tokens"] * float(pr.get("completion", 0) or 0))
            src = "estimate"
        parsed = parse_points(g["text"]) if j["prompt_id"].startswith("P4") else parse_counts(g["text"])
        rec = dict(model=f"or:{short}", repo=model_id, revision="openrouter-" + time.strftime("%Y-%m-%d"),
                   set=set_name, tag=tag, reasoning_effort=effort,
                   reasoning_param=reasoning_param(short, effort), provider_pref=PROVIDER.get(short),
                   repeat=repeat,
                   image_id=j["image_id"], view=j["view"], prompt_id=j["prompt_id"],
                   prompt_version=p["version"], prompt_set_version=p["set_version"], max_tokens=mt,
                   temperature=0, input_wh=[SIDE, SIDE], image_path=os.path.relpath(j["image_path"], HERE),
                   raw=g["text"], reasoning=g["reasoning"], parsed=parsed, error=err, seconds=g["seconds"],
                   prompt_tokens=g["prompt_tokens"], generation_tokens=g["generation_tokens"],
                   reasoning_tokens=g["reasoning_tokens"], cost_usd=cost, cost_source=src,
                   provider=g["provider"], gen_id=g["gen_id"], finish_reason=g["finish_reason"],
                   timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with _lock:
            state["spent"] += float(cost or 0)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            log_cost(dict(timestamp=rec["timestamp"], model=model_id, set=set_name, tag=tag,
                          image_id=j["image_id"], view=j["view"], prompt_id=j["prompt_id"], repeat=repeat,
                          prompt_tokens=g["prompt_tokens"], completion_tokens=g["generation_tokens"],
                          reasoning_tokens=g["reasoning_tokens"], cost_usd=f"{float(cost or 0):.6f}",
                          cost_source=src, seconds=f"{g['seconds']:.2f}", finish_reason=g["finish_reason"],
                          provider=g["provider"], error=(err or "")[:200]))
            short_txt = (g["text"] or "").replace("\n", " ")[-90:]
            print(f"  {short} {j['image_id']}/{j['view']}/{j['prompt_id']}/r{repeat} {g['seconds']:.1f}s "
                  f"tok={g['prompt_tokens']}+{g['generation_tokens']}(r{g['reasoning_tokens']}) "
                  f"${float(cost or 0):.4f} tot=${state['spent']:.2f} ok={parsed.get('ok', parsed.get('n_points'))} "
                  f"{'ERR ' + err[:80] if err else ''}:: {short_txt!r}", flush=True)

    files, futs = {}, []
    with ThreadPoolExecutor(max_workers=workers * len(shorts)) as ex:
        for short in shorts:
            path = raw_path(short, set_name, tag)
            done, fails = load_done(path)
            todo = [j for j in jobs if (j["image_id"], j["view"], j["prompt_id"], repeat) not in done
                    and fails.get((j["image_id"], j["view"], j["prompt_id"], repeat), 0) < 3]
            print(f"[{short}/{set_name}/{tag}/r{repeat}] {len(todo)} calls ({len(done)} cached)", flush=True)
            files[short] = open(path, "a")
            futs += [ex.submit(one, short, j, files[short]) for j in todo]
        for f in as_completed(futs):
            f.result()
    for fh in files.values():
        fh.close()
    print(f"end: spent ${state['spent']:.3f}" + (f"  STOPPED: {state['reason']}" if state["stop"] else ""),
          flush=True)
    return state


# ----------------------------------------------------------------------------
def check():
    """Verify the roster ids against the live models API and snapshot their pricing."""
    d = requests.get(MODELS_URL, timeout=60).json()["data"]
    ids = {m["id"]: m for m in d}
    snap, ok = {}, True
    for short, mid in ROSTER.items():
        m = ids.get(mid)
        if m is None:
            print(f"MISSING  {short}: {mid}")
            ok = False
            continue
        a = m.get("architecture", {})
        snap[mid] = dict(name=m.get("name"), created=m.get("created"), pricing=m.get("pricing"),
                         input_modalities=a.get("input_modalities"),
                         supported_parameters=m.get("supported_parameters"),
                         context_length=m.get("context_length"))
        print(f"ok  {short:18s} {mid:32s} image={'image' in (a.get('input_modalities') or [])} "
              f"in=${float(m['pricing']['prompt'])*1e6:.3f}/M out=${float(m['pricing']['completion'])*1e6:.3f}/M "
              f"temperature={'temperature' in (m.get('supported_parameters') or [])}")
    os.makedirs(LOG_DIR, exist_ok=True)
    snap["_checked"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(snap, open(os.path.join(LOG_DIR, "openrouter_roster_snapshot.json"), "w"), indent=1)
    if load_key():
        r = requests.get(KEY_URL, headers={"Authorization": f"Bearer {load_key()}"}, timeout=30)
        if r.ok:
            k = r.json().get("data", {})
            print(f"key status: HTTP 200, account usage ${k.get('usage'):.3f}, limit {k.get('limit')}")
        else:
            print(f"key status: HTTP {r.status_code} (access blocked?)")
    else:
        print("no key available")
    print(f"this job's logged spend: ${spent_usd():.3f}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ROSTER), help="comma list of roster short names")
    ap.add_argument("--set", default="legacy", choices=["legacy", "legacy_x4", "legacy_x8", "legacy_x16", "rect150"])
    ap.add_argument("--prompts", default="P1,P2,P3")
    ap.add_argument("--tag", default="main", help="run tag, part of the raw file name")
    ap.add_argument("--reasoning", default="min", help="min|default|off|minimal|low|medium|high|{json}")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma list of image ids")
    ap.add_argument("--workers", type=int, default=4, help="concurrent calls per model")
    ap.add_argument("--stop-at", type=float, default=28.0)
    ap.add_argument("--p3-views", default="c", help="quarter patches for P3: c (central, x4) and/or d0..d3")
    ap.add_argument("--max-tokens", type=int, default=None, help="override MAX_TOKENS (pilots)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    shorts = a.models.split(",")
    for s in shorts:
        assert s in ROSTER, f"{s} is not in the roster"
    assert a.stop_at <= HARD_CAP_USD
    run(shorts, a.set, a.prompts.split(","), a.tag, a.reasoning, a.repeat, a.limit,
        a.ids.split(",") if a.ids else None, a.workers, a.stop_at, a.dry_run, tuple(a.p3_views.split(",")), a.max_tokens)
