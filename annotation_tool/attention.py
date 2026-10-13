#!/usr/bin/env python
"""'Check carefully' attention levels for the 76 scene representatives (annotation tool v2.2).

For every scene representative (the photo that is annotated) this module derives a level
(low / medium / high) and a few generic reasons from EXISTING machine outputs only. It never reads
an annotation (nothing under data/saved/, no export, no propagated label) and uses only the standard
library, so it is deterministic: the same inputs always give the same manifest fields.
prepare.py calls it (--attention-only, and in every normal run).

The tool shows the level as a badge with a reason ("Check carefully: estimators disagree on H by 41%")
on photos that are not in the IAA subset and whose record is not blind. It never shows a count, so the
manifest does not hold one either: only the level, the reasons, codes and count-free signals.

Estimators, per axis (H = vertical yarns crossing the H line, V = horizontal yarns crossing the V line;
all of them count on the same rectified crop, so the axes are unambiguous):
  ours       the frozen counter, methods/predictions_ours_150.csv h_count / v_count (its dots are the
             proposals the annotator sees)
  log_gabor  the log-Gabor bank, same file, h_gabor / v_gabor
  gpt_p1     GPT-6.1-sol, prompt P1, vlm/results/rect150_P1_predictions_long.csv (model gpt-6.1-sol,
             parse_ok; vertical = H, horizontal = V)
  cnn_v5b0   the best CNN, V5 EfficientNet-B0 (V4 + TTA), cnn/runs_aug/report/real150_aug_wide_heldout.csv
             cnn-aug-V5-b0__v = H, __h = V. The file is blank on the scenes the CNN was fine-tuned on
             (pseudo-labels), so the CNN is left out there ("where valid").
Counts that are missing, non-finite or <= 0 are left out.

Disagreement on an axis, with l_i = ln(count_i) over the available estimators and m = median(l_i):
    D = exp(max(|l_ours - m|, median_i |l_i - m|)) - 1          (in %)
i.e. how far the proposals are from the estimators' consensus, or how far the estimators typically are
from it, whichever is larger. A single outlier (e.g. one estimator at twice the others) does not raise
D; a 2:2 split between a count and its double gives D = 41 %.

Points (pre-declared on Oct 8 2026, before any human count existed; thresholds are the paper's units):
  disagreement   max over axes D >= 33 %  -> 2   (the estimators do not agree on the counting unit)
                 max over axes D >= 10 %  -> 1   (the paper's tolerance); one reason per axis with D >= 10 %
  px per yarn    S / (consensus count exp(m) of the finer axis) < 4 -> 2, < 5 -> 1
                 (Sec. 5.3: counts break below about 4 px per yarn; a capture should have 4-5)
  resolvability  eta < r_min of the frozen cascade gate -> 2; eta < the 5th percentile on the 150 clean
                 photos (label-free) -> 1. eta: paper/tables/cascade_150_decisions.csv (rho); thresholds:
                 paper/tables/cascade_thresholds.json (fits.vlm9_a10.gate.r_min, fits.real150_clean_rho_pct.p05)
  glare          near-saturated window pixels (all channels >= 240, paper/tables/strata150.csv sat240) > 1 %
                 -> 1 (the pre-declared capture stratum)
  confidence     ours' confidence < t_c of the frozen cascade gate (fits.vlm9_a10.gate.t_c) -> 1
  level          score >= 2 high, 1 medium, 0 low

    <venv2>/bin/python attention.py            # print the table and the distribution (writes nothing)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # .../wvaq2027

VERSION = 1
LEVELS = ("low", "medium", "high")
P_OURS = os.path.join("methods", "predictions_ours_150.csv")
P_VLM = os.path.join("vlm", "results", "rect150_P1_predictions_long.csv")
P_CNN = os.path.join("cnn", "runs_aug", "report", "real150_aug_wide_heldout.csv")
P_DECISIONS = os.path.join("paper", "tables", "cascade_150_decisions.csv")
P_THRESH = os.path.join("paper", "tables", "cascade_thresholds.json")
P_STRATA = os.path.join("paper", "tables", "strata150.csv")
INPUTS = (P_OURS, P_VLM, P_CNN, P_DECISIONS, P_THRESH, P_STRATA)
VLM_MODEL = "gpt-6.1-sol"
CNN_KEY = "cnn-aug-V5-b0"
GATE_FIT = "vlm9_a10"
ESTIMATORS = ("ours", "log_gabor", "gpt_p1", "cnn_v5b0")
ESTIMATOR_SOURCES = {
    "ours": P_OURS + " h_count / v_count (the frozen counter; its dots are the proposals)",
    "log_gabor": P_OURS + " h_gabor / v_gabor",
    "gpt_p1": P_VLM + " model " + VLM_MODEL + ", parse_ok; vertical = H, horizontal = V",
    "cnn_v5b0": P_CNN + " " + CNN_KEY + "__v = H, __h = V (blank on its fine-tuning scenes -> left out)",
}
DIS_STRONG = 100.0 / 3.0   # % -> 2 points
DIS_TOL = 10.0             # % -> 1 point; also the minimum for a disagreement reason
PPY_BREAK = 4.0            # px per yarn -> 2 points
PPY_ASK = 5.0              # px per yarn -> 1 point
GLARE_SAT240 = 0.01        # share of window pixels -> 1 point
REASON_TEXT = {
    "fine_yarns_2": "very fine yarns for this resolution: zoom in well",
    "fine_yarns_1": "fine yarns for this resolution: zoom in",
    "low_resolvability_2": "the photo may be too soft to resolve every yarn (blur)",
    "low_resolvability_1": "the photo may be too soft to resolve every yarn (blur)",
    "glare": "glare hides part of the window: move the line to a clear row",
    "low_confidence": "the automatic counter is unsure about this photo",
}
RULE = ("per axis D = exp(max(|ln ours - m|, median_i |ln e_i - m|)) - 1, m = median of ln counts of the available "
        "estimators (ours, log_gabor, gpt_p1, cnn_v5b0 where valid); points: max D >= 33% -> 2, >= 10% -> 1; "
        "px per yarn = S / exp(m) of the finer axis < 4 -> 2, < 5 -> 1; eta < gate r_min -> 2, < p05 of the 150 "
        "clean photos -> 1; sat240 > 1% -> 1; ours' confidence < gate t_c -> 1; level: >= 2 high, 1 medium, 0 low. "
        "Computed for scene reps only, from predictions only (no annotation is read).")
UI_RULE = ("badge + reasons for high and medium; nothing for low; never on IAA-subset scenes, blind records or "
           "copies; no count is ever shown; 'High attention first' lists the reps where it is visible")


# --------------------------------------------------------------------------- inputs
def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_inputs(root=ROOT):
    """All machine inputs keyed by file id (read-only). Missing files are listed, not fatal (except ours)."""
    p = {k: os.path.join(root, k) for k in INPUTS}
    missing = [k for k in INPUTS if not os.path.exists(p[k])]
    if P_OURS in missing:
        raise SystemExit(f"attention: {p[P_OURS]} is required")
    est = {e: {} for e in ESTIMATORS}   # estimator -> {id: (h, v)}
    conf = {}
    for r in _rows(p[P_OURS]):
        if not str(r.get("id", "")).isdigit():
            continue
        i = int(r["id"])
        est["ours"][i] = (_f(r.get("h_count")), _f(r.get("v_count")))
        est["log_gabor"][i] = (_f(r.get("h_gabor")), _f(r.get("v_gabor")))
        conf[i] = _f(r.get("confidence"))
    if P_VLM not in missing:
        for r in _rows(p[P_VLM]):
            if r.get("model") == VLM_MODEL and str(r.get("parse_ok")).strip().lower() == "true":
                est["gpt_p1"][int(r["image_id"])] = (_f(r.get("vertical")), _f(r.get("horizontal")))
    if P_CNN not in missing:
        for r in _rows(p[P_CNN]):
            est["cnn_v5b0"][int(r["image_id"])] = (_f(r.get(CNN_KEY + "__v")), _f(r.get(CNN_KEY + "__h")))
    eta = {}
    if P_DECISIONS not in missing:
        for r in _rows(p[P_DECISIONS]):
            eta[int(r["image"])] = _f(r.get("rho"))
    th = {"t_c": None, "r_min": None, "eta_p05": None}
    if P_THRESH not in missing:
        with open(p[P_THRESH]) as f:
            fits = json.load(f).get("fits", {})
        gate = fits.get(GATE_FIT, {}).get("gate", {})
        th = {"t_c": _f(gate.get("t_c")), "r_min": _f(gate.get("r_min")),
              "eta_p05": _f(fits.get("real150_clean_rho_pct", {}).get("p05"))}
    strata = {}
    if P_STRATA not in missing:
        for r in _rows(p[P_STRATA]):
            strata[int(r["id"])] = {"S": _f(r.get("S")), "sat240": _f(r.get("sat240"))}
    md5 = {k: md5_of(p[k]) for k in INPUTS if k not in missing}
    return {"est": est, "conf": conf, "eta": eta, "th": th, "strata": strata, "missing": missing, "md5": md5}


# --------------------------------------------------------------------------- rule
def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def disagreement(counts: dict):
    """counts {estimator: count or None} for one axis -> (D in %, n used, consensus count) or (None, n, None).
    D = exp(max(|ln ours - m|, median |ln e - m|)) - 1, m = median of the ln counts. Needs ours and >= 2 counts."""
    logs = {k: math.log(v) for k, v in counts.items() if v is not None and v > 0 and math.isfinite(v)}
    if "ours" not in logs or len(logs) < 2:
        return None, len(logs), (math.exp(_median(list(logs.values()))) if logs else None)
    m = _median(list(logs.values()))
    dev = max(abs(logs["ours"] - m), _median([abs(x - m) for x in logs.values()]))
    return 100.0 * (math.exp(dev) - 1.0), len(logs), math.exp(m)


def assess(i: int, inp: dict) -> dict:
    """Level, score, reasons, codes and count-free signals of one file id."""
    th, est = inp["th"], inp["est"]
    items = []   # (points, order, code, text)
    sig = {"estimators": [e for e in ESTIMATORS if i in est[e] and any(v is not None for v in est[e][i])]}
    cons = {}
    for k, ax in ((0, "h"), (1, "v")):
        counts = {e: (est[e].get(i) or (None, None))[k] for e in ESTIMATORS}
        d, n, c = disagreement(counts)
        sig["d_" + ax] = None if d is None else round(d, 1)   # the reasons quote this value, rounded half up
        sig["n_" + ax] = n
        cons[ax] = c
    ds = [(sig["d_" + ax], ax) for ax in ("h", "v") if sig["d_" + ax] is not None]   # thresholds on the stored value
    dmax = max((d for d, _ in ds), default=None)
    if dmax is not None and dmax >= DIS_TOL:
        pts = 2 if dmax >= DIS_STRONG else 1
        first = True
        for d, ax in sorted(ds, key=lambda t: (-t[0], t[1])):
            if d >= DIS_TOL:
                items.append((pts if first else 0, 0, "disagree_" + ax,
                              f"estimators disagree on {ax.upper()} by {int(math.floor(d + 0.5))}%"))
                first = False
    S = (inp["strata"].get(i) or {}).get("S")
    finer = max((c for c in cons.values() if c), default=None)
    ppy = S / finer if S and finer else None
    sig["ppy_bin"] = None if ppy is None else ("<4" if ppy < PPY_BREAK else "4-5" if ppy < PPY_ASK else ">=5")
    if ppy is not None and ppy < PPY_ASK:
        p = 2 if ppy < PPY_BREAK else 1
        items.append((p, 1, "fine_yarns", REASON_TEXT["fine_yarns_%d" % p]))
    eta = inp["eta"].get(i)
    sig["eta"] = None if eta is None else round(eta, 5)
    if eta is not None and th["r_min"] is not None and eta < th["r_min"]:
        items.append((2, 2, "low_resolvability", REASON_TEXT["low_resolvability_2"]))
    elif eta is not None and th["eta_p05"] is not None and eta < th["eta_p05"]:
        items.append((1, 2, "low_resolvability", REASON_TEXT["low_resolvability_1"]))
    sat = (inp["strata"].get(i) or {}).get("sat240")
    sig["sat240"] = None if sat is None else round(sat, 4)
    if sat is not None and sat > GLARE_SAT240:
        items.append((1, 3, "glare", REASON_TEXT["glare"]))
    conf = inp["conf"].get(i)
    sig["conf"] = None if conf is None else round(conf, 3)
    if conf is not None and th["t_c"] is not None and conf < th["t_c"]:
        items.append((1, 4, "low_confidence", REASON_TEXT["low_confidence"]))
    score = sum(t[0] for t in items)
    items.sort(key=lambda t: (t[1], -t[0]))   # disagreement (largest axis first), then the capture checks
    level = "high" if score >= 2 else "medium" if score == 1 else "low"
    return {"attention": level, "attention_score": int(score),
            "attention_reasons": [t[3] for t in items] if score else [],
            "attention_codes": [t[2] for t in items] if score else [],
            "attention_signals": sig}


ATTN_FIELDS = ("attention", "attention_score", "attention_reasons", "attention_codes", "attention_signals")


def compute(entries: list, iaa=(), root=ROOT) -> dict:
    """Set the attention fields on every scene rep of `entries` (in place) and remove them from every
    other entry. -> the manifest's top-level "attention" block (without its `updated` stamp)."""
    inp = load_inputs(root)
    reps = [e for e in entries if e.get("is_scene_rep")]
    if not reps:
        raise SystemExit("attention: the manifest has no scene reps (run prepare.py --scenes-only first)")
    for e in entries:
        for k in ATTN_FIELDS:
            e.pop(k, None)
    for e in reps:
        e.update(assess(int(e["id"]), inp))
    iaa = set(int(i) for i in iaa)
    count = lambda es: {lv: sum(1 for e in es if e["attention"] == lv) for lv in LEVELS}  # noqa: E731
    th = dict(inp["th"], dis_strong_pct=round(DIS_STRONG, 2), dis_tol_pct=DIS_TOL, ppy_break=PPY_BREAK,
              ppy_ask=PPY_ASK, glare_sat240=GLARE_SAT240)
    return {
        "version": VERSION, "rule": RULE, "ui": UI_RULE, "thresholds": th,
        "estimators": ESTIMATOR_SOURCES, "gate_fit": f"{P_THRESH} fits.{GATE_FIT}.gate",
        "inputs_md5": inp["md5"], "missing_inputs": inp["missing"],
        "n_reps": len(reps), "levels": count(reps),
        "levels_outside_iaa": count([e for e in reps if int(e["id"]) not in iaa]),
        "levels_iaa": count([e for e in reps if int(e["id"]) in iaa]),
        "cnn_valid_reps": sum(1 for e in reps if "cnn_v5b0" in e["attention_signals"]["estimators"]),
        "codes": {c: sum(1 for e in reps if c in e["attention_codes"]) for c in
                  ("disagree_h", "disagree_v", "fine_yarns", "low_resolvability", "glare", "low_confidence")},
    }


def block(old_manifest: dict, entries: list, meta: dict, now: str) -> dict:
    """Top-level block; `updated` only moves when a field or the block changed (idempotent re-runs)."""
    old_by_id = {e["id"]: e for e in old_manifest.get("images", [])}
    same = bool(old_manifest.get("attention")) and all(
        old_by_id.get(e["id"], {}).get(k) == e.get(k) for e in entries for k in ATTN_FIELDS)
    if same and {k: v for k, v in old_manifest["attention"].items() if k != "updated"} == meta:
        return old_manifest["attention"]
    return dict(meta, updated=now)


def main():
    man = json.load(open(os.path.join(HERE, "data", "manifest.json")))
    entries = [dict(e) for e in man["images"]]
    meta = compute(entries, man.get("iaa_subset") or [])
    iaa = set(man.get("iaa_subset") or [])
    print(f"{'id':>4} {'iaa':>3} {'level':>6} {'pts':>3}  {'D_h':>6} {'D_v':>6} n  ppy   reasons")
    for e in sorted((e for e in entries if e.get("is_scene_rep")), key=lambda e: (-e["attention_score"], e["id"])):
        s = e["attention_signals"]
        print(f"{e['id']:>4} {'*' if e['id'] in iaa else '':>3} {e['attention']:>6} {e['attention_score']:>3}  "
              f"{s['d_h'] if s['d_h'] is not None else '-':>6} {s['d_v'] if s['d_v'] is not None else '-':>6} "
              f"{s['n_h']}  {s['ppy_bin'] or '-':<5} {'; '.join(e['attention_reasons'])}")
    print(json.dumps({k: meta[k] for k in ("levels", "levels_outside_iaa", "levels_iaa", "cnn_valid_reps", "codes",
                                           "missing_inputs")}, indent=1))


if __name__ == "__main__":
    main()
