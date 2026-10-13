#!/usr/bin/env python
"""Prepare data for the thread-density annotation tool.

Writes (all under annotation_tool/data/):
    crops/{i}.jpg       rectified 1-inch window (common/rectify.py, JPEG q=92)
    crops_enh/{i}.jpg   same crop with CLAHE on the L channel (display aid only)
    thumbs/{i}.jpg      downscaled original photo with the window polygon drawn
    manifest.json       per-image metadata + crossing-point proposals

Proposals: if ../methods/proposals.json exists, its entries are merged in (rescaled if
its rect_size differs from ours); any image missing there gets a simple built-in
fallback proposal: CLAHE + high-pass; yarn period from the autocorrelation of 1-D
profiles of 9-px bands (averaged over the central half for stability); crossings =
scipy.signal.find_peaks on the 9-px band through the centre row/column with
distance ~ 0.6*period. It is only a typing aid: it often locks onto colour stripes or
onto the 2-yarn float period of plain weaves, so annotators must correct it.

RE-RUN HOOK (refresh proposals without touching crops or annotations):
    python prepare.py --refresh-proposals
Annotations never live in the manifest (they are in the browser's localStorage, the
optional server-side autosave in data/saved/, and the files you download), so
re-running this script can never delete them. In the app, an image picks up the new
proposals automatically if its record is still untouched (opened but never edited);
a record with ANY user edit is never changed -- it shows "Newer proposals exist" and the
annotator may press "Reset to proposals". Blind records never show proposals anyway.

SCENES (tool v2.1): the 150 files are 76 distinct scenes (../camera_aug/originals_provenance.json).
Every manifest image gets, from scenes.py (deterministic; reads the tool crops, never annotations):
    scene_id (= id of the scene's representative), is_scene_rep, scene_members, kind
    (rep | byte_identical | quarter_res_copy | png_copy | notebook_synthetic), copy_of,
    scene_rot_k (crop(file) ~= np.rot90(crop(rep), k), counter-clockwise), scene_swap_hv (k odd),
    scene_rot_ncc, scene_rot_margin (crop NCC of the 4 rotation hypotheses, as aug1016 does)
and the manifest gets a top-level "scenes" block. scenes_to_annotate.txt must agree with the rule.

IAA subset: version 2 = 20 distinct scenes, all reps, stratified by resolution (full vs
quarter-only scenes) x the frozen counter's weave guess (plain / twill / other), seed 2027,
largest-remainder allocation. A version-1 subset (20 files incl. same-scene pairs) is retired
once into "iaa_history"; images of it that already have a DONE record in data/saved/ are kept in
the new subset (never silently dropped). After that the subset is frozen like before.
    python prepare.py --scenes-only     # add/refresh scene fields + IAA v2; proposals untouched
                                        # (attention fields are kept as they are: re-run --attention-only)

ATTENTION (tool v2.2, "Check carefully"): every scene rep gets attention (low | medium | high),
attention_score, attention_reasons (generic text, never a count), attention_codes and count-free
attention_signals from attention.py (deterministic; reads only existing predictions -- ours, log-Gabor,
GPT-6.1-sol P1, CNN V5-B0 held-out -- plus the frozen cascade gate's confidence / resolvability and the
capture strata; never an annotation); copies get none. The manifest gets a top-level "attention" block
(rule, thresholds, input md5s, level counts) whose `updated` stamp only moves when something changed.
    python prepare.py --attention-only  # add/refresh only the attention fields; everything else untouched
A normal run recomputes them too (--no-attention: none).

Accepted proposals.json layouts: {"<id>": entry, ...}, {"proposals": {...}} or
{"images": [entry-with-"id", ...]}; entry = {"rect_size": [W, H], "h_line_y", "v_line_x",
"h_positions": [x...], "v_positions": [y...], optional "method", "weave_guess"} in
rectified-crop pixels (rescaled if W differs from our crop). A line with no positions
(e.g. the method abstained) gets the built-in fallback for that line only.
Optional yarn orientation (tool v2, shown next to the in-browser auto-align estimate,
never applied automatically): "h_angle_deg" / "v_angle_deg" in the tool's convention
(degrees, y down, positive = clockwise on screen), or the lattice method's "angle_x" /
"angle_y" (methods/counting.py), converted as h = angle_x, v = -angle_y. Orientation
fields do not enter the proposal signature, so adding them never resets an annotation.

Usage:
    python prepare.py                    # first run: crops + thumbs + manifest
    python prepare.py --refresh-proposals
    python prepare.py --refresh-proposals --proposals other_method.json
    python prepare.py --force            # rewrite all image files too
    python prepare.py --ids 0 1 22       # restrict to some ids (debugging)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # .../wvaq2027
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402

DATA = os.path.join(HERE, "data")
CROPS = os.path.join(DATA, "crops")
CROPS_ENH = os.path.join(DATA, "crops_enh")
THUMBS = os.path.join(DATA, "thumbs")
MANIFEST = os.path.join(DATA, "manifest.json")
METHOD_PROPOSALS = os.path.join(ROOT, "methods", "proposals.json")

# weave labels: methods/counting.py says "unknown", the annotation tool says "unsure"
WEAVE_GUESSES = ["plain", "twill_2_1", "twill_3_1", "twill_2_2", "satin", "other", "unknown"]
TOOL_WEAVES = ["plain", "twill_2_1", "twill_3_1", "twill_2_2", "satin", "other", "unsure"]
BAND_HALF = 4          # 9-px band
TPI_RANGE = (8, 300)   # plausible threads per inch for the fallback period search
HP_DIV = 120           # high-pass sigma = S / HP_DIV (S = crop side = 1 inch)
IAA_SIZE = 20
IAA_SEED = 2027
IAA_VERSION = 2
SAVED = os.path.join(DATA, "saved")
OURS_PRED = os.path.join(ROOT, "methods", "predictions_ours_150.csv")
SCENE_FIELDS = ("scene_id", "is_scene_rep", "scene_members", "kind", "copy_of", "scene_rot_k", "scene_swap_hv",
                "scene_rot_ncc", "scene_rot_margin")


# --------------------------------------------------------------------------- fallback
def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    S = g.shape[0]
    tiles = max(2, int(round(S / 64)))  # ~64-px tiles regardless of resolution
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tiles, tiles)).apply(g).astype(np.float64)


def _band_profile(g: np.ndarray, axis: str, centre: int) -> np.ndarray:
    lo, hi = max(0, centre - BAND_HALF), min(g.shape[0], centre + BAND_HALF + 1)
    return g[lo:hi, :].mean(axis=0) if axis == "h" else g[:, lo:hi].mean(axis=1)


def _autocorr(prof: np.ndarray, S: int) -> np.ndarray:
    hp = prof - gaussian_filter1d(prof, sigma=max(2.0, S / HP_DIV))
    x = hp - hp.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    return ac / (ac[0] + 1e-12)


def estimate_period(g: np.ndarray, axis: str, S: int):
    """Dominant yarn period (px) along `axis` via autocorrelation, or None.

    The autocorrelation is averaged over all 9-px bands in the central half of the
    crop (more stable than a single band); the band profiles are high-passed with a
    Gaussian of sigma S/HP_DIV to suppress illumination gradients and wide colour
    stripes. The smallest-lag peak within 80% of the best peak is taken (avoids
    picking 2T, 3T), then refined to sub-pixel by a parabola.
    """
    lo, hi = int(S * 0.25), int(S * 0.75)
    sl = g[lo:hi, :] if axis == "h" else g[:, lo:hi].T
    nb = sl.shape[0] // (2 * BAND_HALF + 1)
    if nb == 0:
        return None
    bands = sl[: nb * (2 * BAND_HALF + 1)].reshape(nb, 2 * BAND_HALF + 1, -1).mean(axis=1)
    ac = np.mean([_autocorr(b, S) for b in bands], axis=0)
    min_lag = max(2.0, S / TPI_RANGE[1])
    max_lag = min(len(ac) // 3, S / TPI_RANGE[0])
    peaks, _ = find_peaks(ac[: int(max_lag) + 2], prominence=0.02)
    peaks = peaks[(peaks >= min_lag) & (peaks <= max_lag)]
    if len(peaks) == 0:
        return None
    best = ac[peaks].max()
    if best <= 0.05:
        return None
    lag = int(peaks[np.argmax(ac[peaks] >= 0.8 * best)])
    if 1 <= lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return float(lag + 0.5 * (a - c) / den)
    return float(lag)


def detect_crossings(prof: np.ndarray, period: float, S: int) -> list:
    hp = prof - gaussian_filter1d(prof, sigma=max(2.0, 1.5 * period))
    sm = gaussian_filter1d(hp, sigma=max(0.5, period / 6))
    peaks, _ = find_peaks(sm, distance=max(1.0, 0.6 * period))
    return [round(float(p), 1) for p in peaks]


def fallback_proposal(crop: np.ndarray) -> dict:
    S = crop.shape[0]
    g = _clahe_gray(crop)
    c = S // 2
    out = {"rect_size": [S, S], "h_line_y": c, "v_line_x": c, "weave_guess": "unknown",
           "method": "fallback_autocorr_v2", "h_positions": [], "v_positions": []}
    for axis, key in (("h", "h_positions"), ("v", "v_positions")):
        T = estimate_period(g, axis, S)
        prof = _band_profile(g, axis, c)
        out["period_" + axis] = None if T is None else round(T, 2)
        if T is not None:
            out[key] = detect_crossings(prof, T, S)
    return out


# --------------------------------------------------------------------------- merging
def load_method_proposals(path: str = METHOD_PROPOSALS) -> dict:
    """{str(id): entry} from a proposals file in any of the accepted layouts ({} if absent)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:  # malformed / half-written file: fall back, but say so
        print(f"[warn] could not read {path}: {e}; using fallback only")
        return {}
    for wrap in ("proposals", "images", "records"):
        if isinstance(d, dict) and isinstance(d.get(wrap), (dict, list)):
            d = d[wrap]
            break
    if isinstance(d, list):
        d = {str(e["id"]): e for e in d if isinstance(e, dict) and "id" in e}
    if not isinstance(d, dict):
        return {}
    return {str(k): v for k, v in d.items() if isinstance(v, dict) and str(k).isdigit()}


def normalise_method_proposal(p: dict, S: int):
    """Validate one methods/proposals.json entry and rescale it to an S x S crop."""
    try:
        W, H = (float(v) for v in p.get("rect_size", [S, S]))
        sx, sy = S / W, S / H
        out = {
            "rect_size": [S, S],
            "h_line_y": int(round(float(p.get("h_line_y", H / 2)) * sy)),
            "v_line_x": int(round(float(p.get("v_line_x", W / 2)) * sx)),
            "h_positions": sorted(round(float(x) * sx, 1) for x in p.get("h_positions") or []
                                  if np.isfinite(float(x))),
            "v_positions": sorted(round(float(y) * sy, 1) for y in p.get("v_positions") or []
                                  if np.isfinite(float(y))),
            "weave_guess": p.get("weave_guess", "unknown") if p.get("weave_guess") in WEAVE_GUESSES else "unknown",
            "method": str(p.get("method", "methods_proposals")),
        }
    except (KeyError, TypeError, ValueError):
        return None
    out["h_line_y"] = min(max(out["h_line_y"], 0), S - 1)
    out["v_line_x"] = min(max(out["v_line_x"], 0), S - 1)
    out["h_positions"] = [x for x in out["h_positions"] if 0 <= x <= S]
    out["v_positions"] = [y for y in out["v_positions"] if 0 <= y <= S]
    out.update(proposal_angles(p))
    return out


def proposal_angles(p: dict) -> dict:
    """Optional guide-line angles of a proposals.json entry in the tool's convention."""
    out = {}
    for line, own, lat, sign in (("h", "h_angle_deg", "angle_x", 1.0), ("v", "v_angle_deg", "angle_y", -1.0)):
        a = p.get(own)
        if a is None and p.get(lat) is not None:
            a = sign * float(p[lat])
        try:
            a = float(a)
        except (TypeError, ValueError):
            continue
        if np.isfinite(a) and abs(a) <= 45:
            out[own] = round(a, 2)
    return out


def proposal_key(p: dict) -> tuple:
    return (p.get("method"), p.get("h_line_y"), p.get("v_line_x"),
            tuple(p.get("h_positions", [])), tuple(p.get("v_positions", [])))


def iaa_subset(entries: list) -> list:
    """Deterministic 20-image subset, stratified by high/low capture resolution."""
    rng = np.random.RandomState(IAA_SEED)
    hi = [e["id"] for e in entries if max(e["orig_w"], e["orig_h"]) >= 3000]
    lo = [e["id"] for e in entries if max(e["orig_w"], e["orig_h"]) < 3000]
    n_hi = int(round(IAA_SIZE * len(hi) / max(1, len(hi) + len(lo))))
    pick = list(rng.choice(hi, min(n_hi, len(hi)), replace=False)) + \
        list(rng.choice(lo, min(IAA_SIZE - n_hi, len(lo)), replace=False))
    return sorted(int(i) for i in pick)


# --------------------------------------------------------------------------- scenes (v2.1)
def coarse_weave(w) -> str:
    w = str(w or "").lower()
    if w == "plain":
        return "plain"
    if w.startswith("twill"):
        return "twill"
    return "other"          # satin, other, unknown (the counter abstained)


def counter_weave_guess(entries) -> dict:
    """{id: weave} of the frozen counter ('ours', methods/predictions_ours_150.csv); the manifest
    proposal's weave_guess (same source) where the CSV is missing."""
    out = {e["id"]: (e.get("proposal") or {}).get("weave_guess") for e in entries}
    if os.path.exists(OURS_PRED):
        import csv
        with open(OURS_PRED) as f:
            for r in csv.DictReader(f):
                if r.get("id", "").isdigit() and r.get("weave"):
                    out[int(r["id"])] = r["weave"]
    return out


def done_ids_on_disk(saved_dir: str = SAVED) -> dict:
    """{id: n done records} over the disk autosaves data/saved/*.json (read-only; backups skipped)."""
    import glob
    n = {}
    for p in sorted(glob.glob(os.path.join(saved_dir, "*.json"))):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001 -- a half-written autosave must not stop prepare
            print(f"[warn] could not read {p}: {e}")
            continue
        recs = d.get("records", []) if isinstance(d, dict) else d
        recs = list(recs.values()) if isinstance(recs, dict) else recs
        for r in recs or []:
            if isinstance(r, dict) and r.get("done") and r.get("id") is not None:
                n[int(r["id"])] = n.get(int(r["id"]), 0) + 1
    return n


def iaa_subset_v2(entries: list, forced=()) -> tuple:
    """20 distinct scenes (reps only), stratified by resolution x counter weave guess.
    `forced` ids (old-subset images with done records) are kept and count toward their cell.
    -> (sorted ids, strata table)."""
    reps = [e for e in entries if e.get("is_scene_rep")]
    wg = counter_weave_guess(entries)
    cell = {}
    for e in reps:
        res = "full" if max(e["orig_w"], e["orig_h"]) >= 3000 else "quarter_only"
        cell.setdefault((res, coarse_weave(wg.get(e["id"]))), []).append(e["id"])
    keys = sorted(cell)
    n_tot = sum(len(v) for v in cell.values())
    quota = {k: IAA_SIZE * len(cell[k]) / n_tot for k in keys}
    alloc = {k: int(np.floor(quota[k])) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[: IAA_SIZE - sum(alloc.values())]:
        alloc[k] += 1
    forced = sorted(set(int(i) for i in forced))
    rng = np.random.RandomState(IAA_SEED)
    pick, table = list(forced), []
    for k in keys:
        pool = sorted(i for i in cell[k] if i not in forced)
        need = max(0, alloc[k] - sum(1 for i in forced if i in cell[k]))
        chosen = sorted(int(i) for i in rng.choice(pool, min(need, len(pool)), replace=False)) if need else []
        pick += chosen
        table.append({"resolution": k[0], "counter_weave": k[1], "n_scenes": len(cell[k]),
                      "quota": round(quota[k], 3), "n_drawn": alloc[k], "ids": sorted(chosen + [i for i in forced if i in cell[k]])})
    return sorted(set(pick)), table


def apply_scenes(entries: list) -> dict:
    """Add the per-image scene fields (scenes.py) to the manifest entries in place -> top-level block."""
    import scenes as SC
    st = SC.compute_scenes(crops_dir=CROPS)
    missing = [e["id"] for e in entries if e["id"] not in st]
    if missing:
        raise SystemExit(f"no scene information for ids {missing[:10]}")
    for e in entries:
        s = st[e["id"]]
        for k in SCENE_FIELDS:
            if k == "scene_swap_hv":
                e[k] = bool(s["scene_rot_k"] % 2)
            else:
                e[k] = s[k]
    reps = sorted(e["id"] for e in entries if e["is_scene_rep"])
    kinds = {}
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    marg = [e["scene_rot_margin"] for e in entries if not e["is_scene_rep"]]
    return {
        "n_scenes": len(reps), "reps": reps, "kinds": kinds,
        "rot_convention": "crop(file) ~= numpy.rot90(crop(rep), scene_rot_k), counter-clockwise; "
                          "scene_swap_hv = k odd: copy labels with h<->v (and warp_dir) swapped",
        "rot_method": "crop NCC vs the rep re-rectified with rolled corners (CLAHE grey + gradient magnitude, "
                      "sigma 2 px at 256 px/in, 128 px/in, inner 84%), as aug1016",
        "rot_margin_min": round(float(min(marg)), 4) if marg else None,
        "rot_margin_median": round(float(np.median(marg)), 4) if marg else None,
        "provenance": os.path.relpath(SC.PROVENANCE, ROOT), "provenance_md5": SC.md5_of(SC.PROVENANCE),
        "reps_file": os.path.basename(SC.REPS_FILE),
        "reps_file_md5": SC.md5_of(SC.REPS_FILE) if os.path.exists(SC.REPS_FILE) else None,
    }


def choose_iaa(old_manifest: dict, entries: list, now: str) -> tuple:
    """-> (iaa ids, manifest fields to set). Version 2 is frozen once written; a version-1 subset
    is retired once into iaa_history, keeping any of its images that already have a done record."""
    old_iaa = sorted(old_manifest.get("iaa_subset") or [])
    history = list(old_manifest.get("iaa_history") or [])
    all_ids = {e["id"] for e in entries}
    reps = {e["id"] for e in entries if e.get("is_scene_rep")}
    has_scenes = bool(reps)
    if old_manifest.get("iaa_version") == IAA_VERSION and len(old_iaa) == IAA_SIZE and set(old_iaa) <= all_ids:
        return old_iaa, {"iaa_version": IAA_VERSION, "iaa_policy": old_manifest.get("iaa_policy"),
                         "iaa_history": history}       # frozen
    if not has_scenes:   # no scene information: version-1 behaviour (frozen once written)
        if len(old_iaa) == IAA_SIZE and set(old_iaa) <= all_ids:
            return old_iaa, {}
        return (iaa_subset(entries) if len(entries) >= IAA_SIZE else []), {}
    done = done_ids_on_disk(SAVED)
    print(f"IAA re-draw check: {sum(done.values())} done record(s) in {SAVED}/*.json")
    forced = [i for i in old_iaa if done.get(i)]
    if forced:
        print(f"[warn] old IAA images with DONE records are KEPT in the new subset: {forced} "
              f"(decide with the author whether that is wanted)")
    iaa, table = iaa_subset_v2(entries, forced)
    if old_iaa and old_iaa != iaa:
        history.append({"version": old_manifest.get("iaa_version", 1), "subset": old_iaa,
                        "drawn": old_manifest.get("generated"), "retired": now,
                        "reason": "v1 drew 20 FILES incl. same-scene pairs (10, 33) and (61, 69) and non-rep copies; "
                                  "replaced by 20 distinct scenes (reps) while no record was done "
                                  f"(done records on disk at retirement: {sum(done.values())})",
                        "n_done_on_disk_at_retirement": int(sum(done.values())),
                        "kept_because_done": forced})
    policy = {"rule": "20 distinct scenes, reps only; strata = resolution (full: rep >= 3000 px on the long side, "
                      "else quarter_only) x frozen counter weave guess (plain / twill / other = satin, other, unknown); "
                      "largest-remainder proportional allocation; within a cell numpy RandomState(seed).choice "
                      "of the sorted ids, cells in sorted order",
              "seed": IAA_SEED, "size": IAA_SIZE, "strata": table, "drawn": now,
              "weave_source": os.path.relpath(OURS_PRED, ROOT), "forced_kept": forced}
    bad = [i for i in iaa if i not in reps and i not in forced]
    assert not bad, bad
    return iaa, {"iaa_version": IAA_VERSION, "iaa_policy": policy, "iaa_history": history}


def load_existing_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    with open(MANIFEST) as f:
        return json.load(f)


def write_manifest(manifest: dict):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    os.replace(tmp, MANIFEST)


def scenes_block(old: dict, entries: list, meta: dict, now: str) -> dict:
    """Top-level "scenes" block; its `updated` stamp only moves when a scene field changed
    (so re-running prepare.py on unchanged inputs rewrites an identical manifest)."""
    old_by_id = {e["id"]: e for e in old.get("images", [])}
    same = bool(old.get("scenes")) and all(
        old_by_id.get(e["id"], {}).get(k) == e.get(k) for e in entries for k in SCENE_FIELDS)
    if same and {k: v for k, v in old["scenes"].items() if k != "updated"} == meta:
        return old["scenes"]
    return dict(meta, updated=now)


def scenes_only(args):
    """Add / refresh only the scene fields and the IAA subset; everything else (proposals,
    crops, timestamps) is kept exactly as it is in the existing manifest."""
    old = load_existing_manifest()
    if not old.get("images"):
        raise SystemExit("--scenes-only needs an existing data/manifest.json (run prepare.py first)")
    entries = [dict(e) for e in old["images"]]
    now = _dt.datetime.now().isoformat(timespec="seconds")
    scenes_meta = apply_scenes(entries)
    iaa, iaa_fields = choose_iaa(old, entries, now)
    man = dict(old)
    man["images"] = entries
    man["scenes"] = scenes_block(old, entries, scenes_meta, now)
    man["iaa_subset"] = iaa
    man.update(iaa_fields)
    write_manifest(man)
    print(f"wrote {MANIFEST}: {scenes_meta['n_scenes']} scenes, kinds {scenes_meta['kinds']}; rotation margin "
          f"min {scenes_meta['rot_margin_min']} median {scenes_meta['rot_margin_median']}; IAA v{man.get('iaa_version', 1)} "
          f"{iaa}; proposals untouched")


# --------------------------------------------------------------------------- attention (v2.2)
def apply_attention(old: dict, entries: list, iaa, now: str) -> dict:
    """Attention fields on the reps of `entries` (in place; attention.py) -> top-level "attention" block."""
    import attention as AT
    return AT.block(old, entries, AT.compute(entries, iaa), now)


def attention_only(args):
    """Add / refresh only the attention fields; images, proposals, scenes and the IAA subset are kept
    exactly as they are in the existing manifest. Never reads data/saved/ or any annotation."""
    old = load_existing_manifest()
    if not old.get("images"):
        raise SystemExit("--attention-only needs an existing data/manifest.json (run prepare.py first)")
    entries = [dict(e) for e in old["images"]]
    now = _dt.datetime.now().isoformat(timespec="seconds")
    man = dict(old)
    man["images"] = entries
    man["attention"] = apply_attention(old, entries, old.get("iaa_subset") or [], now)
    write_manifest(man)
    a = man["attention"]
    print(f"wrote {MANIFEST}: attention for {a['n_reps']} scene reps: {a['levels']} (outside the IAA subset "
          f"{a['levels_outside_iaa']}); missing inputs: {a['missing_inputs'] or 'none'}; everything else untouched")


# --------------------------------------------------------------------------- main
def write_jpg(path, img, q):
    tmp = path + ".tmp.jpg"
    if not cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, q]):
        raise IOError(path)
    os.replace(tmp, path)


def make_thumb(img, corners, max_side=720):
    s = max_side / max(img.shape[:2])
    t = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.polylines(t, [(corners * s).round().astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    return t


def enhance(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    tiles = max(2, int(round(crop.shape[0] / 96)))
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tiles, tiles)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main():
    global SAVED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-proposals", action="store_true",
                    help="only recompute/merge proposals and rewrite manifest.json "
                         "(reuses the existing crops; never touches annotations)")
    ap.add_argument("--proposals", default=METHOD_PROPOSALS,
                    help="proposals file to merge (default: ../methods/proposals.json)")
    ap.add_argument("--force", action="store_true", help="rewrite crops/thumbs even if present")
    ap.add_argument("--ids", type=int, nargs="*", help="restrict to these image ids")
    ap.add_argument("--scenes-only", action="store_true",
                    help="only add/refresh the scene fields and the IAA subset (v2) in the existing manifest; "
                         "proposals, crops and annotations untouched")
    ap.add_argument("--no-scenes", action="store_true",
                    help="do not compute scene fields (v2.0 manifest; IAA subset v1 rule)")
    ap.add_argument("--saved-dir", default=SAVED,
                    help="disk autosaves checked (read-only) for done records before an IAA re-draw "
                         "(default: data/saved)")
    ap.add_argument("--attention-only", action="store_true",
                    help="only add/refresh the 'check carefully' attention fields of the scene reps (attention.py; "
                         "predictions only, never annotations); everything else in the manifest untouched")
    ap.add_argument("--no-attention", action="store_true",
                    help="normal run without attention fields (the tool then shows no attention badge or filter)")
    args = ap.parse_args()
    SAVED = os.path.abspath(args.saved_dir)
    if args.scenes_only and args.attention_only:
        raise SystemExit("use --scenes-only and --attention-only one after the other")
    if args.scenes_only:
        return scenes_only(args)
    if args.attention_only:
        return attention_only(args)

    for d in (CROPS, CROPS_ENH, THUMBS):
        os.makedirs(d, exist_ok=True)
    ids = args.ids if args.ids else list(range(R.N_IMAGES))
    method_props = load_method_proposals(args.proposals)
    print(f"{args.proposals}: {'%d entries' % len(method_props) if method_props else 'not found/empty -> fallback only'}")

    old_manifest, old = {}, {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            old_manifest = json.load(f)
        old = {e["id"]: e for e in old_manifest.get("images", [])}

    entries, n_method, n_fallback, n_partial, n_changed, t0 = [], 0, 0, 0, 0, time.time()
    for i in ids:
        corners = R.load_corners(i)
        S = R.default_out_size(corners)
        rel = {k: f"data/{k}/{i}.jpg" for k in ("crops", "crops_enh", "thumbs")}
        crop_path = os.path.join(HERE, rel["crops"])
        crop = cv2.imread(crop_path) if os.path.exists(crop_path) and not args.force else None
        if crop is not None and crop.shape[:2] != (S, S):
            print(f"[warn] {rel['crops']} is {crop.shape[1]}x{crop.shape[0]}, expected {S}x{S} -> regenerating")
            crop = None
        if crop is None:  # (re)generate crop, enhanced crop and thumbnail from the original photo
            img = R.load_image(i)
            crop = R.rectify(i, img=img)
            write_jpg(crop_path, crop, 92)
            write_jpg(os.path.join(HERE, rel["crops_enh"]), enhance(crop), 92)
            write_jpg(os.path.join(HERE, rel["thumbs"]), make_thumb(img, corners), 85)
            crop = cv2.imread(crop_path)  # proposals always use exactly what the annotator sees
        elif not args.refresh_proposals:
            for k, fn in (("crops_enh", lambda im: enhance(crop)), ("thumbs", lambda im: make_thumb(im, corners))):
                if not os.path.exists(os.path.join(HERE, rel[k])):
                    write_jpg(os.path.join(HERE, rel[k]), fn(R.load_image(i)), 92 if k == "crops_enh" else 85)
        with Image.open(R.image_path(i)) as im:  # header only, no decode
            ow, oh = im.size

        prop = None
        if str(i) in method_props:
            prop = normalise_method_proposal(method_props[str(i)], S)
            if prop is None:
                print(f"[warn] proposal for {i} in {args.proposals} is malformed -> fallback")
        if prop is None:
            prop = fallback_proposal(crop)
            n_fallback += 1
        else:
            n_method += 1
            empty = [ax for ax in ("h", "v") if not prop[ax + "_positions"]]
            if empty:  # the method abstained on a line: fallback for that line only
                fb = fallback_proposal(crop)
                for ax in empty:
                    prop[ax + "_positions"] = fb[ax + "_positions"]
                    prop[("h_line_y" if ax == "h" else "v_line_x")] = fb["h_line_y" if ax == "h" else "v_line_x"]
                prop["method"] += "+fallback_" + "".join(empty)
                n_partial += 1
        if i in old and proposal_key(old[i].get("proposal", {})) != proposal_key(prop):
            n_changed += 1

        entries.append({
            "id": i,
            "file": os.path.basename(R.image_path(i)),
            "orig_w": int(ow), "orig_h": int(oh),
            "rect_size": [S, S],
            "polygon": corners.round(1).tolist(),
            "crop": rel["crops"], "crop_enh": rel["crops_enh"], "thumb": rel["thumbs"],
            "proposal": prop,
        })
        print(f"  {i:3d}: {ow}x{oh} -> {S}x{S}  "
              f"H {len(prop['h_positions']):3d}  V {len(prop['v_positions']):3d}  [{prop['method']}]")

    # keep entries for ids not processed in this run (when --ids is used)
    done_ids = {e["id"] for e in entries}
    entries += [e for k, e in old.items() if k not in done_ids]
    entries.sort(key=lambda e: e["id"])

    now = _dt.datetime.now().isoformat(timespec="seconds")
    scenes_meta = None
    import scenes as SC
    if not args.no_scenes and os.path.exists(SC.PROVENANCE) and len(entries) == R.N_IMAGES:
        scenes_meta = scenes_block(old_manifest, entries, apply_scenes(entries), now)
    elif not args.no_scenes:
        print(f"[warn] no scene fields: {SC.PROVENANCE} missing or not all {R.N_IMAGES} images present")
    # IAA: v2 (scenes) frozen once written; a v1 subset is retired once (see choose_iaa)
    iaa, iaa_fields = choose_iaa(old_manifest, entries, now)
    manifest = {
        "version": 1,
        "generated": old_manifest.get("generated", now) if args.refresh_proposals else now,
        "proposals_updated": now,
        "proposals_source": {"methods_file": os.path.abspath(args.proposals) if method_props else None,
                             "n_method": n_method, "n_fallback": n_fallback,
                             "n_method_with_line_fallback": n_partial},
        "n_images": len(entries),
        "iaa_subset": iaa,
        "weave_options": TOOL_WEAVES,
        "images": entries,
    }
    if scenes_meta:
        manifest["scenes"] = scenes_meta
    manifest.update(iaa_fields)
    import attention as AT
    if scenes_meta and not args.no_attention:   # v2.2: 'check carefully' levels of the reps (predictions only)
        manifest["attention"] = apply_attention(old_manifest, entries, iaa, now)
    else:
        for e in entries:   # never keep stale attention fields (e.g. of entries kept from the old manifest)
            for k in AT.ATTN_FIELDS:
                e.pop(k, None)
    write_manifest(manifest)
    print(f"wrote {MANIFEST}: {len(entries)} images ({n_method} method proposals "
          f"[{n_partial} with a fallback line], {n_fallback} fallback); proposals changed for "
          f"{n_changed} images; in {time.time() - t0:.1f}s")
    print("annotations untouched (they live in the browser, data/saved/ and downloads); in the app, "
          "untouched records adopt the new proposals, edited ones keep their dots.")


if __name__ == "__main__":
    main()
