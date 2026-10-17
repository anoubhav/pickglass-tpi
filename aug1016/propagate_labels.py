#!/usr/bin/env python
"""propagate_labels.py -- copy the human thread-count labels of the 150 originals onto every
usable augmented crop of the earlier augmented set (crops/{i}_aug{n}.jpg, manifest.csv).

    <venv2>/bin/python propagate_labels.py                      # default inputs (below)
    <venv2>/bin/python propagate_labels.py A.json B.csv ...     # explicit exports
    <venv2>/bin/python propagate_labels.py --watch 60           # re-run whenever an input changes

Inputs (any mix; JSON or CSV, annotation tool v1 or v2):
  * JSON: the tool's Download export or the server autosave
    annotation_tool/data/saved/annotations_<name>.json ({"records": {...}} or a list).
    Threads per inch are recomputed from the geometry with an exact port of core.js
    lineInfo(): h_tpi / v_tpi = dots on the (possibly rotated, v2) guide line inside the
    window / its length in inches (== h_count / v_count for level lines, i.e. v1).
  * CSV: the tool's Download CSV (47 v1 columns, + h_tpi/v_tpi/... in v2), or a
    cnn/train_cv.py-style CSV (image|id, h_count, v_count | warp, weft, warp_axis, weave,
    group, exclude, uncertain).
  Default (no arguments): annotation_tool/data/saved/*.json and *.csv (not backups/) plus
  aug1016/inbox/*.json|*.csv -- drop an exported annotations_*.csv into inbox/ and run.

Rules (identical to cnn/train_cv.py load_annotations so both see the same labels)
  * only done records when the input has a done field (--include-not-done to override);
    nonwoven / exclude rows are skipped; h and v must be > 0
  * the same (annotator, image) seen in several files (autosave JSON + Download JSON + CSV)
    counts once: the newest `timestamp` wins
  * several annotators of one image: per-axis mean, majority weave, uncertain if any row is
    (uncertain = confidence 1 'low' or uncertain=1)
  * augmentations: only manifest rows with usable=True, after review/review_decisions.csv
    (decision accept / reject, optional k_deg_override) is applied
  * swap_hv (k = 90 / 270): h <-> v for tpi, counts, raw counts and line angles (signs are
    preserved by a rotation), warp_dir vertical <-> horizontal; warp/weft tpi and weave
    unchanged.  k = 0 / 180: nothing changes.

Scenes (default): the 150 originals are 76 scenes (annotation_tool manifest; see
annotation_tool/propagate_scene_labels.py). Before the augmentations are labelled, each scene's label
(the rep's, else the best labelled member's) is copied to EVERY file of the scene in that file's own
orientation (h <-> v for files stored turned 90 / 270 deg relative to the rep), so augmentations of
copies (e.g. 0_aug*, a quarter-res copy of 9) are labelled too; and every augmented crop inherits its
source's scene: `group` = scene_id (use it for group-aware CV). --no-scenes: labels only from a
source's own annotation; --group-by source: group = source_id (both = the original, scene-unaware behaviour).
New columns: scene_id, source_kind, source_scene_rot_k, label_src_id. propagate() itself keeps
group = source_id unless called with group_by="scene".

Output: labels_aug1016.csv (one row per usable augmented crop, `group` = scene_id (default) or
source_id for group-aware CV; `image` = absolute crop path) + labels_aug1016_meta.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAVED = os.path.join(ROOT, "annotation_tool", "data", "saved")
INBOX = os.path.join(HERE, "inbox")
MANIFEST = os.path.join(HERE, "manifest.csv")
DECISIONS = os.path.join(HERE, "review", "review_decisions.csv")
OUT = os.path.join(HERE, "labels_aug1016.csv")
TOOL_MANIFEST = os.path.join(ROOT, "annotation_tool", "data", "manifest.json")
MAX_ANGLE = 45.0


# ---------------------------------------------------------------------------- core.js port
def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _truthy(x):
    return str(x).strip().lower() in ("1", "1.0", "true", "yes", "y")


def line_angle(rec, line):
    a = _num(rec.get(f"{line}_angle_deg"))
    return 0.0 if a is None else min(MAX_ANGLE, max(-MAX_ANGLE, a))


def line_tpi(rec, line):
    """Exact port of core.js lineInfo(rec, line).tpi."""
    S = float(rec["rect_size"])
    a = line_angle(rec, line)
    dots = rec.get(f"{line}_dots") or []
    if a == 0:
        return float(len(dots))
    k = (1 if line == "h" else -1) * math.tan(math.radians(a))
    c0 = float(rec["h_line_y"] if line == "h" else rec["v_line_x"])
    lo, hi = S / 2 + (0 - c0) / k, S / 2 + (S - c0) / k
    if lo > hi:
        lo, hi = hi, lo
    lo, hi = max(0.0, lo), min(S, hi)
    hi = max(hi, lo)
    seg_in = (hi - lo) / math.cos(math.radians(a)) / S
    n_in = sum(1 for p in dots if lo - 0.5 <= p <= hi + 0.5)
    return n_in / seg_in if seg_in > 0 else None


# ---------------------------------------------------------------------------- readers
def _from_tool_record(r, annotator, path):
    if r.get("id") is None or not isinstance(r.get("h_dots"), list) or not isinstance(r.get("v_dots"), list):
        return None
    h = line_tpi(r, "h") if r.get("rect_size") else float(len(r["h_dots"]))
    v = line_tpi(r, "v") if r.get("rect_size") else float(len(r["v_dots"]))
    return dict(source_id=int(r["id"]), annotator=r.get("annotator") or annotator or "", done=bool(r.get("done")),
                has_done=True, h_tpi=h, v_tpi=v, h_count=len(r["h_dots"]), v_count=len(r["v_dots"]),
                weave=r.get("weave"), warp_dir=r.get("warp_dir"), nonwoven=bool(r.get("nonwoven")),
                exclude=False, confidence=_num(r.get("confidence")), uncertain=_num(r.get("confidence")) == 1,
                h_angle_deg=line_angle(r, "h"), v_angle_deg=line_angle(r, "v"), group=None,
                timestamp=r.get("timestamp") or "", file=path)


def read_json(path):
    obj = json.load(open(path))
    recs = obj.get("records", {}) if isinstance(obj, dict) else obj
    if isinstance(recs, dict):
        recs = list(recs.values())
    ann = obj.get("annotator", "") if isinstance(obj, dict) else ""
    return [x for x in (_from_tool_record(r, ann, path) for r in recs) if x]


def _source_from_image_field(s):
    s = str(s).strip()
    if s.replace(".0", "").isdigit():
        return int(float(s))
    base = os.path.splitext(os.path.basename(s))[0]
    return int(base) if base.isdigit() else None


def read_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "image" not in df and "id" in df:
        df["image"] = df["id"]
    if "image" not in df:
        raise ValueError(f"{path}: needs an id or image column")
    out = []
    for r in df.to_dict("records"):
        sid = _source_from_image_field(r["image"])
        if sid is None:
            continue
        h, v = _num(r.get("h_tpi")), _num(r.get("v_tpi"))
        h = h if h is not None else _num(r.get("h_count"))
        v = v if v is not None else _num(r.get("v_count"))
        if h is None or v is None:          # train_cv-style warp / weft
            wa, we = _num(r.get("warp")), _num(r.get("weft"))
            if wa is not None and we is not None:
                axis = str(r.get("warp_axis", "") or "vertical").strip().lower()
                h, v = (wa, we) if axis.startswith("v") or axis in ("nan", "") else (we, wa)
        warp_dir = r.get("warp_dir") or (r.get("warp_axis") or None)
        conf = _num(r.get("confidence"))
        out.append(dict(source_id=sid, annotator=r.get("annotator", ""), done=_truthy(r.get("done", "")),
                        has_done="done" in r, h_tpi=h, v_tpi=v, h_count=_num(r.get("h_count")),
                        v_count=_num(r.get("v_count")), weave=r.get("weave") or None, warp_dir=warp_dir or None,
                        nonwoven=_truthy(r.get("nonwoven", "")), exclude=_num(r.get("exclude")) == 1,
                        confidence=conf, uncertain=_num(r.get("uncertain")) == 1 or conf == 1,
                        h_angle_deg=_num(r.get("h_angle_deg")) or 0.0, v_angle_deg=_num(r.get("v_angle_deg")) or 0.0,
                        group=r.get("group") or None, timestamp=r.get("timestamp", ""), file=path))
    return out


def default_inputs():
    pats = [os.path.join(SAVED, "*.json"), os.path.join(SAVED, "*.csv"),
            os.path.join(INBOX, "*.json"), os.path.join(INBOX, "*.csv")]
    return sorted(p for pat in pats for p in glob.glob(pat))


def read_annotations(paths, include_not_done=False, log=print):
    """-> (per-source merged DataFrame, per-row DataFrame after dedup/filters)."""
    rows = []
    for p in paths:
        try:
            rows += read_json(p) if p.lower().endswith(".json") else read_csv(p)
        except Exception as e:  # a half-written autosave must not stop the pipeline
            log(f"  skipped {p}: {e}")
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(), raw
    # same annotator + image in several files -> newest timestamp wins
    raw = raw.sort_values("timestamp").drop_duplicates(["annotator", "source_id"], keep="last")
    keep = raw[~raw.nonwoven & ~raw.exclude]
    if not include_not_done:
        keep = keep[~keep.has_done | keep.done]
    keep = keep[keep.h_tpi.notna() & keep.v_tpi.notna()]
    keep = keep[(keep.h_tpi > 0) & (keep.v_tpi > 0)]
    merged = []
    for sid, g in keep.groupby("source_id"):
        ws = [w for w in g.weave if w not in (None, "", "unsure")]
        wd = [w for w in g.warp_dir if w in ("vertical", "horizontal")]
        merged.append(dict(
            source_id=int(sid), h_tpi=float(g.h_tpi.mean()), v_tpi=float(g.v_tpi.mean()),
            h_count=float(g.h_count.mean()) if g.h_count.notna().any() else np.nan,
            v_count=float(g.v_count.mean()) if g.v_count.notna().any() else np.nan,
            weave=Counter(ws).most_common(1)[0][0] if ws else (g.weave.iloc[0] or None),
            warp_dir=(wd[0] if len(set(wd)) == 1 else "unknown") if wd else (g.warp_dir.iloc[0] or None),
            h_angle_deg=float(g.h_angle_deg.mean()), v_angle_deg=float(g.v_angle_deg.mean()),
            uncertain=bool(g.uncertain.any()), confidence=g.confidence.min(),
            n_annotations=len(g), annotators=";".join(sorted(set(map(str, g.annotator)))),
            source_group=next((x for x in g.group if x), None),
            label_files=";".join(sorted(set(os.path.basename(f) for f in g.file)))))
    return pd.DataFrame(merged), keep


# ---------------------------------------------------------------------------- manifest + swap
def load_manifest(apply_decisions=True, usable_only=True, include_flagged=False):
    man = pd.read_csv(MANIFEST)
    man["review_decision"] = ""
    if apply_decisions and os.path.exists(DECISIONS):
        dec = pd.read_csv(DECISIONS, dtype=str, keep_default_na=False).set_index("aug_id")
        for aid, d in dec.iterrows():
            sel = man.aug_id == aid
            if not sel.any():
                continue
            choice = d.get("decision", "").strip().lower()
            if choice == "accept":
                man.loc[sel, "usable"] = True
                ko = _num(d.get("k_deg_override", ""))
                if ko is not None:
                    man.loc[sel, "k_deg"] = int(ko) % 360
                    man.loc[sel, "swap_hv"] = int(ko) % 360 in (90, 270)
            elif choice == "reject":
                man.loc[sel, "usable"] = False
            man.loc[sel, "review_decision"] = choice
    if usable_only and not include_flagged:
        man = man[man.usable.astype(bool)]
    man["crop_path"] = [os.path.join(HERE, c) for c in man.crop]
    return man.reset_index(drop=True)


SWAP_PAIRS = [("h_tpi", "v_tpi"), ("h_count", "v_count"), ("h_angle_deg", "v_angle_deg")]
WARP_SWAP = {"vertical": "horizontal", "horizontal": "vertical"}


# ---------------------------------------------------------------------------- scenes
def scene_module(manifest_path=TOOL_MANIFEST):
    """annotation_tool/propagate_scene_labels.py of the tool that owns the manifest."""
    import importlib.util
    tool = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
    spec = importlib.util.spec_from_file_location("propagate_scene_labels", os.path.join(tool, "propagate_scene_labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_scenes(manifest_path=TOOL_MANIFEST, log=print):
    """Per-file scene table of the 150 originals, or None (no scene fields / tool missing)."""
    try:
        return scene_module(manifest_path).load_scenes(manifest_path)
    except (OSError, ValueError, KeyError) as e:
        log(f"  no scene information ({e}); labels only from each source's own annotation")
        return None


def expand_to_scenes(src: pd.DataFrame, scenes: pd.DataFrame, manifest_path=TOOL_MANIFEST) -> pd.DataFrame:
    """Per-source labels (read_annotations) -> one row per FILE of every labelled scene, in that file's
    own orientation (annotation_tool/propagate_scene_labels.py: the rep's label wins, h <-> v for odd
    rotations relative to the rep). Adds scene_id, source_kind, source_scene_rot_k, label_src_id."""
    PS = scene_module(manifest_path)
    own = src.set_index("source_id")
    own = own[own.index.isin(scenes.index)]
    pf, ps = PS.propagate(own, scenes)
    pf = pf.rename(columns={"kind": "source_kind", "scene_rot_k": "source_scene_rot_k"})
    pf["label_src_id"] = [int(ps.at[s, "label_src_id"]) for s in pf.scene_id]
    pf = pf.drop(columns=[c for c in ("is_scene_rep", "swap_hv", "copy_of", "rect_size", "scene_resolution",
                                      "is_test_item", "iaa", "own_h_tpi", "own_v_tpi", "has_own_label") if c in pf])
    pf.index.name = "source_id"
    return pf.reset_index()


def propagate(src: pd.DataFrame, man: pd.DataFrame, group_by: str = "source", scenes=None) -> pd.DataFrame:
    """src: one row per source_id (columns as read_annotations()'s merged frame; only
    source_id/h_tpi/v_tpi are required).  -> one row per (usable) augmented crop.
    group = source_id (default) or, with group_by="scene", the source's scene (scene_id, from src or
    from `scenes`, the per-file scene table of propagate_scene_labels.load_scenes())."""
    m = man.merge(src, on="source_id", how="inner", suffixes=("", "_src"))
    sw = m.swap_hv.astype(bool).values
    for a, b in SWAP_PAIRS:
        if a in m and b in m:
            va, vb = m[a].values.copy(), m[b].values.copy()
            m[a] = np.where(sw, vb, va)
            m[b] = np.where(sw, va, vb)
    if "warp_dir" in m:
        m["warp_dir"] = [WARP_SWAP.get(w, w) if s else w for w, s in zip(m.warp_dir, sw)]
    if "scene_id" not in m and scenes is not None:
        m["scene_id"] = scenes.scene_id.reindex(m.source_id).to_numpy()
    if group_by == "scene":
        if "scene_id" not in m:
            raise ValueError("group_by='scene' needs scene ids (src.scene_id or scenes=)")
        m["group"] = m.scene_id
    else:
        m["group"] = m.source_id
    m["image"] = m.crop_path
    return m


OUT_COLS = ["image", "aug_id", "source_id", "group", "scene_id", "source_kind", "source_scene_rot_k", "label_src_id",
            "k_deg", "swap_hv", "h_tpi", "v_tpi", "h_count", "v_count",
            "weave", "warp_dir", "h_angle_deg", "v_angle_deg", "uncertain", "confidence", "n_annotations",
            "annotators", "source_group", "rect_size", "source_rect_size", "review_decision", "label_files"]


def run(paths, out=OUT, include_not_done=False, include_flagged=False, with_originals=False, log=print,
        scenes=True, group_by="scene"):
    src, rows = read_annotations(paths, include_not_done, log)
    n_own = len(src)
    man = load_manifest(include_flagged=include_flagged)
    sc = load_scenes(log=log) if scenes else None
    if sc is None and group_by == "scene":
        group_by = "source"
    if sc is not None and not src.empty:
        src = expand_to_scenes(src, sc)
    if src.empty:
        lab = pd.DataFrame(columns=OUT_COLS)
    else:
        lab = propagate(src, man, group_by=group_by, scenes=sc)
        lab = lab[[c for c in OUT_COLS if c in lab]]
        if with_originals:
            o = src.copy()
            o["image"], o["aug_id"], o["k_deg"], o["swap_hv"] = o.source_id, "", 0, False
            o["group"] = o.scene_id if group_by == "scene" else o.source_id
            lab = pd.concat([o[[c for c in OUT_COLS if c in o]], lab], ignore_index=True)
    tmp = out + ".tmp"
    lab.to_csv(tmp, index=False)
    os.replace(tmp, out)
    meta = dict(written=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs={p: os.path.getmtime(p) for p in paths},
                n_label_rows_used=int(len(rows)), n_sources_with_own_label=int(n_own),
                n_sources_labelled=int(len(src)), scenes=sc is not None, group_by=group_by,
                n_aug_rows=int((lab.aug_id != "").sum()) if len(lab) else 0, n_manifest_usable=int(len(man)),
                include_not_done=include_not_done, include_flagged=include_flagged)
    with open(os.path.splitext(out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    log(f"{len(paths)} input file(s), {len(rows)} usable label rows -> {n_own} sources with an own label -> "
        f"{len(src)} labelled sources{' (scene labels copied to every file of the scene)' if sc is not None else ''} -> "
        f"{meta['n_aug_rows']} augmented crops labelled (of {len(man)} usable) -> {out}")
    return lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="annotation exports (JSON/CSV); default: saved/ + inbox/")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--include-not-done", action="store_true", help="also use records not marked done")
    ap.add_argument("--include-flagged", action="store_true", help="also label rows with usable=False (not advised)")
    ap.add_argument("--with-originals", action="store_true", help="also write the 150 source rows (image = id)")
    ap.add_argument("--no-scenes", action="store_true",
                    help="label augmentations only from their source's own annotation (the original, scene-unaware behaviour)")
    ap.add_argument("--group-by", choices=("scene", "source"), default="scene",
                    help="`group` column: the source's scene (default) or the source id (the original behaviour)")
    ap.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                    help="poll the inputs and re-run whenever one appears or changes")
    a = ap.parse_args()
    kw = dict(out=a.out, include_not_done=a.include_not_done, include_flagged=a.include_flagged,
              with_originals=a.with_originals, scenes=not a.no_scenes, group_by=a.group_by)
    if not a.watch:
        run(a.inputs or default_inputs(), **kw)
        return
    os.makedirs(INBOX, exist_ok=True)
    last = None
    while True:
        paths = a.inputs or default_inputs()
        sig = tuple((p, os.path.getmtime(p)) for p in paths if os.path.exists(p))
        for extra in (DECISIONS, TOOL_MANIFEST):   # review decisions / scene fields changed -> re-run
            if os.path.exists(extra):
                sig += ((extra, os.path.getmtime(extra)),)
        if sig != last:
            print(time.strftime("[%H:%M:%S]"), end=" ")
            run(paths, **kw)
            last = sig
        sys.stdout.flush()
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
