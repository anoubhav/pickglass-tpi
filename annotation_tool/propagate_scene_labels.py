#!/usr/bin/env python
"""propagate_scene_labels.py -- copy each scene's human labels onto every file of the scene.

The 150 files are 76 scenes (prepare.py / scenes.py; manifest fields scene_id, is_scene_rep,
kind, scene_rot_k). Only the scene representative ("rep") is annotated; this script gives every
file of the scene the rep's labels, in the FILE's own crop orientation:

    k = scene_rot_k of the file  (crop(file) ~= numpy.rot90(crop(rep), k), counter-clockwise)
    k even (0, 180 deg): identity
    k odd  (90, 270 deg): h <-> v for threads/inch, counts and guide-line angles (a rotation keeps
                          the sign of the clockwise-positive line angles), warp_dir vertical <->
                          horizontal; weave and quality flags unchanged

This applies to every non-rep kind. Byte-identical, PNG and quarter-resolution copies are the same
exposure, but 41 of these 63 files are stored turned by 90 or 180 deg relative to their rep (e.g. 33
quarter-res copies at k = 3), so copying labels unchanged would be wrong for the 36 with odd k. The 11
notebook_synthetic files are turned +-90 deg relative to the file they were made from, which is an
odd k relative to the rep for 9 of them (k = 0 for 139 and k = 2 for 132, whose sources are rotated copies).

Scene label (in the rep's frame), --policy:
    rep  (default)  the rep's own label; if the rep has none, the label of the best labelled member
                    (real before synthetic, then larger crop, then kind rep > byte_identical >
                    png_copy > quarter_res_copy, then lower id), rotated into the rep's frame
    pool            per-axis mean over all labelled members (rotated into the rep's frame)
A file's own label (if a copy was annotated after "Edit anyway") is kept in own_* columns and is
never used instead of the rep's (only as a fallback when the rep has none).

Evaluation units (used by paper/tools/score150.py, cnn/train_cv.py, aug1016/propagate_labels.py):
    primary   one row per scene = the rep (76 rows)
    secondary one row per file, real files only (139; is_test_item)
    notebook_synthetic files are augmentations: never test items, training only, in their scene's fold.

    <venv2>/bin/python propagate_scene_labels.py                  # data/saved/*.json|csv (read-only)
    <venv2>/bin/python propagate_scene_labels.py A.json B.csv     # explicit exports (JSON or CSV, v1/v2)
    options: --policy rep|pool  --include-not-done  --out-dir DIR (default: propagated/)
Writes DIR/labels_per_file.csv (150 rows, labelled scenes only filled), DIR/labels_per_scene.csv
(76 rows) and DIR/labels_meta.json. Inputs are only read.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "data", "manifest.json")
SAVED = os.path.join(HERE, "data", "saved")
OUT_DIR = os.path.join(HERE, "propagated")
MAX_ANGLE = 45.0
KIND_ORDER = {"rep": 0, "byte_identical": 1, "png_copy": 2, "quarter_res_copy": 3, "notebook_synthetic": 4}
WARP_SWAP = {"vertical": "horizontal", "horizontal": "vertical"}
# column pairs swapped for odd k (whichever are present in a frame)
SWAP_PAIRS = [("h_tpi", "v_tpi"), ("h_count", "v_count"), ("h_angle_deg", "v_angle_deg"), ("h", "v"),
              ("h_count_raw", "v_count_raw"), ("own_h_tpi", "own_v_tpi")]


# ---------------------------------------------------------------------------- scene table
def load_scenes(manifest=MANIFEST) -> pd.DataFrame:
    """Per-file scene table (index = id) from the tool manifest (a path or the loaded dict)."""
    m = manifest
    if isinstance(manifest, str):
        with open(manifest) as f:
            m = json.load(f)
    rows = []
    for e in m["images"]:
        if e.get("scene_id") is None or not isinstance(e.get("is_scene_rep"), bool):
            raise ValueError("the manifest has no scene fields: run annotation_tool/prepare.py --scenes-only")
        k = int(e.get("scene_rot_k") or 0) % 4
        rows.append(dict(id=int(e["id"]), scene_id=int(e["scene_id"]), is_scene_rep=bool(e["is_scene_rep"]),
                         kind=e["kind"], scene_rot_k=k, swap_hv=bool(k % 2), copy_of=e.get("copy_of"),
                         scene_members=" ".join(str(x) for x in e["scene_members"]),
                         rect_size=int(e["rect_size"][0]), orig_w=int(e["orig_w"]), orig_h=int(e["orig_h"]),
                         is_test_item=e["kind"] != "notebook_synthetic"))
    sc = pd.DataFrame(rows).set_index("id").sort_index()
    rep_long = {i: max(r.orig_w, r.orig_h) for i, r in sc[sc.is_scene_rep].iterrows()}
    sc["scene_resolution"] = ["full" if rep_long[s] >= 3000 else "quarter_only" for s in sc.scene_id]
    iaa = set(m.get("iaa_subset") or [])
    sc["iaa"] = [i in iaa for i in sc.index]
    n_reps = int(sc.is_scene_rep.sum())
    if n_reps != sc.scene_id.nunique() or not (sc[sc.is_scene_rep].index == sc[sc.is_scene_rep].scene_id).all():
        raise ValueError("inconsistent scene fields (each scene needs exactly one rep with scene_id == its id)")
    return sc


def swap_hv(df: pd.DataFrame, mask) -> pd.DataFrame:
    """Swap the h/v column pairs (and warp_dir) of the rows where mask is True."""
    df = df.copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return df
    for a, b in SWAP_PAIRS:
        if a in df and b in df:
            va, vb = df[a].to_numpy(copy=True), df[b].to_numpy(copy=True)
            df[a] = np.where(mask, vb, va)
            df[b] = np.where(mask, va, vb)
    if "warp_dir" in df:
        df["warp_dir"] = [WARP_SWAP.get(w, w) if s else w for w, s in zip(df["warp_dir"], mask)]
    return df


def _majority(vals):
    vs = [v for v in vals if isinstance(v, str) and v not in ("", "unsure", "unknown", "nan")]
    if vs:
        return Counter(vs).most_common(1)[0][0]
    vs = [v for v in vals if isinstance(v, str) and v]
    return vs[0] if vs else None


def scene_labels(own: pd.DataFrame, scenes: pd.DataFrame, policy="rep") -> pd.DataFrame:
    """own: per-FILE labels (index = file id; any columns, h/v pairs in the file's frame).
    -> per-SCENE labels (index = scene_id = rep id) in the rep's frame + label_src_id / n_labelled_files."""
    own = own[own.index.isin(scenes.index)]
    if own.empty:
        return pd.DataFrame(columns=list(own.columns) + ["label_src_id", "label_src_kind", "n_labelled_files"])
    sc = scenes.loc[own.index]
    rep_frame = swap_hv(own, sc.swap_hv.to_numpy())             # every file label rotated into its rep's frame
    rep_frame["_scene"] = sc.scene_id.to_numpy()
    # the rep itself always comes first: some copies have a slightly LARGER tool crop than their rep
    # (90 vs 86, 140 vs 95, 145 vs 47, 146 vs 45: same exposure, separately drawn window polygon), and
    # without the leading rep term an "Edit anyway" label on such a copy would beat the rep's label
    rep_frame["_key"] = [(not bool(sc.at[i, "is_scene_rep"]), sc.at[i, "kind"] == "notebook_synthetic",
                          -sc.at[i, "rect_size"], KIND_ORDER.get(sc.at[i, "kind"], 9), i) for i in rep_frame.index]
    out = []
    for s, g in rep_frame.groupby("_scene"):
        g = g.sort_values("_key")
        if policy == "pool" and len(g) > 1:
            row = {}
            for c in g.columns:
                if c.startswith("_"):
                    continue
                col = g[c]
                if pd.api.types.is_bool_dtype(col):
                    row[c] = bool(col.any())
                elif pd.api.types.is_numeric_dtype(col):
                    row[c] = float(col.mean()) if col.notna().any() else np.nan
                elif c in ("weave", "warp_dir"):
                    row[c] = _majority(list(col))
                else:
                    row[c] = col.iloc[0]
            row["label_src_id"] = int(g.index[0])
        else:
            row = {c: g.iloc[0][c] for c in g.columns if not c.startswith("_")}
            row["label_src_id"] = int(g.index[0])
        row["label_src_kind"] = scenes.at[row["label_src_id"], "kind"]
        row["n_labelled_files"] = int(len(g))
        row["scene_id"] = int(s)
        out.append(row)
    return pd.DataFrame(out).set_index("scene_id").sort_index()


def file_labels(scene_lab: pd.DataFrame, scenes: pd.DataFrame, own: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-scene labels (rep frame) -> one row per member file, rotated into the file's frame.
    Unlabelled scenes give no rows. own (optional): per-file own labels, kept as own_h_tpi / own_v_tpi."""
    sc = scenes[scenes.scene_id.isin(scene_lab.index)]
    lab = scene_lab.reindex(sc.scene_id.to_numpy())
    lab.index = sc.index
    lab = swap_hv(lab, sc.swap_hv.to_numpy())
    meta = sc[["scene_id", "is_scene_rep", "kind", "scene_rot_k", "swap_hv", "copy_of", "rect_size",
               "scene_resolution", "is_test_item", "iaa"]]
    out = meta.join(lab.drop(columns=[c for c in meta.columns if c in lab.columns]))
    if own is not None:
        hv = [("h_tpi", "v_tpi")] if "h_tpi" in own else [("h", "v")]
        a, b = hv[0]
        out["has_own_label"] = out.index.isin(own.index)
        out["own_h_tpi"] = own[a].reindex(out.index)
        out["own_v_tpi"] = own[b].reindex(out.index)
    return out.sort_index()


def propagate(own: pd.DataFrame, scenes: pd.DataFrame | None = None, policy="rep"):
    """own (per-file labels, index = id) -> (per-file labels for every member of a labelled scene,
    per-scene labels in the rep frame)."""
    scenes = load_scenes() if scenes is None else scenes
    sl = scene_labels(own, scenes, policy)
    return file_labels(sl, scenes, own), sl


# ---------------------------------------------------------------------------- readers (tool exports)
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
    """Exact port of core.js lineInfo(rec, line).tpi (as aug1016/propagate_labels.py)."""
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


def _row(id_, ann, done, has_done, h, v, hc, vc, weave, warp, nonwoven, conf, ha, va, ts, path, proposals_seen=None):
    return dict(id=int(id_), annotator=ann or "", done=bool(done), has_done=has_done, h_tpi=h, v_tpi=v, h_count=hc,
                v_count=vc, weave=weave or None, warp_dir=warp or None, nonwoven=bool(nonwoven), confidence=conf,
                uncertain=conf == 1, h_angle_deg=ha, v_angle_deg=va, timestamp=ts or "", file=path,
                proposals_seen=proposals_seen)


def read_json(path):
    obj = json.load(open(path))
    recs = obj.get("records", {}) if isinstance(obj, dict) else obj
    recs = list(recs.values()) if isinstance(recs, dict) else recs
    ann = obj.get("annotator", "") if isinstance(obj, dict) else ""
    out = []
    for r in recs or []:
        if not isinstance(r, dict) or r.get("id") is None or not isinstance(r.get("h_dots"), list) \
                or not isinstance(r.get("v_dots"), list):
            continue
        h = line_tpi(r, "h") if r.get("rect_size") else float(len(r["h_dots"]))
        v = line_tpi(r, "v") if r.get("rect_size") else float(len(r["v_dots"]))
        out.append(_row(r["id"], r.get("annotator") or ann, r.get("done"), True, h, v, len(r["h_dots"]),
                        len(r["v_dots"]), r.get("weave"), r.get("warp_dir"), r.get("nonwoven"), _num(r.get("confidence")),
                        line_angle(r, "h"), line_angle(r, "v"), r.get("timestamp"), path, bool(r.get("proposals_seen"))))
    return out


def read_csv(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "id" not in df and "image" in df:
        df["id"] = df["image"]
    out = []
    for r in df.to_dict("records"):
        s = str(r.get("id", "")).strip()
        base = os.path.splitext(os.path.basename(s))[0]
        if not base.replace(".0", "").isdigit():
            continue
        h, v = _num(r.get("h_tpi")), _num(r.get("v_tpi"))
        h = h if h is not None else _num(r.get("h_count"))
        v = v if v is not None else _num(r.get("v_count"))
        out.append(_row(int(float(base)), r.get("annotator", ""), _truthy(r.get("done", "")), "done" in r, h, v,
                        _num(r.get("h_count")), _num(r.get("v_count")), r.get("weave"), r.get("warp_dir"),
                        _truthy(r.get("nonwoven", "")), _num(r.get("confidence")), _num(r.get("h_angle_deg")) or 0.0,
                        _num(r.get("v_angle_deg")) or 0.0, r.get("timestamp", ""), path,
                        _truthy(r.get("proposals_seen", "")) if "proposals_seen" in r else None))
    return out


def default_inputs(saved=SAVED):
    return sorted(p for pat in ("*.json", "*.csv") for p in glob.glob(os.path.join(saved, pat)))


def read_own_labels(paths, include_not_done=False, log=print) -> pd.DataFrame:
    """Per-file own labels (index = id): newest record per (annotator, id); done (unless
    include_not_done), not nonwoven, h and v > 0; several annotators -> per-axis mean, majority weave."""
    rows = []
    for p in paths:
        try:
            rows += read_json(p) if p.lower().endswith(".json") else read_csv(p)
        except Exception as e:  # noqa: BLE001 -- a half-written autosave must not stop the pipeline
            log(f"  skipped {p}: {e}")
    raw = pd.DataFrame(rows)
    cols = ["h_tpi", "v_tpi", "h_count", "v_count", "weave", "warp_dir", "h_angle_deg", "v_angle_deg", "uncertain",
            "confidence", "n_annotations", "annotators"]
    if raw.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="id"))
    raw = raw.sort_values("timestamp").drop_duplicates(["annotator", "id"], keep="last")
    keep = raw[~raw.nonwoven]
    if not include_not_done:
        keep = keep[~keep.has_done | keep.done]
    keep = keep[keep.h_tpi.notna() & keep.v_tpi.notna()]
    keep = keep[(keep.h_tpi > 0) & (keep.v_tpi > 0)]
    out = []
    for i, g in keep.groupby("id"):
        wd = [w for w in g.warp_dir if w in ("vertical", "horizontal")]
        out.append(dict(id=int(i), h_tpi=float(g.h_tpi.mean()), v_tpi=float(g.v_tpi.mean()),
                        h_count=float(pd.to_numeric(g.h_count).mean()), v_count=float(pd.to_numeric(g.v_count).mean()),
                        weave=_majority(list(g.weave)),
                        warp_dir=(wd[0] if len(set(wd)) == 1 else "unknown") if wd else (g.warp_dir.iloc[0] or None),
                        h_angle_deg=float(g.h_angle_deg.mean()), v_angle_deg=float(g.v_angle_deg.mean()),
                        uncertain=bool(g.uncertain.any()), confidence=g.confidence.min(),
                        n_annotations=int(len(g)), annotators=";".join(sorted(set(map(str, g.annotator))))))
    return pd.DataFrame(out, columns=["id"] + cols).set_index("id")


def run(paths, out_dir=OUT_DIR, policy="rep", include_not_done=False, manifest=MANIFEST, log=print):
    scenes = load_scenes(manifest)
    own = read_own_labels(paths, include_not_done, log)
    per_file, per_scene = propagate(own, scenes, policy)
    os.makedirs(out_dir, exist_ok=True)
    full = scenes[["scene_id", "is_scene_rep", "kind", "scene_rot_k", "swap_hv", "is_test_item", "iaa"]]
    pf = full.join(per_file.drop(columns=[c for c in full.columns if c in per_file.columns]), how="left")
    pf["labelled"] = pf.index.isin(per_file.index)
    reps = scenes[scenes.is_scene_rep][["scene_members", "scene_resolution", "iaa", "rect_size"]]
    reps.index.name = "scene_id"
    ps = reps.join(per_scene, how="left")
    ps["labelled"] = ps.index.isin(per_scene.index)
    for df, name in ((pf, "labels_per_file.csv"), (ps, "labels_per_scene.csv")):
        tmp = os.path.join(out_dir, name + ".tmp")
        df.to_csv(tmp)
        os.replace(tmp, os.path.join(out_dir, name))
    meta = dict(written=time.strftime("%Y-%m-%dT%H:%M:%S"), inputs=paths, policy=policy,
                include_not_done=include_not_done, n_files_with_own_label=int(len(own)),
                n_scenes_labelled=int(len(per_scene)), n_files_labelled=int(len(per_file)),
                n_scenes=int(scenes.is_scene_rep.sum()), n_files=int(len(scenes)),
                n_swapped_files_labelled=int(per_file.swap_hv.sum()) if len(per_file) else 0,
                labels_from_non_rep=sorted(int(i) for i in per_scene.index[per_scene.label_src_id != per_scene.index])
                if len(per_scene) else [])
    with open(os.path.join(out_dir, "labels_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"{len(paths)} input file(s): {len(own)} files with an own label -> {len(per_scene)}/{meta['n_scenes']} scenes "
        f"labelled -> {len(per_file)}/{meta['n_files']} files labelled ({meta['n_swapped_files_labelled']} with h/v "
        f"swapped) -> {out_dir}")
    return pf, ps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="annotation exports (JSON/CSV); default: data/saved/*.json|csv")
    ap.add_argument("--policy", choices=("rep", "pool"), default="rep")
    ap.add_argument("--include-not-done", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()
    run(a.inputs or default_inputs(), a.out_dir, a.policy, a.include_not_done, a.manifest)


if __name__ == "__main__":
    main()
