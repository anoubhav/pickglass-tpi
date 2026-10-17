"""build_stress.py -- PickGlass-Stress: deterministic phone-camera renders of the pick-glass crops.

Every source crop is rendered through each camera profile of camera_profiles.json with camera_sim.py
(camera-only: no LED torch), plus three harsh capture/transport variants and the degrade.py glare /
defocus families at 2 severities on top of the budget phone.  The benchmark file of a render is the
crop-store JPEG (q92, 4:2:0) that annotation_tool/prepare.py would have written for that capture.

SETS
  legacy13  vlm/crops/legacy/{name}.png  (canonical crops every method saw; 13 hand-counted images)
  real150   annotation_tool/data/crops/{id}.jpg  (labels arrive from the human annotation by id)

CONDITIONS (name: profile, overlay, transport, scene degradation)            reps legacy13 / real150
  noop_store            no camera; crop-store re-encode only (JPEG q92)          1 / 1
  noop_q75              libjpeg q75 4:2:0 resave + crop store (= our originals)   1 / 1
  budget_binned_12mp    budget_binned_12mp                                       3 / 1
  budget_fullres_50mp   budget_fullres_50mp                                      3 / 1
  midrange_binned       midrange_binned                                          3 / 1
  flagship              flagship                                                 3 / 1
  legacy_16mp_2019      legacy_16mp_2019                                         3 / 1
  budget_lowlight       budget_binned_12mp + low_light overlay                   3 / 1
  budget_whatsapp       budget_binned_12mp + messaging_standard transport        3 / 1
  legacy_quarter        legacy_16mp_2019 + resave_q75_x0.25 (our real quarter-res copies)  3 / 1
  budget+glare0.05 / budget+glare0.3 / budget+gblur3 / budget+gblur6
                        degrade.apply(scene, kind, level, seed=crc32(id) % 100003) then the budget
                        camera of rep 0 (same severities and seed rule as paper Table 6)   1 / 1
Seeds: seed = stable_seed('PickGlass-Stress-v1', set, id, profile, rep) -- the profile, not the
condition, enters the seed, so budget / budget_lowlight / budget_whatsapp / budget+glare share the
same camera draw (overlay / transport / degradation are isolated); likewise legacy_16mp_2019 and
legacy_quarter.

LABELS: identity for threads per inch (the crop is always the same 1-inch window) times the exact
radial-distortion factor of the render (camera_sim docstring):  h_label = h_src x label_factor_h,
v_label = v_src x label_factor_v (area-mean; *_centre columns give the guide-line factors).
legacy13 labels are written now (legacy harness parser, axis convention h <-> warp, v <-> weft);
real150 labels with --labels (annotation-tool CSV exports, read with paper/tools/score150.py's
load_labels / gt_table: mean over annotators, h_tpi / v_tpi).

OUTPUT  camera_aug/stress/{set}/{condition}/{id}__r{rep}.jpg
        camera_aug/stress/{set}/manifest.csv   (one row per render: seed, every sampled parameter
                                                 as JSON, sizes, label factors, md5)
        camera_aug/stress/{set}/labels.csv      (legacy13 now; real150 after --labels)

    <venv2>/bin/python build_stress.py --set legacy13 --workers 4
    <venv2>/bin/python build_stress.py --set real150 --workers 4
    <venv2>/bin/python build_stress.py --set real150 --labels ../annotation_tool/data/saved/A.csv,B.csv
    <venv2>/bin/python build_stress.py --set legacy13 --verify [--limit 20]   # re-render, compare md5
Deterministic: same OpenCV / NumPy / libjpeg build -> byte-identical files (md5 in the manifest).
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import camera_sim as CS  # noqa: E402

BENCH = "PickGlass-Stress-v1"
OUT = os.path.join(HERE, "stress")
LEGACY_DIR = os.path.join(ROOT, "vlm", "crops", "legacy")
CROPS_150 = os.path.join(ROOT, "annotation_tool", "data", "crops")
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]

# name, profile, overlay, transport, scene degradation (kind, level), reps legacy13, reps real150
CONDITIONS = [
    ("noop_store", None, None, None, None, 1, 1),
    ("noop_q75", None, None, None, None, 1, 1),
    ("budget_binned_12mp", "budget_binned_12mp", None, "none", None, 3, 1),
    ("budget_fullres_50mp", "budget_fullres_50mp", None, "none", None, 3, 1),
    ("midrange_binned", "midrange_binned", None, "none", None, 3, 1),
    ("flagship", "flagship", None, "none", None, 3, 1),
    ("legacy_16mp_2019", "legacy_16mp_2019", None, "none", None, 3, 1),
    ("budget_lowlight", "budget_binned_12mp", "low_light", "none", None, 3, 1),
    ("budget_whatsapp", "budget_binned_12mp", None, "messaging_standard", None, 3, 1),
    ("legacy_quarter", "legacy_16mp_2019", None, "resave_q75_x0.25", None, 3, 1),
    ("budget+glare0.05", "budget_binned_12mp", None, "none", ("glare", 0.05), 1, 1),
    ("budget+glare0.3", "budget_binned_12mp", None, "none", ("glare", 0.30), 1, 1),
    ("budget+gblur3", "budget_binned_12mp", None, "none", ("gblur", 3), 1, 1),
    ("budget+gblur6", "budget_binned_12mp", None, "none", ("gblur", 6), 1, 1),
]
COND = {c[0]: c for c in CONDITIONS}
CAMERA_ONLY = ["budget_binned_12mp", "budget_fullres_50mp", "midrange_binned", "flagship", "legacy_16mp_2019"]


def degrade_seed(src_id):
    return zlib.crc32(str(src_id).encode()) % 100003       # == methods/degrade_vlmcrops.seed_for


def sources(set_name):
    if set_name == "legacy13":
        return [(n, os.path.join(LEGACY_DIR, f"{n}.png")) for n in NAMES13]
    if set_name == "real150":
        return [(str(i), os.path.join(CROPS_150, f"{i}.jpg")) for i in range(150)]
    raise ValueError(set_name)


def render_seed(set_name, src_id, profile, rep):
    return CS.stable_seed(BENCH, set_name, src_id, profile, rep)


def jobs_for(set_name, conds=None):
    out = []
    for sid, path in sources(set_name):
        for c in CONDITIONS:
            if conds and c[0] not in conds:
                continue
            reps = c[5] if set_name == "legacy13" else c[6]
            for rep in range(reps):
                out.append((set_name, sid, path, c[0], rep))
    return out


def rel(p):
    return os.path.relpath(p, HERE)


def render_one(job, write=True):
    set_name, sid, path, cname, rep = job
    _, profile, overlay, transport, sdeg, _, _ = COND[cname]
    src_bytes = open(path, "rb").read()
    src = cv2.imdecode(np.frombuffer(src_bytes, np.uint8), cv2.IMREAD_COLOR)
    t0 = time.time()
    row = dict(benchmark=BENCH, set=set_name, source_id=sid, source_path=os.path.relpath(path, ROOT),
               source_md5=hashlib.md5(src_bytes).hexdigest(), S_src=int(src.shape[0]), condition=cname, rep=rep,
               profile=profile, overlay=overlay, transport=transport,
               scene_degrade=None if sdeg is None else f"{sdeg[0]}{sdeg[1]:g}", seed=None)
    if profile is None:
        out, jb = CS.noop_store(src, q75_first=(cname == "noop_q75"))
        P = {}
        info = dict(S_out=int(out.shape[0]), ppi_out=float(out.shape[0]), label_factor_h=1.0, label_factor_v=1.0,
                    label_factor_h_centre=1.0, label_factor_v_centre=1.0, label_bound_ca=0.0)
    else:
        seed = render_seed(set_name, sid, profile, rep)
        row["seed"] = seed
        P = CS.sample_params(profile, seed, overlay=overlay, transport=transport)
        dg = None if sdeg is None else (sdeg[0], sdeg[1], degrade_seed(sid))
        out, info, jb = CS.render(src, P, scene_degrade=dg, return_bytes=True)
    fn = os.path.join(OUT, set_name, cname, f"{sid}__r{rep}.jpg")
    if write:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        tmp = fn + ".tmp"
        with open(tmp, "wb") as f:
            f.write(jb)
        os.replace(tmp, fn)
    keep = ("S_out", "ppi_out", "ppi_profile", "ppi_sensor", "ppi_capped", "scene_upsampled", "z_eff", "tf_eff",
            "label_factor_h",
            "label_factor_v", "label_factor_h_centre", "label_factor_v_centre", "label_bound_ca",
            "sigma_noise_dn_pred", "clipped_frac", "pitch_scale")
    row.update({k: info.get(k) for k in keep})
    row.update(path=rel(fn), md5=hashlib.md5(jb).hexdigest(), bytes=len(jb), render_s=round(time.time() - t0, 2),
               sim_version=CS.SIM_VERSION, params_json=json.dumps(P, sort_keys=True))
    # a few headline parameters as columns (all are in params_json)
    for k in ("iso", "frames", "demosaic", "cfa", "jpeg_q", "zoom", "k1", "motion_u", "theta_deg",
              "transport_q", "transport_factor", "transport_filter"):
        row["p_" + k] = P.get(k)
    return row


def _init():
    cv2.setNumThreads(1)


def build(set_name, workers=4, conds=None, limit=0):
    jobs = jobs_for(set_name, conds)
    if limit:
        jobs = jobs[:limit]
    man_path = os.path.join(OUT, set_name, "manifest.csv")
    old = pd.read_csv(man_path, dtype={"source_id": str}) if os.path.exists(man_path) else None
    if old is not None:   # resume: skip renders already in the manifest whose file is present with that md5
        done = {(r.source_id, r.condition, int(r.rep)) for r in old.itertuples()
                if os.path.exists(os.path.join(HERE, r.path))}
        jobs = [j for j in jobs if (j[1], j[3], j[4]) not in done]
    print(f"{set_name}: {len(jobs)} renders to do (workers={workers})", flush=True)
    rows, t0 = [], time.time()
    os.makedirs(os.path.join(OUT, set_name), exist_ok=True)
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(render_one, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
                _save(old, rows, man_path)
    _save(old, rows, man_path)
    if set_name == "legacy13":
        write_labels_legacy13()


def _save(old, rows, man_path):
    d = pd.DataFrame(rows)
    if old is not None:
        d = pd.concat([old, d], ignore_index=True)
    d = d.drop_duplicates(["source_id", "condition", "rep"], keep="last")
    order = {c[0]: k for k, c in enumerate(CONDITIONS)}
    d["_o"] = d.condition.map(order)
    d["_s"] = d.source_id.astype(str)
    d = d.sort_values(["_s", "_o", "rep"]).drop(columns=["_o", "_s"])
    tmp = man_path + ".tmp"
    d.to_csv(tmp, index=False)
    os.replace(tmp, man_path)


# =============================================================================================
# labels
# =============================================================================================
def legacy_gt():
    spec = importlib.util.spec_from_file_location("legacy_unified", os.path.join(ROOT, "paper", "tools",
                                                                                 "legacy_unified.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_gt()          # index image; warp, weft, weave, unclear, is_range, *_formula


def _propagate(man, gt, hcol, vcol):
    m = man.merge(gt, left_on="source_id", right_index=True, how="left")
    m["h_label"] = m[hcol] * m.label_factor_h
    m["v_label"] = m[vcol] * m.label_factor_v
    m["h_label_centre"] = m[hcol] * m.label_factor_h_centre
    m["v_label_centre"] = m[vcol] * m.label_factor_v_centre
    m["ppt_out_fine"] = m.S_out / m[[hcol, vcol]].max(axis=1)        # px per yarn on the finer axis
    return m


def write_labels_legacy13():
    man = pd.read_csv(os.path.join(OUT, "legacy13", "manifest.csv"), dtype={"source_id": str})
    gt = legacy_gt()
    gt.index = gt.index.astype(str)
    m = _propagate(man, gt[["warp", "weft", "weave", "unclear", "is_range", "warp_formula", "weft_formula"]],
                   "warp", "weft")
    m = m.rename(columns={"warp": "h_src", "weft": "v_src"})
    cols = ["source_id", "condition", "rep", "path", "md5", "S_out", "h_src", "v_src", "label_factor_h",
            "label_factor_v", "h_label", "v_label", "h_label_centre", "v_label_centre", "ppt_out_fine", "weave",
            "unclear", "is_range", "warp_formula", "weft_formula"]
    m[cols].to_csv(os.path.join(OUT, "legacy13", "labels.csv"), index=False)
    print("wrote", os.path.join(OUT, "legacy13", "labels.csv"))


def write_labels_real150(label_paths):
    spec = importlib.util.spec_from_file_location("score150", os.path.join(ROOT, "paper", "tools", "score150.py"))
    s150 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s150)
    rec = s150.load_labels(label_paths)
    gt = s150.gt_table(rec)
    gt.index = gt.index.astype(int).astype(str)
    man = pd.read_csv(os.path.join(OUT, "real150", "manifest.csv"), dtype={"source_id": str})
    m = _propagate(man, gt[["h", "v", "n_ann", "weave"]], "h", "v").rename(columns={"h": "h_src", "v": "v_src"})
    cols = ["source_id", "condition", "rep", "path", "md5", "S_out", "h_src", "v_src", "n_ann", "label_factor_h",
            "label_factor_v", "h_label", "v_label", "h_label_centre", "v_label_centre", "ppt_out_fine", "weave"]
    m[cols].to_csv(os.path.join(OUT, "real150", "labels.csv"), index=False)
    print(f"wrote {OUT}/real150/labels.csv: {m.h_src.notna().sum()} of {len(m)} renders labelled")


def verify(set_name, limit=0, workers=4):
    man = pd.read_csv(os.path.join(OUT, set_name, "manifest.csv"), dtype={"source_id": str})
    src = dict(sources(set_name))
    jobs = [(set_name, r.source_id, src[r.source_id], r.condition, int(r.rep)) for r in man.itertuples()]
    if limit:
        idx = np.random.default_rng(0).choice(len(jobs), min(limit, len(jobs)), replace=False)
        jobs = [jobs[i] for i in sorted(idx)]
    from functools import partial
    with Pool(workers, initializer=_init) as pool:
        rows = pool.map(partial(render_one, write=False), jobs)
    ref = {(r.source_id, r.condition, int(r.rep)): r.md5 for r in man.itertuples()}
    same = sum(ref[(r["source_id"], r["condition"], r["rep"])] == r["md5"] for r in rows)
    disk = sum(hashlib.md5(open(os.path.join(HERE, r["path"]), "rb").read()).hexdigest() == r["md5"] for r in rows)
    print(f"verify {set_name}: re-rendered {len(rows)}; md5 == manifest: {same}/{len(rows)}; "
          f"md5 == file on disk: {disk}/{len(rows)}")
    return same == len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["legacy13", "real150"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conds", default="", help="comma list of condition names (default: all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels", default="", help="real150: annotation CSV export(s), comma-separated")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.set, a.limit, a.workers) else 1)
    if a.labels:
        write_labels_real150([p for p in a.labels.split(",") if p])
        sys.exit(0)
    build(a.set, a.workers, [c for c in a.conds.split(",") if c] or None, a.limit)
