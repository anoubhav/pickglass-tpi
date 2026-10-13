#!/usr/bin/env python
"""augment_optics_verify.py -- label-safety audit of augment_optics.py + 'minimum phone resolution'.

Protocol = verify_augment.py's (the label-safety audit of augment.py), applied to the new transforms:
for every image the frozen classical counter ("ours", methods/results/counting_frozen_final.py with the
release conf_model.json) is run on the clean crop and on each transform x strength.  Per image and
axis:  drift = observed per-inch count / (predicted label factor x clean per-inch count) - 1, where the
predicted factor is the CROSSING factor of the transform's OLT evaluated with ours' clean-run yarn
tilt (ours measures guide-line crossing counts); a failed run counts as 100 % drift.  Only values
whose expected per-crop count lies in ours' search band [1.1 x 18, 0.40 x N_work] (clean and
transformed) enter the PASS statistics (the phone-resolution CURVE below uses every value).

Noise floor and PASS rule (unchanged from verify_augment.py, fixed before any run):
  floor = pooled drift of the null re-encodings (JPEG q95 re-encode; 0.5-px sub-pixel shift),
  same set.  within-noise iff  median |drift| <= floor + 0.5 pp,  p90 <= floor + 2 pp,
  P(|drift| > 10 %) <= floor + 2 pp,  |median signed drift| <= 1 %.
  Size-changing transforms with a size-matched floor (transform + JPEG q95 vs transform) use
  max(floor, size-matched floor).  label_preserving (whitelist) = within-noise on real150.
  legacy13 (the held-out hand-counted test set) is REPORTED, never used for a decision.
  Extra for label-changing transforms: 'drift if label ignored' (= obs/clean - 1), i.e. evidence
  that the counter really sees the predicted change, and a pooled tracking regression of the
  measured log change on the predicted log change.

Minimum phone resolution: phone_capture / area_downsample at target px per yarn (finer axis); error
of every axis value vs ours' clean-crop prediction (real150, split high-res / low-res) and vs the
human counts (legacy13; target size set from the GT counts there).  Also bicubic x2 up-sampling of
the phone image before counting (does the failure come from ours' 0.42 cycles/px search band?).

    python augment_optics_verify.py prep                 # (Mac) writes aug_optics/legacy13_gt.json
    python augment_optics_verify.py run --workers 20     # ours on real150 + legacy13 (resumable)
    python augment_optics_verify.py summarise            # verification CSV, whitelist, curve, figures
    python augment_optics_verify.py examples             # example figure (one panel per transform)
Paths can be overridden with AOV_CROPS, AOV_LEGACY, AOV_OURS, AOV_CONF, AOV_OUT (server runs).
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import zlib

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import augment_optics as AO  # noqa: E402

CROPS = os.environ.get("AOV_CROPS", os.path.join(ROOT, "annotation_tool", "data", "crops"))
LEGACY = os.environ.get("AOV_LEGACY", os.path.join(ROOT, "vlm", "crops", "legacy"))
OURS_FILE = os.environ.get("AOV_OURS", os.path.join(ROOT, "methods", "results", "counting_frozen_final.py"))
CONF_FILE = os.environ.get("AOV_CONF", os.path.join(ROOT, "methods", "conf_model.json"))
OUT = os.environ.get("AOV_OUT", os.path.join(HERE, "aug_optics"))
SYNTH = os.environ.get("AOV_SYNTH", os.path.join(HERE, "data", "synth_val"))
LIMIT = int(os.environ.get("AOV_LIMIT", "0"))
LEGACY13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
R_MIN = 18.0
BAND_LO, BAND_HI = 1.1 * R_MIN, 0.40
PPTS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
RAW_COLS = ["set", "image", "cond", "model", "side", "out_w", "out_h", "ext_x", "ext_y", "pred_h", "pred_v",
            "pred_weave", "conf", "angle_x", "angle_y", "n_work", "runtime_s", "status", "fc_x", "fc_y",
            "f0_x", "f0_y", "fd_x", "fd_y", "spread_x_lo", "spread_x_hi", "spread_y_lo", "spread_y_hi",
            "area_scale_x", "area_scale_y", "ppt_target", "count_used", "side_target", "clipped_frac",
            "sigma_max_px", "label_bound", "applicable", "fh1_x", "fh1_y", "fh2_x", "fh2_y",
            "lab_cross_h", "lab_cross_v", "lab_cross0_h", "lab_cross0_v"]


# =============================================================================================
# data
# =============================================================================================
def cap_side(img, max_side=1024):
    """== cnn_common.cap_side / verify_augment.cap_side."""
    if img.shape[0] > max_side or img.shape[1] > max_side:
        return cv2.resize(img, (max_side, max_side), interpolation=cv2.INTER_AREA)
    if img.shape[0] != img.shape[1]:
        s = max(img.shape[:2])
        return cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
    return img


def load_img(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cap_side(im, 1024)


def load_sets(names=("real150", "legacy13")):
    out = {}
    if "synth100" in names:   # exact labels (intrinsic epi / ppi + exact tilt), as in verify_augment.py
        df = pd.read_csv(os.path.join(SYNTH, "labels.csv")).iloc[:100]
        out["synth100"] = [dict(image=os.path.splitext(r.file)[0], path=os.path.join(SYNTH, r.file),
                                epi=float(r.h_count), ppi=float(r.v_count), rot=float(r.rotation_deg),
                                skew=float(r.skew_deg)) for r in df.itertuples()]
    if "legacy13" in names:
        out["legacy13"] = [dict(image=n, path=os.path.join(LEGACY, f"{n}.png")) for n in LEGACY13]
    if "real150" in names:
        out["real150"] = [dict(image=str(i), path=os.path.join(CROPS, f"{i}.jpg")) for i in range(150)]
    if LIMIT:
        out = {k: v[:LIMIT] for k, v in out.items()}
    return out


def legacy_gt():
    p = os.path.join(OUT, "legacy13_gt.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# =============================================================================================
# conditions
# =============================================================================================
def _c(cond, family, fn, level, kind="candidate", **params):
    return dict(cond=cond, family=family, fn=fn, strength=level, kind=kind, params=params)


def conditions():
    C = [_c("clean", "clean", "identity", 0, "clean")]
    C += [_c("null_jpeg95", "null", "jpeg", 95, "null"), _c("null_subpix", "null", "subpix", 0.5, "null")]
    C += [_c(f"vig_{s:g}", "1_vignette", "vignette_cos4", s, strength=s) for s in (0.2, 0.4, 0.6)]
    C += [_c(f"dist_{k:+g}", "1_distortion", "radial_distortion", k, k1=k)
          for k in (0.005, 0.01, 0.02, 0.05, 0.1, -0.005, -0.01, -0.02, -0.05, -0.1)]
    C += [_c(f"ca_{e:g}", "1_chromatic", "chromatic_aberration", e, eps=e) for e in (0.002, 0.005, 0.01, 0.02)]
    C += [_c(f"dof_{s:g}pitch", "1_dof", "dof_gradient", s, sigma_pitch=s) for s in (0.1, 0.2, 0.3, 0.45)]
    C += [_c(f"led_{a:g}", "1_led", "led_hotspot", a, amp=a) for a in (0.5, 1.0, 2.0, 4.0)]
    C += [_c(f"phone_ppt{p:g}", "2_phone", "phone_capture", p, ppt=p) for p in PPTS]
    C += [_c(f"area_ppt{p:g}", "2_area", "area_downsample", p, ppt=p) for p in PPTS]
    C += [_c(f"phoneup2_ppt{p:g}", "2_phone_up2", "phone_up2", p, ppt=p) for p in (2.0, 2.5, 3.0)]
    C += [_c(f"zoom_{s:g}", "3_zoom", "zoom", s, s=s) for s in (1.1, 1.25, 1.5, 2.0, 0.9, 0.8)]
    C += [_c(f"strx_{s:g}", "3_stretch", "stretch", s, sx=s, sy=1.0) for s in (1.15, 1.3, 1.6, 0.85)]
    C += [_c("stry_1.3", "3_stretch", "stretch", 1.3, sx=1.0, sy=1.3)]
    C += [_c(f"sheary_{g:g}", "3_shear", "shear", g, gamma=g, axis="y", mode="inner") for g in (0.05, 0.1, 0.18, 0.27)]
    C += [_c("shearx_0.1", "3_shear", "shear", 0.1, gamma=0.1, axis="x", mode="inner"),
          _c("sheary_0.1_frame", "3_shear", "shear", 0.1, gamma=0.1, axis="y", mode="frame")]
    C += [_c(f"bowy_{d:g}", "3_bow", "bow", d, depth=d, axis="y") for d in (0.01, 0.02, 0.04)]
    C += [_c(f"{b}+jpeg95", "z_sizefloor", "then_jpeg95", 0, "sizefloor", base=b) for b in SIZEFLOOR_BASES]
    return C


SIZEFLOOR_BASES = ("zoom_1.25", "zoom_1.5", "zoom_2", "strx_1.6", "sheary_0.18", "phone_ppt3", "phone_ppt4")
COND = None


def cond_map():
    global COND
    if COND is None:
        COND = {c["cond"]: c for c in conditions()}
    return COND


def apply_cond(c, img, seed_key, ctx):
    """-> (image, OLT, extras) or None if not applicable.  ctx: pitch_px, count (finer-axis per-crop
    count used for the phone targets)."""
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    fn, p = c["fn"], c["params"]
    ex = {}
    if fn == "identity":
        return img, AO.identity_lt(img), ex
    if fn == "jpeg":
        return AO._jpeg(img, p.get("q", c["strength"])), AO.identity_lt(img, name="jpeg95"), ex
    if fn == "subpix":
        H, W = img.shape[:2]
        M = np.array([[1, 0, 0.5], [0, 1, 0.5]], float)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101), \
            AO.identity_lt(img, name="subpix"), ex
    if fn == "then_jpeg95":
        r = apply_cond(cond_map()[p["base"]], img, seed_key.replace(c["cond"], p["base"]), ctx)
        if r is None:
            return None
        t, lt, ex = r
        return AO._jpeg(t, 95), lt, ex
    if fn in ("phone_capture", "area_downsample", "phone_up2"):
        cnt = ctx["count"]
        S = min(img.shape[:2])
        if not (np.isfinite(cnt) and cnt > 0):
            return None
        side = int(round(p["ppt"] * cnt))
        ex.update(ppt_target=p["ppt"], count_used=cnt, side_target=side)
        if side > 0.9 * S:          # the crop does not have that many px per yarn
            return None
        if fn == "area_downsample":
            t, lt = AO.area_downsample(img, rng=rng, ppt=p["ppt"], count=cnt)
        else:
            t, lt = AO.phone_capture(img, rng=rng, ppt=p["ppt"], count=cnt)
            if fn == "phone_up2":
                t2, l2 = AO.upsample(t, factor=2.0)
                t, lt = t2, AO.compose(lt, l2)
        return t, lt, ex
    if fn == "dof_gradient":
        t, lt = AO.dof_gradient(img, rng=rng, pitch_px=ctx["pitch_px"], sigma_pitch=p["sigma_pitch"])
        ex["sigma_max_px"] = lt.params["sigma_max_px"]
        return t, lt, ex
    if fn == "radial_distortion":
        t, lt = AO.radial_distortion(img, rng=rng, k1=p["k1"], jitter=0.1)
        ex["area_scale_x"], ex["area_scale_y"] = lt.params["area_mean_scale"]
        return t, lt, ex
    if fn == "chromatic_aberration":
        t, lt = AO.chromatic_aberration(img, rng=rng, eps=p["eps"])
        ex["label_bound"] = lt.params["label_bound"]
        return t, lt, ex
    if fn == "led_hotspot":
        t, lt = AO.led_hotspot(img, rng=rng, amp=p["amp"], r0=0.08)
        ex["clipped_frac"] = lt.params["clipped_frac"]
        return t, lt, ex
    if fn == "vignette_cos4":
        t, lt = AO.vignette_cos4(img, rng=rng, strength=p["strength"])
        return t, lt, ex
    if fn in ("zoom", "stretch", "shear", "bow"):
        kw = dict(p)
        if fn in ("zoom", "stretch"):
            kw["center"] = (0.5, 0.5)
        t, lt = AO.TRANSFORMS[fn](img, rng=rng, **kw)
        return t, lt, ex
    raise ValueError(fn)


def lt_cols(lt, ang):
    hc, vc, _ = lt.crossing((1.0, 1.0), ang)
    h0, v0, _ = lt.crossing((1.0, 1.0), (0.0, 0.0))
    hd, vd, _ = lt.density((1.0, 1.0), ang, kind="crossing")
    sp = lt.row_spread(ang)
    w1, w2 = lt.weighted_scale("hann"), lt.weighted_scale("hann2")
    return dict(fh1_x=w1[0], fh1_y=w1[1], fh2_x=w2[0], fh2_y=w2[1],
                fc_x=hc, fc_y=vc, f0_x=h0, f0_y=v0, fd_x=hd, fd_y=vd, ext_x=lt.extent_out[0],
                ext_y=lt.extent_out[1], out_w=lt.out_size[0], out_h=lt.out_size[1], status=lt.status,
                spread_x_lo=sp[0], spread_x_hi=sp[1], spread_y_lo=sp[2], spread_y_hi=sp[3])


# =============================================================================================
# ours (CPU pool)
# =============================================================================================
_O = None


def _init_ours():
    global _O
    spec = importlib.util.spec_from_file_location("ours_frozen", OURS_FILE)
    _O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_O)
    _O.CONF_MODEL = json.load(open(CONF_FILE))


def _ours(img):
    t0 = time.time()
    try:
        e = _O.estimate(img, "ours")
    except Exception as err:  # a crash is a failed run (counts as 100 % drift)
        return dict(pred_h=np.nan, pred_v=np.nan, pred_weave=f"error:{type(err).__name__}", conf=np.nan,
                    angle_x=np.nan, angle_y=np.nan, n_work=np.nan, runtime_s=time.time() - t0)
    lat = e["debug"].get("lattice") or {}
    return dict(pred_h=e["h_count"], pred_v=e["v_count"], pred_weave=e["weave"], conf=e["confidence"],
                angle_x=lat.get("angle_x", np.nan), angle_y=lat.get("angle_y", np.nan),
                n_work=e["debug"].get("N_work", np.nan), runtime_s=time.time() - t0)


def ours_job(a):
    sname, item, clean, conds, gtmax = a
    lab = None
    if sname == "synth100":          # exact intrinsic labels + exact tilt (verify_augment convention)
        lab = ((item["epi"], item["ppi"]), (item["rot"], -(item["rot"] + item["skew"])))
        gtmax = max(item["epi"], item["ppi"])
    img = load_img(item["path"])
    S = img.shape[0]
    rows = []
    ang, pitch, count = (0.0, 0.0), S / 60.0, np.nan
    if clean is not None:
        ax, ay = clean["angle_x"], clean["angle_y"]
        ang = (0.0 if not np.isfinite(ax) else ax, 0.0 if not np.isfinite(ay) else ay)
        m = np.nanmax([clean["pred_h"], clean["pred_v"]])
        if np.isfinite(m) and m > 0:
            pitch, count = S / m, m
    if gtmax is not None:           # legacy13: phone targets from the human counts
        count = gtmax
    ctx = dict(pitch_px=pitch, count=count)
    for cn in conds:
        c = cond_map()[cn]
        r = dict(set=sname, image=item["image"], cond=cn, model="ours", side=S)
        res = apply_cond(c, img, f"{sname}|{item['image']}|{cn}", ctx)
        if res is None:
            r.update(applicable=False, ppt_target=c["params"].get("ppt"), count_used=count)
            rows.append(r)
            continue
        t, lt, ex = res
        r.update(_ours(t))
        r.update(lt_cols(lt, ang))
        r.update(ex)
        r["applicable"] = True
        if lab is not None:
            ch, cv, _ = lt.crossing(lab[0], lab[1], kind="density")
            c0 = AO.identity_lt(img).crossing(lab[0], lab[1], kind="density")
            r.update(lab_cross_h=ch * lt.extent_out[0], lab_cross_v=cv * lt.extent_out[1],   # per crop
                     lab_cross0_h=c0[0], lab_cross0_v=c0[1])
        rows.append(r)
    return rows


def run_ours(sets, workers, conds=None, tag=""):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    conds = conds or [c["cond"] for c in conditions() if c["cond"] != "clean"]
    S = load_sets(sets)
    gt = legacy_gt()
    t0 = time.time()
    clean_path = os.path.join(OUT, "clean_ours.csv")
    have = set()
    if os.path.exists(clean_path):
        d0 = pd.read_csv(clean_path, usecols=["set", "image"], dtype={"image": str})
        have = set(zip(d0["set"], d0["image"]))
    jobs = [(s, it, None, ["clean"], None) for s in S for it in S[s] if (s, it["image"]) not in have]
    if jobs:
        with Pool(workers, initializer=_init_ours) as pool:
            rows = [r for rr in pool.imap_unordered(ours_job, jobs, chunksize=1) for r in rr]
        cols = list(pd.read_csv(clean_path, nrows=0).columns) if have else RAW_COLS   # keep the file's header
        pd.DataFrame(rows).reindex(columns=cols).to_csv(clean_path, mode="a", header=not have, index=False)
        print(f"clean: {len(rows)} new rows, {time.time() - t0:.0f}s", flush=True)
    clean = pd.read_csv(clean_path, dtype={"image": str}).set_index(["set", "image"])
    raw = os.path.join(OUT, f"raw_ours{tag}.csv")
    done = set()
    if os.path.exists(raw):
        d = pd.read_csv(raw, usecols=["set", "image"], dtype={"image": str})
        done = set(zip(d["set"], d["image"]))
    jobs = []
    for s in S:
        for it in S[s]:
            if (s, it["image"]) in done:
                continue
            g = gt.get(it["image"]) if s == "legacy13" else None
            gm = float(max(g["warp"], g["weft"])) if g else None
            jobs.append((s, it, clean.loc[(s, it["image"])].to_dict(), conds, gm))
    print(f"ours: {len(jobs)} images x {len(conds)} conditions, {workers} workers", flush=True)
    header = not os.path.exists(raw)
    k = 0
    with Pool(workers, initializer=_init_ours) as pool:
        for rows in pool.imap_unordered(ours_job, jobs, chunksize=1):
            pd.DataFrame(rows).reindex(columns=RAW_COLS).to_csv(raw, mode="a", header=header, index=False)
            header = False
            k += 1
            print(f"  {k}/{len(jobs)} images  {time.time() - t0:.0f}s", flush=True)


# =============================================================================================
# statistics (== verify_augment.py)
# =============================================================================================
def stats(x):
    a = np.abs(x)
    return dict(med=100 * np.median(a), p90=100 * np.percentile(a, 90), gross10=100 * np.mean(a > 0.10),
                gt5=100 * np.mean(a > 0.05), bias=100 * np.median(np.log1p(x)))


def passes(s, f):
    return (s["med"] <= f["med"] + 0.5 and s["p90"] <= f["p90"] + 2.0 and s["gross10"] <= f["gross10"] + 2.0
            and abs(s["bias"]) <= 1.0)


def boot_delta(e, f, B=400, seed=0, col="primary"):
    rng = np.random.default_rng(seed)
    imgs = np.union1d(e.image.unique(), f.image.unique())
    ge = {k: g.values for k, g in e.groupby("image")[col]}
    gf = {k: g.values for k, g in f.groupby("image")[col]}
    res = {k: [] for k in ("med", "p90", "gross10")}
    for _ in range(B):
        s = rng.choice(imgs, len(imgs), replace=True)
        xe = np.concatenate([ge[i] for i in s if i in ge] or [np.zeros(0)])
        xf = np.concatenate([gf[i] for i in s if i in gf] or [np.zeros(0)])
        if not len(xe) or not len(xf):
            continue
        a, b = stats(xe), stats(xf)
        for k in res:
            res[k].append(a[k] - b[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
            for k, v in res.items()}


def load_raw():
    """All raw_ours*.csv in OUT; a later file (raw_ours_dist / _synth) supersedes earlier rows of the
    same (set, image, cond) -- the supplementary distortion run adds the window-weighted factors."""
    import glob
    files = sorted(glob.glob(os.path.join(OUT, "raw_ours*.csv")), key=lambda f: (os.path.basename(f) != "raw_ours.csv", f))
    raw = pd.concat([pd.read_csv(f, dtype={"image": str}).assign(_src=os.path.basename(f)) for f in files],
                    ignore_index=True)
    raw = raw.drop_duplicates(["set", "image", "cond"], keep="last").reset_index(drop=True)
    cl = pd.read_csv(os.path.join(OUT, "clean_ours.csv"), dtype={"image": str})
    return raw, cl


def long_drift(raw, clean):
    d = raw[raw.applicable.astype(str) == "True"].merge(clean, on=["set", "image"], suffixes=("", "_0"))
    out = []
    for ax in ("x", "y"):
        hv = "h" if ax == "x" else "v"
        pred = d[f"pred_{hv}"]
        base = d[f"pred_{hv}_0"]
        ext = d[f"ext_{ax}"]
        obs = pred / ext
        n_work = np.minimum(np.maximum(d["out_w"], d["out_h"]), 1024)
        exp_crop = base * d[f"fc_{ax}"] * ext
        side0 = d["side"].clip(upper=1024)
        e = pd.DataFrame({"set": d["set"], "image": d["image"], "cond": d["cond"], "axis": ax, "obs": obs,
                          "base": base, "fc": d[f"fc_{ax}"], "fd": d[f"fd_{ax}"], "f0": d[f"f0_{ax}"],
                          "fh2": d.get(f"fh2_{ax}", np.nan), "fh1": d.get(f"fh1_{ax}", np.nan), "ext": ext,
                          "conf0": d["conf_0"], "side": d["side"], "out_side": np.maximum(d["out_w"], d["out_h"]),
                          "ppt_target": d["ppt_target"], "count_used": d["count_used"], "status": d["status"],
                          "pred_crop": pred, "lab_crop": d.get(f"lab_cross_{hv}", np.nan),
                          "lab0": d.get(f"lab_cross0_{hv}", np.nan)})
        e["in_band"] = ((exp_crop >= BAND_LO) & (exp_crop <= BAND_HI * n_work) & (base >= BAND_LO)
                        & (base <= BAND_HI * side0))
        out.append(e)
    e = pd.concat(out, ignore_index=True)
    e = e[np.isfinite(e.base) & (e.base > 0)].copy()
    e["failed"] = ~(np.isfinite(e.obs) & (e.obs > 0))
    e["primary"] = e.obs / (e.fc * e.base) - 1
    e["drift_dens"] = e.obs / (e.fd * e.base) - 1
    e["drift_ignored"] = e.obs / e.base - 1
    e["drift_hann2"] = e.obs / (e.fh2 * e.base) - 1
    e["drift_hann1"] = e.obs / (e.fh1 * e.base) - 1
    e["exact_err"] = e.pred_crop / e.lab_crop - 1            # synth100: vs the exact transformed label
    e["exact_err0"] = e.base / e.lab0 - 1                    # synth100: clean prediction vs exact label
    for c in ("primary", "drift_dens", "drift_ignored", "drift_hann2", "drift_hann1", "exact_err"):
        e.loc[e.failed, c] = 1.0
    return e


def summarise():
    from scipy.stats import wilcoxon
    raw, cl = load_raw()
    clean = cl[["set", "image", "pred_h", "pred_v", "conf", "angle_x", "angle_y"]].rename(
        columns={"pred_h": "pred_h_0", "pred_v": "pred_v_0", "conf": "conf_0", "angle_x": "angle_x_0",
                 "angle_y": "angle_y_0"})
    e = long_drift(raw, clean)
    e.to_csv(os.path.join(OUT, "drift_long.csv.gz"), index=False)
    C = cond_map()
    rows = []
    for s, g in e.groupby("set"):
        gb = g[g.in_band]
        floor_rows = gb[gb.cond.isin(["null_jpeg95", "null_subpix"])]
        for cn in [c["cond"] for c in conditions() if c["cond"] != "clean"]:
            h_all = g[g.cond == cn]
            h = gb[gb.cond == cn]
            c = C[cn]
            n_app = raw[(raw.set == s) & (raw.cond == cn)]
            r = dict(set=s, model="ours", cond=cn, family=c["family"], kind=c["kind"], transform=c["fn"],
                     strength=c["strength"], n_images_total=int(n_app.image.nunique()),
                     n_images_applicable=int((n_app.applicable.astype(str) == "True").sum()),
                     n_values=len(h), n_values_total=len(h_all),
                     frac_in_band=100 * len(h) / max(1, len(h_all)))
            if not len(h) or not len(floor_rows):
                rows.append(r)
                continue
            fr = floor_rows[floor_rows.cond != cn] if c["kind"] == "null" else floor_rows
            st, f2 = stats(h.primary.values), stats(fr.primary.values)
            ci = boot_delta(h, fr, seed=zlib.crc32(cn.encode()))
            sz = None
            if cn in SIZEFLOOR_BASES and (cn + "+jpeg95") in set(gb.cond):
                zz = gb[gb.cond == cn + "+jpeg95"][["image", "axis", "obs"]]
                mm = h.merge(zz, on=["image", "axis"], suffixes=("", "_j"))
                if len(mm):
                    sz = stats((mm.obs_j / mm.obs - 1).fillna(1.0).values)
            ok = passes(st, f2)
            ok_sz = passes(st, {k: max(f2[k], sz[k]) if k != "bias" else 0 for k in f2}) if sz else ok
            pi_t = h.assign(a=h.primary.abs()).groupby("image").a.mean()
            pi_f = fr.assign(a=fr.primary.abs()).groupby("image").a.mean()
            jj = pi_t.index.intersection(pi_f.index)
            dd = (pi_t[jj] - pi_f[jj]).values
            try:
                wil_p = float(wilcoxon(dd, alternative="greater").pvalue) if len(dd) > 5 and np.any(dd != 0) else np.nan
            except ValueError:
                wil_p = np.nan
            hc, fc_ = h[h.conf0 >= 0.5], fr[fr.conf0 >= 0.5]
            ok_conf = passes(stats(hc.primary.values), stats(fc_.primary.values)) if len(hc) and len(fc_) else np.nan
            sd, si = stats(h.drift_dens.values), stats(h.drift_ignored.values)
            center_ok = bool(abs(st["bias"]) <= 1.0 and st["med"] <= 1.0)
            r.update(med_abs_drift_pct=st["med"], p90_abs_drift_pct=st["p90"], pct_gt5=st["gt5"],
                     pct_gt10=st["gross10"], median_signed_drift_pct=st["bias"],
                     floor_med_pct=f2["med"], floor_p90_pct=f2["p90"], floor_pct_gt10=f2["gross10"],
                     d_med_ci_lo=ci["med"][0], d_med_ci_hi=ci["med"][1], d_p90_ci_lo=ci["p90"][0],
                     d_p90_ci_hi=ci["p90"][1], d_gt10_ci_lo=ci["gross10"][0], d_gt10_ci_hi=ci["gross10"][1],
                     sizefloor_med_pct=sz["med"] if sz else np.nan, sizefloor_p90_pct=sz["p90"] if sz else np.nan,
                     sizefloor_pct_gt10=sz["gross10"] if sz else np.nan,
                     wilcoxon_n_images=len(jj), wilcoxon_p=wil_p,
                     PASS=bool(ok), PASS_sizematched=bool(ok_sz), PASS_conf=ok_conf,
                     PASS_center_posthoc=center_ok, flip_excess_pp=st["gross10"] - f2["gross10"],
                     conf_n_values=len(hc), fail_rate_pct=100 * float(h.failed.mean()),
                     ignored_med_abs_drift_pct=si["med"], ignored_median_signed_drift_pct=si["bias"],
                     ignored_PASS=bool(passes(si, f2)),
                     dens_med_abs_drift_pct=sd["med"], dens_median_signed_drift_pct=sd["bias"],
                     label_status=str(h.status.mode().iloc[0]) if len(h) else "")
            for ax in ("x", "y"):
                ha = h[(h.axis == ax) & ~h.failed]
                if len(ha):
                    r[f"pred_factor_{ax}_median"] = float(np.median(ha.fc))
                    r[f"meas_factor_{ax}_median"] = float(np.median(ha.obs / ha.base))
                    if ha.fh1.notna().any():
                        r[f"hann_factor_{ax}_median"] = float(np.nanmedian(ha.fh1))
                        r[f"hann2_factor_{ax}_median"] = float(np.nanmedian(ha.fh2))
            if h.fh1.notna().any() and c["family"] == "1_distortion":
                s1, s2 = stats(h.drift_hann1.dropna().values), stats(h.drift_hann2.dropna().values)
                r.update(hann_med_abs_drift_pct=s1["med"], hann_median_signed_drift_pct=s1["bias"],
                         hann_PASS=bool(passes(s1, f2)), hann2_med_abs_drift_pct=s2["med"],
                         hann2_median_signed_drift_pct=s2["bias"], hann2_PASS=bool(passes(s2, f2)))
            if s == "synth100" and h.lab_crop.notna().any():
                x1 = np.abs(h.exact_err.values)
                x0 = np.abs(h.exact_err0.values)
                r.update(exact_MAPE_pct=100 * float(np.nanmean(x1)), exact_MdAPE_pct=100 * float(np.nanmedian(x1)),
                         exact_acc10_pct=100 * float(np.nanmean(x1 <= 0.10)),
                         clean_exact_MAPE_pct=100 * float(np.nanmean(x0)), clean_exact_MdAPE_pct=100 * float(np.nanmedian(x0)),
                         clean_exact_acc10_pct=100 * float(np.nanmean(x0 <= 0.10)))
            rows.append(r)
    S = pd.DataFrame(rows)
    S["wilcoxon_p_holm"] = np.nan
    for _, g in S.groupby("set"):
        g = g[g.wilcoxon_p.notna()].sort_values("wilcoxon_p")
        m_ = len(g)
        if m_:
            adj = np.minimum(1.0, np.maximum.accumulate((m_ - np.arange(m_)) * g.wilcoxon_p.values))
            S.loc[g.index, "wilcoxon_p_holm"] = adj
    order = {c["cond"]: k for k, c in enumerate(conditions())}
    sorder = {"real150": 0, "synth100": 1, "legacy13": 2}
    S["_o"] = S.cond.map(order)
    S["_s"] = S.set.map(sorder)
    S = S.sort_values(["_s", "_o"]).drop(columns=["_o", "_s"])
    S.to_csv(os.path.join(HERE, "augment_optics_verification.csv"), index=False, float_format="%.4g")
    print(f"wrote augment_optics_verification.csv ({len(S)} rows)")
    T = tracking(e)
    T.to_csv(os.path.join(OUT, "label_tracking.csv"), index=False, float_format="%.4g")
    print(T.round(4).to_string(index=False))
    wl = make_whitelist(S, T)
    json.dump(wl, open(os.path.join(HERE, "augment_optics_whitelist.json"), "w"), indent=1)
    print("wrote augment_optics_whitelist.json")
    R = resolution_curve(raw, cl)
    R.to_csv(os.path.join(HERE, "augment_optics_min_resolution.csv"), index=False, float_format="%.4g")
    print("wrote augment_optics_min_resolution.csv")
    figure_verification(S)
    figure_tracking(e)
    figure_resolution(R)
    return S, e, R


TRACK_FAMS = {"exact geometric (zoom/stretch/shear/bow)": {"3_zoom", "3_stretch", "3_shear", "3_bow"},
              "radial distortion": {"1_distortion"}}


def tracking(e, B=1000):
    """Measured vs predicted label change, robust to the counter's hypothesis flips: one point per
    (condition, axis) = median over images of log(obs/clean) (y) and of log(predicted factor) (x);
    OLS slope through these points; 95 % CI by bootstrapping IMAGES (medians recomputed)."""
    C = cond_map()
    out = []
    for s, g in e.groupby("set"):
        for fam_name, fams in TRACK_FAMS.items():
            for lab in (("fc", "uniform (guide-line) label"), ("fh1", "Hann-weighted label"),
                        ("fh2", "Hann^2-weighted label")):
                col, lname = lab
                h = g[g.in_band & ~g.failed & g.cond.map(lambda c: C[c]["family"] in fams)].copy()
                h = h[np.isfinite(h[col])]
                if len(h) < 10:
                    continue
                h["key"] = h.cond + "|" + h.axis
                h["x"] = np.log(h[col])
                h["y"] = np.log(h.obs / h.base)
                imgs = np.sort(h.image.unique())
                keys = np.sort(h.key.unique())
                ii = {k: i for i, k in enumerate(imgs)}
                kk = {k: i for i, k in enumerate(keys)}
                X = np.full((len(imgs), len(keys)), np.nan)
                Y = np.full((len(imgs), len(keys)), np.nan)
                X[h.image.map(ii).values, h.key.map(kk).values] = h.x.values
                Y[h.image.map(ii).values, h.key.map(kk).values] = h.y.values

                def fit(rows):
                    xm, ym = np.nanmedian(X[rows], 0), np.nanmedian(Y[rows], 0)
                    ok = np.isfinite(xm) & np.isfinite(ym)
                    b, a = np.polyfit(xm[ok], ym[ok], 1)
                    return b, a, xm[ok], ym[ok]

                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b, a, xm, ym = fit(np.arange(len(imgs)))
                    rng = np.random.default_rng(0)
                    bs = [fit(rng.integers(0, len(imgs), len(imgs)))[0] for _ in range(B)]
                res = ym - xm
                flips = np.abs(h.y - h.x) > np.log(1.1)
                out.append(dict(set=s, family=fam_name, label=lname, n_points=len(xm), n_values=len(h),
                                n_images=len(imgs), slope=b, intercept=a, slope_ci_lo=np.percentile(bs, 2.5),
                                slope_ci_hi=np.percentile(bs, 97.5),
                                max_abs_median_resid_pct=100 * float(np.max(np.abs(res))),
                                median_abs_median_resid_pct=100 * float(np.median(np.abs(res))),
                                pct_values_within_2pct=100 * float(np.mean(np.abs(h.y - h.x) < np.log(1.02))),
                                pct_values_flipped_gt10=100 * float(flips.mean())))
    return pd.DataFrame(out)


LADDERS = {
    "vignette_cos4": {"+": ["vig_0.2", "vig_0.4", "vig_0.6"]},
    "radial_distortion": {"+": ["dist_+0.005", "dist_+0.01", "dist_+0.02", "dist_+0.05", "dist_+0.1"],
                          "-": ["dist_-0.005", "dist_-0.01", "dist_-0.02", "dist_-0.05", "dist_-0.1"]},
    "chromatic_aberration": {"+": ["ca_0.002", "ca_0.005", "ca_0.01", "ca_0.02"]},
    "dof_gradient": {"+": ["dof_0.1pitch", "dof_0.2pitch", "dof_0.3pitch", "dof_0.45pitch"]},
    "led_hotspot": {"+": ["led_0.5", "led_1", "led_2", "led_4"]},
    "phone_capture": {"-": [f"phone_ppt{p:g}" for p in sorted(PPTS, reverse=True)]},
    "area_downsample": {"-": [f"area_ppt{p:g}" for p in sorted(PPTS, reverse=True)]},
    "zoom": {"+": ["zoom_1.1", "zoom_1.25", "zoom_1.5", "zoom_2"], "-": ["zoom_0.9", "zoom_0.8"]},
    "stretch": {"+": ["strx_1.15", "strx_1.3", "strx_1.6"], "-": ["strx_0.85"]},
    "stretch_y": {"+": ["stry_1.3"]},
    "shear": {"+": ["sheary_0.05", "sheary_0.1", "sheary_0.18", "sheary_0.27"]},
    "shear_x": {"+": ["shearx_0.1"]},
    "shear_frame": {"+": ["sheary_0.1_frame"]},
    "bow": {"+": ["bowy_0.01", "bowy_0.02", "bowy_0.04"]},
}
SIZE_CHANGING = set(SIZEFLOOR_BASES)
DECISION_SETS = ("real150", "synth100")


def make_whitelist(S, T):
    C = cond_map()
    strict, center = {}, {}
    for (s, cn), g in S.groupby(["set", "cond"]):
        g = g.iloc[0]
        if pd.isna(g.get("PASS")):
            continue
        strict[(s, cn)] = bool(g.PASS_sizematched if cn in SIZE_CHANGING else g.PASS)
        center[(s, cn)] = bool(g.PASS_center_posthoc)
    wl = {"rule_strict": "PRE-REGISTERED (== verify_augment.py): within clean-run noise for the frozen counter on "
                         "real150 AND synth100: median |drift| <= floor+0.5pp, p90 <= floor+2pp, P(>10%) <= floor+2pp, "
                         "|median signed drift| <= 1%; drift is measured against the PREDICTED label change (crossing "
                         "factor, clean-run yarn tilt); floor = pooled JPEG-q95 re-encode + 0.5-px shift; transforms "
                         "with a size-matched floor use max(floor, size-matched floor).  legacy-13 reported only.",
          "rule_label_exact": "POST HOC (added after seeing that the counter's lattice-hypothesis flips reject every "
                              "geometric resampling, including the 2-deg rotations of the original audit): "
                              "|median signed drift| <= 1% AND median |drift| <= 1% on real150 AND synth100 (the "
                              "centre of the drift distribution follows the predicted label); tail flips are reported "
                              "as flip_excess_pp.  Label maths itself is verified exactly on analytic gratings "
                              "(test_augment_optics.py) and against exact synthetic labels (exact_MAPE in the CSV).",
          "transforms": {}, "policy_strict": {}, "policy": {}, "tracking": T.to_dict("records")}
    for key, dirs in LADDERS.items():
        res = {"tested": [], "safe_strict": {}, "safe_label_exact": {}}
        for d, ladder in dirs.items():
            up_s, up_c, stop_s, stop_c = None, None, False, False
            for cn in ladder:
                ok_s = all(strict.get((s, cn), False) for s in DECISION_SETS)
                ok_c = all(center.get((s, cn), False) for s in DECISION_SETS)
                g = S[(S.set == "real150") & (S.cond == cn)]
                res["tested"].append({"cond": cn, "level": C[cn]["strength"], "strict": ok_s, "label_exact": ok_c,
                                      "real150_strict": strict.get(("real150", cn)),
                                      "synth100_strict": strict.get(("synth100", cn)),
                                      "legacy13_strict": strict.get(("legacy13", cn)),
                                      "real150_flip_excess_pp": float(g.flip_excess_pp.iloc[0]) if len(g) and "flip_excess_pp" in g else None})
                if not ok_s:
                    stop_s = True
                if not ok_c:
                    stop_c = True
                if not stop_s:
                    up_s = C[cn]["strength"]
                if not stop_c:
                    up_c = C[cn]["strength"]
            res["safe_strict"][d] = up_s
            res["safe_label_exact"][d] = up_c
        wl["transforms"][key] = res

    def policy(tier):
        T_ = {k: v[f"safe_{tier}"] for k, v in wl["transforms"].items()}
        pol = {}
        if T_["vignette_cos4"].get("+"):
            pol["vignette_cos4"] = {"p": 0.3, "strength": [0.0, float(T_["vignette_cos4"]["+"])]}
        a, b = T_["radial_distortion"].get("+"), T_["radial_distortion"].get("-")
        if a or b:
            pol["radial_distortion"] = {"p": 0.2, "k1": [float(b) if b else 0.0, float(a) if a else 0.0]}
        if T_["chromatic_aberration"].get("+"):
            pol["chromatic_aberration"] = {"p": 0.2, "eps": [0.0, float(T_["chromatic_aberration"]["+"])]}
        if T_["dof_gradient"].get("+"):
            pol["dof_gradient"] = {"p": 0.2, "sigma_pitch": [0.0, float(T_["dof_gradient"]["+"])]}
        if T_["led_hotspot"].get("+"):
            pol["led_hotspot"] = {"p": 0.2, "amp": [0.2, float(T_["led_hotspot"]["+"])]}
        if T_["phone_capture"].get("-"):
            pol["phone_capture"] = {"p": 0.3, "ppt": [float(T_["phone_capture"]["-"]), 10.0]}
        z_hi, z_lo = T_["zoom"].get("+"), T_["zoom"].get("-")
        if z_hi or z_lo:
            pol["zoom"] = {"p": 0.4, "s": [float(z_lo) if z_lo else 1.0, float(z_hi) if z_hi else 1.0]}
        if T_["stretch"].get("+"):
            pol["stretch"] = {"p": 0.3, "s": [1.0, float(T_["stretch"]["+"])]}
        if T_["shear"].get("+"):
            g = float(T_["shear"]["+"])
            pol["shear"] = {"p": 0.2, "gamma": [-g, g]}
        if T_["bow"].get("+"):
            pol["bow"] = {"p": 0.1, "depth": [0.0, float(T_["bow"]["+"])]}
        return pol

    wl["policy_strict"] = policy("strict")
    wl["policy"] = policy("label_exact")
    wl["policy_note"] = ("augment_optics.sample() uses 'policy' (label-exact tier).  Use 'policy_strict' to restrict "
                         "to transforms under which the frozen counter itself is invariant.")
    return wl


# =============================================================================================
# minimum phone resolution
# =============================================================================================
def resolution_curve(raw, cl):
    gt = legacy_gt()
    rr = raw[raw.cond.str.match(r"^(phone|area|phoneup2)_ppt[0-9.]+$")].copy()
    c0 = cl.set_index(["set", "image"])
    man = os.path.join(ROOT, "annotation_tool", "data", "manifest.json")
    hi = set()
    if os.path.exists(man):
        hi = {str(m["id"]) for m in json.load(open(man))["images"] if m["rect_size"][0] > 1000}
    rows = []
    for r in rr.itertuples():
        base = c0.loc[(r.set, r.image)]
        app = str(r.applicable) == "True"
        for ax, col, gcol, lcol in (("x", "pred_h", "warp", "lab_cross0_h"), ("y", "pred_v", "weft", "lab_cross0_v")):
            ref = base[col]
            truth = np.nan
            if r.set == "legacy13" and r.image in gt:
                truth = float(gt[r.image][gcol])
            elif r.set == "synth100":
                truth = float(getattr(r, lcol)) if app else np.nan
            ref_count = truth if np.isfinite(truth) else ref
            side_out = r.out_w if app else np.nan
            if app and r.cond.startswith("phoneup2"):
                side_out = r.out_w / 2.0         # sensor px, before the x2 up-sampling
            pred = getattr(r, col) if app else np.nan
            rows.append(dict(set=r.set, image=r.image, pipeline=r.cond.split("_ppt")[0],
                             ppt_target=float(r.cond.split("_ppt")[1]), axis=ax, applicable=app, side_native=r.side,
                             side_out=side_out, ref=ref, truth=truth, pred=pred,
                             ppt_axis=side_out / ref_count if np.isfinite(ref_count) and ref_count > 0 else np.nan))
    L = pd.DataFrame(rows)
    L["subset"] = np.where(L.set == "legacy13", "legacy13", np.where(L.set == "synth100", "synth100",
                           np.where(L.image.isin(hi), "real150_highres", "real150_lowres")))
    L["err_ref"] = L.pred / L.ref - 1
    L["err_gt"] = L.pred / L.truth - 1
    bad = L.applicable & ~(np.isfinite(L.pred) & (L.pred > 0))
    L.loc[bad, ["err_ref", "err_gt"]] = 1.0
    L.to_csv(os.path.join(OUT, "resolution_long.csv"), index=False, float_format="%.5g")
    rng = np.random.default_rng(1)

    def summ(h, errcol):
        h = h[np.isfinite(h[errcol])]
        imgs = np.sort(h.image.unique())
        a = np.abs(h[errcol].values)
        G = [np.abs(h[h.image == i][errcol].values) for i in imgs]
        bs_med, bs_w10 = [], []
        for _ in range(1000):
            x = np.concatenate([G[i] for i in rng.integers(0, len(imgs), len(imgs))])
            bs_med.append(np.median(x))
            bs_w10.append(np.mean(x <= 0.10))
        return dict(n_images=len(imgs), n_values=len(a), median_ape_pct=100 * np.median(a),
                    median_ape_ci_lo=100 * np.percentile(bs_med, 2.5), median_ape_ci_hi=100 * np.percentile(bs_med, 97.5),
                    mean_ape_pct=100 * np.mean(a), p90_ape_pct=100 * np.percentile(a, 90),
                    within5_pct=100 * np.mean(a <= 0.05), within10_pct=100 * np.mean(a <= 0.10),
                    within10_ci_lo=100 * np.percentile(bs_w10, 2.5), within10_ci_hi=100 * np.percentile(bs_w10, 97.5),
                    median_signed_pct=100 * np.median(h[errcol].values),
                    median_ppt_axis=float(np.nanmedian(h.ppt_axis)))

    REF = {"real150_highres": [("ref", "clean prediction")], "real150_lowres": [("ref", "clean prediction")],
           "legacy13": [("gt", "human GT"), ("ref", "clean prediction")],
           "synth100": [("gt", "exact label"), ("ref", "clean prediction")]}
    out = []
    for (sub, pipe, p), h in L[L.applicable].groupby(["subset", "pipeline", "ppt_target"]):
        for code, name in REF[sub]:
            r = dict(view="by_target_ppt", subset=sub, pipeline=pipe, ppt=p, reference=name)
            r.update(summ(h, "err_" + code))
            r["n_images_not_applicable"] = int(L[(L.subset == sub) & (L.pipeline == pipe) & (L.ppt_target == p)
                                                 & ~L.applicable].image.nunique())
            out.append(r)
    bins = [0, 2.25, 2.75, 3.25, 3.75, 4.5, 5.5, 7, 9, 12, 1e9]
    labels = ["2", "2.5", "3", "3.5", "4", "5", "6", "8", "10", ">12"]
    L["ppt_bin"] = pd.cut(L.ppt_axis, bins, labels=labels)
    for (sub, pipe, b), h in L[L.applicable].groupby(["subset", "pipeline", "ppt_bin"], observed=True):
        for code, name in REF[sub]:
            r = dict(view="by_axis_ppt_bin", subset=sub, pipeline=pipe, ppt=b, reference=name)
            r.update(summ(h, "err_" + code))
            out.append(r)
    # clean, native resolution (<= 1024 px working size)
    cc = cl.copy()
    cc["subset"] = np.where(cc.set == "legacy13", "legacy13", np.where(cc.set == "synth100", "synth100",
                            np.where(cc.image.isin(hi), "real150_highres", "real150_lowres")))
    labs = raw[raw.set == "synth100"].drop_duplicates("image").set_index("image")
    for sub in ("real150_highres", "real150_lowres", "legacy13", "synth100"):
        h = cc[cc.subset == sub]
        if not len(h):
            continue
        rr2 = []
        for x in h.itertuples():
            for col, gcol, lcol in (("pred_h", "warp", "lab_cross0_h"), ("pred_v", "weft", "lab_cross0_v")):
                t = np.nan
                if sub == "legacy13" and x.image in gt:
                    t = float(gt[x.image][gcol])
                elif sub == "synth100" and x.image in labs.index:
                    t = float(labs.loc[x.image, lcol])
                v = getattr(x, col)
                cnt = t if np.isfinite(t) else v
                rr2.append(dict(image=x.image, err_gt=(v / t - 1) if np.isfinite(t) else np.nan, err_ref=0.0,
                                ppt_axis=min(x.side, 1024) / cnt if cnt > 0 else np.nan))
        hh = pd.DataFrame(rr2)
        for code, name in REF[sub]:
            if code == "ref":
                continue
            r = dict(view="clean_native", subset=sub, pipeline="clean", ppt=np.nan, reference=name)
            r.update(summ(hh, "err_" + code))
            out.append(r)
    return pd.DataFrame(out)


# =============================================================================================
# figures
# =============================================================================================
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#8a8984", "#1f9e6e"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "DejaVu Sans"})
    return plt


def figure_verification(S):
    plt = _style()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    conds = [c for c in conditions() if c["kind"] not in ("clean", "sizefloor")]
    sets = [("real150", "real-150 (decision set)"), ("synth100", "synthetic-100, exact labels (decision set)"),
            ("legacy13", "legacy-13 (reported only)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.148 * len(conds) + 1.5), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    ypos, y, last = {}, 0, None
    for c in conds:
        if c["family"] != last and last is not None:
            y += 0.6
        last = c["family"]
        ypos[c["cond"]] = y
        y += 1
    for ax, (st, title) in zip(axes, sets):
        g = S[S.set == st].set_index("cond")
        if not len(g) or "floor_p90_pct" not in g:
            continue
        thr = float(g.floor_p90_pct.dropna().median() + 2.0)
        ax.axvspan(0.005, thr, color="#f0efec", lw=0, zorder=0)
        ax.axvline(thr, color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.axvline(1.0, color=INK2, lw=0.5, ls=(0, (1, 2)))
        for c in conds:
            cn = c["cond"]
            if cn not in g.index or pd.isna(g.loc[cn].get("med_abs_drift_pct", np.nan)):
                continue
            r = g.loc[cn]
            ok = bool(r.PASS_sizematched if cn in SIZE_CHANGING else r.PASS)
            okc = bool(r.get("PASS_center_posthoc", False))
            col = BLUE if ok else (GREEN if okc else ORANGE)
            lo, hi = max(0.006, r.med_abs_drift_pct), max(0.006, r.p90_abs_drift_pct)
            yy = ypos[cn]
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, solid_capstyle="round")
            ax.plot(lo, yy, "o", ms=3.0, color=col, mfc=col if (ok or okc) else "white", mew=1)
            ax.plot(hi, yy, "|", ms=5, color=col)
            fam = c["family"]
            if fam.startswith("3_") or fam == "1_distortion":
                im = r.get("ignored_med_abs_drift_pct", np.nan)
                if np.isfinite(im) and im > 0.05:
                    ax.plot(max(0.006, im), yy, "x", ms=3.2, color=GREY)
        ax.set_xscale("log")
        ax.set_xlim(0.006, 200)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.5)
    axes[1].set_xlabel("|drift| from the predicted label (%):  dot = median, bar end = p90;  grey x = median if the "
                       "label change were ignored;  blue = within clean-run noise (pre-registered rule), green = "
                       "median-level only (post hoc), orange = neither;  shaded = p90 threshold", fontsize=6.8)
    axes[0].set_yticks([ypos[c["cond"]] for c in conds])
    axes[0].set_yticklabels([c["cond"] for c in conds], fontsize=6.2)
    axes[0].invert_yaxis()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figs", f"augment_optics_verification.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_tracking(e):
    """Measured vs predicted count change, one point per (condition, axis): median over images, bars =
    interquartile range.  Left: exact label-transforming family.  Right: radial distortion with the
    uniform guide-line label (open) and the Hann^2-weighted label (filled)."""
    plt = _style()
    C = cond_map()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw=dict(wspace=0.25))
    col = {"real150": BLUE, "synth100": GREEN, "legacy13": ORANGE}
    off = {"real150": -0.004, "synth100": 0.0, "legacy13": 0.004}
    for k, (fams, xcols) in enumerate(((("3_zoom", "3_stretch", "3_shear", "3_bow"), ("fc",)),
                                       (("1_distortion",), ("fc", "fh1")))):
        ax = axes[k]
        lo, hi = 0.45, 1.3
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6)
        for s in ("real150", "synth100", "legacy13"):
            h = e[(e.set == s) & e.in_band & ~e.failed & e.cond.map(lambda c: C[c]["family"] in fams)]
            for xc in xcols:
                if not h[xc].notna().any():
                    continue
                for (cn, axn), g in h.groupby(["cond", "axis"]):
                    x = float(np.nanmedian(g[xc]))
                    y = g.obs / g.base
                    q1, q2, q3 = np.percentile(y, [25, 50, 75])
                    filled = (xc == "fh1") or (k == 0)
                    ax.errorbar(x + off[s], q2, yerr=[[q2 - q1], [q3 - q2]], fmt="o", ms=3.2, color=col[s],
                                mfc=col[s] if filled else "white", elinewidth=0.7, capsize=0, alpha=0.9)
        ax.set_xlabel("predicted count factor" if k == 0 else
                      "predicted factor: open = guide-line (area-mean) label,\nfilled = Hann-window-weighted local scale")
        ax.set_ylabel("measured factor (ours: transformed / clean)")
        ax.grid(color=GRID, lw=0.5)
        if k == 1:
            ax.set_xlim(0.83, 1.03)
            ax.set_ylim(0.83, 1.06)
        else:
            ax.set_xlim(0.45, 1.3)
            ax.set_ylim(0.45, 1.3)
    axes[0].set_title("(a) zoom, stretch, shear, bow: median over images, IQR bars", fontsize=8, loc="left")
    axes[1].set_title("(b) radial distortion", fontsize=8, loc="left")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=col[s], label=s) for s in col], fontsize=7,
                   frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figs", f"augment_optics_label_tracking.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_resolution(R):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw=dict(wspace=0.28))
    spec = [("real150_highres", "phone", "clean prediction", BLUE, "-", "high-res real (65): phone ISP, vs clean"),
            ("real150_highres", "area", "clean prediction", BLUE, ":", "high-res real: area resize only"),
            ("legacy13", "phone", "human GT", ORANGE, "-", "legacy-13: phone ISP, vs human GT"),
            ("legacy13", "area", "human GT", ORANGE, ":", "legacy-13: area only, vs human GT"),
            ("synth100", "phone", "exact label", GREEN, "-", "synthetic-100: phone ISP, vs exact label"),
            ("real150_highres", "phoneup2", "clean prediction", GREY, "--", "high-res: phone + x2 bicubic up-sampling")]
    V = R[R.view == "by_target_ppt"].copy()
    V["ppt"] = pd.to_numeric(V.ppt, errors="coerce")      # the CSV mixes levels with bin labels
    for sub, pipe, ref, col, ls, lab in spec:
        h = V[(V.subset == sub) & (V.pipeline == pipe) & (V.reference == ref)].sort_values("ppt")
        if not len(h):
            continue
        for ax, m, lo, hi in ((axes[0], "median_ape_pct", "median_ape_ci_lo", "median_ape_ci_hi"),
                              (axes[1], "within10_pct", "within10_ci_lo", "within10_ci_hi")):
            ax.plot(h.ppt, h[m], ls, color=col, lw=1.4, marker="o", ms=3, label=lab)
            if ls == "-":
                ax.fill_between(h.ppt.astype(float), h[lo], h[hi], color=col, alpha=0.13, lw=0)
    for sub, ref, col in (("legacy13", "human GT", ORANGE), ("synth100", "exact label", GREEN)):
        cn = R[(R.view == "clean_native") & (R.subset == sub) & (R.reference == ref)]
        if len(cn):
            axes[0].axhline(cn.median_ape_pct.iloc[0], color=col, lw=0.7, ls=(0, (1, 2)))
            axes[1].axhline(cn.within10_pct.iloc[0], color=col, lw=0.7, ls=(0, (1, 2)))
    for ax in axes:
        ax.axvline(1 / 0.42, color=GREY, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([2, 2.5, 3, 4, 5, 6, 8, 10])
        ax.set_xticklabels(["2", "2.5", "3", "4", "5", "6", "8", "10"])
        ax.minorticks_off()
        ax.set_xlabel("pixels per yarn on the finer axis (sensor px)")
        ax.grid(color=GRID, lw=0.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median |error| (%)")
    axes[1].set_ylabel("axis counts within 10 % (%)")
    axes[0].text(1 / 0.42 * 1.03, 0.30, "ours' search\nlimit 2.38 px", transform=axes[0].get_xaxis_transform(),
                 fontsize=6.3, color=GREY, va="top")
    axes[0].set_title("dotted: clean native-resolution error (legacy GT, synthetic)", fontsize=7, loc="left", color=INK2)
    axes[1].legend(fontsize=5.9, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figs", f"augment_optics_min_resolution.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def examples(img_id="80", ca_id="77", ca_eps=0.003):
    """One example per transform (full crop + 3x zoomed inset), predicted label factor annotated."""
    plt = _style()
    src = load_img(os.path.join(CROPS, f"{img_id}.jpg"))
    cl = pd.read_csv(os.path.join(OUT, "clean_ours.csv"), dtype={"image": str}) if os.path.exists(
        os.path.join(OUT, "clean_ours.csv")) else None
    h0, v0 = 45.4, 54.1
    if cl is not None:
        r = cl[(cl.set == "real150") & (cl.image == img_id)]
        if len(r):
            h0, v0 = float(r.pred_h.iloc[0]), float(r.pred_v.iloc[0])
    cnt = max(h0, v0)
    S = src.shape[0]
    rng = lambda k: np.random.default_rng(k)  # noqa: E731
    panels = [("clean", src, AO.identity_lt(src))]
    panels.append(("vignette cos$^4$ 0.6",) + AO.vignette_cos4(src, rng=rng(3), strength=0.6))
    panels.append(("barrel k1=+0.1",) + AO.radial_distortion(src, k1=0.1))
    panels.append(("pincushion k1=-0.1",) + AO.radial_distortion(src, k1=-0.1))
    panels.append(("lateral CA eps=0.01",) + AO.chromatic_aberration(src, eps=0.01, center=(0.5, 0.5), sign=1))
    panels.append(("DOF ramp 0.45 pitch",) + AO.dof_gradient(src, angle=20, pitch_px=S / cnt, sigma_pitch=0.45))
    panels.append(("LED hot spot amp 2",) + AO.led_hotspot(src, amp=2.0, center=(0.4, 0.45)))
    panels.append(("phone 3 px/yarn",) + AO.phone_capture(src, rng=rng(1), ppt=3, count=cnt))
    panels.append(("area 3 px/yarn",) + AO.area_downsample(src, ppt=3, count=cnt))
    panels.append(("zoom 1.5",) + AO.zoom(src, s=1.5))
    panels.append(("stretch x 1.3",) + AO.stretch(src, sx=1.3))
    panels.append(("weft skew g=0.18",) + AO.shear(src, gamma=0.18))
    panels.append(("bowed weft 0.04",) + AO.bow(src, depth=0.04))
    ncol = 5
    nrow = int(math.ceil((len(panels) + 2) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.35 * nrow))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for k, (name, im, lt) in enumerate(panels):
        ax = axes[k]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb, interpolation="nearest" if im.shape[0] < 400 else "antialiased")
        H, W = im.shape[:2]
        z = rgb[int(0.04 * H):int(0.04 * H) + H // 4, int(0.71 * W):int(0.71 * W) + W // 4]
        ins = ax.inset_axes([0.62, 0.62, 0.38, 0.38])
        ins.imshow(z, interpolation="nearest")
        ins.set_xticks([])
        ins.set_yticks([])
        for sp in ins.spines.values():
            sp.set_color("white")
            sp.set_linewidth(1.2)
        h, v, _ = lt.crossing((h0, v0))
        lab = "label: unchanged" if abs(h / h0 - 1) < 1e-6 and abs(v / v0 - 1) < 1e-6 else \
            f"label: h x{h / h0:.3f}, v x{v / v0:.3f}"
        if lt.status == "approx":
            lab += " (approx.)"
        ax.set_title(f"{name}\n{lab}", fontsize=7, color=INK, loc="left")
    # real CA on photo #77 (corner, native resolution) for reference
    try:
        real = cv2.imread(os.path.join(CROPS, f"{ca_id}.jpg"))
        Hh = real.shape[0]
        corner = cv2.cvtColor(real[:Hh // 6, -Hh // 6:], cv2.COLOR_BGR2RGB)
        ax = axes[len(panels)]
        ax.imshow(corner, interpolation="nearest")
        ax.set_title(f"real photo #{ca_id}: corner, native res\n(colour fringes = lateral CA)", fontsize=7, loc="left")
        sim, _ = AO.chromatic_aberration(load_img(os.path.join(CROPS, f"{ca_id}.jpg")), eps=ca_eps,
                                         center=(0.5, 0.5), sign=1)
        Hs = sim.shape[0]
        ax = axes[len(panels) + 1]
        ax.imshow(cv2.cvtColor(sim[:Hs // 6, -Hs // 6:], cv2.COLOR_BGR2RGB), interpolation="nearest")
        ax.set_title(f"#{ca_id} + simulated CA eps={ca_eps:g}\n(real crops: eps p90 0.002, p99 0.004)", fontsize=7, loc="left")
    except Exception as err:   # noqa: BLE001
        print("CA reference panel skipped:", err)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figs", f"augment_optics_examples.{ext}"), dpi=170)
    plt.close(fig)
    print("wrote figs/augment_optics_examples.pdf/.png")


def prep():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "paper", "tools"))
    import legacy_unified as L   # the scorer used for paper/tables/legacy13_unified.csv
    gt = L.load_gt()
    d = {k: dict(warp=float(r.warp), weft=float(r.weft), weave=str(r.weave), unclear=bool(r.unclear),
                 is_range=bool(r.is_range)) for k, r in gt.iterrows()}
    assert sorted(d) == sorted(LEGACY13), sorted(d)
    json.dump(d, open(os.path.join(OUT, "legacy13_gt.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "legacy13_gt.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "summarise", "examples", "figs"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sets", default="legacy13,real150")
    ap.add_argument("--conds", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "run":
        run_ours(tuple(a.sets.split(",")), a.workers, [c for c in a.conds.split(",") if c] or None, a.tag)
    elif a.cmd == "summarise":
        summarise()
    elif a.cmd == "examples":
        examples()
    elif a.cmd == "figs":       # re-plot from the written CSVs (no statistics recomputed)
        figure_verification(pd.read_csv(os.path.join(HERE, "augment_optics_verification.csv")))
        figure_tracking(pd.read_csv(os.path.join(OUT, "drift_long.csv.gz"), dtype={"image": str}))
        figure_resolution(pd.read_csv(os.path.join(HERE, "augment_optics_min_resolution.csv")))
