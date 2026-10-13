"""combined_v1.py -- fit, freeze-support and evaluation of the PRE-REGISTERED combined counter "combined-v1"
(PG-Lattice "ours" + log-Gabor bank; the rule itself lives in methods/results/combined_frozen_v1.py).

Written on 2026-10-11 (US Eastern), before any human count of the 76 PickGlass scenes existed.  Nothing here
reads a human count of the 150 photos; the only real labels used anywhere are the 13 legacy (2019) hand counts,
and only in --eval-legacy, AFTER the freeze, and reported as POST HOC (legacy-13 was used during development).

Steps (in this order; time stamps are written into every output):
  --fit            thresholds (R_MIN, D_AGREE) on SYNTHETIC S-cal only (both generator variants, 19 conditions,
                   exact labels): methods/results/cascade_synth_cal_{legacygrid,default}.csv
                   -> methods/results/combined_v1_fit.json, combined_v1_fit_grid.csv
                   Objective (declared before fitting): maximise P5 = share of counts (2 per crop) with
                   |pred/true - 1| <= 5 %; ties (equal number of hits) -> lower MAPE.  Non-finite prediction =
                   miss with APE 100 %.  Secondary (sensitivity) variant: minimise MAPE, ties -> higher P5.
  (freeze)         thresholds copied into combined_frozen_v1.py; PREREG_combined_v1.json written (--prereg)
  --predict        apply the frozen rule to label-free outputs of the frozen counter that already exist:
                   the 150 rectified crops (methods/predictions_ours_150.csv), legacy-13 canonical crops and
                   their 19 + 10 degradations (methods/results/legacy13_vlmcrops_degrade{,_extra}.csv), and
                   PickGlass-Stress legacy-13 renders (camera_aug/results/stress_legacy13_classical.csv)
  --heldout-make   render a FRESH synthetic held-out set (never used before): seeds 970000..970199
                   ("legacygrid": sensor grid ~ U[1630, 1940] px/inch, crop not capped) and 975000..975199
                   ("default": generator defaults, crop <= 1024 px) -> scratch crops + labels
  --heldout-score  frozen counter (ours + log-Gabor) on the held-out crops x the same 19 conditions as S-cal
                   -> methods/results/combined_v1_synth_heldout.csv
  --legacy-recount combined_frozen_v1.estimate() end to end on the 20 canonical legacy crops (checks that the frozen
                   counter loaded by path reproduces the stored ours / log-Gabor counts; adds sharpness)
  --prereg         write methods/results/PREREG_combined_v1.json (never overwritten)
  Evaluation (held-out synthetic test, POST HOC legacy tables): methods/combined_v1_eval.py (written after the freeze).

    .tools/venv/bin/python methods/combined_v1.py --fit
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("SYNTH_PROFILE", "v2")

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
PAPER = os.path.join(ROOT, "paper")
SCAL = {v: os.path.join(RES, f"cascade_synth_cal_{v}.csv") for v in ("legacygrid", "default")}
FIT_JSON = os.path.join(RES, "combined_v1_fit.json")
FIT_GRID = os.path.join(RES, "combined_v1_fit_grid.csv")
RULE_PY = os.path.join(RES, "combined_frozen_v1.py")
PREREG = os.path.join(RES, "PREREG_combined_v1.json")
HELDOUT_CSV = os.path.join(RES, "combined_v1_synth_heldout.csv")
SCRATCH = os.environ.get(
    "COMBINED_SCRATCH",
    os.path.join((os.environ.get("PICKGLASS_SCRATCH") or os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "scratch")), "wf_round2/prereg-combined/heldout"))
HELDOUT_SEEDS = {"legacygrid": 970000, "default": 975000}
GRID_LEGACY = (1630.0, 1940.0)  # as methods/cascade_synth.py
P5_TOL, P10_TOL = 5.0, 10.0
APE_MISS = 100.0
B = 10000
SEED_BOOT = 20261011


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rule_module():
    spec = importlib.util.spec_from_file_location("combined_frozen_v1", RULE_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ape(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    a = 100 * np.abs(pred / true - 1)
    return np.where(np.isfinite(a), a, APE_MISS)


# =================================================================================== synthetic data
def load_synth(path, set_name):
    s = pd.read_csv(path)
    s["level"] = s.level.astype(float).round(4)
    s["set"] = set_name
    s["image"] = set_name + ":" + s.name.astype(str)
    return s


def synth_counts(s, R, r_min=None, d=None, variant="primary"):
    """(n, 2) APE matrices for ours, gabor, combined on a synthetic frame."""
    out = R.combine_arrays(s.ours_h, s.ours_v, s.gabor_h, s.gabor_v, s.sharpness, s.N_work,
                           r_min=r_min, d_agree=d, variant=variant)
    A = {"ours": np.c_[ape(s.ours_h, s.h_count), ape(s.ours_v, s.v_count)],
         "gabor": np.c_[ape(s.gabor_h, s.h_count), ape(s.gabor_v, s.v_count)],
         "combined": np.c_[ape(out["h"], s.h_count), ape(out["v"], s.v_count)]}
    return A, out


def fit(args):
    R = rule_module()
    s = pd.concat([load_synth(SCAL[v], f"S-cal-{v}") for v in ("legacygrid", "default")], ignore_index=True)
    t_true = np.c_[s.h_count, s.v_count]
    oh, ov, gh, gv = (s[c].to_numpy(float) for c in ("ours_h", "ours_v", "gabor_h", "gabor_v"))
    rho = R.rho_of(oh, ov, s.sharpness, s.N_work)
    r_grid = np.unique(np.r_[0.0, np.quantile(rho, np.arange(0.0025, 0.3001, 0.0025))])
    d_grid = np.round(np.arange(0.0, 0.6001, 0.01), 2)
    rows = []
    for r in r_grid:
        for d in d_grid:
            o = R.combine_arrays(oh, ov, gh, gv, s.sharpness, s.N_work, r_min=r, d_agree=d)
            a = np.c_[ape(o["h"], t_true[:, 0]), ape(o["v"], t_true[:, 1])]
            rows.append(dict(r_min=float(r), d_agree=float(d), hits5=int((a <= P5_TOL).sum()),
                             P5=float(100 * (a <= P5_TOL).mean()), MAPE=float(a.mean()),
                             P10=float(100 * (a <= P10_TOL).mean()), fallback_pct=float(100 * o["fallback"].mean()),
                             mean_pct=float(100 * np.mean(np.r_[o["src_h"], o["src_v"]] == "mean"))))
    G = pd.DataFrame(rows)
    prim = G.sort_values(["hits5", "MAPE", "r_min", "d_agree"], ascending=[False, True, True, True]).iloc[0]
    sec = G.sort_values(["MAPE", "hits5", "r_min", "d_agree"], ascending=[True, False, True, True]).iloc[0]
    A = {"ours": np.c_[ape(oh, t_true[:, 0]), ape(ov, t_true[:, 1])],
         "gabor": np.c_[ape(gh, t_true[:, 0]), ape(gv, t_true[:, 1])]}
    comp = {k: dict(P5=float(100 * (a <= P5_TOL).mean()), MAPE=float(a.mean()), P10=float(100 * (a <= P10_TOL).mean()))
            for k, a in A.items()}
    # how flat is the objective near the optimum (reported, not used)
    near = G[G.hits5 >= prim.hits5 - 0.001 * 2 * len(s)]
    # marginal: best P5 with the defocus fallback forced ON at the cascade gate's r_min (sensitivity, reported)
    fb_only = G[G.d_agree == 0.0].sort_values(["hits5", "MAPE"], ascending=[False, True])
    out = {
        "rule": "combined-v1 (methods/results/combined_frozen_v1.py): rho < R_MIN -> log-Gabor both axes; else per axis "
                "|ln(ours/gabor)| <= D_AGREE -> geometric mean; else ours",
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fit_data": {k: dict(path=os.path.relpath(p, ROOT), sha256=sha256(p)) for k, p in SCAL.items()},
        "n_crops": int(s.image.nunique()), "n_rows": int(len(s)), "n_counts": int(2 * len(s)),
        "conditions": sorted({f"{k}:{lv:g}" for k, lv in zip(s.kind, s.level)}),
        "objective_primary": "maximise P5 = share of counts with |pred/true-1| <= 5% (non-finite = miss, APE 100%); "
                             "ties (equal hits) -> lower MAPE; remaining ties -> smaller R_MIN, then smaller D_AGREE",
        "objective_secondary": "minimise MAPE; ties -> more hits within 5%; then smaller R_MIN, smaller D_AGREE",
        "grid": {"r_min": f"0 and the 0.25%..30% quantiles (step 0.25%) of rho over the fit rows ({len(r_grid)} values)",
                 "d_agree": "0.00..0.60 step 0.01 (61 values)"},
        "primary": {k: (float(prim[k]) if k != "hits5" else int(prim[k])) for k in prim.index},
        "secondary": {k: (float(sec[k]) if k != "hits5" else int(sec[k])) for k in sec.index},
        "components_on_fit_set": comp,
        "r_min_range_within_0.1pt_of_best_P5": [float(near.r_min.min()), float(near.r_min.max())],
        "d_agree_range_within_0.1pt_of_best_P5": [float(near.d_agree.min()), float(near.d_agree.max())],
        "best_fallback_only_rule_d0": {k: float(fb_only.iloc[0][k]) for k in ("r_min", "P5", "MAPE", "fallback_pct")},
        "labels_used": "synthetic generator labels only (exact); no human count, no legacy count",
    }
    G.to_csv(FIT_GRID, index=False)
    json.dump(out, open(FIT_JSON, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("primary", "secondary", "components_on_fit_set",
                                          "r_min_range_within_0.1pt_of_best_P5", "d_agree_range_within_0.1pt_of_best_P5",
                                          "best_fallback_only_rule_d0")}, indent=1))
    # per-condition view on the fit set at the chosen primary thresholds
    A2, o2 = synth_counts(s, R, r_min=float(prim.r_min), d=float(prim.d_agree))
    cond = np.repeat((s.kind + ":" + s.level.map(lambda x: f"{x:g}")).to_numpy()[:, None], 2, 1)
    tab = []
    for c in sorted(set(cond[:, 0])):
        m = cond == c
        tab.append(dict(cond=c, **{f"P5_{k}": 100 * (a[m] <= P5_TOL).mean() for k, a in A2.items()},
                        **{f"MAPE_{k}": a[m].mean() for k, a in A2.items()}))
    print(pd.DataFrame(tab).round(1).to_string(index=False))


# =================================================================================== held-out synthetic
def _gen(a):
    seed, variant, outdir = a
    import cv2
    cv2.setNumThreads(1)
    sys.path.insert(0, os.path.join(ROOT, "cnn"))
    import synth  # noqa: E402  (SYNTH_PROFILE=v2 is set above, read at import)
    rng = np.random.default_rng(seed)
    p = synth.sample_params(rng, seed)
    if variant == "legacygrid":
        r2 = np.random.default_rng(seed + 7_000_000)
        p.r_sensor = float(max(p.r_eff, r2.uniform(*GRID_LEGACY)))
        _, crop, lab = synth.generate(seed, params=p, max_crop_side=2048, frame_margin=0.1)
    else:
        _, crop, lab = synth.generate(seed, params=p, max_crop_side=1024, frame_margin=0.1)
    name = f"h{seed:07d}"
    cv2.imwrite(os.path.join(outdir, name + ".png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    lab["name"] = name
    lab["S"] = int(crop.shape[1])
    lab.pop("corners", None)
    return lab


def heldout_make(args):
    from multiprocessing import Pool
    for variant, s0 in HELDOUT_SEEDS.items():
        outdir = os.path.join(SCRATCH, variant)
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        with Pool(args.workers) as pool:
            labs = pool.map(_gen, [(s0 + k, variant, outdir) for k in range(args.n)], chunksize=2)
        df = pd.DataFrame(labs).sort_values("seed")
        df.to_csv(os.path.join(outdir, "labels.csv"), index=False)
        print(f"{variant}: {len(df)} crops -> {outdir} ({time.time() - t0:.0f}s); S median {df.S.median():.0f} "
              f"[{df.S.min()}, {df.S.max()}]", flush=True)


_C = None


def _counter():
    global _C
    if _C is None:
        _C = rule_module().load_frozen_counter()
    return _C


def _score_job(a):
    name, kind, level, variant = a
    import cv2
    cv2.setNumThreads(1)
    sys.path.insert(0, HERE)
    import degrade as D  # noqa: E402
    C = _counter()
    crop = cv2.imread(os.path.join(SCRATCH, variant, name + ".png"), cv2.IMREAD_COLOR)
    img = D.apply(crop, kind, level, seed=zlib.crc32(str(name).encode()) % 100003)
    t0 = time.time()
    try:
        res = C.estimate_many(img, ("ours", "gabor"), preproc="lcn", glare="mask")
    except Exception:  # e.g. log-Gabor on a tiny crop: one estimator at a time (as cascade_synth.py)
        res = {}
        for est in ("ours", "gabor"):
            try:
                res[est] = C.estimate(img, est, preproc="lcn", glare="mask", with_weave=False)
            except Exception as ex:  # noqa: BLE001
                res[est] = {"h_count": np.nan, "v_count": np.nan, "confidence": 0.0, "weave": "unknown",
                            "debug": {"error": repr(ex)[:200]}}
    o, g = res["ours"], res["gabor"]
    d = o["debug"]
    return dict(variant=variant, name=name, kind=kind, level=level, S_deg=int(img.shape[1]),
                N_work=int(min(max(img.shape[:2]), 1024)), ours_h=o["h_count"], ours_v=o["v_count"],
                ours_conf=o["confidence"], sharpness=d.get("sharpness"), glare_frac=d.get("glare_frac"),
                fusion="|".join(d["fusion"]) if d.get("fusion") else None,
                gabor_h=g["h_count"], gabor_v=g["v_count"], gabor_conf=g["confidence"],
                wall_s=round(time.time() - t0, 3))


def heldout_score(args):
    from multiprocessing import Pool
    sys.path.insert(0, HERE)
    import degrade as D  # noqa: E402
    conds = [("clean", 0)] + [(k, v) for k in ["gblur", "down", "jpeg", "glare", "corners"] for v in D.LEVELS[k]]
    jobs, labs = [], []
    for variant in HELDOUT_SEEDS:
        lab = pd.read_csv(os.path.join(SCRATCH, variant, "labels.csv"))
        lab["variant"] = variant
        labs.append(lab)
        jobs += [(n, k, v, variant) for n in lab.name for k, v in conds]
    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_score_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    lab = pd.concat(labs, ignore_index=True)
    keep = ["variant", "name", "seed", "weave", "weave_coarse", "h_count", "v_count", "r_eff", "r_sensor",
            "blur_sigma_px", "glare", "shadow", "S"]
    df = df.merge(lab[keep], on=["variant", "name"], how="left").sort_values(["variant", "name", "kind", "level"])
    df.to_csv(HELDOUT_CSV, index=False)
    import cv2
    meta = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_rows=int(len(df)),
                seeds={v: f"{s}..{s + args.n - 1}" for v, s in HELDOUT_SEEDS.items()}, conditions=conds,
                counter=os.path.relpath(rule_module().FROZEN_COUNTER, ROOT),
                counter_sha256=sha256(rule_module().FROZEN_COUNTER),
                conf_model_sha256=sha256(os.path.join(HERE, "conf_model.json")),
                degrade_sha256=sha256(os.path.join(HERE, "degrade.py")),
                synth_sha256=sha256(os.path.join(ROOT, "cnn", "synth.py")),
                numpy=np.__version__, cv2=cv2.__version__, wall_s=round(time.time() - t0, 1),
                prereg_sha256=sha256(PREREG) if os.path.exists(PREREG) else None)
    json.dump(meta, open(HELDOUT_CSV.replace(".csv", "_meta.json"), "w"), indent=1)
    print(f"wrote {HELDOUT_CSV}: {len(df)} rows ({time.time() - t0:.0f}s)")


# =================================================================================== predictions (label-free)
P150 = os.path.join(HERE, "predictions_ours_150.csv")
P_LEG_EST = os.path.join(RES, "legacy13_vlmcrops_estimators.csv")
P_LEG_DEG = os.path.join(RES, "legacy13_vlmcrops_degrade.csv")
P_LEG_DEGX = os.path.join(RES, "legacy13_vlmcrops_degrade_extra.csv")
P_STRESS = os.path.join(ROOT, "camera_aug", "results", "stress_legacy13_classical.csv")
LEG_CROPS = os.path.join(ROOT, "vlm", "crops", "legacy")
OUT150_WIDE = os.path.join(RES, "combined_v1_150_wide.csv")
OUT150_LONG = os.path.join(RES, "predictions_combined_v1_150.csv")
OUT_LEG_RECOUNT = os.path.join(RES, "legacy13_vlmcrops_combined_v1_recount.csv")
OUT_LEG_EST = os.path.join(RES, "legacy13_vlmcrops_estimators_combined_v1.csv")
OUT_LEG_DEG = os.path.join(RES, "legacy13_vlmcrops_degrade_combined_v1.csv")
OUT_LEG_DEGX = os.path.join(RES, "legacy13_vlmcrops_degrade_extra_combined_v1.csv")
OUT_STRESS = os.path.join(RES, "stress_legacy13_combined_v1.csv")
VARIANTS = (("primary", "combined_v1", "combined-v1"), ("secondary", "combined_v1_mape", "combined-v1-mape"))


def _recount_job(path):
    import cv2
    cv2.setNumThreads(1)
    R = rule_module()
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    t0 = time.time()
    out = R.estimate(img)
    out["image"] = os.path.splitext(os.path.basename(path))[0]
    out["S_in_w"], out["S_in_h"] = int(img.shape[1]), int(img.shape[0])
    out["md5"] = hashlib.md5(open(path, "rb").read()).hexdigest()
    out["wall_s"] = round(time.time() - t0, 2)
    return out


def legacy_recount(args):
    """End-to-end check of combined_frozen_v1.estimate() (frozen counter loaded by path) on the 20 canonical legacy
    crops: must reproduce the stored ours / log-Gabor counts; also gives ours' sharpness for all 20 crops."""
    from multiprocessing import Pool
    paths = sorted(os.path.join(LEG_CROPS, f) for f in os.listdir(LEG_CROPS) if f.endswith(".png"))
    with Pool(min(args.workers, len(paths))) as pool:
        rows = pool.map(_recount_job, paths)
    df = pd.DataFrame(rows).sort_values("image")
    e = pd.read_csv(P_LEG_EST)
    e = e[e.source == "vlmcrop"]
    for est, ch, cv in (("ours", "ours_h", "ours_v"), ("gabor", "gabor_h", "gabor_v")):
        s = e[e.estimator == est].set_index("image")
        dh = (df.set_index("image")[ch] - s.h).abs().max()
        dv = (df.set_index("image")[cv] - s.v).abs().max()
        print(f"recount vs stored {est}: max |dh| {dh:.2e}, max |dv| {dv:.2e} ({len(s)} crops)")
        df[f"{est}_matches_stored"] = bool(dh < 1e-6 and dv < 1e-6)
    df.to_csv(OUT_LEG_RECOUNT, index=False)
    print(f"wrote {OUT_LEG_RECOUNT}")


def _apply(R, frame, oh, ov, gh, gv, sharp, nwork):
    out = {}
    for var, est, key in VARIANTS:
        c = R.combine_arrays(frame[oh], frame[ov], frame[gh], frame[gv], frame[sharp] if sharp else np.nan,
                             frame[nwork], variant=var)
        out[est] = c
    return out


def predict(args):
    R = rule_module()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # ---------------------------------------------------------------- the 150 rectified crops (no labels exist)
    o = pd.read_csv(P150)
    o["n_work"] = np.minimum(o.S, R.N_WORK_MAX)
    c = _apply(R, o, "h_count", "v_count", "h_gabor", "v_gabor", "sharpness", "n_work")
    W = pd.DataFrame({"image_id": o.id})
    for var, est, key in VARIANTS:
        W[f"{key}__v"] = c[est]["h"]     # VLM wide convention (score150.py --extra-wide): __v = VERTICAL yarns = our h
        W[f"{key}__h"] = c[est]["v"]
        W[f"{key}__source"] = [f"{a}|{b}" for a, b in zip(c[est]["src_h"], c[est]["src_v"])]
    W["rho"] = c["combined_v1"]["rho"]
    W.sort_values("image_id").to_csv(OUT150_WIDE, index=False)
    Lg = pd.DataFrame({"id": o.id, "S": o.S, "n_work": o.n_work, "ours_h": o.h_count, "ours_v": o.v_count,
                       "ours_conf": o.confidence, "gabor_h": o.h_gabor, "gabor_v": o.v_gabor, "sharpness": o.sharpness,
                       "rho": c["combined_v1"]["rho"]})
    for var, est, key in VARIANTS:
        Lg[f"{est}_h"], Lg[f"{est}_v"] = c[est]["h"], c[est]["v"]
        Lg[f"{est}_src_h"], Lg[f"{est}_src_v"] = c[est]["src_h"], c[est]["src_v"]
    Lg["created"] = stamp
    Lg.sort_values("id").to_csv(OUT150_LONG, index=False)
    srcp = np.r_[c["combined_v1"]["src_h"], c["combined_v1"]["src_v"]]
    srcs = np.r_[c["combined_v1_mape"]["src_h"], c["combined_v1_mape"]["src_v"]]
    print(f"150: primary sources {pd.Series(srcp).value_counts().to_dict()}; secondary {pd.Series(srcs).value_counts().to_dict()}")
    # ---------------------------------------------------------------- legacy-13 canonical crops (clean)
    rc = pd.read_csv(OUT_LEG_RECOUNT)
    assert rc.ours_matches_stored.all() and rc.gabor_matches_stored.all(), "recount differs from stored estimates"
    c = _apply(R, rc, "ours_h", "ours_v", "gabor_h", "gabor_v", "sharpness", "n_work")
    rows = []
    for var, est, key in VARIANTS:
        for i, r in rc.reset_index(drop=True).iterrows():
            rows.append(dict(image=r.image, source="vlmcrop", S_in_w=r.S_in_w, S_in_h=r.S_in_h, estimator=est,
                             h=c[est]["h"][i], v=c[est]["v"][i], conf=np.nan, weave=np.nan,
                             fusion=f"{c[est]['src_h'][i]}|{c[est]['src_v'][i]}", g_min=np.nan, tile_support=np.nan,
                             runtime_s_shared=np.nan, wall_s_all_estimators=r.wall_s, rho=c[est]["rho"][i],
                             sharpness=r.sharpness, created=stamp))
    pd.DataFrame(rows).to_csv(OUT_LEG_EST, index=False)
    # ---------------------------------------------------------------- legacy-13 degradations (19 + 10 conditions)
    for src, dst in ((P_LEG_DEG, OUT_LEG_DEG), (P_LEG_DEGX, OUT_LEG_DEGX)):
        d = pd.read_csv(src)
        d["level"] = d.level.astype(float).round(4)
        k = ["image", "kind", "level", "tag"]
        ou = d[d.estimator == "ours"].set_index(k)
        gb = d[d.estimator == "gabor"].set_index(k)
        j = ou.join(gb[["h", "v"]], rsuffix="_gabor", how="inner").reset_index()
        j["n_work"] = np.minimum(j.S_deg, R.N_WORK_MAX)
        c = _apply(R, j, "h", "v", "h_gabor", "v_gabor", "sharpness", "n_work")
        rows = []
        cols = list(d.columns)
        for var, est, key in VARIANTS:
            x = j[[cc for cc in cols if cc in j.columns]].copy()
            x["estimator"] = est
            x["h"], x["v"] = c[est]["h"], c[est]["v"]
            for cc in ("conf", "weave", "g_min", "margin", "tile_support", "agree_prof", "lat_h", "lat_v", "runtime_s",
                       "wall_s", "md5"):
                if cc in x:
                    x[cc] = np.nan
            x["fusion"] = [f"{a}|{b}" for a, b in zip(c[est]["src_h"], c[est]["src_v"])]
            x["rho"] = c[est]["rho"]
            x["created"] = stamp
            rows.append(x)
        pd.concat(rows, ignore_index=True)[cols + ["rho", "created"]].to_csv(dst, index=False)
        print(f"wrote {dst}")
    # ---------------------------------------------------------------- PickGlass-Stress legacy-13 renders
    s = pd.read_csv(P_STRESS, dtype={"source_id": str})
    k = ["source_id", "condition", "rep"]
    ou = s[s.estimator == "ours"].set_index(k)
    gb = s[s.estimator == "gabor"].set_index(k)
    j = ou.join(gb[["h", "v"]], rsuffix="_gabor", how="inner").reset_index()
    j["n_work"] = np.minimum(j.S_in, R.N_WORK_MAX)
    # stored stress rows carry no sharpness: the primary (R_MIN = 0) does not need it; the secondary is left NaN
    cp = R.combine_arrays(j.h, j.v, j.h_gabor, j.v_gabor, np.full(len(j), np.nan), j.n_work, variant="primary")
    x = j[k + ["S_in"]].copy()
    x["estimator"] = "combined_v1"
    x["h"], x["v"] = cp["h"], cp["v"]
    x["source"] = [f"{a}|{b}" for a, b in zip(cp["src_h"], cp["src_v"])]
    x["created"] = stamp
    x.to_csv(OUT_STRESS, index=False)
    print(f"wrote {OUT150_WIDE}, {OUT150_LONG}, {OUT_LEG_EST}, {OUT_STRESS}")


# =================================================================================== pre-registration record
def prereg(args):
    R = rule_module()
    fitj = json.load(open(FIT_JSON))
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    h = lambda p: {"path": rel(p), "sha256": sha256(p)}  # noqa: E731
    counter = R.FROZEN_COUNTER
    doc = {
        "id": "PREREG_combined_v1",
        "title": "Pre-registered offline combined counter combined-v1 (PG-Lattice + log-Gabor bank), WVAQ @ WACV 2027",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statement_no_labels": (
            "When this record and every prediction file listed below were written, no human count of the 76 PickGlass "
            "scene representatives (or of any of the 150 files) existed: the single external annotator's (FL_SA) "
            "exports are due Mon 2026-10-12 18:00 IST (08:30 US Eastern). This job did not look for, open or use any "
            "annotation export (annotation_tool/data/saved, freelancer exports) or any propagated label. The only labels "
            "used were exact synthetic-generator labels (fit) and, AFTER this freeze and reported as post hoc, the 13 "
            "legacy (2019) hand counts, which had already been used during the development of the counters."),
        "motivation": (
            "Known pattern on the legacy-13 degradation study (post hoc): ours (PG-Lattice) is better clean-ish and under "
            "x8 down-scaling, strong glare and phone-camera profiles, but over-counts catastrophically under Gaussian "
            "defocus 1-4u (MAPE 50-246 %) where the log-Gabor bank degrades gracefully (10-40 %). The rule family "
            "therefore contains a defocus fallback to log-Gabor driven by the resolvability ratio rho of the cascade "
            "(supp. S14), plus an agreement term; thresholds were left to the synthetic fit."),
        "rule": {
            "file": h(RULE_PY),
            "rule_id": R.RULE_ID,
            "definition": (
                "Per photo: rho = sharpness / (4 sin^2(pi f))^2, f = max(ours_h, ours_v) / n_work, n_work = min(max(H, W), "
                "1024); if rho < R_MIN -> log-Gabor on both axes; else per axis: if |ln(ours/gabor)| <= D_AGREE -> "
                "sqrt(ours * gabor), else ours. A non-finite component is replaced by the other; both non-finite -> NaN. "
                "Inputs are the frozen counter's ours and log-Gabor outputs on the same pixels (preproc lcn, glare mask)."),
            "thresholds_primary": {"R_MIN": R.R_MIN, "D_AGREE": R.D_AGREE},
            "thresholds_secondary": {"rule_id": R.RULE_ID_SEC, "R_MIN": R.R_MIN_SEC, "D_AGREE": R.D_AGREE_SEC,
                                     "status": "sensitivity analysis only (MAPE objective); not for Tab. 1"},
            "effective_primary_rule_in_words": (
                "The P5 objective switched the defocus fallback off (R_MIN = 0): combined-v1 = per axis, the geometric "
                "mean of ours and log-Gabor when they agree within a factor e^0.18 = 1.197, otherwise ours."),
            "deterministic": True, "label_free_at_inference": True, "n_thresholds": 2,
        },
        "objective": {
            "primary": fitj["objective_primary"],
            "secondary": fitj["objective_secondary"],
            "grid": fitj["grid"],
            "declared_before_fitting": True,
            "source": "objective fixed by the task brief ('maximise share of counts within 5 % (P5), tie-break MAPE')",
        },
        "fit": {
            "script": h(os.path.join(HERE, "combined_v1.py")),
            "fit_json": h(FIT_JSON), "fit_grid_csv": h(FIT_GRID), "fitted_at": fitj["fitted_at"],
            "synthetic_data": {
                "S-cal-legacygrid": dict(h(SCAL["legacygrid"]), seeds="930000..930199",
                                         generator="cnn/synth.py v2, r_sensor ~ U[1630, 1940] px/inch, crop not capped",
                                         scored_on="M2 Max (numpy 1.x), methods/cascade_synth.py, Oct 7"),
                "S-cal-default": dict(h(SCAL["default"]), seeds="930000..930199",
                                      generator="cnn/synth.py v2 defaults, crop <= 1024 px",
                                      scored_on="gpu-server (numpy 2.2.6, cv2 5.0.0), methods/cascade_synth.py, Oct 7"),
                "conditions": fitj["conditions"], "n_crops": fitj["n_crops"], "n_counts": fitj["n_counts"],
                "labels": "exact generator labels (h_count, v_count)",
                "note": "S-cal was produced on Oct 7 for the cascade (paper/tools/cascade_tradeoff.py); it had never been "
                        "used to choose anything about ours vs log-Gabor. Same seeds in both variants (different sensor "
                        "grids), so the two variants are not independent fabrics."},
            "result_primary": fitj["primary"], "result_secondary": fitj["secondary"],
            "components_on_fit_set": fitj["components_on_fit_set"],
            "objective_flatness": {"r_min_within_0.1pt": fitj["r_min_range_within_0.1pt_of_best_P5"],
                                   "d_agree_within_0.1pt": fitj["d_agree_range_within_0.1pt_of_best_P5"]},
        },
        "frozen_dependencies": {
            "counter": dict(h(counter), md5=hashlib.md5(open(counter, "rb").read()).hexdigest(),
                            note="byte-identical, md5 3d0520d6... as recorded in paper/PLAN.md; imported by path, "
                                 "never modified"),
            "conf_model": h(os.path.join(HERE, "conf_model.json")),
            "degrade": h(os.path.join(HERE, "degrade.py")),
            "synth_generator": h(os.path.join(ROOT, "cnn", "synth.py")),
            "scorer_score150": dict(h(os.path.join(PAPER, "tools", "score150.py")),
                                    md5=hashlib.md5(open(os.path.join(PAPER, "tools", "score150.py"), "rb").read()).hexdigest()),
        },
        "prediction_inputs": {
            "real150": dict(h(P150), note="methods/run_ours_150.py (methods/counting.py, whose estimators are identical "
                                          "to the frozen counter; the two files differ only in guide_positions())"),
            "legacy13_clean": dict(h(P_LEG_EST), recount=h(OUT_LEG_RECOUNT),
                                   note="recount = combined_frozen_v1.estimate() on vlm/crops/legacy/*.png; reproduces "
                                        "the stored ours and log-Gabor counts exactly"),
            "legacy13_degradations": h(P_LEG_DEG), "legacy13_degradations_extra": h(P_LEG_DEGX),
            "legacy13_pickglass_stress": h(P_STRESS),
        },
        "predictions": {
            "real150_wide_for_score150": dict(h(OUT150_WIDE), keys=["combined-v1", "combined-v1-mape"],
                                              convention="<key>__v = VERTICAL yarns = our h (VLM wide format)"),
            "real150_long": h(OUT150_LONG),
            "legacy13_clean": h(OUT_LEG_EST),
            "legacy13_degradations": h(OUT_LEG_DEG), "legacy13_degradations_extra": h(OUT_LEG_DEGX),
            "legacy13_pickglass_stress_primary_only": h(OUT_STRESS),
        },
        "heldout_test": {
            "status": "declared here, run after this record was written (methods/combined_v1.py --heldout-score, "
                      "--eval-synth); crops rendered before the freeze but never scored before it",
            "data": {"H-legacygrid": "fresh cnn/synth.py v2 seeds 970000..970199, legacy-like sensor grid (as S-cal)",
                     "H-default": "fresh seeds 975000..975199, generator defaults (crop <= 1024 px)",
                     "conditions": "the same 19 as S-cal (clean; gblur 1,2,3,4,6u; down x2,4,8; jpeg 90,50,20; glare "
                                   "5,10,20,30 %; corners 2,5,10 %), methods/degrade.py, seed crc32(name) % 100003",
                     "counter": "methods/results/counting_frozen_final.py by path (ours + log-Gabor)",
                     "seed_check": "no other study uses 970000+ or 975000+ (training 0..102399, synth_val 900000+, "
                                   "synth_val_v2 910000+, S-cal 930000+, guard 960000+/965000+)"},
            "primary_metric": "P5 over all held-out counts (both sets, all 19 conditions, 2 counts per row)",
            "comparisons": ["combined-v1 - ours", "combined-v1 - log-Gabor"],
            "inference": "paired cluster bootstrap over synthetic crops (both counts and all 19 conditions of a crop "
                         "resampled together), 10 000 resamples, seed 20261011; 95 % percentile CI of the P5 difference "
                         "(percentage points) and of the MAPE difference",
            "decision_rule": "combined-v1 is 'better than both components' iff its held-out P5 exceeds both ours' and "
                             "log-Gabor's AND both 95 % CIs of the P5 difference exclude 0. Otherwise the record says so "
                             "and recommends NOT adding combined-v1 to Tab. 1.",
            "also_reported": "MAPE, P10, per condition, per generator variant, clean only; the secondary variant",
        },
        "label_day_plan": {
            "scoring": "tools/score150.py ... --extra-wide ../methods/results/combined_v1_150_wide.csv::combined-v1 "
                       "(per scene, n = labelled representatives; same metrics as every Tab. 1 row: MAE, MAPE [CI], w5, "
                       "w10, both5, bias_med)",
            "paired": "label_day.py paired(): per-scene mean APE, combined-v1 vs Ours and vs Log-Gabor bank (scene "
                      "bootstrap 10 000, Wilcoxon)",
            "condition": "only if the held-out synthetic test passes; the secondary variant only in the supplement",
            "no_refit": "thresholds are never refitted on human labels",
        },
        "legacy_disclosure": "Legacy-13 results (clean, degradations, PickGlass-Stress) are POST HOC: the 13 hand counts "
                             "were used during development of ours and the defocus pattern motivated this rule family.",
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
    }
    if os.path.exists(PREREG):
        raise SystemExit(f"{PREREG} exists: a pre-registration is never overwritten (move it to a backup first)")
    json.dump(doc, open(PREREG, "w"), indent=1)
    print(f"wrote {PREREG} (sha256 {sha256(PREREG)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--heldout-make", action="store_true")
    ap.add_argument("--heldout-score", action="store_true")
    ap.add_argument("--legacy-recount", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.fit:
        fit(a)
    if a.legacy_recount:
        legacy_recount(a)
    if a.predict:
        predict(a)
    if a.prereg:
        prereg(a)
    if a.heldout_make:
        heldout_make(a)
    if a.heldout_score:
        heldout_score(a)
