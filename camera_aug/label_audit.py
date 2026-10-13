"""label_audit.py -- label-safety audit of camera_sim.py (is the TPI label transform really the identity?)

Protocol (the project's audit, cnn/augment_optics_verify.py / verify_augment.py): apply the transform,
run the FROZEN counter "ours" (methods/results/counting_frozen_final.py, estimate_many(..., ["ours"],
lcn, mask)), and compare the measured per-inch count with the PREDICTED label change against the
no-op re-encode baseline.  Per image and axis
        drift = observed / (predicted factor x clean observed) - 1,
predicted factor = camera_sim's label_factor_h / _v (1 for every stage except radial distortion).
A failed run (NaN) counts as 100 % drift.  Only values whose expected count lies inside ours' search
band [1.1 x 18, 0.40 x N_work] (N_work = min(S_out, 1024)) enter the PASS statistics (observability);
the px-per-yarn curves use every value.

PASS rule (fixed before running; = verify_augment.py): with floor = the no-op baseline distribution
(noop_store: crop-store JPEG q92 re-encode; noop_q75: q75 resave + store, the last encoder of our
originals; the larger of the two is used), a condition is within-noise iff
   median |drift| <= floor + 0.5 pp,  p90 |drift| <= floor + 2 pp,
   P(|drift| > 10 %) <= floor + 2 pp,  |median signed drift| <= 1 %.
Statistics are computed per image and summarised over all 150 crops and over SCENES (76 groups of
camera_aug/originals_provenance.json; CIs = cluster bootstrap over scenes).

PARTS
  real150   150 crops x {clean, noop_store, noop_q75} + ISOLATED stages of budget_binned_12mp (each
            stage alone at the profile's sampled parameters; 42 % of the market mix), the 'cfa' stage
            (demosaic / quad remosaic) of the other 4 profiles, low-light noise / denoise / motion, and
            the 4 transports alone.
            (The FULL pipeline per profile = the real150 stress renders, counted by
            eval_stress_classical.py --set real150 --ests ours.)
  sweep     px-per-yarn sweep on one full-resolution crop per scene (+ legacy13, GT counts):
            target side = ppt x count of the finer axis, ppt in PPTS, routes
            'area' (INTER_AREA only), 'sensor' (camera_sim sample + cfa: lens PSF, pixel aperture,
            rotated grid, Bayer + bilinear demosaic, no noise / ISP), 'phone' (full budget pipeline
            at that px/inch, k1 = 0, zoom 1), 'messaging' (budget camera at its own px/inch, then a
            messaging-app resize -- cv2 bilinear without anti-aliasing -- down to the target and
            JPEG q78).  The threshold = smallest ppt whose bin passes the PASS rule.
  gratings  analytic plaid gratings (luma and isoluminant chroma) through every profile and every
            isolated stage, frequency measured by zero-padded FFT (no counter involved) -> exact
            check of the geometry / label factor, and of CFA aliasing.

    <venv2>/bin/python label_audit.py run --part real150 --workers 6
    <venv2>/bin/python label_audit.py run --part sweep --workers 6
    <venv2>/bin/python label_audit.py gratings
    <venv2>/bin/python label_audit.py summarise
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import camera_sim as CS  # noqa: E402
import eval_stress_classical as EC  # noqa: E402

RES = os.path.join(HERE, "results")
CROPS = os.path.join(ROOT, "annotation_tool", "data", "crops")
LEGACY = os.path.join(ROOT, "vlm", "crops", "legacy")
NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
PPTS = (1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0)
R_MIN = 18.0
SYNTH_FILES = {131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144}
ISO_STAGES = [s for s in CS.STAGES if s != "transport"]


# =============================================================================================
# conditions
# =============================================================================================
def real150_conditions():
    C = [dict(cond="clean"), dict(cond="noop_store", noop="store"), dict(cond="noop_q75", noop="q75")]
    for st in ISO_STAGES:
        C.append(dict(cond=f"budget_binned_12mp:{st}", profile="budget_binned_12mp", stages=[st], family="stage"))
    for prof in ("budget_fullres_50mp", "midrange_binned", "flagship", "legacy_16mp_2019"):
        C.append(dict(cond=f"{prof}:cfa", profile=prof, stages=["cfa"], family="stage"))
    for st in ("noise", "denoise", "motion"):
        C.append(dict(cond=f"budget_lowlight:{st}", profile="budget_binned_12mp", overlay="low_light", stages=[st],
                      family="stage"))
    C.append(dict(cond="budget_lowlight:noise+denoise", profile="budget_binned_12mp", overlay="low_light",
                  stages=["noise", "denoise"], family="stage"))
    for tr in ("messaging_standard", "messaging_hd", "resave_q75", "resave_q75_x0.25"):
        C.append(dict(cond=f"transport:{tr}", profile="budget_binned_12mp" if tr.startswith("messaging") else
                      "legacy_16mp_2019", transport=tr, stages=["transport"], family="transport"))
    return C


def _params(c, sid, tag="audit"):
    P = CS.sample_params(c["profile"], CS.stable_seed(tag, sid, c["profile"]), overlay=c.get("overlay"),
                         transport=c.get("transport", "none"))
    if c["stages"] == ["zoom"]:   # isolated zoom: force a zoom (sampled zoom is 1 for 80 % of captures)
        P["zoom"] = float(np.random.default_rng(CS.stable_seed("zoom", sid)).uniform(1.2, 1.6))
    return P


def count(img):
    r = EC.frozen().estimate_many(img, ["ours"], preproc="lcn", glare="mask")["ours"]
    return float(r["h_count"]), float(r["v_count"]), float(r["confidence"])


def _real150_job(a):
    sid, path, c = a
    cv2.setNumThreads(1)
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    t0 = time.time()
    info = dict(S_out=src.shape[0], label_factor_h=1.0, label_factor_v=1.0)
    if c["cond"] == "clean":
        out = src
    elif "noop" in c:
        out, _ = CS.noop_store(src, q75_first=c["noop"] == "q75")
    else:
        P = _params(c, sid)
        out, info = CS.render(src, P, enabled=set(c["stages"]))
    h, v, conf = count(out)
    return dict(source_id=sid, cond=c["cond"], family=c.get("family", "baseline"), profile=c.get("profile"),
                S_src=src.shape[0], S_out=int(info["S_out"]), f_h=info["label_factor_h"], f_v=info["label_factor_v"],
                h=h, v=v, conf=conf, wall_s=round(time.time() - t0, 2))


# =============================================================================================
# px-per-yarn sweep
# =============================================================================================
def sweep_sources():
    prov = json.load(open(os.path.join(HERE, "originals_provenance.json")))
    man = {e["id"]: e for e in json.load(open(os.path.join(ROOT, "annotation_tool", "data", "manifest.json")))["images"]}
    ours = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    src = []
    for gi, g in enumerate(prov["groups"]):
        hi = [i for i in g if max(man[i]["orig_w"], man[i]["orig_h"]) >= 4000 and i not in SYNTH_FILES]
        if not hi:
            continue
        i = min(hi)
        src.append(dict(set="real150", source_id=str(i), scene=gi, path=os.path.join(CROPS, f"{i}.jpg"),
                        count_fine=float(max(ours.loc[i, "h_count"], ours.loc[i, "v_count"])),
                        S=int(man[i]["rect_size"][0])))
    gt = json.load(open(os.path.join(ROOT, "cnn", "aug_optics", "legacy13_gt.json")))
    for n in NAMES13:
        src.append(dict(set="legacy13", source_id=n, scene=f"L{n}", path=os.path.join(LEGACY, f"{n}.png"),
                        count_fine=float(max(gt[n]["warp"], gt[n]["weft"])),
                        S=int(cv2.imread(os.path.join(LEGACY, f"{n}.png")).shape[0])))
    return src


def _sweep_job(a):
    s, route, ppt = a
    cv2.setNumThreads(1)
    src = cv2.imread(s["path"], cv2.IMREAD_COLOR)
    S = src.shape[0]
    side = int(round(ppt * s["count_fine"]))
    t0 = time.time()
    if route == "clean":
        out = src
    elif route == "area":
        out = cv2.resize(src, (side, side), interpolation=cv2.INTER_AREA)
    else:
        P = CS.sample_params("budget_binned_12mp", CS.stable_seed("sweep", s["source_id"]))
        P["k1"], P["zoom"] = 0.0, 1.0
        if route in ("sensor", "phone"):
            P["ppi_profile"] = float(side)
            en = {"sample", "cfa"} if route == "sensor" else CS.ALL_STAGES - {"distortion"}
            out, _ = CS.render(src, P, enabled=en)
        elif route == "messaging":
            out, _ = CS.render(src, P, enabled=CS.ALL_STAGES - {"distortion", "transport"})
            out = CS.resize(out, (side, side), "bilinear")
            out, _ = CS.jpeg_roundtrip(out, 78, "420", progressive=True)
            out, _ = CS.jpeg_roundtrip(out, CS.STORE_Q, "420")
        else:
            raise ValueError(route)
    h, v, conf = count(out)
    return dict(set=s["set"], source_id=s["source_id"], scene=s["scene"], route=route, ppt_target=ppt,
                count_fine=s["count_fine"], S_src=S, S_out=out.shape[0], ppt_out=out.shape[0] / s["count_fine"],
                h=h, v=v, conf=conf, wall_s=round(time.time() - t0, 2))


# =============================================================================================
# runner
# =============================================================================================
def run(part, workers):
    if part == "real150":
        out = os.path.join(RES, "audit_real150_raw.csv")
        conds = real150_conditions()
        jobs = [(str(i), os.path.join(CROPS, f"{i}.jpg"), c) for c in conds for i in range(150)]
        key = lambda r: (str(r["source_id"]), r["cond"])  # noqa: E731
        jkey = lambda j: (j[0], j[2]["cond"])  # noqa: E731
        fn = _real150_job
    elif part == "sweep":
        out = os.path.join(RES, "audit_sweep_raw.csv")
        srcs = sweep_sources()
        jobs = [(s, "clean", 0.0) for s in srcs]
        jobs += [(s, r, p) for p in PPTS for r in ("area", "sensor", "phone", "messaging") for s in srcs
                 if p * s["count_fine"] < 0.95 * s["S"]]
        key = lambda r: (str(r["source_id"]), r["route"], float(r["ppt_target"]))  # noqa: E731
        jkey = lambda j: (j[0]["source_id"], j[1], float(j[2]))  # noqa: E731
        fn = _sweep_job
    else:
        raise ValueError(part)
    old = pd.read_csv(out, dtype={"source_id": str}) if os.path.exists(out) else None
    if old is not None:
        done = {key(r) for r in old.to_dict("records")}
        jobs = [j for j in jobs if jkey(j) not in done]
    # big crops first (better packing), then the rest
    print(f"{part}: {len(jobs)} jobs, workers={workers}", flush=True)
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                d = pd.DataFrame(rows)
                if old is not None:
                    d = pd.concat([old, d], ignore_index=True)
                d.to_csv(out + ".tmp", index=False)
                os.replace(out + ".tmp", out)
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print("wrote", out)


def _gray(img, channel):
    x = img.astype(np.float64)
    if channel == "luma":
        return 0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]
    if channel == "r":
        return x[..., 2]
    if channel == "pca":                                   # first principal component (as ours' pca_gray)
        v = x.reshape(-1, 3) - x.reshape(-1, 3).mean(0)
        w = np.linalg.eigh(v.T @ v)[1][:, -1]
        return (v @ w).reshape(x.shape[:2])
    raise ValueError(channel)


def measure_cycles(img, f0, axis="h", channel="luma", a=0.1, k1=0.0):
    """Crossing count by phase unwrapping: band-pass every row (axis 'h') or column ('v') around f0
    (0.6-1.4 f0, analytic signal), count cycles between x = a S and x = (1 - a) S, average over the
    rows; also the PREDICTED count from the corner-rectified radial map of camera_sim (window units).
    Returns (measured, predicted) cycles per inch of the central (1 - 2a) span, both / (1 - 2a)."""
    g = _gray(img, channel)
    if axis == "v":
        g = g.T
    n = g.shape[1]
    F = np.fft.fft(g - g.mean(1, keepdims=True), axis=1)
    f = np.fft.fftfreq(n, 1.0 / n)
    F[:, ~((f > 0.6 * f0) & (f < 1.4 * f0))] = 0
    ph = np.unwrap(np.angle(np.fft.ifft(F, axis=1)), axis=1)
    i0, i1 = int(round(a * n)), int(round((1 - a) * n)) - 1
    meas = float(np.mean(ph[:, i1] - ph[:, i0]) / (2 * np.pi))
    w0, w1 = (i0 + 0.5) / n, (i1 + 0.5) / n
    wy = (np.arange(n) + 0.5) / n
    G = 1.0 + 2.0 * k1

    def D(wx):
        r2 = ((wx - 0.5) ** 2 + (wy - 0.5) ** 2) / 0.25
        return 0.5 + (wx - 0.5) * (1.0 + k1 * r2) / G

    pred = float(f0 * np.mean(D(w1) - D(w0)))
    span = w1 - w0
    return meas / span, pred / span


def gratings(out=os.path.join(RES, "audit_gratings.csv")):
    """Analytic gratings: geometry + label factor (no counter), and CFA aliasing vs px per cycle."""
    rows = []
    S = 1000
    for kind in ("luma", "chroma"):
        for fh, fv in ((61.0, 47.0), (113.0, 89.0)):
            scene = CS.grating_scene(S, fh, fv, kind)
            for prof in CS.PROFILE_NAMES:
                P = CS.sample_params(prof, CS.stable_seed("grating", kind, fh, prof))
                for st in ("full",) + tuple(CS.STAGES):
                    en = CS.ALL_STAGES if st == "full" else {st}
                    o, inf = CS.render(None, P, enabled=en, src_is_linear_rgb=scene)
                    k1 = P["k1"] if st in ("full", "distortion") else 0.0
                    for ax, f0 in (("h", fh), ("v", fv)):
                        fac = inf["label_factor_" + ax]
                        for ch in (("luma", "pca") if kind == "luma" else ("r", "pca")):
                            fm, fp = measure_cycles(o, f0, ax, ch, k1=k1)
                            rows.append(dict(test="stage", kind=kind, channel=ch, f_true=f0, axis=ax, profile=prof,
                                             stage=st, S_out=inf["S_out"], ppc=inf["S_out"] / f0, f_meas=fm,
                                             f_pred_central=fp, label_factor=fac, k1=P["k1"],
                                             err_identity=fm / f0 - 1, err_pred=fm / fp - 1,
                                             f_fft_peak=CS.measure_freq(o, ax) if ch == "luma" else np.nan))
        # CFA aliasing: px per cycle sweep through 'sample' + 'cfa' (bilinear) vs 'sample' only
        scene = CS.grating_scene(1600, 100.0, 100.0, kind)
        for ppc in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
            for route, en in (("sample_only", {"sample"}), ("sample+cfa", {"sample", "cfa"})):
                for dem in ("bilinear", "malvar_he_cutler", "edge_aware"):
                    if route == "sample_only" and dem != "bilinear":
                        continue
                    P = CS.sample_params("budget_binned_12mp", CS.stable_seed("alias", kind, ppc))
                    P.update(ppi_profile=100.0 * ppc, theta_deg=0.0, demosaic=dem, lens_sigma_px=0.4)
                    o, inf = CS.render(None, P, enabled=en, src_is_linear_rgb=scene)
                    fm = CS.measure_freq(o, "h")
                    fr = CS.measure_freq(o, "h", channel="r")
                    fp = CS.measure_freq(cv2.merge([np.clip(_gray(o, "pca") - _gray(o, "pca").min(), 0, 255)
                                                    .astype(np.float32)] * 3), "h", channel="g")
                    rows.append(dict(test="alias", kind=kind, f_true=100.0, axis="h", route=route, demosaic=dem,
                                     ppc=inf["S_out"] / 100.0, S_out=inf["S_out"], f_meas=fm, f_meas_r=fr,
                                     f_meas_pca=fp, err_identity=fm / 100.0 - 1, err_r=fr / 100.0 - 1,
                                     err_pca=fp / 100.0 - 1))
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    st = d[d.test == "stage"]
    print("max |measured / predicted - 1| (phase-unwrapped crossing count, central 80 %), by stage:")
    print(st.groupby(["kind", "channel", "stage"]).err_pred.agg(lambda s: np.abs(s).max()).unstack([0, 1])
          .round(4).to_string())
    print("max |error| if the distortion factor were ignored:",
          st[st.stage == "distortion"].err_identity.abs().max().round(4))
    al = d[d.test == "alias"]
    for col in ("err_identity", "err_r", "err_pca"):
        print(f"aliasing test, {col} (FFT peak; 0 = correct frequency):")
        print(al.pivot_table(index="ppc", columns=["kind", "route", "demosaic"], values=col).round(3).to_string())
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", required=True, choices=["real150", "sweep"])
    r.add_argument("--workers", type=int, default=6)
    sub.add_parser("gratings")
    sub.add_parser("summarise")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.part, a.workers)
    elif a.cmd == "gratings":
        gratings()
    elif a.cmd == "summarise":
        import audit_summarise
        audit_summarise.main()
