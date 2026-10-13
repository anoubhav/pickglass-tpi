"""ruler_eval.py -- validate ruler_calib.py on the 150 photos WITHOUT thread labels.

Stages (each cached under methods/results/ruler/):
  calib     run ruler_calib.calibrate on every original photo (r10 fixed, W measured) plus a
            free fit (r10 and W free) for the glass constants; compare with the human window
            polygon: corner error in mm (polygon scale), side/scale error in %.
  count     run the frozen counter ('ours') on the ruler crops and compare with the
            GT-corner crops (agreement, downstream stability).
  perturb   corner-perturbation robustness (degrade.perturb_corners, same seeds as ablate.py):
            counter on perturbed-corner crops vs. on ruler crops initialised from the
            perturbed corners (init_corners mode).

    python ruler_eval.py calib [--workers 6]
    python ruler_eval.py count [--workers 6]
    python ruler_eval.py perturb [--workers 6]
  photodeg  calibration on degraded ORIGINAL photos (downscale x2/x4, defocus 1-3 u, JPEG 20,
            noise 20): success and corner error.
    python ruler_eval.py photodeg [--workers 6]
  blur      ruler blur meter (tick width) on photos defocused by sigma = 0..6 u, to be joined with
            the counter's self-consistency at the same sigma (results/all150_degrade.csv).
    python ruler_eval.py blur [--workers 6]
Then: python ruler_summary.py  (numbers, CIs, paired tests, drop-in table).
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import os
import pickle
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
sys.path.insert(0, os.path.join(ROOT, "common"))
import rectify as R  # noqa: E402
import ruler_calib as RC  # noqa: E402

OUT = os.path.join(HERE, "results", "ruler")
os.makedirs(OUT, exist_ok=True)
INCH = 25.4


def _corner_metrics(corners, gt):
    """Corner error in mm using the human polygon's own scale (side = 25.4 mm by the dataset
    convention), plus side-length ratio."""
    side_gt = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1)
    mm_per_px = INCH / side_gt.mean()
    err = np.linalg.norm(corners - gt, axis=1) * mm_per_px
    side = np.linalg.norm(corners - np.roll(corners, -1, 0), axis=1)
    return err, side / side_gt


def _calib_job(i):
    t0 = time.time()
    img = R.load_image(i)
    gt = R.load_corners(i)
    try:
        cal = RC.calibrate(img)
    except Exception as e:  # noqa: BLE001
        import traceback
        cal = {"ok": False, "fail": "exception:" + repr(e)[:120], "n_segments": -1, "n_rows_cand": -1}
        traceback.print_exc()
    row = {"id": i, "W_img": img.shape[1], "H_img": img.shape[0], "group": "high" if max(img.shape[:2]) >= 3000 else "low",
           "ok": cal["ok"], "fail": cal.get("fail"), "n_segments": cal["n_segments"], "n_rows_cand": cal["n_rows_cand"]}
    keep = {}
    if cal["ok"]:
        err, ratio = _corner_metrics(cal["corners"], gt)
        q = cal["quality"]
        row.update({f"cerr{j}": e for j, e in enumerate(err)})
        row.update({"cerr_mean": err.mean(), "cerr_max": err.max(), "side_ratio_mean": ratio.mean(),
                    "W_mm": cal["W_mm"], "rms_mm": q["rms_mm"], "n_ticks": q["n_ticks"], "coverage": q["coverage"],
                    "W_dev": q["W_dev"], "layout_ok": q["layout_ok"], "qscore": q["score"],
                    "blur_u": cal.get("blur_u"), "blur_px": cal.get("blur_px")})
        for sd in RC.SIDES:
            o = cal["obs"].get(sd)
            row[f"{sd}_scale"] = o["scale"] if o else None
            row[f"{sd}_n"] = len(o["k"]) if o else 0
            row[f"{sd}_ppmm"] = cal["px_per_mm"].get(sd)
        # px/mm of the human polygon along each side (side px / 25.4) vs ruler px/mm
        side_gt = np.linalg.norm(np.roll(gt, -1, 0) - gt, axis=1)  # TL-TR, TR-BR, BR-BL, BL-TL
        for sd, L in zip(("top", "right", "bottom", "left"), side_gt):
            row[f"{sd}_ppmm_gtpoly"] = L / INCH
        # free fit: glass constants
        try:
            ff = RC.fit_metric(cal["obs"], cal["corners_lines"], r10=None, W=None)
            row["W_free"], row["r10_free"] = ff["W"], ff["r10"]
            row["rms_free"] = float(np.sqrt(np.mean(np.concatenate(list(ff["res_mm"].values())) ** 2)))
        except Exception as e:  # noqa: BLE001
            row["free_err"] = repr(e)
        # GT-frame measurements of the ruler (tick pitch in polygon-mm units)
        dst = np.float32([[0, 0], [INCH, 0], [INCH, INCH], [0, INCH]])
        Hg = cv2.getPerspectiveTransform(gt.astype(np.float32), dst).astype(np.float64)
        for sd, o in cal["obs"].items():
            ax = RC.EDGE_AX[sd][0]
            u = RC._homog_apply(Hg, o["feet"])[:, ax]
            if len(np.unique(o["k"])) >= 4:
                p, c = np.polyfit(o["k"], u, 1)
                row[f"{sd}_pitch_gtmm"] = abs(p)
        keep = {"corners": cal["corners"], "corners_lines": cal["corners_lines"], "H": cal["H_img2mm"],
                "W_mm": cal["W_mm"], "obs": {sd: {k: v for k, v in o.items()} for sd, o in cal["obs"].items()},
                "quality": cal["quality"]}
    row["runtime_s"] = time.time() - t0
    return row, keep


def stage_calib(workers):
    with Pool(workers) as p:
        res = p.map(_calib_job, range(R.N_IMAGES), chunksize=1)
    rows = [r for r, _ in res]
    df = pd.DataFrame(rows).sort_values("id")
    df.to_csv(os.path.join(OUT, "calib150.csv"), index=False)
    with open(os.path.join(OUT, "calib150.pkl"), "wb") as f:
        pickle.dump({r["id"]: k for r, k in res}, f)
    print(summarise_calib(df))


def summarise_calib(df):
    lines = []
    for g, d in [("all", df)] + list(df.groupby("group")):
        ok = d[d.ok]
        lines.append(f"[{g}] n={len(d)} ok={len(ok)} ({100 * len(ok) / len(d):.0f}%)  "
                     f"corner err mm: median {ok.cerr_mean.median():.3f}  p90(max) {ok.cerr_max.quantile(.9):.3f}  "
                     f"max<=0.5mm {100 * (ok.cerr_max <= 0.5).mean():.0f}%  "
                     f"W median {ok.W_mm.median():.3f} IQR {ok.W_mm.quantile(.25):.3f}-{ok.W_mm.quantile(.75):.3f}  "
                     f"r10_free median {ok.r10_free.median():.4f}  rms {ok.rms_mm.median():.3f}")
    return "\n".join(lines)


# ============================================================================ counting
def _counter():
    import counting as C  # same estimator code as results/counting_frozen_final.py (only
    return C              # guide_positions differs); loads methods/conf_model.json


def _load_cal():
    with open(os.path.join(OUT, "calib150.pkl"), "rb") as f:
        return pickle.load(f)


def _crop_from(photo, corners, S):
    import degrade as D
    return D.rectify_with_corners(photo, corners, S)


def _count(C, crop):
    e = C.estimate(crop, "ours")
    return e["h_count"], e["v_count"], e["confidence"]


def _count_job(args):
    i, k = args
    C = _counter()
    photo = R.load_image(i)
    gt = R.load_corners(i)
    S_gt = R.default_out_size(gt)
    row = {"id": i, "S_gt": S_gt}
    row["h_gt"], row["v_gt"], row["conf_gt"] = _count(C, R.rectify(i, img=photo))
    if k:
        S_r = R.default_out_size(k["corners"])
        cal = {"corners": k["corners"], "W_mm": k["W_mm"], "H_img2mm": k["H"]}
        crop = RC.rectify_metric(photo, cal, S=S_r, mode="window")
        row["S_ruler"] = S_r
        row["h_ruler"], row["v_ruler"], row["conf_ruler"] = _count(C, crop)
        row["qscore"] = k["quality"]["score"]
        row["W_mm"] = k["W_mm"]
    return row


def stage_count(workers):
    K = _load_cal()
    with Pool(workers) as p:
        rows = p.map(_count_job, [(i, K.get(i)) for i in range(R.N_IMAGES)], chunksize=1)
    df = pd.DataFrame(rows).sort_values("id")
    df["drift_h"] = df.h_ruler / df.h_gt - 1
    df["drift_v"] = df.v_ruler / df.v_gt - 1
    df["absdrift"] = np.fmax(df.drift_h.abs(), df.drift_v.abs())
    df.to_csv(os.path.join(OUT, "count150.csv"), index=False)
    ref = pd.read_csv(os.path.join(HERE, "predictions_ours_150.csv"))[["id", "h_count", "v_count", "confidence"]]
    m = df.merge(ref, on="id")
    print("GT-crop reproduction vs predictions_ours_150.csv: max |dh| %.2e, |dv| %.2e, |dconf| %.2e" % (
        (m.h_gt - m.h_count).abs().max(), (m.v_gt - m.v_count).abs().max(), (m.conf_gt - m.confidence).abs().max()))
    ok = df.absdrift.notna()
    print(f"ruler vs GT crops: n={ok.sum()} median |drift| {100 * df.absdrift.median():.2f}%  "
          f"within 2% {100 * (df.absdrift <= .02).mean():.1f}%  within 5% {100 * (df.absdrift <= .05).mean():.1f}%  "
          f"within 10% {100 * (df.absdrift <= .10).mean():.1f}%")


# ============================================================================ corner perturbation
PERT_LEVELS = (0.02, 0.05, 0.10)


def _seed(i):
    return 0 * 100003 + zlib.crc32(str(i).encode()) % 100003  # == ablate.py --seed 0, key = image id


def _perturb_job(i):
    import degrade as D
    C = _counter()
    photo = R.load_image(i)
    gt = R.load_corners(i)
    S = R.default_out_size(gt)
    side = np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
    mm = INCH / side
    out = []
    for lev in PERT_LEVELS:
        pc = D.perturb_corners(gt, lev, _seed(i))
        row = {"id": i, "level": lev, "cerr_pert_mean": float(np.linalg.norm(pc - gt, axis=1).mean() * mm)}
        row["h_pert"], row["v_pert"], row["conf_pert"] = _count(C, _crop_from(photo, pc, S))
        try:
            cal = RC.calibrate(photo, init_corners=pc)
        except Exception as e:  # noqa: BLE001
            cal = {"ok": False, "fail": repr(e)}
        row["ruler_ok"] = bool(cal["ok"])
        if cal["ok"]:
            row["cerr_ruler_mean"] = float(np.linalg.norm(cal["corners"] - gt, axis=1).mean() * mm)
            row["cerr_ruler_max"] = float(np.linalg.norm(cal["corners"] - gt, axis=1).max() * mm)
            row["qscore"] = cal["quality"]["score"]
            crop = RC.rectify_metric(photo, cal, S=R.default_out_size(cal["corners"]), mode="window")
            row["h_rinit"], row["v_rinit"], row["conf_rinit"] = _count(C, crop)
        out.append(row)
    return out


def stage_perturb(workers):
    with Pool(workers) as p:
        res = p.map(_perturb_job, range(R.N_IMAGES), chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(OUT, "perturb150.csv"), index=False)
    print(df.groupby("level")[["cerr_pert_mean", "cerr_ruler_mean", "ruler_ok"]].median())


# ============================================================================ photo degradations
PHOTO_DEG = [("down", 2), ("down", 4), ("gblur", 1), ("gblur", 2), ("gblur", 3), ("jpeg", 20), ("noise", 20)]


def _degrade_photo(photo, kind, level, S_gt, seed):
    import degrade as D
    if kind == "down":
        return D.downscale(photo, level), float(level)
    if kind == "gblur":  # sigma in u (1e-3 inch) of the WINDOW, as in degrade.py
        s = level * S_gt / 1000.0
        return cv2.GaussianBlur(photo, (0, 0), s), 1.0
    if kind == "jpeg":
        return D.jpeg(photo, level), 1.0
    if kind == "noise":
        return D.noise(photo, level, seed), 1.0
    raise ValueError(kind)


def _photodeg_job(i):
    photo = R.load_image(i)
    gt = R.load_corners(i)
    S = R.default_out_size(gt)
    mm = INCH / np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
    out = []
    for kind, lev in PHOTO_DEG:
        img, f = _degrade_photo(photo, kind, lev, S, _seed(i))
        try:
            cal = RC.calibrate(img)
        except Exception as e:  # noqa: BLE001
            cal = {"ok": False, "fail": repr(e)}
        row = {"id": i, "kind": kind, "level": lev, "ok": bool(cal["ok"])}
        if cal["ok"]:
            c = cal["corners"] * f
            row["cerr_mean"] = float(np.linalg.norm(c - gt, axis=1).mean() * mm)
            row["cerr_max"] = float(np.linalg.norm(c - gt, axis=1).max() * mm)
            row["qscore"] = cal["quality"]["score"]
            row["W_mm"] = cal["W_mm"]
        out.append(row)
    return out


def stage_photodeg(workers):
    with Pool(workers) as p:
        res = p.map(_photodeg_job, range(R.N_IMAGES), chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(OUT, "photodeg150.csv"), index=False)
    print(df.groupby(["kind", "level"])[["ok", "cerr_mean"]].agg(["mean", "median"]))


# ============================================================================ ruler blur meter
BLUR_LEVELS = (0, 1, 2, 3, 4, 6)  # defocus sigma in u, as degrade.py / all150_degrade.csv


def _blur_job(i):
    photo = R.load_image(i)
    S = R.default_out_size(R.load_corners(i))
    out = []
    for su in BLUR_LEVELS:
        img = cv2.GaussianBlur(photo, (0, 0), su * S / 1000.0) if su > 0 else photo
        try:
            cal = RC.calibrate(img)
        except Exception as e:  # noqa: BLE001
            cal = {"ok": False, "fail": repr(e)}
        out.append({"id": i, "level": su, "ok": bool(cal["ok"]), "blur_u": cal.get("blur_u", np.nan),
                    "blur_px": cal.get("blur_px", np.nan), "n_ticks_blur": cal.get("n_ticks_blur", 0),
                    "qscore": cal["quality"]["score"] if cal["ok"] else np.nan})
    return out


def stage_blur(workers):
    with Pool(workers) as p:
        res = p.map(_blur_job, range(R.N_IMAGES), chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(OUT, "blur150.csv"), index=False)
    print(df.groupby("level")[["ok", "blur_u"]].median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calib", "count", "perturb", "photodeg", "blur", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage == "calib":
        stage_calib(a.workers)
    elif a.stage == "count":
        stage_count(a.workers)
    elif a.stage == "perturb":
        stage_perturb(a.workers)
    elif a.stage == "photodeg":
        stage_photodeg(a.workers)
    elif a.stage == "blur":
        stage_blur(a.workers)
    elif a.stage == "summary":
        import ruler_summary  # noqa: F401
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
