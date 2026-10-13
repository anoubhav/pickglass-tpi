#!/usr/bin/env python
"""invariance_check.py -- label-free check that the propagated labels are valid (and a
real-capture-style robustness result): run the FROZEN counter
(methods/results/counting_frozen_final.py, method "ours") on every augmented crop and compare
with its own predictions on the corresponding original (methods/predictions_ours_150.csv),
after swapping h/v for k = 90/270.

Runs (all through the same counter, all crops JPEG q=92 like the tool crops):
  aug        the 853 augmented crops in crops/ (what training would see)
  noop       baseline 1: the 150 tool crops (annotation_tool/data/crops/{i}.jpg) = the
             in-memory crops of predictions_ours_150.csv re-encoded once
  rot        baseline 2: np.rot90(source crop, j) for j = 90/180/270 (pixel permutation, no
             resampling) -> counter rotation equivariance
  matched    baseline 3: per augmentation, the SOURCE image re-rectified at the augmentation's
             rect_size and rotated by its k (rectify.py convention, re-encoded) -> isolates
             'only the resolution and rotation changed' from the real augmentation
             (photometric, noise, JPEG q70-80, double interpolation, perspective resampling)
rel. diff = |pred - ref| / ref per axis, ref = source prediction mapped through the swap.

Outputs: invariance/per_crop.csv, invariance/summary.csv, invariance/summary.md

    <venv2>/bin/python invariance_check.py [--workers 6]
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import importlib.util
import os
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aug_common as A  # noqa: E402

R = A.R
COUNTER = os.path.join(A.ROOT, "methods", "results", "counting_frozen_final.py")
PRED150 = os.path.join(A.ROOT, "methods", "predictions_ours_150.csv")
OUT = os.path.join(A.HERE, "invariance")
_CF = None


def counter():
    global _CF
    if _CF is None:
        spec = importlib.util.spec_from_file_location("counting_frozen_final", COUNTER)
        _CF = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CF)
    return _CF


def reencode(img):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, A.JPEG_Q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def job(t):
    kind, key, src, k_deg, S = t
    t0 = time.time()
    if kind == "aug":
        crop = cv2.imread(os.path.join(A.HERE, "crops", f"{key}.jpg"))
    elif kind == "noop":
        crop = cv2.imread(os.path.join(A.TOOL_CROPS, f"{src}.jpg"))
    elif kind == "rot":
        crop = reencode(np.ascontiguousarray(np.rot90(R.rectify(src), k_deg // 90)))
    elif kind == "matched":
        img = R.load_image(src)
        c = R.load_corners(src)
        crop = reencode(A.source_rotations(img, c, int(S))[k_deg // 90])
    else:
        raise ValueError(kind)
    e = counter().estimate(crop, method="ours")
    return dict(kind=kind, key=key, source_id=src, k_deg=k_deg, S=crop.shape[0], pred_h=e["h_count"],
                pred_v=e["v_count"], conf=e["confidence"], weave=e["weave"], t_s=round(time.time() - t0, 2))


def stats(d, label):
    r = np.concatenate([d.rel_h.values, d.rel_v.values]) * 100
    if not len(r):
        return None
    per_src = d.groupby("source_id")[["rel_h", "rel_v"]].median().mean(axis=1) * 100
    return {"subset": label, "n_crops": len(d), "n_sources": d.source_id.nunique(),
            "median_%": round(float(np.median(r)), 2), "p90_%": round(float(np.percentile(r, 90)), 2),
            "mean_%": round(float(np.mean(r)), 2),
            "within_2%": round(float(100 * np.mean(r <= 2)), 1), "within_5%": round(float(100 * np.mean(r <= 5)), 1),
            "within_10%": round(float(100 * np.mean(r <= 10)), 1),
            "median_h_%": round(float(100 * d.rel_h.median()), 2), "median_v_%": round(float(100 * d.rel_v.median()), 2),
            "p90_h_%": round(float(100 * d.rel_h.quantile(.9)), 2), "p90_v_%": round(float(100 * d.rel_v.quantile(.9)), 2),
            "median_of_source_medians_%": round(float(per_src.median()), 2),
            "flip_%": round(float(100 * (d[["rel_h", "rel_v"]].max(axis=1) > 0.10).mean()), 1),
            "weave_same_%": round(float(100 * d.weave_same.mean()), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summarise-only", action="store_true", help="re-summarise invariance/per_crop.csv")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man = pd.read_csv(A.MANIFEST)
    p150 = pd.read_csv(PRED150).set_index("id")
    tasks = [("aug", r.aug_id, int(r.source_id), int(r.k_deg), int(r.rect_size)) for r in man.itertuples()]
    tasks += [("noop", f"{i}_noop", i, 0, 0) for i in range(150)]
    tasks += [("rot", f"{i}_rot{j}", i, j, 0) for i in range(150) for j in (90, 180, 270)]
    tasks += [("matched", f"{r.aug_id}_matched", int(r.source_id), int(r.k_deg), int(r.rect_size))
              for r in man.itertuples()]
    # big crops first for load balance
    tasks.sort(key=lambda t: -(t[4] or 1000))
    t0 = time.time()
    if a.summarise_only:
        df = pd.read_csv(os.path.join(OUT, "per_crop.csv"))
        df = df[["kind", "key", "source_id", "k_deg", "S", "pred_h", "pred_v", "conf", "weave", "t_s"]]
    else:
        with Pool(a.workers) as p:
            rows = p.map(job, tasks, chunksize=4)
        df = pd.DataFrame(rows)
    swap = df.k_deg.isin([90, 270])
    sh, sv = p150.loc[df.source_id, "h_count"].values, p150.loc[df.source_id, "v_count"].values
    df["ref_h"] = np.where(swap, sv, sh)
    df["ref_v"] = np.where(swap, sh, sv)
    df["rel_h"] = (df.pred_h - df.ref_h).abs() / df.ref_h
    df["rel_v"] = (df.pred_v - df.ref_v).abs() / df.ref_v
    df["src_conf"] = p150.loc[df.source_id, "confidence"].values
    df["weave_same"] = df.weave.values == p150.loc[df.source_id, "weave"].values
    df = df.merge(man[["aug_id", "usable", "t_rot90", "t_persp", "t_ssr", "t_bc", "t_noise", "t_jpeg", "rect_ratio"]]
                  .rename(columns={"aug_id": "key"}), on="key", how="left")
    df.sort_values(["kind", "source_id", "key"]).to_csv(os.path.join(OUT, "per_crop.csv"), index=False)

    conf_med = float(p150.confidence.median())
    S = []
    aug = df[df.kind == "aug"]
    ok = aug[aug.usable == True]  # noqa: E712
    mat = df[df.kind == "matched"]
    mat_ok = mat[mat.key.str.replace("_matched", "").isin(ok.key)]
    S.append(stats(df[df.kind == "noop"], "baseline: no-op re-encode (150 tool crops)"))
    S.append(stats(df[df.kind == "rot"], "baseline: pure rot90 + re-encode (150 x 3)"))
    S.append(stats(mat_ok, "baseline: matched (source re-rectified at aug S, rotated by k)"))
    S.append(stats(ok, "AUG: all usable"))
    S.append(stats(aug, "AUG: all 853 incl. flagged"))
    S.append(stats(aug[~(aug.usable == True)], "AUG: flagged only"))  # noqa: E712
    S.append(stats(ok[ok.src_conf >= conf_med], f"AUG usable, source confidence >= median ({conf_med:.2f})"))
    S.append(stats(ok[ok.src_conf < conf_med], "AUG usable, source confidence < median"))
    for k in (0, 90, 180, 270):
        S.append(stats(ok[ok.k_deg == k], f"AUG usable, k = {k}"))
    for c, name in [("t_persp", "Perspective"), ("t_ssr", "ShiftScale (no persp.)"), ("t_bc", "BrightnessContrast"),
                    ("t_noise", "GaussNoise"), ("t_jpeg", "JPEG q70-80")]:
        for v in (True, False):
            S.append(stats(ok[ok[c] == v], f"AUG usable, {name} {'yes' if v else 'no'}"))
    for lo, hi in [(0, 0.97), (0.97, 1.03), (1.03, 1.15), (1.15, 9)]:
        S.append(stats(ok[(ok.rect_ratio >= lo) & (ok.rect_ratio < hi)], f"AUG usable, rect_size/source in [{lo},{hi})"))
    clean = ok[(ok.t_noise == False) & (ok.t_bc == False) & (ok.t_persp == False)]  # noqa: E712
    S.append(stats(clean, "AUG usable, no noise / no BC / no perspective"))
    # sources on which the counter itself is bistable: > 10 % change from a re-encode or a rot90 alone
    base = df[df.kind.isin(["noop", "rot"])]
    unstable = sorted(set(base.source_id[base[["rel_h", "rel_v"]].max(axis=1) > 0.10]))
    st = ok[~ok.source_id.isin(unstable)]
    S.append(stats(st, f"AUG usable, excl. {len(unstable)} counter-unstable sources"))
    S.append(stats(st[st.t_noise == False], "AUG usable, excl. unstable sources, no GaussNoise"))  # noqa: E712
    S.append(stats(mat_ok[~mat_ok.source_id.isin(unstable)], "baseline: matched, excl. unstable sources"))
    with open(os.path.join(OUT, "counter_unstable_sources.txt"), "w") as f:
        f.write("# sources whose frozen-counter output changes by > 10 % under a JPEG re-encode or a pure rot90\n")
        f.write(" ".join(map(str, unstable)) + "\n")
    summ = pd.DataFrame([s for s in S if s])
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    cols = ["subset", "n_crops", "n_sources", "median_%", "p90_%", "within_5%", "within_10%", "median_h_%",
            "median_v_%", "median_of_source_medians_%", "flip_%", "weave_same_%"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in summ.iterrows()]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Frozen-counter invariance on the augmented crops\n\n"
                "relative difference |pred - ref| / ref pooled over both axes; ref = the counter's prediction on "
                "the original (h/v swapped for k = 90/270); flip_% = crops whose larger per-axis difference "
                "exceeds 10 % (octave / stripe-vs-yarn jumps of a bistable estimate)\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    print(f"{len(df)} counter runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
