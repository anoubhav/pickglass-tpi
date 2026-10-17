"""tune_synth.py -- choose every native_res_guard parameter on SYNTHETIC data only (synth_dev.py).

No real crop (legacy-13, R-HR, the 150) is read here.  Stages (outputs in methods/guard/results/,
curves in the scratch folder synth_dev.SCRATCH):

  --features  per (crop, condition): per-axis floor curves of the power spectrum   -> synth_curves.pkl
  --count     frozen counter, unguarded (max_work 1024), every (crop, condition)  -> synth_counts_unguarded.csv
  --locate    cut-off LOCATOR (dead_delta, live_eps, ref_win, mid_frac) scored against the known native
              side n_true of the up-sampled x4 / x8 conditions (bandwidth ground truth): robust spread of
              log(n_eff / n_true) + spread of the per-condition medians + missed detections
                                                                                  -> synth_locator_grid.csv
  --kappa     frozen counter at max_work = round(kappa * n_eff) (locator fixed, depth >= 0.5),
              kappa in KAPPAS                                                     -> synth_counts_kappa.csv
  --choose    (kappa, depth_min, fire_tol) on COUNT accuracy: minimise the macro MAPE over the
              conditions that need a guard (x4 / x8 up-sampling with 4 kernels, x4 + JPEG variants,
              defocus 1-6 milli-inch), subject to the guard never firing on a clean / native-LR
              synthetic crop; ppy_min from the error-vs-px/yarn curve       -> guard_params.json, synth_tuning.json
  --verify    count_guarded() itself, frozen parameters, whole synthetic set  -> synth_counts_guarded.csv

    PY=<venv2>/bin/python
    $PY tune_synth.py --features --count --locate --kappa --choose --verify --workers 10
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import native_res_guard as G  # noqa: E402
import synth_dev as SD  # noqa: E402

RES = os.path.join(HERE, "results")
CURVES = os.path.join(SD.SCRATCH, "synth_curves.pkl")
UNG_CSV = os.path.join(RES, "synth_counts_unguarded.csv")
KAP_CSV = os.path.join(RES, "synth_counts_kappa.csv")
VER_CSV = os.path.join(RES, "synth_counts_guarded.csv")
LOC_CSV = os.path.join(RES, "synth_locator_grid.csv")
LOC_JSON = os.path.join(RES, "synth_locator_choice.json")
TUNE_JSON = os.path.join(RES, "synth_tuning.json")
KAPPAS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)
DEPTH_FLOOR = 0.5      # kappa runs cover every crop with depth >= 0.5 (below any depth_min candidate)

# fixed a priori (not tuned): floor-curve construction and limits
FIXED = dict(n_bands_per_decade=G.DEFAULTS["n_bands_per_decade"], floor_pct=G.DEFAULTS["floor_pct"],
             f_lo=G.DEFAULTS["f_lo"], n_min=G.DEFAULTS["n_min"], max_work=G.DEFAULTS["max_work"])
LOC_GRID = dict(dead_delta=(0.5, 0.65, 0.8, 1.0), live_eps=(0.2, 0.3, 0.4, 0.5),
                ref_win=((0.3, 0.5), (0.4, 0.6), (0.5, 0.75)), mid_frac=(0.3, 0.4, 0.5, 0.6, 0.7))
CHOOSE_GRID = dict(kappa=KAPPAS, depth_min=(0.75, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
                   fire_tol=(0.1, 0.15, 0.2, 0.25, 0.3))
DESIGN_LOG = [
    "all steps on synthetic data only; no real crop read before guard_params.json was frozen",
    "v1 locator (cut-off at the first empty band, depth window [0.55,0.9] f_c): missed ~50 % of bicubic x4 "
    "(replica bumps above L_dead) -> v2 locator: bump-tolerant empty region f_d + half-height crossing f_c",
    "locator grid scored against the known native side n_true of x4/x8 up-sampled crops; tie-break within 3 % "
    "of the best score: smallest dead_delta (fewer empty floors on clean crops)",
    "first choice (ungated: re-count whenever eligible): kappa 0.9, depth_min 1.2, fire_tol 0.3, macro MAPE "
    "93.3 -> 28.7; but defocus MEDIAN APE rose (2.5 -> 18.5 at 2u, 9.2 -> 58.2 at 6u): under blur the yarn line "
    "survives beyond the background floor's cut-off (f0 > guarded band top for 43-81 % of defocused crops)",
    "-> consistency gate added (parameter-free): re-count only if the unguarded count lies outside the band "
    "searched at the native side; same objective re-run (macro MAPE, zero clean firing, ties -> conservative)",
]


def family(cond):
    if cond in ("clean", "jpeg50"):
        return "clean"
    if cond.startswith("none"):
        return "none"
    if cond.startswith("gblur"):
        return "defocus"
    if cond.endswith("_x2"):
        return "upsampled_x2"
    return "upsampled"


def jobs_all():
    lab = SD.labels()
    return [(r.name, r.set, cond) for r in lab.itertuples() for cond in SD.conditions(r.set)]


# ----------------------------------------------------------------------------- features
def _feat(a):
    name, set_name, cond = a
    img, n_true = SD.apply_condition(SD.load_crop(name), cond)
    g = G._gray(img)
    H, W = g.shape
    axes = {}
    for ax, gg, n in (("x", g, W), ("y", g.T, H)):
        P = G.axis_spectrum(gg)
        cen, lo, hi, L, wts = G.floor_curve(P, FIXED["f_lo"], FIXED["n_bands_per_decade"], FIXED["floor_pct"])
        axes[ax] = dict(cen=cen, lo=lo, hi=hi, L=L, wts=wts, n=int(n))
    return dict(name=name, set=set_name, cond=cond, S=int(max(H, W)), n_true=int(n_true), axes=axes)


def features(workers):
    t0 = time.time()
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = pool.map(_feat, jobs_all(), chunksize=4)
    with open(CURVES, "wb") as f:
        pickle.dump(rows, f)
    print(f"features: {len(rows)} images in {time.time() - t0:.0f}s -> {CURVES}")


def load_curves():
    with open(CURVES, "rb") as f:
        return pickle.load(f)


def detect_from_curves(row, det):
    """G.estimate_bandwidth() evaluated from the stored curves (same _cutoff code)."""
    p = {**G.DEFAULTS, **FIXED, **det}
    axes = {}
    for ax in ("x", "y"):
        a = row["axes"][ax]
        c = G._cutoff(a["cen"], a["lo"], a["hi"], a["L"], a["wts"], p, a["n"])
        c["frac_nyq"] = float(c["f_c"] / (a["n"] / 2.0))
        axes[ax] = c
    return dict(S=row["S"], axes=axes, **G.combine_axes(axes, row["S"]))


# ----------------------------------------------------------------------------- locator
def locate():
    rows = [r for r in load_curves() if family(r["cond"]) == "upsampled"]
    out = []
    for dd, le, rw, mf in itertools.product(*LOC_GRID.values()):
        det = dict(dead_delta=dd, live_eps=le, ref_win=rw, mid_frac=mf, depth_min=-99.0)
        rec = []
        for r in rows:
            d = detect_from_curves(r, det)
            rec.append((r["cond"], d["dead_band"], math_log(d["n_eff"] / r["n_true"])))
        R = pd.DataFrame(rec, columns=["cond", "dead", "lr"])
        D = R[R.dead]
        km = D.groupby("cond").lr.median()
        out.append(dict(dead_delta=dd, live_eps=le, ref_win=list(rw), mid_frac=mf, detected=float(R.dead.mean()),
                        ratio_median=float(np.exp(D.lr.median())), mad_log=float((D.lr - D.lr.median()).abs().mean()),
                        p90_p10_log=float(D.lr.quantile(.9) - D.lr.quantile(.1)),
                        cond_median_range_log=float(km.max() - km.min()),
                        worst_cond=str(km.idxmax()), best_cond=str(km.idxmin())))
    O = pd.DataFrame(out)
    O["score"] = O.mad_log + 0.5 * O.cond_median_range_log + 0.5 * (1 - O.detected)
    O = O.sort_values("score").reset_index(drop=True)
    O.to_csv(LOC_CSV, index=False)
    # conservative tie-break: within 3 % of the best score, the smallest dead_delta (a floor must come
    # closer to the quantisation level to count as empty -> fewer dead bands on clean crops), then score
    near = O[O.score <= 1.03 * O.score.min()].sort_values(["dead_delta", "score"])
    best = near.iloc[0].to_dict()
    choice = {k: best[k] for k in LOC_GRID}
    choice["ref_win"] = list(choice["ref_win"])
    json.dump(dict(choice=choice, score=best["score"], ratio_median=best["ratio_median"],
                   detected=best["detected"], note="locator scored on up-sampled x4/x8 synthetic only "
                   "(score = mean |log(n_eff/n_true) - median| + 0.5 range of per-condition medians "
                   "+ 0.5 missed detections; depth_min not applied); tie-break within 3 % of the best "
                   "score: smallest dead_delta"), open(LOC_JSON, "w"), indent=1)
    print(O.head(12).round(3).to_string())
    print("locator:", choice)
    return choice


def math_log(x):
    return float(np.log(x)) if x > 0 else float("nan")


def locator_choice():
    return json.load(open(LOC_JSON))["choice"]


# ----------------------------------------------------------------------------- counting
def _count(a):
    name, set_name, cond, max_work, tag = a
    C = G.frozen_counter()
    img, n_true = SD.apply_condition(SD.load_crop(name), cond)
    t0 = time.time()
    try:
        e = C.estimate(img, method="ours", max_work=int(max_work))
        d = e["debug"]
        row = dict(h=e["h_count"], v=e["v_count"], conf=e["confidence"], N_work=d.get("N_work"),
                   fusion="|".join(d["fusion"]) if d.get("fusion") else None)
    except Exception as ex:  # noqa: BLE001
        row = dict(h=np.nan, v=np.nan, conf=0.0, N_work=None, fusion=None, error=repr(ex)[:200])
    row.update(name=name, set=set_name, cond=cond, max_work=int(max_work), tag=tag, S_in=int(img.shape[1]),
               n_true=int(n_true), seconds=round(time.time() - t0, 2))
    return row


def _run(jobs, workers, out_csv):
    t0 = time.time()
    rows = []
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_count, jobs, chunksize=1)):
            rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} rows, {time.time() - t0:.0f}s")
    return df


def count_unguarded(workers):
    jobs = [(n, s, c, 1024, "unguarded") for n, s, c in jobs_all()]
    jobs.sort(key=lambda j: 0 if j[2].startswith("none") else -1)
    _run(jobs, workers, UNG_CSV)


def target_of(n_eff, kappa):
    return int(max(FIXED["n_min"], round(kappa * n_eff)))


def count_kappa(workers):
    loc = locator_choice()
    det = dict(loc, depth_min=DEPTH_FLOOR)
    jobs = []
    for r in load_curves():
        d = detect_from_curves(r, det)
        if not d["dead_band"]:
            continue
        N_work = min(r["S"], 1024)
        for t in sorted({target_of(d["n_eff"], k) for k in KAPPAS}):
            if t < N_work:
                jobs.append((r["name"], r["set"], r["cond"], t, "kappa"))
    jobs.sort(key=lambda j: -j[3])
    print(f"kappa runs: {len(jobs)}")
    _run(jobs, workers, KAP_CSV)


# ----------------------------------------------------------------------------- choose
def _ape(h, v, gh, gv):
    ah = np.where(np.isfinite(h), 100 * np.abs(h / gh - 1), 100.0)
    av = np.where(np.isfinite(v), 100 * np.abs(v / gv - 1), 100.0)
    return ah, av


def simulate(loc, depth_min, kappa, fire_tol, rows=None, ung=None, kap=None, lab=None, gate=True):
    """Guarded outcome of every synthetic (crop, condition) for one parameter set, from the
    stored unguarded counts and the kappa runs (exact: same target, eligibility and consistency-gate
    rules as the module; gate=False reproduces the ungated design of the first choice)."""
    rows = load_curves() if rows is None else rows
    lab = SD.labels() if lab is None else lab
    truth = dict(zip(lab.name, zip(lab.h_count, lab.v_count)))
    det = dict(loc, depth_min=depth_min)
    out = []
    for r in rows:
        d = detect_from_curves(r, det)
        S = r["S"]
        N_work = min(S, 1024)
        n_eff = d["n_eff"] if d["dead_band"] else float(S)
        t = target_of(n_eff, kappa)
        eligible = bool(d["dead_band"] and t < (1 - fire_tol) * N_work)
        uh, uv = ung[(r["name"], r["cond"])]
        outside = G.claim_outside_band(uh, uv, t) if gate else True
        fire = eligible and outside
        if fire:
            gh_, gv_ = kap.get((r["name"], r["cond"], t), (np.nan, np.nan))
            missing = (r["name"], r["cond"], t) not in kap
        else:
            gh_, gv_, missing = uh, uv, False
        th, tv = truth[r["name"]]
        out.append(dict(name=r["name"], set=r["set"], cond=r["cond"], fam=family(r["cond"]), fire=fire,
                        eligible=eligible, outside=outside,
                        dead=d["dead_band"], depth=d["depth"], n_eff=n_eff, n_true=r["n_true"], S=S, target=t,
                        ug_h=uh, ug_v=uv, h=gh_, v=gv_, gt_h=th, gt_v=tv, missing=missing))
    df = pd.DataFrame(out)
    ah, av = _ape(df.h.values, df.v.values, df.gt_h.values, df.gt_v.values)
    uah, uav = _ape(df.ug_h.values, df.ug_v.values, df.gt_h.values, df.gt_v.values)
    df["ape"], df["ape_ug"] = 0.5 * (ah + av), 0.5 * (uah + uav)
    df["ape_max"] = np.maximum(ah, av)
    chg = np.fmax(np.abs(df.h / df.ug_h - 1), np.abs(df.v / df.ug_v - 1))
    df["change"] = np.where(np.isfinite(chg), chg, np.where(df.fire, 1.0, 0.0))
    return df


def choose():
    lab = SD.labels()
    rows = load_curves()
    loc = locator_choice()
    U = pd.read_csv(UNG_CSV)
    K = pd.read_csv(KAP_CSV)
    ung = {(r.name, r.cond): (r.h, r.v) for r in U.itertuples()}
    kap = {(r.name, r.cond, int(r.max_work)): (r.h, r.v) for r in K.itertuples()}
    res = []
    for (kappa, dm, ft), gate in itertools.product(itertools.product(*CHOOSE_GRID.values()), (True, False)):
        df = simulate(loc, dm, kappa, ft, rows, ung, kap, lab, gate=gate)
        need = df[df.fam.isin(["upsampled", "defocus"])]
        clean = df[df.fam.isin(["clean", "none"])]
        per = need.groupby("cond").ape.mean()
        res.append(dict(gate=gate, kappa=kappa, depth_min=dm, fire_tol=ft, macro_mape_need=float(per.mean()),
                        macro_median_need=float(need.groupby("cond").ape.median().mean()),
                        worse5_need=float((need.ape > need.ape_ug + 5).mean()),
                        better5_need=float((need.ape < need.ape_ug - 5).mean()),
                        macro_mape_need_ug=float(need.groupby("cond").ape_ug.mean().mean()),
                        fire_need=float(need.fire.mean()), fire_clean=float(clean.fire.mean()),
                        n_fire_clean=int(clean.fire.sum()), chg1_clean=float((clean.change > 0.01).mean()),
                        mape_clean=float(clean.ape.mean()), mape_clean_ug=float(clean.ape_ug.mean()),
                        mape_up=float(need[need.fam == "upsampled"].groupby("cond").ape.mean().mean()),
                        mape_defocus=float(need[need.fam == "defocus"].groupby("cond").ape.mean().mean()),
                        mape_x2=float(df[df.fam == "upsampled_x2"].ape.mean()),
                        mape_x2_ug=float(df[df.fam == "upsampled_x2"].ape_ug.mean()),
                        missing=int(df.missing.sum())))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(RES, "synth_choose_grid.csv"), index=False)
    # design (declared before any real crop was read): the consistency gate is part of the guard
    # (see DESIGN_LOG); the ungated rows are kept in the grid CSV for the record
    feas = R[(R.n_fire_clean == 0) & (R.missing == 0) & R.gate]
    if feas.empty:
        raise SystemExit("no parameter set without clean firing")
    # minimise macro MAPE; ties (within 0.05 points) -> larger depth_min, then larger fire_tol (more conservative)
    best_val = feas.macro_mape_need.min()
    near = feas[feas.macro_mape_need <= best_val + 0.05].sort_values(["depth_min", "fire_tol"], ascending=False)
    best = near.iloc[0].to_dict()
    print(feas.sort_values("macro_mape_need").head(20).round(3).to_string())
    print("best:", best)
    params = dict(G.DEFAULTS)
    params.update(FIXED)
    params.update(loc)
    params.update(kappa=float(best["kappa"]), depth_min=float(best["depth_min"]), fire_tol=float(best["fire_tol"]))
    # ppy_min: smallest value in the optics rule's range [4, 5] maximising Youden's J for flagging
    # guarded synthetic counts with an error > 10 % (px per yarn from n_eff and the guarded counts)
    df = simulate(loc, params["depth_min"], params["kappa"], params["fire_tol"], rows, ung, kap, lab, gate=True)
    dens = np.fmax(df.h, df.v)
    df["ppy"] = df.n_eff / dens
    fail = df.ape_max > 10
    yj = []
    for thr in (4.0, 4.25, 4.5, 4.75, 5.0):
        flag = df.ppy < thr
        tpr = float((flag & fail).sum() / max(1, fail.sum()))
        fpr = float((flag & ~fail).sum() / max(1, (~fail).sum()))
        yj.append(dict(ppy_min=thr, tpr=tpr, fpr=fpr, J=tpr - fpr, flagged=float(flag.mean())))
    YJ = pd.DataFrame(yj)
    params["ppy_min"] = float(YJ.sort_values(["J", "ppy_min"], ascending=[False, True]).iloc[0].ppy_min)
    print(YJ.round(3).to_string())
    params["ref_win"] = list(params["ref_win"])
    out = dict(params=params, chosen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               data="synthetic only: methods/guard/synth_dev.py (cnn/synth.py v2, seeds 960000-960119 hr, "
                    "965000-965059 lr)", locator=json.load(open(LOC_JSON)),
               choose_best=best, ppy_youden=YJ.to_dict(orient="records"), design_log=DESIGN_LOG,
               ungated_best=R[(R.n_fire_clean == 0) & (R.missing == 0) & ~R.gate]
               .sort_values("macro_mape_need").iloc[0].to_dict())
    json.dump(out, open(G.PARAMS_PATH, "w"), indent=1)
    json.dump(out, open(TUNE_JSON, "w"), indent=1)
    print("wrote", G.PARAMS_PATH)


# ----------------------------------------------------------------------------- verify
def _verify(a):
    name, set_name, cond = a
    img, n_true = SD.apply_condition(SD.load_crop(name), cond)
    t0 = time.time()
    r = G.count_guarded(img)
    g = r["guard"]
    return dict(name=name, set=set_name, cond=cond, n_true=n_true, S=g["S"], h=r["h_count"], v=r["v_count"],
                conf=r["confidence"], fired=g["fired"], eligible=g["eligible"], unsupported=g["unsupported_claim"],
                flag=g["flag"], n_eff=g["n_eff"], depth=g["depth"],
                N_work=g["N_work_guarded"], ppy=g["px_per_yarn"], margin=g["px_per_yarn_margin"],
                seconds=round(time.time() - t0, 2))


def verify(workers):
    t0 = time.time()
    jobs = jobs_all()
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = pool.map(_verify, jobs, chunksize=2)
    df = pd.DataFrame(rows)
    lab = SD.labels()
    U = pd.read_csv(UNG_CSV)[["name", "cond", "h", "v"]].rename(columns={"h": "ug_h", "v": "ug_v"})
    df = df.merge(U, on=["name", "cond"]).merge(lab[["name", "h_count", "v_count"]], on="name")
    ah, av = _ape(df.h.values, df.v.values, df.h_count.values, df.v_count.values)
    uh, uv = _ape(df.ug_h.values, df.ug_v.values, df.h_count.values, df.v_count.values)
    df["ape"], df["ape_ug"] = 0.5 * (ah + av), 0.5 * (uh + uv)
    df["change"] = np.fmax(np.abs(df.h / df.ug_h - 1), np.abs(df.v / df.ug_v - 1))
    df.to_csv(VER_CSV, index=False)
    df["better"] = df.ape < df.ape_ug - 1e-9
    df["worse"] = df.ape > df.ape_ug + 1e-9
    summ = df.groupby(["set", "cond"]).agg(n=("ape", "size"), eligible=("eligible", "mean"), fired=("fired", "mean"),
                                           mape_ug=("ape_ug", "mean"), mape_guard=("ape", "mean"),
                                           med_ug=("ape_ug", "median"), med_guard=("ape", "median"),
                                           n_better=("better", "sum"), n_worse=("worse", "sum"),
                                           chg1=("change", lambda x: (x > 0.01).mean()),
                                           chg5=("change", lambda x: (x > 0.05).mean()),
                                           insufficient=("flag", lambda x: x.str.contains("insufficient").mean()),
                                           ratio_med=("n_eff", "median"))
    summ.to_csv(os.path.join(RES, "synth_guarded_summary.csv"))
    print(summ.round(3).to_string())
    print(f"verify: {len(df)} in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in ("features", "count", "locate", "kappa", "choose", "verify"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.features:
        features(a.workers)
    if a.count:
        count_unguarded(a.workers)
    if a.locate:
        locate()
    if a.kappa:
        count_kappa(a.workers)
    if a.choose:
        choose()
    if a.verify:
        verify(a.workers)


if __name__ == "__main__":
    main()
