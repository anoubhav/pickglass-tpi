"""posthoc_supplement.py -- POST-HOC robustness and descriptive analyses of the one-shot real evaluation.

Written AFTER eval_real.py / analyze_real.py had been run and read.  Nothing here
changes the guard (native_res_guard.py and guard_params.json are only imported / read), and no
guard parameter is chosen here.  Every output is labelled post hoc in RESULTS.md.

  --tables      (reads results/real_counts.csv, results/real_posthoc_ungated.csv; writes)
     ph_rhr_scene_cluster.csv   R-HR cells with a SCENE-cluster bootstrap: the 53 R-HR crops are only
                                40 distinct scenes (camera_aug/originals_provenance.json; e.g. 86 and
                                90 are byte-identical), so image-level CIs are too narrow
     ph_clean150_scenes.csv     clean-crop invariance counted over the 76 distinct scenes
     ph_defocus_detail.csv      legacy-13 defocus / blur conditions: median APE, within-10 %, over- and
                                under-counts (> +10 % / < -10 %), and the guarded confidence of fired crops
     ph_cascade_policies.csv    guarded ours vs the stored VLM cascade (paired, cluster bootstrap over the
                                13 photos) + 2 EXPLORATORY routers using the guarded confidence with the
                                cascade's own synthetic-fitted thresholds (nothing fitted here)
     ph_flag_vs_failure.csv     the guard's flags as failure detectors (APE > 10 % on either count)
     real_posthoc_ungated_summary_repro.csv   re-derivation of real_posthoc_ungated_summary.csv, whose
                                generating code was not saved by the interrupted run
  --rectified   the guard on the 150 crops re-rectified live from the originals (common/rectify.py, the
                exact inputs of methods/predictions_ours_150.csv; the one-shot run used the saved JPEG
                crops annotation_tool/data/crops/{i}.jpg)   -> ph_rectified150.csv

    <venv2>/bin/python posthoc_supplement.py --tables --rectified --workers 6
"""
from __future__ import annotations

import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2

cv2.setNumThreads(1)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = os.path.dirname(HERE)
ROOT = os.path.dirname(METHODS)
sys.path.insert(0, HERE)
import analyze_real as A  # noqa: E402

RES = os.path.join(HERE, "results")
B = A.B
SEED = A.SEED
PROV = os.path.join(ROOT, "camera_aug", "originals_provenance.json")


def scene_of():
    d = json.load(open(PROV))
    sc = {}
    for gi, g in enumerate(d["groups"]):
        for i in g:
            sc[int(i)] = gi
    assert len(sc) == 150
    return sc


def cluster_boot(vals_list, keys, rng, nb=B):
    """Percentile CIs of the mean of each array in vals_list, resampling clusters `keys`."""
    keys = np.asarray(keys)
    uk = sorted(set(keys.tolist()))
    members = [np.nonzero(keys == k)[0] for k in uk]
    outs = [np.empty(nb) for _ in vals_list]
    for b in range(nb):
        sel = np.concatenate([members[j] for j in rng.integers(0, len(uk), len(uk))])
        for o, v in zip(outs, vals_list):
            o[b] = v[sel].mean()
    return outs, len(uk)


# ------------------------------------------------------------------------------------------------ tables
def rhr_scene_cluster(df):
    sc = scene_of()
    G = df[(df.group == "genai") & (df.set == "R-HR")].copy()
    G["scene"] = G.key.str[2:].astype(int).map(sc)
    rows = []
    for rest in ["clean"] + A.REST_ORDER:
        for cond in (["clean"] if rest == "clean" else ["D4", "D8", "D4J"]):
            d = G[(G.restorer == rest) & (G.cond == cond)]
            if d.empty:
                continue
            u, g = d.ape_ug.to_numpy(float), d.ape_g.to_numpy(float)
            rng = np.random.default_rng(SEED)
            (bu, bg, bd), n_sc = cluster_boot([u, g, g - u], d.scene.to_numpy(), rng)
            # scene-weighted point estimate: every scene counts once (mean over its copies)
            sw = d.assign(u=u, g=g).groupby("scene")[["u", "g"]].mean()
            rows.append(dict(restorer=rest, cond=cond, n_images=len(d), n_scenes=n_sc,
                             mape_ug=u.mean(), ug_lo=np.percentile(bu, 2.5), ug_hi=np.percentile(bu, 97.5),
                             mape_g=g.mean(), g_lo=np.percentile(bg, 2.5), g_hi=np.percentile(bg, 97.5),
                             d=g.mean() - u.mean(), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5),
                             scene_weighted_mape_ug=float(sw.u.mean()), scene_weighted_mape_g=float(sw.g.mean()),
                             fire_pct=100 * d.fired.mean(), n_fired=int(d.fired.sum())))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "ph_rhr_scene_cluster.csv"), index=False)
    return T


def clean150_scenes(df):
    sc = scene_of()
    C = df[df.group == "clean150"].copy()
    C["id"] = C.key.astype(int)
    C["scene"] = C.id.map(sc)
    per = C.groupby("scene").agg(any_fire=("fired", "any"), any_chg1=("change", lambda x: (x > 0.01).any()),
                                 any_chg5=("change", lambda x: (x > 0.05).any()),
                                 any_eligible=("eligible", "any"), any_upflag=("upsampled", "any"),
                                 any_insuff=("insufficient", "any"))
    reps = []
    for line in open(os.path.join(ROOT, "annotation_tool", "scenes_to_annotate.txt")):
        if line.strip() and not line.startswith("#"):
            reps.append(int(line.split(":")[0]))
    R = C[C.id.isin(reps)]
    rows = [dict(unit="scenes (76; any file of the scene)", n=len(per), fired=int(per.any_fire.sum()),
                 chg1=int(per.any_chg1.sum()), chg5=int(per.any_chg5.sum()), eligible=int(per.any_eligible.sum()),
                 flag_upsampled=int(per.any_upflag.sum()), flag_insufficient=int(per.any_insuff.sum())),
            dict(unit="scene representatives (scenes_to_annotate.txt)", n=len(R), fired=int(R.fired.sum()),
                 chg1=int((R.change > 0.01).sum()), chg5=int((R.change > 0.05).sum()), eligible=int(R.eligible.sum()),
                 flag_upsampled=int(R.upsampled.sum()), flag_insufficient=int(R.insufficient.sum())),
            dict(unit="files (150)", n=len(C), fired=int(C.fired.sum()), chg1=int((C.change > 0.01).sum()),
                 chg5=int((C.change > 0.05).sum()), eligible=int(C.eligible.sum()),
                 flag_upsampled=int(C.upsampled.sum()), flag_insufficient=int(C.insufficient.sum()))]
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "ph_clean150_scenes.csv"), index=False)
    flagged = C[C.upsampled | C.eligible][["id", "scene", "S", "eligible", "fired", "upsampled", "frac_nyq", "depth",
                                            "ug_conf"]].sort_values("id")
    return T, flagged


def defocus_detail(df):
    C = df[df.group == "cascade"].copy()
    tc = json.load(open(os.path.join(ROOT, "paper", "tables", "cascade_thresholds.json")))["fits"]["vlm9_a10"]["conf"]["t_c"]
    rows = []
    for cond in ["clean", "gblur1", "gblur2", "gblur3", "gblur4", "gblur6", "down4", "down8"]:
        d = C[C.cond == cond]
        r = dict(cond=cond, n=len(d), n_fired=int(d.fired.sum()))
        for p in ("ug", "g"):
            sh = 100 * (d[f"{p}_h"] / d.ref_h - 1)
            sv = 100 * (d[f"{p}_v"] / d.ref_v - 1)
            s = np.concatenate([sh, sv])
            r.update({f"mape_{p}": float(d[f"ape_{p}"].mean()), f"median_ape_{p}": float(np.median(np.abs(s))),
                      f"within10_{p}": int((np.abs(s) <= 10).sum()), f"over10_{p}": int((s > 10).sum()),
                      f"under10_{p}": int((s < -10).sum()), f"counts": 2 * len(d)})
        f = d[d.fired]
        r.update(fired_gconf_median=float(f.g_conf.median()) if len(f) else np.nan,
                 fired_gconf_below_tc=int((f.g_conf < tc).sum()), fired_ugconf_below_tc=int((f.ug_conf < tc).sum()),
                 t_c=tc)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "ph_defocus_detail.csv"), index=False)
    return T


def cascade_policies(df):
    inp = pd.read_csv(os.path.join(ROOT, "paper", "tables", "cascade_legacy13_inputs.csv"))
    inp["tag"] = inp.cond.map(A.COND_TAG)
    C = df[df.group == "cascade"].merge(inp[["image", "tag", "gt_w", "gt_f", "conf", "rho", "gpt_w", "gpt_f", "gpt_cost",
                                             "ours_w", "ours_f"]],
                                        left_on=["key", "cond"], right_on=["image", "tag"], how="inner")
    assert len(C) == 117
    # the stored cascade inputs were made with the same frozen counter: check
    same = np.allclose(C.ours_w, C.ug_h, rtol=1e-4) and np.allclose(C.ours_f, C.ug_v, rtol=1e-4)
    thr = json.load(open(os.path.join(ROOT, "paper", "tables", "cascade_thresholds.json")))["fits"]["vlm9_a10"]
    gate, tconf = thr["gate"], thr["conf"]["t_c"]

    def ape(h, v):
        return (0.5 * (100 * np.abs(np.asarray(h, float) / C.gt_w - 1) + 100 * np.abs(np.asarray(v, float) / C.gt_f - 1))
                ).to_numpy()
    esc_paper = ((C.conf < gate["t_c"]) | (C.rho < gate["r_min"])).to_numpy()
    pol = {
        "ours unguarded": (C.ug_h, C.ug_v, np.zeros(len(C), bool)),
        "ours guarded (no VLM)": (C.g_h, C.g_v, np.zeros(len(C), bool)),
        "cascade (paper): unguarded ours -> VLM if conf < t_c or rho < r_min": (
            np.where(esc_paper, C.gpt_w, C.ug_h), np.where(esc_paper, C.gpt_f, C.ug_v), esc_paper),
        "VLM only (stored GPT-6.1-sol P1)": (C.gpt_w, C.gpt_f, np.ones(len(C), bool)),
    }
    e1 = (C.g_conf < tconf).to_numpy()
    pol["EXPLORATORY guarded -> VLM if guarded conf < t_c(conf-only, 0.574)"] = (
        np.where(e1, C.gpt_w, C.g_h), np.where(e1, C.gpt_f, C.g_v), e1)
    e2 = ((C.g_conf < gate["t_c"]) | (C.rho < gate["r_min"])).to_numpy()
    pol["EXPLORATORY guarded -> VLM if guarded conf < t_c(gate) or rho < r_min (paper gate on guarded conf)"] = (
        np.where(e2, C.gpt_w, C.g_h), np.where(e2, C.gpt_f, C.g_v), e2)
    A_ = {k: ape(h, v) for k, (h, v, _) in pol.items()}
    base_g = A_["ours guarded (no VLM)"]
    base_c = A_["cascade (paper): unguarded ours -> VLM if conf < t_c or rho < r_min"]
    rows = []
    for name, (h, v, esc) in pol.items():
        a = A_[name]
        rng = np.random.default_rng(SEED)
        (bm, bdg, bdc), _ = cluster_boot([a, a - base_g, a - base_c], C.key.to_numpy(), rng)
        per = pd.Series(a).groupby(C.tag.to_numpy()).mean()
        rows.append(dict(policy=name, MAPE=a.mean(), lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5),
                         d_vs_guarded=a.mean() - base_g.mean(), dg_lo=np.percentile(bdg, 2.5), dg_hi=np.percentile(bdg, 97.5),
                         d_vs_cascade=a.mean() - base_c.mean(), dc_lo=np.percentile(bdc, 2.5), dc_hi=np.percentile(bdc, 97.5),
                         escalated_pct=100 * esc.mean(), usd_per_1k=float(np.where(esc, C.gpt_cost, 0).sum() / len(C) * 1000),
                         worst_cond=str(per.idxmax()), worst_MAPE=float(per.max()),
                         **{f"MAPE_{t}": float(per.get(t, np.nan)) for t in A.CASCADE_CONDS},
                         stored_inputs_match_this_run=same))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "ph_cascade_policies.csv"), index=False)
    return T


def flag_vs_failure(df):
    """Flags of the guarded output as detectors of a guarded count off by > 10 % (either axis)."""
    d = df[df.group.isin(["genai", "cascade"])].copy()
    d["fail"] = (d.ape_g_h > 10) | (d.ape_g_v > 10)
    d["fam"] = np.where(d.group == "cascade", "cascade legacy-13 (19 conditions incl. clean)",
                        np.where(d.restorer == "none", "genai native LR ('none')",
                                 np.where(d.restorer == "clean", "genai clean refs", "genai up-sampled / restored")))
    rows = []
    for fam, x in list(d.groupby("fam")) + [("ALL genai + cascade", d)]:
        for flag in ("insufficient", "upsampled", "unsupported", "fired"):
            f = x[flag].astype(bool)
            nf = int(x.fail.sum())
            rows.append(dict(set=fam, flag=flag, n=len(x), n_fail=nf, flagged_pct=100 * f.mean(),
                             tpr=100 * (f & x.fail).sum() / max(1, nf),
                             fpr=100 * (f & ~x.fail).sum() / max(1, (~x.fail).sum()),
                             precision=100 * (f & x.fail).sum() / max(1, f.sum())))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "ph_flag_vs_failure.csv"), index=False)
    return T


def ungated_summary_repro(df):
    P = pd.read_csv(os.path.join(RES, "real_posthoc_ungated.csv"))
    P["key"] = P.key.astype(str)
    df = df.copy()
    df["key"] = df.key.astype(str)
    m = df.merge(P[["group", "set", "key", "cond", "restorer", "ung_h", "ung_v"]],
                 on=["group", "set", "key", "cond", "restorer"], how="left")
    extra = m.ung_h.notna()
    m["u2_h"] = np.where(extra, m.ung_h, m.g_h)
    m["u2_v"] = np.where(extra, m.ung_v, m.g_v)
    for ax in ("h", "v"):
        a = 100 * (m[f"u2_{ax}"] / m[f"ref_{ax}"] - 1).abs()
        m[f"ape_u2_{ax}"] = a.where(m[f"u2_{ax}"].notna(), 100.0)
    m["ape_u2"] = 0.5 * (m.ape_u2_h + m.ape_u2_v)
    rows = []
    for (grp, s, rest, cond), d in m[m.group != "clean150"].groupby(["group", "set", "restorer", "cond"]):
        n_extra = int(d.ung_h.notna().sum())
        if n_extra == 0:
            continue
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(d), size=(B, len(d)))
        diff = (d.ape_u2 - d.ape_g).to_numpy()
        bd = diff[idx].mean(1)
        rows.append(dict(group=grp, set=s, restorer=rest, cond=cond, n=len(d), n_extra_fires=n_extra,
                         mape_unguarded=d.ape_ug.mean(), mape_frozen_gated=d.ape_g.mean(),
                         mape_posthoc_ungated=d.ape_u2.mean(), d_ungated_minus_gated=diff.mean(),
                         d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "real_posthoc_ungated_summary_repro.csv"), index=False)
    old = pd.read_csv(os.path.join(RES, "real_posthoc_ungated_summary.csv"))
    k = ["group", "set", "restorer", "cond"]
    c = old.merge(T, on=k, suffixes=("_old", "_new"))
    cmp = {col: float(np.nanmax(np.abs(c[f"{col}_old"] - c[f"{col}_new"])))
           for col in ("n_extra_fires", "mape_unguarded", "mape_frozen_gated", "mape_posthoc_ungated",
                       "d_ungated_minus_gated", "d_lo", "d_hi")}
    return T, dict(rows_old=len(old), rows_new=len(T), rows_matched=len(c), max_abs_diff=cmp)


# ------------------------------------------------------------------------------------------- rectified
def _rect_job(i):
    sys.path.insert(0, os.path.join(ROOT, "common"))
    import rectify as R  # noqa: E402
    import native_res_guard as G  # noqa: E402
    t0 = time.time()
    crop = R.rectify(i)
    C = G.frozen_counter()
    bw = G.estimate_bandwidth(crop)
    ung = C.estimate(crop, method="ours")
    gd = G.count_guarded(crop, _unguarded=ung, _bandwidth=bw)
    g = gd["guard"]
    return dict(id=i, S=int(crop.shape[1]), ug_h=ung["h_count"], ug_v=ung["v_count"], ug_conf=ung["confidence"],
                g_h=gd["h_count"], g_v=gd["v_count"], fired=g["fired"], eligible=g["eligible"], flag=g["flag"],
                upsampled=g["upsampled"], insufficient=g["insufficient_resolution"], dead_band=g["dead_band"],
                frac_nyq=g["frac_nyq"], depth=g["depth"], n_eff=g["n_eff"], target=g["target"],
                seconds=round(time.time() - t0, 2))


def rectified(workers):
    t0 = time.time()
    with Pool(workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        rows = pool.map(_rect_job, range(150), chunksize=1)
    df = pd.DataFrame(rows).sort_values("id")
    s = pd.read_csv(os.path.join(METHODS, "predictions_ours_150.csv"))
    df = df.merge(s[["id", "h_count", "v_count", "confidence"]].rename(
        columns={"h_count": "stored_h", "v_count": "stored_v", "confidence": "stored_conf"}), on="id")
    df["repro_rel"] = np.fmax((df.ug_h / df.stored_h - 1).abs(), (df.ug_v / df.stored_v - 1).abs())
    chg = np.fmax((df.g_h / df.ug_h - 1).abs(), (df.g_v / df.ug_v - 1).abs())
    df["change"] = np.where(np.isfinite(chg), chg, np.where(df.fired, 1.0, 0.0))
    df.to_csv(os.path.join(RES, "ph_rectified150.csv"), index=False)
    sc = scene_of()
    df["scene"] = df.id.map(sc)
    summ = dict(n=len(df), fired=int(df.fired.sum()), eligible=int(df.eligible.sum()),
                chg1=int((df.change > 0.01).sum()), chg5=int((df.change > 0.05).sum()),
                scenes_fired=int(df.groupby("scene").fired.any().sum()),
                flag_upsampled=int(df.upsampled.sum()), flag_insufficient=int(df.insufficient.sum()),
                repro_identical_1e6=int((df.repro_rel < 1e-6).sum()), repro_max_rel=float(df.repro_rel.max()),
                wall_s=round(time.time() - t0))
    print("rectified150:", summ)
    json.dump(summ, open(os.path.join(RES, "ph_rectified150_summary.json"), "w"), indent=1)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--rectified", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    if a.tables:
        df = A.load()
        print("== R-HR, scene-cluster bootstrap\n", rhr_scene_cluster(df).round(2).to_string())
        T, fl = clean150_scenes(df)
        print("== clean 150 by scene\n", T.to_string(), "\n", fl.round(3).to_string())
        print("== defocus detail\n", defocus_detail(df).round(2).to_string())
        print("== cascade policies\n", cascade_policies(df).round(2).to_string())
        print("== flags vs failure\n", flag_vs_failure(df).round(1).to_string())
        T, cmp = ungated_summary_repro(df)
        print("== post hoc ungated summary re-derivation:", cmp)
    if a.rectified:
        rectified(a.workers)


if __name__ == "__main__":
    main()
