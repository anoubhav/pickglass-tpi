"""ruler_summary.py -- numbers, CIs and paired tests for the ruler self-calibration (Job 3).

Reads methods/results/ruler/{calib150,count150,perturb150,photodeg150}.csv (+ calib150.pkl) and
the existing self-consistency results (results/all150_crops.csv: segmentation crops;
results/all150_degrade.csv: corner perturbation without the ruler).  Writes
  methods/results/ruler/summary.json            every number quoted in the paper text
  paper/tables/ruler_calib.csv / ruler_calib.tex   drop-in table (accuracy by resolution group,
                                                   downstream agreement)
No thread labels are used anywhere.

    <venv2>/bin/python ruler_summary.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
import ruler_calib as RC  # noqa: E402

RES = os.path.join(HERE, "results", "ruler")
TAB = os.path.join(ROOT, "paper", "tables")
B = 10000
rng = np.random.default_rng(0)


def boot_ci(x, fn=np.median, b=B):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan, np.nan]
    idx = rng.integers(0, len(x), (b, len(x)))
    v = np.array([fn(x[i]) for i in idx])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def mcnemar(a, b):
    """Exact McNemar on paired booleans a, b."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def duplicate_groups(thr=0.995):
    """Groups of photos that are the same exposure (resized / re-encoded copies): Pearson
    correlation >= thr between 128x96 grey thumbnails (portrait photos rotated to landscape).
    Observed gap: duplicates >= 0.9999, distinct photos <= 0.942.  Cached in duplicate_groups.json."""
    path = os.path.join(RES, "duplicate_groups.json")
    if os.path.exists(path):
        return json.load(open(path))["groups"]
    import cv2
    import rectify as R
    sig = {}
    for i in range(R.N_IMAGES):
        im = cv2.cvtColor(R.load_image(i), cv2.COLOR_BGR2GRAY).astype(np.float32)
        land = im.shape[1] >= im.shape[0]
        if not land:
            im = im.T[::-1]
        g = cv2.resize(im, (128, 96), interpolation=cv2.INTER_AREA)
        sig[i] = ((g - g.mean()) / (g.std() + 1e-6), land)
    par = list(range(R.N_IMAGES))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a in range(R.N_IMAGES):
        for b in range(a + 1, R.N_IMAGES):
            if sig[a][1] == sig[b][1] and float((sig[a][0] * sig[b][0]).mean()) > thr:
                par[f(a)] = f(b)
    G = {}
    for i in range(R.N_IMAGES):
        G.setdefault(f(i), []).append(i)
    groups = [sorted(v) for v in G.values() if len(v) > 1]
    json.dump({"groups": groups, "unique_scenes": len(G)}, open(path, "w"))
    return groups


def main():
    out = {}
    cal = pd.read_csv(os.path.join(RES, "calib150.csv"))
    out["n_photos"] = int(len(cal))
    # ---------------------------------------------------------------- calibration accuracy
    acc = {}
    for g, d in [("all", cal)] + list(cal.groupby("group")):
        ok = d[d.ok]
        side_err = 100 * (ok.side_ratio_mean - 1)
        acc[g] = {
            "n": int(len(d)), "ok": int(len(ok)), "ok_pct": 100 * len(ok) / len(d),
            "ok_ci": wilson(len(ok), len(d)),
            "cerr_mean_median_mm": float(ok.cerr_mean.median()), "cerr_mean_median_ci": boot_ci(ok.cerr_mean),
            "cerr_all_corners_median_mm": float(np.median(ok[[f"cerr{j}" for j in range(4)]].values)),
            "cerr_max_median_mm": float(ok.cerr_max.median()), "cerr_max_p90_mm": float(ok.cerr_max.quantile(.9)),
            "cerr_max_le_0p5_pct": float(100 * (ok.cerr_max <= 0.5).mean()),
            "cerr_max_le_1_pct": float(100 * (ok.cerr_max <= 1.0).mean()),
            "side_err_mean_pct": float(side_err.mean()), "side_err_sd_pct": float(side_err.std()),
            "side_abs_err_median_pct": float(side_err.abs().median()),
            "rms_tick_mm_median": float(ok.rms_mm.median()),
            "n_ticks_median": float(ok.n_ticks.median()), "coverage_median": float(ok.coverage.median()),
            "runtime_s_median": float(d.runtime_s.median()),
        }
    out["accuracy"] = acc
    # signed per-edge offsets (ruler edge vs human edge, outward positive), mm
    K = pickle.load(open(os.path.join(RES, "calib150.pkl"), "rb"))
    import rectify as R
    offs = []
    for i, k in K.items():
        if not k:
            continue
        gt = R.load_corners(i)
        c = k["corners"]
        mm = 25.4 / np.linalg.norm(gt - np.roll(gt, -1, 0), axis=1).mean()
        cen = gt.mean(0)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            m = (gt[a] + gt[b]) / 2
            t = (gt[b] - gt[a]) / np.linalg.norm(gt[b] - gt[a])
            n = np.array([-t[1], t[0]])
            if (cen - m) @ n > 0:
                n = -n
            u = c[b] - c[a]
            s_, _ = np.linalg.solve(np.array([n, -u]).T, c[a] - m)
            offs.append(s_ * mm)
    offs = np.array(offs)
    out["edge_offset_mm"] = {"median": float(np.median(offs)), "mean": float(offs.mean()), "sd": float(offs.std()),
                             "median_ci": boot_ci(offs), "n_edges": int(len(offs))}
    # ---------------------------------------------------------------- glass constants
    ok = cal[cal.ok]
    gc = {}
    for col in ("W_mm", "r10_free", "W_free"):
        x = ok[col].dropna().values
        gc[col] = {"median": float(np.median(x)), "ci": boot_ci(x), "mean": float(x.mean()), "sd": float(x.std()),
                   "n": int(len(x))}
        for half, sel in (("even", ok.id % 2 == 0), ("odd", ok.id % 2 == 1)):
            xh = ok.loc[sel, col].dropna().values
            gc[col][half] = {"median": float(np.median(xh)), "ci": boot_ci(xh), "n": int(len(xh))}
        for grp, d in ok.groupby("group"):
            gc[col][grp] = {"median": float(d[col].median()), "ci": boot_ci(d[col]), "n": int(d[col].notna().sum())}
    # 2.54 vs 2.5: one-sample Wilcoxon of r10_free
    x = ok.r10_free.dropna().values
    gc["r10_test"] = {"wilcoxon_vs_2.54_p": float(stats.wilcoxon(x - 2.54).pvalue),
                      "wilcoxon_vs_2.50_p": float(stats.wilcoxon(x - 2.50).pvalue),
                      "share_closer_to_2.50": float(np.mean(np.abs(x - 2.5) < np.abs(x - 2.54)))}
    gc["window_inch_if_fine_is_mm"] = gc["W_mm"]["median"] / 25.4
    gc["window_inch_if_coarse_is_0.1in"] = gc["W_mm"]["median"] / (10 * gc["r10_free"]["median"])
    # pitch in human-polygon units (polygon taken as 25.4 mm)
    for sc in ("mm", "inch"):
        v = []
        for sd in RC.SIDES:
            m = ok[f"{sd}_scale"] == sc
            v += list(ok.loc[m, f"{sd}_pitch_gtmm"].dropna())
        gc[f"pitch_polygon_mm_{sc}"] = {"median": float(np.median(v)), "ci": boot_ci(v), "n": len(v)}
    out["glass"] = gc
    # ---------------------------------------------------------------- quality score validity
    q = ok[["qscore", "cerr_max", "rms_mm", "W_dev"]].dropna()
    out["quality"] = {"spearman_q_vs_cerrmax": float(stats.spearmanr(q.qscore, q.cerr_max)[0]),
                      "spearman_p": float(stats.spearmanr(q.qscore, q.cerr_max)[1]),
                      "n_flag_q_lt_0.05": int((q.qscore < 0.05).sum()),
                      "flagged_cerr_max": [float(v) for v in q[q.qscore < 0.05].cerr_max],
                      "unflagged_cerr_max_gt_1mm": int(((q.qscore >= 0.05) & (q.cerr_max > 1)).sum()),
                      "unflagged_cerr_max_p90": float(q[q.qscore >= 0.05].cerr_max.quantile(.9)),
                      "unflagged_n": int((q.qscore >= 0.05).sum())}
    # ---------------------------------------------------------------- lens-distortion residual pattern
    end_res, mid_res = [], []
    for i, k in K.items():
        if not k:
            continue
        fit = RC.fit_metric(k["obs"], k["corners_lines"], r10=RC.GLASS["r10"], W=None)
        for sd, o in k["obs"].items():
            ax = RC.EDGE_AX[sd][0]
            P = RC._homog_apply(fit["H"], o["feet"])
            u = P[:, ax]
            r = fit["res_mm"][sd]
            pos = np.abs(u / fit["W"] - 0.5)  # 0 = mid-edge, 0.5 = corner
            end_res += list(r[pos > 0.4])
            mid_res += list(r[pos < 0.2])
    out["residual_mm"] = {"end_ticks_median": float(np.median(end_res)), "mid_ticks_median": float(np.median(mid_res)),
                          "n_end": len(end_res), "n_mid": len(mid_res)}
    # ---------------------------------------------------------------- downstream counting
    cnt = pd.read_csv(os.path.join(RES, "count150.csv"))
    n = len(cnt)
    ad = cnt.absdrift.fillna(np.inf)
    ds = {"n": n, "median_absdrift_pct": float(100 * np.median(ad)), "median_ci": [100 * v for v in boot_ci(ad)]}
    for t in (0.02, 0.05, 0.10):
        k_ = int((ad <= t).sum())
        ds[f"within{int(t * 100)}_pct"] = 100 * k_ / n
        ds[f"within{int(t * 100)}_ci"] = wilson(k_, n)
    # by resolution group
    cnt = cnt.merge(cal[["id", "group"]], on="id")
    for g, d in cnt.groupby("group"):
        a_ = d.absdrift.fillna(np.inf)
        ds[g] = {"n": int(len(d)), "within5_pct": float(100 * (a_ <= .05).mean()),
                 "within2_pct": float(100 * (a_ <= .02).mean()), "median_absdrift_pct": float(100 * np.median(a_))}
    # ideal-counter drift implied by the geometry alone (side ratio), for reference
    ds["geom_side_ratio_abs_median_pct"] = float(100 * (cal[cal.ok].side_ratio_mean - 1).abs().median())
    # large drifts: factor between ruler and GT counts (hypothesis flips)
    big = cnt[cnt.absdrift > 0.05]
    fac = np.where(big.drift_h.abs() >= big.drift_v.abs(), big.h_ruler / big.h_gt, big.v_ruler / big.v_gt)
    ds["n_drift_gt5"] = int(len(big))
    ds["drift_gt5_factors"] = sorted([float(round(v, 2)) for v in fac])
    ds["drift_gt5_cerr_max_median"] = float(cal.set_index("id").loc[big.id, "cerr_max"].median()) if len(big) else None
    # paired comparisons: segmentation crops (results/all150_crops.csv) and corner perturbation 2 %
    seg = pd.read_csv(os.path.join(HERE, "results", "all150_crops.csv"))
    seg = seg[(seg.estimator == "ours") & (seg.kind == "clean")]
    ref = seg[seg.crop == "gt"][["image", "h", "v"]].rename(columns={"h": "h0", "v": "v0"})
    sg = seg[seg.crop == "seg_final"].merge(ref, on="image")
    sg["absdrift_seg"] = np.fmax((sg.h / sg.h0 - 1).abs(), (sg.v / sg.v0 - 1).abs()).fillna(np.inf)
    m = cnt.merge(sg[["image", "absdrift_seg"]], left_on="id", right_on="image")
    a5, s5 = (m.absdrift.fillna(np.inf) <= .05), (m.absdrift_seg <= .05)
    ds["vs_segmentation"] = {"n": int(len(m)), "ruler_within5_pct": float(100 * a5.mean()),
                             "seg_within5_pct": float(100 * s5.mean()),
                             "seg_within10_pct": float(100 * (m.absdrift_seg <= .10).mean()),
                             "seg_within2_pct": float(100 * (m.absdrift_seg <= .02).mean()),
                             "mcnemar_within5": mcnemar(a5, s5),
                             "wilcoxon_absdrift_p": float(stats.wilcoxon(np.minimum(m.absdrift.fillna(9), 9),
                                                                         np.minimum(m.absdrift_seg, 9)).pvalue)}
    deg = pd.read_csv(os.path.join(HERE, "results", "all150_degrade.csv"))
    deg = deg[(deg.estimator == "ours") & (deg.set == "all150")]
    c0 = deg[deg.kind == "clean"][["image", "h", "v"]].rename(columns={"h": "h0", "v": "v0"})
    for lev in (0.02, 0.05, 0.10):
        dd = deg[(deg.kind == "corners") & (np.isclose(deg.level, lev))].merge(c0, on="image")
        dd["ad"] = np.fmax((dd.h / dd.h0 - 1).abs(), (dd.v / dd.v0 - 1).abs()).fillna(np.inf)
        mm_ = cnt.merge(dd[["image", "ad"]], left_on="id", right_on="image")
        r5, p5 = mm_.absdrift.fillna(np.inf) <= .05, mm_.ad <= .05
        ds[f"vs_corners{int(lev * 100):02d}"] = {"n": int(len(mm_)), "pert_within5_pct": float(100 * p5.mean()),
                                                 "ruler_within5_pct": float(100 * r5.mean()),
                                                 "mcnemar_within5": mcnemar(r5, p5)}
    # noise floor of the counter under imperceptible changes (same CSV)
    for kind, lev in (("jpeg", 90), ("noise", 5)):
        dd = deg[(deg.kind == kind) & (np.isclose(deg.level, lev))].merge(c0, on="image")
        dd["ad"] = np.fmax((dd.h / dd.h0 - 1).abs(), (dd.v / dd.v0 - 1).abs()).fillna(np.inf)
        ds[f"floor_{kind}{lev}_within5_pct"] = float(100 * (dd.ad <= .05).mean())
    out["downstream"] = ds
    # ---------------------------------------------------------------- perturbation
    pp = os.path.join(RES, "perturb150.csv")
    if os.path.exists(pp):
        pt = pd.read_csv(pp).merge(cnt[["id", "h_gt", "v_gt", "h_ruler", "v_ruler"]], on="id")
        pr = {}
        for lev, d in pt.groupby("level"):
            adp = np.fmax((d.h_pert / d.h_gt - 1).abs(), (d.v_pert / d.v_gt - 1).abs()).fillna(np.inf)
            adr = np.fmax((d.h_rinit / d.h_gt - 1).abs(), (d.v_rinit / d.v_gt - 1).abs()).fillna(np.inf)
            same = np.isclose(d.h_rinit, d.h_ruler) & np.isclose(d.v_rinit, d.v_ruler)
            pr[f"{lev:.2f}"] = {
                "n": int(len(d)), "cerr_pert_mean_median_mm": float(d.cerr_pert_mean.median()),
                "ruler_ok_pct": float(100 * d.ruler_ok.mean()),
                "cerr_ruler_mean_median_mm": float(d.cerr_ruler_mean.median()),
                "pert_within5_pct": float(100 * (adp <= .05).mean()), "pert_within5_ci": wilson(int((adp <= .05).sum()), len(d)),
                "pert_within2_pct": float(100 * (adp <= .02).mean()), "rinit_within2_pct": float(100 * (adr <= .02).mean()),
                "rinit_within5_pct": float(100 * (adr <= .05).mean()), "rinit_within5_ci": wilson(int((adr <= .05).sum()), len(d)),
                "pert_median_absdrift_pct": float(100 * np.median(adp)), "rinit_median_absdrift_pct": float(100 * np.median(adr)),
                "same_as_auto_pct": float(100 * same.mean()),
                "mcnemar_within5": mcnemar(adr <= .05, adp <= .05)}
        out["perturb"] = pr
    # ---------------------------------------------------------------- photo degradations
    pd_ = os.path.join(RES, "photodeg150.csv")
    if os.path.exists(pd_):
        ph = pd.read_csv(pd_).merge(cal[["id", "group"]], on="id")
        pdg = {}
        for (kind, lev), d in ph.groupby(["kind", "level"]):
            e = {"ok_pct": float(100 * d.ok.mean()), "cerr_mean_median_mm": float(d.cerr_mean.median()),
                 "cerr_max_le_0p5_pct_of_all": float(100 * (d.cerr_max <= 0.5).mean())}
            for g, dg in d.groupby("group"):
                e[g] = {"ok_pct": float(100 * dg.ok.mean()), "cerr_mean_median_mm": float(dg.cerr_mean.median())}
            pdg[f"{kind}_{lev}"] = e
        out["photodeg"] = pdg
    # ---------------------------------------------------------------- ruler blur meter vs. confidence
    bp = os.path.join(RES, "blur150.csv")
    if os.path.exists(bp):
        from sklearn.metrics import roc_auc_score
        bl = pd.read_csv(bp).merge(cal[["id", "group"]], on="id")
        bl["blur_gate"] = bl.blur_u.where(bl.ok, np.inf)        # no calibration -> flagged
        g0 = deg[deg.kind == "gblur"][["image", "level", "h", "v", "conf"]].merge(c0, on="image")
        g0["ad"] = np.fmax((g0.h / g0.h0 - 1).abs(), (g0.v / g0.v0 - 1).abs()).fillna(np.inf)
        j = bl[bl.level > 0].merge(g0, left_on=["id", "level"], right_on=["image", "level"])
        j["unstable"] = j.ad > 0.05
        bm = {"n_pairs": int(len(j)), "n_unstable": int(j.unstable.sum()),
              "blur_u_by_level": {str(k): float(v) for k, v in bl.groupby("level").blur_u.median().items()},
              "ok_by_level": {str(k): float(100 * v) for k, v in bl.groupby("level").ok.mean().items()},
              "spearman_blur_vs_level": float(stats.spearmanr(bl.level, bl.blur_gate.replace(np.inf, 99))[0])}
        yb = j.blur_gate.replace(np.inf, 99).values
        yc = -j.conf.fillna(0).values
        y = j.unstable.values
        bm["auc_blur"] = float(roc_auc_score(y, yb))
        bm["auc_conf"] = float(roc_auc_score(y, yc))
        ids = j.id.unique()
        by = {i: np.nonzero(j.id.values == i)[0] for i in ids}
        diffs, ab, ac = [], [], []
        for _ in range(2000):
            samp = rng.choice(ids, len(ids))
            ix = np.concatenate([by[i] for i in samp])
            if y[ix].all() or (~y[ix]).all():
                continue
            a1, a2 = roc_auc_score(y[ix], yb[ix]), roc_auc_score(y[ix], yc[ix])
            ab.append(a1)
            ac.append(a2)
            diffs.append(a1 - a2)
        bm["auc_blur_ci"] = [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))]
        bm["auc_conf_ci"] = [float(np.percentile(ac, 2.5)), float(np.percentile(ac, 97.5))]
        bm["auc_diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        for grp, d in j.groupby("group"):
            yy = d.unstable.values
            if yy.any() and (~yy).any():
                bm[f"auc_blur_{grp}"] = float(roc_auc_score(yy, d.blur_gate.replace(np.inf, 99)))
                bm[f"auc_conf_{grp}"] = float(roc_auc_score(yy, -d.conf.fillna(0)))
        # label-free gate: 95th percentile of the clean photos' blur estimate
        tau = float(np.nanpercentile(bl[(bl.level == 0) & bl.ok].blur_u, 95))
        flag = j.blur_gate > tau
        bm["gate_tau_u"] = tau
        bm["gate_recall_unstable_pct"] = float(100 * flag[j.unstable].mean())
        bm["gate_flag_stable_pct"] = float(100 * flag[~j.unstable].mean())
        bm["gate_flag_clean_pct"] = float(100 * (bl[bl.level == 0].blur_gate > tau).mean())
        # same false-flag rate for the confidence, for comparison
        thr_c = np.nanpercentile(-j.conf[~j.unstable].fillna(0), 100 - bm["gate_flag_stable_pct"])
        bm["conf_recall_at_same_flag_rate_pct"] = float(100 * ((-j.conf.fillna(0)) > thr_c)[j.unstable].mean())
        for lev, d in j.groupby("level"):
            bm[f"level{lev}"] = {"unstable_pct": float(100 * d.unstable.mean()),
                                 "gate_flag_pct": float(100 * (d.blur_gate > tau).mean()),
                                 "mean_conf": float(d.conf.mean())}
        # POST-HOC (added after seeing auc_blur < auc_conf): blur in yarn periods, using the count
        # returned on the same blurred image (deployment-time information only)
        rel = yb * np.fmax(j.h, j.v).fillna(999).values / 1000.0
        bm["posthoc_auc_blur_x_count"] = float(roc_auc_score(y, rel))
        out["blur_meter"] = bm
    # ---------------------------------------------------------------- de-duplicated photos
    groups = duplicate_groups()
    drop = sorted(i for g_ in groups for i in g_[1:])
    keep = cal[~cal.id.isin(drop)]
    okd = keep[keep.ok]
    kc = cnt[~cnt.id.isin(drop)]
    md = kc.merge(sg[["image", "absdrift_seg"]], left_on="id", right_on="image")
    a5d, s5d = (md.absdrift.fillna(np.inf) <= .05), (md.absdrift_seg <= .05)
    out["dedup"] = {"n_groups": len(groups), "n_photos_in_groups": int(sum(map(len, groups))),
                    "n_unique": int(len(keep)), "dropped_ids": drop, "groups": groups,
                    "ok": int(len(okd)), "ok_by_group": {g_: [int(d.ok.sum()), int(len(d))] for g_, d in keep.groupby("group")},
                    "cerr_mean_median_mm": float(okd.cerr_mean.median()), "cerr_mean_median_ci": boot_ci(okd.cerr_mean),
                    "cerr_max_le_0p5_pct": float(100 * (okd.cerr_max <= .5).mean()),
                    "W_median": float(okd.W_mm.median()), "W_ci": boot_ci(okd.W_mm),
                    "r10_median": float(okd.r10_free.median()), "r10_ci": boot_ci(okd.r10_free),
                    "r10_vs_2.54_p": float(stats.wilcoxon(okd.r10_free.dropna() - 2.54).pvalue),
                    "within5_pct": float(100 * (kc.absdrift.fillna(np.inf) <= .05).mean()),
                    "within5_ci": wilson(int((kc.absdrift.fillna(np.inf) <= .05).sum()), len(kc)),
                    "within2_pct": float(100 * (kc.absdrift.fillna(np.inf) <= .02).mean()),
                    "seg_within5_pct": float(100 * s5d.mean()), "mcnemar_vs_seg": mcnemar(a5d, s5d)}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    # ---------------------------------------------------------------- drop-in table
    rows = []
    for g, lab in (("high", r"Ruler, $\approx$1890\,px/in"), ("low", r"Ruler, $\approx$485\,px/in"), ("all", "Ruler, all")):
        a = acc[g]
        dsg = ds[g] if g in ds else ds
        rows.append({"source": lab, "n": a["n"], "calibrated": a["ok"],
                     "corner_err_mm": f"{a['cerr_mean_median_mm']:.2f}",
                     "corner_err_ci": a["cerr_mean_median_ci"],
                     "within5": dsg["within5_pct"], "within2": dsg["within2_pct"]})
    vs = ds["vs_segmentation"]
    rows.append({"source": "2019 segmentation", "n": vs["n"], "calibrated": None, "corner_err_mm": "--",
                 "corner_err_ci": None, "within5": vs["seg_within5_pct"], "within2": vs["seg_within2_pct"]})
    if "perturb" in out:
        pr = out["perturb"]
        L = ("0.02", "0.05", "0.10")
        rows.append({"source": "Annotated + 2/5/10\\% error", "n": 150, "calibrated": None,
                     "corner_err_mm": "/".join(f"{pr[l]['cerr_pert_mean_median_mm']:.2f}" for l in L), "corner_err_ci": None,
                     "within5": "/".join(f"{pr[l]['pert_within5_pct']:.0f}" for l in L),
                     "within2": "/".join(f"{pr[l]['pert_within2_pct']:.0f}" for l in L)})
        rows.append({"source": "Ruler started there", "n": 150, "calibrated": None,
                     "corner_err_mm": "/".join(f"{pr[l]['cerr_ruler_mean_median_mm']:.2f}" for l in L), "corner_err_ci": None,
                     "within5": "/".join(f"{pr[l]['rinit_within5_pct']:.0f}" for l in L),
                     "within2": "/".join(f"{pr[l]['rinit_within2_pct']:.0f}" for l in L)})
    t = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    t.to_csv(os.path.join(TAB, "ruler_calib.csv"), index=False)

    def f(v):
        return v if isinstance(v, str) else f"{v:.0f}"
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"Window from & Corner err.\ [mm] & \multicolumn{2}{c}{Count within (\%)} \\",
             r" & median & 5\% & 2\% \\", r"\midrule"]
    for _, r in t.iterrows():
        ce = r["corner_err_mm"]
        if r["corner_err_ci"] is not None and isinstance(r["corner_err_ci"], list):
            ce = f"{ce} [{r['corner_err_ci'][0]:.2f}, {r['corner_err_ci'][1]:.2f}]"
        cal_txt = f" ({int(r['calibrated'])}/{int(r['n'])})" if r['calibrated'] is not None and not pd.isna(r['calibrated']) else ""
        lines.append(f"{r['source']}{cal_txt} & {ce} & {f(r['within5'])} & {f(r['within2'])} \\\\")
        if r["source"] == "Ruler, all":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "ruler_calib.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
