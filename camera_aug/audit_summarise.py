"""audit_summarise.py -- tables of the camera_sim label-safety audit (see label_audit.py for the protocol).

Reads   results/audit_real150_raw.csv          isolated stages / transports / no-op baselines (ours)
        results/stress_real150_classical.csv   full pipelines = the real150 stress renders (ours)
        results/audit_sweep_raw.csv            px-per-yarn sweep
        results/audit_gratings.csv             analytic gratings
        methods/predictions_ours_150.csv + originals_provenance.json   real full / quarter-res pairs
Writes  results/label_audit_summary.csv, results/label_audit_sweep.csv, results/label_audit_pairs.csv,
        results/label_audit_thresholds.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
R_MIN = 18.0
SEED = 7
SYNTH = {131, 132, 134, 135, 136, 137, 138, 139, 141, 142, 144}


def scenes():
    prov = json.load(open(os.path.join(HERE, "originals_provenance.json")))
    return {str(i): gi for gi, g in enumerate(prov["groups"]) for i in g}


def long_drift(d, clean, fh="f_h", fv="f_v"):
    """-> one row per (source, cond, axis) with drift, applicability."""
    c = clean.set_index("source_id")
    rows = []
    for ax, col, fcol in (("h", "h", fh), ("v", "v", fv)):
        t = d[["source_id", "cond", "family", "S_src", "S_out", fcol, col]].copy()
        t.columns = ["source_id", "cond", "family", "S_src", "S_out", "factor", "obs"]
        t["axis"] = ax
        t["clean"] = t.source_id.map(c[col])
        t["pred"] = t.factor * t.clean
        t["drift"] = t.obs / t.pred - 1
        t["drift_ignore_label"] = t.obs / t.clean - 1
        t.loc[~np.isfinite(t.obs), "drift"] = 1.0           # failed run = 100 % drift
        nw = np.minimum(t.S_out, 1024)
        nw0 = np.minimum(t.S_src, 1024)
        t["applicable"] = (t.pred >= 1.1 * R_MIN) & (t.pred <= 0.40 * nw) & (t.clean <= 0.40 * nw0) & np.isfinite(t.clean)
        rows.append(t)
    x = pd.concat(rows, ignore_index=True)
    x["scene"] = x.source_id.astype(str).map(scenes())
    x["res"] = np.where(x.S_src >= 1500, "hi", "lo")
    return x


def stats_of(x):
    a = x.drift.abs()
    return dict(n=len(x), n_scenes=x.scene.nunique(), med_abs=100 * a.median(), p90_abs=100 * a.quantile(0.9),
                gt10=100 * (a > 0.10).mean(), med_signed=100 * x.drift.median(), mean_abs=100 * a.mean())


def scene_ci(x, B=2000):
    """cluster bootstrap over scenes of the mean |drift| (pp)."""
    g = x.groupby("scene").drift.apply(lambda s: s.abs().mean())
    if len(g) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    v = g.to_numpy()
    bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def pass_rule(s, f):
    return dict(pass_med=s["med_abs"] <= f["med_abs"] + 0.5, pass_p90=s["p90_abs"] <= f["p90_abs"] + 2.0,
                pass_gt10=s["gt10"] <= f["gt10"] + 2.0, pass_bias=abs(s["med_signed"]) <= 1.0)


def summary_real150():
    raw = pd.read_csv(os.path.join(RES, "audit_real150_raw.csv"), dtype={"source_id": str})
    clean = raw[raw.cond == "clean"][["source_id", "h", "v"]]
    parts = [raw[raw.cond != "clean"]]
    p_full = os.path.join(RES, "stress_real150_classical.csv")
    if os.path.exists(p_full):
        s = pd.read_csv(p_full, dtype={"source_id": str})
        s = s[(s.estimator == "ours") & (s.condition != "clean") & ~s.condition.str.startswith("noop")]
        man = pd.read_csv(os.path.join(HERE, "stress", "real150", "manifest.csv"), dtype={"source_id": str})
        s = s.merge(man[["source_id", "condition", "rep", "S_src", "S_out", "label_factor_h", "label_factor_v"]],
                    on=["source_id", "condition", "rep"])
        parts.append(pd.DataFrame(dict(source_id=s.source_id, cond="full:" + s.condition, family="full",
                                       S_src=s.S_src, S_out=s.S_out, f_h=s.label_factor_h, f_v=s.label_factor_v,
                                       h=s.h, v=s.v)))
    d = pd.concat(parts, ignore_index=True)
    x = long_drift(d, clean)
    x = x[~x.source_id.astype(int).isin(SYNTH)]           # the 11 synthetic PNG re-crops are not captures
    fl = {}
    for nm in ("noop_store", "noop_q75"):
        fl[nm] = stats_of(x[(x.cond == nm) & x.applicable])
    floor = {k: max(fl["noop_store"][k], fl["noop_q75"][k]) for k in ("med_abs", "p90_abs", "gt10")}
    rows = []
    for cond, g in x.groupby("cond"):
        for sub, gg in (("all", g), ("hi", g[g.res == "hi"]), ("lo", g[g.res == "lo"])):
            ap_ = gg[gg.applicable]
            if not len(ap_):
                continue
            s = stats_of(ap_)
            lo, hi = scene_ci(ap_)
            pr = pass_rule(s, floor)
            r = dict(cond=cond, family=gg.family.iloc[0], subset=sub, **s, mean_abs_lo=lo, mean_abs_hi=hi,
                     n_not_applicable=int((~gg.applicable).sum()), **pr, PASS=all(pr.values()),
                     factor_min=float(gg.factor.min()), factor_max=float(gg.factor.max()),
                     med_abs_if_label_ignored=100 * ap_.drift_ignore_label.abs().median())
            # tracking: slope of log(obs/clean) on log(factor) when the factor varies
            if gg.factor.std() > 1e-4:
                ok = ap_[np.isfinite(ap_.obs) & (ap_.obs > 0)]
                X, Y = np.log(ok.factor), np.log(ok.obs / ok.clean)
                r["tracking_slope"] = float(np.polyfit(X, Y, 1)[0]) if len(ok) > 5 else np.nan
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "label_audit_summary.csv"), index=False)
    x.to_csv(os.path.join(RES, "label_audit_long.csv.gz"), index=False)
    return out, floor


def summary_sweep(floor):
    p = os.path.join(RES, "audit_sweep_raw.csv")
    if not os.path.exists(p):
        return None, None
    d = pd.read_csv(p, dtype={"source_id": str})
    c = d[d.route == "clean"].set_index("source_id")
    gt = json.load(open(os.path.join(ROOT, "cnn", "aug_optics", "legacy13_gt.json")))
    rows = []
    for ax in ("h", "v"):
        t = d[d.route != "clean"].copy()
        t["axis"] = ax
        t["obs"] = t[ax]
        t["clean"] = t.source_id.map(c[ax])
        t["drift"] = t.obs / t.clean - 1
        t.loc[~np.isfinite(t.obs), "drift"] = 1.0
        gtc = {n: (g["warp"] if ax == "h" else g["weft"]) for n, g in gt.items()}
        t["gt_count"] = t.source_id.map(gtc)
        t["err_gt"] = t.obs / t["gt_count"] - 1
        # px per yarn of THIS axis
        t["ppt_axis"] = t.S_out / t.clean
        rows.append(t)
    x = pd.concat(rows, ignore_index=True)
    x = x[np.isfinite(x.clean) & (x.clean >= 1.1 * R_MIN)]
    x["ppt_bin"] = pd.cut(x.ppt_axis, [0, 1.875, 2.125, 2.375, 2.625, 2.875, 3.25, 3.75, 4.5, 5.5, 7, 100],
                          labels=["<1.9", "2.0", "2.25", "2.5", "2.75", "3.0", "3.5", "4.0", "5.0", "6.0", ">7"])
    rows = []
    for (route, b), g in x.groupby(["route", "ppt_bin"], observed=True):
        s = stats_of(g.assign(scene=g.source_id))
        pr = pass_rule(s, floor)
        lg = g[g.set == "legacy13"]
        rows.append(dict(route=route, ppt_bin=str(b), ppt_med=float(g.ppt_axis.median()), **s, **pr,
                         PASS=all(pr.values()), n_legacy=len(lg),
                         legacy_mape_vs_gt=100 * lg.err_gt.abs().mean() if len(lg) else np.nan))
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(RES, "label_audit_sweep.csv"), index=False)
    order = ["<1.9", "2.0", "2.25", "2.5", "2.75", "3.0", "3.5", "4.0", "5.0", "6.0", ">7"]
    # the strict PASS rule (no-op re-encode floor) fails even for pure INTER_AREA down-scaling at > 7 px/yarn:
    # ours' result depends on its working resolution for ~5 % of counts.  Two descriptive thresholds instead
    # (smallest bin from which EVERY larger bin satisfies the rule):
    #   collapse: median |drift| <= 3 %  (systematic failure / aliasing below it)
    #   plateau : share |drift| > 10 % <= resampling plateau + 5 pp, plateau = 'area' route pooled over >= 5 px/yarn
    pl = x[(x.route == "area") & (x.ppt_axis >= 4.5)]
    plateau = 100 * (pl.drift.abs() > 0.10).mean()
    sw["collapse_ok"] = sw.med_abs <= 3.0
    sw["plateau_ok"] = sw.gt10 <= plateau + 5.0
    thr = {"strict_PASS": {}, "collapse_median_abs_le_3pct": {}, "gross_le_plateau_plus_5pp": {},
           "resampling_plateau_gt10_pct": float(plateau)}
    for route, g in sw.groupby("route"):
        g = g.set_index("ppt_bin").reindex(order).dropna(subset=["n"])
        bins = list(g.index)
        for key, col in (("strict_PASS", "PASS"), ("collapse_median_abs_le_3pct", "collapse_ok"),
                         ("gross_le_plateau_plus_5pp", "plateau_ok")):
            ok = g[col].astype(bool).tolist()
            th = None
            for i in range(len(bins)):
                if all(ok[i:]):
                    th = bins[i]
                    break
            thr[key][route] = th
    sw.to_csv(os.path.join(RES, "label_audit_sweep.csv"), index=False)
    return sw, thr


def summary_pairs():
    prov = json.load(open(os.path.join(HERE, "originals_provenance.json")))
    o = pd.read_csv(os.path.join(ROOT, "methods", "predictions_ours_150.csv")).set_index("id")
    sc = scenes()
    rows = []
    for p in prov["fine_pairs"]:
        f, q, k = p["full_res"], p["quarter_res"], p["rot90_k"]
        if f in SYNTH or q in SYNTH:
            continue
        hf, vf = o.loc[f, "h_count"], o.loc[f, "v_count"]
        hq, vq = o.loc[q, "h_count"], o.loc[q, "v_count"]
        if k % 2 == 1:
            hq, vq = vq, hq
        for ax, a, b in (("h", hf, hq), ("v", vf, vq)):
            rows.append(dict(full_res=f, quarter_res=q, rot90_k=k, axis=ax, S_full=o.loc[f, "S"], S_quarter=o.loc[q, "S"],
                             full=a, quarter=b, drift=b / a - 1, ppt_quarter=o.loc[q, "S"] / a, scene=sc[str(f)]))
    d = pd.DataFrame(rows)
    # simulated counterpart: legacy_quarter render of the full-res crop (real150 stress set, ours)
    p_full = os.path.join(RES, "stress_real150_classical.csv")
    if os.path.exists(p_full):
        s = pd.read_csv(p_full, dtype={"source_id": str})
        s = s[s.estimator == "ours"]
        cl = s[s.condition == "clean"].set_index("source_id")
        for cond in ("legacy_quarter", "noop_q75", "budget_whatsapp"):
            q = s[s.condition == cond].set_index("source_id")
            for ax in ("h", "v"):
                m = d.axis == ax
                ids = d.loc[m, "full_res"].astype(str)
                d.loc[m, f"sim_{cond}"] = (q[ax].reindex(ids).to_numpy() / cl[ax].reindex(ids).to_numpy()) - 1
    # the matched simulation: the ISOLATED q75 x0.25 resave of the same full-res crop (label_audit real150 part)
    p_iso = os.path.join(RES, "audit_real150_raw.csv")
    if os.path.exists(p_iso):
        raw = pd.read_csv(p_iso, dtype={"source_id": str})
        cl = raw[raw.cond == "clean"].set_index("source_id")
        q = raw[raw.cond == "transport:resave_q75_x0.25"].set_index("source_id")
        for ax in ("h", "v"):
            m = d.axis == ax
            ids = d.loc[m, "full_res"].astype(str)
            d.loc[m, "sim_iso_resave_q75_x0.25"] = q[ax].reindex(ids).to_numpy() / cl[ax].reindex(ids).to_numpy() - 1
            d.loc[m, "S_sim"] = q["S_out"].reindex(ids).to_numpy()
    d.to_csv(os.path.join(RES, "label_audit_pairs.csv"), index=False)
    return d


def main():
    out, floor = summary_real150()
    pd.set_option("display.width", 220)
    cols = ["cond", "subset", "n", "n_scenes", "med_abs", "p90_abs", "gt10", "med_signed", "mean_abs_lo", "mean_abs_hi",
            "PASS", "n_not_applicable", "factor_min", "factor_max", "med_abs_if_label_ignored", "tracking_slope"]
    cols = [c for c in cols if c in out]
    print("floor (max of no-op baselines):", {k: round(v, 2) for k, v in floor.items()})
    print(out[out.subset == "all"][cols].round(2).to_string(index=False))
    print("\nby source resolution (hi >= 1500 px, lo ~490 px):")
    print(out[out.subset != "all"][["cond", "subset", "n", "med_abs", "p90_abs", "gt10", "med_signed", "PASS"]]
          .round(2).to_string(index=False))
    sw, thr = summary_sweep(floor)
    if sw is not None:
        print("\npx-per-yarn sweep:")
        print(sw[["route", "ppt_bin", "n", "med_abs", "p90_abs", "gt10", "med_signed", "PASS", "collapse_ok",
                  "plateau_ok", "legacy_mape_vs_gt"]].round(2).to_string(index=False))
        print("thresholds (px per yarn; smallest bin from which every larger bin satisfies the rule):", thr)
    pr = summary_pairs()
    a = pr.drift.abs()
    print(f"\nreal full/quarter pairs: {pr[['full_res', 'quarter_res']].drop_duplicates().shape[0]} pairs, "
          f"{pr.scene.nunique()} scenes; |drift| median {100 * a.median():.2f} %, p90 {100 * a.quantile(.9):.2f} %, "
          f">10 %: {100 * (a > .1).mean():.1f} %; signed median {100 * pr.drift.median():+.2f} %")
    for c in ("sim_iso_resave_q75_x0.25", "sim_legacy_quarter", "sim_noop_q75"):
        if c in pr:
            b = pr[c].abs().dropna()
            print(f"  simulated {c[4:]} on the same full-res crops: |drift| median {100 * b.median():.2f} %, "
                  f"p90 {100 * b.quantile(.9):.2f} %, >10 %: {100 * (b > .1).mean():.1f} %")
    json.dump(dict(floor=floor, thresholds=thr), open(os.path.join(RES, "label_audit_thresholds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
