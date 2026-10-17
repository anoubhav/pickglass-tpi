"""analyze.py -- all tables of the genai_sr study (README sections 6-7, deviation D1).

    <venv2>/bin/python analyze.py            # reads results/*.csv (synced from gpu-server) -> results/table_*.csv

Reference counts: R-HR = ours on the clean reference (primary, for every estimator) and each estimator's own clean
prediction (secondary, '_self'); R-LEG = human counts (h <-> warp, v <-> weft).  Per-count APE in %.
CIs: percentile bootstrap over images (both counts together), B = 10 000, seed 20261008.  Paired tests against
'bicubic' (pre-registered control) and 'none' (deviation D1): paired-bootstrap CI of dMAPE, Wilcoxon signed-rank on
per-image mean APE, exact McNemar on per-count within-10 %; Holm within (set, estimator, comparator).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

import gsr_common as G

RES = os.path.join(G.GSR, "results")
B = 10_000
GEN = ["swinir", "realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
ORDER = ["clean", "none", "bicubic"] + GEN
FAMILY = {"none": "degraded (no restoration)", "bicubic": "interpolation (control)", "swinir": "regression SR",
          "realesrgan": "GAN SR", "seedvr2": "one-step diffusion", "invsr": "diffusion inversion",
          "flux2klein": "flow editor (open)", "nbpro": "commercial AI editor", "clean": "reference (clean)"}
NAME = {"none": "Degraded (LR, no restoration)", "bicubic": "Bicubic", "swinir": "SwinIR-M x4",
        "realesrgan": "Real-ESRGAN x4plus", "seedvr2": "SeedVR2-3B", "invsr": "InvSR (1 step)",
        "flux2klein": "FLUX.2 klein 4B", "nbpro": "Nano Banana Pro", "clean": "Reference (clean)"}
ESTS = ["ours", "gabor", "cnn"]


def load_counts():
    cl = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "counts_classical_*.csv"))], ignore_index=True)
    cl = cl[["key", "set", "cond", "restorer", "ours_h", "ours_v", "ours_conf", "gabor_h", "gabor_v"]]
    cn = [pd.read_csv(p) for p in glob.glob(os.path.join(RES, "counts_cnn_*.csv"))]
    if cn:
        cn = pd.concat(cn, ignore_index=True)[["key", "cond", "restorer", "cnn_h", "cnn_v"]]
        cl = cl.merge(cn, on=["key", "cond", "restorer"], how="left")
    else:
        cl["cnn_h"] = cl["cnn_v"] = np.nan
    return cl


def references(df):
    clean = df[df.restorer == "clean"].set_index("key")
    gt = pd.read_csv(os.path.join(G.INPUTS, "gt_legacy13.csv")).set_index("key")
    ref = {}
    for k in clean.index:
        if k.startswith("hr"):
            ref[k] = dict(ref_h=clean.loc[k, "ours_h"], ref_v=clean.loc[k, "ours_v"])
        else:
            ref[k] = dict(ref_h=gt.loc[k, "warp"], ref_v=gt.loc[k, "weft"])
        for e in ESTS:
            ref[k][f"self_{e}_h"], ref[k][f"self_{e}_v"] = clean.loc[k, f"{e}_h"], clean.loc[k, f"{e}_v"]
    return pd.DataFrame(ref).T


def long_ape(df, ref, self_ref=False):
    """one row per (key, set, cond, restorer, estimator, axis) with pred, ref, ape, conf."""
    rows = []
    for e in ESTS:
        for ax in ("h", "v"):
            t = df[["key", "set", "cond", "restorer", f"{e}_{ax}", "ours_conf"]].rename(columns={f"{e}_{ax}": "pred"})
            t["estimator"], t["axis"] = e, ax
            col = f"self_{e}_{ax}" if self_ref else f"ref_{ax}"
            t["ref"] = t.key.map(ref[col].astype(float))
            rows.append(t)
    L = pd.concat(rows, ignore_index=True)
    L["ape"] = (L.pred / L.ref - 1).abs() * 100
    L["spe"] = (L.pred / L.ref - 1) * 100
    return L


def boot_ci(per_img, rng):
    """per_img: (n_img, 2) APE array (nan allowed) -> MAPE CI over images."""
    n = len(per_img)
    if n == 0:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(B, n))
    s = np.nansum(per_img[idx], axis=(1, 2))
    c = np.sum(np.isfinite(per_img[idx]), axis=(1, 2))
    m = s / np.maximum(c, 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(L, tag):
    rng = np.random.default_rng(G.SEED)
    out = []
    for (s, c, r, e), g in L.groupby(["set", "cond", "restorer", "estimator"]):
        piv = g.pivot_table(index="key", columns="axis", values="ape", dropna=False)
        arr = piv[["h", "v"]].to_numpy(float)
        lo, hi = boot_ci(arr, rng)
        a = g.ape.to_numpy(float)
        fin = np.isfinite(a)
        out.append(dict(ref=tag, set=s, cond=c, restorer=r, estimator=e, n_img=len(piv), n_counts=int(fin.sum()),
                        n_fail=int((~fin).sum()), MAPE=np.nanmean(a), MAPE_lo=lo, MAPE_hi=hi, medAPE=np.nanmedian(a),
                        w5=np.mean(a[fin] <= 5) if fin.any() else np.nan, w10=np.mean(a[fin] <= 10) if fin.any() else np.nan,
                        gross25=np.mean(a[fin] > 25) if fin.any() else np.nan, bias=np.nanmean(g.spe)))
    return pd.DataFrame(out)


def mcnemar_exact(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def paired(L, comparator, tag):
    rng = np.random.default_rng(G.SEED + 1)
    out = []
    for (s, e), g in L.groupby(["set", "estimator"]):
        for c in G.CONDS:
            base = g[(g.cond == c) & (g.restorer == comparator)].pivot_table(index="key", columns="axis", values="ape")
            for r in ["bicubic"] + GEN if comparator == "none" else GEN:
                if r == comparator:
                    continue
                t = g[(g.cond == c) & (g.restorer == r)].pivot_table(index="key", columns="axis", values="ape")
                keys = t.index.intersection(base.index)
                if len(keys) < 3:
                    continue
                A, Bm = t.loc[keys, ["h", "v"]].to_numpy(float), base.loc[keys, ["h", "v"]].to_numpy(float)
                ok = np.isfinite(A) & np.isfinite(Bm)
                A, Bm = np.where(ok, A, np.nan), np.where(ok, Bm, np.nan)
                d = np.nanmean(A) - np.nanmean(Bm)
                idx = rng.integers(0, len(keys), size=(B, len(keys)))
                ds = np.nanmean(A[idx].reshape(B, -1), axis=1) - np.nanmean(Bm[idx].reshape(B, -1), axis=1)
                pa, pb = np.nanmean(A, axis=1), np.nanmean(Bm, axis=1)
                fin = np.isfinite(pa) & np.isfinite(pb)
                try:
                    pw = stats.wilcoxon(pa[fin], pb[fin]).pvalue if np.any(pa[fin] != pb[fin]) else 1.0
                except ValueError:
                    pw = 1.0
                a10, b10 = (A <= 10)[ok], (Bm <= 10)[ok]
                out.append(dict(ref=tag, set=s, estimator=e, cond=c, restorer=r, comparator=comparator, n_img=len(keys),
                                MAPE=np.nanmean(A), MAPE_comp=np.nanmean(Bm), dMAPE=d,
                                dMAPE_lo=np.percentile(ds, 2.5), dMAPE_hi=np.percentile(ds, 97.5),
                                w10=a10.mean(), w10_comp=b10.mean(), p_wilcoxon=pw, p_mcnemar=mcnemar_exact(a10, b10)))
    P = pd.DataFrame(out)
    if len(P):
        P["p_wilcoxon_holm"] = np.nan
        for _, g in P.groupby(["ref", "set", "estimator", "comparator"]):
            P.loc[g.index, "p_wilcoxon_holm"] = holm(g.p_wilcoxon.values)
        P["verdict"] = np.where(P.dMAPE_lo > 0, "worse", np.where(P.dMAPE_hi < 0, "better", "n.s."))
    return P


def drift(df, ref):
    """count drift of each restoration vs bicubic and vs none, per estimator (ours, gabor, cnn)."""
    out = []
    for e in ESTS:
        for comp in ("bicubic", "none"):
            base = df[df.restorer == comp].set_index(["key", "cond"])
            for r in ["bicubic"] + GEN:
                if r == comp:
                    continue
                t = df[df.restorer == r].set_index(["key", "cond"])
                idx = t.index.intersection(base.index)
                for (s, c), g in t.loc[idx].groupby(["set", "cond"]):
                    ii = g.index
                    rows = []
                    for ax in ("h", "v"):
                        cm, cb = g[f"{e}_{ax}"].astype(float), base.loc[ii, f"{e}_{ax}"].astype(float)
                        rv = np.array([ref.loc[k, f"ref_{ax}"] for k, _ in ii], float)
                        dr = (cm / cb - 1).abs() * 100
                        toward = (cm - rv).abs() < (cb - rv).abs()
                        rows.append(pd.DataFrame(dict(dr=dr.values, toward=toward.values, changed=dr.values > 5)))
                    R = pd.concat(rows)
                    out.append(dict(estimator=e, comparator=comp, set=s, cond=c, restorer=r, n_counts=len(R),
                                    mean_drift=np.nanmean(R.dr), median_drift=np.nanmedian(R.dr),
                                    share_gt5=np.mean(R.dr > 5), share_gt10=np.mean(R.dr > 10),
                                    share_changed_toward=np.mean(R.toward[R.changed]) if R.changed.any() else np.nan))
    return pd.DataFrame(out)


def silent(L):
    """among crops where ours misses by > 10 % on either axis: share with ours confidence >= t_c."""
    o = L[L.estimator == "ours"]
    w = o.pivot_table(index=["set", "cond", "restorer", "key"], columns="axis", values="ape")
    conf = o.groupby(["set", "cond", "restorer", "key"]).ours_conf.first()
    w["bad"] = (w[["h", "v"]].max(axis=1) > 10)
    w["conf_hi"] = conf >= G.T_C
    out = []
    for (s, c, r), g in w.groupby(level=[0, 1, 2]):
        out.append(dict(set=s, cond=c, restorer=r, n_img=len(g), n_bad=int(g.bad.sum()),
                        n_bad_confident=int((g.bad & g.conf_hi).sum()),
                        share_bad_confident=float((g.bad & g.conf_hi).sum() / g.bad.sum()) if g.bad.any() else np.nan,
                        accepted=int(g.conf_hi.sum()),
                        accepted_bad=int((g.bad & g.conf_hi).sum()),
                        risk_accepted=float((g.bad & g.conf_hi).sum() / g.conf_hi.sum()) if g.conf_hi.any() else np.nan))
    return pd.DataFrame(out)


def lattice_and_iq():
    lat = [pd.read_csv(p) for p in glob.glob(os.path.join(RES, "lattice_*.csv")) if not p.endswith("lattice_refs.csv")]
    iq = [pd.read_csv(p) for p in glob.glob(os.path.join(RES, "iq_*.csv"))]
    lat = pd.concat(lat, ignore_index=True) if lat else pd.DataFrame()
    iq = pd.concat(iq, ignore_index=True) if iq else pd.DataFrame()
    out = []
    if len(lat):
        for ax in ("x", "y"):
            lat[f"dom_{ax}_bad"] = (lat[f"dom_{ax}_rel"].abs() > 0.05).astype(float)
            lat[f"domP_{ax}_bad"] = (lat[f"domP_{ax}_rel"].abs() > 0.05).astype(float)
        lat["dom_bad"] = lat[["dom_x_bad", "dom_y_bad"]].mean(axis=1)
        lat["domP_bad"] = lat[["domP_x_bad", "domP_y_bad"]].mean(axis=1)
        for (s, c, r), g in lat.groupby(["set", "cond", "restorer"]):
            row = dict(set=s, cond=c, restorer=r, n_img=len(g), hpef_mean=g.hpef.mean(), hpef_median=g.hpef.median(),
                       hpef_novel_mean=g.hpef_novel.mean(), hpef_replica_mean=g.hpef_replica.mean(),
                       n_novel_mean=g.n_novel.mean(), share_img_novel=float((g.n_novel > 0).mean()),
                       erase_below_mean=g.erase_below.mean(), resynth_above_mean=g.resynth_above.mean(),
                       n_img_above=int(g.n_lat_above.fillna(0).gt(0).sum()),
                       dom_shift_gt5=g.dom_bad.mean(), domP_shift_gt5=g.domP_bad.mean())
            for comp in ("bicubic", "none"):
                if r == comp:
                    continue
                b = lat[(lat.set == s) & (lat.cond == c) & (lat.restorer == comp)].set_index("key")
                gg = g.set_index("key")
                k = gg.index.intersection(b.index)
                for m in ("hpef", "hpef_novel", "erase_below"):
                    x, y = gg.loc[k, m].astype(float), b.loc[k, m].astype(float)
                    f = np.isfinite(x) & np.isfinite(y)
                    try:
                        p = stats.wilcoxon(x[f], y[f]).pvalue if np.any(x[f] != y[f]) else 1.0
                    except ValueError:
                        p = np.nan
                    row[f"p_{m}_vs_{comp}"] = p
                    row[f"d_{m}_vs_{comp}"] = float(np.mean(x[f] - y[f])) if f.any() else np.nan
            out.append(row)
    LAT = pd.DataFrame(out)
    out = []
    if len(iq):
        for (s, c, r), g in iq.groupby(["set", "cond", "restorer"]):
            row = dict(set=s, cond=c, restorer=r, n_img=len(g))
            for m in ("psnr_y", "ssim_y", "lpips_alex", "musiq", "clipiqa"):
                if m in g:
                    row[m] = g[m].mean()
            if r not in ("bicubic", "clean"):
                b = iq[(iq.set == s) & (iq.cond == c) & (iq.restorer == "bicubic")].set_index("key")
                gg = g.set_index("key")
                k = gg.index.intersection(b.index)
                for m in ("lpips_alex", "musiq", "clipiqa", "psnr_y"):
                    if m in gg and len(k) > 2:
                        x, y = gg.loc[k, m].astype(float), b.loc[k, m].astype(float)
                        row[f"d_{m}_vs_bicubic"] = float((x - y).mean())
                        row[f"p_{m}_vs_bicubic"] = stats.wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            out.append(row)
    IQ = pd.DataFrame(out)
    return LAT, IQ


def gpt_table():
    """GPT-6.1-sol P1 on R-LEG D8: this study's runs (bicubic / seedvr2 / nbpro restorations) + the paper's stored runs
    (clean crop, legacy_x8 = LR resized straight to 1536 px) from vlm/results/legacy_predictions.csv, repeat 0."""
    import json
    gt = pd.read_csv(os.path.join(G.INPUTS, "gt_legacy13.csv")).set_index("image")
    rows = []
    for p in glob.glob(os.path.join(G.GSR, "raw", "gpt-6.1-sol__*_D8.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r.get("error"):
                continue
            pr = r["parsed"]
            rows.append(dict(source=f"{r['restorer']} (this job)", image=r["key"][4:], vertical=pr.get("vertical"),
                             horizontal=pr.get("horizontal"), usd=r.get("cost_usd")))
    v = pd.read_csv(os.path.join(G.WVAQ, "vlm", "results", "legacy_predictions.csv"))
    v = v[(v.model == "gpt-6.1-sol") & (v.repeat == 0) & v.method.isin(["P1", "P1x8"]) & v.set.isin(["legacy", "legacy_x8"])]
    for r in v.itertuples():
        rows.append(dict(source={"legacy": "clean crop (paper run)", "legacy_x8": "LR x8 -> 1536 bicubic (paper run)"}[r.set],
                         image=r.image, vertical=r.vertical, horizontal=r.horizontal, usd=r.usd))
    D = pd.DataFrame(rows)
    D = D[D.image.isin(gt.index)]
    D["warp"], D["weft"] = D.image.map(gt.warp), D.image.map(gt.weft)
    out, rng = [], np.random.default_rng(G.SEED + 2)
    for src, g in D.groupby("source"):
        for assign in ("axis", "sorted"):
            a = g[["vertical", "horizontal"]].astype(float).to_numpy()
            t = g[["warp", "weft"]].to_numpy(float)
            if assign == "sorted":
                a, t = np.sort(a, axis=1), np.sort(t, axis=1)
            ape = np.abs(a / t - 1) * 100
            lo, hi = boot_ci(ape, rng)
            out.append(dict(source=src, assign=assign, n_img=len(g), MAPE=np.nanmean(ape), MAPE_lo=lo, MAPE_hi=hi,
                            w10=np.nanmean(ape <= 10), usd_total=g.usd.astype(float).sum()))
    return pd.DataFrame(out), D


def seedvar(df, ref):
    """EXPLORATORY (D3): spread of ours' count over the 3 seeds (20261008, 1, 2) for the same LR input."""
    out = []
    for m in ("flux2klein", "invsr", "seedvr2"):
        names = [m, f"{m}_s1", f"{m}_s2"]
        t = df[df.restorer.isin(names)]
        if t.restorer.nunique() < 3:
            continue
        for e in ESTS:
            for ax in ("h", "v"):
                w = t.pivot_table(index=["set", "cond", "key"], columns="restorer", values=f"{e}_{ax}").reindex(columns=names).dropna()
                rng_rel = (w.max(axis=1) / w.min(axis=1) - 1) * 100
                rv = np.array([ref.loc[k, f"ref_{ax}"] for _, _, k in w.index], float)
                ok = (np.abs(w.to_numpy() / rv[:, None] - 1) <= 0.10)
                for (s, c), g in pd.DataFrame(dict(r=rng_rel.values, flip=(ok.any(1) & ~ok.all(1))),
                                              index=w.index).groupby(level=[0, 1]):
                    out.append(dict(model=m, estimator=e, axis=ax, set=s, cond=c, n=len(g),
                                    median_range_pct=g.r.median(), share_range_gt5=np.mean(g.r > 5),
                                    share_range_gt10=np.mean(g.r > 10), share_correct_flips=g.flip.mean()))
    return pd.DataFrame(out)


def main():
    df = load_counts()
    ref = references(df)
    df_all = df
    df = df[~df.restorer.str.contains("_s")]
    L = long_ape(df, ref)
    Ls = long_ape(df, ref, self_ref=True)
    T = pd.concat([summarise(L, "primary"), summarise(Ls[Ls.set == "R-HR"], "self")], ignore_index=True)
    T["family"] = T.restorer.map(FAMILY)
    T["o"] = T.restorer.map({r: i for i, r in enumerate(ORDER)})
    T = T.sort_values(["ref", "set", "estimator", "cond", "o"]).drop(columns="o")
    T.to_csv(os.path.join(RES, "table_counts.csv"), index=False)
    P = pd.concat([paired(L, "bicubic", "primary"), paired(L, "none", "primary"),
                   paired(Ls[Ls.set == "R-HR"], "none", "self")], ignore_index=True)
    P.to_csv(os.path.join(RES, "table_paired.csv"), index=False)
    D = drift(df, ref)
    D.to_csv(os.path.join(RES, "table_drift.csv"), index=False)
    S = silent(L)
    S.to_csv(os.path.join(RES, "table_silent.csv"), index=False)
    LAT, IQ = lattice_and_iq()
    LAT.to_csv(os.path.join(RES, "table_lattice.csv"), index=False)
    IQ.to_csv(os.path.join(RES, "table_iq.csv"), index=False)
    L.to_csv(os.path.join(RES, "per_count_long.csv"), index=False)
    GPT, GD = gpt_table()
    GPT.to_csv(os.path.join(RES, "table_gpt_p1.csv"), index=False)
    GD.to_csv(os.path.join(RES, "gpt_p1_predictions.csv"), index=False)
    SV = seedvar(df_all, ref)
    SV.to_csv(os.path.join(RES, "table_seedvar.csv"), index=False)
    # compact console view
    pd.set_option("display.width", 250)
    v = T[(T.ref == "primary") & (T.estimator == "ours")][["set", "cond", "restorer", "n_img", "MAPE", "MAPE_lo",
                                                           "MAPE_hi", "w10", "bias"]]
    print(v.round(1).to_string(index=False))
    return T, P, D, S, LAT, IQ


if __name__ == "__main__":
    main()
