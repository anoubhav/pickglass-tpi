"""check/recompute_headlines.py -- independent re-computation of the genai_sr headline numbers.

Reads only the raw per-image CSVs in genai_sr/results (counts_*, lattice_*, iq_*, gpt_p1_predictions.csv),
inputs/gt_legacy13.csv, and the original project files (predictions_ours_150.csv, crops manifest). It does NOT
import analyze.py, so it is an independent re-implementation. Bootstrap uses a different seed (12345) on purpose:
verdicts that flip under a different seed are borderline and are reported as such.

    <venv2>/bin/python check/recompute_headlines.py  > check/recompute_headlines.out
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

GSR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WVAQ = os.path.dirname(GSR)
RES = os.path.join(GSR, "results")
GENS = ["realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]
T_C = 0.5740764461064712
B = 10_000


def load():
    cl = []
    for p in glob.glob(os.path.join(RES, "counts_classical_*.csv")):
        cl.append(pd.read_csv(p)[["key", "set", "cond", "restorer", "ours_h", "ours_v", "ours_conf", "gabor_h", "gabor_v"]])
    cl = pd.concat(cl, ignore_index=True)
    cn = pd.concat([pd.read_csv(p)[["key", "cond", "restorer", "cnn_h", "cnn_v"]]
                    for p in glob.glob(os.path.join(RES, "counts_cnn_*.csv"))], ignore_index=True)
    df = cl.merge(cn, on=["key", "cond", "restorer"], how="left", validate="one_to_one")
    return df


def ape_table(df, est, set_name, self_ref):
    """returns dict[(cond, restorer)] -> DataFrame index=key cols h,v (APE %)."""
    gt = pd.read_csv(os.path.join(GSR, "inputs", "gt_legacy13.csv")).set_index("key")
    clean = df[(df.restorer == "clean")].set_index("key")
    d = df[(df.set == set_name)]
    out = {}
    for (c, r), g in d.groupby(["cond", "restorer"]):
        g = g.set_index("key")
        t = pd.DataFrame(index=g.index)
        for ax, gcol in (("h", "warp"), ("v", "weft")):
            if set_name == "R-LEG":
                ref = gt.loc[g.index, gcol].astype(float)
            elif self_ref:
                ref = clean.loc[g.index, f"{est}_{ax}"].astype(float)
            else:
                ref = clean.loc[g.index, f"ours_{ax}"].astype(float)
            t[ax] = (g[f"{est}_{ax}"].astype(float) / ref - 1).abs() * 100
        t["conf"] = g["ours_conf"]
        out[(c, r)] = t
    return out


def main():
    rng = np.random.default_rng(12345)
    df = load()
    df = df[~df.restorer.str.contains("_s")]
    print("rows", len(df), "restorers", sorted(df.restorer.unique()))
    # ---------------- completeness: every LR x restorer scored? ----------------
    man = pd.read_csv(os.path.join(GSR, "inputs", "manifest.csv"))
    print("\n[completeness] LR inputs:", len(man))
    for r in ["none", "bicubic", "swinir", "realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]:
        n = (df.restorer == r).sum()
        nn = df[df.restorer == r][["ours_h", "ours_v"]].isna().any(axis=1).sum()
        nc = df[df.restorer == r][["cnn_h", "cnn_v"]].isna().any(axis=1).sum()
        print(f"  {r:11s} rows={n:4d}  ours-NaN={nn}  cnn-NaN={nc}")
    # ---------------- R-HR selection ----------------
    pr = pd.read_csv(os.path.join(WVAQ, "methods", "predictions_ours_150.csv"))
    sel = sorted(pr[(pr.S >= 1500) & (pr.confidence >= T_C)].id.astype(int).tolist())
    hr = sorted(int(k[2:]) for k in df[df.set == "R-HR"].key.unique())
    print("\n[R-HR selection] recomputed n=%d, used n=%d, identical=%s; S>=1500 total=%d" %
          (len(sel), len(hr), sel == hr, (pr.S >= 1500).sum()))

    # ---------------- Headline 1: ours MAPE, none vs bicubic at D8 ----------------
    print("\n[H1] ours MAPE (primary ref) with 95% bootstrap CI over images")
    tabs = {s: ape_table(df, "ours", s, False) for s in ("R-LEG", "R-HR")}
    def mape_ci(t):
        a = t[["h", "v"]].to_numpy(float)
        idx = rng.integers(0, len(a), size=(B, len(a)))
        m = np.nanmean(a[idx].reshape(B, -1), axis=1)
        return np.nanmean(a), np.percentile(m, 2.5), np.percentile(m, 97.5), len(a)
    for s in ("R-LEG", "R-HR"):
        for c in ("clean", "D4", "D8"):
            for r in (["clean"] if c == "clean" else ["none", "bicubic", "swinir", "seedvr2", "realesrgan", "nbpro"]):
                if (c, r) in tabs[s]:
                    m, lo, hi, n = mape_ci(tabs[s][(c, r)])
                    print(f"  {s:5s} {c:5s} {r:11s} n={n:2d} MAPE={m:6.1f} [{lo:6.1f},{hi:6.1f}]")
    # outlier diagnosis on R-LEG clean
    t = tabs["R-LEG"][("clean", "clean")]
    print("  R-LEG clean per-crop APE (h,v):", {k: (round(a, 1), round(b, 1)) for k, a, b in zip(t.index, t.h, t.v)})
    t = tabs["R-LEG"][("D8", "none")]
    print("  R-LEG D8 none per-crop APE (h,v):", {k: (round(a, 1), round(b, 1)) for k, a, b in zip(t.index, t.h, t.v)})

    # ---------------- Headline 2: verdict tally vs none ----------------
    print("\n[H2] verdict tally vs 'none' (paired bootstrap CI of dMAPE excludes 0), seed 12345")
    rows = []
    for e in ("ours", "gabor", "cnn"):
        for s in ("R-LEG", "R-HR"):
            for selfref in ((False,) if s == "R-LEG" else (True, False)):
                T = ape_table(df, e, s, selfref)
                for c in ("D4", "D8", "D4J"):
                    base = T[(c, "none")]
                    for r in ["bicubic", "swinir"] + GENS:
                        if (c, r) not in T:
                            continue
                        t = T[(c, r)]
                        k = t.index.intersection(base.index)
                        A, Bm = t.loc[k, ["h", "v"]].to_numpy(float), base.loc[k, ["h", "v"]].to_numpy(float)
                        ok = np.isfinite(A) & np.isfinite(Bm)
                        A, Bm = np.where(ok, A, np.nan), np.where(ok, Bm, np.nan)
                        idx = rng.integers(0, len(k), size=(B, len(k)))
                        ds = np.nanmean(A[idx].reshape(B, -1), 1) - np.nanmean(Bm[idx].reshape(B, -1), 1)
                        lo, hi = np.percentile(ds, [2.5, 97.5])
                        pa, pb = np.nanmean(A, 1), np.nanmean(Bm, 1)
                        f = np.isfinite(pa) & np.isfinite(pb)
                        pw = stats.wilcoxon(pa[f], pb[f]).pvalue if np.any(pa[f] != pb[f]) else 1.0
                        rows.append(dict(est=e, set=s, ref="self" if selfref else "primary", cond=c, restorer=r,
                                         n=len(k), dMAPE=np.nanmean(A) - np.nanmean(Bm), lo=lo, hi=hi, p_w=pw))
    P = pd.DataFrame(rows)
    P["verdict"] = np.where(P.lo > 0, "worse", np.where(P.hi < 0, "better", "n.s."))
    # Holm within (est, set, ref) over restorations x conds, as in analyze.py (bicubic + swinir + GENS)
    P["p_holm"] = np.nan
    for _, g in P.groupby(["est", "set", "ref"]):
        p = g.p_w.to_numpy()
        o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
        for i, j in enumerate(o):
            run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
        P.loc[g.index, "p_holm"] = adj
    P["verdict_holm"] = np.where(P.p_holm < 0.05, np.where(P.dMAPE > 0, "worse", "better"), "n.s.")
    P["grp"] = P.restorer.map(lambda r: "generative" if r in GENS else r)
    main_ref = ((P.set == "R-LEG") & (P.ref == "primary")) | ((P.set == "R-HR") & (P.ref == "self"))
    for lab, msk in (("paper reference (R-LEG hand, R-HR self)", main_ref),
                     ("all-primary reference (R-HR = ours clean for every estimator)",
                      (P.ref == "primary"))):
        print("  --", lab)
        for gname in ("generative", "swinir", "bicubic"):
            g = P[msk & (P.grp == gname)]
            print(f"   {gname:10s} n={len(g):3d} CI-verdict {g.verdict.value_counts().to_dict()}  "
                  f"Holm-Wilcoxon {g.verdict_holm.value_counts().to_dict()}")
        g = P[msk & (P.grp == "generative")]
        for e in ("ours", "gabor", "cnn"):
            print(f"     generative/{e}: {g[g.est == e].verdict.value_counts().to_dict()}  Holm {g[g.est == e].verdict_holm.value_counts().to_dict()}")
    # compare with stored table_paired verdicts (same reference choice)
    SP = pd.read_csv(os.path.join(RES, "table_paired.csv"))
    SP = SP[SP.comparator == "none"]
    SP["ref2"] = SP.ref
    mm = P.merge(SP, left_on=["est", "set", "ref", "cond", "restorer"], right_on=["estimator", "set", "ref2", "cond", "restorer"],
                 suffixes=("", "_stored"))
    flips = mm[mm.verdict != mm.verdict_stored]
    print(f"  verdict agreement with stored table_paired: {len(mm) - len(flips)}/{len(mm)}; flips:")
    for r in flips.itertuples():
        print(f"     {r.est} {r.set} {r.ref} {r.cond} {r.restorer}: mine {r.verdict} [{r.lo:+.2f},{r.hi:+.2f}] vs stored {r.verdict_stored} [{r.dMAPE_lo:+.2f},{r.dMAPE_hi:+.2f}]")
    print("  max |dMAPE - stored| =", float((mm.dMAPE - mm.dMAPE_stored).abs().max()))
    # key ours rows
    for s, c, r in (("R-HR", "D8", "seedvr2"), ("R-HR", "D8", "realesrgan"), ("R-LEG", "D8", "seedvr2"),
                    ("R-LEG", "D8", "nbpro"), ("R-HR", "D8", "flux2klein"), ("R-LEG", "D8", "bicubic")):
        g = P[main_ref & (P.est == "ours") & (P.set == s) & (P.cond == c) & (P.restorer == r)].iloc[0]
        print(f"  ours {s} {c} {r}: dMAPE {g.dMAPE:+.1f} [{g.lo:+.1f},{g.hi:+.1f}] n={g.n} Holm p={g.p_holm:.3g}")

    # ---------------- Headline 3: novel off-lattice energy + MUSIQ ----------------
    print("\n[H3] lattice: mean hpef_novel / hpef at D8; share of crops with >=1 novel peak")
    lat = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "lattice_*.csv"))
                     if not p.endswith("lattice_refs.csv") and "_s1" not in p and "_s2" not in p], ignore_index=True)
    for s in ("R-HR", "R-LEG"):
        for r in ["bicubic", "swinir"] + GENS:
            g = lat[(lat.set == s) & (lat.cond == "D8") & (lat.restorer == r)]
            if len(g):
                print(f"  {s:5s} {r:11s} n={len(g):2d} novel={100 * g.hpef_novel.mean():5.1f}%  hpef={100 * g.hpef.mean():5.1f}%  "
                      f"img_with_novel={100 * (g.n_novel > 0).mean():5.1f}%  median_novel={100 * g.hpef_novel.median():4.1f}%")
    g1 = lat[(lat.set == "R-HR") & (lat.cond == "D8") & (lat.restorer == "realesrgan")].set_index("key")
    g2 = lat[(lat.set == "R-HR") & (lat.cond == "D8") & (lat.restorer == "bicubic")].set_index("key")
    k = g1.index.intersection(g2.index)
    print("  Real-ESRGAN vs bicubic novel, R-HR D8: Wilcoxon p =", stats.wilcoxon(g1.loc[k, "hpef_novel"], g2.loc[k, "hpef_novel"]).pvalue)
    print("\n[H3b] MUSIQ / PSNR at D8 vs clean reference")
    iq = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "iq_*.csv"))], ignore_index=True)
    for s in ("R-HR", "R-LEG"):
        cl = iq[(iq.set == s) & (iq.restorer == "clean")].set_index("key")
        print(f"  {s} clean MUSIQ mean {cl.musiq.mean():.2f}")
        for c in ("D4", "D8", "D4J"):
            for r in ["bicubic", "swinir"] + GENS:
                g = iq[(iq.set == s) & (iq.cond == c) & (iq.restorer == r)].set_index("key")
                if not len(g):
                    continue
                k = g.index.intersection(cl.index)
                d = g.loc[k, "musiq"] - cl.loc[k, "musiq"]
                p = stats.wilcoxon(d).pvalue
                if c == "D8" or r in GENS:
                    print(f"   {c:3s} {r:11s} MUSIQ {g.musiq.mean():5.1f} (vs clean {d.mean():+5.1f}, Wilcoxon p={p:.2g}, "
                          f"share>clean {100 * (d > 0).mean():3.0f}%)  PSNR-Y {g.psnr_y.mean():5.1f}  LPIPS {g.lpips_alex.mean():.2f}")

    # ---------------- silent failures ----------------
    print("\n[H4] silent failures (ours APE>10% on either axis, conf>=t_c)")
    nb = nc = n0b = n0c = 0
    for s in ("R-LEG", "R-HR"):
        for (c, r), t in tabs[s].items():
            bad = t[["h", "v"]].max(axis=1) > 10
            conf = t.conf >= T_C
            if r in GENS:
                nb += bad.sum(); nc += (bad & conf).sum()
            if r == "none":
                n0b += bad.sum(); n0c += (bad & conf).sum()
    print(f"  generative: {nc}/{nb} = {100 * nc / nb:.1f}%   none: {n0c}/{n0b} = {100 * n0c / n0b:.1f}%")
    print("  Fisher exact (generative vs none):", stats.fisher_exact([[nc, nb - nc], [n0c, n0b - n0c]]))

    # ---------------- GPT ----------------
    print("\n[H5] GPT-6.1-sol P1, R-LEG D8, axis assignment (vertical<->warp, horizontal<->weft as in analyze.py)")
    G = pd.read_csv(os.path.join(RES, "gpt_p1_predictions.csv"))
    G["ape_w"] = (G.vertical / G.warp - 1).abs() * 100
    G["ape_f"] = (G.horizontal / G.weft - 1).abs() * 100
    G["mean_ape"] = G[["ape_w", "ape_f"]].mean(axis=1)
    piv = G.pivot_table(index="image", columns="source", values="mean_ape")
    print(G.groupby("source")[["ape_w", "ape_f"]].apply(lambda x: pd.Series(dict(n=len(x), MAPE=np.nanmean(x.to_numpy())))).round(2))
    cols = list(piv.columns)
    b = [c for c in cols if c.startswith("bicubic")][0]
    for c in cols:
        if c == b:
            continue
        k = piv[[b, c]].dropna()
        d = (k[c] - k[b]).to_numpy()
        idx = rng.integers(0, len(d), size=(B, len(d)))
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        p = stats.wilcoxon(k[c], k[b]).pvalue if np.any(d != 0) else 1
        print(f"  {c:40s} - bicubic: d={d.mean():+.2f} [{lo:+.2f},{hi:+.2f}] n={len(d)} Wilcoxon p={p:.2f}")


if __name__ == "__main__":
    main()
