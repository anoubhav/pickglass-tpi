"""make_paper_table.py -- one compact, paper-facing CSV (+ LaTeX snippet) from the analysis tables.

    <venv2>/bin/python make_paper_table.py
-> results/genai_hallucination_table.csv  (copied to paper/tables/genai_hallucination.csv)
   results/genai_hallucination_summary.json (headline counts used in the text)
   paper/tables/genai_hallucination.tex       (compact D8 table)

Reference per set: R-LEG = hand counts (all estimators); R-HR = each estimator's own clean prediction
('self'; for ours this equals the pre-registered primary pseudo-GT).  Deltas are against 'none' (the degraded crop
counted as is, deviation D1); 'vs_bicubic' columns keep the pre-registered control.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

import gsr_common as G
from analyze import FAMILY, NAME, ORDER

RES = os.path.join(G.GSR, "results")
GENS = ["realesrgan", "seedvr2", "invsr", "flux2klein", "nbpro"]


def main():
    T = pd.read_csv(os.path.join(RES, "table_counts.csv"))
    P = pd.read_csv(os.path.join(RES, "table_paired.csv"))
    D = pd.read_csv(os.path.join(RES, "table_drift.csv"))
    S = pd.read_csv(os.path.join(RES, "table_silent.csv"))
    LAT = pd.read_csv(os.path.join(RES, "table_lattice.csv"))
    IQ = pd.read_csv(os.path.join(RES, "table_iq.csv"))
    ref_of = lambda s: "primary" if s == "R-LEG" else "self"  # noqa: E731
    rows = []
    for (s, c, r), _ in T[(T.cond != "clean")].groupby(["set", "cond", "restorer"]):
        row = dict(set=s, cond=c, restorer=r, name=NAME[r], family=FAMILY[r])
        for e in ("ours", "gabor", "cnn"):
            t = T[(T.ref == ref_of(s)) & (T.set == s) & (T.cond == c) & (T.restorer == r) & (T.estimator == e)]
            if len(t):
                t = t.iloc[0]
                row.update({f"{e}_n_img": t.n_img, f"{e}_MAPE": t.MAPE, f"{e}_MAPE_lo": t.MAPE_lo, f"{e}_MAPE_hi": t.MAPE_hi,
                            f"{e}_w5": t.w5, f"{e}_w10": t.w10})
            for comp in ("none", "bicubic"):
                ref = ref_of(s) if comp == "none" else "primary"
                if comp == "bicubic" and s == "R-HR" and e != "ours":
                    continue  # pre-registered bicubic comparison is primary-reference only
                p = P[(P.ref == ref) & (P.set == s) & (P.cond == c) & (P.restorer == r) & (P.estimator == e)
                      & (P.comparator == comp)]
                if len(p):
                    p = p.iloc[0]
                    row.update({f"{e}_dMAPE_vs_{comp}": p.dMAPE, f"{e}_dMAPE_vs_{comp}_lo": p.dMAPE_lo,
                                f"{e}_dMAPE_vs_{comp}_hi": p.dMAPE_hi, f"{e}_verdict_vs_{comp}": p.verdict,
                                f"{e}_p_holm_vs_{comp}": p.p_wilcoxon_holm})
        d = D[(D.estimator == "ours") & (D.comparator == "none") & (D.set == s) & (D.cond == c) & (D.restorer == r)]
        if len(d):
            row.update(ours_drift_vs_none_gt10=d.iloc[0].share_gt10, ours_drift_vs_none_toward_ref=d.iloc[0].share_changed_toward)
        sl = S[(S.set == s) & (S.cond == c) & (S.restorer == r)]
        if len(sl):
            sl = sl.iloc[0]
            row.update(ours_n_wrong=sl.n_bad, ours_n_wrong_confident=sl.n_bad_confident,
                       ours_share_wrong_confident=sl.share_bad_confident)
        la = LAT[(LAT.set == s) & (LAT.cond == c) & (LAT.restorer == r)]
        if len(la):
            la = la.iloc[0]
            row.update(hpef=la.hpef_mean, hpef_novel=la.hpef_novel_mean, hpef_replica=la.hpef_replica_mean,
                       share_img_novel_peaks=la.share_img_novel, lattice_erased_below_nyq=la.erase_below_mean,
                       lattice_resynth_above_nyq=la.resynth_above_mean, dom_peak_shift_gt5=la.dom_shift_gt5,
                       dom_rawpeak_shift_gt5=la.domP_shift_gt5)
        q = IQ[(IQ.set == s) & (IQ.cond == c) & (IQ.restorer == r)]
        if len(q):
            q = q.iloc[0]
            row.update({k: q.get(k) for k in ("psnr_y", "ssim_y", "lpips_alex", "musiq", "clipiqa")})
        rows.append(row)
    out = pd.DataFrame(rows)
    clean_iq = IQ[IQ.restorer == "clean"].set_index("set")[["musiq", "clipiqa"]]
    out["musiq_clean_ref"] = out.set.map(clean_iq.musiq)
    out["clipiqa_clean_ref"] = out.set.map(clean_iq.clipiqa)
    out["o"] = out.restorer.map({r: i for i, r in enumerate(ORDER)})
    out = out.sort_values(["set", "cond", "o"]).drop(columns="o")
    p = os.path.join(RES, "genai_hallucination_table.csv")
    out.to_csv(p, index=False)
    shutil.copyfile(p, os.path.join(G.WVAQ, "paper", "tables", "genai_hallucination.csv"))

    # headline counts (vs none, per-set reference): generative / swinir / bicubic x 3 estimators x 2 sets x 3 conds
    pp = P[(P.comparator == "none") & (((P.ref == "primary") & (P.set == "R-LEG")) | ((P.ref == "self") & (P.set == "R-HR")))]
    grp = pp.restorer.map(lambda r: "generative" if r in GENS else r)
    summ = {g: pp[grp == g].verdict.value_counts().to_dict() for g in ("generative", "swinir", "bicubic")}
    summ["generative_by_estimator"] = {e: pp[(grp == "generative") & (pp.estimator == e)].verdict.value_counts().to_dict()
                                       for e in ("ours", "gabor", "cnn")}
    sg = S[S.restorer.isin(GENS)]
    summ["ours_wrong_but_confident_generative"] = dict(n_wrong=int(sg.n_bad.sum()), n_confident=int(sg.n_bad_confident.sum()))
    sn = S[S.restorer == "none"]
    summ["ours_wrong_but_confident_none"] = dict(n_wrong=int(sn.n_bad.sum()), n_confident=int(sn.n_bad_confident.sum()))
    json.dump(summ, open(os.path.join(RES, "genai_hallucination_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # compact LaTeX (D8, both sets)
    k = out[out.cond == "D8"].copy()
    L = ["% auto-generated by genai_sr/make_paper_table.py -- do not edit by hand",
         "\\begin{tabular}{llrrrrrr}", "\\toprule",
         " & & \\multicolumn{3}{c}{MAPE (\\%), $\\Delta$ vs.\\ degraded} & & & \\\\",
         "Set & Restoration & ours & log-Gabor & CNN & novel-peak energy & LPIPS$\\downarrow$ & MUSIQ$\\uparrow$ \\\\", "\\midrule"]

    def cell(r, e):
        m = r.get(f"{e}_MAPE")
        dl = r.get(f"{e}_dMAPE_vs_none")
        v = r.get(f"{e}_verdict_vs_none")
        if m is None or not np.isfinite(m):
            return "--"
        if r["restorer"] == "none" or dl is None or not np.isfinite(dl):
            return f"{m:.1f}"
        mark = "$^\\uparrow$" if v == "worse" else ("$^\\downarrow$" if v == "better" else "")
        return f"{m:.1f} ({dl:+.1f}){mark}"
    for s in ("R-LEG", "R-HR"):
        for _, r in k[k.set == s].iterrows():
            nov = "--" if pd.isna(r.get("hpef_novel")) else f"{100 * r['hpef_novel']:.1f}\\%"
            lp = "--" if pd.isna(r.get("lpips_alex")) else f"{r['lpips_alex']:.2f}"
            mu = "--" if pd.isna(r.get("musiq")) else f"{r['musiq']:.0f}"
            nm = r["name"].replace(" x4", " $\\times$4")
            L.append(f"{s} & {nm} & {cell(r, 'ours')} & {cell(r, 'gabor')} & {cell(r, 'cnn')} & {nov} & {lp} & {mu} \\\\")
        L.append("\\midrule" if s == "R-LEG" else "\\bottomrule")
    L.append("\\end{tabular}")
    open(os.path.join(G.WVAQ, "paper", "tables", "genai_hallucination.tex"), "w").write("\n".join(L) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
