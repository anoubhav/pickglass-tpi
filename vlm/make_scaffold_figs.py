"""Figures + LaTeX table for the scaffolded-prompting study.

  paper/figs/fig_scaffold_examples.pdf  the four scaffolds rendered on one legacy crop (a2)
  paper/figs/fig_scaffold_results.pdf   MAPE [95 % CI] per model x condition (legacy-13, axis-matched)
  paper/tables/scaffold_legacy13.tex    compact table (MAPE [CI], <=10 %, paired dMAPE vs own P1)
  paper/tables/scaffold_legacy13.csv    the numbers behind both

    <venv2>/bin/python make_scaffold_figs.py
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import scaffolds as SC                   # noqa: E402

RES = os.path.join(SC.SC_DIR, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
TABS = os.path.join(ROOT, "paper", "tables")
MODELS = ["gpt-6.1-sol", "gemini-3.8-flash", "qwen3.8-27b", "molmo2_8b_bf16"]
TITLE = {"gpt-6.1-sol": "GPT-6.1-sol", "gemini-3.8-flash": "Gemini 3.8 Flash",
         "qwen3.8-27b": "Qwen3.8-27B (open)", "molmo2_8b_bf16": "Molmo2-8B bf16 (open)"}
CONDS = ["P1", "S1t", "S1", "S2", "S3", "S4"]
CLABEL = {"P1": "P1 direct", "S1t": "S1t text only", "S1": "S1 ruler", "S2": "S2 guide lines",
          "S3": "S3 0.1-in insets", "S4": "S4 profile plot"}
INK, MUTED, GRID, ACCENT, BASE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#8a8984"
REF_OURS, REF_LOO = 11.6, 32.5     # paper/tables/legacy13_unified.csv (axis-matched MAPE)


def examples():
    crop = SC.load_crop("legacy", "a2")
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95))
    for a, sid, t in zip(ax, ["S1", "S2", "S3", "S4"],
                         ["S1 ruler (0.1 in ticks)", "S2 guide lines, 10 segments", "S3 four 0.1-in insets",
                          "S4 band brightness profiles"]):
        im = cv2.resize(SC.render(crop, sid), (540, 540), interpolation=cv2.INTER_AREA)
        a.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        a.set_title(t, fontsize=7, color=INK, pad=2)
        a.axis("off")
    plt.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.03)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_scaffold_examples.{ext}"), dpi=300)
    plt.close(fig)
    # photo panels compress badly in the PDF (Flate); a JPEG copy is ~10x smaller for \includegraphics
    cv2.imwrite(os.path.join(FIGS, "fig_scaffold_examples.jpg"),
                cv2.imread(os.path.join(FIGS, "fig_scaffold_examples.png")), [cv2.IMWRITE_JPEG_QUALITY, 88])


def results(T, PR):
    ax_ = T[(T["assign"] == "axis") & (T.repeat.astype(str) == "0")]
    gpt_p1 = float(ax_[(ax_.model == "gpt-6.1-sol") & (ax_.cond == "P1")].MAPE.iloc[0])
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
    for a, m in zip(axs, MODELS):
        y = np.arange(len(CONDS))[::-1]
        for yy, c in zip(y, CONDS):
            r = ax_[(ax_.model == m) & (ax_.cond == c)]
            if not len(r):
                continue
            r = r.iloc[0]
            col = BASE if c == "P1" else ACCENT
            a.plot([r.MAPE_lo, r.MAPE_hi], [yy, yy], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            a.plot([r.MAPE], [yy], "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=3)
            if c != "P1":
                p = PR[(PR["assign"] == "axis") & (PR.model_A == m) & (PR.cond_A == c) & (PR.rep_A.astype(str) == "0")
                       & (PR.cond_B == "P1") & (PR.model_B == m) & (PR.rep_B.astype(str) == "0")]
                if len(p) and (p.iloc[0].dMAPE_hi < 0 or p.iloc[0].dMAPE_lo > 0):
                    a.text(r.MAPE_hi * 1.08, yy, "*", fontsize=8, va="center", color=INK)
            n_ok = int(r.n_ok)
            if n_ok < 13:
                a.text(3.2, yy + 0.3, f"{n_ok}/13 parsed", fontsize=4.8, color=MUTED, va="center")
        a.axvline(gpt_p1, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
        a.axvline(REF_LOO, color=MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        a.set_xscale("log")
        a.set_xlim(3, 200)
        a.set_xticks([5, 10, 20, 50, 100])
        a.set_xticklabels(["5", "10", "20", "50", "100"], fontsize=6, color=MUTED)
        a.minorticks_off()
        a.tick_params(axis="x", length=2, color=GRID)
        a.tick_params(axis="y", length=0)
        a.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.set_title(TITLE[m], fontsize=7, color=INK, pad=3)
        a.set_xlabel("MAPE (%), log scale", fontsize=6, color=MUTED, labelpad=1)
    axs[0].set_yticks(np.arange(len(CONDS))[::-1])
    axs[0].set_yticklabels([CLABEL[c] for c in CONDS], fontsize=6.5, color=INK)
    fig.text(0.995, 0.015, "dashed: GPT-6.1-sol P1 (%.1f%%)   dotted: leave-one-out mean (%.1f%%)   "
             "grey: P1   *: paired 95%% CI of the change vs P1 excludes 0" % (gpt_p1, REF_LOO),
             fontsize=5.2, color=MUTED, ha="right", va="bottom")
    plt.subplots_adjust(left=0.115, right=0.995, top=0.88, bottom=0.25, wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_scaffold_results.{ext}"), dpi=300)
    plt.close(fig)


def table(T, PR):
    ax_ = T[(T["assign"] == "axis") & (T.repeat.astype(str) == "0") & T.cond.isin(CONDS) & T.model.isin(MODELS)]
    rows = []
    for m in MODELS:
        for c in CONDS:
            r = ax_[(ax_.model == m) & (ax_.cond == c)]
            if not len(r):
                continue
            r = r.iloc[0]
            d = dict(model=m, cond=c, n_ok=int(r.n_ok), MAPE=r.MAPE, MAPE_lo=r.MAPE_lo, MAPE_hi=r.MAPE_hi,
                     MAE=r.MAE, within10=int(r.within10), n_values=int(r.n_values), bias=r.bias,
                     MAPE_best=float(T[(T["assign"] == "sorted") & (T.model == m) & (T.cond == c)
                                       & (T.repeat.astype(str) == "0")].MAPE.iloc[0]))
            if c != "P1":
                p = PR[(PR["assign"] == "axis") & (PR.model_A == m) & (PR.cond_A == c) & (PR.rep_A.astype(str) == "0")
                       & (PR.model_B == m) & (PR.cond_B == "P1") & (PR.rep_B.astype(str) == "0")].iloc[0]
                d.update(dMAPE=p.dMAPE, dMAPE_lo=p.dMAPE_lo, dMAPE_hi=p.dMAPE_hi, p_wil=p.p_wilcoxon_abs_pct_err_26,
                         p_holm=p.p_holm, kind=p.kind, n_pair=int(p.n_counts))
            rows.append(d)
    D = pd.DataFrame(rows)
    D.round(4).to_csv(os.path.join(TABS, "scaffold_legacy13.csv"), index=False)

    def f(x, nd=1):
        return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"
    L = [r"% Generated by vlm/make_scaffold_figs.py from vlm/scaffold/results -- do not edit by hand.",
         r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}llrccr@{}}", r"\toprule",
         r"Model & Input & $n$ & MAPE [95\% CI] & $\le$10\% & $\Delta$ vs P1 [95\% CI] \\", r"\midrule"]
    for i, m in enumerate(MODELS):
        if i:
            L.append(r"\midrule")
        g = D[D.model == m]
        for j, r in enumerate(g.itertuples()):
            name = TITLE[m] if j == 0 else ""
            dd = "--" if r.cond == "P1" else (f"${r.dMAPE:+.1f}$ [${r.dMAPE_lo:+.1f}$, ${r.dMAPE_hi:+.1f}$]"
                                             + (r"$^\ast$" if (r.dMAPE_hi < 0 or r.dMAPE_lo > 0) else ""))
            cond = r.cond + (r"$^{\dagger}$" if r.cond == "S3" else "")
            L.append(f"{name} & {cond} & {r.n_ok} & {f(r.MAPE)} [{f(r.MAPE_lo)}, {f(r.MAPE_hi)}] & "
                     f"{r.within10}/{r.n_values} & {dd} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Scaffolded visual prompting on legacy-13 (axis-matched; 26 counts when all 13 images parse). "
          r"P1: the unchanged direct prompt; S1t: S1's count-a-tenth-and-multiply instruction without the drawn "
          r"ruler (byte-identical image to P1); S1: rulers with 0.1-inch ticks; S2: guide lines in ten numbered "
          r"0.1-inch segments; S3: four magnified 0.1-inch insets; S4: crop plus band brightness profiles. "
          r"$\Delta$: paired difference in MAPE (points) to the same model's P1 on the images both parsed, "
          r"image-level bootstrap CI; $^\ast$ CI excludes 0. $^{\dagger}$ S3 was pre-declared as the primary "
          r"scaffold on synthetic images before any legacy run; all other rows are exploratory.}",
          r"\label{tab:scaffolds}", r"\end{table*}"]
    open(os.path.join(TABS, "scaffold_legacy13.tex"), "w").write("\n".join(L) + "\n")
    return D


def compact(D, PR, T):
    """Main-paper table: conditions x models, cell = MAPE (paired change vs own P1)."""
    S4o = PR[(PR["assign"] == "axis") & (PR.kind == "S4-oracle") & (PR.model_A == "profile-oracle")
             & (PR.model_B == "est:ours")].iloc[0]
    L = [r"% Generated by vlm/make_scaffold_figs.py -- do not edit by hand.", r"\begin{table}[t]", r"\centering",
         r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Input & GPT-6.1 & Gem.\,Flash & Qwen-27B$^{o}$ & Molmo2$^{o}$ \\", r"\midrule"]
    for c in CONDS:
        cells = []
        for m in MODELS:
            r = D[(D.model == m) & (D.cond == c)]
            if not len(r):
                cells.append("--")
                continue
            r = r.iloc[0]
            if c == "P1":
                cells.append(f"{r.MAPE:.1f}")
            else:
                sig = (r.dMAPE_hi < 0 or r.dMAPE_lo > 0)
                val = (r"\textbf{%.1f}" % r.MAPE) if sig and r.dMAPE < 0 else f"{r.MAPE:.1f}"
                cells.append(val + r"\,{\tiny$" + f"{r.dMAPE:+.1f}" + r"$}" + (r"$^\ast$" if sig and r.dMAPE > 0 else ""))
        name = {"P1": "P1 direct", "S1t": "S1t text only", "S1": "S1 ruler", "S2": "S2 guide lines",
                "S3": r"S3 insets$^{\dagger}$", "S4": "S4 profile"}[c]
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Scaffolded visual prompting on legacy-13 (26 counts, axis-matched MAPE in \%; small: paired "
          r"change vs the same model's P1 in points; bold: 95\% image-bootstrap CI of the change below 0). "
          + f"Without any VLM, counting the peaks of the S4 plot (find\\_peaks, set on synthetic) gives {S4o.MAPE_A:.1f}; "
          + f"\\ours\\ {S4o.MAPE_B:.1f}; leave-one-out mean {REF_LOO:.1f}. "
          r"$^{\dagger}$S3 pre-declared as primary on synthetic images (Holm over 4 models); other rows exploratory. "
          r"$^{o}$open weights. S1t: S1's instruction without the ruler (same image as P1).}",
          r"\label{tab:scaffolds_compact}", r"\end{table}"]
    open(os.path.join(TABS, "scaffold_legacy13_compact.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    T = pd.read_csv(os.path.join(RES, "legacy13_scaffold_metrics.csv"))
    PR = pd.read_csv(os.path.join(RES, "legacy13_scaffold_paired.csv"))
    examples()
    results(T, PR)
    D = table(T, PR)
    compact(D, PR, T)
    print(D.round(2).to_string(index=False))
