"""Job C (WVAQ core): capture-quality experiment on the 13 hand-counted legacy crops, all methods
on byte-identical degraded inputs -> table, figure and confidence analysis.

Inputs (all built from the canonical VLM crops vlm/crops/legacy/{name}.png, degraded once by
methods/degrade_vlmcrops.py with degrade.py):
  methods/results/legacy13_vlmcrops_degrade.csv                 ours (frozen), log-Gabor, FFT
  cnn/runs/synth_r18_v2mix_aniso/degrade_vlmcrops/predictions.csv  synthetic-only CNN (zero-shot)
  vlm/results/legacy_deg_predictions.csv                         GPT-6.1-sol P1/P3, Gemini 3.8 Flash P3
      (vlm/run_degrade_vlm.py --collect; clean = the main legacy run, repeat 0, identical inputs)
  vlm/results/legacy_predictions.csv                             earlier GPT P1 x4/x8 resolution runs
      (near-replicates of down4/down8: W//k instead of round(W/k) -> listed with run='earlier')

Outputs:
  paper/tables/degradation_legacy13.csv            method x degradation x level x assignment
  paper/tables/degradation_legacy13_per_image.csv  every prediction used (traceability)
  paper/tables/degradation_legacy13_conf.csv       ours' confidence: drop per condition + error prediction
  paper/figs/degradation_curves.pdf / .png

Metrics are pooled over the 26 counts (13 images x warp/weft).  'axis' = vertical yarns <-> warp
(the convention of every legacy row in the paper), 'sorted' = best assignment.  MAE / MAPE / bias
are over the counts of images whose answer parsed; n_w10 counts the 26 GT values within 10 %
(an image without a count counts as 2 misses); parse_fail_rate = images without a usable count / 13
(includes abstentions); n_zero_answers = images answered with a count of 0 (scored as given, i.e.
100 % error; Gemini does this on whited-out patches where GPT abstains); abstain_rate = images where the VLM explicitly declined (null counts and/or
a stated reason such as "too blurred" / "almost entirely white") / 13.  95 % CIs:
image-level (cluster) bootstrap, 5000 resamples; dMAPE = paired difference to the same method's
clean run on the images parsed in both.

    <venv2>/bin/python tools/degradation_table.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figs")

NAMES13 = ["a1", "a2", "a4", "a5", "a6", "a8", "a10", "r2", "r3", "r4", "r6", "r7", "r8"]
FAMILY = {"clean": "clean", "gblur": "defocus", "down": "downscale", "jpeg": "jpeg", "glare": "glare",
          "corners": "corners"}
VLM_CONDS = {("gblur", 3.0), ("gblur", 6.0), ("down", 4.0), ("down", 8.0), ("jpeg", 20.0),
             ("glare", 0.05), ("glare", 0.30), ("corners", 0.05)}
LEVELS = {"gblur": [1, 2, 3, 4, 6], "down": [2, 4, 8], "jpeg": [90, 50, 20], "glare": [0.05, 0.10, 0.20, 0.30],
          "corners": [0.02, 0.05, 0.10]}
B = 5000
RNG = np.random.default_rng(20261007)


def sev_rank(kind, level):
    return 0 if kind == "clean" else 1 + LEVELS[kind].index(min(LEVELS[kind], key=lambda x: abs(x - level)))


# ----------------------------------------------------------------------------- load
def load():
    rows = []
    m = pd.read_csv(os.path.join(ROOT, "methods", "results", "legacy13_vlmcrops_degrade.csv"))
    for r in m.itertuples():
        rows.append(dict(method={"ours": "ours", "gabor": "log-Gabor", "fft": "FFT"}[r.estimator], run="main",
                         image=r.image, kind=r.kind, level=float(r.level), warp_pred=r.h, weft_pred=r.v,
                         ok=bool(np.isfinite(r.h) and np.isfinite(r.v)), conf=r.conf, gt_warp=r.gt_warp,
                         gt_weft=r.gt_weft, cost_usd=0.0, md5=r.md5, abstained=False))
    c = pd.read_csv(os.path.join(ROOT, "cnn", "runs", "synth_r18_v2mix_aniso", "degrade_vlmcrops", "predictions.csv"))
    for r in c.itertuples():
        rel_iqr = 0.5 * (r.h_iqr / r.pred_h + r.v_iqr / r.pred_v)
        rows.append(dict(method="CNN", run="main", image=r.image, kind=r.kind, level=float(r.level),
                         warp_pred=r.pred_h, weft_pred=r.pred_v, ok=True, conf=-rel_iqr, gt_warp=r.warp,
                         gt_weft=r.weft, cost_usd=0.0, md5=r.md5, abstained=False))
    gt = {r.image: (r.gt_warp, r.gt_weft) for r in m[m.estimator == "ours"].drop_duplicates("image").itertuples()}
    v = pd.read_csv(os.path.join(ROOT, "vlm", "results", "legacy_deg_predictions.csv"))
    v = v[v.model.isin(["gpt-6.1-sol", "gemini-3.8-flash"])]
    keep = {("gpt-6.1-sol", "P1"), ("gpt-6.1-sol", "P3"), ("gemini-3.8-flash", "P3")}
    for r in v.itertuples():
        if (r.model, r.prompt) not in keep:
            continue
        if r.repeat != 0 and not (r.kind == "clean" and r.prompt == "P1" and r.model == "gpt-6.1-sol"):
            continue
        ok = bool(r.ok) and pd.notna(r.vertical) and pd.notna(r.horizontal)
        rows.append(dict(method=f"{r.model} {r.prompt}", run="main" if r.repeat == 0 else f"repeat{r.repeat}",
                         image=r.image, kind=r.kind, level=float(r.level),
                         warp_pred=r.vertical if ok else np.nan, weft_pred=r.horizontal if ok else np.nan, ok=ok,
                         conf=np.nan, gt_warp=gt[r.image][0], gt_weft=gt[r.image][1],
                         cost_usd=float(r.cost_usd or 0), md5=None, abstained=bool(r.abstained)))
    lp = pd.read_csv(os.path.join(ROOT, "vlm", "results", "legacy_predictions.csv"))
    lp = lp[(lp.model == "gpt-6.1-sol") & lp.set.isin(["legacy_x4", "legacy_x8"]) & lp.image.isin(NAMES13)]
    for r in lp.itertuples():
        ok = bool(r.ok) and pd.notna(r.vertical) and pd.notna(r.horizontal)
        rows.append(dict(method="gpt-6.1-sol P1", run="earlier", image=r.image, kind="down",
                         level=float(r.set.split("_x")[1]), warp_pred=r.vertical if ok else np.nan,
                         weft_pred=r.horizontal if ok else np.nan, ok=ok, conf=np.nan,
                         gt_warp=gt[r.image][0], gt_weft=gt[r.image][1], cost_usd=float(r.usd or 0), md5=None,
                         abstained=False))
    d = pd.DataFrame(rows)
    d["level"] = d.level.round(4)
    d["family"] = d.kind.map(FAMILY)
    d["severity_rank"] = [sev_rank(k, lv) for k, lv in zip(d.kind, d.level)]
    for a in ("warp", "weft"):
        d[f"ape_{a}"] = 100 * (d[f"{a}_pred"] - d[f"gt_{a}"]).abs() / d[f"gt_{a}"]
    lo = np.minimum(d.warp_pred, d.weft_pred)
    hi = np.maximum(d.warp_pred, d.weft_pred)
    glo = np.minimum(d.gt_warp, d.gt_weft)
    ghi = np.maximum(d.gt_warp, d.gt_weft)
    d["ape_sorted_lo"] = 100 * (lo - glo).abs() / glo
    d["ape_sorted_hi"] = 100 * (hi - ghi).abs() / ghi
    d["ape_img"] = 0.5 * (d.ape_warp + d.ape_weft)
    d["both_w10"] = (d.ape_warp <= 10) & (d.ape_weft <= 10)
    return d


# ----------------------------------------------------------------------------- metrics
def per_image_errors(g, assign):
    """-> dict image -> (abs err[2], ape[2], signed err[2]) for parsed images."""
    out = {}
    for r in g.itertuples():
        if not r.ok:
            continue
        if assign == "axis":
            p, t = np.array([r.warp_pred, r.weft_pred]), np.array([r.gt_warp, r.gt_weft])
        else:
            p, t = np.sort([r.warp_pred, r.weft_pred]), np.sort([r.gt_warp, r.gt_weft])
        e = p - t
        out[r.image] = (np.abs(e), 100 * np.abs(e) / t, e)
    return out


def pooled(errs, images):
    if not images:
        return np.nan, np.nan
    ae = np.concatenate([errs[i][0] for i in images])
    pe = np.concatenate([errs[i][1] for i in images])
    return ae.mean(), pe.mean()


def boot_ci(errs, stat_fn, imgs):
    imgs = list(imgs)
    if len(imgs) < 3:
        return np.nan, np.nan
    vals = []
    for _ in range(B):
        s = RNG.choice(imgs, len(imgs), replace=True)
        vals.append(stat_fn(s))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def summarise(d):
    out = []
    main = d[d.run.isin(["main", "repeat1", "repeat2", "earlier"])]
    for (method, run, kind, level), g in main.groupby(["method", "run", "kind", "level"]):
        g = g[g.image.isin(NAMES13)]
        clean = d[(d.method == method) & (d.run == "main") & (d.kind == "clean")]
        for assign in ("axis", "sorted"):
            E = per_image_errors(g, assign)
            Ec = per_image_errors(clean, assign)
            imgs = sorted(E)
            mae, mape = pooled(E, imgs)
            pe_all = np.concatenate([E[i][1] for i in imgs]) if imgs else np.array([])
            se_all = np.concatenate([E[i][2] for i in imgs]) if imgs else np.array([])
            ae_all = np.concatenate([E[i][0] for i in imgs]) if imgs else np.array([])
            lo, hi = boot_ci(E, lambda s: np.mean(np.concatenate([E[i][1] for i in s])), imgs)
            both = sorted(set(E) & set(Ec))
            if kind != "clean" and both:
                dm = np.mean(np.concatenate([E[i][1] for i in both])) - np.mean(np.concatenate([Ec[i][1] for i in both]))
                dlo, dhi = boot_ci(None, lambda s: np.mean(np.concatenate([E[i][1] for i in s]))
                                   - np.mean(np.concatenate([Ec[i][1] for i in s])), both)
            else:
                dm = dlo = dhi = np.nan
            out.append(dict(method=method, run=run, family=FAMILY[kind], kind=kind, level=level,
                            severity_rank=sev_rank(kind, level), vlm_condition=(kind, level) in VLM_CONDS or kind == "clean",
                            assign=assign, n_images=len(g), n_ok=len(imgs),
                            parse_fail_rate=round(1 - len(imgs) / len(g), 3) if len(g) else np.nan,
                            abstain_rate=round(g.abstained.astype(bool).mean(), 3) if len(g) else np.nan,
                            n_zero_answers=int((g.ok & ((g.warp_pred == 0) | (g.weft_pred == 0))).sum()),
                            n_counts_scored=int(pe_all.size), MAE=mae, MAPE=mape, MAPE_lo=lo, MAPE_hi=hi,
                            medAE=float(np.median(ae_all)) if ae_all.size else np.nan,
                            bias=float(se_all.mean()) if se_all.size else np.nan,
                            n_w10=int((pe_all <= 10).sum()), n_counts=2 * len(g),
                            w10_pct=100 * (pe_all <= 10).sum() / (2 * len(g)) if len(g) else np.nan,
                            n_w5=int((pe_all <= 5).sum()), dMAPE_vs_clean=dm, dMAPE_lo=dlo, dMAPE_hi=dhi,
                            n_paired=len(both), cost_usd=g.cost_usd.sum()))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- confidence
def spearman_ci(x, y, groups):
    x, y, groups = map(np.asarray, (x, y, groups))
    m = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[m], y[m], groups[m]
    rho = stats.spearmanr(x, y).correlation
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(2000):
        s = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug), replace=True)])
        vals.append(stats.spearmanr(x[s], y[s]).correlation)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return rho, lo, hi, int(m.sum()), len(ug)


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    m = np.isfinite(score)
    score, label = score[m], label[m]
    if label.all() or (~label).all():
        return np.nan
    return stats.mannwhitneyu(score[label], score[~label]).statistic / (label.sum() * (~label).sum())


def confidence(d):
    out = []
    ours = d[(d.method == "ours") & (d.run == "main")]
    clean = ours[ours.kind == "clean"].set_index("image")
    # A. does the confidence drop under each degradation (paired over the 13 images)?
    for (kind, level), g in ours.groupby(["kind", "level"]):
        g = g.set_index("image").loc[NAMES13]
        dc = g.conf - clean.loc[NAMES13, "conf"]
        p = stats.wilcoxon(dc).pvalue if kind != "clean" and (dc != 0).any() else np.nan
        out.append(dict(part="A_conf_by_condition", family=FAMILY[kind], kind=kind, level=level,
                        n=len(g), conf_mean=g.conf.mean(), conf_median=g.conf.median(),
                        dconf_mean=dc.mean(), n_conf_lower=int((dc < 0).sum()), wilcoxon_p=p,
                        n_conf_ge_0p9=int((g.conf >= 0.9).sum()), n_both_w10=int(g.both_w10.sum()),
                        mean_ape_img=g.ape_img.mean()))
    # B. does the confidence predict error?  (Spearman over image x condition, cluster CI by image)
    scopes = {"all 19 conditions": lambda x: x,
              "18 degraded only": lambda x: x[x.kind != "clean"],
              "VLM conditions (clean + 8)": lambda x: x[[(k, lv) in VLM_CONDS or k == "clean"
                                                        for k, lv in zip(x.kind, x.level)]]}
    preds = {"ours conf": ("ours", "conf"), "FFT peak-height conf": ("FFT", "conf"),
             "log-Gabor agreement conf": ("log-Gabor", "conf"), "CNN -rel. tile IQR": ("CNN", "conf")}
    targets = ["ours", "log-Gabor", "CNN", "gpt-6.1-sol P1", "gpt-6.1-sol P3", "gemini-3.8-flash P3"]
    key = ["image", "kind", "level"]
    for sname, sf in scopes.items():
        for pname, (pm, pc) in preds.items():
            P = d[(d.method == pm) & (d.run == "main")][key + [pc]].rename(columns={pc: "score"})
            for tm in targets:
                if sname == "all 19 conditions" and tm.startswith(("gpt", "gemini")):
                    continue  # VLMs were run on clean + 8 conditions only: see the other two scopes
                if pname != "ours conf" and tm not in (pm, "ours"):
                    continue  # other confidences: only against their own method's error (and ours)
                T = d[(d.method == tm) & (d.run == "main") & d.ok][key + ["ape_img", "both_w10"]]
                J = sf(P.merge(T, on=key))
                if len(J) < 10:
                    continue
                rho, lo, hi, n, ng = spearman_ci(J.score, J.ape_img, J.image)
                hiconf = J.score >= 0.9 if pm == "ours" else None
                out.append(dict(part="B_conf_vs_error", scope=sname, predictor=pname, target_method=tm,
                                target="mean APE of the 2 counts (axis)", n=n, n_images=ng, spearman=rho,
                                spearman_lo=lo, spearman_hi=hi,
                                auc_both_w10=auc(J.score, J.both_w10), n_both_w10=int(J.both_w10.sum()),
                                n_conf_ge_0p9=int(hiconf.sum()) if hiconf is not None else np.nan,
                                acc_both_w10_conf_ge_0p9=J.both_w10[hiconf].mean() if hiconf is not None and hiconf.any() else np.nan,
                                acc_both_w10_conf_lt_0p9=J.both_w10[~hiconf].mean() if hiconf is not None and (~hiconf).any() else np.nan))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figure
CAT = {"ours": "#2a78d6", "gpt": "#eb6834", "cnn": "#1baf7a", "gemini": "#4a3aa7"}  # validated all-pairs
INK, INK2, GRID, CTX = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
SERIES = [  # (method, label, color, marker, linestyle, lw)
    ("ours", os.environ.get("OURS_NAME", "PG-Lattice"), CAT["ours"], "o", "-", 1.8),  # = \ours (preamble.tex)
    ("CNN", "CNN (synthetic only)", CAT["cnn"], "s", "-", 1.4),
    ("gpt-6.1-sol P1", "GPT-6.1-sol P1 (whole inch)", CAT["gpt"], "D", "--", 1.4),
    ("gpt-6.1-sol P3", "GPT-6.1-sol P3 (¼-inch ×4)", CAT["gpt"], "^", ":", 1.4),
    ("gemini-3.8-flash P3", "Gemini 3.8 Flash P3 (¼-inch ×4)", CAT["gemini"], "v", ":", 1.4),
    ("log-Gabor", "log-Gabor (context)", CTX, None, "-", 0.9),
]
PANELS = [("gblur", "Defocus σ (u = 10$^{-3}$ in)"), ("down", "Downscale factor"), ("jpeg", "JPEG quality"),
          ("glare", "Glare (% of window)"), ("corners", "Corner error (% side)")]


TICKS = {"gblur": ["clean", "1", "2", "3", "4", "6"], "down": ["clean", "×2", "×4", "×8"],
         "jpeg": ["clean", "90", "50", "20"], "glare": ["clean", "5", "10", "20", "30"],
         "corners": ["clean", "2", "5", "10"]}


def figure(S, loo_mape):
    """MAPE (axis-matched, log scale) against ordinal severity, one panel per family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    # printed at 0.64\linewidth of a figure* (about 0.63x), so 9-10 pt here prints at about 6 pt (Job Fig2, Oct 12)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.8, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    A = S[(S["assign"] == "axis") & (S.run == "main")]
    gpt_clean = S[(S["assign"] == "axis") & (S.method == "gpt-6.1-sol P1") & (S.kind == "clean")].MAPE
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.3), sharey=True)
    y_lo, y_hi = 1.2, 400
    for ax, (kind, xl) in zip(axes, PANELS):
        # run-to-run spread of GPT-6.1-sol P1 on the clean crops (3 repeats), as a noise reference
        ax.plot([-0.22, -0.22], [gpt_clean.min(), gpt_clean.max()], color=CAT["gpt"], lw=2.2, alpha=0.45,
                solid_capstyle="butt", zorder=1)
        for method, label, col, mk, ls, lw in SERIES:
            g = A[(A.method == method) & A.kind.isin(["clean", kind])].sort_values("severity_rank")
            if g.empty:
                continue
            x = g.severity_rank.to_numpy(float)
            # break the line at partially abstained points: their MAPE covers answered images only
            y_line = g.MAPE.where(g.parse_fail_rate == 0) if mk is not None else g.MAPE
            ax.plot(x, y_line, ls=ls, color=col, lw=lw, zorder=3 if method == "ours" else 2,
                    label=label if mk is None else None)
            if mk is None:
                continue
            full = (g.parse_fail_rate == 0).to_numpy()
            ax.plot(x[full], g.MAPE[full], ls="none", marker=mk, ms=4.2, color=col, mec="white", mew=0.6,
                    zorder=4 if method == "ours" else 3)
            # partial abstention: MAPE over answered images only, so grey it out (it is not comparable)
            ax.plot(x[~full], g.MAPE[~full], ls="none", marker=mk, ms=4.2, mfc="white", mec=CTX, mew=1.0, zorder=4)
            ax.plot([], [], ls=ls, color=col, lw=lw, marker=mk, ms=4.2, mec="white", mew=0.6, label=label)
            for r in g[~full].itertuples():
                k = int(round(r.parse_fail_rate * r.n_images))
                # label placement (points): first level -> below right, last level -> above left with a leader
                if r.severity_rank <= 1:
                    xy, ha, va = (5, -9), "left", "top"
                elif r.severity_rank == len(TICKS[kind]) - 1 and r.MAPE > 5:
                    xy, ha, va = (-8, -45), "right", "center"
                else:
                    xy, ha, va = (-5, 0), "right", "center"
                ax.annotate(f"{k}/13 abstained", (r.severity_rank, r.MAPE), xytext=xy, textcoords="offset points",
                            fontsize=6.8, color=INK2, ha=ha, va=va,
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=0, shrinkB=3))
        ax.axhline(loo_mape, color=INK2, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(-0.45, len(TICKS[kind]) - 0.6)
        ax.set_xticks(range(len(TICKS[kind])))
        ax.set_xticklabels(TICKS[kind])
        ax.set_xlabel(xl)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MAPE vs hand count (%)")
    ticks = [2, 5, 10, 20, 50, 100, 200]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([str(t) for t in ticks])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].text(3.35, loo_mape * 1.1, "leave-one-out mean", fontsize=6.8, color=INK2, ha="right", va="bottom")
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in [s[1] for s in SERIES]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.8, columnspacing=1.4)
    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.2, top=0.78, wspace=0.12)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"degradation_curves.{ext}"), dpi=300)
    plt.close(fig)


def loo_mean(d):
    """Leave-one-out mean predictor (axis-matched MAPE) on the same GT: the 'uninformative' reference."""
    g = d[(d.method == "ours") & (d.kind == "clean")].set_index("image").loc[NAMES13]
    gw, gf = g.gt_warp.to_numpy(float), g.gt_weft.to_numpy(float)
    n = len(gw)
    lw_, lf_ = (gw.sum() - gw) / (n - 1), (gf.sum() - gf) / (n - 1)
    return 100 * np.mean(np.r_[np.abs(lw_ - gw) / gw, np.abs(lf_ - gf) / gf])


def main():
    d = load()
    if "--fig-only" in sys.argv:  # redraw from the written table (skips the bootstrap)
        S = pd.read_csv(os.path.join(TAB, "degradation_legacy13.csv"))
        figure(S, loo_mean(d))
        return
    S = summarise(d)
    S = S.sort_values(["assign", "method", "run", "family", "severity_rank"]).reset_index(drop=True)
    os.makedirs(TAB, exist_ok=True)
    S.round(3).to_csv(os.path.join(TAB, "degradation_legacy13.csv"), index=False)
    cols = ["method", "run", "image", "family", "kind", "level", "severity_rank", "warp_pred", "weft_pred", "ok",
            "gt_warp", "gt_weft", "ape_warp", "ape_weft", "ape_sorted_lo", "ape_sorted_hi", "conf", "abstained",
            "cost_usd", "md5"]
    d.sort_values(["method", "run", "family", "severity_rank", "image"])[cols].round(4).to_csv(
        os.path.join(TAB, "degradation_legacy13_per_image.csv"), index=False)
    Cf = confidence(d)
    Cf.round(4).to_csv(os.path.join(TAB, "degradation_legacy13_conf.csv"), index=False)
    loo = loo_mean(d)
    figure(S, loo)
    A = S[(S["assign"] == "axis") & S.vlm_condition & (S.run == "main")]
    piv = A.pivot_table(index=["family", "kind", "level"], columns="method", values="MAPE").round(1)
    print(piv.to_string())
    print(f"\nleave-one-out mean predictor MAPE (axis) = {loo:.1f}%")


if __name__ == "__main__":
    main()
